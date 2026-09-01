"""Joint DSRL position correction and contact-gated VLM force control."""

from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from clean_dsrl_wrapper import CleanDSRLLabPickWrapper
from online_dsrl_metrics import FAILURE_FLAG_KEYS, any_flag, classify_terminal, scalar
from vlm_force import (
    EpisodeFeedback,
    EpisodeForceAdaptationLoop,
    ForceControllerConfig,
    ForceRange,
    TactileForceController,
    diagnose_episode_failure,
)


ExperimentMode = Literal[
    "ours_tactile",
    "dsrl_tactile",
    "joint",
    "joint_bilateral",
    "guarded_joint",
    "dsrl",
    "vlm",
    "base",
    "flow_rwr",
    "flow_ppo",
]


def _tensor_row(value: torch.Tensor) -> list[float]:
    return value.detach().reshape(value.shape[0], -1)[0].cpu().float().tolist()


class VLMDSRLLabPickWrapper(CleanDSRLLabPickWrapper):
    """Apply force feedback to gripper width while DSRL controls pose.

    In ``joint`` mode, an online SAC agent supplies Flow-noise actions and the
    force controller replaces only CAFE action index 9 after contact.
    ``joint_bilateral`` additionally requires both tactile sensors and both
    per-finger forces to cross their contact thresholds before first
    activation. ``dsrl`` disables the advisor/controller, while ``vlm`` uses
    native Flow noise and retains the same force-adaptation path. ``base`` runs
    the frozen native BC policy without DSRL updates, an advisor, or force
    control. The wrapper owns durable, per-physics-step trajectories for every
    outcome.
    """

    def __init__(
        self,
        env: Any,
        policy_checkpoint: str | Path,
        *,
        mode: ExperimentMode,
        output_dir: str | Path,
        adaptation: EpisodeForceAdaptationLoop | None,
        controller_config: ForceControllerConfig | None = None,
        save_advisor_images: bool = False,
        record_videos: bool = False,
        video_dir: str | Path | None = None,
        video_camera: str = "third",
        video_every_n_physics_steps: int = 4,
        video_fps: int = 30,
        **kwargs: Any,
    ) -> None:
        if mode not in {
            "ours_tactile",
            "dsrl_tactile",
            "joint",
            "joint_bilateral",
            "guarded_joint",
            "dsrl",
            "vlm",
            "base",
            "flow_rwr",
            "flow_ppo",
        }:
            raise ValueError(f"Unsupported experiment mode: {mode!r}.")
        if (mode in {"ours_tactile", "joint", "joint_bilateral", "vlm"}) != (
            adaptation is not None
        ):
            raise ValueError(
                "ours_tactile/joint/joint_bilateral/vlm modes require adaptation; "
                "dsrl_tactile/dsrl/base/flow_rwr/flow_ppo modes forbid it."
            )
        self.mode = mode
        self.experiment_seed = kwargs.get("seed")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir = self.output_dir / "trajectories"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.episode_path = self.output_dir / "episodes.jsonl"
        if self.episode_path.exists():
            raise FileExistsError(
                f"Refusing to append to existing episode log: {self.episode_path}"
            )
        self._episode_stream = self.episode_path.open(
            "x", encoding="utf-8", buffering=1
        )
        self.adaptation = adaptation
        self.save_advisor_images = bool(save_advisor_images)
        self.record_videos = bool(record_videos)
        self.video_camera = str(video_camera)
        self.video_every_n_physics_steps = int(video_every_n_physics_steps)
        self.video_fps = int(video_fps)
        if self.video_camera not in {
            "third",
            "wrist",
            "viewer",
            "tactile_left",
            "tactile_right",
        }:
            raise ValueError(f"Unsupported video camera: {self.video_camera!r}.")
        if self.video_every_n_physics_steps < 1 or self.video_fps < 1:
            raise ValueError("Video sampling interval and FPS must be positive.")
        self.video_dir = (
            self.output_dir / "videos"
            if video_dir is None
            else Path(video_dir).expanduser().resolve()
        )
        if self.record_videos:
            self.video_dir.mkdir(parents=True, exist_ok=True)
        self.force_control_enabled = mode in {
            "ours_tactile",
            "joint",
            "joint_bilateral",
            "vlm",
        }
        resolved_controller_config = (
            ForceControllerConfig(
                require_contact_mask_for_activation=mode == "joint_bilateral"
            )
            if controller_config is None
            else controller_config
        )
        if (
            mode == "joint_bilateral"
            and not resolved_controller_config.require_contact_mask_for_activation
        ):
            raise ValueError(
                "joint_bilateral requires explicit contact-mask activation gating."
            )
        self.controller = (
            None
            if adaptation is None
            else TactileForceController(
                adaptation.current_range_n,
                resolved_controller_config,
            )
        )
        self.completed_episodes = 0
        self.total_outer_interactions = 0
        self._active_reset_seed: int | None = None
        self.results: list[dict[str, Any]] = []
        self._trajectory_stream: Any | None = None
        self._pending_episode: dict[str, Any] | None = None
        self._last_episode_images: dict[str, np.ndarray] = {}
        self._video_writer: Any | None = None
        self._video_temporary_path: Path | None = None
        self._video_frame_count = 0
        super().__init__(env, policy_checkpoint, **kwargs)
        self._reset_hybrid_episode_state()

    @property
    def advisor_calls(self) -> int:
        if self.adaptation is None:
            return 0
        return int(getattr(self.adaptation.advisor, "call_count", len(self.results)))

    def reset(self, **kwargs: Any):
        seed = kwargs.get("seed")
        self._active_reset_seed = None if seed is None else int(seed)
        return super().reset(**kwargs)

    def _reset_hybrid_episode_state(self) -> None:
        self._episode_physics_steps = 0
        self._episode_outer_interactions = 0
        self._episode_return = 0.0
        self._episode_flags = {reason: False for reason, _ in FAILURE_FLAG_KEYS}
        self._episode_flags.update({"success": False, "timeout": False})
        self._touched = False
        self._tactile_contact_samples = 0
        self._bilateral_contact_samples = 0
        self._loaded_contact_samples = 0
        self._contact_force_sum_n = 0.0
        self._squared_force_error_sum_n2 = 0.0
        self._peak_contact_force_n = 0.0
        self._peak_net_contact_force_n = 0.0
        self._max_lift_m = 0.0
        self._min_grasp_distance_m = float("inf")
        self._controller_active_steps = 0
        self._safety_override_steps = 0
        self._max_non_gripper_action_delta = 0.0
        self._first_controller_activation_physics_step: int | None = None
        self._first_controller_activation_bilateral: bool | None = None
        self._first_controller_activation_left_force_n: float | None = None
        self._first_controller_activation_right_force_n: float | None = None
        self._last_episode_images = {}
        self._video_frame_count = 0
        self._attempted_range_n = (
            None if self.adaptation is None else self.adaptation.current_range_n
        )

    def _after_episode_reset(self) -> None:
        if self._pending_episode is not None:
            raise RuntimeError(
                "The terminal episode must be committed after the DSRL update "
                "and before the next reset."
            )
        if self._trajectory_stream is not None and not self._trajectory_stream.closed:
            raise RuntimeError("Previous trajectory stream is still open.")
        if self._video_writer is not None:
            raise RuntimeError("Previous episode video writer is still open.")
        self._reset_hybrid_episode_state()
        if self.controller is not None:
            assert self._attempted_range_n is not None
            self.controller.reset()
            self.controller.set_target_range(self._attempted_range_n)
        path = self.trajectory_dir / f"episode_{self.completed_episodes:03d}.jsonl.gz"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite trajectory: {path}")
        self._trajectory_stream = gzip.open(path, "xt", encoding="utf-8")
        self._video_temporary_path = None
        if self.record_videos:
            self._video_temporary_path = (
                self.video_dir
                / f"episode_{self.completed_episodes:03d}_pending_{self.video_camera}.mp4"
            )
            if self._video_temporary_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite video: {self._video_temporary_path}"
                )

    def _capture_video_frame(self, *, terminal: bool) -> None:
        if not self.record_videos:
            return
        if (
            self._episode_physics_steps != 1
            and self._episode_physics_steps % self.video_every_n_physics_steps != 0
            and not terminal
        ):
            return
        if self._video_temporary_path is None:
            raise RuntimeError("Video path is unavailable before a physics step.")
        if self._video_writer is None:
            import imageio.v2 as imageio

            self._video_writer = imageio.get_writer(
                str(self._video_temporary_path),
                fps=self.video_fps,
                macro_block_size=1,
            )
        frame = np.asarray(
            self.env.unwrapped.get_video_frame(self.video_camera)[:, :, :3]
        )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.float32)
            if frame.size and float(frame.max()) <= 1.0:
                frame *= 255.0
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
        self._video_writer.append_data(frame)
        self._video_frame_count += 1

    def _close_video_writer(self) -> None:
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None

    def _finalize_episode_video(
        self, episode_index: int, *, success: bool, reason: str
    ) -> Path | None:
        if not self.record_videos:
            return None
        if (
            self._video_temporary_path is None
            or not self._video_temporary_path.is_file()
        ):
            raise RuntimeError("Recorded episode has no video file.")
        if self._video_frame_count < 1:
            raise RuntimeError("Recorded episode has no video frames.")
        status = "success" if success else "failure"
        safe_reason = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_" for char in reason
        )
        final_path = self.video_dir / (
            f"episode_{episode_index:03d}_{status}_{safe_reason}_{self.video_camera}.mp4"
        )
        if final_path.exists():
            raise FileExistsError(f"Refusing to overwrite video: {final_path}")
        self._video_temporary_path.replace(final_path)
        self._video_temporary_path = None
        return final_path

    def _cache_advisor_images(self) -> None:
        if not (
            self.save_advisor_images
            or (
                self.adaptation is not None
                and self.adaptation.advisor.__class__.__name__
                == "OpenAICompatibleVLMAdvisor"
            )
        ):
            return
        base = self.env.unwrapped
        self._last_episode_images = {
            name: np.asarray(base.get_video_frame(name)[:, :, :3]).copy()
            for name in ("third", "wrist", "tactile_left", "tactile_right")
        }

    def _transform_physical_action(
        self, policy_action: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        base = self.env.unwrapped
        force_metrics = base.get_contact_force_metrics()
        left_touch, right_touch = base.tactile_contact_depths()
        tactile_contact = (left_touch > base.cfg.tactile_threshold_mm) | (
            right_touch > base.cfg.tactile_threshold_mm
        )
        both_contact = (left_touch > base.cfg.tactile_threshold_mm) & (
            right_touch > base.cfg.tactile_threshold_mm
        )
        grip_force = force_metrics["grip_force_n"]
        left_force = force_metrics["left_finger_force_n"]
        right_force = force_metrics["right_finger_force_n"]
        break_force = force_metrics["max_finger_force_n"]
        net_force = force_metrics["net_force_n"]
        robot_state = base.get_cafe_observation()["robot0_pos"]
        relative_position, object_rot6d = base.get_privileged_object_pose()
        self._cache_advisor_images()

        if self.controller is None:
            executed_action = policy_action.clone()
            controller_active = False
            safety_override = False
            filtered_force = scalar(grip_force)
            target_force = None
            force_error = None
        else:
            if self.mode == "joint_bilateral":
                force_gate = (
                    left_force >= self.controller.config.contact_on_force_n
                ) & (right_force >= self.controller.config.contact_on_force_n)
                requested_contact = both_contact & force_gate
            else:
                requested_contact = tactile_contact & (
                    grip_force >= self.controller.config.contact_on_force_n
                )
            executed_action, diagnostics = self.controller.control(
                policy_action,
                contact_force_n=grip_force,
                safety_force_n=break_force,
                contact_mask=requested_contact,
                dt_s=float(base.physics_dt),
            )
            controller_active = any_flag(diagnostics.active)
            safety_override = any_flag(diagnostics.safety_override)
            filtered_force = scalar(diagnostics.filtered_force_n)
            target_force = float(diagnostics.target_force_n)
            force_error = scalar(diagnostics.force_error_n)

        non_gripper = [
            index
            for index in range(policy_action.shape[-1])
            if self.controller is None
            or index != self.controller.config.gripper_width_index
        ]
        max_non_gripper_delta = scalar(
            (executed_action[:, non_gripper] - policy_action[:, non_gripper])
            .abs()
            .max()
        )
        if max_non_gripper_delta > 1.0e-7:
            raise RuntimeError(
                "Force arbitration changed a non-gripper policy dimension: "
                f"max delta={max_non_gripper_delta:.3e}."
            )

        metadata = {
            "pre_robot_state": _tensor_row(robot_state),
            "pre_relative_object_position": _tensor_row(relative_position),
            "pre_object_rot6d": _tensor_row(object_rot6d),
            "pre_left_touch_mm": scalar(left_touch),
            "pre_right_touch_mm": scalar(right_touch),
            "pre_tactile_contact": any_flag(tactile_contact),
            "pre_bilateral_contact": any_flag(both_contact),
            "pre_grip_force_n": scalar(grip_force),
            "pre_left_force_n": scalar(left_force),
            "pre_right_force_n": scalar(right_force),
            "pre_break_force_n": scalar(break_force),
            "pre_net_force_n": scalar(net_force),
            "controller_active": controller_active,
            "safety_override": safety_override,
            "filtered_force_n": filtered_force,
            "target_force_n": target_force,
            "force_error_n": force_error,
            "max_non_gripper_action_delta": max_non_gripper_delta,
        }
        return executed_action, metadata

    def _after_physics_step(
        self,
        *,
        policy_action: torch.Tensor,
        executed_action: torch.Tensor,
        reward: Any,
        terminated: Any,
        truncated: Any,
        info: dict[str, Any],
        action_metadata: dict[str, Any],
        final_metrics: dict[str, Any],
    ) -> None:
        del info
        self._episode_physics_steps += 1
        reward_value = scalar(reward)
        self._episode_return += reward_value
        for name, value in final_metrics["flags"].items():
            self._episode_flags[name] |= bool(value)

        tactile_contact = bool(action_metadata["pre_tactile_contact"])
        bilateral_contact = bool(action_metadata["pre_bilateral_contact"])
        grip_force_n = float(action_metadata["pre_grip_force_n"])
        contact_on_force_n = (
            0.01
            if self.controller is None
            else self.controller.config.contact_on_force_n
        )
        loaded_contact = grip_force_n >= contact_on_force_n
        target_force_n = action_metadata["target_force_n"]
        self._touched |= tactile_contact
        self._tactile_contact_samples += int(tactile_contact)
        self._bilateral_contact_samples += int(bilateral_contact)
        self._loaded_contact_samples += int(loaded_contact)
        if loaded_contact:
            self._contact_force_sum_n += grip_force_n
            if target_force_n is not None:
                self._squared_force_error_sum_n2 += (
                    grip_force_n - float(target_force_n)
                ) ** 2
        self._peak_contact_force_n = max(
            self._peak_contact_force_n, final_metrics["contact_force_n"]
        )
        self._peak_net_contact_force_n = max(
            self._peak_net_contact_force_n, final_metrics["net_contact_force_n"]
        )
        self._max_lift_m = max(self._max_lift_m, final_metrics["lift_m"])
        self._min_grasp_distance_m = min(
            self._min_grasp_distance_m, final_metrics["grasp_distance_m"]
        )
        self._controller_active_steps += int(action_metadata["controller_active"])
        self._safety_override_steps += int(action_metadata["safety_override"])
        self._max_non_gripper_action_delta = max(
            self._max_non_gripper_action_delta,
            float(action_metadata["max_non_gripper_action_delta"]),
        )
        if (
            bool(action_metadata["controller_active"])
            and self._first_controller_activation_physics_step is None
        ):
            self._first_controller_activation_physics_step = self._episode_physics_steps
            self._first_controller_activation_bilateral = bilateral_contact
            self._first_controller_activation_left_force_n = float(
                action_metadata["pre_left_force_n"]
            )
            self._first_controller_activation_right_force_n = float(
                action_metadata["pre_right_force_n"]
            )
            activation_contract_satisfied = (
                bilateral_contact
                and self._first_controller_activation_left_force_n >= contact_on_force_n
                and self._first_controller_activation_right_force_n
                >= contact_on_force_n
            )
            if self.mode == "joint_bilateral" and not activation_contract_satisfied:
                raise RuntimeError(
                    "Bilateral force control activated before bilateral tactile and "
                    "per-finger force thresholds were satisfied."
                )

        row = {
            "schema_version": 1,
            "episode_index": self.completed_episodes,
            "physics_step": self._episode_physics_steps,
            "policy_action": _tensor_row(policy_action),
            "executed_action": _tensor_row(executed_action),
            "reward": reward_value,
            "terminated": any_flag(terminated),
            "truncated": any_flag(truncated),
            "post_metrics": final_metrics,
            "attempted_force_range_n": (
                None
                if self._attempted_range_n is None
                else self._attempted_range_n.as_list()
            ),
            **action_metadata,
        }
        if self._trajectory_stream is None:
            raise RuntimeError(
                "Trajectory stream is unavailable before a physics step."
            )
        self._trajectory_stream.write(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        )
        self._capture_video_frame(terminal=any_flag(terminated) or any_flag(truncated))

    def _after_episode_complete(
        self,
        *,
        total_reward: Any,
        terminated: Any,
        truncated: Any,
        chunk_flags: dict[str, bool],
        final_metrics: dict[str, Any],
    ) -> None:
        del total_reward, chunk_flags, final_metrics
        if self._pending_episode is not None:
            raise RuntimeError("A second terminal episode arrived before commit.")
        status, success, raw_failure_reason = classify_terminal(
            self._episode_flags,
            terminated=any_flag(terminated),
            truncated=any_flag(truncated),
        )
        raw_reason = "success" if success else str(raw_failure_reason or "unknown")
        mean_force_n = self._contact_force_sum_n / max(self._loaded_contact_samples, 1)
        force_rmse_n = math.sqrt(
            self._squared_force_error_sum_n2 / max(self._loaded_contact_samples, 1)
        )
        contact_fraction = self._loaded_contact_samples / max(
            self._episode_physics_steps, 1
        )
        tactile_fraction = self._tactile_contact_samples / max(
            self._episode_physics_steps, 1
        )
        bilateral_fraction = self._bilateral_contact_samples / max(
            self._episode_physics_steps, 1
        )
        diagnosed_reason = raw_reason
        feedback = None
        if self._attempted_range_n is not None:
            diagnosed_reason = diagnose_episode_failure(
                raw_reason,
                touched=self._touched,
                contact_fraction=contact_fraction,
                bilateral_contact_fraction=bilateral_fraction,
                mean_force_n=mean_force_n,
                attempted_range_n=self._attempted_range_n,
            )
            feedback = EpisodeFeedback(
                episode_index=self.completed_episodes,
                success=success,
                failure_reason=diagnosed_reason,
                attempted_range_n=self._attempted_range_n,
                target_force_n=self._attempted_range_n.center_n,
                mean_contact_force_n=mean_force_n,
                peak_contact_force_n=self._peak_contact_force_n,
                force_rmse_n=force_rmse_n,
                contact_fraction=contact_fraction,
                max_lift_m=self._max_lift_m,
                metadata={
                    "terminal_reason": raw_reason,
                    "tactile_contact_fraction": tactile_fraction,
                    "bilateral_contact_fraction": bilateral_fraction,
                    "break_force_threshold_n": self.break_force_threshold_n,
                },
            )
        if self._trajectory_stream is None:
            raise RuntimeError("Terminal episode has no trajectory stream.")
        self._trajectory_stream.flush()
        os.fsync(self._trajectory_stream.fileno())
        self._trajectory_stream.close()
        self._close_video_writer()
        self._pending_episode = {
            "status": status,
            "success": success,
            "terminal_reason": raw_reason,
            "diagnosed_failure_reason": diagnosed_reason,
            "feedback": feedback,
            "mean_contact_force_n": mean_force_n,
            "force_rmse_n": force_rmse_n,
            "contact_fraction": contact_fraction,
            "tactile_contact_fraction": tactile_fraction,
            "bilateral_contact_fraction": bilateral_fraction,
        }

    def _save_cached_images(self, episode_index: int) -> list[Path]:
        if not self._last_episode_images:
            return []
        directory = self.output_dir / "advisor_images"
        directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for name, array in self._last_episode_images.items():
            path = directory / f"episode_{episode_index:03d}_{name}.png"
            Image.fromarray(array).save(path)
            paths.append(path)
        return paths

    def _extra_episode_result(self) -> dict[str, Any]:
        """Extension hook for method-specific episode-level diagnostics."""

        return {}

    def complete_pending_episode(
        self, *, dsrl_updates_completed: int
    ) -> dict[str, Any]:
        """Run the advisor after the terminal DSRL update and persist the result."""

        if self._pending_episode is None:
            raise RuntimeError("No terminal episode is waiting to be committed.")
        pending = self._pending_episode
        episode_index = self.completed_episodes
        feedback = pending.pop("feedback")
        decision = None
        image_paths: list[Path] = []
        if self.adaptation is not None:
            if feedback is None:
                raise RuntimeError("VLM mode completed an episode without feedback.")
            image_paths = self._save_cached_images(episode_index)
            calls_before = self.advisor_calls
            decision = self.adaptation.complete_episode(
                feedback, image_paths=image_paths
            )
            if self.advisor_calls != calls_before + 1:
                raise RuntimeError(
                    "The advisor must be called exactly once per episode."
                )
        video_path = self._finalize_episode_video(
            episode_index,
            success=bool(pending["success"]),
            reason=str(pending["diagnosed_failure_reason"]),
        )

        result = {
            "schema_version": 1,
            "episode_index": episode_index,
            "experiment_seed": self.experiment_seed,
            "reset_seed": self._active_reset_seed,
            "reset_sequence_index": episode_index,
            "mode": self.mode,
            "success": bool(pending["success"]),
            "terminal_reason": pending["terminal_reason"],
            "diagnosed_failure_reason": pending["diagnosed_failure_reason"],
            "episode_return": self._episode_return,
            "physics_steps": self._episode_physics_steps,
            "outer_interactions": self._episode_outer_interactions,
            "ending_outer_interaction": self.total_outer_interactions,
            "dsrl_updates_completed": int(dsrl_updates_completed),
            "max_lift_m": self._max_lift_m,
            "min_grasp_distance_m": (
                None
                if not math.isfinite(self._min_grasp_distance_m)
                else self._min_grasp_distance_m
            ),
            "peak_contact_force_n": self._peak_contact_force_n,
            "peak_net_contact_force_n": self._peak_net_contact_force_n,
            "terminal_tactile_actor": self._decision_tactile_actor[0]
            .detach()
            .cpu()
            .tolist(),
            "mean_contact_force_n": pending["mean_contact_force_n"],
            "force_rmse_n": pending["force_rmse_n"],
            "contact_fraction": pending["contact_fraction"],
            "tactile_contact_fraction": pending["tactile_contact_fraction"],
            "bilateral_contact_fraction": pending["bilateral_contact_fraction"],
            "controller_active_fraction": self._controller_active_steps
            / max(self._episode_physics_steps, 1),
            "safety_override_fraction": self._safety_override_steps
            / max(self._episode_physics_steps, 1),
            "max_non_gripper_action_delta": self._max_non_gripper_action_delta,
            "first_controller_activation_physics_step": (
                self._first_controller_activation_physics_step
            ),
            "first_controller_activation_bilateral": (
                self._first_controller_activation_bilateral
            ),
            "first_controller_activation_left_force_n": (
                self._first_controller_activation_left_force_n
            ),
            "first_controller_activation_right_force_n": (
                self._first_controller_activation_right_force_n
            ),
            "bilateral_activation_contract_satisfied": (
                None
                if self.mode != "joint_bilateral"
                else self._first_controller_activation_physics_step is None
                or (
                    self.controller is not None
                    and bool(self._first_controller_activation_bilateral)
                    and self._first_controller_activation_left_force_n
                    >= self.controller.config.contact_on_force_n
                    and self._first_controller_activation_right_force_n
                    >= self.controller.config.contact_on_force_n
                )
            ),
            "break_force_threshold_n": self.break_force_threshold_n,
            "attempted_force_range_n": (
                None
                if self._attempted_range_n is None
                else self._attempted_range_n.as_list()
            ),
            "vlm_recommended_force_range_n": (
                None if decision is None else decision.vlm_range_n.as_list()
            ),
            "next_force_range_n": (
                None if decision is None else decision.target_range_n.as_list()
            ),
            "update_kind": None if decision is None else decision.update_kind,
            "trajectory": str(
                self.trajectory_dir / f"episode_{episode_index:03d}.jsonl.gz"
            ),
            "advisor_images": [str(path) for path in image_paths],
            "video": None if video_path is None else str(video_path),
            "video_frame_count": self._video_frame_count,
            "video_camera": self.video_camera if video_path is not None else None,
            "video_fps": self.video_fps if video_path is not None else None,
            "video_every_n_physics_steps": (
                self.video_every_n_physics_steps if video_path is not None else None
            ),
            **self._extra_episode_result(),
        }
        self._episode_stream.write(
            json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
        )
        self._episode_stream.flush()
        os.fsync(self._episode_stream.fileno())
        self.results.append(result)
        self.completed_episodes += 1
        self._pending_episode = None
        temporary = self.output_dir / "results.partial.json.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "completed_episodes": self.completed_episodes,
                    "results": self.results,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.output_dir / "results.partial.json")
        return result

    def begin_auto_reset_episode(self) -> None:
        """Start logging the episode already reset internally by IsaacLab."""

        self._active_reset_seed = None
        self._after_episode_reset()

    def _step_noise(self, policy_action: torch.Tensor | None):
        self.total_outer_interactions += 1
        self._episode_outer_interactions += 1
        return super()._step_noise(policy_action)

    def close(self):
        self._close_video_writer()
        if self._trajectory_stream is not None and not self._trajectory_stream.closed:
            self._trajectory_stream.flush()
            os.fsync(self._trajectory_stream.fileno())
            self._trajectory_stream.close()
        if not self._episode_stream.closed:
            self._episode_stream.flush()
            os.fsync(self._episode_stream.fileno())
            self._episode_stream.close()
        return super().close()


__all__ = ["ExperimentMode", "VLMDSRLLabPickWrapper"]
