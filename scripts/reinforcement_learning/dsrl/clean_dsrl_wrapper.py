"""LabPick wrapper for base-anchored Flow-noise DSRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .clean_dsrl_sac import CleanDSRLLayout
    from .flow_matching_noise_adapter import FlowMatchingNoiseAdapter
    from .online_dsrl_metrics import (
        FAILURE_FLAG_KEYS,
        OnlineDSRLJSONLLogger,
        any_flag,
        classify_terminal,
        extract_step_metrics,
        scalar,
        utc_now,
    )
    from .tactile_observation import TACTILE_ACTOR_DIM, build_tactile_actor_from_env
except ImportError:
    from clean_dsrl_sac import CleanDSRLLayout
    from flow_matching_noise_adapter import FlowMatchingNoiseAdapter
    from online_dsrl_metrics import (
        FAILURE_FLAG_KEYS,
        OnlineDSRLJSONLLogger,
        any_flag,
        classify_terminal,
        extract_step_metrics,
        scalar,
        utc_now,
    )
    from tactile_observation import TACTILE_ACTOR_DIM, build_tactile_actor_from_env


FLOW_HORIZON = 32
FLOW_NOISE_DIM = 10
DEFAULT_EXECUTE_STEPS = 32
ACTION_REPEAT = 2
MAX_OBSERVATION_UPDATES = 32


def _validate_matrix(tensor: torch.Tensor, *, name: str, width: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.ndim != 2 or tensor.shape[-1] != width:
        raise ValueError(f"{name} must have shape [B, {width}], got {tuple(tensor.shape)}.")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype, received {tensor.dtype}.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values.")


def pack_dsrl_critic_state(
    normalized_proprioception: torch.Tensor,
    relative_object_position: torch.Tensor,
    object_rot6d: torch.Tensor,
) -> torch.Tensor:
    """Pack the same 19-D privileged state used by clean residual SAC."""

    fields = (
        (normalized_proprioception, "normalized_proprioception", 10),
        (relative_object_position, "relative_object_position", 3),
        (object_rot6d, "object_rot6d", 6),
    )
    for tensor, name, width in fields:
        _validate_matrix(tensor, name=name, width=width)
    if len({tensor.shape[0] for tensor, _, _ in fields}) != 1:
        raise ValueError("All critic-state fields must have the same batch size.")
    return torch.cat(
        (
            normalized_proprioception,
            relative_object_position.to(normalized_proprioception),
            object_rot6d.to(normalized_proprioception),
        ),
        dim=-1,
    )


class CleanDSRLLabPickWrapper(gym.Wrapper):
    """Decode a short SAC noise action through a frozen Flow policy.

    Each outer step samples the frozen Flow policy's native Gaussian noise,
    expands the learned correction with ``repeat_last``, and adds a bounded
    scaled correction before decoding. A zero correction is exactly native BC.
    The actor receives frozen Flow encoder features; the critic additionally
    receives the simulator-only 19-D object/proprioception state.
    """

    flow_horizon = FLOW_HORIZON
    flow_noise_dim = FLOW_NOISE_DIM
    action_repeat = ACTION_REPEAT

    def __init__(
        self,
        env: gym.Env,
        policy_checkpoint: str | Path,
        *,
        device: str = "cuda",
        learned_noise_steps: int = 1,
        padding_mode: str = "repeat_last",
        noise_residual_scale: float = 0.25,
        chunk_execute_steps: int = DEFAULT_EXECUTE_STEPS,
        chunk_discount: float = 0.99,
        flow_num_inference_steps: int = 20,
        phase_horizon_steps: int = 383,
        camera_warmup_steps: int = 8,
        use_visual_xy_override: bool = True,
        online_metrics_dir: str | Path | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        base = env.unwrapped
        if int(getattr(base, "num_envs", 0)) != 1:
            raise ValueError("Clean DSRL currently requires exactly one environment.")
        cfg = getattr(base, "cfg", None)
        if cfg is None or bool(getattr(cfg, "rl_align_cafe_action_yaw", True)):
            raise ValueError(
                "Clean DSRL requires rl_align_cafe_action_yaw=False so the "
                "decoded Flow action remains unmodified."
            )
        if not callable(getattr(base, "get_cafe_observation", None)):
            raise TypeError("The wrapped environment must implement get_cafe_observation().")
        if not callable(getattr(base, "get_privileged_object_pose", None)):
            raise TypeError("The wrapped environment must implement get_privileged_object_pose().")
        if not callable(getattr(base, "tactile_contact_depths", None)):
            raise TypeError("The wrapped environment must implement tactile_contact_depths().")
        if not hasattr(base, "has_touched"):
            raise TypeError("The wrapped environment must expose has_touched.")
        if not isinstance(policy_checkpoint, (str, Path)) or not str(policy_checkpoint):
            raise TypeError("policy_checkpoint must be a non-empty path string.")
        for value, name, minimum in (
            (learned_noise_steps, "learned_noise_steps", 1),
            (chunk_execute_steps, "chunk_execute_steps", 1),
            (flow_num_inference_steps, "flow_num_inference_steps", 1),
            (phase_horizon_steps, "phase_horizon_steps", 1),
            (camera_warmup_steps, "camera_warmup_steps", 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}.")
        if chunk_execute_steps > self.flow_horizon:
            raise ValueError("chunk_execute_steps cannot exceed the Flow horizon.")
        if not 0.0 < chunk_discount <= 1.0:
            raise ValueError("chunk_discount must lie in (0, 1].")

        self.chunk_execute_steps = int(chunk_execute_steps)
        self.chunk_discount = float(chunk_discount)
        self.phase_horizon_steps = int(phase_horizon_steps)
        self.camera_warmup_steps = int(camera_warmup_steps)
        self.break_force_threshold_n = float(
            getattr(cfg, "terminate_break_force_threshold_n", float("nan"))
        )
        flow_noise_seed = 0 if seed is None else int(seed)
        self.adapter = FlowMatchingNoiseAdapter.from_pretrained(
            policy_checkpoint,
            device=device,
            num_inference_steps=int(flow_num_inference_steps),
            visual_xy_lock_phase=0.30,
            use_visual_xy_override=bool(use_visual_xy_override),
            seed=flow_noise_seed,
        )
        adapter_horizon = int(getattr(self.adapter, "n_action_steps", -1))
        adapter_action_dim = int(getattr(self.adapter, "action_dim", -1))
        if adapter_horizon != self.flow_horizon:
            raise ValueError(
                f"Clean DSRL requires a Flow horizon of {self.flow_horizon}, "
                f"received {adapter_horizon}."
            )
        if adapter_action_dim != self.flow_noise_dim:
            raise ValueError(
                f"Clean DSRL requires {self.flow_noise_dim}-D Flow noise, "
                f"received {adapter_action_dim}."
            )
        self._layout = CleanDSRLLayout(
            policy=int(self.adapter.observation_dim) + TACTILE_ACTOR_DIM,
            noise_dim=adapter_action_dim,
            flow_horizon=adapter_horizon,
            learned_noise_steps=int(learned_noise_steps),
            padding_mode=padding_mode,
            residual_scale=float(noise_residual_scale),
        )
        self._flow_policy_steps = 0
        self._flow_needs_warmup = True
        self.last_decoded_action_chunk: torch.Tensor | None = None
        self._last_tactile_actor = torch.zeros(
            (self.num_envs, TACTILE_ACTOR_DIM), device=self.device, dtype=torch.float32
        )
        self._online_logger = (
            None
            if online_metrics_dir is None
            else OnlineDSRLJSONLLogger(online_metrics_dir)
        )
        self._online_interaction_index = 0
        self._online_episode_index = 1
        self._reset_online_episode_state()

        action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.layout.action_dim,),
            dtype=np.float32,
        )
        policy_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.layout.policy_dim,),
            dtype=np.float32,
        )
        critic_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.layout.state_dim,),
            dtype=np.float32,
        )
        self.action_space = action_space
        self.single_action_space = action_space
        base.single_action_space = action_space
        self.single_observation_space = gym.spaces.Dict(
            {"policy": policy_space, "critic": critic_space}
        )
        base.single_observation_space = self.single_observation_space
        base.observation_space = gym.vector.utils.batch_space(policy_space, self.num_envs)
        base.state_space = gym.vector.utils.batch_space(critic_space, self.num_envs)
        self.observation_space = base.observation_space
        self.state_space = base.state_space

    @property
    def device(self) -> torch.device | str:
        return self.env.unwrapped.device

    @property
    def num_envs(self) -> int:
        return int(self.env.unwrapped.num_envs)

    @property
    def layout(self) -> CleanDSRLLayout:
        return self._layout

    @property
    def outer_discount_factor(self) -> float:
        """Bootstrap discount for a full non-terminal decoded prefix."""

        return self.chunk_discount**self.chunk_execute_steps

    def _current_phase(self) -> float:
        return min(self._flow_policy_steps / float(self.phase_horizon_steps), 1.0)

    def _camera_numpy(self, camera: Any) -> np.ndarray:
        rgb = camera.data.output["rgb"][:, :, :, :3].permute(0, 3, 1, 2).float()
        rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        return rgb[0].permute(1, 2, 0).clamp(0, 255).byte().detach().cpu().numpy()

    def _raw_flow_observation(self) -> dict[str, Any]:
        base = self.env.unwrapped
        cafe = base.get_cafe_observation()
        proprioception = cafe["robot0_pos"]
        _validate_matrix(proprioception, name="physical_proprioception", width=10)
        return {
            "robot0_pos": proprioception[0].detach().cpu().numpy().astype(np.float32),
            "robot0_image": self._camera_numpy(base.wrist_camera),
            "robot0_image_third": self._camera_numpy(base.third_person_camera),
            "phase": self._current_phase(),
        }

    def _warmup_cameras(self) -> None:
        if self.camera_warmup_steps == 0:
            return
        base = self.env.unwrapped
        hold_action = base.get_cafe_observation()["robot0_pos"].clone()
        for _ in range(self.camera_warmup_steps):
            base._pre_physics_step(hold_action)
            base._apply_action()
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(dt=base.physics_dt)
            base.sim.render()
        base.step_count.zero_()
        base.has_touched.zero_()

    def _ensure_flow_observation_ready(self) -> None:
        if self._flow_needs_warmup:
            self._warmup_cameras()
            self._flow_needs_warmup = False
        for _ in range(MAX_OBSERVATION_UPDATES):
            if self.adapter.is_ready:
                return
            self.adapter.update(self._raw_flow_observation())
        if not self.adapter.is_ready:
            raise RuntimeError(
                "Flow adapter did not become ready after "
                f"{MAX_OBSERVATION_UPDATES} observation updates."
            )

    def _critic_state(self) -> torch.Tensor:
        base = self.env.unwrapped
        physical_proprioception = base.get_cafe_observation()["robot0_pos"].to(
            device=self.device,
            dtype=torch.float32,
        )
        normalized_proprioception = self.adapter.normalize_proprioception(
            physical_proprioception,
            phase=self._current_phase(),
        ).to(device=self.device, dtype=torch.float32)
        relative_position, object_rot6d = base.get_privileged_object_pose()
        state = pack_dsrl_critic_state(
            normalized_proprioception,
            relative_position.to(device=self.device, dtype=torch.float32),
            object_rot6d.to(device=self.device, dtype=torch.float32),
        )
        self.layout.validate_states(state)
        return state

    def _policy_observation(self) -> dict[str, torch.Tensor]:
        self._ensure_flow_observation_ready()
        flow_condition_embedding = self.adapter.encode_observation().to(
            device=self.device,
            dtype=torch.float32,
        )
        _validate_matrix(
            flow_condition_embedding,
            name="flow_condition_embedding",
            width=self.layout.flow_condition_dim,
        )
        tactile_actor = build_tactile_actor_from_env(self.env.unwrapped).to(
            device=self.device, dtype=torch.float32
        )
        self._last_tactile_actor = tactile_actor.detach().clone()
        policy = torch.cat((flow_condition_embedding, tactile_actor), dim=-1)
        self.layout.validate_policy_observations(policy)
        return {"policy": policy, "critic": self._critic_state()}

    @staticmethod
    def _episode_done(terminated: Any, truncated: Any) -> bool:
        done = terminated | truncated
        if isinstance(done, torch.Tensor):
            return bool(done.any().item())
        return bool(np.asarray(done).any())

    def _reset_flow_after_episode(self) -> None:
        self.adapter.reset()
        self._flow_policy_steps = 0
        self._flow_needs_warmup = True

    def _reset_online_episode_state(self) -> None:
        self._online_episode_step = 0
        self._online_episode_return = 0.0
        self._online_episode_action_steps = 0
        self._online_episode_physics_steps = 0
        self._online_episode_peak_contact_force_n = 0.0
        self._online_episode_peak_net_contact_force_n = 0.0

    def _after_episode_reset(self) -> None:
        """Extension hook called after the physical environment is reset."""

    def _transform_physical_action(
        self, policy_action: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Extension hook for high-rate action arbitration.

        The clean DSRL path is intentionally an identity transform. Specialized
        wrappers may replace selected physical action dimensions while keeping
        the decoded DSRL action available for replay and diagnostics.
        """

        return policy_action, {}

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
        """Extension hook called once after every physical interaction step."""

    def _after_episode_complete(
        self,
        *,
        total_reward: Any,
        terminated: Any,
        truncated: Any,
        chunk_flags: dict[str, bool],
        final_metrics: dict[str, Any],
    ) -> None:
        """Extension hook called exactly once after a terminal transition."""

    def _log_online_transition(
        self,
        *,
        reward: Any,
        terminated: Any,
        truncated: Any,
        executed_actions: int,
        physics_steps: int,
        policy_noise_rms: Any,
        decoder_noise_rms: Any,
        chunk_flags: dict[str, bool],
        chunk_peak_contact_force_n: float,
        chunk_peak_net_contact_force_n: float,
        final_metrics: dict[str, Any],
    ) -> None:
        if self._online_logger is None:
            return

        self._online_interaction_index += 1
        self._online_episode_step += 1
        reward_value = scalar(reward)
        self._online_episode_return += reward_value
        self._online_episode_action_steps += executed_actions
        self._online_episode_physics_steps += physics_steps
        self._online_episode_peak_contact_force_n = max(
            self._online_episode_peak_contact_force_n,
            chunk_peak_contact_force_n,
        )
        self._online_episode_peak_net_contact_force_n = max(
            self._online_episode_peak_net_contact_force_n,
            chunk_peak_net_contact_force_n,
        )
        terminated_value = any_flag(terminated)
        truncated_value = any_flag(truncated)
        terminal = terminated_value or truncated_value
        status, success, failure_reason = classify_terminal(
            chunk_flags,
            terminated=terminated_value,
            truncated=truncated_value,
        )
        if not terminal:
            status = "ongoing"
            success_value: bool | None = None
            failure_reason = None
            terminal_reason = None
        else:
            success_value = success
            terminal_reason = "success" if success else failure_reason

        interaction_row = {
            "schema_version": 1,
            "recorded_at_utc": utc_now(),
            "interaction_index": self._online_interaction_index,
            "episode_index": self._online_episode_index,
            "episode_step": self._online_episode_step,
            "status": status,
            "terminal": terminal,
            "terminated": terminated_value,
            "truncated": truncated_value,
            "success": success_value,
            "failure_reason": failure_reason,
            "terminal_reason": terminal_reason,
            "reward": reward_value,
            "episode_return_so_far": self._online_episode_return,
            "action_steps_executed": executed_actions,
            "physics_steps_executed": physics_steps,
            "episode_action_steps_so_far": self._online_episode_action_steps,
            "episode_physics_steps_so_far": self._online_episode_physics_steps,
            "policy_noise_rms": scalar(policy_noise_rms),
            "decoder_noise_rms": scalar(decoder_noise_rms),
            "chunk_peak_contact_force_n": chunk_peak_contact_force_n,
            "chunk_peak_net_contact_force_n": chunk_peak_net_contact_force_n,
            "episode_peak_contact_force_n": self._online_episode_peak_contact_force_n,
            "episode_peak_net_contact_force_n": self._online_episode_peak_net_contact_force_n,
            "final_contact_force_n": final_metrics["contact_force_n"],
            "final_net_contact_force_n": final_metrics["net_contact_force_n"],
            "final_lift_m": final_metrics["lift_m"],
            "final_grasp_distance_m": final_metrics["grasp_distance_m"],
            "break_force_threshold_n": self.break_force_threshold_n,
            "terminal_flags": dict(chunk_flags),
            "tactile_actor": self._decision_tactile_actor[0].detach().cpu().tolist(),
        }
        self._online_logger.log_interaction(interaction_row)

        if terminal:
            episode_row = {
                "schema_version": 1,
                "recorded_at_utc": interaction_row["recorded_at_utc"],
                "episode_index": self._online_episode_index,
                "ending_interaction_index": self._online_interaction_index,
                "num_interactions": self._online_episode_step,
                "status": status,
                "success": success_value,
                "failure_reason": failure_reason,
                "terminal_reason": terminal_reason,
                "terminated": terminated_value,
                "truncated": truncated_value,
                "episode_return": self._online_episode_return,
                "action_steps_executed": self._online_episode_action_steps,
                "physics_steps_executed": self._online_episode_physics_steps,
                "peak_contact_force_n": self._online_episode_peak_contact_force_n,
                "peak_net_contact_force_n": self._online_episode_peak_net_contact_force_n,
                "break_force_threshold_n": self.break_force_threshold_n,
                "terminal_flags": dict(chunk_flags),
                "terminal_tactile_actor": self._decision_tactile_actor[0]
                .detach()
                .cpu()
                .tolist(),
            }
            self._online_logger.log_episode(episode_row)
            self._online_episode_index += 1
            self._reset_online_episode_state()

    @torch.inference_mode()
    def reset(self, **kwargs):
        flow_noise_seed = kwargs.pop("flow_noise_seed", None)
        if flow_noise_seed is not None:
            if isinstance(flow_noise_seed, bool) or not isinstance(flow_noise_seed, int):
                raise TypeError("flow_noise_seed must be an integer or None.")
            if flow_noise_seed < 0:
                raise ValueError("flow_noise_seed must be non-negative.")
        self._reset_flow_after_episode()
        if flow_noise_seed is not None:
            generator = getattr(self.adapter.runner, "generator", None)
            if generator is None:
                raise RuntimeError("A native Flow noise seed requires a seeded policy runner.")
            generator.manual_seed(flow_noise_seed)
        _, info = self.env.reset(**kwargs)
        self._after_episode_reset()
        return self._policy_observation(), info

    @torch.inference_mode()
    def _step_noise(self, policy_action: torch.Tensor | None):
        self._ensure_flow_observation_ready()
        self._decision_tactile_actor = self._last_tactile_actor.detach().clone()
        native_noise = self.adapter.sample_native_noise(batch_size=self.num_envs)
        if policy_action is None:
            decoder_noise = native_noise
            policy_noise_rms = torch.zeros((), device=self.device)
        else:
            policy_action = torch.as_tensor(
                policy_action,
                dtype=torch.float32,
                device=self.device,
            )
            if policy_action.ndim == 1:
                policy_action = policy_action.unsqueeze(0)
            self.layout.validate_actions(policy_action, enforce_bounds=True)
            if policy_action.shape[0] != self.num_envs:
                raise ValueError(
                    f"DSRL action batch must equal num_envs={self.num_envs}, "
                    f"received {policy_action.shape[0]}."
                )
            decoder_noise = self.layout.compose_noise(native_noise, policy_action)
            policy_noise_rms = policy_action.square().mean().sqrt().detach()
        action_chunk = self.adapter.decode(decoder_noise.reshape(self.num_envs, -1))
        native_noise_rms = native_noise.square().mean().sqrt().detach()
        decoder_noise_rms = decoder_noise.square().mean().sqrt().detach()

        _validate_matrix(action_chunk, name="decoded_action_chunk", width=10)
        if action_chunk.shape[0] != self.flow_horizon:
            raise ValueError(
                f"Flow decoder must return [{self.flow_horizon}, 10], "
                f"received {tuple(action_chunk.shape)}."
            )
        self.last_decoded_action_chunk = action_chunk.detach().clone()

        total_reward = None
        terminated = truncated = info = None
        executed_actions = 0
        physics_steps = 0
        episode_done = False
        chunk_flags = {reason: False for reason, _ in FAILURE_FLAG_KEYS}
        chunk_flags.update({"success": False, "timeout": False})
        chunk_peak_contact_force_n = 0.0
        chunk_peak_net_contact_force_n = 0.0
        final_metrics: dict[str, Any] | None = None
        for action_index in range(self.chunk_execute_steps):
            physical_action = action_chunk[action_index].reshape(1, -1).to(
                device=self.device,
                dtype=torch.float32,
            )
            reward_for_action = None
            for _ in range(self.action_repeat):
                executed_action, action_metadata = self._transform_physical_action(
                    physical_action
                )
                action_metadata = dict(action_metadata)
                action_metadata["tactile_actor"] = (
                    self._decision_tactile_actor[0].detach().cpu().tolist()
                )
                action_metadata["outer_policy_decision"] = action_index == 0
                _validate_matrix(
                    executed_action,
                    name="executed_physical_action",
                    width=10,
                )
                if executed_action.shape != physical_action.shape:
                    raise ValueError(
                        "Physical action transform must preserve the action shape, "
                        f"received {tuple(executed_action.shape)} instead of "
                        f"{tuple(physical_action.shape)}."
                    )
                _, reward, terminated, truncated, info = self.env.step(executed_action)
                final_metrics = extract_step_metrics(info)
                self._after_physics_step(
                    policy_action=physical_action,
                    executed_action=executed_action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    action_metadata=action_metadata,
                    final_metrics=final_metrics,
                )
                for name, value in final_metrics["flags"].items():
                    chunk_flags[name] |= bool(value)
                chunk_peak_contact_force_n = max(
                    chunk_peak_contact_force_n,
                    final_metrics["contact_force_n"],
                )
                chunk_peak_net_contact_force_n = max(
                    chunk_peak_net_contact_force_n,
                    final_metrics["net_contact_force_n"],
                )
                reward_for_action = reward if reward_for_action is None else reward_for_action + reward
                physics_steps += 1
                episode_done = self._episode_done(terminated, truncated)
                if episode_done:
                    break
            if reward_for_action is None:
                raise RuntimeError("Clean DSRL wrapper executed no physics step.")
            discounted_reward = (self.chunk_discount**action_index) * reward_for_action
            total_reward = discounted_reward if total_reward is None else total_reward + discounted_reward
            executed_actions += 1
            self._flow_policy_steps += 1
            if episode_done:
                self._reset_flow_after_episode()
                break
            self.adapter.update(self._raw_flow_observation())

        if (
            total_reward is None
            or terminated is None
            or truncated is None
            or info is None
            or final_metrics is None
        ):
            raise RuntimeError("Clean DSRL wrapper executed no decoded action.")

        info = dict(info)
        info["clean_dsrl/action_steps_executed"] = executed_actions
        info["clean_dsrl/physics_steps_executed"] = physics_steps
        info["clean_dsrl/policy_noise_rms"] = policy_noise_rms
        info["clean_dsrl/native_noise_rms"] = native_noise_rms
        info["clean_dsrl/decoder_noise_rms"] = decoder_noise_rms
        info["tactile_actor"] = self._decision_tactile_actor.detach().clone()
        log = dict(info.get("log", {}))
        log["CleanDSRL/action_steps_executed"] = float(executed_actions)
        log["CleanDSRL/policy_noise_rms"] = policy_noise_rms
        log["CleanDSRL/native_noise_rms"] = native_noise_rms
        log["CleanDSRL/decoder_noise_rms"] = decoder_noise_rms
        info["log"] = log
        self._log_online_transition(
            reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            executed_actions=executed_actions,
            physics_steps=physics_steps,
            policy_noise_rms=policy_noise_rms,
            decoder_noise_rms=decoder_noise_rms,
            chunk_flags=chunk_flags,
            chunk_peak_contact_force_n=chunk_peak_contact_force_n,
            chunk_peak_net_contact_force_n=chunk_peak_net_contact_force_n,
            final_metrics=final_metrics,
        )
        if self._episode_done(terminated, truncated):
            self._after_episode_complete(
                total_reward=total_reward,
                terminated=terminated,
                truncated=truncated,
                chunk_flags=chunk_flags,
                final_metrics=final_metrics,
            )
        terminated_for_learning = terminated | truncated
        return (
            self._policy_observation(),
            total_reward,
            terminated_for_learning,
            truncated,
            info,
        )

    def step(self, policy_action):
        return self._step_noise(policy_action)

    def step_bc(self):
        """Execute the exact zero-correction frozen-BC branch."""

        return self._step_noise(None)

    def close(self):
        if self._online_logger is not None:
            self._online_logger.close()
        return super().close()


__all__ = [
    "ACTION_REPEAT",
    "CleanDSRLLabPickWrapper",
    "DEFAULT_EXECUTE_STEPS",
    "FLOW_HORIZON",
    "FLOW_NOISE_DIM",
    "pack_dsrl_critic_state",
]
