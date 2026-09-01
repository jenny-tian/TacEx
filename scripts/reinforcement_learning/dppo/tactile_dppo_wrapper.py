"""Isaac Lab observation/action bridge for tactile DPPO."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import torch.nn.functional as F

try:
    from runtime import disable_optional_transformers_discovery
except ImportError:
    from scripts.reinforcement_learning.dsrl.runtime import disable_optional_transformers_discovery

disable_optional_transformers_discovery()

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

try:
    from tactile_observation import build_tactile_actor_from_env
except ImportError:
    from scripts.reinforcement_learning.dsrl.tactile_observation import build_tactile_actor_from_env

try:
    from tactile_dppo import TactileDPPO
except ImportError:
    from scripts.reinforcement_learning.dppo.tactile_dppo import TactileDPPO


MATCHED_CAMERA_SHAPE = (224, 224)
DIFFUSION_BC_CAMERA_CONTRACT = "matched_full_frame_visual_xy_residual_224x224_v2"


class TactileDPPOLabPickWrapper(gym.Wrapper):
    """Execute one normalized Diffusion action chunk per outer transition."""

    action_repeat = 2

    def __init__(
        self,
        env: gym.Env,
        policy_checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        camera_warmup_steps: int = 8,
        chunk_discount: float = 0.99,
        phase_horizon_steps: int = 383,
        visual_xy_lock_phase: float | None = 0.30,
        dppo_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(env)
        base = env.unwrapped
        if int(getattr(base, "num_envs", 0)) != 1:
            raise ValueError("Tactile DPPO requires exactly one Isaac environment.")
        if not callable(getattr(base, "get_cafe_observation", None)):
            raise TypeError("LabPick environment must expose get_cafe_observation().")
        if not callable(getattr(base, "tactile_contact_depths", None)):
            raise TypeError("LabPick environment must expose tactile_contact_depths().")
        if not hasattr(base, "has_touched"):
            raise TypeError("LabPick environment must expose has_touched.")
        if camera_warmup_steps < 0:
            raise ValueError("camera_warmup_steps cannot be negative.")
        if not 0.0 < chunk_discount <= 1.0:
            raise ValueError("chunk_discount must be in (0, 1].")
        if phase_horizon_steps < 1:
            raise ValueError("phase_horizon_steps must be positive.")
        if visual_xy_lock_phase is not None and not 0.0 <= visual_xy_lock_phase <= 1.0:
            raise ValueError("visual_xy_lock_phase must lie in [0,1] or be None.")

        checkpoint = str(Path(policy_checkpoint).expanduser().resolve())
        policy = DiffusionPolicy.from_pretrained(checkpoint).to(device)
        if policy.config.noise_scheduler_type != "DDPM":
            raise ValueError("Tactile DPPO runtime requires a DDPM Diffusion BC checkpoint.")
        crop_shape = (
            None
            if policy.config.crop_shape is None
            else tuple(int(value) for value in policy.config.crop_shape)
        )
        if crop_shape != MATCHED_CAMERA_SHAPE or bool(policy.config.crop_is_random):
            raise ValueError(
                "Tactile DPPO requires the matched deterministic full-frame Diffusion BC "
                f"camera contract {DIFFUSION_BC_CAMERA_CONTRACT!r}; received "
                f"crop_shape={crop_shape}, crop_is_random={policy.config.crop_is_random}."
            )
        if not bool(getattr(policy.config, "use_visual_xy_residual", False)):
            raise ValueError(
                "Tactile DPPO requires the matched visual-x/y residual Diffusion BC contract."
            )
        camera_count = len(policy.config.image_features)
        if int(policy.config.visual_xy_camera_index) % camera_count != camera_count - 1:
            raise ValueError("Matched DPPO requires the final (third-person) camera for visual x/y.")
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": device}},
        )
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.agent = TactileDPPO(policy, **(dppo_kwargs or {})).to(device)
        self.camera_warmup_steps = int(camera_warmup_steps)
        self.chunk_discount = float(chunk_discount)
        self.phase_horizon_steps = int(phase_horizon_steps)
        self.visual_xy_lock_phase = visual_xy_lock_phase
        self._policy_steps = 0
        self._locked_visual_xy: torch.Tensor | None = None
        self._history: dict[str, deque[torch.Tensor]] = {
            OBS_STATE: deque(maxlen=int(policy.config.n_obs_steps)),
            OBS_IMAGES: deque(maxlen=int(policy.config.n_obs_steps)),
        }
        self._last_observation: dict[str, torch.Tensor] | None = None
        self._last_tactile = torch.zeros((1, 5), device=device, dtype=torch.float32)

    def _reset_visual_xy_state(self) -> None:
        self._policy_steps = 0
        self._locked_visual_xy = None

    @property
    def device(self) -> torch.device | str:
        return self.env.unwrapped.device

    @property
    def num_envs(self) -> int:
        return int(self.env.unwrapped.num_envs)

    @property
    def policy(self) -> DiffusionPolicy:
        return self.agent.base_policy

    def _camera_tensor(self, camera: Any) -> torch.Tensor:
        rgb = camera.data.output["rgb"][:, :, :, :3].permute(0, 3, 1, 2).float()
        rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        return (rgb / 255.0).clamp(0.0, 1.0)

    def _raw_observation(self) -> dict[str, torch.Tensor]:
        base = self.env.unwrapped
        return {
            "observation.state": base.get_cafe_observation()["robot0_pos"],
            "observation.images.rgb": self._camera_tensor(base.wrist_camera),
            "observation.images.rgb_third": self._camera_tensor(base.third_person_camera),
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

    def _append_observation(self, *, initialize: bool = False, refresh: bool = True) -> None:
        processed = self.preprocessor(self._raw_observation())
        state = processed[OBS_STATE]
        images = torch.stack(
            [processed[key] for key in self.policy.config.image_features], dim=-4
        )
        if initialize:
            for history in self._history.values():
                history.clear()
            for _ in range(int(self.policy.config.n_obs_steps)):
                self._history[OBS_STATE].append(state.detach().clone())
                self._history[OBS_IMAGES].append(images.detach().clone())
        else:
            self._history[OBS_STATE].append(state.detach())
            self._history[OBS_IMAGES].append(images.detach())
        if refresh:
            self._refresh_observation()

    def _refresh_observation(self) -> None:
        if any(len(history) != int(self.policy.config.n_obs_steps) for history in self._history.values()):
            raise RuntimeError("DPPO observation history is incomplete.")
        processed_history = {
            key: torch.stack(list(history), dim=1) for key, history in self._history.items()
        }
        global_condition, visual_xy = self.agent.encode_observation(processed_history)
        phase = min(self._policy_steps / float(self.phase_horizon_steps), 1.0)
        if (
            self.visual_xy_lock_phase is not None
            and phase >= self.visual_xy_lock_phase
            and self._locked_visual_xy is None
        ):
            self._locked_visual_xy = visual_xy.detach().clone()
        selected_visual_xy = (
            self._locked_visual_xy if self._locked_visual_xy is not None else visual_xy
        )
        global_condition = self.agent.condition_with_visual_xy(
            global_condition, selected_visual_xy
        )
        tactile = build_tactile_actor_from_env(self.env.unwrapped).to(
            device=global_condition.device, dtype=torch.float32
        )
        self._last_tactile = tactile.detach().clone()
        self._last_observation = {
            "global_condition": global_condition,
            "tactile_actor": tactile,
            "visual_xy": selected_visual_xy,
        }

    def current_observation(self) -> dict[str, torch.Tensor]:
        if self._last_observation is None:
            raise RuntimeError("Call reset() before requesting the DPPO observation.")
        return {key: value.detach().clone() for key, value in self._last_observation.items()}

    @torch.inference_mode()
    def reset(self, **kwargs: Any):
        _, info = self.env.reset(**kwargs)
        self._reset_visual_xy_state()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self._warmup_cameras()
        self._append_observation(initialize=True)
        return self.current_observation(), info

    @staticmethod
    def _done(terminated: Any, truncated: Any) -> bool:
        return bool(torch.as_tensor(terminated | truncated).any().item())

    @torch.inference_mode()
    def step(self, normalized_action_chunk: torch.Tensor):
        chunk = torch.as_tensor(
            normalized_action_chunk, device=self.agent.device, dtype=torch.float32
        )
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)
        expected = (
            1,
            int(self.policy.config.n_action_steps),
            int(self.policy.config.action_feature.shape[0]),
        )
        if tuple(chunk.shape) != expected:
            raise ValueError(f"DPPO action chunk must have shape {expected}, got {tuple(chunk.shape)}.")
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("DPPO action chunk contains non-finite values.")

        decision_tactile = self._last_tactile.detach().clone()
        total_reward = None
        terminated = truncated = info = None
        physics_steps = 0
        action_steps = 0
        episode_done = False
        for action_index in range(expected[1]):
            physical_action = self.postprocessor(chunk[:, action_index]).to(self.device)
            action_steps += 1
            self._policy_steps += 1
            for _ in range(self.action_repeat):
                _, reward, terminated, truncated, info = self.env.step(physical_action)
                discounted = (self.chunk_discount**action_index) * reward
                total_reward = discounted if total_reward is None else total_reward + discounted
                physics_steps += 1
                episode_done = self._done(terminated, truncated)
                if episode_done:
                    break
            if episode_done:
                break
            self._append_observation(refresh=False)

        if total_reward is None or terminated is None or truncated is None or info is None:
            raise RuntimeError("DPPO wrapper executed no physical action.")
        if episode_done:
            # DirectRLEnv has already reset the physical task.
            self.preprocessor.reset()
            self.postprocessor.reset()
            self._reset_visual_xy_state()
            self._warmup_cameras()
            self._append_observation(initialize=True)
        else:
            self._refresh_observation()

        info = dict(info)
        info["dppo/action_steps_executed"] = action_steps
        info["dppo/action_repeat"] = self.action_repeat
        info["dppo/physics_steps_executed"] = physics_steps
        info["tactile_actor"] = decision_tactile
        return (
            self.current_observation(),
            total_reward,
            terminated | truncated,
            truncated,
            info,
        )


__all__ = [
    "DIFFUSION_BC_CAMERA_CONTRACT",
    "MATCHED_CAMERA_SHAPE",
    "TactileDPPOLabPickWrapper",
]
