from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .runtime import disable_optional_transformers_discovery
except ImportError:
    from runtime import disable_optional_transformers_discovery

disable_optional_transformers_discovery()

try:
    from .diffusion_noise_adapter import DiffusionNoiseAdapter
except ImportError:
    from diffusion_noise_adapter import DiffusionNoiseAdapter

try:
    from .flow_matching_noise_adapter import FlowMatchingNoiseAdapter
except ImportError:
    from flow_matching_noise_adapter import FlowMatchingNoiseAdapter


class LabPickDSRLWrapper(gym.Wrapper):
    """Turn a LabPick CAFE-action environment into a DSRL noise environment.

    One outer environment step corresponds to one SAC action and up to
    ``n_action_steps`` physical simulation steps decoded by a frozen BC policy.
    SAC can either control the full initial-noise tensor or a low-dimensional
    Cartesian/gripper residual applied to the native Flow Matching chunk. The
    first integration is intentionally single-environment because
    LeRobot's action/observation queues do not independently reset individual
    vector slots and LabPick camera/tactile rendering is already GPU-heavy.
    """

    def __init__(
        self,
        env: gym.Env,
        policy_checkpoint: str,
        *,
        device: str = "cuda",
        noise_magnitude: float = 1.5,
        chunk_discount: float = 0.99,
        action_repeat: int = 2,
        policy_type: str = "auto",
        action_mode: str = "noise",
        residual_position_scale_m: tuple[float, float, float] = (0.03, 0.03, 0.01),
        residual_width_scale_m: float = 0.002,
        curriculum_steps: int = 0,
        curriculum_start_step: int = 0,
        curriculum_start_xy_m: tuple[float, float] = (0.05, 0.05),
        curriculum_end_xy_m: tuple[float, float] = (0.10, 0.10),
        curriculum_start_yaw_rad: float = 0.5235987756,
        curriculum_end_yaw_rad: float = 0.7853981634,
        flow_num_inference_steps: int = 20,
        flow_chunk_execute_steps: int = 32,
        flow_phase_horizon_steps: int = 383,
        flow_camera_warmup_steps: int = 8,
    ) -> None:
        super().__init__(env)
        if int(env.unwrapped.num_envs) != 1:
            raise ValueError("LabPick DSRL currently requires --num_envs 1.")
        if policy_type not in {"auto", "diffusion", "flow_matching"}:
            raise ValueError(f"Unsupported DSRL policy type: {policy_type}")
        if policy_type == "auto":
            policy_type = "flow_matching" if Path(policy_checkpoint).is_file() else "diffusion"
        if action_mode not in {"noise", "residual"}:
            raise ValueError(f"Unsupported DSRL action mode: {action_mode}")
        if action_mode == "residual" and policy_type != "flow_matching":
            raise ValueError("Residual DSRL currently requires a Flow Matching policy.")
        self.policy_type = policy_type
        self.action_mode = action_mode
        if policy_type == "flow_matching":
            self.adapter = FlowMatchingNoiseAdapter.from_pretrained(
                policy_checkpoint,
                device=device,
                num_inference_steps=flow_num_inference_steps,
            )
        else:
            self.adapter = DiffusionNoiseAdapter.from_pretrained(policy_checkpoint, device=device)
        self.noise_magnitude = float(noise_magnitude)
        self.chunk_discount = float(chunk_discount)
        self.action_repeat = int(action_repeat)
        self.residual_position_scale_m = tuple(float(value) for value in residual_position_scale_m)
        self.residual_width_scale_m = float(residual_width_scale_m)
        self.curriculum_steps = int(curriculum_steps)
        self.curriculum_start_step = int(curriculum_start_step)
        self.curriculum_start_xy_m = tuple(float(value) for value in curriculum_start_xy_m)
        self.curriculum_end_xy_m = tuple(float(value) for value in curriculum_end_xy_m)
        self.curriculum_start_yaw_rad = float(curriculum_start_yaw_rad)
        self.curriculum_end_yaw_rad = float(curriculum_end_yaw_rad)
        self.flow_chunk_execute_steps = int(flow_chunk_execute_steps)
        self.flow_phase_horizon_steps = int(flow_phase_horizon_steps)
        self.flow_camera_warmup_steps = int(flow_camera_warmup_steps)
        self._outer_steps = self.curriculum_start_step
        self._curriculum_progress = 0.0
        self._flow_policy_steps = 0
        self._flow_needs_warmup = policy_type == "flow_matching"
        self._completed_episodes = 0
        self._successful_episodes = 0
        self._broken_episodes = 0
        if self.action_repeat < 1:
            raise ValueError(f"action_repeat must be at least 1, received {self.action_repeat}.")
        if len(self.residual_position_scale_m) != 3 or any(value < 0.0 for value in self.residual_position_scale_m):
            raise ValueError("residual_position_scale_m must contain three non-negative values.")
        if self.residual_width_scale_m < 0.0:
            raise ValueError("residual_width_scale_m must be non-negative.")
        if len(self.curriculum_start_xy_m) != 2 or len(self.curriculum_end_xy_m) != 2:
            raise ValueError("Curriculum XY ranges must contain two values.")
        if any(value < 0.0 for value in (*self.curriculum_start_xy_m, *self.curriculum_end_xy_m)):
            raise ValueError("Curriculum XY ranges must be non-negative.")
        if self.curriculum_steps < 0 or self.curriculum_start_step < 0:
            raise ValueError("Curriculum steps must be non-negative.")
        if policy_type == "flow_matching" and not 1 <= self.flow_chunk_execute_steps <= self.adapter.n_action_steps:
            raise ValueError("flow_chunk_execute_steps must be within the policy horizon.")
        if self.flow_phase_horizon_steps < 1:
            raise ValueError("flow_phase_horizon_steps must be at least 1.")
        if self.flow_camera_warmup_steps < 0:
            raise ValueError("flow_camera_warmup_steps must be non-negative.")
        if self.action_mode == "residual":
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(
                low=-self.noise_magnitude,
                high=self.noise_magnitude,
                shape=(self.adapter.noise_dim,),
                dtype=np.float32,
            )
        # SKRL's Isaac wrapper reads the unwrapped environment metadata.
        env.unwrapped.single_action_space = self.action_space
        self.single_action_space = self.action_space
        self._update_randomization_curriculum()

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def num_envs(self) -> int:
        return int(self.env.unwrapped.num_envs)

    def _update_randomization_curriculum(self) -> None:
        if self.curriculum_steps <= 0:
            return
        progress = min(max(self._outer_steps / float(self.curriculum_steps), 0.0), 1.0)
        self._curriculum_progress = progress
        xy_range = tuple(
            start + progress * (end - start)
            for start, end in zip(self.curriculum_start_xy_m, self.curriculum_end_xy_m, strict=True)
        )
        yaw_range = self.curriculum_start_yaw_rad + progress * (
            self.curriculum_end_yaw_rad - self.curriculum_start_yaw_rad
        )
        base_cfg = self.env.unwrapped.cfg
        base_cfg.randomize_labware_position = True
        base_cfg.labware_pos_randomization_xy = xy_range
        base_cfg.labware_yaw_randomization = yaw_range

    def reset(self, **kwargs):
        self._update_randomization_curriculum()
        self.adapter.reset()
        self._flow_policy_steps = 0
        observation, info = self.env.reset(**kwargs)
        if self.policy_type == "flow_matching":
            self._flow_needs_warmup = True
            self._ensure_flow_observation_ready()
            observation = self.env.unwrapped._get_observations()
        return observation, info

    def _warmup_flow_cameras(self) -> None:
        base = self.env.unwrapped
        if self.flow_camera_warmup_steps <= 0:
            return
        hold_action = base.get_cafe_observation()["robot0_pos"].clone()
        for _ in range(self.flow_camera_warmup_steps):
            base._pre_physics_step(hold_action)
            base._apply_action()
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(dt=base.physics_dt)
            base.sim.render()
        base.step_count.zero_()
        base.has_touched.zero_()

    def _camera_numpy(self, camera) -> np.ndarray:
        rgb = camera.data.output["rgb"][:, :, :, :3].permute(0, 3, 1, 2).float()
        rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        return rgb[0].permute(1, 2, 0).clamp(0, 255).byte().detach().cpu().numpy()

    def _raw_flow_observation(self) -> dict[str, Any]:
        base = self.env.unwrapped
        cafe = base.get_cafe_observation()
        wrist_image = self._camera_numpy(base.wrist_camera)
        third_image = self._camera_numpy(base.third_person_camera)
        phase = min(
            self._flow_policy_steps / float(self.flow_phase_horizon_steps),
            1.0,
        )
        return {
            "robot0_pos": cafe["robot0_pos"][0].detach().cpu().numpy().astype(np.float32),
            "robot0_image": wrist_image,
            "robot0_image_third": third_image,
            "phase": phase,
        }

    def _ensure_flow_observation_ready(self) -> None:
        if self._flow_needs_warmup:
            self._warmup_flow_cameras()
            self._flow_needs_warmup = False
        if not self.adapter.is_ready:
            self.adapter.update(self._raw_flow_observation())

    def _camera_tensor(self, camera) -> torch.Tensor:
        rgb = camera.data.output["rgb"][:, :, :, :3].permute(0, 3, 1, 2).float()
        rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        return (rgb / 255.0).clamp(0.0, 1.0)

    def _raw_policy_observation(self) -> dict[str, Any]:
        base = self.env.unwrapped
        cafe = base.get_cafe_observation()
        return {
            "observation.state": cafe["robot0_pos"],
            "observation.images.rgb": self._camera_tensor(base.wrist_camera),
            "observation.images.rgb_third": self._camera_tensor(base.third_person_camera),
        }

    @staticmethod
    def _log_flag(log: dict[str, Any], key: str, legacy_key: str) -> bool:
        value = log.get(key, log.get(legacy_key, 0.0))
        if isinstance(value, torch.Tensor):
            value = value.reshape(-1)[0].item()
        return bool(value)

    def _add_episode_metrics(self, info: dict[str, Any], episode_done: bool) -> dict[str, Any]:
        info = dict(info)
        log = dict(info.get("log", {}))
        success = self._log_flag(log, "LabPick/success_terminal_step", "LabPick/success_rate")
        broken = self._log_flag(log, "LabPick/broken_terminal_step", "LabPick/broken_rate")
        if episode_done:
            self._completed_episodes += 1
            self._successful_episodes += int(success)
            self._broken_episodes += int(broken)
        denominator = max(self._completed_episodes, 1)
        log["LabPick/episode_success_rate"] = self._successful_episodes / denominator
        log["LabPick/episode_broken_rate"] = self._broken_episodes / denominator
        log["LabPick/completed_episodes"] = float(self._completed_episodes)
        log["LabPick/dsrl_action_mode_residual"] = float(self.action_mode == "residual")
        log["LabPick/randomization_curriculum_progress"] = self._curriculum_progress
        log["LabPick/randomization_xy_x_m"] = float(self.env.unwrapped.cfg.labware_pos_randomization_xy[0])
        log["LabPick/randomization_xy_y_m"] = float(self.env.unwrapped.cfg.labware_pos_randomization_xy[1])
        log["LabPick/randomization_yaw_rad"] = float(self.env.unwrapped.cfg.labware_yaw_randomization)
        info["log"] = log
        return info

    @torch.inference_mode()
    def _next_physical_action(self, flat_noise: torch.Tensor | None) -> torch.Tensor:
        if self.adapter.preprocessor is None:
            raise RuntimeError("The Diffusion checkpoint is missing its preprocessor.")
        batch = self.adapter.preprocessor(self._raw_policy_observation())
        noise = self.adapter.reshape_noise(flat_noise) if flat_noise is not None else None
        action = self.adapter.policy.select_action(batch, noise=noise)
        if self.adapter.postprocessor is not None:
            action = self.adapter.postprocessor(action)
        return action.to(self.device)

    def _step_decoded_chunk(self, flat_noise: torch.Tensor | None):
        total_reward = None
        terminated = truncated = None
        observation = info = None
        executed_actions = 0
        executed_physics_steps = 0
        episode_done = False
        for action_step in range(self.adapter.policy.config.n_action_steps):
            physical_action = self._next_physical_action(flat_noise if action_step == 0 else None)
            executed_actions += 1
            for _ in range(self.action_repeat):
                observation, reward, terminated, truncated, info = self.env.step(physical_action)
                discounted_reward = (self.chunk_discount**action_step) * reward
                total_reward = discounted_reward if total_reward is None else total_reward + discounted_reward
                executed_physics_steps += 1
                episode_done = bool((terminated | truncated).any().item())
                if episode_done:
                    self.adapter.reset()
                    break
            if episode_done:
                break

        if observation is None or total_reward is None or terminated is None or truncated is None or info is None:
            raise RuntimeError("DSRL wrapper executed no physical actions.")
        info = self._add_episode_metrics(info, episode_done)
        info["dsrl/action_steps_executed"] = executed_actions
        info["dsrl/action_repeat"] = self.action_repeat
        info["dsrl/physics_steps_executed"] = executed_physics_steps
        return observation, total_reward, terminated, truncated, info

    def _apply_flow_residual(self, action_chunk: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if action_chunk.ndim != 2 or action_chunk.shape[-1] < 10:
            raise ValueError(f"Expected a [horizon, 10+] Flow Matching action chunk, got {tuple(action_chunk.shape)}.")
        residual = residual.reshape(-1)
        if residual.numel() != 4:
            raise ValueError(f"Expected a 4-D residual action, got {residual.numel()} values.")
        adjusted = action_chunk.clone()
        position_scale = torch.as_tensor(
            self.residual_position_scale_m,
            dtype=adjusted.dtype,
            device=adjusted.device,
        )
        adjusted[:, :3] += residual[:3].to(adjusted) * position_scale
        closing = adjusted[:, 9] < 0.038
        width_delta = residual[3].to(adjusted) * self.residual_width_scale_m
        adjusted[closing, 9] = (adjusted[closing, 9] + width_delta).clamp(0.006, 0.04)
        return adjusted

    def _step_flow_chunk(self, flat_action: torch.Tensor | None, *, apply_residual: bool = True):
        self._ensure_flow_observation_ready()
        if self.action_mode == "residual":
            action_chunk = self.adapter.decode(None)
            if apply_residual:
                if flat_action is None:
                    raise ValueError("Residual action mode requires a SAC residual action.")
                action_chunk = self._apply_flow_residual(action_chunk, flat_action)
        else:
            action_chunk = self.adapter.decode(flat_action)
        total_reward = None
        terminated = truncated = None
        observation = info = None
        executed_actions = 0
        executed_physics_steps = 0
        episode_done = False
        for action_step in range(self.flow_chunk_execute_steps):
            physical_action = (
                action_chunk[action_step]
                .view(1, -1)
                .to(device=self.device, dtype=torch.float32)
            )
            executed_actions += 1
            for _ in range(self.action_repeat):
                observation, reward, terminated, truncated, info = self.env.step(physical_action)
                discounted_reward = (self.chunk_discount**action_step) * reward
                total_reward = (
                    discounted_reward
                    if total_reward is None
                    else total_reward + discounted_reward
                )
                executed_physics_steps += 1
                episode_done = bool((terminated | truncated).any().item())
                if episode_done:
                    break
            self._flow_policy_steps += 1
            if episode_done:
                self.adapter.reset()
                self._flow_policy_steps = 0
                self._flow_needs_warmup = True
                break
            self.adapter.update(self._raw_flow_observation())

        if observation is None or total_reward is None or terminated is None or truncated is None or info is None:
            raise RuntimeError("DSRL wrapper executed no physical actions.")
        info = self._add_episode_metrics(info, episode_done)
        info["dsrl/action_steps_executed"] = executed_actions
        info["dsrl/action_repeat"] = self.action_repeat
        info["dsrl/physics_steps_executed"] = executed_physics_steps
        info["dsrl/policy_type"] = self.policy_type
        return observation, total_reward, terminated, truncated, info

    def step(self, flat_action):
        if not isinstance(flat_action, torch.Tensor):
            flat_action = torch.as_tensor(flat_action, dtype=torch.float32, device=self.device)
        if flat_action.ndim == 1:
            flat_action = flat_action.unsqueeze(0)
        limit = 1.0 if self.action_mode == "residual" else self.noise_magnitude
        flat_action = flat_action.clamp(-limit, limit)
        if self.policy_type == "flow_matching":
            result = self._step_flow_chunk(flat_action)
        else:
            result = self._step_decoded_chunk(flat_action)
        self._outer_steps += 1
        self._update_randomization_curriculum()
        return result

    def step_bc(self):
        """Execute one frozen-BC action chunk using the policy native Gaussian noise."""

        if self.policy_type == "flow_matching":
            return self._step_flow_chunk(None, apply_residual=False)
        return self._step_decoded_chunk(None)
