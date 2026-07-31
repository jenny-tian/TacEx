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

    One outer environment step corresponds to one SAC noise action and up to
    ``n_action_steps`` physical simulation steps decoded by a frozen Diffusion
    Policy. The first integration is intentionally single-environment because
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
        self.policy_type = policy_type
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
        self.flow_chunk_execute_steps = int(flow_chunk_execute_steps)
        self.flow_phase_horizon_steps = int(flow_phase_horizon_steps)
        self.flow_camera_warmup_steps = int(flow_camera_warmup_steps)
        self._flow_policy_steps = 0
        self._flow_needs_warmup = policy_type == "flow_matching"
        self._completed_episodes = 0
        self._successful_episodes = 0
        self._broken_episodes = 0
        if self.action_repeat < 1:
            raise ValueError(f"action_repeat must be at least 1, received {self.action_repeat}.")
        if policy_type == "flow_matching" and not 1 <= self.flow_chunk_execute_steps <= self.adapter.n_action_steps:
            raise ValueError("flow_chunk_execute_steps must be within the policy horizon.")
        if self.flow_phase_horizon_steps < 1:
            raise ValueError("flow_phase_horizon_steps must be at least 1.")
        if self.flow_camera_warmup_steps < 0:
            raise ValueError("flow_camera_warmup_steps must be non-negative.")
        self.action_space = gym.spaces.Box(
            low=-self.noise_magnitude,
            high=self.noise_magnitude,
            shape=(self.adapter.noise_dim,),
            dtype=np.float32,
        )
        # SKRL's Isaac wrapper reads the unwrapped environment metadata.
        env.unwrapped.single_action_space = self.action_space
        self.single_action_space = self.action_space

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def num_envs(self) -> int:
        return int(self.env.unwrapped.num_envs)

    def reset(self, **kwargs):
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

    def _step_flow_chunk(self, flat_noise: torch.Tensor | None):
        self._ensure_flow_observation_ready()
        action_chunk = self.adapter.decode(flat_noise)
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

    def step(self, flat_noise):
        if not isinstance(flat_noise, torch.Tensor):
            flat_noise = torch.as_tensor(flat_noise, dtype=torch.float32, device=self.device)
        if flat_noise.ndim == 1:
            flat_noise = flat_noise.unsqueeze(0)
        flat_noise = flat_noise.clamp(-self.noise_magnitude, self.noise_magnitude)
        if self.policy_type == "flow_matching":
            return self._step_flow_chunk(flat_noise)
        return self._step_decoded_chunk(flat_noise)

    def step_bc(self):
        """Execute one frozen-BC action chunk using the policy native Gaussian noise."""

        if self.policy_type == "flow_matching":
            return self._step_flow_chunk(None)
        return self._step_decoded_chunk(None)
