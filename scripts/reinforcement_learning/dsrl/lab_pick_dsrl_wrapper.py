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
    try:
        from diffusion_noise_adapter import DiffusionNoiseAdapter
    except ImportError:
        DiffusionNoiseAdapter = None

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
        chunk_discount: float = 1.0,
        action_repeat: int = 2,
        policy_type: str = "auto",
        residual_mode: str = "latent",
        physical_residual_segments: int = 4,
        flow_num_inference_steps: int = 20,
        flow_chunk_execute_steps: int = 16,
        flow_phase_horizon_steps: int = 383,
        flow_camera_warmup_steps: int = 8,
        gate_enabled: bool = False,
        gate_temperature: float = 0.5,
        gate_penalty: float = 5.0,
        gate_min: float = 0.02,
        gate_max: float = 0.2,
        base_noise_seed: int = 42,
        visual_pose_probe: str | None = None,
    ) -> None:
        super().__init__(env)
        if int(env.unwrapped.num_envs) != 1:
            raise ValueError("LabPick DSRL currently requires --num_envs 1.")
        if policy_type not in {"auto", "diffusion", "flow_matching"}:
            raise ValueError(f"Unsupported DSRL policy type: {policy_type}")
        if residual_mode not in {"latent", "physical"}:
            raise ValueError(f"Unsupported DSRL residual mode: {residual_mode}")
        if residual_mode == "physical" and policy_type not in {"auto", "flow_matching"}:
            raise ValueError("Physical residual mode currently requires a Flow Matching BC policy.")
        if policy_type == "auto":
            policy_type = "flow_matching" if Path(policy_checkpoint).is_file() else "diffusion"
        self.policy_type = policy_type
        self.residual_mode = residual_mode
        self.physical_residual_segments = int(physical_residual_segments)
        if policy_type == "flow_matching":
            self.adapter = FlowMatchingNoiseAdapter.from_pretrained(
                policy_checkpoint,
                device=device,
                num_inference_steps=flow_num_inference_steps,
                visual_pose_probe=visual_pose_probe,
            )
        else:
            if DiffusionNoiseAdapter is None:
                raise ImportError("Diffusion DSRL requires the optional LeRobot/draccus dependencies.")
            self.adapter = DiffusionNoiseAdapter.from_pretrained(policy_checkpoint, device=device)
        self.noise_magnitude = float(noise_magnitude)
        self.chunk_discount = float(chunk_discount)
        self.action_repeat = int(action_repeat)
        self.flow_chunk_execute_steps = int(flow_chunk_execute_steps)
        self.flow_phase_horizon_steps = int(flow_phase_horizon_steps)
        self.flow_camera_warmup_steps = int(flow_camera_warmup_steps)
        self.gate_enabled = bool(gate_enabled)
        self.gate_temperature = float(gate_temperature)
        self.gate_penalty = float(gate_penalty)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.noise_dim = int(self.adapter.noise_dim)
        self.observation_dim = (
            int(self.adapter.observation_dim)
            if policy_type == "flow_matching"
            else int(env.unwrapped.single_observation_space.shape[-1])
        )
        # Physical residual mode exposes the current frozen-BC action in the
        # SAC observation. This makes the residual problem Markov even though
        # the BC decoder is stochastic in latent mode.
        self.base_action_dim = (
            10 * self.physical_residual_segments if residual_mode == "physical" else 0
        )
        # The frozen BC already contains image-supervised XY and rotation
        # heads. Expose their predictions to SAC without feeding simulator
        # object state or changing the BC checkpoint.
        self.visual_pose_probe_enabled = bool(visual_pose_probe)
        self.visual_pose_dim = (10 if self.visual_pose_probe_enabled else 8) if residual_mode == "physical" else 0
        self.time_observation_dim = 1 if residual_mode == "physical" else 0
        self.observation_dim += self.base_action_dim + self.visual_pose_dim + self.time_observation_dim
        self.base_noise_seed = int(base_noise_seed)
        self._physical_base_noise: torch.Tensor | None = None
        self._physical_noise_generator: torch.Generator | None = None
        if residual_mode == "physical":
            self._physical_noise_generator = torch.Generator(device=self.device)
            self._physical_noise_generator.manual_seed(self.base_noise_seed)
            self._resample_physical_base_noise()
        self._flow_policy_steps = 0
        self._flow_needs_warmup = policy_type == "flow_matching"
        self._completed_episodes = 0
        self._successful_episodes = 0
        self._broken_episodes = 0
        self._timed_out_episodes = 0
        self._other_failed_episodes = 0
        if self.action_repeat < 1:
            raise ValueError(f"action_repeat must be at least 1, received {self.action_repeat}.")
        if self.physical_residual_segments < 1:
            raise ValueError("physical_residual_segments must be at least 1.")
        if self.physical_residual_segments > self.flow_chunk_execute_steps:
            raise ValueError("physical_residual_segments cannot exceed flow_chunk_execute_steps.")
        if policy_type == "flow_matching" and not 1 <= self.flow_chunk_execute_steps <= self.adapter.n_action_steps:
            raise ValueError("flow_chunk_execute_steps must be within the policy horizon.")
        if self.flow_phase_horizon_steps < 1:
            raise ValueError("flow_phase_horizon_steps must be at least 1.")
        if self.flow_camera_warmup_steps < 0:
            raise ValueError("flow_camera_warmup_steps must be non-negative.")
        if self.gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive.")
        if self.gate_penalty < 0.0:
            raise ValueError("gate_penalty must be non-negative.")
        if not 0.0 <= self.gate_min < self.gate_max <= 1.0:
            raise ValueError("gate_min and gate_max must satisfy 0 <= min < max <= 1.")
        residual_dim = (
            10 * self.physical_residual_segments if residual_mode == "physical" else self.noise_dim
        )
        action_dim = residual_dim + int(self.gate_enabled)
        action_bound = 1.0 if residual_mode == "physical" else self.noise_magnitude
        self.action_space = gym.spaces.Box(
            low=-action_bound,
            high=action_bound,
            shape=(action_dim,),
            dtype=np.float32,
        )
        # SKRL's Isaac wrapper reads the unwrapped environment metadata.
        env.unwrapped.single_action_space = self.action_space
        self.single_action_space = self.action_space
        policy_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict({"policy": policy_observation_space})
        env.unwrapped.single_observation_space = self.observation_space
        self.single_observation_space = self.observation_space

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
            if self.residual_mode == "physical":
                self._resample_physical_base_noise()
            self._flow_needs_warmup = True
            self._ensure_flow_observation_ready()
            observation = {"policy": self._encoded_flow_observation()}
        return observation, info

    @torch.inference_mode()
    def _resample_physical_base_noise(self) -> None:
        """Draw native BC noise like the standalone BC evaluator.

        A fixed noise tensor across all episodes can make the residual wrapper
        appear much worse than BC if that one realization is unfavorable. The
        generator remains seeded for reproducibility, but each episode receives
        a fresh native-distribution sample that remains fixed within the episode.
        """
        if self._physical_noise_generator is None:
            return
        self._physical_base_noise = torch.randn(
            (1, self.noise_dim),
            device=self.device,
            generator=self._physical_noise_generator,
        )

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

    def _encoded_flow_observation(self) -> torch.Tensor:
        self._ensure_flow_observation_ready()
        encoded = self.adapter.encode_observation().to(device=self.device, dtype=torch.float32)
        if self.residual_mode != "physical":
            return encoded
        base_action = self._decode_base_action()[: self.flow_chunk_execute_steps]
        if self.visual_pose_dim:
            visual_pose = (
                self.adapter.visual_pose_probe_estimate()
                if self.visual_pose_probe_enabled
                else self.adapter.visual_pose_estimate()
            )
            if self.visual_pose_probe_enabled:
                # Do not expose an untrusted pose at the same scale as a
                # confident estimate. The confidence itself remains in the
                # observation as the last probe dimension.
                visual_pose = visual_pose.clone()
                visual_pose[..., :9] = visual_pose[..., :9] * visual_pose[..., 9:10]
        else:
            visual_pose = None
        # Position and width are normalized to roughly unit scale; rot6d is
        # already dimensionless. The SAC residual policy can therefore see
        # the action it is correcting without privileged object state.
        segment_ids = self._segment_ids(base_action.shape[0], base_action.device)
        base_summary = torch.stack(
            [base_action[segment_ids == index].mean(dim=0) for index in range(self.physical_residual_segments)]
        )
        base_summary[:, :3] = base_summary[:, :3] / 0.5
        base_summary[:, 9] = base_summary[:, 9] / 0.04
        pieces = [encoded]
        if visual_pose is not None:
            pieces.append(visual_pose)
        pieces.append(base_summary.reshape(1, -1))
        base = self.env.unwrapped
        remaining_time = (
            1.0 - base.episode_length_buf.float() / float(base.max_episode_length)
        ).clamp(0.0, 1.0).unsqueeze(-1)
        pieces.append(remaining_time)
        return torch.cat(pieces, dim=-1)

    @torch.inference_mode()
    def _decode_base_action(self) -> torch.Tensor:
        if self.policy_type != "flow_matching":
            raise RuntimeError("Physical residual mode requires Flow Matching decoding.")
        if self._physical_base_noise is None:
            raise RuntimeError("Physical BC base noise has not been initialized.")
        return self.adapter.decode(self._physical_base_noise)

    @staticmethod
    def _normalize_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
        matrix = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
        first = F.normalize(matrix[..., :, 0], dim=-1, eps=1.0e-6)
        second_raw = matrix[..., :, 1]
        second = F.normalize(
            second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first,
            dim=-1,
            eps=1.0e-6,
        )
        return torch.stack((first, second), dim=-1).reshape(*rot6d.shape[:-1], 6)

    def _apply_physical_residual(
        self, action_chunk: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        if action_chunk.ndim != 2 or action_chunk.shape[-1] != 10:
            raise ValueError(f"Expected a (horizon, 10) BC action chunk, got {tuple(action_chunk.shape)}")
        expected_shape = (self.physical_residual_segments, 10)
        if residual.numel() != self.physical_residual_segments * 10:
            raise ValueError(f"Expected a {expected_shape} physical residual, got {tuple(residual.shape)}")
        residual = residual.reshape(expected_shape)
        scales = torch.tensor(
            [0.015, 0.015, 0.015, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.004],
            device=action_chunk.device,
            dtype=action_chunk.dtype,
        )
        segment_ids = self._segment_ids(action_chunk.shape[0], action_chunk.device)
        delta = residual.to(action_chunk)[segment_ids] * scales
        result = action_chunk.clone()
        result[:, :3] += delta[:, :3]
        result[:, 3:9] = self._normalize_rot6d(result[:, 3:9] + delta[:, 3:9])
        result[:, 9] = (result[:, 9] + delta[:, 9]).clamp(0.0, 0.04)
        return result

    def _segment_ids(self, chunk_length: int, device: torch.device) -> torch.Tensor:
        if chunk_length < self.physical_residual_segments:
            raise ValueError("Action chunk is shorter than the configured physical residual segments.")
        indices = torch.arange(chunk_length, device=device)
        return torch.div(
            indices * self.physical_residual_segments,
            chunk_length,
            rounding_mode="floor",
        ).clamp_max(self.physical_residual_segments - 1)

    def _gate_from_action(self, raw_gate: torch.Tensor) -> torch.Tensor:
        # Recover an unbounded gate logit from the actor's bounded [-1, 1]
        # output so values close to zero remain representable.
        bounded_gate = raw_gate.clamp(-0.999999, 0.999999)
        gate_logit = torch.atanh(bounded_gate) / self.gate_temperature
        return self.gate_min + (self.gate_max - self.gate_min) * torch.sigmoid(gate_logit)

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
        timed_out = self._log_flag(log, "LabPick/timeout_terminal_step", "LabPick/timeout_rate")
        if not episode_done:
            info["log"] = log
            return info

        self._completed_episodes += 1
        # Outcomes are exclusive: breakage takes precedence over a lift
        # detected during the same terminal physics step.
        outcome = "other_failure"
        if broken:
            self._broken_episodes += 1
            outcome = "broken"
        elif success:
            self._successful_episodes += 1
            outcome = "success"
        elif timed_out:
            self._timed_out_episodes += 1
            outcome = "timeout"
        else:
            self._other_failed_episodes += 1

        # Each rate receives exactly one binary sample per completed episode.
        # SKRL averages these samples over its TensorBoard write interval, so
        # this is an episode success/failure rate rather than a step frequency.
        metrics = {
            "LabPick/episode_success_rate": float(outcome == "success"),
            "LabPick/episode_failure_rate": float(outcome != "success"),
            "LabPick/episode_broken_rate": float(outcome == "broken"),
            "LabPick/episode_timeout_rate": float(outcome == "timeout"),
            "LabPick/episode_other_failure_rate": float(outcome == "other_failure"),
            # Counts are monotonic. The ``(max)`` suffix asks SKRL to retain
            # the last/highest count in each write interval instead of its mean.
            "LabPick/completed_episodes (max)": float(self._completed_episodes),
            "LabPick/successful_episodes (max)": float(self._successful_episodes),
            "LabPick/broken_episodes (max)": float(self._broken_episodes),
            "LabPick/timed_out_episodes (max)": float(self._timed_out_episodes),
            "LabPick/other_failed_episodes (max)": float(self._other_failed_episodes),
        }
        # SKRL only forwards scalar tensors from infos["log"] to TensorBoard.
        log.update({key: torch.tensor(value, device=self.device) for key, value in metrics.items()})
        info["log"] = log
        return info

    def _blend_gated_noise(
        self,
        flat_action: torch.Tensor,
        native_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not self.gate_enabled:
            return flat_action, None, None
        if flat_action.shape[-1] != self.noise_dim + 1:
            raise ValueError(
                f"Expected {self.noise_dim + 1} gated DSRL actions, received {flat_action.shape[-1]}."
            )
        rl_noise = flat_action[..., : self.noise_dim]
        raw_gate = flat_action[..., self.noise_dim :]
        gate = self._gate_from_action(raw_gate)
        if native_noise is None:
            native_noise = torch.randn_like(rl_noise)
        elif native_noise.shape != rl_noise.shape:
            raise ValueError(
                f"Native BC noise must have shape {tuple(rl_noise.shape)}, received {tuple(native_noise.shape)}."
            )
        blended_noise = torch.lerp(native_noise, rl_noise, gate)
        return blended_noise, gate, native_noise

    def _apply_gate_reward_and_metrics(
        self,
        total_reward: torch.Tensor,
        info: dict[str, Any],
        gate: torch.Tensor | None,
        correction: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if gate is None:
            return total_reward, info
        correction = gate if correction is None else correction
        correction_per_env = correction.square().mean(dim=-1)
        penalty = self.gate_penalty * correction_per_env
        total_reward = total_reward - penalty.to(total_reward)
        info = dict(info)
        log = dict(info.get("log", {}))
        log["DSRL/gate_mean"] = gate.mean().detach()
        log["DSRL/gate_penalty"] = penalty.mean().detach()
        log["DSRL/correction_rms"] = correction_per_env.sqrt().mean().detach()
        info["log"] = log
        info["dsrl/gate"] = float(gate.mean().detach().cpu())
        return total_reward, info

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

    def _step_decoded_chunk(self, flat_noise: torch.Tensor | None, gate: torch.Tensor | None = None):
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
        total_reward, info = self._apply_gate_reward_and_metrics(total_reward, info, gate)
        info = self._add_episode_metrics(info, episode_done)
        info["dsrl/action_steps_executed"] = executed_actions
        info["dsrl/action_repeat"] = self.action_repeat
        info["dsrl/physics_steps_executed"] = executed_physics_steps
        return observation, total_reward, terminated, truncated, info

    def _step_flow_chunk(self, flat_noise: torch.Tensor | None, gate: torch.Tensor | None = None):
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
        # Intermediate actions are open-loop from the decoded frozen-BC chunk,
        # so defer the costly vision encoder update until the complete chunk
        # has executed.
        if not episode_done:
            self.adapter.update(self._raw_flow_observation())

        if observation is None or total_reward is None or terminated is None or truncated is None or info is None:
            raise RuntimeError("DSRL wrapper executed no physical actions.")
        total_reward, info = self._apply_gate_reward_and_metrics(total_reward, info, gate)
        info = self._add_episode_metrics(info, episode_done)
        info["dsrl/action_steps_executed"] = executed_actions
        info["dsrl/action_repeat"] = self.action_repeat
        info["dsrl/physics_steps_executed"] = executed_physics_steps
        info["dsrl/policy_type"] = self.policy_type
        observation = {"policy": self._encoded_flow_observation()}
        return observation, total_reward, terminated, truncated, info

    def _step_flow_physical_chunk(
        self, flat_residual: torch.Tensor, gate: torch.Tensor | None = None
    ):
        """Execute a low-dimensional residual around a deterministic BC chunk."""
        self._ensure_flow_observation_ready()
        base_chunk = self._decode_base_action()
        residual_dim = 10 * self.physical_residual_segments
        residual = flat_residual[..., :residual_dim]
        if gate is not None:
            residual = residual * gate
        action_chunk = self._apply_physical_residual(base_chunk, residual[0])

        total_reward = None
        terminated = truncated = None
        observation = info = None
        executed_actions = 0
        executed_physics_steps = 0
        episode_done = False
        for action_step in range(self.flow_chunk_execute_steps):
            physical_action = action_chunk[action_step].view(1, -1).to(self.device, dtype=torch.float32)
            executed_actions += 1
            for _ in range(self.action_repeat):
                observation, reward, terminated, truncated, info = self.env.step(physical_action)
                discounted_reward = (self.chunk_discount**action_step) * reward
                total_reward = discounted_reward if total_reward is None else total_reward + discounted_reward
                executed_physics_steps += 1
                episode_done = bool((terminated | truncated).any().item())
                if episode_done:
                    break
            self._flow_policy_steps += 1
            if episode_done:
                self.adapter.reset()
                self._resample_physical_base_noise()
                self._flow_policy_steps = 0
                self._flow_needs_warmup = True
                break
            # The residual is applied to an already decoded frozen-BC chunk;
            # no new visual encoding is needed between its actions.

        if not episode_done:
            self.adapter.update(self._raw_flow_observation())
            # Keep one native Flow Matching noise template for the whole
            # episode. Re-sampling here changes the frozen BC trajectory at
            # every replan and makes DSRL incomparable with BC evaluation.

        if observation is None or total_reward is None or terminated is None or truncated is None or info is None:
            raise RuntimeError("DSRL wrapper executed no physical residual actions.")
        total_reward, info = self._apply_gate_reward_and_metrics(
            total_reward, info, gate, correction=residual
        )
        info = self._add_episode_metrics(info, episode_done)
        info["dsrl/action_steps_executed"] = executed_actions
        info["dsrl/action_repeat"] = self.action_repeat
        info["dsrl/physics_steps_executed"] = executed_physics_steps
        info["dsrl/policy_type"] = self.policy_type
        info["dsrl/residual_mode"] = self.residual_mode
        observation = {"policy": self._encoded_flow_observation()}
        return observation, total_reward, terminated, truncated, info

    def step(self, flat_noise):
        if not isinstance(flat_noise, torch.Tensor):
            flat_noise = torch.as_tensor(flat_noise, dtype=torch.float32, device=self.device)
        if flat_noise.ndim == 1:
            flat_noise = flat_noise.unsqueeze(0)
        if self.residual_mode == "physical":
            flat_noise = flat_noise.clamp(-1.0, 1.0)
            gate = None
            residual_dim = 10 * self.physical_residual_segments
            if self.gate_enabled:
                if flat_noise.shape[-1] != residual_dim + 1:
                    raise ValueError(
                        f"Expected {residual_dim + 1} physical residual actions, got {flat_noise.shape[-1]}"
                    )
                raw_gate = flat_noise[..., residual_dim:]
                gate = self._gate_from_action(raw_gate)
            elif flat_noise.shape[-1] != residual_dim:
                raise ValueError(f"Expected {residual_dim} physical residual actions, got {flat_noise.shape[-1]}")
            return self._step_flow_physical_chunk(flat_noise, gate=gate)
        flat_noise = flat_noise.clamp(-self.noise_magnitude, self.noise_magnitude)
        flat_noise, gate, _ = self._blend_gated_noise(flat_noise)
        if self.policy_type == "flow_matching":
            return self._step_flow_chunk(flat_noise, gate=gate)
        return self._step_decoded_chunk(flat_noise, gate=gate)

    def step_bc(self):
        """Execute one frozen-BC action chunk using the policy native Gaussian noise."""

        if self.residual_mode == "physical":
            residual_dim = 10 * self.physical_residual_segments
            zero = torch.zeros((1, residual_dim + int(self.gate_enabled)), device=self.device)
            return self._step_flow_physical_chunk(zero, gate=None)
        if self.policy_type == "flow_matching":
            return self._step_flow_chunk(None)
        return self._step_decoded_chunk(None)
