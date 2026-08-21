"""Minimal LabPick environment contract for clean residual SAC.

The frozen Flow Matching policy predicts a 32-action CAFE chunk.  SAC sees and
corrects exactly one action at a time, and the BC is replanned after the first
10 actions.  Its 4-D post-tanh action changes only normalized CAFE
``(x, y, z, width)``; the six BC Rot6D coordinates are preserved verbatim.

This wrapper deliberately contains no reward shaping, residual penalty,
warm-up noise, trust region, potential, or n-step-return logic.  Those belong
to neither the environment nor the clean one-step SAC contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .clean_residual_sac import CleanResidualLayout
    from .flow_matching_noise_adapter import FlowMatchingNoiseAdapter
except ImportError:
    from clean_residual_sac import CleanResidualLayout
    from flow_matching_noise_adapter import FlowMatchingNoiseAdapter


FLOW_HORIZON = 32
REPLAN_STEPS = 10
ACTION_REPEAT = 2
MAX_OBSERVATION_UPDATES = 32


def _validate_matrix(tensor: torch.Tensor, *, name: str, width: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, received {type(tensor).__name__}.")
    if tensor.ndim != 2 or tensor.shape[-1] != width:
        raise ValueError(f"{name} must have shape [B, {width}], got {tuple(tensor.shape)}.")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype, received {tensor.dtype}.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values.")


def compose_normalized_action(
    full_bc_action: torch.Tensor,
    residual: torch.Tensor,
    *,
    layout: CleanResidualLayout | None = None,
) -> torch.Tensor:
    """Return a complete normalized CAFE action with an xyz/width correction.

    The input is never modified.  In particular, columns ``3:9`` in the return
    value are bit-for-bit copies of the frozen BC Rot6D columns.
    """

    resolved_layout = CleanResidualLayout() if layout is None else layout
    if not isinstance(resolved_layout, CleanResidualLayout):
        raise TypeError(
            "layout must be a CleanResidualLayout, received "
            f"{type(resolved_layout).__name__}."
        )
    _validate_matrix(
        full_bc_action,
        name="full_bc_action",
        width=resolved_layout.full_bc_context,
    )
    resolved_layout.validate_residuals(residual)
    if full_bc_action.shape[0] != residual.shape[0]:
        raise ValueError(
            "full_bc_action and residual must have the same batch size, received "
            f"{full_bc_action.shape[0]} and {residual.shape[0]}."
        )

    composed = full_bc_action.clone()
    index = torch.as_tensor(
        resolved_layout.indices,
        dtype=torch.long,
        device=composed.device,
    )
    controlled = resolved_layout.compose_controlled_action(
        full_bc_action,
        residual.to(full_bc_action),
    )
    composed.index_copy_(-1, index, controlled)
    return composed


def pack_clean_residual_observation(
    normalized_proprioception: torch.Tensor,
    relative_object_position: torch.Tensor,
    object_rot6d: torch.Tensor,
    current_bc_action: torch.Tensor,
    *,
    layout: CleanResidualLayout | None = None,
) -> dict[str, torch.Tensor]:
    """Pack the public 29-D policy and 19-D asymmetric-Critic observations."""

    resolved_layout = CleanResidualLayout() if layout is None else layout
    if not isinstance(resolved_layout, CleanResidualLayout):
        raise TypeError(
            "layout must be a CleanResidualLayout, received "
            f"{type(resolved_layout).__name__}."
        )
    fields = (
        (normalized_proprioception, "normalized_proprioception", 10),
        (relative_object_position, "relative_object_position", 3),
        (object_rot6d, "object_rot6d", 6),
        (current_bc_action, "current_bc_action", 10),
    )
    for tensor, name, width in fields:
        _validate_matrix(tensor, name=name, width=width)
    batch_sizes = {tensor.shape[0] for tensor, _, _ in fields}
    if len(batch_sizes) != 1:
        raise ValueError(
            "All observation fields must have the same batch size, received "
            f"{sorted(batch_sizes)}."
        )

    relative_object_position = relative_object_position.to(normalized_proprioception)
    object_rot6d = object_rot6d.to(normalized_proprioception)
    current_bc_action = current_bc_action.to(normalized_proprioception)
    critic = torch.cat(
        (normalized_proprioception, relative_object_position, object_rot6d),
        dim=-1,
    )
    policy = torch.cat((critic, current_bc_action), dim=-1)
    resolved_layout.validate_states(critic)
    resolved_layout.split_policy_observation(policy)
    return {"policy": policy, "critic": critic}


class CleanResidualLabPickWrapper(gym.Wrapper):
    """Q1 Flow-BC residual environment for the minimal alpha-zero SAC path."""

    flow_horizon = FLOW_HORIZON
    replan_steps = REPLAN_STEPS
    action_repeat = ACTION_REPEAT

    def __init__(
        self,
        env: gym.Env,
        policy_checkpoint: str | Path,
        *,
        device: str = "cuda",
        residual_scale: float = 0.15,
        flow_num_inference_steps: int = 20,
        phase_horizon_steps: int = 383,
        camera_warmup_steps: int = 8,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        base = env.unwrapped
        if int(getattr(base, "num_envs", 0)) != 1:
            raise ValueError("Clean residual SAC requires exactly one environment.")
        cfg = getattr(base, "cfg", None)
        if cfg is None or not bool(getattr(cfg, "rl_align_cafe_action_yaw", False)):
            raise ValueError(
                "Clean residual SAC requires rl_align_cafe_action_yaw=True: "
                "the environment supplies oracle yaw while SAC leaves BC Rot6D unchanged."
            )
        if not callable(getattr(base, "get_cafe_observation", None)):
            raise TypeError("The wrapped environment must implement get_cafe_observation().")
        if not callable(getattr(base, "get_privileged_object_pose", None)):
            raise TypeError("The wrapped environment must implement get_privileged_object_pose().")
        if not isinstance(policy_checkpoint, (str, Path)) or not str(policy_checkpoint):
            raise TypeError("policy_checkpoint must be a non-empty path string.")
        integer_options = (
            (flow_num_inference_steps, "flow_num_inference_steps", 1),
            (phase_horizon_steps, "phase_horizon_steps", 1),
            (camera_warmup_steps, "camera_warmup_steps", 0),
        )
        for value, name, minimum in integer_options:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}.")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("seed must be an integer or None.")
            if seed < 0:
                raise ValueError("seed must be non-negative when provided.")

        self._layout = CleanResidualLayout(scale=residual_scale)
        self.phase_horizon_steps = int(phase_horizon_steps)
        self.camera_warmup_steps = int(camera_warmup_steps)
        self.adapter = FlowMatchingNoiseAdapter.from_pretrained(
            policy_checkpoint,
            device=device,
            num_inference_steps=int(flow_num_inference_steps),
            visual_xy_lock_phase=0.30,
            use_visual_xy_override=True,
            seed=seed,
        )
        adapter_horizon = int(getattr(self.adapter, "n_action_steps", -1))
        adapter_action_dim = int(getattr(self.adapter, "action_dim", -1))
        if adapter_horizon != self.flow_horizon:
            raise ValueError(
                f"Clean residual SAC requires a Flow BC horizon of {self.flow_horizon}, "
                f"received {adapter_horizon}."
            )
        if adapter_action_dim != self.layout.full_bc_context:
            raise ValueError(
                "Clean residual SAC requires 10-D CAFE actions, received "
                f"{adapter_action_dim}."
            )

        self._flow_policy_steps = 0
        self._flow_needs_warmup = True
        self._normalized_bc_chunk: torch.Tensor | None = None
        self._chunk_offset = 0

        single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.layout.action,),
            dtype=np.float32,
        )
        policy_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.layout.policy,),
            dtype=np.float32,
        )
        critic_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.layout.state,),
            dtype=np.float32,
        )
        self.action_space = single_action_space
        self.single_action_space = single_action_space
        base.single_action_space = single_action_space
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
    def layout(self) -> CleanResidualLayout:
        """Return the immutable tensor/action contract shared with the SAC models."""

        return self._layout

    @property
    def chunk_offset(self) -> int:
        """Index of the BC action exposed to the policy for the next step."""

        return self._chunk_offset

    @property
    def current_bc_action(self) -> torch.Tensor:
        """Return a defensive copy of the normalized 10-D BC action to execute next."""

        if self._normalized_bc_chunk is None:
            raise RuntimeError("No Flow BC chunk is prepared; call reset() first.")
        if not 0 <= self._chunk_offset < self.replan_steps:
            raise RuntimeError(f"Invalid BC chunk offset {self._chunk_offset}.")
        return self._normalized_bc_chunk[self._chunk_offset].reshape(1, -1).detach().clone()

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

    def _prepare_bc_chunk(self) -> None:
        self._ensure_flow_observation_ready()
        physical, normalized = self.adapter.decode_with_normalized()
        _validate_matrix(physical, name="physical_bc_chunk", width=10)
        _validate_matrix(normalized, name="normalized_bc_chunk", width=10)
        if physical.shape[0] != self.flow_horizon or normalized.shape[0] != self.flow_horizon:
            raise ValueError(
                f"Flow BC must decode [{self.flow_horizon}, 10] chunks, received "
                f"{tuple(physical.shape)} and {tuple(normalized.shape)}."
            )
        self._normalized_bc_chunk = normalized.to(
            device=self.device,
            dtype=torch.float32,
        ).detach().clone()
        self._chunk_offset = 0

    def _compose_normalized_action(
        self,
        full_bc_action: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Instance form of :func:`compose_normalized_action` for tests/runtime."""

        return compose_normalized_action(full_bc_action, residual, layout=self.layout)

    def _policy_observation(self) -> dict[str, torch.Tensor]:
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
        return pack_clean_residual_observation(
            normalized_proprioception,
            relative_position.to(device=self.device, dtype=torch.float32),
            object_rot6d.to(device=self.device, dtype=torch.float32),
            self.current_bc_action,
            layout=self.layout,
        )

    @staticmethod
    def _episode_done(terminated: Any, truncated: Any) -> bool:
        done = terminated | truncated
        if isinstance(done, torch.Tensor):
            return bool(done.any().item())
        return bool(np.asarray(done).any())

    @torch.inference_mode()
    def reset(self, **kwargs):
        self.adapter.reset()
        self._flow_policy_steps = 0
        self._flow_needs_warmup = True
        self._normalized_bc_chunk = None
        self._chunk_offset = 0
        _, info = self.env.reset(**kwargs)
        self._prepare_bc_chunk()
        return self._policy_observation(), info

    @torch.inference_mode()
    def step(self, residual):
        residual = torch.as_tensor(residual, dtype=torch.float32, device=self.device)
        if residual.ndim == 1:
            residual = residual.unsqueeze(0)
        self.layout.validate_residuals(residual, enforce_bounds=True)
        if residual.shape[0] != self.num_envs:
            raise ValueError(
                f"Residual batch must equal num_envs={self.num_envs}, "
                f"received {residual.shape[0]}."
            )

        action_index = self._chunk_offset
        normalized_action = self._compose_normalized_action(
            self.current_bc_action,
            residual,
        )
        physical_action = self.adapter.unnormalize_action(normalized_action).to(
            device=self.device,
            dtype=torch.float32,
        )
        _validate_matrix(physical_action, name="physical_action", width=10)
        if physical_action.shape[0] != self.num_envs:
            raise ValueError(
                f"Physical action batch must equal num_envs={self.num_envs}, "
                f"received {physical_action.shape[0]}."
            )

        total_reward = None
        terminated = truncated = info = None
        episode_done = False
        physics_steps = 0
        for _ in range(self.action_repeat):
            _, reward, terminated, truncated, info = self.env.step(physical_action)
            total_reward = reward if total_reward is None else total_reward + reward
            physics_steps += 1
            episode_done = self._episode_done(terminated, truncated)
            if episode_done:
                break
        if total_reward is None or terminated is None or truncated is None or info is None:
            raise RuntimeError("Clean residual wrapper executed no physical action.")

        self._flow_policy_steps += 1
        replanned = False
        if episode_done:
            # DirectRLEnv has already auto-reset.  Rebuild all BC state from the
            # reset observation, while the terminal mask below prevents SAC
            # from bootstrapping across the episode boundary.
            self.adapter.reset()
            self._flow_policy_steps = 0
            self._flow_needs_warmup = True
            self._normalized_bc_chunk = None
            self._chunk_offset = 0
            self._prepare_bc_chunk()
            replanned = True
        else:
            self.adapter.update(self._raw_flow_observation())
            self._chunk_offset += 1
            if self._chunk_offset == self.replan_steps:
                self._prepare_bc_chunk()
                replanned = True

        info = dict(info)
        residual_rms = residual.square().mean().sqrt().detach()
        info["clean_residual/action_index"] = action_index
        info["clean_residual/action_repeat"] = physics_steps
        info["clean_residual/replanned"] = replanned
        info["clean_residual/residual_rms"] = residual_rms
        info["clean_residual/effective_residual_rms"] = residual_rms * self.layout.scale
        log = dict(info.get("log", {}))
        log["CleanResidual/raw_residual_rms"] = residual_rms
        log["CleanResidual/effective_residual_rms"] = residual_rms * self.layout.scale
        info["log"] = log
        terminated_for_learning = terminated | truncated
        return (
            self._policy_observation(),
            total_reward,
            terminated_for_learning,
            truncated,
            info,
        )


__all__ = [
    "ACTION_REPEAT",
    "CleanResidualLabPickWrapper",
    "FLOW_HORIZON",
    "REPLAN_STEPS",
    "compose_normalized_action",
    "pack_clean_residual_observation",
]
