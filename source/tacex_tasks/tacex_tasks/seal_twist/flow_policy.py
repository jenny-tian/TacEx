"""6D-wrench-conditioned Flow Matching policy for SealTwist.

The policy follows the same conditional Flow Matching construction used by
the LabPick BC stack: an observation-conditioned vector field transports a
standard Gaussian action chunk at ``t=0`` to a normalized CAFE action chunk
at ``t=1``.  Unlike the earlier SealTwist MLP baseline, the observation
explicitly includes the full ``[Fx, Fy, Fz, Tx, Ty, Tz]`` task wrench.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


FLOW_CONTRACT_VERSION = 4
FLOW_OBSERVATION_DIM = 28
FLOW_ACTION_DIM = 10
WRENCH_OBSERVATION_SLICE = slice(10, 16)
TOOL_POSITION_OBSERVATION_SLICE = slice(0, 3)
RELATIVE_CAP_POSITION_OBSERVATION_SLICE = slice(19, 22)
PREVIOUS_GRIPPER_WIDTH_OBSERVATION_INDEX = 9
FLOW_ACTION_REPRESENTATION = (
    "delta_xyz_absolute_rot6d_delta_gripper_relative_to_current_observation"
)

FLOW_OBSERVATION_FIELDS = (
    "robot0_tool_pose_xyz_rot6d[9]",
    "previous_commanded_gripper_width[1]",
    "task_wrench_Fx_Fy_Fz_Tx_Ty_Tz[6]",
    "cap_position_in_robot_base[3]",
    "cap_position_relative_to_tool[3]",
    "turn_angle_seal_torque_seal_compression[3]",
    "left_right_fingertip_force_magnitude[2]",
    "episode_progress[1]",
)


@dataclass(frozen=True)
class SealTwistFlowConfig:
    observation_dim: int = FLOW_OBSERVATION_DIM
    action_dim: int = FLOW_ACTION_DIM
    horizon: int = 8
    hidden_dim: int = 512
    depth: int = 4
    time_embedding_dim: int = 64

    def __post_init__(self) -> None:
        for name in (
            "observation_dim",
            "action_dim",
            "horizon",
            "hidden_dim",
            "depth",
            "time_embedding_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.observation_dim != FLOW_OBSERVATION_DIM:
            raise ValueError(
                f"SealTwist Flow observation_dim must be {FLOW_OBSERVATION_DIM} so the "
                "full 6D wrench is part of every observation"
            )
        if self.action_dim != FLOW_ACTION_DIM:
            raise ValueError(f"SealTwist CAFE action_dim must be {FLOW_ACTION_DIM}")
        if self.time_embedding_dim % 2:
            raise ValueError("time_embedding_dim must be even")

    @property
    def flat_action_dim(self) -> int:
        return self.horizon * self.action_dim


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        frequencies = torch.exp(
            torch.linspace(0.0, math.log(1000.0), dimension // 2, dtype=torch.float32)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = time.reshape(-1, 1) * self.frequencies.reshape(1, -1)
        angles = angles * (2.0 * math.pi)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class SealTwistFlowPolicy(nn.Module):
    """Observation-conditioned vector field over normalized action chunks."""

    def __init__(self, config: SealTwistFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.time_embedding = FourierTimeEmbedding(config.time_embedding_dim)
        input_dim = (
            config.observation_dim
            + config.flat_action_dim
            + config.time_embedding_dim
        )
        self.input_projection = nn.Linear(input_dim, config.hidden_dim)
        self.blocks = nn.Sequential(
            *(ResidualBlock(config.hidden_dim) for _ in range(config.depth))
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.flat_action_dim),
        )

    def vector_field(
        self,
        observation: torch.Tensor,
        noisy_action: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        if observation.ndim != 2 or observation.shape[-1] != config.observation_dim:
            raise ValueError(
                f"observation must have shape [B, {config.observation_dim}], "
                f"got {tuple(observation.shape)}"
            )
        expected_action_shape = (config.horizon, config.action_dim)
        if noisy_action.ndim != 3 or tuple(noisy_action.shape[1:]) != expected_action_shape:
            raise ValueError(
                f"noisy_action must have shape [B, {config.horizon}, "
                f"{config.action_dim}], got {tuple(noisy_action.shape)}"
            )
        if len(observation) != len(noisy_action):
            raise ValueError("observation and noisy_action batch sizes differ")
        time = torch.as_tensor(time, dtype=observation.dtype, device=observation.device)
        if time.numel() == 1:
            time = time.expand(len(observation))
        if time.ndim != 1 or len(time) != len(observation):
            raise ValueError("time must be scalar or have shape [B]")
        inputs = torch.cat(
            (
                observation,
                noisy_action.reshape(len(noisy_action), -1),
                self.time_embedding(time).to(observation),
            ),
            dim=-1,
        )
        hidden = self.blocks(self.input_projection(inputs))
        return self.output(hidden).reshape_as(noisy_action)

    def flow_matching_loss(
        self,
        observation: torch.Tensor,
        target_action: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Straight-path conditional Flow Matching regression loss."""

        if noise is None:
            noise = torch.randn_like(target_action)
        if noise.shape != target_action.shape:
            raise ValueError("noise and target_action must have identical shapes")
        if time is None:
            time = torch.rand(len(target_action), device=target_action.device)
        time = torch.as_tensor(time, dtype=target_action.dtype, device=target_action.device)
        if time.numel() == 1:
            time = time.expand(len(target_action))
        interpolation = (
            (1.0 - time[:, None, None]) * noise
            + time[:, None, None] * target_action
        )
        target_velocity = target_action - noise
        prediction = self.vector_field(observation, interpolation, time)
        return torch.mean((prediction - target_velocity).square())

    @torch.inference_mode()
    def decode(
        self,
        observation: torch.Tensor,
        initial_noise: torch.Tensor,
        *,
        integration_steps: int = 16,
        solver: str = "heun",
    ) -> torch.Tensor:
        """Integrate the learned ODE from Gaussian noise to an action chunk."""

        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        state = initial_noise.clone()
        dt = 1.0 / float(integration_steps)
        for index in range(integration_steps):
            time = torch.full(
                (len(state),),
                index * dt,
                dtype=state.dtype,
                device=state.device,
            )
            velocity = self.vector_field(observation, state, time)
            if solver == "euler" or index == integration_steps - 1:
                state = state + dt * velocity
            else:
                proposal = state + dt * velocity
                next_velocity = self.vector_field(observation, proposal, time + dt)
                state = state + 0.5 * dt * (velocity + next_velocity)
        return state


def trajectory_flow_observation(trajectory, *, max_episode_steps: int) -> np.ndarray:
    """Build the exact 28D Flow observation from one serialized trajectory.

    The collector writes ``robot0_pos[:, 9]`` after the expert has selected the
    action for that same sample.  Using it directly would leak the gripper
    target.  At deployment the available value is the command left by the
    preceding policy sample, so reconstruct that causal quantity by shifting
    the recorded gripper action by one sample.
    """

    serialized_robot_state = np.asarray(trajectory["robot0_pos"], dtype=np.float32)
    robot_pose = serialized_robot_state[:, :9]
    action = np.asarray(trajectory["action"], dtype=np.float32)
    if action.ndim != 2 or action.shape != serialized_robot_state.shape:
        raise ValueError(
            "trajectory action and robot0_pos must both have shape [T, 10]"
        )
    previous_gripper_width = np.empty((len(action), 1), dtype=np.float32)
    previous_gripper_width[0, 0] = serialized_robot_state[0, 9]
    previous_gripper_width[1:, 0] = action[:-1, 9]
    wrench = np.asarray(trajectory["wrench"], dtype=np.float32)
    cap_pos_b = np.asarray(trajectory["cap_pose_b"], dtype=np.float32)[:, :3]
    relative_cap_pos_b = cap_pos_b - robot_pose[:, :3]
    task_state = np.stack(
        (
            trajectory["turn_angle_rad"],
            trajectory["seal_torque_nm"],
            trajectory["seal_compression_m"],
        ),
        axis=-1,
    ).astype(np.float32)
    finger_force_n = np.asarray(trajectory["finger_force_n"], dtype=np.float32)
    step_fraction = (
        np.asarray(trajectory["step"], dtype=np.float32)[:, None]
        / float(max_episode_steps)
    )
    observation = np.concatenate(
        (
            robot_pose,
            previous_gripper_width,
            wrench,
            cap_pos_b,
            relative_cap_pos_b,
            task_state,
            finger_force_n,
            step_fraction,
        ),
        axis=-1,
    ).astype(np.float32)
    if observation.shape[1] != FLOW_OBSERVATION_DIM:
        raise ValueError(
            f"expected {FLOW_OBSERVATION_DIM}-D Flow observations, got {observation.shape}"
        )
    return observation


def environment_flow_observation(env, *, max_episode_steps: int) -> torch.Tensor:
    """Build the runtime observation with the same ordering used for training."""

    serialized_robot_state = env.get_cafe_observation()["robot0_pos"]
    robot_pose = serialized_robot_state[:, :9]
    previous_gripper_width = serialized_robot_state[:, 9:10]
    wrench = env.get_task_wrench()
    cap_pos_b = env.labware.data.root_pos_w - env._robot.data.root_link_pos_w
    relative_cap_pos_b = cap_pos_b - robot_pose[:, :3]
    flags = env.get_task_flags()
    task_state = torch.stack(
        (
            flags["turn_angle_rad"],
            flags["seal_torque_nm"],
            flags["seal_compression_m"],
        ),
        dim=-1,
    )
    finger_force_n = torch.stack(
        (flags["left_finger_force_n"], flags["right_finger_force_n"]), dim=-1
    )
    step_fraction = env.step_count.float().unsqueeze(-1) / float(max_episode_steps)
    observation = torch.cat(
        (
            robot_pose,
            previous_gripper_width,
            wrench,
            cap_pos_b,
            relative_cap_pos_b,
            task_state,
            finger_force_n,
            step_fraction,
        ),
        dim=-1,
    )
    if observation.shape[1] != FLOW_OBSERVATION_DIM:
        raise ValueError(
            f"expected {FLOW_OBSERVATION_DIM}-D Flow observations, got {tuple(observation.shape)}"
        )
    return observation


@dataclass
class LoadedSealTwistFlowPolicy:
    model: SealTwistFlowPolicy
    observation_mean: torch.Tensor
    observation_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    max_episode_steps: int
    metadata: dict[str, Any]

    def normalize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation - self.observation_mean) / self.observation_std

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std + self.action_mean


def save_flow_checkpoint(
    path: str | Path,
    *,
    model: SealTwistFlowPolicy,
    observation_mean: torch.Tensor,
    observation_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    max_episode_steps: int,
    metadata: dict[str, Any],
) -> None:
    checkpoint = {
        "contract": "seal_twist_6d_wrench_conditional_flow_matching",
        "contract_version": FLOW_CONTRACT_VERSION,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "observation_mean": observation_mean.detach().cpu(),
        "observation_std": observation_std.detach().cpu(),
        "action_mean": action_mean.detach().cpu(),
        "action_std": action_std.detach().cpu(),
        "max_episode_steps": int(max_episode_steps),
        "observation_fields": list(FLOW_OBSERVATION_FIELDS),
        "wrench_observation_indices": [
            WRENCH_OBSERVATION_SLICE.start,
            WRENCH_OBSERVATION_SLICE.stop,
        ],
        "metadata": metadata,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)


def load_flow_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> LoadedSealTwistFlowPolicy:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != "seal_twist_6d_wrench_conditional_flow_matching":
        raise ValueError("checkpoint is not a SealTwist 6D-wrench Flow Matching policy")
    if int(checkpoint.get("contract_version", -1)) != FLOW_CONTRACT_VERSION:
        raise ValueError("unsupported SealTwist Flow checkpoint contract version")
    if checkpoint.get("wrench_observation_indices") != [10, 16]:
        raise ValueError("checkpoint does not expose the required 6D wrench observation")
    metadata = dict(checkpoint.get("metadata", {}))
    if metadata.get("action_representation") != FLOW_ACTION_REPRESENTATION:
        raise ValueError("checkpoint does not use the causal relative CAFE action contract")
    target_device = torch.device(device)
    model = SealTwistFlowPolicy(SealTwistFlowConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device).eval()
    return LoadedSealTwistFlowPolicy(
        model=model,
        observation_mean=checkpoint["observation_mean"].to(target_device),
        observation_std=checkpoint["observation_std"].to(target_device),
        action_mean=checkpoint["action_mean"].to(target_device),
        action_std=checkpoint["action_std"].to(target_device),
        max_episode_steps=int(checkpoint["max_episode_steps"]),
        metadata=metadata,
    )


__all__ = [
    "FLOW_ACTION_DIM",
    "FLOW_ACTION_REPRESENTATION",
    "FLOW_CONTRACT_VERSION",
    "FLOW_OBSERVATION_DIM",
    "FLOW_OBSERVATION_FIELDS",
    "RELATIVE_CAP_POSITION_OBSERVATION_SLICE",
    "PREVIOUS_GRIPPER_WIDTH_OBSERVATION_INDEX",
    "TOOL_POSITION_OBSERVATION_SLICE",
    "WRENCH_OBSERVATION_SLICE",
    "LoadedSealTwistFlowPolicy",
    "SealTwistFlowConfig",
    "SealTwistFlowPolicy",
    "environment_flow_observation",
    "load_flow_checkpoint",
    "save_flow_checkpoint",
    "trajectory_flow_observation",
]
