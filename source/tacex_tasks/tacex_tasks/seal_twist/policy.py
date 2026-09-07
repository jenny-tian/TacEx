"""State/tactile behavior-cloning policy utilities for SealTwist."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


POLICY_OBSERVATION_DIM = 22
POLICY_ACTION_DIM = 10


class SealTwistBCPolicy(nn.Module):
    """Compact MLP mapping robot, cap, seal and fingertip state to CAFE actions."""

    def __init__(self, hidden_dim: int = 256, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = POLICY_OBSERVATION_DIM
        for _ in range(depth):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.SiLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, POLICY_ACTION_DIM))
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def trajectory_policy_observation(trajectory, *, max_episode_steps: int) -> np.ndarray:
    robot_state = np.asarray(trajectory["robot0_pos"], dtype=np.float32)
    cap_pos_b = np.asarray(trajectory["cap_pose_b"], dtype=np.float32)[:, :3]
    relative_cap_pos_b = cap_pos_b - robot_state[:, :3]
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
            robot_state,
            cap_pos_b,
            relative_cap_pos_b,
            task_state,
            finger_force_n,
            step_fraction,
        ),
        axis=-1,
    ).astype(np.float32)
    if observation.shape[1] != POLICY_OBSERVATION_DIM:
        raise ValueError(
            f"Expected {POLICY_OBSERVATION_DIM} policy inputs, received {observation.shape}"
        )
    return observation


def environment_policy_observation(env, *, max_episode_steps: int) -> torch.Tensor:
    robot_state = env.get_cafe_observation()["robot0_pos"]
    cap_pos_b = env.labware.data.root_pos_w - env._robot.data.root_link_pos_w
    relative_cap_pos_b = cap_pos_b - robot_state[:, :3]
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
            robot_state,
            cap_pos_b,
            relative_cap_pos_b,
            task_state,
            finger_force_n,
            step_fraction,
        ),
        dim=-1,
    )
    if observation.shape[1] != POLICY_OBSERVATION_DIM:
        raise ValueError(
            f"Expected {POLICY_OBSERVATION_DIM} policy inputs, received {observation.shape}"
        )
    return observation


def normalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value - mean) / std.clamp_min(1.0e-6)


def denormalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return value * std.clamp_min(1.0e-6) + mean
