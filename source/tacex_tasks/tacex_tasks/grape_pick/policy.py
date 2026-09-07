from __future__ import annotations

import numpy as np
import torch
from torch import nn


POLICY_OBSERVATION_DIM = 11
POLICY_ACTION_DIM = 10


class GrapeBCPolicy(nn.Module):
    """Compact tactile-conditioned behavior-cloning baseline.

    Inputs include both physical pad-normal contact loads.  These are sampled
    online on every control step rather than reconstructed from target labels.
    """

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
    step_fraction = np.asarray(trajectory["step"], dtype=np.float32)[:, None] / float(max_episode_steps)
    observation = np.concatenate(
        (
            trajectory["grape_pos_b"],
            trajectory["stem_anchor_pos_b"],
            trajectory["finger_normal_force_n"],
            trajectory["stem_tension_n"][:, None],
            trajectory["stem_attached"][:, None].astype(np.float32),
            step_fraction,
        ),
        axis=-1,
    ).astype(np.float32)
    if observation.shape[1] != POLICY_OBSERVATION_DIM:
        raise ValueError(f"Expected {POLICY_OBSERVATION_DIM} policy inputs, received {observation.shape}")
    return observation


def environment_policy_observation(env, *, max_episode_steps: int) -> torch.Tensor:
    forces = env.get_contact_force_metrics()
    root_pos_w = env._robot.data.root_link_pos_w
    grape_pos_b = env.grape.data.root_pos_w - root_pos_w
    anchor_pos_b = env.stem_anchor_pos_w - root_pos_w
    finger_normals = torch.stack(
        (forces["left_normal_force_n"], forces["right_normal_force_n"]), dim=-1
    )
    step_fraction = env.step_count.float().unsqueeze(-1) / float(max_episode_steps)
    observation = torch.cat(
        (
            grape_pos_b,
            anchor_pos_b,
            finger_normals,
            env.stem_tension_n.unsqueeze(-1),
            env.stem_attached.float().unsqueeze(-1),
            step_fraction,
        ),
        dim=-1,
    )
    if observation.shape[1] != POLICY_OBSERVATION_DIM:
        raise ValueError(f"Expected {POLICY_OBSERVATION_DIM} policy inputs, received {observation.shape}")
    return observation


def normalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value - mean) / std.clamp_min(1.0e-6)


def denormalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return value * std.clamp_min(1.0e-6) + mean
