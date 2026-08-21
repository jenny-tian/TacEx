"""Minimal model contract for the 4-D clean residual SAC path.

The policy action is the bounded residual over normalized CAFE coordinates
``[x, y, z, width]``.  The environment owns composition with the frozen BC
action; critics therefore receive the post-tanh residual itself rather than a
second, internally composed action representation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


@dataclass(frozen=True)
class CleanResidualLayout:
    """Tensor and action-composition contract for the clean residual path."""

    policy: int = 29
    state: int = 19
    action: int = 4
    full_bc_context: int = 10
    indices: tuple[int, ...] = (0, 1, 2, 9)
    scale: float = 0.15

    def __post_init__(self) -> None:
        expected = {
            "policy": 29,
            "state": 19,
            "action": 4,
            "full_bc_context": 10,
        }
        for name, expected_value in expected.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, received {type(value).__name__}.")
            if value != expected_value:
                raise ValueError(
                    f"The clean residual v1 contract fixes {name}={expected_value}, "
                    f"received {value}."
                )
        if self.policy != self.state + self.full_bc_context:
            raise ValueError(
                "policy must equal state + full_bc_context, received "
                f"{self.policy}, {self.state}, and {self.full_bc_context}."
            )
        if not isinstance(self.indices, tuple):
            raise TypeError(
                "indices must be the immutable tuple (0, 1, 2, 9), received "
                f"{type(self.indices).__name__}."
            )
        if self.indices != (0, 1, 2, 9):
            raise ValueError(
                "The controlled normalized CAFE indices must be (0, 1, 2, 9), "
                f"received {self.indices}."
            )
        if len(self.indices) != self.action:
            raise ValueError("The number of controlled indices must equal the residual action dimension.")
        if isinstance(self.scale, bool) or not isinstance(self.scale, Real):
            raise TypeError(
                f"scale must be a real scalar, received {type(self.scale).__name__}."
            )
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise ValueError(f"scale must be finite and positive, received {self.scale}.")
        object.__setattr__(self, "scale", float(self.scale))

    @property
    def policy_dim(self) -> int:
        return self.policy

    @property
    def state_dim(self) -> int:
        return self.state

    @property
    def action_dim(self) -> int:
        return self.action

    @property
    def bc_action_dim(self) -> int:
        return self.full_bc_context

    @property
    def full_bc_context_dim(self) -> int:
        return self.full_bc_context

    @property
    def controlled_action_indices(self) -> tuple[int, ...]:
        return self.indices

    @property
    def residual_scale(self) -> float:
        return float(self.scale)

    @property
    def critic_input_dim(self) -> int:
        return self.state + self.full_bc_context + self.action

    @staticmethod
    def _validate_matrix(tensor: torch.Tensor, *, name: str, width: int) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, received {type(tensor).__name__}.")
        if tensor.ndim != 2 or tensor.shape[-1] != width:
            raise ValueError(f"{name} must have shape [B, {width}], got {tuple(tensor.shape)}.")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating dtype, received {tensor.dtype}.")

    def split_policy_observation(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the 19-D state context and complete 10-D frozen-BC action."""

        self._validate_matrix(observations, name="observations", width=self.policy)
        return observations[:, : self.state], observations[:, self.state :]

    def validate_states(self, states: torch.Tensor) -> None:
        self._validate_matrix(states, name="states", width=self.state)

    def validate_residuals(
        self,
        residuals: torch.Tensor,
        *,
        enforce_bounds: bool = False,
    ) -> None:
        self._validate_matrix(residuals, name="residuals", width=self.action)
        if enforce_bounds and bool((residuals.abs() > 1.0 + 1.0e-6).any()):
            raise ValueError("residuals must satisfy the post-tanh [-1, 1] action contract.")

    def select_controlled_bc_action(self, full_bc_action: torch.Tensor) -> torch.Tensor:
        self._validate_matrix(
            full_bc_action,
            name="full_bc_action",
            width=self.full_bc_context,
        )
        index = torch.as_tensor(self.indices, device=full_bc_action.device, dtype=torch.long)
        return full_bc_action.index_select(-1, index)

    def compose_controlled_action(
        self,
        full_bc_action: torch.Tensor,
        residuals: torch.Tensor,
    ) -> torch.Tensor:
        """Compose xyz/width residuals in the frozen policy's normalized space."""

        controlled_bc_action = self.select_controlled_bc_action(full_bc_action)
        self.validate_residuals(residuals, enforce_bounds=True)
        if controlled_bc_action.shape[0] != residuals.shape[0]:
            raise ValueError(
                "full_bc_action and residuals must have the same batch size, received "
                f"{controlled_bc_action.shape[0]} and {residuals.shape[0]}."
            )
        return controlled_bc_action + self.scale * residuals.to(controlled_bc_action)


class TanhSquashedGaussianActor(GaussianMixin, Model):
    """Gaussian policy with an explicit reparameterized tanh transform."""

    def _init_squashed_gaussian(
        self,
        observation_space,
        action_space,
        device,
        *,
        min_log_std: float,
        max_log_std: float,
    ) -> None:
        if not math.isfinite(min_log_std) or not math.isfinite(max_log_std):
            raise ValueError("log standard-deviation bounds must be finite.")
        if min_log_std >= max_log_std:
            raise ValueError(
                f"min_log_std must be below max_log_std, received {min_log_std} and {max_log_std}."
            )
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_mean_actions=False,
            clip_log_std=True,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

    @staticmethod
    def _tanh_log_abs_det_jacobian(pre_tanh: torch.Tensor) -> torch.Tensor:
        return 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))

    def deterministic_action(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        role: str = "policy",
    ) -> torch.Tensor:
        mean, _ = self.compute(inputs, role)
        return torch.tanh(mean)

    def act(self, inputs, *, role: str = ""):
        mean, outputs = self.compute(inputs, role)
        if mean.ndim != 2 or mean.shape[-1] != self.num_actions:
            raise ValueError(
                f"policy mean must have shape [B, {self.num_actions}], got {tuple(mean.shape)}."
            )
        if "log_std" not in outputs:
            raise KeyError("Policy compute output must contain 'log_std'.")
        log_std = outputs["log_std"]
        if log_std.shape != mean.shape:
            raise ValueError(
                f"log_std must match the policy mean shape {tuple(mean.shape)}, "
                f"got {tuple(log_std.shape)}."
            )
        log_std = torch.clamp(log_std, min=self._g_min_log_std, max=self._g_max_log_std)
        outputs["log_std"] = log_std

        distribution = Normal(mean, log_std.exp())
        self._g_distribution = distribution
        pre_tanh = distribution.rsample()
        actions = torch.tanh(pre_tanh)

        taken_actions = inputs.get("taken_actions")
        if taken_actions is None:
            log_prob_pre_tanh = pre_tanh
        else:
            if taken_actions.shape != mean.shape:
                raise ValueError(
                    f"taken_actions must have shape {tuple(mean.shape)}, "
                    f"got {tuple(taken_actions.shape)}."
                )
            if not taken_actions.is_floating_point():
                raise TypeError(
                    f"taken_actions must use a floating dtype, received {taken_actions.dtype}."
                )
            tolerance = 1.0e-6
            if bool((taken_actions.abs() > 1.0 + tolerance).any()):
                raise ValueError("taken_actions must satisfy the post-tanh [-1, 1] contract.")
            eps = torch.finfo(taken_actions.dtype).eps
            log_prob_pre_tanh = torch.atanh(
                taken_actions.clamp(min=-1.0 + eps, max=1.0 - eps)
            )

        log_prob = distribution.log_prob(log_prob_pre_tanh)
        log_prob -= self._tanh_log_abs_det_jacobian(log_prob_pre_tanh)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = torch.tanh(mean)
        outputs["pre_tanh_actions"] = pre_tanh
        return actions, outputs


class CleanResidualActor(TanhSquashedGaussianActor):
    """29-D observation to one 4-D post-tanh xyz/width residual."""

    hidden_dims: tuple[int, int] = (256, 256)

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        *,
        layout: CleanResidualLayout | None = None,
        initial_log_std: float = -2.0,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ) -> None:
        self.layout = CleanResidualLayout() if layout is None else layout
        if not isinstance(self.layout, CleanResidualLayout):
            raise TypeError(
                "layout must be a CleanResidualLayout, received "
                f"{type(self.layout).__name__}."
            )
        if not math.isfinite(initial_log_std):
            raise ValueError(f"initial_log_std must be finite, received {initial_log_std}.")
        if not min_log_std <= initial_log_std <= max_log_std:
            raise ValueError(
                "initial_log_std must lie inside the configured bounds, received "
                f"{initial_log_std} outside [{min_log_std}, {max_log_std}]."
            )
        self.initial_log_std = float(initial_log_std)
        self._init_squashed_gaussian(
            observation_space,
            action_space,
            device,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )
        if self.num_observations != self.layout.policy:
            raise ValueError(
                f"Actor observation space must flatten to {self.layout.policy}, "
                f"received {self.num_observations}."
            )
        if self.num_actions != self.layout.action:
            raise ValueError(
                f"Actor action space must flatten to {self.layout.action}, "
                f"received {self.num_actions}."
            )

        self.trunk = nn.Sequential(
            nn.Linear(self.layout.policy, self.hidden_dims[0]),
            nn.ELU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ELU(),
        )
        self.mean_head = nn.Linear(self.hidden_dims[-1], self.layout.action)
        self.log_std_head = nn.Linear(self.hidden_dims[-1], self.layout.action)
        self.reset_output_heads()

    def reset_output_heads(self) -> None:
        """Initialize the policy at zero residual and the requested deviation."""

        with torch.no_grad():
            self.mean_head.weight.zero_()
            self.mean_head.bias.zero_()
            self.log_std_head.weight.zero_()
            self.log_std_head.bias.fill_(self.initial_log_std)

    def compute(self, inputs, role):
        del role
        if "observations" not in inputs:
            raise KeyError("Actor inputs must contain 'observations'.")
        observations = inputs["observations"]
        self.layout.split_policy_observation(observations)
        features = self.trunk(observations)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        return mean, {"log_std": log_std}


class CleanResidualCritic(DeterministicMixin, Model):
    """Plain Q-function over state, full BC context, and raw residual."""

    hidden_dims: tuple[int, int] = (256, 256)

    def __init__(
        self,
        state_space,
        action_space,
        device,
        *,
        layout: CleanResidualLayout | None = None,
    ) -> None:
        self.layout = CleanResidualLayout() if layout is None else layout
        if not isinstance(self.layout, CleanResidualLayout):
            raise TypeError(
                "layout must be a CleanResidualLayout, received "
                f"{type(self.layout).__name__}."
            )
        Model.__init__(
            self,
            observation_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        if self.num_observations != self.layout.state:
            raise ValueError(
                f"Critic state space must flatten to {self.layout.state}, "
                f"received {self.num_observations}."
            )
        if self.num_actions != self.layout.action:
            raise ValueError(
                f"Critic action space must flatten to {self.layout.action}, "
                f"received {self.num_actions}."
            )

        self.network_input_dim = self.layout.critic_input_dim
        self.net = nn.Sequential(
            nn.Linear(self.network_input_dim, self.hidden_dims[0]),
            nn.ELU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ELU(),
            nn.Linear(self.hidden_dims[-1], 1),
        )

    def build_network_input(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = {"states", "observations", "taken_actions"}.difference(inputs)
        if missing:
            raise KeyError(f"Critic inputs are missing required keys: {sorted(missing)}.")
        states = inputs["states"]
        observations = inputs["observations"]
        residuals = inputs["taken_actions"]
        self.layout.validate_states(states)
        _, full_bc_action = self.layout.split_policy_observation(observations)
        self.layout.validate_residuals(residuals)
        batch_sizes = {states.shape[0], full_bc_action.shape[0], residuals.shape[0]}
        if len(batch_sizes) != 1:
            raise ValueError(
                "states, observations, and taken_actions must have one batch size, received "
                f"{states.shape[0]}, {full_bc_action.shape[0]}, and {residuals.shape[0]}."
            )
        network_input = torch.cat(
            (
                states,
                full_bc_action.to(states),
                residuals.to(states),
            ),
            dim=-1,
        )
        if network_input.shape[-1] != self.network_input_dim:
            raise RuntimeError(
                f"Critic input must have width {self.network_input_dim}, "
                f"got {network_input.shape[-1]}."
            )
        return network_input

    def compute(self, inputs, role):
        del role
        values = self.net(self.build_network_input(inputs))
        if values.ndim != 2 or values.shape[-1] != 1:
            raise RuntimeError(f"Critic output must have shape [B, 1], got {tuple(values.shape)}.")
        return values, {}


def build_clean_residual_sac_models(
    observation_space,
    state_space,
    action_space,
    device,
    *,
    layout: CleanResidualLayout | None = None,
    initial_log_std: float = -2.0,
    min_log_std: float = -5.0,
    max_log_std: float = 2.0,
) -> dict[str, Model]:
    """Build the five independent models required by the clean SAC agent."""

    resolved_layout = CleanResidualLayout() if layout is None else layout
    if not isinstance(resolved_layout, CleanResidualLayout):
        raise TypeError(
            "layout must be a CleanResidualLayout, received "
            f"{type(resolved_layout).__name__}."
        )

    policy = CleanResidualActor(
        observation_space,
        action_space,
        device,
        layout=resolved_layout,
        initial_log_std=initial_log_std,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
    ).to(device)

    def new_critic() -> CleanResidualCritic:
        return CleanResidualCritic(
            state_space,
            action_space,
            device,
            layout=resolved_layout,
        ).to(device)

    models: dict[str, Model] = {
        "policy": policy,
        "critic_1": new_critic(),
        "critic_2": new_critic(),
        "target_critic_1": new_critic(),
        "target_critic_2": new_critic(),
    }
    if len({id(model) for model in models.values()}) != len(models):
        raise RuntimeError("Model factory produced shared model instances.")
    parameter_ids = [
        {id(parameter) for parameter in model.parameters()}
        for model in models.values()
    ]
    for index, left in enumerate(parameter_ids):
        for right in parameter_ids[index + 1 :]:
            if not left.isdisjoint(right):
                raise RuntimeError("Model factory produced shared parameter objects.")
    return models


__all__ = [
    "CleanResidualActor",
    "CleanResidualCritic",
    "CleanResidualLayout",
    "TanhSquashedGaussianActor",
    "build_clean_residual_sac_models",
]
