"""Model contract for tactile-conditioned, base-anchored Flow-noise DSRL-SAC.

The frozen Flow Matching policy consumes a ``[32, 10]`` initial-noise tensor.
The actor learns only a short, bounded correction. The last learned row is
repeated to construct the full correction and is added to a native independent
Gaussian Flow sample. Thus a zero actor action is exactly the frozen base
policy instead of the degenerate all-zero/repeated-noise policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, Model

try:
    from .clean_residual_sac import TanhSquashedGaussianActor
except ImportError:
    from clean_residual_sac import TanhSquashedGaussianActor


CLEAN_DSRL_CONTRACT_VERSION = 4
CLEAN_DSRL_CONTRACT_BUFFER = "_clean_dsrl_contract_version"


def validate_absolute_dsrl_policy_state(policy_state: Mapping[str, object]) -> None:
    """Reject checkpoints that predate the base-anchored tactile v4 contract."""

    if CLEAN_DSRL_CONTRACT_BUFFER not in policy_state:
        raise ValueError(
            "Legacy Clean DSRL checkpoint is incompatible with the base-anchored tactile v4 "
            "contract: the policy has no contract-version marker."
        )
    version = torch.as_tensor(policy_state[CLEAN_DSRL_CONTRACT_BUFFER]).reshape(-1)
    if version.numel() != 1 or int(version.item()) != CLEAN_DSRL_CONTRACT_VERSION:
        received = version.tolist()
        raise ValueError(
            "Clean DSRL checkpoint contract mismatch: expected base-anchored tactile "
            f"v{CLEAN_DSRL_CONTRACT_VERSION}, received {received}."
        )


@dataclass(frozen=True)
class CleanDSRLLayout:
    """Dimensions and noise-expansion rules shared by the DSRL stack."""

    policy: int
    state: int = 19
    noise_dim: int = 10
    flow_horizon: int = 32
    learned_noise_steps: int = 1
    padding_mode: str = "repeat_last"
    tactile: int = 5
    residual_scale: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "policy",
            "state",
            "noise_dim",
            "flow_horizon",
            "learned_noise_steps",
            "tactile",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive, received {value}.")
        if self.state != 19:
            raise ValueError(f"The LabPick DSRL critic state is fixed at 19-D, received {self.state}.")
        if self.noise_dim != 10:
            raise ValueError(f"The Flow policy noise width is fixed at 10-D, received {self.noise_dim}.")
        if self.tactile != 5:
            raise ValueError(f"The tactile actor vector is fixed at 5-D, received {self.tactile}.")
        if self.policy <= self.tactile:
            raise ValueError("policy must include a non-empty Flow condition plus the 5-D tactile vector.")
        if self.learned_noise_steps > self.flow_horizon:
            raise ValueError("learned_noise_steps cannot exceed flow_horizon.")
        if self.padding_mode not in {"repeat_last", "zeros"}:
            raise ValueError("padding_mode must be 'repeat_last' or 'zeros'.")
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0.0:
            raise ValueError("residual_scale must be finite and non-negative.")

    @property
    def policy_dim(self) -> int:
        return self.policy

    @property
    def state_dim(self) -> int:
        return self.state

    @property
    def action_dim(self) -> int:
        return self.learned_noise_steps * self.noise_dim

    @property
    def flow_condition_dim(self) -> int:
        return self.policy - self.tactile

    @property
    def tactile_dim(self) -> int:
        return self.tactile

    @property
    def critic_input_dim(self) -> int:
        return self.policy + self.state + self.action_dim

    @staticmethod
    def _validate_matrix(tensor: torch.Tensor, *, name: str, width: int) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.ndim != 2 or tensor.shape[-1] != width:
            raise ValueError(f"{name} must have shape [B, {width}], got {tuple(tensor.shape)}.")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating dtype, received {tensor.dtype}.")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains non-finite values.")

    def validate_policy_observations(self, observations: torch.Tensor) -> None:
        self._validate_matrix(observations, name="observations", width=self.policy)

    def validate_states(self, states: torch.Tensor) -> None:
        self._validate_matrix(states, name="states", width=self.state)

    def validate_actions(self, actions: torch.Tensor, *, enforce_bounds: bool = False) -> None:
        self._validate_matrix(actions, name="actions", width=self.action_dim)
        if enforce_bounds and bool((actions.abs() > 1.0 + 1.0e-6).any()):
            raise ValueError("DSRL policy actions must satisfy the post-tanh [-1, 1] contract.")

    def expand_noise(self, actions: torch.Tensor) -> torch.Tensor:
        """Expand a bounded SAC correction to the full Flow-noise tensor."""

        self.validate_actions(actions, enforce_bounds=True)
        learned = actions.reshape(-1, self.learned_noise_steps, self.noise_dim)
        if self.learned_noise_steps == self.flow_horizon:
            return learned
        pad_steps = self.flow_horizon - self.learned_noise_steps
        if self.padding_mode == "repeat_last":
            padding = learned[:, -1:, :].expand(-1, pad_steps, -1)
        else:
            padding = learned.new_zeros((learned.shape[0], pad_steps, self.noise_dim))
        return torch.cat((learned, padding), dim=1)

    def compose_noise(
        self,
        native_noise: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Add the scaled DSRL correction to an independent native Flow sample."""

        if not isinstance(native_noise, torch.Tensor):
            raise TypeError("native_noise must be a torch.Tensor.")
        expected = (actions.shape[0], self.flow_horizon, self.noise_dim)
        if tuple(native_noise.shape) != expected:
            raise ValueError(
                f"native_noise must have shape {expected}, got {tuple(native_noise.shape)}."
            )
        if not native_noise.is_floating_point() or not bool(torch.isfinite(native_noise).all()):
            raise ValueError("native_noise must be finite and floating point.")
        correction = self.expand_noise(actions).to(native_noise)
        return native_noise + self.residual_scale * correction


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    dims = [int(input_dim), *[int(value) for value in hidden_dims]]
    if len(dims) == 1:
        raise ValueError("hidden_dims must contain at least one layer.")
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        if out_dim < 1:
            raise ValueError("hidden dimensions must be positive.")
        layers.extend((nn.Linear(in_dim, out_dim), nn.ELU()))
    layers.append(nn.Linear(dims[-1], int(output_dim)))
    return nn.Sequential(*layers)


class CleanDSRLActor(TanhSquashedGaussianActor):
    """Flow-condition encoder features to a short learned noise chunk."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        *,
        layout: CleanDSRLLayout,
        hidden_dims: Sequence[int] = (512, 512, 512),
        initial_log_std: float = -2.0,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ) -> None:
        if not isinstance(layout, CleanDSRLLayout):
            raise TypeError("layout must be a CleanDSRLLayout.")
        if not min_log_std <= initial_log_std <= max_log_std:
            raise ValueError("initial_log_std must lie inside the configured bounds.")
        self.layout = layout
        self.initial_log_std = float(initial_log_std)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self._init_squashed_gaussian(
            observation_space,
            action_space,
            device,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )
        self.register_buffer(
            CLEAN_DSRL_CONTRACT_BUFFER,
            torch.tensor(CLEAN_DSRL_CONTRACT_VERSION, dtype=torch.int64),
            persistent=True,
        )
        if self.num_observations != layout.policy_dim:
            raise ValueError(
                f"Actor observation space must flatten to {layout.policy_dim}, "
                f"received {self.num_observations}."
            )
        if self.num_actions != layout.action_dim:
            raise ValueError(
                f"Actor action space must flatten to {layout.action_dim}, received {self.num_actions}."
            )
        trunk_layers: list[nn.Module] = []
        trunk_dims = (layout.policy_dim, *self.hidden_dims)
        for in_dim, out_dim in zip(trunk_dims[:-1], trunk_dims[1:]):
            trunk_layers.extend((nn.Linear(in_dim, out_dim), nn.ELU()))
        self.trunk = nn.Sequential(*trunk_layers)
        self.mean_head = nn.Linear(self.hidden_dims[-1], layout.action_dim)
        self.log_std_head = nn.Linear(self.hidden_dims[-1], layout.action_dim)
        self.reset_output_heads()

    def reset_output_heads(self) -> None:
        """Start from a zero-mean, state-independent noise distribution."""

        with torch.no_grad():
            self.mean_head.weight.zero_()
            self.mean_head.bias.zero_()
            self.log_std_head.weight.zero_()
            self.log_std_head.bias.fill_(self.initial_log_std)

    def compute(self, inputs, role):
        del role
        observations = inputs.get("observations")
        if observations is None:
            raise KeyError("Actor inputs must contain 'observations'.")
        self.layout.validate_policy_observations(observations)
        features = self.trunk(observations)
        return self.mean_head(features), {"log_std": self.log_std_head(features)}


class CleanDSRLCritic(DeterministicMixin, Model):
    """Twin-Q component over Flow features, privileged state, and learned noise."""

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        *,
        layout: CleanDSRLLayout,
        hidden_dims: Sequence[int] = (512, 512, 512),
    ) -> None:
        if not isinstance(layout, CleanDSRLLayout):
            raise TypeError("layout must be a CleanDSRLLayout.")
        self.layout = layout
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        if self.num_observations != layout.policy_dim:
            raise ValueError("Critic observation space does not match the DSRL layout.")
        if self.num_actions != layout.action_dim:
            raise ValueError("Critic action space does not match the DSRL layout.")
        state_width = int(state_space.shape[-1])
        if state_width != layout.state_dim:
            raise ValueError(
                f"Critic state space must flatten to {layout.state_dim}, received {state_width}."
            )
        self.net = _mlp(layout.critic_input_dim, self.hidden_dims, 1)

    def build_network_input(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = {"observations", "states", "taken_actions"}.difference(inputs)
        if missing:
            raise KeyError(f"Critic inputs are missing required keys: {sorted(missing)}.")
        observations = inputs["observations"]
        states = inputs["states"]
        actions = inputs["taken_actions"]
        self.layout.validate_policy_observations(observations)
        self.layout.validate_states(states)
        self.layout.validate_actions(actions)
        if len({observations.shape[0], states.shape[0], actions.shape[0]}) != 1:
            raise ValueError("observations, states, and actions must have the same batch size.")
        return torch.cat((observations, states.to(observations), actions.to(observations)), dim=-1)

    def compute(self, inputs, role):
        del role
        return self.net(self.build_network_input(inputs)), {}


def build_clean_dsrl_sac_models(
    observation_space,
    state_space,
    action_space,
    device,
    *,
    layout: CleanDSRLLayout,
    actor_hidden_dims: Sequence[int] = (512, 512, 512),
    critic_hidden_dims: Sequence[int] = (512, 512, 512),
    initial_log_std: float = -2.0,
) -> dict[str, Model]:
    """Build independent policy, online critics, and target critics."""

    policy = CleanDSRLActor(
        observation_space,
        action_space,
        device,
        layout=layout,
        hidden_dims=actor_hidden_dims,
        initial_log_std=initial_log_std,
    ).to(device)

    def new_critic() -> CleanDSRLCritic:
        return CleanDSRLCritic(
            observation_space,
            state_space,
            action_space,
            device,
            layout=layout,
            hidden_dims=critic_hidden_dims,
        ).to(device)

    models: dict[str, Model] = {
        "policy": policy,
        "critic_1": new_critic(),
        "critic_2": new_critic(),
        "target_critic_1": new_critic(),
        "target_critic_2": new_critic(),
    }
    parameter_ids = [{id(parameter) for parameter in model.parameters()} for model in models.values()]
    for index, left in enumerate(parameter_ids):
        for right in parameter_ids[index + 1 :]:
            if not left.isdisjoint(right):
                raise RuntimeError("Model factory produced shared parameter objects.")
    return models


__all__ = [
    "CLEAN_DSRL_CONTRACT_BUFFER",
    "CLEAN_DSRL_CONTRACT_VERSION",
    "CleanDSRLActor",
    "CleanDSRLCritic",
    "CleanDSRLLayout",
    "build_clean_dsrl_sac_models",
    "validate_absolute_dsrl_policy_state",
]
