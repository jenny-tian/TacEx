"""PPO baseline for the latent noise of a frozen Flow Matching policy.

The policy predicts the bounded initial noise consumed by the existing frozen
Flow decoder. This keeps the action/observation contract identical to Clean
DSRL while replacing its off-policy SAC optimizer with on-policy PPO.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, Model

from clean_dsrl_sac import CleanDSRLActor, CleanDSRLLayout


class FlowPPOValue(DeterministicMixin, Model):
    """Privileged value function used only while training Flow-PPO."""

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
        if not self.hidden_dims or any(value < 1 for value in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions.")
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        state_width = int(state_space.shape[-1])
        if state_width != layout.state_dim:
            raise ValueError(
                f"Value state space must flatten to {layout.state_dim}, "
                f"received {state_width}."
            )
        dimensions = (layout.state_dim, *self.hidden_dims)
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.ELU()))
        layers.append(nn.Linear(dimensions[-1], 1))
        self.net = nn.Sequential(*layers)

    def compute(self, inputs, role):
        del role
        states = inputs.get("states")
        if states is None:
            raise KeyError("Flow-PPO value inputs must contain 'states'.")
        self.layout.validate_states(states)
        return self.net(states), {}


def build_flow_ppo_models(
    observation_space,
    state_space,
    action_space,
    device,
    *,
    layout: CleanDSRLLayout,
    actor_hidden_dims: Sequence[int] = (512, 512, 512),
    value_hidden_dims: Sequence[int] = (512, 512, 512),
    initial_log_std: float = -2.0,
) -> dict[str, Model]:
    """Build independent stochastic policy and privileged value networks."""

    policy = CleanDSRLActor(
        observation_space,
        action_space,
        device,
        layout=layout,
        hidden_dims=actor_hidden_dims,
        initial_log_std=initial_log_std,
    ).to(device)
    value = FlowPPOValue(
        observation_space,
        state_space,
        action_space,
        device,
        layout=layout,
        hidden_dims=value_hidden_dims,
    ).to(device)
    if not {id(parameter) for parameter in policy.parameters()}.isdisjoint(
        {id(parameter) for parameter in value.parameters()}
    ):
        raise RuntimeError("Flow-PPO policy and value networks share parameters.")
    return {"policy": policy, "value": value}


__all__ = ["FlowPPOValue", "build_flow_ppo_models"]
