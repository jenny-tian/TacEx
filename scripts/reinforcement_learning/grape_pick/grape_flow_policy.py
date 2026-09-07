"""State/tactile-conditioned Flow Matching policy for GrapePick.

The frozen policy maps independent Gaussian noise with shape ``[B, H, 10]``
to normalized CAFE action chunks.  DSRL is deliberately represented by a
bounded correction to that *initial noise*, never by an action-space residual.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


FLOW_CONTRACT_VERSION = 1
FLOW_NOISE_SEED_OFFSET = 321_991


@dataclass(frozen=True)
class GrapeFlowConfig:
    observation_dim: int = 11
    action_dim: int = 10
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
        if self.time_embedding_dim % 2:
            raise ValueError("time_embedding_dim must be even")

    @property
    def flat_action_dim(self) -> int:
        return self.horizon * self.action_dim


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        half = dimension // 2
        frequencies = torch.exp(
            torch.linspace(0.0, math.log(1000.0), half, dtype=torch.float32)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = time.reshape(-1, 1)
        angles = time * self.frequencies.reshape(1, -1) * (2.0 * math.pi)
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


class GrapeFlowPolicy(nn.Module):
    """Conditional vector field for straight-line conditional flow matching."""

    def __init__(self, config: GrapeFlowConfig) -> None:
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
        if noisy_action.ndim != 3 or tuple(noisy_action.shape[1:]) != (
            config.horizon,
            config.action_dim,
        ):
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
            raise ValueError("time must be scalar or shape [B]")
        flat_action = noisy_action.reshape(len(noisy_action), -1)
        inputs = torch.cat(
            (observation, flat_action, self.time_embedding(time).to(observation)),
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
        if noise is None:
            noise = torch.randn_like(target_action)
        if noise.shape != target_action.shape:
            raise ValueError("noise and target_action must have identical shapes")
        if time is None:
            time = torch.rand(len(target_action), device=target_action.device)
        time = torch.as_tensor(time, device=target_action.device, dtype=target_action.dtype)
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
        """Integrate the learned ODE from noise (t=0) to action (t=1)."""

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
                next_time = time + dt
                next_velocity = self.vector_field(observation, proposal, next_time)
                state = state + 0.5 * dt * (velocity + next_velocity)
        return state


@dataclass(frozen=True)
class FlowNoiseDSRLLayout:
    """Short bounded SAC correction expanded into a Flow-noise tensor."""

    horizon: int
    action_dim: int
    learned_noise_steps: int = 1
    residual_scale: float = 0.25
    padding_mode: str = "repeat_last"
    active_action_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.action_dim < 1 or self.learned_noise_steps < 1:
            raise ValueError("noise dimensions must be positive")
        if self.learned_noise_steps > self.horizon:
            raise ValueError("learned_noise_steps cannot exceed horizon")
        if self.padding_mode not in {"repeat_last", "zeros"}:
            raise ValueError("padding_mode must be repeat_last or zeros")
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0.0:
            raise ValueError("residual_scale must be finite and non-negative")
        active = (
            tuple(range(self.action_dim))
            if self.active_action_indices is None
            else tuple(int(index) for index in self.active_action_indices)
        )
        if not active:
            raise ValueError("at least one Flow-noise action index must be active")
        if len(set(active)) != len(active):
            raise ValueError("Flow-noise action indices must be unique")
        if any(index < 0 or index >= self.action_dim for index in active):
            raise ValueError(
                f"Flow-noise action indices must lie in [0, {self.action_dim})"
            )
        object.__setattr__(self, "active_action_indices", active)

    @property
    def correction_dim(self) -> int:
        assert self.active_action_indices is not None
        return self.learned_noise_steps * len(self.active_action_indices)

    def expand_correction(self, correction: torch.Tensor) -> torch.Tensor:
        if correction.ndim != 2 or correction.shape[-1] != self.correction_dim:
            raise ValueError(
                f"correction must have shape [B, {self.correction_dim}], "
                f"got {tuple(correction.shape)}"
            )
        if bool((correction.abs() > 1.0 + 1.0e-6).any()):
            raise ValueError("bounded DSRL correction must lie in [-1, 1]")
        assert self.active_action_indices is not None
        learned_active = correction.reshape(
            len(correction), self.learned_noise_steps, len(self.active_action_indices)
        )
        learned = correction.new_zeros(
            (len(correction), self.learned_noise_steps, self.action_dim)
        )
        learned[:, :, list(self.active_action_indices)] = learned_active
        if self.learned_noise_steps == self.horizon:
            return learned
        pad_steps = self.horizon - self.learned_noise_steps
        if self.padding_mode == "repeat_last":
            padding = learned[:, -1:, :].expand(-1, pad_steps, -1)
        else:
            padding = learned.new_zeros((len(learned), pad_steps, self.action_dim))
        return torch.cat((learned, padding), dim=1)

    def compose_noise(
        self, native_noise: torch.Tensor, correction: torch.Tensor
    ) -> torch.Tensor:
        expected = (len(correction), self.horizon, self.action_dim)
        if tuple(native_noise.shape) != expected:
            raise ValueError(
                f"native_noise must have shape {expected}, got {tuple(native_noise.shape)}"
            )
        expanded = self.expand_correction(correction).to(native_noise)
        return native_noise + self.residual_scale * expanded


def pull_phase_noise_gate(
    physics_step: int | torch.Tensor,
    bilateral_contact: torch.Tensor,
    stem_attached: torch.Tensor,
    *,
    minimum_step: int,
) -> torch.Tensor:
    """Return where a Flow-noise correction may be applied and learned.

    The frozen Flow policy owns approach, alignment, and grasp.  DSRL becomes
    active only after the demonstrated pull phase begins while both fingers
    remain in physical contact with an attached grape.
    """

    if minimum_step < 0:
        raise ValueError("minimum_step must be non-negative")
    bilateral = bilateral_contact.bool()
    attached = stem_attached.bool()
    if bilateral.shape != attached.shape:
        raise ValueError("bilateral_contact and stem_attached shapes must match")
    step = torch.as_tensor(physics_step, device=bilateral.device)
    return (step >= minimum_step) & bilateral & attached


@dataclass
class LoadedGrapeFlowPolicy:
    model: GrapeFlowPolicy
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
    model: GrapeFlowPolicy,
    observation_mean: torch.Tensor,
    observation_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    max_episode_steps: int,
    metadata: dict[str, Any],
) -> None:
    checkpoint = {
        "contract": "grape_state_tactile_conditional_flow_matching",
        "contract_version": FLOW_CONTRACT_VERSION,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "observation_mean": observation_mean.detach().cpu(),
        "observation_std": observation_std.detach().cpu(),
        "action_mean": action_mean.detach().cpu(),
        "action_std": action_std.detach().cpu(),
        "max_episode_steps": int(max_episode_steps),
        "metadata": metadata,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)


def load_flow_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> LoadedGrapeFlowPolicy:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != "grape_state_tactile_conditional_flow_matching":
        raise ValueError("checkpoint is not a Grape Flow Matching policy")
    if int(checkpoint.get("contract_version", -1)) != FLOW_CONTRACT_VERSION:
        raise ValueError("unsupported Grape Flow checkpoint contract version")
    target_device = torch.device(device)
    model = GrapeFlowPolicy(GrapeFlowConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device).eval()
    return LoadedGrapeFlowPolicy(
        model=model,
        observation_mean=checkpoint["observation_mean"].to(target_device),
        observation_std=checkpoint["observation_std"].to(target_device),
        action_mean=checkpoint["action_mean"].to(target_device),
        action_std=checkpoint["action_std"].to(target_device),
        max_episode_steps=int(checkpoint["max_episode_steps"]),
        metadata=dict(checkpoint.get("metadata", {})),
    )


__all__ = [
    "FLOW_CONTRACT_VERSION",
    "FlowNoiseDSRLLayout",
    "FLOW_NOISE_SEED_OFFSET",
    "GrapeFlowConfig",
    "GrapeFlowPolicy",
    "LoadedGrapeFlowPolicy",
    "load_flow_checkpoint",
    "pull_phase_noise_gate",
    "save_flow_checkpoint",
]
