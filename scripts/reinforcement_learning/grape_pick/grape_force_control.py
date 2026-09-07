"""Episode-level VLM force-range adaptation and 120 Hz gripper force control."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ForceRange:
    minimum_n: float
    maximum_n: float

    def __post_init__(self) -> None:
        low = float(self.minimum_n)
        high = float(self.maximum_n)
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("force range endpoints must be finite")
        if low < 0.0 or high <= low:
            raise ValueError("force range must satisfy 0 <= low < high")
        object.__setattr__(self, "minimum_n", low)
        object.__setattr__(self, "maximum_n", high)

    @property
    def center_n(self) -> float:
        return 0.5 * (self.minimum_n + self.maximum_n)

    @property
    def width_n(self) -> float:
        return self.maximum_n - self.minimum_n

    def as_list(self) -> list[float]:
        return [self.minimum_n, self.maximum_n]


@dataclass(frozen=True)
class ForceRangeAdapterConfig:
    physical_minimum_n: float = 2.60
    physical_maximum_n: float = 3.25
    minimum_width_n: float = 0.15
    smoothing: float = 0.35

    def __post_init__(self) -> None:
        if self.physical_minimum_n < 0.0:
            raise ValueError("physical_minimum_n must be non-negative")
        if self.physical_maximum_n <= self.physical_minimum_n:
            raise ValueError("physical force bounds are invalid")
        if not 0.0 < self.minimum_width_n <= (
            self.physical_maximum_n - self.physical_minimum_n
        ):
            raise ValueError("minimum_width_n is invalid")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must lie in (0, 1]")


class OnlineForceRangeAdapter:
    """Project and smooth one VLM proposal after every completed episode."""

    def __init__(
        self,
        initial_range: ForceRange,
        config: ForceRangeAdapterConfig | None = None,
    ) -> None:
        self.config = ForceRangeAdapterConfig() if config is None else config
        self.current_range = self._project(initial_range)
        self.decisions: list[dict[str, Any]] = []

    def _project(self, proposed: ForceRange) -> ForceRange:
        config = self.config
        low = min(
            max(proposed.minimum_n, config.physical_minimum_n),
            config.physical_maximum_n - config.minimum_width_n,
        )
        high = min(max(proposed.maximum_n, low + config.minimum_width_n), config.physical_maximum_n)
        if high - low < config.minimum_width_n:
            low = max(config.physical_minimum_n, high - config.minimum_width_n)
        return ForceRange(low, high)

    def update_after_episode(
        self,
        *,
        episode_index: int,
        proposed_range: ForceRange | None,
        raw_response: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Create the range applied to episode ``episode_index + 1``."""

        if episode_index != len(self.decisions):
            raise RuntimeError(
                f"expected completed episode {len(self.decisions)}, got {episode_index}"
            )
        previous = self.current_range
        if proposed_range is None:
            projected = previous
            applied = previous
            update_kind = "vlm_invalid_keep_previous"
        else:
            projected = self._project(proposed_range)
            alpha = self.config.smoothing
            applied = self._project(
                ForceRange(
                    (1.0 - alpha) * previous.minimum_n
                    + alpha * projected.minimum_n,
                    (1.0 - alpha) * previous.maximum_n
                    + alpha * projected.maximum_n,
                )
            )
            update_kind = "vlm_projected_smoothed"
        self.current_range = applied
        decision = {
            "completed_episode_index": int(episode_index),
            "applies_to_episode_index": int(episode_index + 1),
            "previous_range_n": previous.as_list(),
            "vlm_proposed_range_n": (
                None if proposed_range is None else proposed_range.as_list()
            ),
            "projected_range_n": projected.as_list(),
            "applied_range_n": applied.as_list(),
            "update_kind": update_kind,
            "raw_response": str(raw_response),
            "error": error,
            "safety_bounds_n": [
                self.config.physical_minimum_n,
                self.config.physical_maximum_n,
            ],
            "smoothing": self.config.smoothing,
        }
        self.decisions.append(decision)
        return decision


@dataclass(frozen=True)
class ForceControllerConfig:
    action_dim: int = 10
    gripper_width_index: int = 9
    minimum_width_m: float = 0.010
    maximum_width_m: float = 0.040
    kp_width_rate_per_n: float = 0.006
    ki_width_rate_per_n_s: float = 0.018
    kd_width_per_n: float = 0.00008
    maximum_width_rate_m_s: float = 0.006
    free_space_close_rate_m_s: float = 0.012
    safety_open_rate_m_s: float = 0.006
    integral_limit_n_s: float = 1.5
    force_filter_alpha: float = 0.35
    contact_on_force_n: float = 0.20
    contact_off_force_n: float = 0.08
    release_hysteresis_steps: int = 6
    contact_settle_steps: int = 2
    hard_force_limit_n: float = 3.30

    def __post_init__(self) -> None:
        if self.action_dim < 1 or not 0 <= self.gripper_width_index < self.action_dim:
            raise ValueError("action layout is invalid")
        if self.maximum_width_m <= self.minimum_width_m:
            raise ValueError("width limits are invalid")
        if (
            self.maximum_width_rate_m_s <= 0.0
            or self.free_space_close_rate_m_s <= 0.0
            or self.safety_open_rate_m_s <= 0.0
        ):
            raise ValueError("controller rates must be positive")
        if not 0.0 < self.force_filter_alpha <= 1.0:
            raise ValueError("force_filter_alpha must lie in (0, 1]")


@dataclass(frozen=True)
class ForceControlDiagnostics:
    active: torch.Tensor
    filtered_force_n: torch.Tensor
    target_force_n: float
    error_n: torch.Tensor
    command_width_m: torch.Tensor
    safety_override: torch.Tensor


class GrapeForceController:
    """PI-D force servo that owns only CAFE action index 9 after contact."""

    def __init__(
        self,
        target_range: ForceRange,
        config: ForceControllerConfig | None = None,
    ) -> None:
        self.config = ForceControllerConfig() if config is None else config
        self.set_target_range(target_range)
        self.reset()

    def set_target_range(self, target_range: ForceRange) -> None:
        if target_range.maximum_n > self.config.hard_force_limit_n:
            raise ValueError("target force range exceeds the hard force limit")
        self.target_range = target_range

    def reset(self) -> None:
        self._command_width: torch.Tensor | None = None
        self._filtered_force: torch.Tensor | None = None
        self._previous_force: torch.Tensor | None = None
        self._integral: torch.Tensor | None = None
        self._active: torch.Tensor | None = None
        self._dropout_steps: torch.Tensor | None = None
        self._active_age: torch.Tensor | None = None
        self._last_executed_width: torch.Tensor | None = None

    def _initialize(self, action: torch.Tensor, measured_force: torch.Tensor) -> None:
        batch = len(action)
        width = action[:, self.config.gripper_width_index]
        self._command_width = width.detach().clone()
        self._filtered_force = measured_force.detach().clone()
        self._previous_force = measured_force.detach().clone()
        self._integral = torch.zeros_like(measured_force)
        self._active = torch.zeros(batch, dtype=torch.bool, device=action.device)
        self._dropout_steps = torch.zeros(batch, dtype=torch.long, device=action.device)
        self._active_age = torch.zeros(batch, dtype=torch.long, device=action.device)
        self._last_executed_width = width.detach().clone()

    def control(
        self,
        policy_action: torch.Tensor,
        *,
        contact_force_n: torch.Tensor,
        bilateral_contact: torch.Tensor,
        dt_s: float,
    ) -> tuple[torch.Tensor, ForceControlDiagnostics]:
        config = self.config
        if policy_action.ndim != 2 or policy_action.shape[-1] != config.action_dim:
            raise ValueError(
                f"policy_action must have shape [B, {config.action_dim}]"
            )
        measured = torch.as_tensor(
            contact_force_n, dtype=policy_action.dtype, device=policy_action.device
        ).reshape(-1)
        bilateral = torch.as_tensor(
            bilateral_contact, dtype=torch.bool, device=policy_action.device
        ).reshape(-1)
        if len(measured) != len(policy_action) or len(bilateral) != len(policy_action):
            raise ValueError("force/contact batch does not match action batch")
        if not math.isfinite(float(dt_s)) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if self._command_width is None or len(self._command_width) != len(policy_action):
            self._initialize(policy_action, measured)

        assert self._command_width is not None
        assert self._filtered_force is not None
        assert self._previous_force is not None
        assert self._integral is not None
        assert self._active is not None
        assert self._dropout_steps is not None
        assert self._active_age is not None
        assert self._last_executed_width is not None

        requested = bilateral & (measured >= config.contact_on_force_n)
        loaded = measured >= config.contact_off_force_n
        self._dropout_steps = torch.where(
            self._active & ~requested & ~loaded,
            self._dropout_steps + 1,
            torch.zeros_like(self._dropout_steps),
        )
        keep_active = self._active & (
            loaded | (self._dropout_steps <= config.release_hysteresis_steps)
        )
        next_active = requested | keep_active
        just_activated = next_active & ~self._active
        just_released = ~next_active & self._active
        raw_policy_width = policy_action[:, config.gripper_width_index]
        # Before physical bilateral contact the frozen Flow BC owns the
        # gripper trajectory exactly.  Rate-limiting this free-space command
        # delays closure by hundreds of 120 Hz ticks and can prevent contact
        # altogether.  The force loop takes ownership only after contact (or
        # an independent hard-force safety override).
        policy_width = raw_policy_width.clamp(
            config.minimum_width_m, config.maximum_width_m
        )
        self._command_width = torch.where(
            ~self._active, policy_width, self._command_width
        )
        self._filtered_force = torch.where(
            just_activated,
            measured,
            config.force_filter_alpha * measured
            + (1.0 - config.force_filter_alpha) * self._filtered_force,
        )
        derivative = (self._filtered_force - self._previous_force) / float(dt_s)
        derivative = torch.where(just_activated, torch.zeros_like(derivative), derivative)
        error = self.target_range.center_n - self._filtered_force
        candidate_integral = torch.clamp(
            self._integral + error * float(dt_s),
            -config.integral_limit_n_s,
            config.integral_limit_n_s,
        )
        self._integral = torch.where(
            next_active, candidate_integral, torch.zeros_like(candidate_integral)
        )
        width_rate = (
            -config.kp_width_rate_per_n * error
            - config.ki_width_rate_per_n_s * self._integral
            + config.kd_width_per_n * derivative
        ).clamp(-config.maximum_width_rate_m_s, config.maximum_width_rate_m_s)
        settling = next_active & (self._active_age < config.contact_settle_steps)
        width_rate = torch.where(
            settling, torch.clamp(width_rate, min=0.0), width_rate
        )
        safety_threshold = min(
            self.target_range.maximum_n, config.hard_force_limit_n
        )
        safety_override = measured >= safety_threshold
        width_rate = torch.where(
            safety_override,
            torch.full_like(width_rate, config.safety_open_rate_m_s),
            width_rate,
        )
        candidate_width = torch.clamp(
            self._command_width + width_rate * float(dt_s),
            config.minimum_width_m,
            config.maximum_width_m,
        )
        self._command_width = torch.where(
            next_active | safety_override, candidate_width, policy_width
        )
        self._command_width = torch.where(
            just_released, policy_width, self._command_width
        )
        self._previous_force = self._filtered_force.detach().clone()
        self._active = next_active
        self._active_age = torch.where(
            next_active,
            torch.where(
                just_activated,
                torch.zeros_like(self._active_age),
                self._active_age + 1,
            ),
            torch.zeros_like(self._active_age),
        )

        executed = policy_action.clone()
        executed[:, config.gripper_width_index] = torch.where(
            self._active | safety_override, self._command_width, policy_width
        )
        self._last_executed_width = executed[
            :, config.gripper_width_index
        ].detach().clone()
        diagnostics = ForceControlDiagnostics(
            active=self._active.detach().clone(),
            filtered_force_n=self._filtered_force.detach().clone(),
            target_force_n=self.target_range.center_n,
            error_n=error.detach().clone(),
            command_width_m=self._command_width.detach().clone(),
            safety_override=safety_override.detach().clone(),
        )
        return executed, diagnostics


__all__ = [
    "ForceControlDiagnostics",
    "ForceControllerConfig",
    "ForceRange",
    "ForceRangeAdapterConfig",
    "GrapeForceController",
    "OnlineForceRangeAdapter",
]
