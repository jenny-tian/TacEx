"""High-rate tactile force feedback that masks only selected action dimensions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .contracts import ForceRange


@dataclass(frozen=True)
class ForceControllerConfig:
    action_dim: int = 10
    gripper_width_index: int = 9
    minimum_width_m: float = 0.0
    maximum_width_m: float = 0.04
    kp_width_rate_per_n: float = 0.006
    ki_width_rate_per_n_s: float = 0.018
    kd_width_per_n: float = 0.00008
    maximum_width_rate_m_s: float = 0.018
    integral_limit_n_s: float = 1.5
    force_filter_alpha: float = 0.35
    contact_on_force_n: float = 0.01
    contact_off_force_n: float = 0.005
    release_hysteresis_steps: int = 6
    contact_settle_steps: int = 2
    require_contact_mask_for_activation: bool = False
    hard_force_limit_n: float = 3.5
    safety_open_rate_m_s: float = 0.12

    def __post_init__(self) -> None:
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive.")
        if not 0 <= self.gripper_width_index < self.action_dim:
            raise ValueError("gripper_width_index is outside the action vector.")
        if self.maximum_width_m <= self.minimum_width_m:
            raise ValueError("maximum_width_m must exceed minimum_width_m.")
        if self.maximum_width_rate_m_s <= 0.0 or self.safety_open_rate_m_s <= 0.0:
            raise ValueError("Width rates must be positive.")
        if not 0.0 < self.force_filter_alpha <= 1.0:
            raise ValueError("force_filter_alpha must be in (0, 1].")
        if self.release_hysteresis_steps < 0:
            raise ValueError("release_hysteresis_steps must be non-negative.")
        if self.contact_settle_steps < 0:
            raise ValueError("contact_settle_steps must be non-negative.")


@dataclass(frozen=True)
class ForceControlDiagnostics:
    active: torch.Tensor
    measured_force_n: torch.Tensor
    filtered_force_n: torch.Tensor
    target_force_n: float
    force_error_n: torch.Tensor
    width_command_m: torch.Tensor
    safety_override: torch.Tensor


class TactileForceController:
    """PI-D gripper-width controller executed at the physics/sensor rate.

    The returned action is cloned from the policy action and only the configured
    gripper-width column is replaced while contact control is active.
    """

    def __init__(
        self, target_range_n: ForceRange, config: ForceControllerConfig | None = None
    ) -> None:
        self.config = ForceControllerConfig() if config is None else config
        self.set_target_range(target_range_n)
        self._command_width_m: torch.Tensor | None = None
        self._filtered_force_n: torch.Tensor | None = None
        self._previous_force_n: torch.Tensor | None = None
        self._integral_error_n_s: torch.Tensor | None = None
        self._active: torch.Tensor | None = None
        self._dropout_steps: torch.Tensor | None = None
        self._active_age_steps: torch.Tensor | None = None

    @property
    def target_force_n(self) -> float:
        return self.target_range_n.center_n

    def set_target_range(self, target_range_n: ForceRange) -> None:
        if target_range_n.maximum_n >= self.config.hard_force_limit_n:
            raise ValueError(
                "Target range must stay strictly below hard_force_limit_n."
            )
        self.target_range_n = target_range_n

    def reset(self) -> None:
        self._command_width_m = None
        self._filtered_force_n = None
        self._previous_force_n = None
        self._integral_error_n_s = None
        self._active = None
        self._dropout_steps = None
        self._active_age_steps = None

    def _initialize(
        self, policy_action: torch.Tensor, measured_force_n: torch.Tensor
    ) -> None:
        batch = policy_action.shape[0]
        self._command_width_m = (
            policy_action[:, self.config.gripper_width_index].detach().clone()
        )
        self._filtered_force_n = measured_force_n.detach().clone()
        self._previous_force_n = measured_force_n.detach().clone()
        self._integral_error_n_s = torch.zeros(
            batch, device=policy_action.device, dtype=policy_action.dtype
        )
        self._active = torch.zeros(batch, device=policy_action.device, dtype=torch.bool)
        self._dropout_steps = torch.zeros(
            batch, device=policy_action.device, dtype=torch.long
        )
        self._active_age_steps = torch.zeros(
            batch, device=policy_action.device, dtype=torch.long
        )

    def control(
        self,
        policy_action: torch.Tensor,
        *,
        contact_force_n: torch.Tensor,
        contact_mask: torch.Tensor,
        dt_s: float,
        safety_force_n: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ForceControlDiagnostics]:
        if policy_action.ndim != 2 or policy_action.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"policy_action must have shape [B, {self.config.action_dim}]."
            )
        if not policy_action.is_floating_point() or not bool(
            torch.isfinite(policy_action).all()
        ):
            raise ValueError("policy_action must be a finite floating tensor.")
        batch = policy_action.shape[0]
        measured = torch.as_tensor(
            contact_force_n, device=policy_action.device, dtype=policy_action.dtype
        ).reshape(-1)
        requested_contact = torch.as_tensor(
            contact_mask, device=policy_action.device, dtype=torch.bool
        ).reshape(-1)
        if measured.shape[0] != batch or requested_contact.shape[0] != batch:
            raise ValueError("Force and contact batches must match policy_action.")
        if not bool(torch.isfinite(measured).all()) or bool((measured < 0.0).any()):
            raise ValueError("contact_force_n must contain finite non-negative values.")
        if safety_force_n is None:
            safety_force = measured
        else:
            safety_force = torch.as_tensor(
                safety_force_n, device=policy_action.device, dtype=policy_action.dtype
            ).reshape(-1)
            if safety_force.shape[0] != batch:
                raise ValueError("safety_force_n batch must match policy_action.")
            if not bool(torch.isfinite(safety_force).all()) or bool(
                (safety_force < 0.0).any()
            ):
                raise ValueError(
                    "safety_force_n must contain finite non-negative values."
                )
        if not math.isfinite(float(dt_s)) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive.")
        if self._command_width_m is None or self._command_width_m.shape[0] != batch:
            self._initialize(policy_action, measured)

        assert self._command_width_m is not None
        assert self._filtered_force_n is not None
        assert self._previous_force_n is not None
        assert self._integral_error_n_s is not None
        assert self._active is not None
        assert self._dropout_steps is not None
        assert self._active_age_steps is not None

        sensor_contact = measured >= self.config.contact_on_force_n
        activate = (
            requested_contact
            if self.config.require_contact_mask_for_activation
            else requested_contact | sensor_contact
        )
        still_loaded = measured >= self.config.contact_off_force_n
        self._dropout_steps = torch.where(
            self._active & ~activate & ~still_loaded,
            self._dropout_steps + 1,
            torch.zeros_like(self._dropout_steps),
        )
        keep_active = self._active & (
            still_loaded | (self._dropout_steps <= self.config.release_hysteresis_steps)
        )
        next_active = activate | keep_active
        just_activated = next_active & ~self._active
        just_released = ~next_active & self._active

        policy_width = policy_action[:, self.config.gripper_width_index]
        self._command_width_m = torch.where(
            ~self._active, policy_width, self._command_width_m
        )
        self._filtered_force_n = torch.where(
            just_activated,
            measured,
            self.config.force_filter_alpha * measured
            + (1.0 - self.config.force_filter_alpha) * self._filtered_force_n,
        )
        derivative_n_s = (self._filtered_force_n - self._previous_force_n) / float(dt_s)
        derivative_n_s = torch.where(
            just_activated, torch.zeros_like(derivative_n_s), derivative_n_s
        )
        error_n = self.target_force_n - self._filtered_force_n
        candidate_integral = torch.clamp(
            self._integral_error_n_s + error_n * float(dt_s),
            -self.config.integral_limit_n_s,
            self.config.integral_limit_n_s,
        )
        self._integral_error_n_s = torch.where(
            next_active, candidate_integral, torch.zeros_like(candidate_integral)
        )

        width_rate = (
            -self.config.kp_width_rate_per_n * error_n
            - self.config.ki_width_rate_per_n_s * self._integral_error_n_s
            + self.config.kd_width_per_n * derivative_n_s
        )
        width_rate = width_rate.clamp(
            -self.config.maximum_width_rate_m_s,
            self.config.maximum_width_rate_m_s,
        )
        settling = next_active & (
            self._active_age_steps < self.config.contact_settle_steps
        )
        unloaded = measured < self.config.contact_off_force_n
        # Never ratchet the fingers closed on a noisy tactile-only event. On
        # first contact, hold briefly so the force sample catches up with the
        # position command before applying integral action.
        inhibit_closing = settling | unloaded
        width_rate = torch.where(
            inhibit_closing, torch.clamp(width_rate, min=0.0), width_rate
        )
        # The learned range is itself a safety contract: react at its upper
        # edge instead of waiting until the glass break threshold is crossed.
        safety_threshold_n = min(
            self.config.hard_force_limit_n, self.target_range_n.maximum_n
        )
        safety_override = safety_force >= safety_threshold_n
        width_rate = torch.where(
            safety_override,
            torch.full_like(width_rate, self.config.safety_open_rate_m_s),
            width_rate,
        )
        candidate_width = torch.clamp(
            self._command_width_m + width_rate * float(dt_s),
            self.config.minimum_width_m,
            self.config.maximum_width_m,
        )
        self._command_width_m = torch.where(
            next_active | safety_override, candidate_width, policy_width
        )
        self._command_width_m = torch.where(
            just_released, policy_width, self._command_width_m
        )
        self._previous_force_n = self._filtered_force_n.detach().clone()
        self._active = next_active
        self._active_age_steps = torch.where(
            next_active,
            torch.where(
                just_activated,
                torch.zeros_like(self._active_age_steps),
                self._active_age_steps + 1,
            ),
            torch.zeros_like(self._active_age_steps),
        )

        action = policy_action.clone()
        action[:, self.config.gripper_width_index] = torch.where(
            self._active | safety_override,
            self._command_width_m,
            policy_width,
        )
        diagnostics = ForceControlDiagnostics(
            active=self._active.detach().clone(),
            measured_force_n=measured.detach().clone(),
            filtered_force_n=self._filtered_force_n.detach().clone(),
            target_force_n=self.target_force_n,
            force_error_n=error_n.detach().clone(),
            width_command_m=self._command_width_m.detach().clone(),
            safety_override=safety_override.detach().clone(),
        )
        return action, diagnostics


__all__ = ["ForceControlDiagnostics", "ForceControllerConfig", "TactileForceController"]
