"""Pure-Python/Torch contracts for repaired RaspberryPick online RL.

This module deliberately has no Isaac Lab dependency.  The proposal guard,
environment-grounded reward and semantic residual can therefore be audited in
unit tests without starting the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class RaspberryProfile:
    """One persistent, physically coherent berry/pedicle profile."""

    name: str
    detach_threshold_n: float
    damage_threshold_n: float
    stem_stiffness_n_per_m: float
    stem_damping_n_s_per_m: float


@dataclass(frozen=True)
class OracleForceRange:
    """Force interval computed from simulator-only profile ground truth."""

    minimum_n: float
    maximum_n: float
    source_damage_threshold_n: float
    damage_margin_n: float
    interval_width_n: float

    def as_tuple(self) -> tuple[float, float]:
        return self.minimum_n, self.maximum_n


# ``fixed`` reproduces the diagnosed experiment.  The other profiles are
# candidates only: they must pass force-set-point calibration before use in a
# formal comparison.
RASPBERRY_PROFILES: dict[str, RaspberryProfile] = {
    "fixed": RaspberryProfile("fixed", 0.80, 3.20, 95.0, 0.24),
    "delicate": RaspberryProfile("delicate", 0.68, 2.60, 82.0, 0.21),
    "firm": RaspberryProfile("firm", 0.92, 3.10, 112.0, 0.28),
}


def theoretical_oracle_force_range(
    profile: RaspberryProfile,
    *,
    interval_width_n: float = 0.60,
    damage_margin_n: float = 0.20,
) -> OracleForceRange:
    """Return a safe grip interval using privileged simulator parameters.

    The upper edge stays ``damage_margin_n`` below the true damage threshold;
    the requested controller bandwidth then defines the lower edge.  This is
    an oracle ablation, not a deployable observation: neither the residual
    actor nor a VLM infers these profile values.
    """

    width = float(interval_width_n)
    margin = float(damage_margin_n)
    if width <= 0.0:
        raise ValueError("interval_width_n must be positive")
    if margin <= 0.0:
        raise ValueError("damage_margin_n must be positive")
    maximum = float(profile.damage_threshold_n) - margin
    minimum = maximum - width
    if minimum <= 0.0:
        raise ValueError("oracle force interval must remain positive")
    if maximum <= float(profile.detach_threshold_n):
        raise ValueError("oracle force interval does not clear the detach threshold")
    return OracleForceRange(
        minimum_n=minimum,
        maximum_n=maximum,
        source_damage_threshold_n=float(profile.damage_threshold_n),
        damage_margin_n=margin,
        interval_width_n=width,
    )


@dataclass(frozen=True)
class GuardedForceProposal:
    force_range_n: tuple[float, float]
    evidence: str
    decision: str
    candidate_direction: str
    confirmation_streak: int


def _validate_range(name: str, force_range_n: Sequence[float]) -> tuple[float, float]:
    if len(force_range_n) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    low, high = map(float, force_range_n)
    if not low < high:
        raise ValueError(f"{name} must be strictly ordered")
    return low, high


def guard_force_proposal(
    *,
    current_range_n: Sequence[float],
    proposed_range_n: Sequence[float],
    physical_bounds_n: Sequence[float],
    outcome: str,
    bilateral_contact_fraction: float,
    mean_left_force_n: float,
    mean_right_force_n: float,
    peak_grip_force_n: float,
    damage_threshold_n: float,
    damage_risk_fraction: float,
    recent_candidate_directions: Sequence[str] = (),
    confirmations_required: int = 2,
    maximum_center_step_n: float = 0.05,
    minimum_bilateral_fraction: float = 0.10,
    imbalance_fraction: float = 0.28,
) -> GuardedForceProposal:
    """Guard a force proposal using identifiable, directional evidence.

    A proposal is applied only after consecutive, directionally consistent
    episodes.  Success, missing contact, geometry failures and bilateral force
    imbalance always hold.  Low pedicle tension is intentionally absent from
    the upward-evidence rule: it cannot by itself identify low grip force.
    """

    current_low, current_high = _validate_range("current_range_n", current_range_n)
    proposed_low, proposed_high = _validate_range("proposed_range_n", proposed_range_n)
    physical_low, physical_high = _validate_range("physical_bounds_n", physical_bounds_n)
    if not physical_low <= current_low < current_high <= physical_high:
        raise ValueError("current_range_n violates physical_bounds_n")
    if confirmations_required < 1:
        raise ValueError("confirmations_required must be positive")
    if maximum_center_step_n < 0.0:
        raise ValueError("maximum_center_step_n must be non-negative")

    current = (current_low, current_high)
    current_center = 0.5 * (current_low + current_high)
    proposed_center = 0.5 * (proposed_low + proposed_high)
    requested_delta = proposed_center - current_center
    requested_direction = "up" if requested_delta > 1.0e-8 else "down" if requested_delta < -1.0e-8 else "hold"

    weaker = min(float(mean_left_force_n), float(mean_right_force_n))
    stronger = max(float(mean_left_force_n), float(mean_right_force_n))
    mean_force = 0.5 * (weaker + stronger)
    imbalance = (stronger - weaker) / max(mean_force, 0.20)

    evidence = "no_directional_force_evidence"
    candidate_direction = "hold"
    hard_hold: str | None = None
    if outcome == "success":
        hard_hold = "hold_after_success"
        evidence = "successful_safe_execution"
    elif outcome in {"workspace_failure", "obstacle_collision", "slipped"}:
        hard_hold = "hold_geometry_or_trajectory_failure"
        evidence = outcome
    elif (
        outcome != "crushed"
        and float(bilateral_contact_fraction) < float(minimum_bilateral_fraction)
    ):
        hard_hold = "hold_no_contact"
        evidence = "insufficient_bilateral_contact"
    elif imbalance > float(imbalance_fraction):
        hard_hold = "hold_alignment_or_imbalance"
        evidence = "bilateral_force_imbalance"
    else:
        over_force = (
            outcome == "crushed"
            or float(damage_risk_fraction) >= 0.05
            or weaker > current_high + 0.03
        )
        # Upward evidence requires both fingers to remain below the target and
        # the peak load never to enter the band.  A low tension/no-detach result
        # with force already in band therefore cannot authorize an increase.
        under_force = (
            outcome == "no_detach"
            and stronger < current_low - 0.03
            and float(peak_grip_force_n) < current_low + 0.02
        )
        if over_force:
            candidate_direction = "down"
            evidence = "consistent_over_force_or_damage_risk"
        elif under_force:
            candidate_direction = "up"
            evidence = "consistent_bilateral_under_force"

    matching_history = 0
    if candidate_direction != "hold":
        for direction in reversed(tuple(recent_candidate_directions)):
            if direction != candidate_direction:
                break
            matching_history += 1
    streak = matching_history + int(candidate_direction != "hold")

    if hard_hold is not None:
        return GuardedForceProposal(current, evidence, hard_hold, "hold", 0)
    if requested_direction == "hold":
        return GuardedForceProposal(current, evidence, "hold_no_requested_change", candidate_direction, streak)
    if candidate_direction == "hold" or requested_direction != candidate_direction:
        return GuardedForceProposal(
            current,
            evidence,
            "hold_without_directional_force_evidence",
            candidate_direction,
            streak,
        )
    if streak < confirmations_required:
        return GuardedForceProposal(
            current,
            evidence,
            "hold_pending_confirmation",
            candidate_direction,
            streak,
        )

    bounded_delta = max(
        -float(maximum_center_step_n),
        min(float(maximum_center_step_n), requested_delta),
    )
    width = current_high - current_low
    center = current_center + bounded_delta
    center = max(physical_low + 0.5 * width, min(physical_high - 0.5 * width, center))
    applied = (center - 0.5 * width, center + 0.5 * width)
    return GuardedForceProposal(
        applied,
        evidence,
        "accept_confirmed_directional_evidence",
        candidate_direction,
        streak,
    )


def environment_shaped_reward(
    *,
    left_force_n: float,
    right_force_n: float,
    bilateral_contact: bool,
    pedicle_tension_n: float,
    detach_threshold_n: float,
    damage_threshold_n: float,
    terminal_outcome: str | None = None,
) -> float:
    """Return task reward based only on physical constraints and outcome."""

    if detach_threshold_n <= 0.0 or damage_threshold_n <= 0.0:
        raise ValueError("physical thresholds must be positive")
    left = max(float(left_force_n), 0.0)
    right = max(float(right_force_n), 0.0)
    grip = max(left, right)
    reward = 0.002 * int(bool(bilateral_contact))
    reward += 0.004 * min(max(float(pedicle_tension_n), 0.0) / detach_threshold_n, 1.2)
    if bilateral_contact:
        imbalance = abs(left - right) / max(0.5 * (left + right), 0.20)
        reward -= 0.002 * min(imbalance * imbalance, 4.0)
    risk_start = 0.85 * damage_threshold_n
    if grip > risk_start:
        risk = (grip - risk_start) / max(damage_threshold_n - risk_start, 1.0e-6)
        reward -= 0.004 * min(risk * risk, 4.0)
    if terminal_outcome is not None:
        reward += terminal_outcome_reward(terminal_outcome)
    return reward


def terminal_outcome_reward(outcome: str) -> float:
    """Return the environment-grounded terminal reward for one outcome."""

    return {
        "success": 10.0,
        "crushed": -10.0,
        "slipped": -6.0,
        "obstacle_collision": -10.0,
        "workspace_failure": -6.0,
        "no_detach": -3.0,
        "timeout_after_detach": 1.0,
    }[outcome]


@dataclass(frozen=True)
class SemanticResidualLayout:
    """Bounded post-decode residual with explicit physical units."""

    translation_scale_m: tuple[float, float, float] = (0.006, 0.006, 0.008)
    gripper_scale_m: float = 0.0015

    @property
    def correction_dim(self) -> int:
        return 4

    def apply(self, action_chunk: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        if action_chunk.ndim != 3 or action_chunk.shape[-1] != 10:
            raise ValueError("action_chunk must have shape [B,H,10]")
        if correction.ndim != 2 or correction.shape != (len(action_chunk), self.correction_dim):
            raise ValueError(f"correction must have shape [B,{self.correction_dim}]")
        result = action_chunk.clone()
        scale = correction.new_tensor((*self.translation_scale_m, self.gripper_scale_m))
        physical = correction.clamp(-1.0, 1.0) * scale
        result[:, :, :3] += physical[:, None, :3]
        result[:, :, 9] += physical[:, None, 3]
        return result


@dataclass(frozen=True)
class SemanticXYResidualLayout:
    """Bounded pre-pull XY alignment residual in explicit physical units."""

    translation_scale_m: tuple[float, float] = (0.006, 0.006)

    @property
    def correction_dim(self) -> int:
        return 2

    def apply(self, action_chunk: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        if action_chunk.ndim != 3 or action_chunk.shape[-1] != 10:
            raise ValueError("action_chunk must have shape [B,H,10]")
        if correction.ndim != 2 or correction.shape != (len(action_chunk), self.correction_dim):
            raise ValueError(f"correction must have shape [B,{self.correction_dim}]")
        result = action_chunk.clone()
        scale = correction.new_tensor(self.translation_scale_m)
        result[:, :, :2] += correction.clamp(-1.0, 1.0)[:, None, :] * scale
        return result


def residual_is_active(
    *,
    physics_step: int,
    start_step: int,
    phase: float,
    pull_start_phase: float,
    stem_attached: bool,
    semantic_residual: bool,
) -> bool:
    """Gate residual control to the part of the task it can safely improve.

    Latent residuals retain the legacy behavior.  Semantic residuals are used
    only during the pre-pull alignment/grasp phase; after the pull starts, the
    frozen BC trajectory and force controller regain full authority.  This
    prevents an online learner from cancelling the calibrated pull motion.
    """

    if physics_step < start_step or not stem_attached:
        return False
    return not semantic_residual or float(phase) < float(pull_start_phase)
