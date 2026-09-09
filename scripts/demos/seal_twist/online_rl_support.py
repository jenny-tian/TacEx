"""Pure-Torch contracts for SealTwist online force-conditioned RL.

This module deliberately has no Isaac Lab dependency so the force/state/reward
contracts can be unit tested without launching the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


FLOW_OBSERVATION_DIM = 28
ACTOR_FORCE_FEATURE_DIM = 6
ACTOR_OBSERVATION_DIM = FLOW_OBSERVATION_DIM + ACTOR_FORCE_FEATURE_DIM
CRITIC_FORCE_CONTEXT_DIM = 5


def exploration_scale_for_episode(
    *,
    episode_index: int,
    total_episodes: int,
    exploration_start: float,
    exploration_end: float,
    resumed: bool,
    anneal_episodes: int | None = None,
) -> float:
    """Return exploration without restarting annealing during continuation.

    ``total_episodes`` is the new cumulative endpoint when a run is resumed.
    Recomputing progress against that larger value would incorrectly raise the
    exploration scale after the original run had already annealed to its end.
    """

    if total_episodes < 1:
        raise ValueError("total_episodes must be positive")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    if anneal_episodes is not None and anneal_episodes < 1:
        raise ValueError("anneal_episodes must be positive when provided")
    if resumed:
        return float(exploration_end)
    schedule_episodes = total_episodes if anneal_episodes is None else anneal_episodes
    progress = min(episode_index / max(schedule_episodes - 1, 1), 1.0)
    if progress >= 1.0:
        return float(exploration_end)
    return float(exploration_start + progress * (exploration_end - exploration_start))


def force_balance_action_prior(
    reference: torch.Tensor,
    *,
    action_dim: int,
    actor_coordinate: int,
    target_value: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an episode-level tactile balancing prior for one action.

    Instantaneous fingertip imbalance changes sign during contact transients,
    whereas the terminal imbalance is the quantity used by the success rule.
    The caller therefore supplies a slowly updated terminal statistic rather
    than deriving a target independently at every transition.
    """

    if reference.ndim != 2:
        raise ValueError("reference must be a matrix")
    if action_dim < 1 or not 0 <= actor_coordinate < action_dim:
        raise ValueError("actor_coordinate is outside the residual action")
    if limit < 0.0:
        raise ValueError("force-balance prior limit must be non-negative")
    prior = reference.new_zeros((len(reference), action_dim))
    mask = torch.zeros_like(prior)
    target = max(
        -float(limit),
        min(float(limit), float(target_value)),
    )
    prior[:, actor_coordinate] = target
    mask[:, actor_coordinate] = 1.0
    return prior, mask


def force_conditioned_observation(
    normalized_flow_observation: torch.Tensor,
    *,
    force_low_n: float,
    force_high_n: float,
    left_force_n: torch.Tensor,
    right_force_n: torch.Tensor,
) -> torch.Tensor:
    """Append the VLM target and separate fingertip errors to the actor state.

    The old implementation appended only the error of the *mean* fingertip
    force.  A 4 N / 10 N grasp therefore looked identical to a safe 7 N / 7 N
    grasp even though the former fails the per-finger task constraint.
    """

    if normalized_flow_observation.ndim != 2:
        raise ValueError("normalized_flow_observation must be a matrix")
    if normalized_flow_observation.shape[-1] != FLOW_OBSERVATION_DIM:
        raise ValueError(
            f"expected {FLOW_OBSERVATION_DIM} Flow features, got "
            f"{normalized_flow_observation.shape[-1]}"
        )
    if not force_low_n < force_high_n:
        raise ValueError("force target interval must be strictly ordered")
    batch = len(normalized_flow_observation)
    left = left_force_n.to(normalized_flow_observation).reshape(batch, 1)
    right = right_force_n.to(normalized_flow_observation).reshape(batch, 1)
    low = torch.full_like(left, float(force_low_n) / 10.0)
    high = torch.full_like(left, float(force_high_n) / 10.0)
    center = torch.full_like(left, 0.5 * (float(force_low_n) + float(force_high_n)) / 10.0)
    left_error = center - left / 10.0
    right_error = center - right / 10.0
    imbalance = (left - right) / 10.0
    return torch.cat(
        (
            normalized_flow_observation,
            low,
            high,
            center,
            left_error,
            right_error,
            imbalance,
        ),
        dim=-1,
    )


def critic_force_context(
    *,
    reference: torch.Tensor,
    force_low_n: float,
    force_high_n: float,
    hidden_minimum_n: torch.Tensor,
    hidden_maximum_n: torch.Tensor,
    width_correction_m: torch.Tensor,
    maximum_width_correction_m: float,
) -> torch.Tensor:
    """Build the critic-only context for the hybrid VLM/force action.

    The VLM interval and the force-servo correction are part of the action that
    reached the environment.  Randomized task thresholds are privileged critic
    state: exposing them only to Q removes reward aliasing without leaking them
    to the deployment actor.
    """

    if maximum_width_correction_m <= 0.0:
        raise ValueError("maximum_width_correction_m must be positive")
    batch = len(reference)
    dtype = reference.dtype
    device = reference.device
    low = torch.full((batch, 1), float(force_low_n) / 10.0, dtype=dtype, device=device)
    high = torch.full_like(low, float(force_high_n) / 10.0)
    hidden_low = hidden_minimum_n.to(reference).reshape(batch, 1) / 10.0
    hidden_high = hidden_maximum_n.to(reference).reshape(batch, 1) / 10.0
    width = (
        width_correction_m.to(reference).reshape(batch, 1)
        / float(maximum_width_correction_m)
    ).clamp(-1.0, 1.0)
    return torch.cat((low, high, hidden_low, hidden_high, width), dim=-1)


def force_evidence(
    *,
    left_force_n: float,
    right_force_n: float,
    force_low_n: float,
    force_high_n: float,
) -> str:
    """Classify observed terminal force relative to the requested interval."""

    weaker = min(float(left_force_n), float(right_force_n))
    stronger = max(float(left_force_n), float(right_force_n))
    if weaker >= force_low_n and stronger <= force_high_n:
        return "balanced_in_target"
    if weaker < force_low_n and stronger > force_high_n:
        return "bilateral_imbalance"
    if stronger < force_low_n:
        return "uniform_under_target"
    if weaker > force_high_n:
        return "uniform_over_target"
    return "mixed_near_target"


@dataclass(frozen=True)
class GuardedForceProposal:
    force_range_n: tuple[float, float]
    evidence: str
    decision: str


def guard_vlm_force_proposal(
    *,
    current_range_n: tuple[float, float],
    proposed_range_n: tuple[float, float],
    physical_bounds_n: tuple[float, float],
    terminal_left_force_n: float,
    terminal_right_force_n: float,
    outcome: str,
    maximum_center_step_n: float,
    threshold_schedule: str = "iid",
    force_constraint_violation: str = "none",
    threshold_persists_next_episode: bool | None = None,
) -> GuardedForceProposal:
    """Accept a VLM set-point change only when temporal evidence supports it.

    Under an IID threshold schedule, a completed episode cannot reveal the
    independent limits of the next episode, so only symmetric tracking evidence
    can move the set-point.  Under a persistent/block schedule, a directional
    force-constraint outcome is informative about the next episode in the same
    material batch and may authorize a conservative move even when the measured
    forces were inside or straddled the requested target interval.
    """

    current_low, current_high = map(float, current_range_n)
    proposed_low, proposed_high = map(float, proposed_range_n)
    physical_low, physical_high = map(float, physical_bounds_n)
    if not physical_low <= current_low < current_high <= physical_high:
        raise ValueError("current force range violates physical bounds")
    if not proposed_low < proposed_high:
        raise ValueError("proposed force range must be strictly ordered")
    if maximum_center_step_n < 0.0:
        raise ValueError("maximum_center_step_n must be non-negative")
    schedule = str(threshold_schedule).lower()
    if schedule not in {"iid", "persistent", "block"}:
        raise ValueError("threshold_schedule must be one of: iid, persistent, block")
    violation = str(force_constraint_violation).lower()
    if violation not in {"none", "under", "over"}:
        raise ValueError("force_constraint_violation must be one of: none, under, over")
    if threshold_persists_next_episode is None:
        threshold_persists_next_episode = schedule == "persistent"

    evidence = force_evidence(
        left_force_n=terminal_left_force_n,
        right_force_n=terminal_right_force_n,
        force_low_n=current_low,
        force_high_n=current_high,
    )
    if outcome == "success":
        return GuardedForceProposal((current_low, current_high), evidence, "hold_after_success")

    current_center = 0.5 * (current_low + current_high)
    proposed_center = 0.5 * (proposed_low + proposed_high)
    requested_delta = proposed_center - current_center
    symmetric_direction_is_supported = (
        evidence == "uniform_under_target" and requested_delta > 0.0
    ) or (
        evidence == "uniform_over_target" and requested_delta < 0.0
    )
    persistent_direction_is_supported = (
        schedule in {"persistent", "block"}
        and bool(threshold_persists_next_episode)
        and (
            (violation == "under" and requested_delta > 0.0)
            or (violation == "over" and requested_delta < 0.0)
        )
    )
    direction_is_supported = (
        symmetric_direction_is_supported or persistent_direction_is_supported
    )
    if not direction_is_supported:
        reason = (
            "hold_bilateral_imbalance"
            if evidence == "bilateral_imbalance"
            else "hold_without_symmetric_force_evidence"
        )
        return GuardedForceProposal((current_low, current_high), evidence, reason)

    limited_delta = max(-maximum_center_step_n, min(maximum_center_step_n, requested_delta))
    width = current_high - current_low
    center = current_center + limited_delta
    center = max(physical_low + 0.5 * width, min(physical_high - 0.5 * width, center))
    guarded = (center - 0.5 * width, center + 0.5 * width)
    decision = (
        "accept_symmetric_force_evidence"
        if symmetric_direction_is_supported
        else f"accept_{schedule}_{violation}_constraint_evidence"
    )
    return GuardedForceProposal(guarded, evidence, decision)


def force_shaping_term(
    *,
    left_force_n: torch.Tensor,
    right_force_n: torch.Tensor,
    force_low_n: float,
    force_high_n: float,
    bilateral_contact: torch.Tensor,
) -> torch.Tensor:
    """Dense force reward aligned with the task's two-finger success rule."""

    left = left_force_n.float()
    right = right_force_n.float()
    bilateral = bilateral_contact.float()
    low = float(force_low_n)
    high = float(force_high_n)
    scale = max(0.5 * (high - low), 0.5)
    below = torch.relu(low - torch.minimum(left, right)) / scale
    above = torch.relu(torch.maximum(left, right) - high) / scale
    imbalance = torch.abs(left - right) / max(high - low, 1.0)
    both_in_band = ((left >= low) & (left <= high) & (right >= low) & (right <= high)).float()
    return bilateral * (
        0.008 * both_in_band
        - 0.003 * below.square().clamp(max=9.0)
        - 0.006 * above.square().clamp(max=9.0)
        - 0.0015 * imbalance.square().clamp(max=9.0)
    )


__all__ = [
    "ACTOR_OBSERVATION_DIM",
    "CRITIC_FORCE_CONTEXT_DIM",
    "GuardedForceProposal",
    "critic_force_context",
    "force_balance_action_prior",
    "force_conditioned_observation",
    "force_evidence",
    "force_shaping_term",
    "guard_vlm_force_proposal",
]
