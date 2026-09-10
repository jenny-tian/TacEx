"""Pure-Python/Torch contracts for repaired RaspberryPick online RL.

This module deliberately has no Isaac Lab dependency.  The proposal guard,
environment-grounded reward and semantic residual can therefore be audited in
unit tests without starting the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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


def should_call_vlm(
    *, force_range_source: str, call_policy: str, outcome: str
) -> bool:
    """Return whether an episode should trigger an actual VLM request."""

    if call_policy not in {"every_episode", "failure_only"}:
        raise ValueError(f"unsupported VLM call policy: {call_policy}")
    if force_range_source != "vlm":
        return False
    return call_policy == "every_episode" or outcome != "success"


def validate_agent_checkpoint_contract(
    checkpoint: Mapping[str, object],
    *,
    target_method: str,
    target_contract: str,
    dsrl_contract: str,
    base_checkpoint_sha256: str,
    expected_observation_dim: int,
    expected_residual_mode: str,
    evaluation_mode: bool,
    allow_dsrl_for_controller_only_oracle: bool,
    allow_dsrl_for_oracle_training_branch: bool = False,
) -> str:
    """Validate native loading or an explicit shared-actor oracle ablation.

    A controller-only oracle evaluation may reuse a trained ``dsrl_only`` actor
    only when evaluation is deterministic and update-free.  The exception is
    deliberately narrow: it cannot initialize another training run, cannot
    load a frozen-zero residual checkpoint, and must preserve the actor input
    contract and base policy.  The return value is recorded in the run summary.
    """

    if checkpoint.get("base_checkpoint_sha256") != str(base_checkpoint_sha256):
        raise ValueError("residual checkpoint was trained with a different base policy")
    if int(checkpoint.get("actor_observation_dim", -1)) != int(expected_observation_dim):
        raise ValueError("residual checkpoint actor observation dimension does not match")
    if checkpoint.get("residual_mode") != str(expected_residual_mode):
        raise ValueError("residual checkpoint mode does not match")

    source_method = checkpoint.get("method")
    source_contract = checkpoint.get("contract")
    if source_method == target_method and source_contract == target_contract:
        return "native_method_checkpoint"

    shared_actor = (
        bool(allow_dsrl_for_controller_only_oracle)
        and bool(evaluation_mode)
        and target_method == "oracle_force_dsrl"
        and source_method == "dsrl_only"
        and source_contract == dsrl_contract
        and not bool(checkpoint.get("dsrl_frozen"))
    )
    if shared_actor:
        return "shared_dsrl_actor_controller_only_oracle_evaluation"

    shared_training_history = (
        bool(allow_dsrl_for_oracle_training_branch)
        and not bool(evaluation_mode)
        and target_method == "oracle_force_dsrl"
        and source_method == "dsrl_only"
        and source_contract == dsrl_contract
        and not bool(checkpoint.get("dsrl_frozen"))
    )
    if shared_training_history:
        return "shared_dsrl_training_history_oracle_branch"

    raise ValueError("residual checkpoint contract/method does not match this run")


def validate_training_branch_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    resume_episode: int,
    training_seed: int,
) -> int:
    """Validate the episode/replay state needed for an online-training branch.

    The returned integer is the cumulative transition count at the branch
    point.  Network and optimizer compatibility remains the responsibility of
    :func:`validate_agent_checkpoint_contract` and the agent loader.
    """

    resume_episode = int(resume_episode)
    if resume_episode < 1:
        raise ValueError("training branch resume episode must be positive")
    if int(checkpoint.get("completed_episodes", -1)) != resume_episode:
        raise ValueError("checkpoint completed-episode count does not match resume episode")
    if int(checkpoint.get("training_seed", -1)) != int(training_seed):
        raise ValueError("checkpoint training seed does not match branch seed")

    decisions = checkpoint.get("force_range_decisions")
    if not isinstance(decisions, (list, tuple)) or len(decisions) != resume_episode:
        raise ValueError("checkpoint force-decision history does not match resume episode")
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping):
            raise ValueError("checkpoint force-decision history is malformed")
        if int(decision.get("completed_episode_index", -1)) != index:
            raise ValueError("checkpoint force-decision history is non-contiguous")
        if int(decision.get("applies_to_episode_index", -1)) != index + 1:
            raise ValueError("checkpoint force-decision application index is invalid")

    current_range = checkpoint.get("current_force_range_n")
    if not isinstance(current_range, (list, tuple)):
        raise ValueError("checkpoint current force range is missing")
    _validate_range("checkpoint current_force_range_n", current_range)

    replay_state = checkpoint.get("replay_state")
    if not isinstance(replay_state, Mapping):
        raise ValueError("training branch checkpoint is missing replay state")
    replay_size = int(replay_state.get("size", -1))
    replay_capacity = int(replay_state.get("capacity", -1))
    replay_position = int(replay_state.get("position", -1))
    if replay_capacity < 1 or not 0 <= replay_size <= replay_capacity:
        raise ValueError("checkpoint replay size/capacity is invalid")
    if not 0 <= replay_position < replay_capacity:
        raise ValueError("checkpoint replay position is invalid")
    replay_transitions = int(checkpoint.get("replay_transitions", -1))
    if replay_transitions < replay_size:
        raise ValueError("checkpoint replay transition count is inconsistent")
    return replay_transitions


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
            outcome in {"no_detach", "no_detach_with_contact"}
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


@dataclass(frozen=True)
class BottleStyleLateralResidualLayout:
    """One bounded residual along the gripper closing axis.

    This mirrors the successful SealTwist DSRL design: the learned coordinate
    has direct millimetre-scale physical authority instead of being routed
    through a weak Flow-latent channel.
    """

    maximum_offset_m: float = 0.006

    def __post_init__(self) -> None:
        if self.maximum_offset_m <= 0.0:
            raise ValueError("maximum_offset_m must be positive")

    @property
    def correction_dim(self) -> int:
        return 1

    def apply(
        self,
        action_chunk: torch.Tensor,
        correction: torch.Tensor,
        closing_axis_b: torch.Tensor,
    ) -> torch.Tensor:
        if action_chunk.ndim != 3 or action_chunk.shape[-1] != 10:
            raise ValueError("action_chunk must have shape [B,H,10]")
        if correction.ndim != 2 or correction.shape != (len(action_chunk), 1):
            raise ValueError("correction must have shape [B,1]")
        axis = torch.as_tensor(
            closing_axis_b, dtype=action_chunk.dtype, device=action_chunk.device
        )
        if axis.shape != (len(action_chunk), 3):
            raise ValueError("closing_axis_b must have shape [B,3]")
        norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        if bool((norm <= 1.0e-8).any()):
            raise ValueError("closing_axis_b must be non-zero")
        axis = axis / norm
        offset = correction.clamp(-1.0, 1.0)[:, 0] * float(self.maximum_offset_m)
        result = action_chunk.clone()
        result[:, :, :3] += offset[:, None, None] * axis[:, None, :]
        return result


def force_balance_prior_target(
    *,
    left_force_n: float,
    right_force_n: float,
    force_scale_n: float,
    gain: float,
    limit: float,
) -> float:
    """Return the bounded closing-axis prior used by bottle-style DSRL."""

    if force_scale_n <= 0.0:
        raise ValueError("force_scale_n must be positive")
    if min(gain, limit) < 0.0:
        raise ValueError("gain and limit must be non-negative")
    normalized_imbalance = (float(left_force_n) - float(right_force_n)) / float(
        force_scale_n
    )
    return max(-float(limit), min(float(limit), -float(gain) * normalized_imbalance))


def exploration_scale_for_episode(
    *,
    episode_index: int,
    anneal_episodes: int,
    exploration_start: float,
    exploration_end: float,
) -> float:
    """Linearly anneal exploration, then hold the final scale."""

    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    if anneal_episodes < 1:
        raise ValueError("anneal_episodes must be positive")
    if min(exploration_start, exploration_end) < 0.0:
        raise ValueError("exploration scales must be non-negative")
    progress = min(episode_index / max(anneal_episodes - 1, 1), 1.0)
    if progress >= 1.0:
        return float(exploration_end)
    return float(exploration_start + progress * (exploration_end - exploration_start))


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
