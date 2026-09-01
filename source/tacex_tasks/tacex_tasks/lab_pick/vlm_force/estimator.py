"""Safety-constrained force-range estimator with diminishing episode updates."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import EpisodeFeedback, ForceDecision, ForceRange, VLMRecommendation


@dataclass(frozen=True)
class ForceEstimatorConfig:
    physical_range_n: ForceRange = ForceRange(0.25, 3.25)
    initial_range_n: ForceRange = ForceRange(1.0, 3.0)
    minimum_range_width_n: float = 0.30
    evidence_margin_n: float = 0.08
    success_margin_n: float = 0.18
    update_power: float = 0.65

    def __post_init__(self) -> None:
        if self.initial_range_n.minimum_n < self.physical_range_n.minimum_n:
            raise ValueError("initial_range_n starts below physical_range_n.")
        if self.initial_range_n.maximum_n > self.physical_range_n.maximum_n:
            raise ValueError("initial_range_n ends above physical_range_n.")
        if not 0.0 < self.minimum_range_width_n <= self.physical_range_n.width_n:
            raise ValueError("minimum_range_width_n is invalid.")
        if self.evidence_margin_n <= 0.0 or self.success_margin_n <= 0.0:
            raise ValueError("Estimator margins must be positive.")
        if not 0.5 < self.update_power <= 1.0:
            raise ValueError("update_power must be in (0.5, 1].")


class ConvergentForceEstimator:
    """Fuse VLM proposals with monotone empirical constraints.

    Force-caused failures monotonically tighten an admissible bracket.  The VLM
    proposal is projected into that bracket and blended with a Robbins--Monro
    gain ``n**(-update_power)``. Under consistent feedback and bounded proposal
    noise, the estimate therefore settles instead of oscillating indefinitely.
    """

    _LOW_FORCE_TOKENS = ("slip", "dropped", "insufficient_force", "too_low")
    _HIGH_FORCE_TOKENS = ("broken", "over_force", "excessive_force", "too_high")

    def __init__(self, config: ForceEstimatorConfig | None = None) -> None:
        self.config = ForceEstimatorConfig() if config is None else config
        self.current_range_n = self.config.initial_range_n
        self.evidence_lower_bound_n = self.config.physical_range_n.minimum_n
        self.evidence_upper_bound_n = self.config.physical_range_n.maximum_n
        self.informative_episode_count = 0
        self.success_force_samples_n: list[float] = []
        self.decisions: list[ForceDecision] = []

    @staticmethod
    def _force_reference(episode: EpisodeFeedback) -> float:
        if episode.mean_contact_force_n > 0.0:
            return episode.mean_contact_force_n
        return episode.target_force_n

    def _classify(
        self, episode: EpisodeFeedback, recommendation: VLMRecommendation
    ) -> str:
        if episode.success:
            return "success"
        labels = f"{episode.failure_reason} {recommendation.failure_cause} {recommendation.force_assessment}".lower()
        if any(token in labels for token in self._HIGH_FORCE_TOKENS):
            return "too_high"
        if episode.contact_fraction > 0.0 and any(
            token in labels for token in self._LOW_FORCE_TOKENS
        ):
            return "too_low"
        return "non_force"

    def _tighten_evidence(self, episode: EpisodeFeedback, update_kind: str) -> None:
        margin = self.config.evidence_margin_n
        reference = self._force_reference(episode)
        if update_kind == "too_low":
            candidate = (
                min(
                    episode.target_force_n,
                    max(reference, self.current_range_n.minimum_n),
                )
                + margin
            )
            # Reject contradictory evidence instead of relaxing the opposite,
            # already-established upper bound.
            if candidate < self.evidence_upper_bound_n:
                self.evidence_lower_bound_n = max(
                    self.evidence_lower_bound_n, candidate
                )
            self.informative_episode_count += 1
        elif update_kind == "too_high":
            candidate = (
                max(
                    episode.target_force_n,
                    min(reference, self.current_range_n.maximum_n),
                )
                - margin
            )
            if candidate > self.evidence_lower_bound_n:
                self.evidence_upper_bound_n = min(
                    self.evidence_upper_bound_n, candidate
                )
            self.informative_episode_count += 1
        elif update_kind == "success":
            self.success_force_samples_n.append(reference)
            self.informative_episode_count += 1

        physical = self.config.physical_range_n
        self.evidence_lower_bound_n = max(
            physical.minimum_n, self.evidence_lower_bound_n
        )
        self.evidence_upper_bound_n = min(
            physical.maximum_n, self.evidence_upper_bound_n
        )

    def _success_center_and_width(self) -> tuple[float, float] | None:
        if not self.success_force_samples_n:
            return None
        center = float(statistics.median(self.success_force_samples_n))
        deviations = [abs(value - center) for value in self.success_force_samples_n]
        robust_sigma = 1.4826 * float(statistics.median(deviations))
        width = 2.0 * max(
            0.5 * self.config.minimum_range_width_n,
            robust_sigma + self.config.success_margin_n,
        )
        return center, width

    @staticmethod
    def _clamp_center(
        center: float, *, lower: float, upper: float, width: float
    ) -> float:
        half = 0.5 * width
        return min(upper - half, max(lower + half, center))

    def update(
        self, recommendation: VLMRecommendation, episode: EpisodeFeedback
    ) -> ForceDecision:
        previous = self.current_range_n
        update_kind = self._classify(episode, recommendation)
        self._tighten_evidence(episode, update_kind)

        lower = self.evidence_lower_bound_n
        upper = self.evidence_upper_bound_n
        admissible_width = upper - lower
        minimum_width = min(self.config.minimum_range_width_n, admissible_width)
        proposal = recommendation.target_range_n
        proposed_center = min(upper, max(lower, proposal.center_n))
        success_target = self._success_center_and_width()
        if success_target is not None:
            success_center, success_width = success_target
            confidence = recommendation.confidence
            proposed_center = (
                confidence * proposed_center + (1.0 - confidence) * success_center
            )
            proposed_width = min(proposal.width_n, success_width)
        else:
            proposed_width = proposal.width_n

        if update_kind == "non_force":
            new_width = min(previous.width_n, admissible_width)
            new_center = self._clamp_center(
                previous.center_n, lower=lower, upper=upper, width=new_width
            )
        else:
            n = max(self.informative_episode_count, 1)
            gain = n ** (-self.config.update_power)
            new_center = previous.center_n + gain * (
                proposed_center - previous.center_n
            )
            scheduled_width = self.config.initial_range_n.width_n / math.sqrt(1.0 + n)
            new_width = max(
                minimum_width,
                min(
                    previous.width_n, proposed_width, scheduled_width, admissible_width
                ),
            )
            new_center = self._clamp_center(
                new_center, lower=lower, upper=upper, width=new_width
            )

        self.current_range_n = ForceRange(
            new_center - 0.5 * new_width, new_center + 0.5 * new_width
        )
        decision = ForceDecision(
            episode_index=episode.episode_index,
            previous_range_n=previous,
            vlm_range_n=proposal,
            target_range_n=self.current_range_n,
            evidence_lower_bound_n=lower,
            evidence_upper_bound_n=upper,
            informative_episode_count=self.informative_episode_count,
            update_kind=update_kind,
        )
        self.decisions.append(decision)
        return decision

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                **asdict(self.config),
                "physical_range_n": self.config.physical_range_n.as_list(),
                "initial_range_n": self.config.initial_range_n.as_list(),
            },
            "current_range_n": self.current_range_n.as_list(),
            "evidence_lower_bound_n": self.evidence_lower_bound_n,
            "evidence_upper_bound_n": self.evidence_upper_bound_n,
            "informative_episode_count": self.informative_episode_count,
            "success_force_samples_n": list(self.success_force_samples_n),
            "decisions": [item.to_dict() for item in self.decisions],
        }


__all__ = ["ConvergentForceEstimator", "ForceEstimatorConfig"]
