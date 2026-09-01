"""Typed, JSON-safe contracts shared by the force adaptation modules."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, received {value!r}.")
    return result


@dataclass(frozen=True)
class ForceRange:
    """Closed target interval for one fingertip's normal contact force."""

    minimum_n: float
    maximum_n: float

    def __post_init__(self) -> None:
        lower = _finite(self.minimum_n, name="minimum_n")
        upper = _finite(self.maximum_n, name="maximum_n")
        if lower < 0.0:
            raise ValueError("minimum_n must be non-negative.")
        if upper <= lower:
            raise ValueError("maximum_n must be greater than minimum_n.")
        object.__setattr__(self, "minimum_n", lower)
        object.__setattr__(self, "maximum_n", upper)

    @property
    def center_n(self) -> float:
        return 0.5 * (self.minimum_n + self.maximum_n)

    @property
    def width_n(self) -> float:
        return self.maximum_n - self.minimum_n

    def as_list(self) -> list[float]:
        return [self.minimum_n, self.maximum_n]

    @classmethod
    def from_pair(cls, value: Any, *, name: str = "force range") -> "ForceRange":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must contain exactly [minimum_n, maximum_n].")
        return cls(float(value[0]), float(value[1]))


@dataclass(frozen=True)
class EpisodeFeedback:
    """Compact evidence passed to the VLM exactly once after an episode."""

    episode_index: int
    success: bool
    failure_reason: str
    attempted_range_n: ForceRange
    target_force_n: float
    mean_contact_force_n: float
    peak_contact_force_n: float
    force_rmse_n: float
    contact_fraction: float
    slip_events: int = 0
    max_lift_m: float = 0.0
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.episode_index, bool) or self.episode_index < 0:
            raise ValueError("episode_index must be a non-negative integer.")
        if self.slip_events < 0:
            raise ValueError("slip_events must be non-negative.")
        for name in (
            "target_force_n",
            "mean_contact_force_n",
            "peak_contact_force_n",
            "force_rmse_n",
            "contact_fraction",
            "max_lift_m",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if self.target_force_n < 0.0 or self.mean_contact_force_n < 0.0:
            raise ValueError("Force values must be non-negative.")
        if self.peak_contact_force_n < 0.0 or self.force_rmse_n < 0.0:
            raise ValueError("Force values must be non-negative.")
        if not 0.0 <= self.contact_fraction <= 1.0:
            raise ValueError("contact_fraction must be in [0, 1].")
        reason = str(self.failure_reason).strip() or (
            "success" if self.success else "unknown"
        )
        object.__setattr__(self, "failure_reason", reason)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["attempted_range_n"] = self.attempted_range_n.as_list()
        return result


@dataclass(frozen=True)
class VLMRecommendation:
    """Validated structured response produced by a VLM advisor."""

    target_range_n: ForceRange
    failure_cause: str
    force_assessment: str
    rationale: str
    next_experiment: str
    confidence: float

    def __post_init__(self) -> None:
        confidence = _finite(self.confidence, name="confidence")
        object.__setattr__(self, "confidence", min(1.0, max(0.0, confidence)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_contact_force_range_n": self.target_range_n.as_list(),
            "failure_cause": self.failure_cause,
            "force_assessment": self.force_assessment,
            "rationale": self.rationale,
            "next_experiment": self.next_experiment,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ForceDecision:
    """Safety-projected range used by the next episode's force controller."""

    episode_index: int
    previous_range_n: ForceRange
    vlm_range_n: ForceRange
    target_range_n: ForceRange
    evidence_lower_bound_n: float
    evidence_upper_bound_n: float
    informative_episode_count: int
    update_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "previous_range_n": self.previous_range_n.as_list(),
            "vlm_range_n": self.vlm_range_n.as_list(),
            "target_range_n": self.target_range_n.as_list(),
            "target_force_n": self.target_range_n.center_n,
            "evidence_lower_bound_n": self.evidence_lower_bound_n,
            "evidence_upper_bound_n": self.evidence_upper_bound_n,
            "informative_episode_count": self.informative_episode_count,
            "update_kind": self.update_kind,
        }
