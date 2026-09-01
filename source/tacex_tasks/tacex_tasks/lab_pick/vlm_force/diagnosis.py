"""Episode-level failure attribution for force-range adaptation."""

from __future__ import annotations

from .contracts import ForceRange


def diagnose_episode_failure(
    raw_reason: str,
    *,
    touched: bool,
    contact_fraction: float,
    bilateral_contact_fraction: float,
    mean_force_n: float,
    attempted_range_n: ForceRange,
) -> str:
    """Separate task-force evidence from contact and tracking failures.

    A measured force below the commanded range is evidence that the command was
    not realized, not evidence that the task's desired force is too low.  Only a
    drop after the commanded range was physically reached is allowed to tighten
    the lower force bound.
    """

    if raw_reason in {
        "success",
        "object_broken",
        "object_too_far",
        "ee_outside_workspace",
    }:
        return raw_reason
    if not touched or contact_fraction <= 0.0:
        return "no_contact"
    if bilateral_contact_fraction < 0.01:
        return "bad_alignment"
    if mean_force_n < attempted_range_n.minimum_n:
        return "force_tracking_error"
    if raw_reason == "object_dropped":
        return "object_dropped"
    return "trajectory_error"


__all__ = ["diagnose_episode_failure"]
