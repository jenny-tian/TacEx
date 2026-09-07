"""Small, deterministic physics helpers for the seal-tightening task."""

from __future__ import annotations

import torch


def seal_torque_from_angle(
    turn_angle_rad: torch.Tensor,
    *,
    engagement_angle_rad: float,
    running_torque_nm: float,
    compression_stiffness_nm_per_rad: float,
) -> torch.Tensor:
    """Return the quasi-static reaction torque of the equivalent screw seal.

    Before the elastomer touches the cap, only the thread running torque is
    present.  After engagement, compression raises torque linearly.  Keeping
    this law explicit makes the success window and collected labels auditable.
    """

    angle = torch.clamp(turn_angle_rad, min=0.0)
    compression_angle = torch.clamp(angle - engagement_angle_rad, min=0.0)
    running = torch.where(
        angle > 0.0,
        torch.full_like(angle, running_torque_nm),
        torch.zeros_like(angle),
    )
    return running + compression_stiffness_nm_per_rad * compression_angle


def seal_compression_from_angle(
    turn_angle_rad: torch.Tensor,
    *,
    engagement_angle_rad: float,
    thread_pitch_m_per_turn: float,
    maximum_compression_m: float,
) -> torch.Tensor:
    """Convert post-engagement rotation into axial O-ring compression."""

    compression_angle = torch.clamp(turn_angle_rad - engagement_angle_rad, min=0.0)
    compression = compression_angle * thread_pitch_m_per_turn / (2.0 * torch.pi)
    return torch.clamp(compression, max=maximum_compression_m)


def in_torque_window(torque_nm: torch.Tensor, lower_nm: float, upper_nm: float) -> torch.Tensor:
    """Return whether each torque is inside the inclusive acceptance window."""

    return (torque_nm >= lower_nm) & (torque_nm <= upper_nm)
