from __future__ import annotations

import torch


def _quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Multiply scalar-first quaternions without depending on Isaac Lab."""

    lhs_w, lhs_xyz = lhs[..., :1], lhs[..., 1:]
    rhs_w, rhs_xyz = rhs[..., :1], rhs[..., 1:]
    return torch.cat(
        (
            lhs_w * rhs_w - torch.sum(lhs_xyz * rhs_xyz, dim=-1, keepdim=True),
            lhs_w * rhs_xyz
            + rhs_w * lhs_xyz
            + torch.linalg.cross(lhs_xyz, rhs_xyz, dim=-1),
        ),
        dim=-1,
    )


def compute_stem_restoring_torque(
    fruit_quat_w: torch.Tensor,
    rest_quat_w: torch.Tensor,
    fruit_ang_vel_w: torch.Tensor,
    *,
    stiffness_nm_per_rad: float,
    damping_nm_s_per_rad: float,
    max_torque_nm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a bounded world-frame torque resisting attached-fruit rotation.

    The translational tether alone cannot transmit a moment because its force
    is applied at the fruit's centre of mass.  A real pedicel resists both
    bending and torsion before it detaches, which is represented here by an
    isotropic angular spring-damper about the fruit's reset orientation.
    """

    if fruit_quat_w.shape != rest_quat_w.shape or fruit_quat_w.shape[-1] != 4:
        raise ValueError("fruit and rest quaternions must have matching [..., 4] shapes")
    if fruit_ang_vel_w.shape != fruit_quat_w.shape[:-1] + (3,):
        raise ValueError("angular velocity must match the quaternion batch shape")
    if stiffness_nm_per_rad < 0.0 or damping_nm_s_per_rad < 0.0 or max_torque_nm < 0.0:
        raise ValueError("angular stem parameters must be non-negative")

    fruit_quat = fruit_quat_w / torch.linalg.vector_norm(
        fruit_quat_w, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    rest_quat = rest_quat_w / torch.linalg.vector_norm(
        rest_quat_w, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    rest_inverse = torch.cat((rest_quat[..., :1], -rest_quat[..., 1:]), dim=-1)
    error_quat = _quat_multiply(fruit_quat, rest_inverse)
    # q and -q encode the same orientation.  Selecting the non-negative
    # scalar hemisphere gives the shortest restoring rotation (<= pi).
    error_quat = torch.where(error_quat[..., :1] < 0.0, -error_quat, error_quat)
    error_vector = error_quat[..., 1:]
    sine_half_angle = torch.linalg.vector_norm(error_vector, dim=-1)
    angle = 2.0 * torch.atan2(
        sine_half_angle,
        error_quat[..., 0].clamp_min(0.0),
    )
    axis = error_vector / sine_half_angle.unsqueeze(-1).clamp_min(1.0e-8)
    rotation_vector_w = axis * angle.unsqueeze(-1)

    torque_w = (
        -float(stiffness_nm_per_rad) * rotation_vector_w
        - float(damping_nm_s_per_rad) * fruit_ang_vel_w
    )
    magnitude = torch.linalg.vector_norm(torque_w, dim=-1)
    scale = torch.clamp(
        float(max_torque_nm) / magnitude.clamp_min(1.0e-8),
        max=1.0,
    )
    return torque_w * scale.unsqueeze(-1), angle


def compute_stem_tether_force(
    grape_pos_w: torch.Tensor,
    grape_lin_vel_w: torch.Tensor,
    anchor_pos_w: torch.Tensor,
    *,
    rest_length_m: float,
    stiffness_n_per_m: float,
    damping_n_s_per_m: float,
    max_tension_n: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the world-frame stem force, scalar tension and extension.

    The stem is a tension-only Kelvin-Voigt element.  It therefore pulls a
    grape towards the vine but never pushes it away.  This model is stable,
    deterministic and exposes the same scalar tension that drives detachment.
    """

    anchor_to_grape = grape_pos_w - anchor_pos_w
    distance = torch.linalg.norm(anchor_to_grape, dim=-1)
    direction = anchor_to_grape / distance.unsqueeze(-1).clamp_min(1.0e-8)
    extension = torch.clamp(distance - float(rest_length_m), min=0.0)
    separating_speed = torch.sum(grape_lin_vel_w * direction, dim=-1)
    tension = stiffness_n_per_m * extension + damping_n_s_per_m * separating_speed
    tension = torch.clamp(tension, min=0.0, max=float(max_tension_n))
    force_w = -tension.unsqueeze(-1) * direction
    return force_w, tension, extension


def update_threshold_dwell(
    counter: torch.Tensor,
    values: torch.Tensor,
    threshold: float | torch.Tensor,
    *,
    required_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update consecutive-threshold counters and return newly triggered rows."""

    threshold_value = torch.as_tensor(threshold, dtype=values.dtype, device=values.device)
    over = values >= threshold_value
    updated = torch.where(over, counter + 1, torch.zeros_like(counter))
    triggered = updated >= max(int(required_steps), 1)
    return updated, triggered
