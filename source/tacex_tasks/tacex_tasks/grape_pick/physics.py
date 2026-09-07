from __future__ import annotations

import torch


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
