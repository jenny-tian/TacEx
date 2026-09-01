"""Shared deployable tactile observation contract for comparison policies.

Only current GelSight indentation depths and the environment's cumulative
``has_touched`` flag are accepted here. In particular, this module never
queries simulator contact-force ground truth, object pose, rewards, or terminal
labels.
"""

from __future__ import annotations

from typing import Any

import torch


TACTILE_ACTOR_DIM = 5
TACTILE_INDENTATION_SCALE_MM = 0.28125
TACTILE_CONTACT_THRESHOLD_MM = 0.05
TACTILE_CONTRACT_VERSION = 1


def _column(value: Any, *, name: str, device: torch.device | str | None) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    elif tensor.ndim != 2 or tensor.shape[-1] != 1:
        raise ValueError(f"{name} must have shape [B] or [B, 1], got {tuple(tensor.shape)}.")
    tensor = tensor.float()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf.")
    return tensor


def build_tactile_actor(
    left_indentation_mm: Any,
    right_indentation_mm: Any,
    has_touched_this_episode: Any,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the canonical ``[B, 5]`` tactile policy observation."""

    left = _column(left_indentation_mm, name="left_indentation_mm", device=device)
    right = _column(right_indentation_mm, name="right_indentation_mm", device=device)
    touched = _column(has_touched_this_episode, name="has_touched_this_episode", device=device)
    if len({left.shape[0], right.shape[0], touched.shape[0]}) != 1:
        raise ValueError(
            "Tactile inputs must have the same batch size, received "
            f"{left.shape[0]}, {right.shape[0]}, and {touched.shape[0]}."
        )
    tactile = torch.cat(
        (
            (left / TACTILE_INDENTATION_SCALE_MM).clamp(0.0, 2.0),
            (right / TACTILE_INDENTATION_SCALE_MM).clamp(0.0, 2.0),
            (left > TACTILE_CONTACT_THRESHOLD_MM).float(),
            (right > TACTILE_CONTACT_THRESHOLD_MM).float(),
            (touched > 0.0).float(),
        ),
        dim=-1,
    ).to(dtype=torch.float32)
    if tactile.shape != (left.shape[0], TACTILE_ACTOR_DIM):
        raise RuntimeError(f"Unexpected tactile vector shape {tuple(tactile.shape)}.")
    if not bool(torch.isfinite(tactile).all()):
        raise RuntimeError("Constructed tactile vector contains NaN or Inf.")
    return tactile


def build_tactile_actor_from_env(env: Any) -> torch.Tensor:
    """Read only the allowed current tactile signals from a LabPick env."""

    depth_reader = getattr(env, "tactile_contact_depths", None)
    if not callable(depth_reader):
        raise TypeError("Environment must implement tactile_contact_depths().")
    if not hasattr(env, "has_touched"):
        raise TypeError("Environment must expose the per-episode has_touched flag.")
    left, right = depth_reader()
    return build_tactile_actor(left, right, env.has_touched, device=getattr(env, "device", None))


def tactile_contract_metadata() -> dict[str, Any]:
    return {
        "version": TACTILE_CONTRACT_VERSION,
        "dimension": TACTILE_ACTOR_DIM,
        "source": "LabPickEnv.tactile_contact_depths_and_has_touched",
        "indentation_scale_mm": TACTILE_INDENTATION_SCALE_MM,
        "instantaneous_contact_threshold_mm": TACTILE_CONTACT_THRESHOLD_MM,
        "features": [
            "left_indentation_normalized",
            "right_indentation_normalized",
            "left_contact",
            "right_contact",
            "has_touched_this_episode",
        ],
        "forbidden_sources": [
            "privileged_object_pose",
            "future_information",
            "terminal_or_success_label",
            "PhysX_contact_force",
        ],
    }


__all__ = [
    "TACTILE_ACTOR_DIM",
    "TACTILE_CONTACT_THRESHOLD_MM",
    "TACTILE_CONTRACT_VERSION",
    "TACTILE_INDENTATION_SCALE_MM",
    "build_tactile_actor",
    "build_tactile_actor_from_env",
    "tactile_contract_metadata",
]
