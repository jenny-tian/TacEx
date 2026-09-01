"""Durable, JSON-safe online metrics for Clean DSRL experiments."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


FAILURE_FLAG_KEYS = (
    ("object_broken", "LabPick/broken_terminal_step"),
    ("object_dropped", "LabPick/object_dropped_terminal_step"),
    ("object_too_far", "LabPick/object_too_far_terminal_step"),
    ("ee_outside_workspace", "LabPick/ee_outside_workspace_terminal_step"),
)
SUCCESS_FLAG_KEY = "LabPick/success_terminal_step"
TIMEOUT_FLAG_KEY = "LabPick/timeout_terminal_step"
FORCE_KEYS = {
    "contact_force_n": "LabPick/contact_force_n",
    "net_contact_force_n": "LabPick/net_contact_force_n",
}


def scalar(value: Any, default: float = 0.0) -> float:
    """Convert the scalar-like values emitted by IsaacLab to a finite float."""

    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        result = float(value.detach().reshape(-1)[0].cpu().item())
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return float(default)
        result = float(value.reshape(-1)[0])
    else:
        result = float(value)
    return result if math.isfinite(result) else float(default)


def any_flag(value: Any) -> bool:
    """Return whether any element of a simulator boolean is true."""

    if isinstance(value, torch.Tensor):
        return bool(value.detach().bool().any().cpu().item())
    return bool(np.asarray(value).astype(bool).any())


def extract_step_metrics(info: dict[str, Any]) -> dict[str, Any]:
    """Extract JSON-safe task metrics from one physical environment step."""

    log = info.get("log", {})
    flags = {
        reason: scalar(log.get(key), 0.0) > 0.5
        for reason, key in FAILURE_FLAG_KEYS
    }
    flags["success"] = scalar(log.get(SUCCESS_FLAG_KEY), 0.0) > 0.5
    flags["timeout"] = scalar(log.get(TIMEOUT_FLAG_KEY), 0.0) > 0.5
    return {
        "flags": flags,
        **{
            metric: scalar(log.get(key), 0.0)
            for metric, key in FORCE_KEYS.items()
        },
        "lift_m": scalar(log.get("LabPick/lift_m"), 0.0),
        "grasp_distance_m": scalar(log.get("LabPick/grasp_distance_m"), 0.0),
    }


def classify_terminal(
    flags: dict[str, bool],
    *,
    terminated: bool,
    truncated: bool,
) -> tuple[str, bool, str | None]:
    """Classify an outer transition using a deterministic safety-first order."""

    for reason, _ in FAILURE_FLAG_KEYS:
        if flags.get(reason, False):
            return "failure", False, reason
    if flags.get("success", False):
        return "success", True, None
    if truncated or flags.get("timeout", False):
        return "failure", False, "timeout"
    if terminated:
        return "failure", False, "unknown_terminal"
    return "ongoing", False, None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OnlineDSRLJSONLLogger:
    """Append and fsync one row at a time so interrupted runs remain auditable."""

    interaction_filename = "online_interactions.jsonl"
    episode_filename = "online_episodes.jsonl"

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interaction_path = self.output_dir / self.interaction_filename
        self.episode_path = self.output_dir / self.episode_filename
        existing = [
            str(path)
            for path in (self.interaction_path, self.episode_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Refusing to append a new experiment to existing online metrics: "
                + ", ".join(existing)
            )
        self._interaction_stream = self.interaction_path.open(
            "x", encoding="utf-8", buffering=1
        )
        self._episode_stream = self.episode_path.open(
            "x", encoding="utf-8", buffering=1
        )

    @staticmethod
    def _append(stream, row: dict[str, Any]) -> None:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    def log_interaction(self, row: dict[str, Any]) -> None:
        self._append(self._interaction_stream, row)

    def log_episode(self, row: dict[str, Any]) -> None:
        self._append(self._episode_stream, row)

    def close(self) -> None:
        for stream in (self._interaction_stream, self._episode_stream):
            if not stream.closed:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()


__all__ = [
    "FAILURE_FLAG_KEYS",
    "FORCE_KEYS",
    "OnlineDSRLJSONLLogger",
    "any_flag",
    "classify_terminal",
    "extract_step_metrics",
    "scalar",
    "utc_now",
]
