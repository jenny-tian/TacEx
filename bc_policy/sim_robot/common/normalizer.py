from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ArrayStats:
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray

    @classmethod
    def from_array(cls, array: np.ndarray) -> "ArrayStats":
        array = np.asarray(array, dtype=np.float32)
        return cls(
            mean=array.mean(axis=0).astype(np.float32),
            std=array.std(axis=0).astype(np.float32),
            min=array.min(axis=0).astype(np.float32),
            max=array.max(axis=0).astype(np.float32),
        )

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArrayStats":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            min=np.asarray(data["min"], dtype=np.float32),
            max=np.asarray(data["max"], dtype=np.float32),
        )


class LinearNormalizer:
    """Per-field linear normalizer.

    Unlike the real-robot baseline, the sim dataset normalizes every action
    dimension, including the gripper-width/action-width dimension.
    """

    def __init__(
        self,
        stats: dict[str, ArrayStats] | None = None,
        mode: str = "limits",
        range_eps: float = 1e-4,
    ) -> None:
        if mode not in {"standard", "gaussian", "limits"}:
            raise ValueError("mode must be one of: 'standard', 'gaussian', 'limits'")
        self.stats = stats or {}
        self.mode = "standard" if mode == "gaussian" else mode
        self.range_eps = float(range_eps)

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "range_eps": self.range_eps,
            "stats": {key: value.to_dict() for key, value in self.stats.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        mode = state.get("mode", "limits")
        self.mode = "standard" if mode == "gaussian" else mode
        self.range_eps = float(state.get("range_eps", 1e-4))
        self.stats = {key: ArrayStats.from_dict(value) for key, value in state["stats"].items()}

    def _params_numpy(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        stat = self.stats[key]
        if self.mode == "standard":
            center = stat.mean.astype(np.float32)
            scale = np.where(stat.std < self.range_eps, 1.0, stat.std).astype(np.float32)
            return center, scale

        input_range = stat.max - stat.min
        ignore_dim = input_range < self.range_eps
        center = np.where(ignore_dim, stat.min, (stat.max + stat.min) / 2.0).astype(np.float32)
        scale = np.where(ignore_dim, 1.0, input_range / 2.0).astype(np.float32)
        return center, scale

    def normalize_numpy(self, key: str, value: np.ndarray) -> np.ndarray:
        center, scale = self._params_numpy(key)
        return ((value - center) / scale).astype(np.float32)

    def unnormalize_numpy(self, key: str, value: np.ndarray) -> np.ndarray:
        center, scale = self._params_numpy(key)
        return (value * scale + center).astype(np.float32)

    def normalize_tensor(self, key: str, value: torch.Tensor) -> torch.Tensor:
        center, scale = self._params_numpy(key)
        center_t = torch.as_tensor(center, device=value.device, dtype=value.dtype)
        scale_t = torch.as_tensor(scale, device=value.device, dtype=value.dtype)
        return (value - center_t) / scale_t

    def unnormalize_tensor(self, key: str, value: torch.Tensor) -> torch.Tensor:
        center, scale = self._params_numpy(key)
        center_t = torch.as_tensor(center, device=value.device, dtype=value.dtype)
        scale_t = torch.as_tensor(scale, device=value.device, dtype=value.dtype)
        return value * scale_t + center_t

