#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_KEY = "action"
STATE_KEY = "state"
DEFAULT_DINOV3_REPO = Path.home() / ".cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_HDF5_THIRD_IMAGE_KEYS = (
    "rgb_third",
    "robot0_image_third",
    "robot0_image_third_person",
    "robot0_third_image",
    "third_person_image",
)


def numeric_suffix(name: str) -> int:
    tail = name.rsplit("_", maxsplit=1)[-1]
    return int(tail) if tail.isdigit() else 0


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    return batch


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def warmup_cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def quat_to_rot6d(quat: np.ndarray, order: str = "wxyz") -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32).reshape(-1, 4)
    if order == "xyzw":
        x, y, z, w = np.moveaxis(q, -1, 0)
    else:
        w, x, y, z = np.moveaxis(q, -1, 0)

    norm = np.sqrt(w * w + x * x + y * y + z * z)
    norm = np.maximum(norm, 1.0e-8)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    rot = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - z * w)
    rot[:, 0, 2] = 2.0 * (x * z + y * w)
    rot[:, 1, 0] = 2.0 * (x * y + z * w)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - x * w)
    rot[:, 2, 0] = 2.0 * (x * z - y * w)
    rot[:, 2, 1] = 2.0 * (y * z + x * w)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot[:, :, :2].reshape(q.shape[0], 6)


def record_state_from_parts(xyz: np.ndarray, quat: np.ndarray, width: np.ndarray, quat_order: str) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    width = np.asarray(width, dtype=np.float32).reshape(len(xyz), -1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz has shape {xyz.shape}; expected (T, 3).")
    if width.shape[1] != 1:
        raise ValueError(f"width has shape {width.shape}; expected (T, 1).")
    rot6d = quat_to_rot6d(quat, order=quat_order)
    if not (len(xyz) == len(rot6d) == len(width)):
        raise ValueError(f"state lengths differ: xyz={len(xyz)}, quat={len(rot6d)}, width={len(width)}.")
    return np.concatenate((xyz, rot6d, width), axis=-1).astype(np.float32, copy=False)


def normalize_image_array(image: np.ndarray, name: str = "image") -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"{name} has shape {image.shape}; expected a 3D RGB image.")
    if image.shape[-1] == 3:
        pass
    elif image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"{name} has shape {image.shape}; expected HWC or CHW RGB image.")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    image = image.astype(np.float32)
    if image.size and float(np.nanmax(image)) <= 1.5:
        image = image * 255.0
    return np.ascontiguousarray(np.clip(image, 0.0, 255.0).round().astype(np.uint8))


def clamp_indices(length: int, indices: np.ndarray) -> np.ndarray:
    return np.clip(indices, 0, length - 1)


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
        return {"mean": self.mean, "std": self.std, "min": self.min, "max": self.max}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArrayStats":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            min=np.asarray(data["min"], dtype=np.float32),
            max=np.asarray(data["max"], dtype=np.float32),
        )


class LinearNormalizer:
    def __init__(self, stats: dict[str, ArrayStats] | None = None, mode: str = "limits", range_eps: float = 1e-4):
        if mode not in {"limits", "standard", "gaussian"}:
            raise ValueError("normalizer mode must be one of: limits, standard, gaussian")
        self.stats = stats or {}
        self.mode = "standard" if mode == "gaussian" else mode
        self.range_eps = float(range_eps)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "range_eps": self.range_eps,
            "stats": {key: value.to_dict() for key, value in self.stats.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mode = "standard" if state.get("mode", "limits") == "gaussian" else state.get("mode", "limits")
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


@dataclass(frozen=True)
class EpisodeInfo:
    path: Path | None
    name: str
    index: int
    length: int


def split_episode_indices(num_episodes: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.arange(num_episodes)
    if num_episodes <= 1 or val_ratio <= 0:
        return ids, np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = min(max(1, int(round(num_episodes * val_ratio))), num_episodes - 1)
    val = np.sort(ids[:n_val])
    train = np.sort(ids[n_val:])
    return train, val


def discover_record_dirs(root: Path, image_keys: list[str]) -> list[Path]:
    if root.is_dir() and (root / "aligned").is_dir():
        candidates = [root]
    else:
        candidates = sorted(
            [path for path in root.glob("record_*") if path.is_dir() and (path / "aligned").is_dir()],
            key=lambda path: numeric_suffix(path.name),
        )
    record_dirs = []
    for path in candidates:
        aligned = path / "aligned"
        required = [aligned / "action.npy", aligned / "xyz.npy", aligned / "quat.npy", aligned / "width.npy"]
        required.extend(aligned / f"{key}.npy" for key in image_keys)
        if all(item.exists() for item in required):
            record_dirs.append(path)
    return record_dirs


def record_length(record_dir: Path, image_keys: list[str]) -> int:
    aligned = record_dir / "aligned"
    lengths = [
        len(np.load(aligned / "action.npy", mmap_mode="r")),
        len(np.load(aligned / "xyz.npy", mmap_mode="r")),
        len(np.load(aligned / "quat.npy", mmap_mode="r")),
        len(np.load(aligned / "width.npy", mmap_mode="r")),
    ]
    lengths.extend(len(np.load(aligned / f"{key}.npy", mmap_mode="r")) for key in image_keys)
    length = min(lengths)
    if length <= 0:
        raise ValueError(f"Record episode has no usable frames: {record_dir}")
    if len(set(lengths)) != 1:
        raise ValueError(f"Record episode lengths differ in {record_dir}: {lengths}")
    return length


def compute_records_normalizer(record_dirs: list[Path], normalizer_mode: str, quat_order: str) -> LinearNormalizer:
    state_parts = []
    action_parts = []
    for record_dir in record_dirs:
        aligned = record_dir / "aligned"
        state = record_state_from_parts(
            np.load(aligned / "xyz.npy", mmap_mode="r"),
            np.load(aligned / "quat.npy", mmap_mode="r"),
            np.load(aligned / "width.npy", mmap_mode="r"),
            quat_order=quat_order,
        )
        action = np.asarray(np.load(aligned / "action.npy", mmap_mode="r"), dtype=np.float32)
        state_parts.append(state)
        action_parts.append(action)
    return LinearNormalizer(
        stats={
            STATE_KEY: ArrayStats.from_array(np.concatenate(state_parts, axis=0)),
            ACTION_KEY: ArrayStats.from_array(np.concatenate(action_parts, axis=0)),
        },
        mode=normalizer_mode,
    )


class RecordsSequenceDataset(Dataset):
    def __init__(
        self,
        record_dirs: list[Path],
        normalizer: LinearNormalizer,
        image_keys: list[str],
        state_obs_steps: int,
        image_obs_steps: int,
        chunk_size: int,
        quat_order: str = "wxyz",
    ) -> None:
        super().__init__()
        self.record_dirs = list(record_dirs)
        self.normalizer = normalizer
        self.image_keys = list(image_keys)
        self.state_obs_steps = int(state_obs_steps)
        self.image_obs_steps = int(image_obs_steps)
        self.chunk_size = int(chunk_size)
        self.quat_order = quat_order
        self._cache: dict[int, dict[str, Any]] = {}

        self.episodes = [
            EpisodeInfo(path=path, name=path.name, index=numeric_suffix(path.name), length=record_length(path, self.image_keys))
            for path in self.record_dirs
        ]
        if not self.episodes:
            raise ValueError("No record episodes selected.")

        self.samples: list[tuple[int, int]] = []
        for local_idx, episode in enumerate(self.episodes):
            for t in range(episode.length):
                self.samples.append((local_idx, t))

        first = self._episode_arrays(0)
        self.state_dim = int(first[STATE_KEY].shape[-1])
        self.action_dim = int(first[ACTION_KEY].shape[-1])
        first_img = first["images"][self.image_keys[0]]
        self.image_shape = tuple(int(v) for v in first_img.shape[1:])
        self.num_cameras = len(self.image_keys)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def _episode_arrays(self, local_idx: int) -> dict[str, Any]:
        if local_idx not in self._cache:
            record_dir = self.record_dirs[local_idx]
            aligned = record_dir / "aligned"
            self._cache[local_idx] = {
                STATE_KEY: record_state_from_parts(
                    np.load(aligned / "xyz.npy", mmap_mode="r"),
                    np.load(aligned / "quat.npy", mmap_mode="r"),
                    np.load(aligned / "width.npy", mmap_mode="r"),
                    quat_order=self.quat_order,
                ),
                ACTION_KEY: np.load(aligned / "action.npy", mmap_mode="r"),
                "images": {key: np.load(aligned / f"{key}.npy", mmap_mode="r") for key in self.image_keys},
            }
        return self._cache[local_idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        local_idx, t = self.samples[idx]
        episode = self.episodes[local_idx]
        arrays = self._episode_arrays(local_idx)
        length = episode.length

        state_idx = clamp_indices(length, np.arange(t - self.state_obs_steps + 1, t + 1))
        image_idx = clamp_indices(length, np.arange(t - self.image_obs_steps + 1, t + 1))
        action_idx = clamp_indices(length, np.arange(t, t + self.chunk_size))

        state = arrays[STATE_KEY][state_idx].astype(np.float32)
        action = arrays[ACTION_KEY][action_idx].astype(np.float32)
        state = self.normalizer.normalize_numpy(STATE_KEY, state)
        action = self.normalizer.normalize_numpy(ACTION_KEY, action)

        image_steps = []
        for image_t in image_idx:
            cameras = [
                normalize_image_array(arrays["images"][key][int(image_t)], f"{episode.name}/{key}[{int(image_t)}]")
                for key in self.image_keys
            ]
            image_steps.append(np.stack(cameras, axis=0))
        images = np.stack(image_steps, axis=0)
        images_t = torch.from_numpy(images).permute(0, 1, 4, 2, 3).float() / 255.0

        return {
            "obs": {
                STATE_KEY: torch.from_numpy(state),
                "images": images_t,
            },
            ACTION_KEY: torch.from_numpy(action),
            "episode": torch.tensor(episode.index, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
        }


def lerobot_image_feature_name(key: str) -> str:
    if key.startswith("observation."):
        return key
    return f"observation.images.{key}"


def discover_lerobot_videos(root: Path, image_keys: list[str]) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    for key in image_keys:
        feature = lerobot_image_feature_name(key)
        candidates = sorted((root / "videos" / feature).glob("chunk-*/*.mp4"))
        if not candidates:
            raise FileNotFoundError(f"Missing LeRobot video for {feature!r} under {root / 'videos'}")
        if len(candidates) > 1:
            raise NotImplementedError(
                f"Multiple video shards for {feature!r} are not supported by this compact backend. "
                "Use the records backend or merge videos first."
            )
        videos[key] = candidates[0]
    return videos


def load_lerobot_table(root: Path):
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(f"LeRobot backend requires pandas/pyarrow in this environment: {exc}") from exc

    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {root / 'data'}")
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True)


class LeRobotSequenceDataset(Dataset):
    def __init__(
        self,
        root: Path,
        episode_ids: np.ndarray,
        normalizer: LinearNormalizer,
        image_keys: list[str],
        state_obs_steps: int,
        image_obs_steps: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.root = root
        self.df = load_lerobot_table(root)
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self.normalizer = normalizer
        self.image_keys = list(image_keys)
        self.state_obs_steps = int(state_obs_steps)
        self.image_obs_steps = int(image_obs_steps)
        self.chunk_size = int(chunk_size)
        self.videos = discover_lerobot_videos(root, self.image_keys)

        self.episode_positions: dict[int, np.ndarray] = {}
        self.samples: list[tuple[int, int]] = []
        for episode_id in self.episode_ids:
            positions = self.df.index[self.df["episode_index"] == int(episode_id)].to_numpy(dtype=np.int64)
            positions = positions[np.argsort(self.df.loc[positions, "frame_index"].to_numpy())]
            if len(positions) == 0:
                continue
            self.episode_positions[int(episode_id)] = positions
            for t in range(len(positions)):
                self.samples.append((int(episode_id), t))
        if not self.samples:
            raise ValueError("No LeRobot samples selected.")

        first_episode_id, first_t = self.samples[0]
        first = self.df.iloc[int(self.episode_positions[first_episode_id][first_t])]
        self.state_dim = int(np.asarray(first["observation.state"]).reshape(-1).shape[0])
        self.action_dim = int(np.asarray(first["action"]).reshape(-1).shape[0])
        self.num_cameras = len(self.image_keys)
        with open(root / "meta" / "info.json", "r", encoding="utf-8") as f:
            info = json.load(f)
        first_feature = lerobot_image_feature_name(self.image_keys[0])
        self.image_shape = tuple(info["features"][first_feature]["shape"])

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_video_frame(path: Path, frame_index: int) -> np.ndarray:
        try:
            import imageio.v3 as imageio
        except Exception as exc:
            raise RuntimeError(f"LeRobot video backend requires imageio: {exc}") from exc
        return normalize_image_array(imageio.imread(path, index=int(frame_index)), f"{path}[{frame_index}]")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_id, t = self.samples[idx]
        positions = self.episode_positions[episode_id]
        length = len(positions)
        state_idx = clamp_indices(length, np.arange(t - self.state_obs_steps + 1, t + 1))
        image_idx = clamp_indices(length, np.arange(t - self.image_obs_steps + 1, t + 1))
        action_idx = clamp_indices(length, np.arange(t, t + self.chunk_size))

        state = np.stack(
            [np.asarray(self.df.iloc[int(positions[i])]["observation.state"], dtype=np.float32) for i in state_idx],
            axis=0,
        )
        action = np.stack(
            [np.asarray(self.df.iloc[int(positions[i])]["action"], dtype=np.float32) for i in action_idx],
            axis=0,
        )
        state = self.normalizer.normalize_numpy(STATE_KEY, state)
        action = self.normalizer.normalize_numpy(ACTION_KEY, action)

        image_steps = []
        for i in image_idx:
            row = self.df.iloc[int(positions[i])]
            frame_index = int(row["index"])
            cameras = [self._read_video_frame(self.videos[key], frame_index) for key in self.image_keys]
            image_steps.append(np.stack(cameras, axis=0))
        images = np.stack(image_steps, axis=0)

        return {
            "obs": {
                STATE_KEY: torch.from_numpy(state),
                "images": torch.from_numpy(images).permute(0, 1, 4, 2, 3).float() / 255.0,
            },
            ACTION_KEY: torch.from_numpy(action),
            "episode": torch.tensor(episode_id, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
        }


def compute_lerobot_normalizer(root: Path, episode_ids: np.ndarray, normalizer_mode: str) -> LinearNormalizer:
    df = load_lerobot_table(root)
    state_parts = []
    action_parts = []
    for episode_id in episode_ids:
        rows = df[df["episode_index"] == int(episode_id)]
        state_parts.extend(np.asarray(value, dtype=np.float32) for value in rows["observation.state"])
        action_parts.extend(np.asarray(value, dtype=np.float32) for value in rows["action"])
    return LinearNormalizer(
        stats={
            STATE_KEY: ArrayStats.from_array(np.stack(state_parts, axis=0)),
            ACTION_KEY: ArrayStats.from_array(np.stack(action_parts, axis=0)),
        },
        mode=normalizer_mode,
    )


def hdf5_is_file(path: Path) -> bool:
    if path.suffix.lower() in {".h5", ".hdf5", ".hdf"}:
        return True
    try:
        import h5py

        return bool(h5py.is_hdf5(path))
    except Exception:
        return False


def resolve_hdf5_action_key(actions, requested: str) -> str:
    if requested != "auto":
        if requested not in actions:
            raise KeyError(f"HDF5 actions group is missing {requested!r}; available: {list(actions)}")
        return requested
    if "low" in actions:
        return "low"
    if "high" in actions:
        return "high"
    if len(actions) == 1:
        return next(iter(actions.keys()))
    raise KeyError(f"Could not auto-select HDF5 action key; available: {list(actions)}")


def resolve_hdf5_image_keys(obs, requested: str) -> list[str]:
    if requested != "auto":
        keys = parse_csv(requested)
        missing = [key for key in keys if key not in obs]
        if missing:
            raise KeyError(f"HDF5 obs group is missing image keys {missing}; available: {list(obs)}")
        return keys
    keys = ["robot0_image"] if "robot0_image" in obs else []
    for candidate in DEFAULT_HDF5_THIRD_IMAGE_KEYS:
        if candidate in obs:
            keys.append(candidate)
            break
    if not keys:
        raise KeyError(f"Could not auto-select HDF5 image keys; available obs keys: {list(obs)}")
    return keys


class HDF5SequenceDataset(Dataset):
    def __init__(
        self,
        path: Path,
        episode_ids: np.ndarray,
        normalizer: LinearNormalizer,
        state_key: str,
        action_key: str,
        image_keys: list[str],
        state_obs_steps: int,
        image_obs_steps: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.path = str(path)
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self.normalizer = normalizer
        self.state_key = state_key
        self.action_key = action_key
        self.image_keys = list(image_keys)
        self.state_obs_steps = int(state_obs_steps)
        self.image_obs_steps = int(image_obs_steps)
        self.chunk_size = int(chunk_size)
        self._file = None

        import h5py

        with h5py.File(self.path, "r") as f:
            self.episodes = []
            names = sorted(f["data"].keys(), key=numeric_suffix)
            selected = [names[int(i)] for i in self.episode_ids]
            for name in selected:
                demo = f["data"][name]
                length = min(
                    int(demo["obs"][state_key].shape[0]),
                    int(demo["actions"][action_key].shape[0]),
                    *[int(demo["obs"][key].shape[0]) for key in image_keys],
                )
                self.episodes.append(EpisodeInfo(path=None, name=name, index=numeric_suffix(name), length=length))
            first = f["data"][self.episodes[0].name]
            self.state_dim = int(first["obs"][state_key].shape[-1])
            self.action_dim = int(first["actions"][action_key].shape[-1])
            self.image_shape = tuple(first["obs"][image_keys[0]].shape[1:])
            self.num_cameras = len(image_keys)

        self.samples = []
        for local_idx, episode in enumerate(self.episodes):
            for t in range(episode.length):
                self.samples.append((local_idx, t))

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    @property
    def file(self):
        if self._file is None:
            import h5py

            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        local_idx, t = self.samples[idx]
        episode = self.episodes[local_idx]
        demo = self.file["data"][episode.name]
        length = episode.length
        state_idx = clamp_indices(length, np.arange(t - self.state_obs_steps + 1, t + 1))
        image_idx = clamp_indices(length, np.arange(t - self.image_obs_steps + 1, t + 1))
        action_idx = clamp_indices(length, np.arange(t, t + self.chunk_size))

        state = np.stack([demo["obs"][self.state_key][int(i)] for i in state_idx], axis=0).astype(np.float32)
        action = np.stack([demo["actions"][self.action_key][int(i)] for i in action_idx], axis=0).astype(np.float32)
        state = self.normalizer.normalize_numpy(STATE_KEY, state)
        action = self.normalizer.normalize_numpy(ACTION_KEY, action)
        image_steps = []
        for image_t in image_idx:
            cameras = [
                normalize_image_array(demo["obs"][key][int(image_t)], f"{episode.name}/obs/{key}[{int(image_t)}]")
                for key in self.image_keys
            ]
            image_steps.append(np.stack(cameras, axis=0))
        images = np.stack(image_steps, axis=0)

        return {
            "obs": {
                STATE_KEY: torch.from_numpy(state),
                "images": torch.from_numpy(images).permute(0, 1, 4, 2, 3).float() / 255.0,
            },
            ACTION_KEY: torch.from_numpy(action),
            "episode": torch.tensor(episode.index, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
        }


def compute_hdf5_normalizer(
    path: Path, episode_ids: np.ndarray, state_key: str, action_key: str, normalizer_mode: str
) -> LinearNormalizer:
    import h5py

    state_parts = []
    action_parts = []
    with h5py.File(path, "r") as f:
        names = sorted(f["data"].keys(), key=numeric_suffix)
        for episode_id in episode_ids:
            demo = f["data"][names[int(episode_id)]]
            length = min(int(demo["obs"][state_key].shape[0]), int(demo["actions"][action_key].shape[0]))
            state_parts.append(demo["obs"][state_key][:length])
            action_parts.append(demo["actions"][action_key][:length])
    return LinearNormalizer(
        stats={
            STATE_KEY: ArrayStats.from_array(np.concatenate(state_parts, axis=0)),
            ACTION_KEY: ArrayStats.from_array(np.concatenate(action_parts, axis=0)),
        },
        mode=normalizer_mode,
    )


def infer_data_format(data_root: Path, requested: str, image_keys: list[str]) -> str:
    if requested != "auto":
        return requested
    if data_root.is_dir() and discover_record_dirs(data_root, image_keys):
        return "records"
    if data_root.is_dir() and (data_root / "meta" / "info.json").exists() and (data_root / "data").exists():
        return "lerobot"
    if data_root.is_file() and hdf5_is_file(data_root):
        return "hdf5"
    raise ValueError(f"Could not infer data format for {data_root}")


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset | None, LinearNormalizer, dict[str, Any]]:
    data_root = resolve_path(args.data_root)
    record_image_keys = parse_csv(args.image_keys)
    data_format = infer_data_format(data_root, args.data_format, record_image_keys)

    if data_format == "records":
        record_dirs = discover_record_dirs(data_root, record_image_keys)
        if not record_dirs:
            raise ValueError(f"No record episodes with image keys {record_image_keys} found under {data_root}")
        train_local, val_local = split_episode_indices(len(record_dirs), args.val_ratio, args.seed)
        train_dirs = [record_dirs[int(i)] for i in train_local]
        val_dirs = [record_dirs[int(i)] for i in val_local]
        normalizer = compute_records_normalizer(train_dirs, args.normalizer_mode, args.quat_order)
        train_set = RecordsSequenceDataset(
            train_dirs, normalizer, record_image_keys, args.state_obs_steps, args.image_obs_steps, args.chunk_size, args.quat_order
        )
        val_set = (
            RecordsSequenceDataset(
                val_dirs,
                normalizer,
                record_image_keys,
                args.state_obs_steps,
                args.image_obs_steps,
                args.chunk_size,
                args.quat_order,
            )
            if val_dirs
            else None
        )
        info = {
            "data_format": data_format,
            "data_root": str(data_root),
            "image_keys": record_image_keys,
            "num_episodes": len(record_dirs),
            "train_episodes": len(train_dirs),
            "val_episodes": len(val_dirs),
        }
        return train_set, val_set, normalizer, info

    if data_format == "lerobot":
        image_keys = record_image_keys
        df = load_lerobot_table(data_root)
        episode_ids_all = np.sort(df["episode_index"].unique().astype(np.int64))
        train_local, val_local = split_episode_indices(len(episode_ids_all), args.val_ratio, args.seed)
        train_ids = episode_ids_all[train_local]
        val_ids = episode_ids_all[val_local]
        normalizer = compute_lerobot_normalizer(data_root, train_ids, args.normalizer_mode)
        train_set = LeRobotSequenceDataset(
            data_root, train_ids, normalizer, image_keys, args.state_obs_steps, args.image_obs_steps, args.chunk_size
        )
        val_set = (
            LeRobotSequenceDataset(
                data_root, val_ids, normalizer, image_keys, args.state_obs_steps, args.image_obs_steps, args.chunk_size
            )
            if len(val_ids)
            else None
        )
        info = {
            "data_format": data_format,
            "data_root": str(data_root),
            "image_keys": image_keys,
            "num_episodes": int(len(episode_ids_all)),
            "train_episodes": int(len(train_ids)),
            "val_episodes": int(len(val_ids)),
        }
        return train_set, val_set, normalizer, info

    if data_format == "hdf5":
        import h5py

        with h5py.File(data_root, "r") as f:
            names = sorted(f["data"].keys(), key=numeric_suffix)
            first = f["data"][names[0]]
            action_key = resolve_hdf5_action_key(first["actions"], args.hdf5_action_key)
            image_keys = resolve_hdf5_image_keys(first["obs"], args.hdf5_image_keys)
            if args.hdf5_state_key not in first["obs"]:
                raise KeyError(f"HDF5 obs group is missing state key {args.hdf5_state_key!r}; available: {list(first['obs'])}")
        train_local, val_local = split_episode_indices(len(names), args.val_ratio, args.seed)
        normalizer = compute_hdf5_normalizer(data_root, train_local, args.hdf5_state_key, action_key, args.normalizer_mode)
        train_set = HDF5SequenceDataset(
            data_root,
            train_local,
            normalizer,
            args.hdf5_state_key,
            action_key,
            image_keys,
            args.state_obs_steps,
            args.image_obs_steps,
            args.chunk_size,
        )
        val_set = (
            HDF5SequenceDataset(
                data_root,
                val_local,
                normalizer,
                args.hdf5_state_key,
                action_key,
                image_keys,
                args.state_obs_steps,
                args.image_obs_steps,
                args.chunk_size,
            )
            if len(val_local)
            else None
        )
        info = {
            "data_format": data_format,
            "data_root": str(data_root),
            "image_keys": image_keys,
            "hdf5_state_key": args.hdf5_state_key,
            "hdf5_action_key": action_key,
            "num_episodes": len(names),
            "train_episodes": int(len(train_local)),
            "val_episodes": int(len(val_local)),
        }
        return train_set, val_set, normalizer, info

    raise ValueError(f"Unsupported data format: {data_format}")


@dataclass
class DINOv3ViTConfig:
    patch_size: int = 16
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    hidden_act: str = "gelu"
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    rope_theta: float = 100.0
    image_size: int = 224
    num_channels: int = 3
    query_bias: bool = True
    key_bias: bool = False
    value_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    layerscale_value: float = 1.0
    drop_path_rate: float = 0.0
    use_gated_mlp: bool = False
    num_register_tokens: int = 4

    @classmethod
    def from_json(cls, path: Path) -> "DINOv3ViTConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = {field for field in cls.__dataclass_fields__}
        values = {key: data[key] for key in allowed if key in data}
        return cls(**values)


def find_default_dino_path() -> Path:
    snapshots = DEFAULT_DINOV3_REPO / "snapshots"
    if snapshots.exists():
        candidates = sorted(snapshots.glob("*"), key=lambda path: path.stat().st_mtime, reverse=True)
        for candidate in candidates:
            if (candidate / "config.json").exists() and (candidate / "model.safetensors").exists():
                return candidate
    raise FileNotFoundError(
        "Could not find local DINOv3 snapshot. Pass --dino-path pointing to a directory with "
        "config.json and model.safetensors."
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches
    q_prefix, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
    k_prefix, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)
    cos = cos[None, None, :, :].to(device=q.device, dtype=q.dtype)
    sin = sin[None, None, :, :].to(device=q.device, dtype=q.dtype)
    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)
    return torch.cat((q_prefix, q_patches), dim=-2), torch.cat((k_prefix, k_patches), dim=-2)


class DINOv3ViTEmbeddings(nn.Module):
    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.empty(1, config.num_register_tokens, config.hidden_size))
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeddings = self.patch_embeddings(pixel_values.to(dtype=self.patch_embeddings.weight.dtype))
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        return torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)


class DINOv3ViTRopePositionEmbedding(nn.Module):
    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.base = float(config.rope_theta)
        self.head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.config.patch_size
        num_patches_w = width // self.config.patch_size
        device = pixel_values.device
        coords_h = torch.arange(0.5, num_patches_h, dtype=torch.float32, device=device) / num_patches_h
        coords_w = torch.arange(0.5, num_patches_w, dtype=torch.float32, device=device) / num_patches_w
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]
        angles = angles.flatten(1, 2).repeat(1, 2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return cos.to(dtype=pixel_values.dtype), sin.to(dtype=pixel_values.dtype)


class DINOv3ViTAttention(nn.Module):
    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(config.attention_dropout)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.key_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.value_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.query_bias)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.proj_bias)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch_size, num_tokens, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, position_embeddings[0], position_embeddings[1])
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim).contiguous()
        return self.o_proj(out)


class DINOv3ViTLayerScale(nn.Module):
    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


class DINOv3ViTMLP(nn.Module):
    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        if config.hidden_act != "gelu":
            raise ValueError(f"Unsupported DINOv3 activation {config.hidden_act!r}; expected gelu.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class DINOv3ViTGatedMLP(nn.Module):
    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x.div(keep_prob) * random_tensor


class DINOv3ViTLayer(nn.Module):
    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DINOv3ViTAttention(config)
        self.layer_scale1 = DINOv3ViTLayerScale(config)
        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = DINOv3ViTGatedMLP(config) if config.use_gated_mlp else DINOv3ViTMLP(config)
        self.layer_scale2 = DINOv3ViTLayerScale(config)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, position_embeddings)
        hidden_states = self.drop_path(self.layer_scale1(hidden_states)) + residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.drop_path(self.layer_scale2(hidden_states)) + residual
        return hidden_states


class MinimalDINOv3ViT(nn.Module):
    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = DINOv3ViTEmbeddings(config)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(config)
        self.layer = nn.ModuleList([DINOv3ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @classmethod
    def from_pretrained_dir(cls, path: Path) -> "MinimalDINOv3ViT":
        config = DINOv3ViTConfig.from_json(path / "config.json")
        model = cls(config)
        state = load_safetensors(path / "model.safetensors")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"DINOv3 state dict mismatch: missing={missing}, unexpected={unexpected}")
        return model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_hw = int(self.config.image_size)
        if pixel_values.shape[-2:] != (target_hw, target_hw):
            pixel_values = F.interpolate(pixel_values, size=(target_hw, target_hw), mode="bicubic", align_corners=False)
        hidden_states = self.embeddings(pixel_values)
        position_embeddings = self.rope_embeddings(pixel_values)
        for layer in self.layer:
            hidden_states = layer(hidden_states, position_embeddings)
        hidden_states = self.norm(hidden_states)
        prefix = 1 + int(self.config.num_register_tokens)
        return hidden_states[:, prefix:, :]


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=x.device) * -scale)
        emb = x[:, None].float() * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class CrossAttentionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        cond_dim: int,
        n_cond_tokens: int,
        n_layer: int = 6,
        n_head: int = 8,
        n_emb: int = 512,
        dropout: float = 0.1,
        n_cond_layers: int = 2,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, n_cond_tokens + 1, n_emb))
        self.drop = nn.Dropout(dropout)

        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_cond_layers)
        else:
            self.encoder = nn.Sequential(nn.Linear(n_emb, 4 * n_emb), nn.GELU(), nn.Linear(4 * n_emb, n_emb))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
        self.apply(self._init_weights)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, sample: torch.Tensor, time_value: torch.Tensor | float, cond_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = sample.shape[0]
        if not torch.is_tensor(time_value):
            time_value = torch.full((batch_size,), float(time_value), device=sample.device, dtype=sample.dtype)
        elif time_value.ndim == 0:
            time_value = time_value[None].expand(batch_size).to(device=sample.device, dtype=sample.dtype)
        else:
            time_value = time_value.to(device=sample.device, dtype=sample.dtype)

        time_token = self.time_emb(time_value).unsqueeze(1).to(dtype=cond_tokens.dtype)
        obs_tokens = self.cond_obs_emb(cond_tokens)
        cond = torch.cat([time_token, obs_tokens], dim=1)
        cond = self.drop(cond + self.cond_pos_emb[:, : cond.shape[1]])
        memory = self.encoder(cond)

        x = self.input_emb(sample)
        x = self.drop(x + self.pos_emb[:, : x.shape[1]])
        x = self.decoder(tgt=x, memory=memory)
        return self.head(self.ln_f(x))


class VisionTokenPooler(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_queries: int, num_heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_queries, input_dim) * 0.02)
        self.attn = nn.MultiheadAttention(input_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        self.projector = nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = patch_tokens.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        pooled = self.attn(query=query, key=patch_tokens, value=patch_tokens, need_weights=False)[0]
        return self.projector(self.norm(pooled))


@dataclass
class DINOFlowMatchingConfig:
    state_dim: int = 10
    action_dim: int = 10
    num_cameras: int = 2
    image_size: int = 224
    state_obs_steps: int = 1
    image_obs_steps: int = 1
    chunk_size: int = 32
    vision_tokens_per_image: int = 8
    dino_hidden_size: int = 384
    cond_dim: int = 512
    transformer_layers: int = 6
    transformer_heads: int = 8
    transformer_dim: int = 512
    transformer_cond_layers: int = 2
    dropout: float = 0.1
    time_embed_scale: float = 1000.0
    num_inference_steps: int = 4
    ode_solver: str = "euler"
    clip_sample: bool = True
    freeze_dino: bool = True
    unfreeze_dino_last_n: int = 0
    dino_path: str = ""
    image_keys: tuple[str, ...] = ("rgb", "rgb_third")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DINOFlowMatchingConfig":
        if isinstance(data.get("image_keys"), list):
            data = dict(data)
            data["image_keys"] = tuple(data["image_keys"])
        return cls(**data)


class DINOFlowMatchingPolicy(nn.Module):
    def __init__(self, config: DINOFlowMatchingConfig, dino: MinimalDINOv3ViT) -> None:
        super().__init__()
        if config.transformer_dim % config.transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads")
        if config.ode_solver not in {"euler", "heun"}:
            raise ValueError("ode_solver must be 'euler' or 'heun'")
        self.config = config
        self.dino = dino
        self._configure_dino_trainability()

        self.vision_pooler = VisionTokenPooler(
            input_dim=config.dino_hidden_size,
            output_dim=config.cond_dim,
            num_queries=config.vision_tokens_per_image,
            num_heads=max(1, min(8, config.dino_hidden_size // 64)),
        )
        self.state_projector = nn.Sequential(
            nn.Linear(config.state_dim, config.cond_dim),
            nn.LayerNorm(config.cond_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.cond_dim, config.cond_dim),
            nn.LayerNorm(config.cond_dim),
            nn.GELU(),
        )
        self.state_step_emb = nn.Parameter(torch.randn(1, config.state_obs_steps, config.cond_dim) * 0.02)
        self.image_step_emb = nn.Parameter(torch.randn(1, config.image_obs_steps, 1, 1, config.cond_dim) * 0.02)
        self.camera_emb = nn.Parameter(torch.randn(1, 1, config.num_cameras, 1, config.cond_dim) * 0.02)
        self.modality_emb = nn.Parameter(torch.randn(1, 2, config.cond_dim) * 0.02)

        n_cond_tokens = config.state_obs_steps + config.image_obs_steps * config.num_cameras * config.vision_tokens_per_image
        self.velocity_net = CrossAttentionTransformer(
            input_dim=config.action_dim,
            output_dim=config.action_dim,
            horizon=config.chunk_size,
            cond_dim=config.cond_dim,
            n_cond_tokens=n_cond_tokens,
            n_layer=config.transformer_layers,
            n_head=config.transformer_heads,
            n_emb=config.transformer_dim,
            dropout=config.dropout,
            n_cond_layers=config.transformer_cond_layers,
        )
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def _configure_dino_trainability(self) -> None:
        for parameter in self.dino.parameters():
            parameter.requires_grad_(not self.config.freeze_dino)
        if self.config.freeze_dino and self.config.unfreeze_dino_last_n > 0:
            for parameter in self.dino.parameters():
                parameter.requires_grad_(False)
            n = min(int(self.config.unfreeze_dino_last_n), len(self.dino.layer))
            for layer in self.dino.layer[-n:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.dino.norm.parameters():
                parameter.requires_grad_(True)

    @property
    def dino_is_fully_frozen(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.dino.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        if self.dino_is_fully_frozen:
            self.dino.eval()
        return self

    def _model_time(self, t: torch.Tensor) -> torch.Tensor:
        return t * float(self.config.time_embed_scale)

    def encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        state = obs[STATE_KEY]
        images = obs["images"]
        batch_size, image_steps, num_cameras, channels, height, width = images.shape
        if num_cameras != self.config.num_cameras:
            raise ValueError(f"Expected {self.config.num_cameras} cameras, got {num_cameras}.")

        state_tokens = self.state_projector(state) + self.state_step_emb[:, : state.shape[1]]
        state_tokens = state_tokens + self.modality_emb[:, 0:1]

        flat_images = images.reshape(batch_size * image_steps * num_cameras, channels, height, width)
        flat_images = (flat_images - self.image_mean.to(dtype=flat_images.dtype)) / self.image_std.to(dtype=flat_images.dtype)
        if self.dino_is_fully_frozen:
            with torch.no_grad():
                patch_tokens = self.dino(flat_images)
        else:
            patch_tokens = self.dino(flat_images)
        vision_tokens = self.vision_pooler(patch_tokens)
        vision_tokens = vision_tokens.reshape(
            batch_size, image_steps, num_cameras, self.config.vision_tokens_per_image, self.config.cond_dim
        )
        vision_tokens = vision_tokens + self.image_step_emb[:, :image_steps] + self.camera_emb[:, :, :num_cameras]
        vision_tokens = vision_tokens + self.modality_emb[:, 1].view(1, 1, 1, 1, -1)
        vision_tokens = vision_tokens.reshape(batch_size, -1, self.config.cond_dim)
        return torch.cat([state_tokens, vision_tokens], dim=1)

    def _velocity(self, sample: torch.Tensor, t: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        return self.velocity_net(sample, self._model_time(t), cond_tokens=cond_tokens)

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        obs = batch["obs"]
        action = batch[ACTION_KEY]
        batch_size = action.shape[0]
        cond_tokens = self.encode_obs(obs)
        x0 = torch.randn_like(action)
        x1 = action
        t = torch.rand(batch_size, device=action.device, dtype=action.dtype)
        t_view = t.view(batch_size, *([1] * (action.ndim - 1)))
        xt = (1.0 - t_view) * x0 + t_view * x1
        target_velocity = x1 - x0
        pred_velocity = self._velocity(xt, t, cond_tokens)
        loss = F.mse_loss(pred_velocity, target_velocity)
        return {"loss": loss}

    @torch.no_grad()
    def predict_action(
        self,
        obs: dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        obs = {key: value.to(device) for key, value in obs.items()}
        batch_size = obs[STATE_KEY].shape[0]
        steps = self.config.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        if steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        cond_tokens = self.encode_obs(obs)
        action = torch.randn(
            batch_size,
            self.config.chunk_size,
            self.config.action_dim,
            device=device,
            generator=generator,
        )
        dt = 1.0 / float(steps)
        for i in range(steps):
            t0 = torch.full((batch_size,), i / float(steps), device=device, dtype=action.dtype)
            v0 = self._velocity(action, t0, cond_tokens)
            if self.config.ode_solver == "heun" and i < steps - 1:
                proposal = action + dt * v0
                t1 = torch.full((batch_size,), (i + 1) / float(steps), device=device, dtype=action.dtype)
                v1 = self._velocity(proposal, t1, cond_tokens)
                action = action + 0.5 * dt * (v0 + v1)
            else:
                action = action + dt * v0
            if self.config.clip_sample:
                action = action.clamp(-1.0, 1.0)
        return {"action": action, "action_pred": action}


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.averaged_model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for parameter in self.averaged_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def step(self, model: nn.Module) -> None:
        for averaged, current in zip(self.averaged_model.parameters(), model.parameters()):
            averaged.mul_(self.decay).add_(current.detach(), alpha=1.0 - self.decay)
        for averaged, current in zip(self.averaged_model.buffers(), model.buffers()):
            averaged.copy_(current)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "averaged_model": self.averaged_model.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.averaged_model.load_state_dict(state["averaged_model"])


def load_dino_for_config(dino_path: Path | None) -> tuple[MinimalDINOv3ViT, Path]:
    resolved = resolve_path(dino_path) if dino_path is not None else find_default_dino_path()
    if not (resolved / "config.json").exists() or not (resolved / "model.safetensors").exists():
        raise FileNotFoundError(f"DINO path must contain config.json and model.safetensors: {resolved}")
    dino = MinimalDINOv3ViT.from_pretrained_dir(resolved)
    return dino, resolved


def run_epoch(
    model: DINOFlowMatchingPolicy,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    ema: EMAModel | None = None,
    scaler: torch.amp.GradScaler | None = None,
    train: bool = True,
    epoch: int = 0,
    max_steps: int | None = None,
    base_lr: float = 1e-4,
    total_steps: int = 1,
    warmup_steps: int = 0,
    global_step: int = 0,
    grad_clip: float = 1.0,
    amp: bool = False,
) -> tuple[dict[str, float], int]:
    model.train(train)
    sums: dict[str, float] = {"loss": 0.0}
    n_batches = 0
    for local_step, batch in enumerate(loader):
        if max_steps is not None and local_step >= max_steps:
            break
        batch = move_to_device(batch, device)
        if train:
            assert optimizer is not None
            lr = warmup_cosine_lr(global_step, total_steps, warmup_steps, base_lr)
            set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if train and scaler is not None and scaler.is_enabled():
                with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    loss_dict = model.compute_loss(batch)
                    loss = loss_dict["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    loss_dict = model.compute_loss(batch)
                    loss = loss_dict["loss"]
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        if train and ema is not None:
            ema.step(model)
        values = {key: float(value.detach().cpu()) for key, value in loss_dict.items()}
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + value
        n_batches += 1
        if train:
            global_step += 1

    mean = {key: value / max(n_batches, 1) for key, value in sums.items()}
    prefix = "train" if train else "val"
    print(f"epoch {epoch:04d} {prefix} batches={n_batches} loss={mean['loss']:.6f}")
    return mean, global_step


def save_checkpoint(
    path: Path,
    model: DINOFlowMatchingPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    normalizer_state: dict[str, Any],
    train_config: dict[str, Any],
    ema: EMAModel | None = None,
) -> None:
    checkpoint = {
        "model_type": "tacex_dinov3_flow_matching_bc",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "normalizer": normalizer_state,
        "policy_config": model.config.to_dict(),
        "train_config": train_config,
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, map_location=None) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_policy(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    use_ema: bool = True,
) -> tuple[DINOFlowMatchingPolicy, LinearNormalizer, dict[str, Any]]:
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    config = DINOFlowMatchingConfig.from_dict(ckpt["policy_config"])
    dino_path = Path(config.dino_path) if config.dino_path else None
    dino, _ = load_dino_for_config(dino_path)
    model = DINOFlowMatchingPolicy(config, dino).to(device)
    if use_ema and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"]["averaged_model"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(ckpt["normalizer"])
    return model, normalizer, ckpt


class FlowMatchingBCRunner:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        use_ema: bool = True,
        num_inference_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.model, self.normalizer, self.checkpoint = load_policy(checkpoint_path, device=device, use_ema=use_ema)
        self.config = self.model.config
        self.device = next(self.model.parameters()).device
        self.num_inference_steps = num_inference_steps
        self.state_history: deque[np.ndarray] = deque(maxlen=self.config.state_obs_steps)
        self.image_history: deque[np.ndarray] = deque(maxlen=self.config.image_obs_steps)
        self.generator = None
        if seed is not None:
            try:
                self.generator = torch.Generator(device=self.device)
            except RuntimeError:
                self.generator = torch.Generator()
            self.generator.manual_seed(int(seed))

    def reset(self) -> None:
        self.state_history.clear()
        self.image_history.clear()

    def update(self, state: np.ndarray, images: dict[str, np.ndarray] | list[np.ndarray] | np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.config.state_dim:
            raise ValueError(f"state must have shape ({self.config.state_dim},), got {state.shape}")
        if isinstance(images, dict):
            cameras = [normalize_image_array(images[key], key) for key in self.config.image_keys]
        elif isinstance(images, list):
            cameras = [normalize_image_array(image, f"image_{i}") for i, image in enumerate(images)]
        else:
            image_array = np.asarray(images)
            if image_array.ndim == 4:
                cameras = [normalize_image_array(image_array[i], f"image_{i}") for i in range(image_array.shape[0])]
            else:
                cameras = [normalize_image_array(image_array, "image")]
        if len(cameras) != self.config.num_cameras:
            raise ValueError(f"Expected {self.config.num_cameras} camera images, got {len(cameras)}")
        camera_stack = np.stack(cameras, axis=0)

        if not self.state_history:
            for _ in range(self.config.state_obs_steps):
                self.state_history.append(state.copy())
            for _ in range(self.config.image_obs_steps):
                self.image_history.append(camera_stack.copy())
            return
        self.state_history.append(state.copy())
        self.image_history.append(camera_stack.copy())

    def is_ready(self) -> bool:
        return len(self.state_history) == self.config.state_obs_steps and len(self.image_history) == self.config.image_obs_steps

    def build_model_obs(self) -> dict[str, torch.Tensor]:
        if not self.is_ready():
            raise RuntimeError("Observation history is not ready. Call update() first.")
        state = np.stack(list(self.state_history), axis=0).astype(np.float32)
        state = self.normalizer.normalize_numpy(STATE_KEY, state)
        images = np.stack(list(self.image_history), axis=0)
        images_t = torch.from_numpy(images).permute(0, 1, 4, 2, 3).float() / 255.0
        return {
            STATE_KEY: torch.from_numpy(state).unsqueeze(0).to(self.device),
            "images": images_t.unsqueeze(0).to(self.device),
        }

    @torch.inference_mode()
    def predict_action_chunk(
        self, state: np.ndarray | None = None, images: dict[str, np.ndarray] | list[np.ndarray] | np.ndarray | None = None
    ) -> np.ndarray:
        if state is not None or images is not None:
            if state is None or images is None:
                raise ValueError("state and images must be provided together.")
            self.update(state, images)
        obs = self.build_model_obs()
        result = self.model.predict_action(obs, generator=self.generator, num_inference_steps=self.num_inference_steps)
        action_norm = result[ACTION_KEY].detach().cpu().numpy()[0]
        return self.normalizer.unnormalize_numpy(ACTION_KEY, action_norm)


def make_policy_config(
    args: argparse.Namespace,
    train_set: Dataset,
    dino_path: Path,
    dino_hidden_size: int,
    image_keys: list[str],
) -> DINOFlowMatchingConfig:
    return DINOFlowMatchingConfig(
        state_dim=int(train_set.state_dim),
        action_dim=int(train_set.action_dim),
        num_cameras=int(train_set.num_cameras),
        image_size=int(train_set.image_shape[0]),
        state_obs_steps=int(args.state_obs_steps),
        image_obs_steps=int(args.image_obs_steps),
        chunk_size=int(args.chunk_size),
        vision_tokens_per_image=int(args.vision_tokens_per_image),
        dino_hidden_size=int(dino_hidden_size),
        cond_dim=int(args.cond_dim),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
        transformer_dim=int(args.transformer_dim),
        transformer_cond_layers=int(args.transformer_cond_layers),
        dropout=float(args.dropout),
        time_embed_scale=float(args.time_embed_scale),
        num_inference_steps=int(args.num_inference_steps),
        ode_solver=args.ode_solver,
        clip_sample=not args.no_clip_sample,
        freeze_dino=bool(args.freeze_dino),
        unfreeze_dino_last_n=int(args.unfreeze_dino_last_n),
        dino_path=str(dino_path),
        image_keys=tuple(image_keys),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TacEx BC with DINOv3 vision and flow matching.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-format", choices=("auto", "records", "lerobot", "hdf5"), default="auto")
    parser.add_argument("--image-keys", type=str, default="rgb,rgb_third")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--state-obs-steps", type=int, default=1)
    parser.add_argument("--image-obs-steps", type=int, default=1)
    parser.add_argument("--vision-tokens-per-image", type=int, default=8)
    parser.add_argument("--dino-path", type=Path, default=None)
    parser.add_argument("--freeze-dino", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze-dino-last-n", type=int, default=0)
    parser.add_argument("--quat-order", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument("--hdf5-action-key", type=str, default="auto")
    parser.add_argument("--hdf5-state-key", type=str, default="robot0_pos")
    parser.add_argument("--hdf5-image-keys", type=str, default="auto")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--normalizer-mode", choices=("limits", "standard", "gaussian"), default="limits")
    parser.add_argument("--cond-dim", type=int, default=512)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-dim", type=int, default=512)
    parser.add_argument("--transformer-cond-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--time-embed-scale", type=float, default=1000.0)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--ode-solver", choices=("euler", "heun"), default="euler")
    parser.add_argument("--no-clip-sample", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set, normalizer, dataset_info = build_datasets(args)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            drop_last=False,
        )

    dino, dino_path = load_dino_for_config(args.dino_path)
    policy_config = make_policy_config(
        args,
        train_set,
        dino_path,
        dino_hidden_size=dino.config.hidden_size,
        image_keys=list(dataset_info["image_keys"]),
    )
    model = DINOFlowMatchingPolicy(policy_config, dino).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.95, 0.999),
    )
    ema = None if args.no_ema else EMAModel(model, decay=args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    total_train_steps = max(len(train_loader), 1) * args.epochs
    train_config = vars(args).copy()
    train_config.update(
        {
            "model_type": "tacex_dinov3_flow_matching_bc",
            "policy_config": policy_config.to_dict(),
            "dataset": dataset_info,
            "num_parameters_trainable": count_parameters(model),
            "train_samples": len(train_set),
            "val_samples": 0 if val_set is None else len(val_set),
            "state_dim": train_set.state_dim,
            "action_dim": train_set.action_dim,
            "image_shape": train_set.image_shape,
            "num_cameras": train_set.num_cameras,
        }
    )
    (output_dir / "config.json").write_text(json.dumps(json_ready(train_config), indent=2), encoding="utf-8")
    print(
        json.dumps(
            json_ready(
                {
                    "model_type": train_config["model_type"],
                    "dataset": dataset_info,
                    "policy_config": policy_config.to_dict(),
                    "num_parameters_trainable": train_config["num_parameters_trainable"],
                    "train_samples": len(train_set),
                    "val_samples": 0 if val_set is None else len(val_set),
                }
            ),
            indent=2,
        )
    )

    best_val = float("inf")
    global_step = 0
    log_path = output_dir / "logs.jsonl"
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_metrics, global_step = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            ema=ema,
            scaler=scaler,
            train=True,
            epoch=epoch,
            max_steps=1 if args.debug else args.max_train_steps,
            base_lr=args.lr,
            total_steps=total_train_steps,
            warmup_steps=args.warmup_steps,
            global_step=global_step,
            grad_clip=args.grad_clip,
            amp=args.amp and device.type == "cuda",
        )
        val_metrics = None
        eval_model = ema.averaged_model if ema is not None else model
        if val_loader is not None:
            val_metrics, _ = run_epoch(
                eval_model,
                val_loader,
                device,
                train=False,
                epoch=epoch,
                max_steps=1 if args.debug else args.max_val_steps,
                amp=args.amp and device.type == "cuda",
            )
        elapsed = time.time() - start_time
        metric = train_metrics["loss"] if val_metrics is None else val_metrics["loss"]
        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_sec": elapsed,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_ready(entry)) + "\n")

        print(
            f"epoch {epoch:04d} done train_loss={train_metrics['loss']:.6f}"
            + ("" if val_metrics is None else f" val_loss={val_metrics['loss']:.6f}")
            + f" time={elapsed:.1f}s"
        )
        save_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            epoch,
            global_step,
            normalizer.state_dict(),
            train_config,
            ema=ema,
        )
        if metric < best_val:
            best_val = metric
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                normalizer.state_dict(),
                train_config,
                ema=ema,
            )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                global_step,
                normalizer.state_dict(),
                train_config,
                ema=ema,
            )
        if args.debug:
            break

    if args.debug:
        runner = FlowMatchingBCRunner(output_dir / "best.pt", device=str(device), use_ema=ema is not None, num_inference_steps=2, seed=args.seed)
        sample = train_set[0]
        state_np = normalizer.unnormalize_numpy(STATE_KEY, sample["obs"][STATE_KEY].numpy())[-1]
        image_np = (sample["obs"]["images"].numpy()[-1].transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
        action_chunk = runner.predict_action_chunk(state_np, image_np)
        if not np.isfinite(action_chunk).all():
            raise RuntimeError("Reloaded policy produced non-finite actions.")
        print(f"debug reload action_chunk shape={action_chunk.shape} finite={np.isfinite(action_chunk).all()}")


if __name__ == "__main__":
    main()
