from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from sim_robot.common.normalizer import ArrayStats, LinearNormalizer


@dataclass(frozen=True)
class EpisodeInfo:
    name: str
    index: int
    length: int


def _demo_index(name: str) -> int:
    return int(name.split("_")[-1])


def normalize_image_keys(image_key: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(image_key, str):
        keys = tuple(part.strip() for part in image_key.split(",") if part.strip())
    else:
        keys = tuple(str(part).strip() for part in image_key if str(part).strip())
    if not keys:
        raise ValueError("At least one image key is required.")
    return keys


def demonstration_mode_value(demo: h5py.Group) -> float:
    mode = demo.attrs.get("demonstration_mode", "safe")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8")
    return {
        "position_failure": -1.0,
        "safe": 0.0,
        "overforce": 1.0,
    }.get(str(mode), 0.0)


def demonstration_mode_name(demo: h5py.Group) -> str:
    mode = demo.attrs.get("demonstration_mode", "safe")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8")
    return str(mode)


def _read_episode_length(
    demo: h5py.Group, action_key: str, state_key: str, image_key: str | tuple[str, ...] | list[str]
) -> int:
    state_len = int(demo["obs"][state_key].shape[0])
    image_keys = normalize_image_keys(image_key)
    image_len = min(int(demo["obs"][key].shape[0]) for key in image_keys)
    action_len = int(demo["actions"][action_key].shape[0])
    attr_key = f"length_{action_key}"
    attr_len = int(demo.attrs[attr_key]) if attr_key in demo.attrs else action_len
    return min(state_len, image_len, action_len, attr_len)


def list_episodes(
    hdf5_path: str | Path,
    action_key: str = "high",
    state_key: str = "robot0_pos",
    image_key: str = "robot0_image",
    success_only: bool = False,
) -> list[EpisodeInfo]:
    with h5py.File(hdf5_path, "r") as f:
        demos = f["data"]
        episodes = []
        for key in sorted(demos.keys(), key=_demo_index):
            demo = demos[key]
            if success_only and not bool(demo.attrs.get("success", True)):
                continue
            index = _demo_index(key)
            length = _read_episode_length(demo, action_key=action_key, state_key=state_key, image_key=image_key)
            episodes.append(EpisodeInfo(name=key, index=index, length=length))
        return episodes


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


def compute_normalizer(
    hdf5_path: str | Path,
    episode_ids: np.ndarray,
    action_key: str = "high",
    state_key: str = "robot0_pos",
    image_key: str = "robot0_image",
    mode: str = "limits",
    include_phase: bool = False,
    include_demo_mode: bool = False,
    safe_close_width_m: float | None = None,
    overforce_close_width_m: float | None = None,
) -> LinearNormalizer:
    pos_parts = []
    action_parts = []
    with h5py.File(hdf5_path, "r") as f:
        demos = f["data"]
        for episode_id in episode_ids:
            demo = demos[f"demo_{int(episode_id)}"]
            length = _read_episode_length(demo, action_key=action_key, state_key=state_key, image_key=image_key)
            pos = demo["obs"][state_key][:length]
            if include_phase:
                phase = np.linspace(0.0, 1.0, num=length, dtype=np.float32)[:, None]
                pos = np.concatenate((pos, phase), axis=-1)
            if include_demo_mode:
                mode_feature = np.full(
                    (length, 1), demonstration_mode_value(demo), dtype=np.float32
                )
                pos = np.concatenate((pos, mode_feature), axis=-1)
            pos_parts.append(pos)
            actions = demo["actions"][action_key][:length]
            if safe_close_width_m is not None and demonstration_mode_name(demo) == "safe":
                actions = np.asarray(actions).copy()
                closing = actions[:, -1] < 0.02
                actions[closing, -1] = np.minimum(actions[closing, -1], float(safe_close_width_m))
            if (
                overforce_close_width_m is not None
                and demonstration_mode_name(demo) == "overforce"
            ):
                actions = np.asarray(actions).copy()
                closing = actions[:, -1] < 0.02
                actions[closing, -1] = np.minimum(
                    actions[closing, -1], float(overforce_close_width_m)
                )
            action_parts.append(actions)
    stats = {
        "robot0_pos": ArrayStats.from_array(np.concatenate(pos_parts, axis=0)),
        "action": ArrayStats.from_array(np.concatenate(action_parts, axis=0)),
    }
    return LinearNormalizer(stats=stats, mode=mode)


class SimRobotHDF5SequenceDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str | Path,
        episode_ids: np.ndarray,
        normalizer: LinearNormalizer,
        n_state_obs_steps: int = 2,
        n_image_obs_steps: int = 2,
        n_action_steps: int = 32,
        action_key: str = "high",
        state_key: str = "robot0_pos",
        image_key: str = "robot0_image",
        cache_images: bool = False,
        include_phase: bool = False,
        include_demo_mode: bool = False,
        safe_close_width_m: float | None = None,
        overforce_close_width_m: float | None = None,
    ) -> None:
        super().__init__()
        self.hdf5_path = str(hdf5_path)
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self.normalizer = normalizer
        self.n_state_obs_steps = int(n_state_obs_steps)
        self.n_image_obs_steps = int(n_image_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.action_key = action_key
        self.state_key = state_key
        self.image_keys = normalize_image_keys(image_key)
        self.image_key = self.image_keys[0]
        self.cache_images = cache_images
        self.include_phase = bool(include_phase)
        self.include_demo_mode = bool(include_demo_mode)
        self.safe_close_width_m = None if safe_close_width_m is None else float(safe_close_width_m)
        self.overforce_close_width_m = (
            None if overforce_close_width_m is None else float(overforce_close_width_m)
        )
        self._file: h5py.File | None = None
        self._image_cache: dict[tuple[int, int, str], np.ndarray] = {}

        for name, value in {
            "n_state_obs_steps": self.n_state_obs_steps,
            "n_image_obs_steps": self.n_image_obs_steps,
            "n_action_steps": self.n_action_steps,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1")

        with h5py.File(self.hdf5_path, "r") as f:
            self.episodes = [
                EpisodeInfo(
                    name=f"demo_{int(ep)}",
                    index=int(ep),
                    length=_read_episode_length(
                        f["data"][f"demo_{int(ep)}"],
                        action_key=action_key,
                        state_key=state_key,
                        image_key=self.image_keys,
                    ),
                )
                for ep in self.episode_ids
            ]
            if not self.episodes:
                raise ValueError("No episodes selected.")
            self.episode_modes = {
                episode.index: demonstration_mode_name(f["data"][episode.name])
                for episode in self.episodes
            }
            first = f["data"][self.episodes[0].name]
            self.robot0_pos_dim = (
                int(first["obs"][state_key].shape[-1])
                + int(self.include_phase)
                + int(self.include_demo_mode)
            )
            self.action_dim = int(first["actions"][action_key].shape[-1])
            self.image_shape = tuple(first["obs"][self.image_keys[0]].shape[1:])
            self.image_shapes = {key: tuple(first["obs"][key].shape[1:]) for key in self.image_keys}
            self.freq_ratio = int(f.attrs.get("freq_ratio", first.attrs.get("freq_ratio", 1)))
            self.instruction = f.attrs.get("instruction", first.attrs.get("instruction", ""))
            self.labware = f.attrs.get("labware", first.attrs.get("labware", ""))

        self.samples: list[tuple[int, int, int]] = []
        for episode in self.episodes:
            for t in range(episode.length):
                self.samples.append((episode.index, episode.length, t))

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_image_cache"] = {}
        return state

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __len__(self) -> int:
        return len(self.samples)

    def sample_weights(
        self,
        safe_weight: float = 1.0,
        overforce_weight: float = 1.0,
        position_failure_weight: float = 1.0,
        safe_close_phase: float = 0.4,
        safe_close_phase_weight: float = 1.0,
    ) -> torch.DoubleTensor:
        weights = {
            "safe": float(safe_weight),
            "overforce": float(overforce_weight),
            "position_failure": float(position_failure_weight),
        }
        if any(value <= 0.0 for value in weights.values()):
            raise ValueError("All demonstration mode sample weights must be positive.")
        if safe_close_phase_weight <= 0.0:
            raise ValueError("safe_close_phase_weight must be positive.")
        return torch.tensor(
            [weights[self.episode_modes[episode_id]] * (float(safe_close_phase_weight) if self.episode_modes[episode_id] == "safe" and t / max(length - 1, 1) >= float(safe_close_phase) else 1.0) for episode_id, length, t in self.samples],
            dtype=torch.double,
        )

    @staticmethod
    def _clamp_indices(length: int, indices: np.ndarray) -> np.ndarray:
        return np.clip(indices, 0, length - 1)

    @staticmethod
    def _read_rows(dataset, indices: np.ndarray) -> np.ndarray:
        if len(indices) > 1 and np.all(np.diff(indices) > 0):
            return dataset[indices]
        return np.stack([dataset[int(i)] for i in indices], axis=0)

    def _read_images(self, demo, episode_id: int, indices: np.ndarray, image_key: str) -> np.ndarray:
        if not self.cache_images:
            return self._read_rows(demo["obs"][image_key], indices)
        parts = []
        for idx in indices:
            cache_key = (episode_id, int(idx), image_key)
            if cache_key not in self._image_cache:
                self._image_cache[cache_key] = demo["obs"][image_key][idx]
            parts.append(self._image_cache[cache_key])
        return np.stack(parts, axis=0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_id, length, t = self.samples[idx]
        demo = self.file["data"][f"demo_{episode_id}"]
        state_idx = self._clamp_indices(length, np.arange(t - self.n_state_obs_steps + 1, t + 1))
        image_idx = self._clamp_indices(length, np.arange(t - self.n_image_obs_steps + 1, t + 1))
        action_idx = self._clamp_indices(length, np.arange(t, t + self.n_action_steps))

        robot0_pos = self._read_rows(demo["obs"][self.state_key], state_idx).astype(np.float32)
        if self.include_phase:
            phase = (state_idx.astype(np.float32) / max(length - 1, 1))[:, None]
            robot0_pos = np.concatenate((robot0_pos, phase), axis=-1)
        if self.include_demo_mode:
            mode_feature = np.full(
                (len(state_idx), 1), demonstration_mode_value(demo), dtype=np.float32
            )
            robot0_pos = np.concatenate((robot0_pos, mode_feature), axis=-1)
        action = self._read_rows(demo["actions"][self.action_key], action_idx).astype(np.float32)
        if self.safe_close_width_m is not None and demonstration_mode_name(demo) == "safe":
            closing = action[:, -1] < 0.02
            action[closing, -1] = np.minimum(action[closing, -1], self.safe_close_width_m)
        if (
            self.overforce_close_width_m is not None
            and demonstration_mode_name(demo) == "overforce"
        ):
            closing = action[:, -1] < 0.02
            action[closing, -1] = np.minimum(action[closing, -1], self.overforce_close_width_m)
        robot0_pos = self.normalizer.normalize_numpy("robot0_pos", robot0_pos)
        action = self.normalizer.normalize_numpy("action", action)

        obs = {"robot0_pos": torch.from_numpy(robot0_pos)}
        for key in self.image_keys:
            image = self._read_images(demo, episode_id, image_idx, key)
            obs_key = "robot0_image" if len(self.image_keys) == 1 else key
            obs[obs_key] = torch.from_numpy(image).permute(0, 3, 1, 2).float() / 255.0
        return {
            "obs": obs,
            "action": torch.from_numpy(action),
            "episode": torch.tensor(episode_id, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
        }


def build_datasets(
    hdf5_path: str | Path,
    n_state_obs_steps: int,
    n_image_obs_steps: int,
    n_action_steps: int,
    val_ratio: float,
    seed: int,
    normalizer_mode: str = "limits",
    action_key: str = "high",
    state_key: str = "robot0_pos",
    image_key: str = "robot0_image",
    success_only: bool = False,
    cache_images: bool = False,
    include_phase: bool = False,
    include_demo_mode: bool = False,
    safe_close_width_m: float | None = None,
    overforce_close_width_m: float | None = None,
) -> tuple[SimRobotHDF5SequenceDataset, SimRobotHDF5SequenceDataset | None, LinearNormalizer]:
    image_keys = normalize_image_keys(image_key)
    episodes = list_episodes(
        hdf5_path,
        action_key=action_key,
        state_key=state_key,
        image_key=image_keys,
        success_only=success_only,
    )
    original_ids = np.asarray([episode.index for episode in episodes], dtype=np.int64)
    train_local, val_local = split_episode_indices(len(original_ids), val_ratio=val_ratio, seed=seed)
    train_ids = np.sort(original_ids[train_local])
    val_ids = np.sort(original_ids[val_local])
    normalizer = compute_normalizer(
        hdf5_path=hdf5_path,
        episode_ids=train_ids,
        action_key=action_key,
        state_key=state_key,
        image_key=image_key,
        mode=normalizer_mode,
        include_phase=include_phase,
        include_demo_mode=include_demo_mode,
        safe_close_width_m=safe_close_width_m,
        overforce_close_width_m=overforce_close_width_m,
    )
    train_set = SimRobotHDF5SequenceDataset(
        hdf5_path=hdf5_path,
        episode_ids=train_ids,
        normalizer=normalizer,
        n_state_obs_steps=n_state_obs_steps,
        n_image_obs_steps=n_image_obs_steps,
        n_action_steps=n_action_steps,
        action_key=action_key,
        state_key=state_key,
        image_key=image_keys,
        cache_images=cache_images,
        include_phase=include_phase,
        include_demo_mode=include_demo_mode,
        safe_close_width_m=safe_close_width_m,
        overforce_close_width_m=overforce_close_width_m,
    )
    val_set = None
    if len(val_ids) > 0:
        val_set = SimRobotHDF5SequenceDataset(
            hdf5_path=hdf5_path,
            episode_ids=val_ids,
            normalizer=normalizer,
            n_state_obs_steps=n_state_obs_steps,
            n_image_obs_steps=n_image_obs_steps,
            n_action_steps=n_action_steps,
            action_key=action_key,
            state_key=state_key,
            image_key=image_keys,
            cache_images=cache_images,
            include_phase=include_phase,
            include_demo_mode=include_demo_mode,
            safe_close_width_m=safe_close_width_m,
            overforce_close_width_m=overforce_close_width_m,
        )
    return train_set, val_set, normalizer
