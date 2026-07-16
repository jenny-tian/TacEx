from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch


class CafeHdf5Writer:
    """Writer for ForceCapture-CAFE compatible multi-frequency HDF5 datasets."""

    def __init__(self, dataset_file: str | Path, freq_ratio: int = 3, include_marker: bool = False):
        self.dataset_file = Path(dataset_file)
        self.dataset_file.parent.mkdir(parents=True, exist_ok=True)
        self.freq_ratio = freq_ratio
        self.include_marker = include_marker
        self.episode_index = 0
        if self.dataset_file.exists():
            with h5py.File(self.dataset_file, "r") as h5:
                self.episode_index = int(h5.attrs.get("num_demos", 0))
                data_group = h5.get("data")
                if data_group is not None:
                    demo_indices = [
                        int(name.split("_", maxsplit=1)[1])
                        for name in data_group.keys()
                        if name.startswith("demo_") and name.split("_", maxsplit=1)[1].isdigit()
                    ]
                    if demo_indices:
                        self.episode_index = max(self.episode_index, max(demo_indices) + 1)
        self.high_pos: list[np.ndarray] = []
        self.high_force: list[np.ndarray] = []
        self.high_action: list[np.ndarray] = []
        self.low_image: list[np.ndarray] = []
        self.low_action: list[np.ndarray] = []
        self.high_marker2d: list[np.ndarray] = []

    def append_high_step(self, obs: dict[str, torch.Tensor], action: torch.Tensor):
        self.high_pos.append(obs["robot0_pos"][0].detach().cpu().numpy().astype(np.float32))
        self.high_force.append(obs["robot0_force"][0].detach().cpu().numpy().astype(np.float32))
        self.high_action.append(action[0].detach().cpu().numpy().astype(np.float32))
        if self.include_marker:
            self.high_marker2d.append(np.zeros((728,), dtype=np.float32))

    def append_low_step(self, image: torch.Tensor, action: torch.Tensor):
        self.low_image.append(image[0].detach().cpu().numpy().astype(np.uint8))
        self.low_action.append(action[0].detach().cpu().numpy().astype(np.float32))

    def flush_episode(
        self,
        *,
        success: bool,
        labware_reset_pos_w: torch.Tensor,
        labware_reset_quat_w: torch.Tensor,
        success_only: bool,
    ):
        if success_only and not success:
            self.clear_episode()
            return False
        if not self.high_action or not self.low_action:
            self.clear_episode()
            return False

        n_low = len(self.low_action)
        n_high = min(len(self.high_action), n_low * self.freq_ratio)
        n_low = n_high // self.freq_ratio
        n_high = n_low * self.freq_ratio
        if n_high == 0 or n_low == 0:
            self.clear_episode()
            return False

        high_pos = self.high_pos[:n_high]
        high_force = self.high_force[:n_high]
        high_action = self.high_action[:n_high]
        low_image = self.low_image[:n_low]
        low_action = self.low_action[:n_low]
        high_marker2d = self.high_marker2d[:n_high]

        with h5py.File(self.dataset_file, "a") as h5:
            h5.attrs["num_demos"] = max(int(h5.attrs.get("num_demos", 0)), self.episode_index + 1)
            h5.attrs["include_images"] = True
            h5.attrs["freq_ratio"] = self.freq_ratio
            h5.attrs["high_freq_obs_keys"] = (
                "robot0_pos,robot0_force,robot0_marker2d" if self.include_marker else "robot0_pos,robot0_force"
            )
            h5.attrs["low_freq_obs_keys"] = "robot0_image"
            h5.attrs["high_freq_action_key"] = "high"
            h5.attrs["low_freq_action_key"] = "low"

            data_group = h5.require_group("data")
            demo_group = data_group.create_group(f"demo_{self.episode_index}")
            actions_group = demo_group.create_group("actions")
            actions_group.create_dataset("high", data=np.stack(high_action, axis=0), dtype=np.float32)
            actions_group.create_dataset("low", data=np.stack(low_action, axis=0), dtype=np.float32)

            obs_group = demo_group.create_group("obs")
            obs_group.create_dataset("robot0_pos", data=np.stack(high_pos, axis=0), dtype=np.float32)
            obs_group.create_dataset("robot0_force", data=np.stack(high_force, axis=0), dtype=np.float32)
            obs_group.create_dataset("robot0_image", data=np.stack(low_image, axis=0), dtype=np.uint8)
            if self.include_marker:
                obs_group.create_dataset("robot0_marker2d", data=np.stack(high_marker2d, axis=0), dtype=np.float32)

            demo_group.attrs["length_high"] = n_high
            demo_group.attrs["length_low"] = n_low
            demo_group.attrs["freq_ratio"] = self.freq_ratio
            demo_group.attrs["success"] = bool(success)
            demo_group.attrs["labware_reset_pos_w"] = labware_reset_pos_w[0].detach().cpu().numpy()
            demo_group.attrs["labware_reset_quat_w"] = labware_reset_quat_w[0].detach().cpu().numpy()

        self.episode_index += 1
        self.clear_episode()
        return True

    def clear_episode(self):
        self.high_pos.clear()
        self.high_force.clear()
        self.high_action.clear()
        self.low_image.clear()
        self.low_action.clear()
        self.high_marker2d.clear()


class CafeRecordWriter:
    """Writer for ForceCapture-CAFE style raw record directories and aligned 60Hz arrays."""

    def __init__(self, record_dir: str | Path):
        self.record_dir = Path(record_dir)
        self.aligned_samples: list[dict[str, np.ndarray | float]] = []
        self.camera_timestamps: list[float] = []
        self.camera_rgb: list[np.ndarray] = []
        self.third_camera_timestamps: list[float] = []
        self.third_camera_rgb: list[np.ndarray] = []
        self.ft_timestamps: list[float] = []
        self.ft: list[np.ndarray] = []
        self.tracker_timestamps: list[float] = []
        self.tracker_xyz: list[np.ndarray] = []
        self.tracker_quat: list[np.ndarray] = []
        self.encoder_timestamps: list[float] = []
        self.encoder_width: list[np.ndarray] = []
        self.xense_timestamps: list[float] = []
        self.xense_marker2d: list[np.ndarray] = []

    def append_aligned_sample(self, timestamp: float, sample: dict[str, np.ndarray]):
        self.aligned_samples.append(
            {
                "timestamp": float(timestamp),
                "xyz": np.asarray(sample["xyz"], dtype=np.float32).reshape(3),
                "quat": np.asarray(sample["quat"], dtype=np.float32).reshape(4),
                "width": np.asarray(sample["width"], dtype=np.float32).reshape(1),
                "ft": np.asarray(sample["ft"], dtype=np.float32).reshape(6),
                "marker2d": self._flatten_marker2d(sample["marker2d"]),
                "rgb": np.asarray(sample["rgb"], dtype=np.uint8),
                "third_rgb": np.asarray(sample["third_rgb"], dtype=np.uint8),
                "action": np.asarray(sample["action"], dtype=np.float32).reshape(-1),
            }
        )

    def append_camera_sample(self, timestamp: float, rgb: np.ndarray, third_rgb: np.ndarray):
        self.camera_timestamps.append(float(timestamp))
        self.camera_rgb.append(np.asarray(rgb, dtype=np.uint8))
        self.third_camera_timestamps.append(float(timestamp))
        self.third_camera_rgb.append(np.asarray(third_rgb, dtype=np.uint8))

    def append_ft_sample(self, timestamp: float, ft: np.ndarray):
        self.ft_timestamps.append(float(timestamp))
        self.ft.append(np.asarray(ft, dtype=np.float32).reshape(6))

    def append_tracker_sample(self, timestamp: float, xyz: np.ndarray, quat: np.ndarray):
        self.tracker_timestamps.append(float(timestamp))
        self.tracker_xyz.append(np.asarray(xyz, dtype=np.float32).reshape(3))
        self.tracker_quat.append(np.asarray(quat, dtype=np.float32).reshape(4))

    def append_encoder_sample(self, timestamp: float, width: np.ndarray):
        self.encoder_timestamps.append(float(timestamp))
        self.encoder_width.append(np.asarray(width, dtype=np.float32).reshape(1))

    def append_xense_sample(self, timestamp: float, marker2d: np.ndarray):
        self.xense_timestamps.append(float(timestamp))
        self.xense_marker2d.append(np.asarray(marker2d, dtype=np.float32))

    def flush_episode(
        self,
        *,
        success: bool,
        labware_reset_pos_w: np.ndarray,
        labware_reset_quat_w: np.ndarray,
    ):
        if not self.aligned_samples:
            self.clear_episode()
            return False

        self._mkdirs()
        aligned = self.record_dir / "aligned_60Hz"
        np.save(aligned / "timestamps.npy", self._stack_aligned("timestamp", dtype=np.float64))
        np.save(aligned / "xyz.npy", self._stack_aligned("xyz", dtype=np.float32))
        np.save(aligned / "quat.npy", self._stack_aligned("quat", dtype=np.float32))
        np.save(aligned / "width.npy", self._stack_aligned("width", dtype=np.float32))
        np.save(aligned / "ft.npy", self._stack_aligned("ft", dtype=np.float32))
        np.save(aligned / "marker2d.npy", self._stack_aligned("marker2d", dtype=np.float32))
        np.save(aligned / "rgb.npy", self._stack_aligned("rgb", dtype=np.uint8))
        np.save(aligned / "third_rgb.npy", self._stack_aligned("third_rgb", dtype=np.uint8))
        np.save(aligned / "action.npy", self._stack_aligned("action", dtype=np.float32))

        np.save(self.record_dir / "encoder" / "width.npy", self._stack_or_empty(self.encoder_width, (0, 1), np.float32))
        np.save(
            self.record_dir / "encoder" / "timestamps.npy",
            np.asarray(self.encoder_timestamps, dtype=np.float64),
        )

        np.save(self.record_dir / "tracker" / "xyz.npy", self._stack_or_empty(self.tracker_xyz, (0, 3), np.float32))
        np.save(self.record_dir / "tracker" / "quat.npy", self._stack_or_empty(self.tracker_quat, (0, 4), np.float32))
        np.save(
            self.record_dir / "tracker" / "timestamps.npy",
            np.asarray(self.tracker_timestamps, dtype=np.float64),
        )

        ft = self._stack_or_empty(self.ft, (0, 6), np.float32)
        np.save(self.record_dir / "ftsensor" / "ft.npy", ft)
        np.save(self.record_dir / "ftsensor" / "ft_compensated.npy", ft.copy())
        np.save(
            self.record_dir / "ftsensor" / "timestamps.npy",
            np.asarray(self.ft_timestamps, dtype=np.float64),
        )

        marker2d = self._stack_or_empty(self.xense_marker2d, (0, 14, 26, 2), np.float32)
        np.save(self.record_dir / "xense" / "marker2d.npy", marker2d)
        np.save(self.record_dir / "xense" / "marker2d_flatten.npy", marker2d.reshape(marker2d.shape[0], -1))
        np.save(
            self.record_dir / "xense" / "timestamps.npy",
            np.asarray(self.xense_timestamps, dtype=np.float64),
        )

        np.save(
            self.record_dir / "camera" / "color" / "timestamps.npy",
            np.asarray(self.camera_timestamps, dtype=np.float64),
        )
        if self.camera_rgb:
            np.save(self.record_dir / "camera" / "color" / "rgb.npy", np.stack(self.camera_rgb, axis=0))
        else:
            np.save(self.record_dir / "camera" / "color" / "rgb.npy", np.zeros((0, 480, 640, 3), dtype=np.uint8))

        third_color = self.record_dir / "camera" / "third" / "color"
        np.save(third_color / "timestamps.npy", np.asarray(self.third_camera_timestamps, dtype=np.float64))
        if self.third_camera_rgb:
            np.save(third_color / "rgb.npy", np.stack(self.third_camera_rgb, axis=0))
        else:
            np.save(third_color / "rgb.npy", np.zeros((0, 720, 1280, 3), dtype=np.uint8))

        np.savez(
            self.record_dir / "metadata.npz",
            success=np.asarray(success, dtype=np.bool_),
            labware_reset_pos_w=np.asarray(labware_reset_pos_w, dtype=np.float32).reshape(3),
            labware_reset_quat_w=np.asarray(labware_reset_quat_w, dtype=np.float32).reshape(4),
        )

        self.clear_episode()
        return True

    def clear_episode(self):
        self.aligned_samples.clear()
        self.camera_timestamps.clear()
        self.camera_rgb.clear()
        self.third_camera_timestamps.clear()
        self.third_camera_rgb.clear()
        self.ft_timestamps.clear()
        self.ft.clear()
        self.tracker_timestamps.clear()
        self.tracker_xyz.clear()
        self.tracker_quat.clear()
        self.encoder_timestamps.clear()
        self.encoder_width.clear()
        self.xense_timestamps.clear()
        self.xense_marker2d.clear()

    def _mkdirs(self):
        for relative_path in (
            "encoder",
            "tracker",
            "ftsensor",
            "xense",
            "camera/color",
            "camera/third/color",
            "aligned_60Hz",
        ):
            (self.record_dir / relative_path).mkdir(parents=True, exist_ok=True)

    def _stack_aligned(self, key: str, dtype: np.dtype):
        values = [sample[key] for sample in self.aligned_samples]
        if key == "timestamp":
            return np.asarray(values, dtype=dtype)
        return np.stack(values, axis=0).astype(dtype, copy=False)

    def _flatten_marker2d(self, marker2d: np.ndarray) -> np.ndarray:
        marker = np.asarray(marker2d, dtype=np.float32)
        return marker.reshape(-1)

    def _stack_or_empty(self, values: list[np.ndarray], shape: tuple[int, ...], dtype: np.dtype):
        if not values:
            return np.zeros(shape, dtype=dtype)
        return np.stack(values, axis=0).astype(dtype, copy=False)
