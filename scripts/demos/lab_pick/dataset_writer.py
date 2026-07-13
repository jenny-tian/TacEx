from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class LabPickHdf5Writer:
    FREQ_RATIO = 3

    def __init__(self, dataset_file: str | Path, labware: str, instruction: str, task_id: int):
        self.dataset_file = Path(dataset_file)
        self.dataset_file.parent.mkdir(parents=True, exist_ok=True)
        if self.dataset_file.exists():
            self.dataset_file.unlink()
        self.labware = labware
        self.instruction = instruction
        self.task_id = task_id
        self.episode_index = 0
        self.reset_episode()

    def reset_episode(self):
        self.action = []
        self.low_action = []
        self.robot0_pos = []
        self.robot0_force = []
        self.task_id_frame = []
        self.robot0_image = []

    def append_high(self, env):
        obs = env.cafe_observation()
        action = env.action_vector()
        self.action.append(action[0].detach().cpu().numpy().astype(np.float32))
        self.robot0_pos.append(obs["robot0_pos"][0].detach().cpu().numpy().astype(np.float32))
        self.robot0_force.append(obs["robot0_force"][0].detach().cpu().numpy().astype(np.float32))
        self.task_id_frame.append(np.array([self.task_id], dtype=np.int64))

    def append_low(self, env):
        action = env.action_vector()
        self.low_action.append(action[0].detach().cpu().numpy().astype(np.float32))
        self.robot0_image.append(env.wrist_image_224()[0].detach().cpu().numpy().astype(np.uint8))

    def write_episode(self, env, success: bool):
        length_high = min(len(self.action), len(self.robot0_pos), len(self.robot0_force))
        length_low = min(len(self.low_action), len(self.robot0_image))
        if length_high == 0 or length_low == 0:
            return

        with h5py.File(self.dataset_file, "a") as h5:
            h5.attrs["num_demos"] = self.episode_index + 1
            h5.attrs["labware"] = self.labware
            h5.attrs["instruction"] = self.instruction
            h5.attrs["task_id"] = self.task_id
            h5.attrs["include_images"] = True
            h5.attrs["freq_ratio"] = self.FREQ_RATIO
            h5.attrs["high_freq_obs_keys"] = "robot0_pos,robot0_force"
            h5.attrs["low_freq_obs_keys"] = "robot0_image"
            h5.attrs["high_freq_action_key"] = "high"
            h5.attrs["low_freq_action_key"] = "low"

            data_group = h5.require_group("data")
            demo = data_group.create_group(f"demo_{self.episode_index}")
            actions_group = demo.create_group("actions")
            actions_group.create_dataset("high", data=np.stack(self.action[:length_high], axis=0), dtype=np.float32)
            actions_group.create_dataset("low", data=np.stack(self.low_action[:length_low], axis=0), dtype=np.float32)

            obs_group = demo.create_group("obs")
            obs_group.create_dataset("robot0_pos", data=np.stack(self.robot0_pos[:length_high], axis=0), dtype=np.float32)
            obs_group.create_dataset("robot0_force", data=np.stack(self.robot0_force[:length_high], axis=0), dtype=np.float32)
            obs_group.create_dataset("robot0_image", data=np.stack(self.robot0_image[:length_low], axis=0), dtype=np.uint8)

            demo.attrs["success"] = bool(success)
            demo.attrs["instruction"] = self.instruction
            demo.attrs["task_id"] = self.task_id
            demo.attrs["length_high"] = length_high
            demo.attrs["length_low"] = length_low
            demo.attrs["freq_ratio"] = self.FREQ_RATIO
            demo.attrs["labware_reset_pos_w"] = env.labware_reset_pos_w[0].detach().cpu().numpy().astype(np.float32)
            demo.attrs["labware_reset_quat_w"] = env.labware_reset_quat_w[0].detach().cpu().numpy().astype(np.float32)

        self.episode_index += 1
        self.reset_episode()


class CafeRecordWriter:
    """Writer for ForceCapture-CAFE style raw record directories."""

    def __init__(self, record_dir: str | Path):
        self.record_dir = Path(record_dir)
        self.reset_episode()

    def reset_episode(self):
        self.samples = []

    def append_sample(self, timestamp: float, sample: dict[str, np.ndarray]):
        marker2d = np.asarray(sample["marker2d"], dtype=np.float32)
        self.samples.append(
            {
                "timestamp": float(timestamp),
                "xyz": np.asarray(sample["xyz"], dtype=np.float32).reshape(3),
                "quat": np.asarray(sample["quat"], dtype=np.float32).reshape(4),
                "width": np.asarray(sample["width"], dtype=np.float32).reshape(1),
                "ft": np.asarray(sample["ft"], dtype=np.float32).reshape(6),
                "marker2d": marker2d.reshape(-1),
                "marker2d_grid": marker2d.reshape(14, 26, 2),
                "rgb": np.asarray(sample["rgb"], dtype=np.uint8),
                "action": np.asarray(sample["action"], dtype=np.float32).reshape(-1),
            }
        )

    def flush_episode(
        self,
        *,
        success: bool,
        labware_reset_pos_w: np.ndarray,
        labware_reset_quat_w: np.ndarray,
        force_source: str,
        marker2d_source: str = "placeholder_zero",
        force_frame: str = "world",
    ) -> bool:
        if not self.samples:
            self.reset_episode()
            return False

        self._mkdirs()
        timestamps = np.asarray([sample["timestamp"] for sample in self.samples], dtype=np.float64)
        xyz = self._stack("xyz", np.float32)
        quat = self._stack("quat", np.float32)
        width = self._stack("width", np.float32)
        ft = self._stack("ft", np.float32)
        marker2d = self._stack("marker2d", np.float32)
        marker2d_grid = self._stack("marker2d_grid", np.float32)
        rgb = self._stack("rgb", np.uint8)
        action = self._stack("action", np.float32)

        aligned = self.record_dir / "aligned_60Hz"
        np.save(aligned / "timestamps.npy", timestamps)
        np.save(aligned / "xyz.npy", xyz)
        np.save(aligned / "quat.npy", quat)
        np.save(aligned / "width.npy", width)
        np.save(aligned / "ft.npy", ft)
        np.save(aligned / "marker2d.npy", marker2d)
        np.save(aligned / "rgb.npy", rgb)
        np.save(aligned / "action.npy", action)

        np.save(self.record_dir / "encoder" / "timestamps.npy", timestamps)
        np.save(self.record_dir / "encoder" / "width.npy", width)
        np.save(self.record_dir / "tracker" / "timestamps.npy", timestamps)
        np.save(self.record_dir / "tracker" / "xyz.npy", xyz)
        np.save(self.record_dir / "tracker" / "quat.npy", quat)
        np.save(self.record_dir / "ftsensor" / "timestamps.npy", timestamps)
        np.save(self.record_dir / "ftsensor" / "ft.npy", ft)
        np.save(self.record_dir / "ftsensor" / "ft_compensated.npy", ft.copy())
        np.save(self.record_dir / "xense" / "timestamps.npy", timestamps)
        np.save(self.record_dir / "xense" / "marker2d.npy", marker2d_grid)
        np.save(self.record_dir / "xense" / "marker2d_flatten.npy", marker2d)
        np.save(self.record_dir / "camera" / "color" / "timestamps.npy", timestamps)
        np.save(self.record_dir / "camera" / "color" / "rgb.npy", rgb)
        np.savez(
            self.record_dir / "metadata.npz",
            success=np.asarray(success, dtype=np.bool_),
            labware_reset_pos_w=np.asarray(labware_reset_pos_w, dtype=np.float32).reshape(3),
            labware_reset_quat_w=np.asarray(labware_reset_quat_w, dtype=np.float32).reshape(4),
            force_source=np.asarray(force_source),
            force_frame=np.asarray(force_frame),
            marker2d_source=np.asarray(marker2d_source),
        )
        self.reset_episode()
        return True

    def _mkdirs(self):
        for relative in ("aligned_60Hz", "encoder", "tracker", "ftsensor", "xense", "camera/color"):
            (self.record_dir / relative).mkdir(parents=True, exist_ok=True)

    def _stack(self, key: str, dtype: np.dtype) -> np.ndarray:
        return np.stack([sample[key] for sample in self.samples], axis=0).astype(dtype, copy=False)
