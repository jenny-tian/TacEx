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
