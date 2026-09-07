"""Trajectory serialization and validation for SealTwist demonstrations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


SCHEMA_VERSION = 3


def force_vectors_to_maps(force_w: np.ndarray, size: int = 32) -> np.ndarray:
    """Encode each contact-force magnitude as a normalized Gaussian force map.

    The map is a compact tactile-learning proxy rather than a pressure value in
    pascals.  Its pixel sum is exactly the corresponding force magnitude in N.
    """

    coordinates = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(coordinates, coordinates, indexing="ij")
    kernel = np.exp(-(grid_x**2 + grid_y**2) / (2.0 * 0.28**2)).astype(np.float32)
    kernel /= np.sum(kernel)
    magnitude_n = np.linalg.norm(force_w, axis=-1).astype(np.float32)
    return magnitude_n[:, None, None] * kernel[None, :, :]


class SealTwistTrajectoryBuffer:
    """Accumulate a single vectorized rollout without synchronizing each step."""

    def __init__(self):
        self.steps: list[torch.Tensor] = []
        self.rgb_third: list[torch.Tensor] = []
        self.rgb_wrist: list[torch.Tensor] = []
        self.tactile_left: list[torch.Tensor] = []
        self.tactile_right: list[torch.Tensor] = []

    def append(self, env, action: torch.Tensor, *, save_rgb: bool, save_tactile: bool):
        cafe = env.get_cafe_observation()
        flags = env.get_task_flags()
        left_w = env._contact_force_from_sensor(env.left_finger_contact_sensor)
        right_w = env._contact_force_from_sensor(env.right_finger_contact_sensor)
        cap_pose_b = env.labware.data.root_state_w[:, :7].clone()
        cap_pose_b[:, :3] -= env._robot.data.root_link_pos_w
        packed = torch.cat(
            (
                cafe["robot0_pos"],
                env.get_task_wrench(),
                action,
                left_w,
                right_w,
                torch.stack(
                    (flags["left_finger_force_n"], flags["right_finger_force_n"]), dim=-1
                ),
                cap_pose_b,
                torch.stack(
                    (
                        flags["turn_angle_rad"],
                        flags["turn_rate_rad_s"],
                        flags["seal_torque_nm"],
                        flags["seal_compression_m"],
                        env.expert_phase.float(),
                        env.step_count.float(),
                    ),
                    dim=-1,
                ),
            ),
            dim=-1,
        ).detach().clone()
        self.steps.append(packed)
        if save_rgb:
            self.rgb_third.append(env.get_camera_rgb("third"))
            self.rgb_wrist.append(env.get_camera_rgb("wrist"))
        if save_tactile:
            self.tactile_left.append(env.get_tactile_rgb("left"))
            self.tactile_right.append(env.get_tactile_rgb("right"))

    def save(
        self,
        output_dir: Path,
        env,
        *,
        seed: int,
        success: bool,
        sample_period_s: float,
        env_index: int = 0,
        failure_reason: str = "",
    ) -> Path:
        if not self.steps:
            raise ValueError("No SealTwist trajectory samples were recorded")
        output_dir.mkdir(parents=True, exist_ok=False)
        if not 0 <= env_index < env.num_envs:
            raise IndexError(f"env_index {env_index} is outside [0, {env.num_envs})")
        packed = (
            torch.stack(self.steps, dim=0)[:, env_index]
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        arrays: dict[str, np.ndarray] = {
            "robot0_pos": packed[:, 0:10],
            "wrench": packed[:, 10:16],
            "action": packed[:, 16:26],
            "left_force_w": packed[:, 26:29],
            "right_force_w": packed[:, 29:32],
            "finger_force_n": packed[:, 32:34],
            "cap_pose_b": packed[:, 34:41],
            "turn_angle_rad": packed[:, 41],
            "turn_rate_rad_s": packed[:, 42],
            "seal_torque_nm": packed[:, 43],
            "seal_compression_m": packed[:, 44],
            "phase": packed[:, 45].astype(np.int16),
            "step": packed[:, 46].astype(np.int32),
        }
        arrays["tactile_force_map_left_n"] = force_vectors_to_maps(arrays["left_force_w"])
        arrays["tactile_force_map_right_n"] = force_vectors_to_maps(arrays["right_force_w"])
        if self.rgb_third:
            arrays["rgb_third"] = (
                torch.stack(self.rgb_third)[:, env_index].cpu().numpy().astype(np.uint8)
            )
            arrays["rgb_wrist"] = (
                torch.stack(self.rgb_wrist)[:, env_index].cpu().numpy().astype(np.uint8)
            )
        if self.tactile_left:
            arrays["tactile_left_rgb"] = (
                torch.stack(self.tactile_left)[:, env_index].cpu().numpy().astype(np.uint8)
            )
            arrays["tactile_right_rgb"] = (
                torch.stack(self.tactile_right)[:, env_index].cpu().numpy().astype(np.uint8)
            )
        np.savez_compressed(output_dir / "trajectory.npz", **arrays)

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "task": "TacEx-SealTwist-Torque-v0",
            "instruction": "tighten the cap and stop inside the seal torque window",
            "success": bool(success),
            "failure_reason": str(failure_reason),
            "seed": int(seed),
            "env_index": int(env_index),
            "length": len(self.steps),
            "sample_period_s": float(sample_period_s),
            "torque_window_nm": list(env.cfg.torque_window_nm),
            "expert_target_torque_nm": float(
                env.episode_expert_target_torque_nm[env_index].item()
            ),
            "expert_grasp_width_m": float(
                env.episode_expert_grasp_width_m[env_index].item()
            ),
            "minimum_force_threshold_n": float(env.minimum_force_threshold_n[env_index].item()),
            "maximum_force_threshold_n": float(env.maximum_force_threshold_n[env_index].item()),
            "alignment_tolerance_m": float(env.cfg.alignment_tolerance_m),
            "final_alignment_error_m": float(env.get_alignment_error_m()[env_index].item()),
            "sampled_cap_position_offset_m": env.sampled_cap_position_offset_m[env_index]
            .detach()
            .cpu()
            .tolist(),
            "peak_seal_torque_nm": float(env.peak_torque_nm[env_index].item()),
            "peak_left_finger_force_n": float(env.peak_left_force_n[env_index].item()),
            "peak_right_finger_force_n": float(env.peak_right_force_n[env_index].item()),
            "final_turn_angle_rad": float(env.turn_angle_rad[env_index].item()),
            "final_seal_compression_m": float(env.seal_compression_m[env_index].item()),
            "torque_source": "equivalent screw/seal reaction model",
            "tactile_force_source": "PhysX ContactSensor filtered to cap",
            "scene_rgb_recorded": "rgb_third" in arrays,
            "wrist_rgb_recorded": "rgb_wrist" in arrays,
            "gelsight_rgb_recorded": "tactile_left_rgb" in arrays,
            "gelsight_rgb_dynamic": bool(
                "tactile_left_rgb" in arrays
                and (
                    np.any(arrays["tactile_left_rgb"] != arrays["tactile_left_rgb"][0])
                    or np.any(arrays["tactile_right_rgb"] != arrays["tactile_right_rgb"][0])
                )
            ),
            "tactile_force_map_source": "normalized Gaussian map whose pixel sum equals fingertip force magnitude in N",
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return output_dir


def validate_episode(output_dir: Path) -> dict[str, object]:
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(output_dir / "trajectory.npz") as data:
        length = int(metadata["length"])
        widths = {
            "robot0_pos": 10,
            "wrench": 6,
            "action": 10,
            "left_force_w": 3,
            "right_force_w": 3,
            "finger_force_n": 2,
            "cap_pose_b": 7,
        }
        for key, width in widths.items():
            if key not in data or data[key].shape != (length, width):
                actual = None if key not in data else data[key].shape
                raise ValueError(f"{key} has shape {actual}, expected {(length, width)}")
            if not np.isfinite(data[key]).all():
                raise ValueError(f"{key} contains non-finite values")
        for key in ("turn_angle_rad", "turn_rate_rad_s", "seal_torque_nm", "seal_compression_m", "phase", "step"):
            if key not in data or data[key].shape != (length,):
                raise ValueError(f"{key} must have shape {(length,)}")
        for key, force_key in (
            ("tactile_force_map_left_n", "left_force_w"),
            ("tactile_force_map_right_n", "right_force_w"),
        ):
            if key not in data or data[key].shape != (length, 32, 32):
                raise ValueError(f"{key} must have shape {(length, 32, 32)}")
            map_force = data[key].sum(axis=(1, 2))
            vector_force = np.linalg.norm(data[force_key], axis=-1)
            if not np.allclose(map_force, vector_force, rtol=1.0e-5, atol=1.0e-6):
                raise ValueError(f"{key} does not preserve fingertip force magnitude")
        lower, upper = metadata["torque_window_nm"]
        if bool(metadata["success"]):
            final_torque = float(data["seal_torque_nm"][-1])
            if not lower <= final_torque <= upper:
                raise ValueError("Successful trajectory ends outside its torque window")
            if int(metadata.get("schema_version", 1)) >= 3:
                final_forces = data["finger_force_n"][-1]
                if float(np.min(final_forces)) < float(metadata["minimum_force_threshold_n"]):
                    raise ValueError("Successful trajectory ends below its minimum force threshold")
                if float(np.max(final_forces)) > float(metadata["maximum_force_threshold_n"]):
                    raise ValueError("Successful trajectory ends above its maximum force threshold")
                if float(metadata["final_alignment_error_m"]) > float(
                    metadata["alignment_tolerance_m"]
                ):
                    raise ValueError("Successful trajectory ends outside its alignment tolerance")
    return metadata
