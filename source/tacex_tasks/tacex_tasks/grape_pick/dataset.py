from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


SCHEMA_VERSION = 2


def _numpy(value: torch.Tensor, env_index: int, dtype=np.float32) -> np.ndarray:
    return value[env_index].detach().cpu().numpy().astype(dtype, copy=False)


class GrapeTrajectoryBuffer:
    """Per-environment recorder for synchronized GrapePick trajectories."""

    def __init__(self, num_envs: int):
        self.num_envs = int(num_envs)
        self.packed_steps: list[torch.Tensor] = []
        self.rgb_steps: list[torch.Tensor] = []
        self.rgb_third_steps: list[torch.Tensor] = []
        self.tactile_left_steps: list[torch.Tensor] = []
        self.tactile_right_steps: list[torch.Tensor] = []
        self.height_left_steps: list[torch.Tensor] = []
        self.height_right_steps: list[torch.Tensor] = []

    def append(
        self,
        env,
        action: torch.Tensor,
        active: torch.Tensor,
        *,
        rgb=None,
        rgb_third=None,
        tactile=None,
    ) -> None:
        cafe = env.get_cafe_observation()
        forces = env.get_contact_force_metrics()
        obstacle_contact_force_n = env.get_obstacle_contact_force()
        root_pos_w = env._robot.data.root_link_pos_w
        grape_pos_b = env.grape.data.root_pos_w - root_pos_w
        grape_vel_w = torch.cat((env.grape.data.root_lin_vel_w, env.grape.data.root_ang_vel_w), dim=-1)
        anchor_pos_b = env.stem_anchor_pos_w - root_pos_w
        packed = torch.cat(
            (
                cafe["robot0_pos"],
                cafe["robot0_force"],
                action,
                forces["left_force_w"],
                forces["right_force_w"],
                torch.stack(
                    (forces["left_normal_force_n"], forces["right_normal_force_n"]), dim=-1
                ),
                forces["grip_force_n"].unsqueeze(-1),
                obstacle_contact_force_n.unsqueeze(-1),
                env.stem_tension_n.unsqueeze(-1),
                env.stem_attached.float().unsqueeze(-1),
                grape_pos_b,
                grape_vel_w,
                anchor_pos_b,
                env.expert_phase.float().unsqueeze(-1),
                env.step_count.float().unsqueeze(-1),
            ),
            dim=-1,
        ).detach().clone()
        self.packed_steps.append(packed)
        if rgb is not None:
            self.rgb_steps.append(rgb.detach().clone())
        if rgb_third is not None:
            self.rgb_third_steps.append(rgb_third.detach().clone())
        if tactile is not None:
            self.tactile_left_steps.append(tactile[0].detach().clone())
            self.tactile_right_steps.append(tactile[1].detach().clone())
            self.height_left_steps.append(tactile[2].detach().clone())
            self.height_right_steps.append(tactile[3].detach().clone())

    def save_episode(
        self,
        env_index: int,
        output_dir: str | Path,
        *,
        success: bool,
        failure_reason: str,
        env,
        seed: int,
        sample_period_s: float | None = None,
    ) -> Path:
        if not self.packed_steps:
            raise ValueError(f"Environment {env_index} has no recorded trajectory samples.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        packed = torch.stack(self.packed_steps, dim=0)[:, env_index].cpu().numpy().astype(np.float32, copy=False)
        arrays = {
            "robot0_pos": packed[:, 0:10],
            "robot0_force": packed[:, 10:16],
            "action": packed[:, 16:26],
            "left_force_w": packed[:, 26:29],
            "right_force_w": packed[:, 29:32],
            "finger_normal_force_n": packed[:, 32:34],
            "grip_force_n": packed[:, 34],
            "obstacle_contact_force_n": packed[:, 35],
            "stem_tension_n": packed[:, 36],
            "stem_attached": packed[:, 37].astype(np.uint8),
            "grape_pos_b": packed[:, 38:41],
            "grape_velocity_w": packed[:, 41:47],
            "stem_anchor_pos_b": packed[:, 47:50],
            "phase": packed[:, 50].astype(np.int16),
            "step": packed[:, 51].astype(np.int32),
        }
        if self.rgb_steps:
            arrays["rgb"] = (
                torch.stack(self.rgb_steps, dim=0)[:, env_index].cpu().numpy().astype(np.uint8, copy=False)
            )
        if self.rgb_third_steps:
            arrays["rgb_third"] = (
                torch.stack(self.rgb_third_steps, dim=0)[:, env_index]
                .cpu()
                .numpy()
                .astype(np.uint8, copy=False)
            )
        if self.tactile_left_steps:
            arrays["tactile_left_rgb"] = (
                torch.stack(self.tactile_left_steps, dim=0)[:, env_index]
                .cpu()
                .numpy()
                .astype(np.uint8, copy=False)
            )
            arrays["tactile_right_rgb"] = (
                torch.stack(self.tactile_right_steps, dim=0)[:, env_index]
                .cpu()
                .numpy()
                .astype(np.uint8, copy=False)
            )
            arrays["tactile_left_height_map"] = (
                torch.stack(self.height_left_steps, dim=0)[:, env_index]
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            arrays["tactile_right_height_map"] = (
                torch.stack(self.height_right_steps, dim=0)[:, env_index]
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
        np.savez_compressed(output_dir / "trajectory.npz", **arrays)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "task": "TacEx-GrapePick-Force-v0",
            "instruction": "pick the target grape without exceeding 5 N grip force",
            "success": bool(success),
            "failure_reason": failure_reason,
            "seed": int(seed),
            "env_index": int(env_index),
            "length": len(self.packed_steps),
            "sample_period_s": float(env.physics_dt if sample_period_s is None else sample_period_s),
            "contact_force_source": "PhysX ContactSensor filtered to target_grape",
            "grip_force_definition": "maximum left/right pad-normal contact load",
            "grip_limit_n": float(env.cfg.damage_grip_force_threshold_n),
            "detach_threshold_n": float(env.cfg.detach_force_threshold_n),
            "episode_damage_threshold_n": float(env.damage_threshold_n[env_index].item()),
            "episode_detach_threshold_n": float(env.detach_threshold_n[env_index].item()),
            "peak_grip_force_n": float(env.peak_grip_force_n[env_index].item()),
            "peak_stem_tension_n": float(env.peak_stem_tension_n[env_index].item()),
            "peak_obstacle_contact_force_n": float(
                env.peak_obstacle_contact_force_n[env_index].item()
            ),
            "grape_position_randomization_m": list(env.cfg.grape_position_randomization_m),
            "sampled_grape_position_offset_m": _numpy(
                env.grape_reset_pos_w
                - env.scene.env_origins
                - torch.tensor(env.cfg.grape_initial_pos, device=env.device),
                env_index,
            ).tolist(),
            "grape_reset_pos_w": _numpy(env.grape_reset_pos_w, env_index).tolist(),
            "grape_reset_quat_w": _numpy(env.grape_reset_quat_w, env_index).tolist(),
            "gelsight_rgb_recorded": "tactile_left_rgb" in arrays,
            "scene_rgb_recorded": "rgb_third" in arrays,
            "wrist_rgb_recorded": "rgb" in arrays,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return output_dir


def validate_episode(episode_dir: str | Path) -> dict[str, object]:
    episode_dir = Path(episode_dir)
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(episode_dir / "trajectory.npz") as trajectory:
        required = {
            "robot0_pos": 10,
            "robot0_force": 6,
            "action": 10,
            "left_force_w": 3,
            "right_force_w": 3,
            "finger_normal_force_n": 2,
            "grape_pos_b": 3,
            "grape_velocity_w": 6,
            "stem_anchor_pos_b": 3,
        }
        length = int(metadata["length"])
        for key, width in required.items():
            if key not in trajectory:
                raise ValueError(f"Missing trajectory field: {key}")
            if trajectory[key].shape != (length, width):
                raise ValueError(f"{key} has shape {trajectory[key].shape}; expected {(length, width)}")
            if not np.isfinite(trajectory[key]).all():
                raise ValueError(f"{key} contains non-finite values")
        for key in ("grip_force_n", "stem_tension_n", "stem_attached", "phase", "step"):
            if trajectory[key].shape != (length,):
                raise ValueError(f"{key} has shape {trajectory[key].shape}; expected {(length,)}")
        if int(metadata.get("schema_version", 1)) >= 2:
            key = "obstacle_contact_force_n"
            if key not in trajectory or trajectory[key].shape != (length,):
                raise ValueError(f"{key} must have shape {(length,)} in schema version 2")
            if not np.isfinite(trajectory[key]).all():
                raise ValueError(f"{key} contains non-finite values")
            if bool(metadata["success"]) and float(np.max(trajectory[key])) >= 0.25:
                raise ValueError("Successful episode contains a robot-to-vine/trellis collision")
        measured_peak = float(np.max(trajectory["grip_force_n"]))
        if bool(metadata["success"]) and measured_peak > float(metadata["grip_limit_n"]) + 1.0e-4:
            raise ValueError("Successful episode violates its grip-force limit")
    return metadata
