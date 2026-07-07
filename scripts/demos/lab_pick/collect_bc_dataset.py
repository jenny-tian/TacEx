from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect ForceCapture-CAFE style LabPick records.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_demos", type=int, default=100)
parser.add_argument("--labware", choices=("slide", "coverslip", "cup"), default="slide")
parser.add_argument("--record_dir", type=str, default="/home/tjx/TacEx/datasets/lab_pick_slide_cafe_records")
parser.add_argument("--dataset_file", type=str, default="", help="Deprecated alias; parent directory is used.")
parser.add_argument("--success_only", action="store_true")
parser.add_argument("--max_episode_steps", type=int, default=960)
parser.add_argument("--aligned_hz", type=float, default=60.0)
parser.add_argument("--camera_hz", type=float, default=30.0)
parser.add_argument("--ft_hz", type=float, default=90.0)
parser.add_argument("--tracker_hz", type=float, default=300.0)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from tacex_tasks.lab_pick.bc_dataset import CafeRecordWriter
from tacex_tasks.lab_pick.lab_pick_env import LabPickEnv
from tacex_tasks.lab_pick.lab_pick_env_cfg import LabPickEnvCfg


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor[0].detach().cpu().numpy()


def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)


def _make_cafe_sample(env: LabPickEnv, action: torch.Tensor) -> dict[str, np.ndarray]:
    tool_pos_b, tool_quat_b = env._compute_frame_pose()
    rgb = env.wrist_camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy().astype(np.uint8)
    return {
        "xyz": _to_numpy(tool_pos_b).astype(np.float32),
        "quat": _quat_wxyz_to_xyzw(_to_numpy(tool_quat_b)),
        "width": _to_numpy(env.gripper_width[:, :1]).astype(np.float32),
        "ft": _to_numpy(env.get_cafe_ft()).astype(np.float32),
        "marker2d": _to_numpy(env.get_cafe_marker2d()).astype(np.float32),
        "rgb": rgb,
        "action": _to_numpy(action).astype(np.float32),
    }


def _record_base_dir() -> Path:
    if args_cli.dataset_file:
        return Path(args_cli.dataset_file).expanduser().resolve().parent
    return Path(args_cli.record_dir).expanduser().resolve()


def _due(next_timestamp: float, current_timestamp: float) -> bool:
    return next_timestamp <= current_timestamp + 1.0e-9


def main():
    env_cfg = LabPickEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.labware_name = args_cli.labware
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = LabPickEnv(env_cfg, render_mode="rgb_array")
    record_dir = _record_base_dir()
    record_dir.mkdir(parents=True, exist_ok=True)
    recorded = 0

    try:
        while simulation_app.is_running() and recorded < args_cli.num_demos:
            env.reset()
            writer = CafeRecordWriter(record_dir / f"record_{recorded:06d}")
            next_aligned_t = 1.0 / args_cli.aligned_hz
            next_camera_t = 1.0 / args_cli.camera_hz
            next_ft_t = 1.0 / args_cli.ft_hz
            next_tracker_t = 1.0 / args_cli.tracker_hz
            next_encoder_t = 1.0 / args_cli.aligned_hz
            next_xense_t = 1.0 / args_cli.aligned_hz

            for step in range(args_cli.max_episode_steps):
                env.command_pick_state_machine()
                action = env.get_cafe_action()

                env._pre_physics_step(None)
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
                env.sim.render()

                timestamp = float((step + 1) * env.physics_dt)
                sample = _make_cafe_sample(env, action)

                while _due(next_aligned_t, timestamp):
                    writer.append_aligned_sample(next_aligned_t, sample)
                    next_aligned_t += 1.0 / args_cli.aligned_hz
                while _due(next_camera_t, timestamp):
                    writer.append_camera_sample(next_camera_t, sample["rgb"])
                    next_camera_t += 1.0 / args_cli.camera_hz
                while _due(next_ft_t, timestamp):
                    writer.append_ft_sample(next_ft_t, sample["ft"])
                    next_ft_t += 1.0 / args_cli.ft_hz
                while _due(next_tracker_t, timestamp):
                    writer.append_tracker_sample(next_tracker_t, sample["xyz"], sample["quat"])
                    next_tracker_t += 1.0 / args_cli.tracker_hz
                while _due(next_encoder_t, timestamp):
                    writer.append_encoder_sample(next_encoder_t, sample["width"])
                    next_encoder_t += 1.0 / args_cli.aligned_hz
                while _due(next_xense_t, timestamp):
                    writer.append_xense_sample(next_xense_t, sample["marker2d"])
                    next_xense_t += 1.0 / args_cli.aligned_hz

                terminated, time_out = env._get_dones()
                done = bool((terminated | time_out)[0].item())

                lift_delta = env.labware.data.root_pos_w[:, 2] - env.initial_object_height
                success = bool((lift_delta[0] > env.cfg.success_lift_height).item())
                if done or success:
                    exported = False
                    if success or not args_cli.success_only:
                        exported = writer.flush_episode(
                            success=success,
                            labware_reset_pos_w=_to_numpy(env.labware_reset_pos_w).astype(np.float32),
                            labware_reset_quat_w=_to_numpy(env.labware_reset_quat_w).astype(np.float32),
                        )
                    else:
                        writer.clear_episode()
                    if exported:
                        recorded += 1
                        print(f"[INFO] recorded_demo={recorded}/{args_cli.num_demos} success={success}")
                    break
            else:
                exported = False
                if not args_cli.success_only:
                    exported = writer.flush_episode(
                        success=False,
                        labware_reset_pos_w=_to_numpy(env.labware_reset_pos_w).astype(np.float32),
                        labware_reset_quat_w=_to_numpy(env.labware_reset_quat_w).astype(np.float32),
                    )
                else:
                    writer.clear_episode()
                if exported:
                    recorded += 1
                    print(f"[INFO] recorded_demo={recorded}/{args_cli.num_demos} success=False")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
