from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _bootstrap_isaaclab_source_paths():
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        return

    isaaclab_package_dir = Path(spec.origin).resolve().parent
    source_root = isaaclab_package_dir / "source"
    if not source_root.is_dir():
        return

    for package_name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
        package_source = source_root / package_name
        if (package_source / package_name).is_dir():
            package_source_str = str(package_source)
            if package_source_str not in sys.path:
                sys.path.insert(0, package_source_str)


def _bootstrap_isaacsim_warp_path():
    spec = importlib.util.find_spec("isaacsim")
    if spec is None or spec.origin is None:
        return

    isaacsim_package_dir = Path(spec.origin).resolve().parent
    extcache_dir = isaacsim_package_dir / "extscache"
    if not extcache_dir.is_dir():
        return

    warp_core_paths = sorted(extcache_dir.glob("omni.warp.core-*"), reverse=True)
    for warp_core_path in warp_core_paths:
        if (warp_core_path / "warp" / "__init__.py").is_file():
            warp_core_path_str = str(warp_core_path)
            if warp_core_path_str not in sys.path:
                sys.path.insert(0, warp_core_path_str)
            return


_bootstrap_isaaclab_source_paths()
_bootstrap_isaacsim_warp_path()

from isaaclab.app import AppLauncher


def _patch_isaaclab_missing_exports():
    import isaaclab.utils as isaaclab_utils
    from isaaclab.utils.buffers.circular_buffer import CircularBuffer
    from isaaclab.utils.buffers.delay_buffer import DelayBuffer
    from isaaclab.utils.buffers.timestamped_buffer import TimestampedBuffer

    isaaclab_utils.CircularBuffer = CircularBuffer
    isaaclab_utils.DelayBuffer = DelayBuffer
    isaaclab_utils.TimestampedBuffer = TimestampedBuffer


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

_patch_isaaclab_missing_exports()

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


def _print_collection_stats(*, attempted: int, successful: int, recorded: int, final: bool = False):
    failures = attempted - successful
    success_rate = successful / attempted if attempted else 0.0
    label = "SUMMARY" if final else "STATS"
    print(
        f"[{label}] attempts={attempted} successes={successful} failures={failures} "
        f"success_rate={successful}/{attempted} ({success_rate:.2%}) recorded={recorded}/{args_cli.num_demos}"
    )


def _failure_reasons(env: LabPickEnv) -> list[str]:
    reasons: list[str] = []

    object_drop_delta = env.labware.data.root_pos_w[:, 2] - env.initial_object_height
    if bool((object_drop_delta < -env.cfg.terminate_object_drop_height)[0].item()):
        reasons.append("object_drop")

    object_pos_b = env.labware.data.root_pos_w - env._robot.data.root_link_pos_w
    object_xy_delta = object_pos_b[:, :2] - env.initial_object_pos_b[:, :2]
    if bool((torch.linalg.norm(object_xy_delta, dim=1) > env.cfg.terminate_object_xy_distance)[0].item()):
        reasons.append("object_xy_distance")

    ee_pos_b, _ = env._compute_frame_pose()
    workspace_min = env.workspace_min_b - env.cfg.terminate_ee_workspace_margin
    workspace_max = env.workspace_max_b + env.cfg.terminate_ee_workspace_margin
    if bool(torch.any((ee_pos_b < workspace_min) | (ee_pos_b > workspace_max), dim=1)[0].item()):
        reasons.append("ee_workspace")

    force_norm = torch.linalg.norm(env.get_cafe_ft()[:, :3], dim=1)
    if bool((env.has_touched & (force_norm > env.cfg.terminate_break_force_threshold_n))[0].item()):
        reasons.append("break_force")

    return reasons


def _save_rgb_preview(rgb_path: Path, rgb: np.ndarray) -> Path:
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    try:
        from PIL import Image

        Image.fromarray(rgb_u8).save(rgb_path)
        return rgb_path
    except Exception:
        ppm_path = rgb_path.with_suffix(".ppm")
        height, width = rgb_u8.shape[:2]
        with ppm_path.open("wb") as stream:
            stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            stream.write(rgb_u8[:, :, :3].tobytes())
        return ppm_path


def _write_frame_debug(
    debug_dir: Path,
    *,
    prefix: str,
    sample: dict[str, np.ndarray],
    timestamp: float,
    step: int,
    first_failure_step: int,
    failure_reason: str,
    break_force_threshold_n: float,
):
    debug_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(sample["rgb"], dtype=np.uint8)
    ft = np.asarray(sample["ft"], dtype=np.float32).reshape(6)
    force_norm = float(np.linalg.norm(ft[:3]))
    torque_norm = float(np.linalg.norm(ft[3:]))

    np.save(debug_dir / f"{prefix}_rgb.npy", rgb)
    np.save(debug_dir / f"{prefix}_ft.npy", ft)
    preview_path = _save_rgb_preview(debug_dir / f"{prefix}_rgb.png", rgb)
    summary = (
        f"failure_reason={failure_reason}\n"
        f"{prefix}_step={step}\n"
        f"first_failure_step={first_failure_step}\n"
        f"timestamp={timestamp:.6f}\n"
        f"ft=[{ft[0]:.6f}, {ft[1]:.6f}, {ft[2]:.6f}, {ft[3]:.6f}, {ft[4]:.6f}, {ft[5]:.6f}]\n"
        f"force_norm_n={force_norm:.6f}\n"
        f"torque_norm_nm={torque_norm:.6f}\n"
        f"break_force_threshold_n={break_force_threshold_n:.6f}\n"
        f"rgb_npy={debug_dir / f'{prefix}_rgb.npy'}\n"
        f"rgb_preview={preview_path}\n"
        f"ft_npy={debug_dir / f'{prefix}_ft.npy'}\n"
    )
    (debug_dir / f"{prefix}_info.txt").write_text(summary, encoding="utf-8")
    print(
        f"[WARN] failed_attempt_{prefix} "
        f"reason={failure_reason} step={step} first_failure_step={first_failure_step} "
        f"force_norm_n={force_norm:.6f} ft={ft.round(6).tolist()} debug_dir={debug_dir}"
    )


def _write_failure_debug(
    debug_dir: Path,
    *,
    failure_sample: dict[str, np.ndarray],
    failure_timestamp: float,
    failure_step: int,
    last_sample: dict[str, np.ndarray],
    last_timestamp: float,
    last_step: int,
    first_failure_step: int,
    failure_reason: str,
    break_force_threshold_n: float,
):
    _write_frame_debug(
        debug_dir,
        prefix="failure_frame",
        sample=failure_sample,
        timestamp=failure_timestamp,
        step=failure_step,
        first_failure_step=first_failure_step,
        failure_reason=failure_reason,
        break_force_threshold_n=break_force_threshold_n,
    )
    _write_frame_debug(
        debug_dir,
        prefix="last_frame",
        sample=last_sample,
        timestamp=last_timestamp,
        step=last_step,
        first_failure_step=first_failure_step,
        failure_reason=failure_reason,
        break_force_threshold_n=break_force_threshold_n,
    )


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
    attempted = 0
    successful = 0

    try:
        while simulation_app.is_running() and recorded < args_cli.num_demos:
            env.reset()
            attempt_index = attempted
            attempted += 1
            writer = CafeRecordWriter(record_dir / f"record_{recorded:06d}")
            failure_debug_dir = record_dir / "failed_attempts" / f"attempt_{attempt_index:06d}"
            next_aligned_t = 1.0 / args_cli.aligned_hz
            next_camera_t = 1.0 / args_cli.camera_hz
            next_ft_t = 1.0 / args_cli.ft_hz
            next_tracker_t = 1.0 / args_cli.tracker_hz
            next_encoder_t = 1.0 / args_cli.aligned_hz
            next_xense_t = 1.0 / args_cli.aligned_hz
            episode_failed = False
            first_failure_step = -1
            failure_reason = ""
            failure_sample: dict[str, np.ndarray] | None = None
            failure_timestamp = 0.0
            failure_step = -1
            last_sample: dict[str, np.ndarray] | None = None
            last_timestamp = 0.0
            last_step = -1
            exported = False

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
                last_sample = sample
                last_timestamp = timestamp
                last_step = step

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

                terminated, _time_out = env._get_dones()
                terminated_now = bool(terminated[0].item())
                if terminated_now and not episode_failed:
                    episode_failed = True
                    first_failure_step = step
                    failure_reason = "+".join(_failure_reasons(env)) or "terminated"
                    failure_sample = sample
                    failure_timestamp = timestamp
                    failure_step = step

                lift_delta = env.labware.data.root_pos_w[:, 2] - env.initial_object_height
                success = bool((lift_delta[0] > env.cfg.success_lift_height).item())
                if success and not episode_failed:
                    exported = writer.flush_episode(
                        success=True,
                        labware_reset_pos_w=_to_numpy(env.labware_reset_pos_w).astype(np.float32),
                        labware_reset_quat_w=_to_numpy(env.labware_reset_quat_w).astype(np.float32),
                    )
                    if exported:
                        successful += 1
                        recorded += 1
                        print(f"[INFO] recorded_demo={recorded}/{args_cli.num_demos} success=True")
                    break

            if not exported:
                if not episode_failed:
                    failure_reason = "timeout_or_no_success"
                    first_failure_step = last_step
                    failure_sample = last_sample
                    failure_timestamp = last_timestamp
                    failure_step = last_step
                if last_sample is not None and failure_sample is not None:
                    _write_failure_debug(
                        failure_debug_dir,
                        failure_sample=failure_sample,
                        failure_timestamp=failure_timestamp,
                        failure_step=failure_step,
                        last_sample=last_sample,
                        last_timestamp=last_timestamp,
                        last_step=last_step,
                        first_failure_step=first_failure_step,
                        failure_reason=failure_reason,
                        break_force_threshold_n=env.cfg.terminate_break_force_threshold_n,
                    )
                if args_cli.success_only:
                    writer.clear_episode()
                else:
                    exported = writer.flush_episode(
                        success=False,
                        labware_reset_pos_w=_to_numpy(env.labware_reset_pos_w).astype(np.float32),
                        labware_reset_quat_w=_to_numpy(env.labware_reset_quat_w).astype(np.float32),
                    )
                if exported:
                    recorded += 1
                    print(f"[INFO] recorded_demo={recorded}/{args_cli.num_demos} success=False")
            _print_collection_stats(attempted=attempted, successful=successful, recorded=recorded)
    finally:
        _print_collection_stats(attempted=attempted, successful=successful, recorded=recorded, final=True)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
