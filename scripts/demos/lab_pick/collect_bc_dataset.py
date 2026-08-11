from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path


def _reexec_with_torch_cuda_library_path() -> None:
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / python_dir / "site-packages"
    required_dirs = [
        site_packages / "torch" / "lib",
        site_packages / "nvidia" / "cudnn" / "lib",
    ]
    current_paths = [value for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if value]
    missing = [str(path) for path in required_dirs if path.is_dir() and str(path) not in current_paths]
    if not missing:
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join([*missing, *current_paths])
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


_reexec_with_torch_cuda_library_path()


import numpy as np
import torch
import torch.nn.functional as F


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
parser.add_argument("--failure_only", action="store_true")
parser.add_argument("--max_attempts", type=int, default=0, help="Stop after this many attempts; 0 means no explicit cap.")
parser.add_argument("--max_episode_steps", type=int, default=960)
parser.add_argument("--aligned_hz", type=float, default=60.0)
parser.add_argument("--camera_hz", type=float, default=30.0, help="Deprecated; camera streams are saved only in aligned.")
parser.add_argument("--ft_hz", type=float, default=90.0)
parser.add_argument("--tracker_hz", type=float, default=300.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--break_force_threshold_n", type=float, default=0.0, help="Override break-force threshold; <=0 keeps env default.")
parser.add_argument(
    "--safe_demo_fraction",
    type=float,
    default=0.5,
    help="Fraction of recorded demonstrations commanded with the safe slide close width.",
)
parser.add_argument(
    "--position_failure_demo_fraction",
    type=float,
    default=0.25,
    help="Fraction of demonstrations intentionally offset from the slide to record position failures.",
)
parser.add_argument("--safe_close_width_m", type=float, default=0.0065)
parser.add_argument("--overforce_close_width_m", type=float, default=0.0015)
parser.add_argument(
    "--position_failure_offset_m",
    type=float,
    default=0.03,
    help="Absolute y offset applied to position-failure expert targets.",
)
parser.add_argument(
    "--labware_random_xy",
    type=float,
    nargs=2,
    metavar=("X", "Y"),
    default=None,
    help="Override the uniform x/y half ranges in meters.",
)
parser.add_argument(
    "--labware_random_yaw_degrees",
    type=float,
    default=None,
    help="Override the uniform yaw half range in degrees.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.success_only and args_cli.failure_only:
    parser.error("--success_only and --failure_only are mutually exclusive")
if not 0.0 <= args_cli.safe_demo_fraction <= 1.0:
    parser.error("--safe_demo_fraction must be in [0, 1]")
if not 0.0 <= args_cli.position_failure_demo_fraction <= 1.0:
    parser.error("--position_failure_demo_fraction must be in [0, 1]")
if args_cli.safe_demo_fraction + args_cli.position_failure_demo_fraction > 1.0:
    parser.error("--safe_demo_fraction + --position_failure_demo_fraction must be <= 1")
if not 0.0 <= args_cli.safe_close_width_m <= 0.04:
    parser.error("--safe_close_width_m must be in [0, 0.04]")
if not 0.0 <= args_cli.overforce_close_width_m <= 0.04:
    parser.error("--overforce_close_width_m must be in [0, 0.04]")
if args_cli.position_failure_offset_m <= 0.0:
    parser.error("--position_failure_offset_m must be > 0")
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


def _camera_rgb_224(camera_rgb: torch.Tensor) -> np.ndarray:
    rgb = camera_rgb[0, :, :, :3].permute(2, 0, 1).unsqueeze(0).float()
    rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
    rgb = rgb.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte()
    return rgb.detach().cpu().numpy()


def _make_cafe_observation(env: LabPickEnv) -> dict[str, np.ndarray]:
    tool_pos_b, tool_quat_b = env._compute_frame_pose()
    rgb = _camera_rgb_224(env.wrist_camera.data.output["rgb"])
    rgb_third = _camera_rgb_224(env.third_person_camera.data.output["rgb"])
    return {
        "xyz": _to_numpy(tool_pos_b).astype(np.float32),
        "quat": _quat_wxyz_to_xyzw(_to_numpy(tool_quat_b)),
        # Capture the previous command width before the expert mutates it for
        # the next action. This matches the observation available at deploy.
        "width": _to_numpy(env.gripper_width[:, :1]).astype(np.float32),
        "ft": _to_numpy(env.get_cafe_ft()).astype(np.float32),
        "marker2d": _to_numpy(env.get_cafe_marker2d()).astype(np.float32),
        "rgb": rgb,
        "rgb_third": rgb_third,
    }


def _make_cafe_sample(observation: dict[str, np.ndarray], action: torch.Tensor) -> dict[str, np.ndarray]:
    return {**observation, "action": _to_numpy(action).astype(np.float32)}


def _record_base_dir() -> Path:
    if args_cli.dataset_file:
        return Path(args_cli.dataset_file).expanduser().resolve().parent
    return Path(args_cli.record_dir).expanduser().resolve()


def _due(next_timestamp: float, current_timestamp: float) -> bool:
    return next_timestamp <= current_timestamp + 1.0e-9


def _build_demo_modes(
    num_demos: int,
    safe_fraction: float,
    position_failure_fraction: float,
    seed: int,
) -> list[str]:
    fractions = np.asarray(
        [safe_fraction, 1.0 - safe_fraction - position_failure_fraction, position_failure_fraction],
        dtype=np.float64,
    )
    exact_counts = fractions * num_demos
    counts = np.floor(exact_counts).astype(np.int64)
    remainder = num_demos - int(counts.sum())
    if remainder:
        order = np.argsort(-(exact_counts - counts), kind="stable")
        counts[order[:remainder]] += 1
    safe_count, overforce_count, position_failure_count = map(int, counts)
    modes = (
        ["safe"] * safe_count
        + ["overforce"] * overforce_count
        + ["position_failure"] * position_failure_count
    )
    np.random.default_rng(seed).shuffle(modes)
    return modes


def _flag_is_set(flags: dict[str, torch.Tensor], name: str) -> bool:
    return bool(flags[name].reshape(-1)[0].item())


def _print_collection_stats(*, attempted: int, successful: int, recorded: int, final: bool = False):
    failures = attempted - successful
    success_rate = successful / attempted if attempted else 0.0
    label = "SUMMARY" if final else "STATS"
    print(
        f"[{label}] attempts={attempted} successes={successful} failures={failures} "
        f"success_rate={successful}/{attempted} ({success_rate:.2%}) recorded={recorded}/{args_cli.num_demos}"
    )


def _failure_reasons(flags: dict[str, torch.Tensor]) -> list[str]:
    reasons: list[str] = []
    if _flag_is_set(flags, "object_broken"):
        reasons.append("break_force")
    if _flag_is_set(flags, "object_dropped"):
        reasons.append("object_drop")
    if _flag_is_set(flags, "object_too_far"):
        reasons.append("object_xy_distance")
    if _flag_is_set(flags, "ee_outside_workspace"):
        reasons.append("ee_workspace")
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
    rgb_third = np.asarray(sample["rgb_third"], dtype=np.uint8)
    ft = np.asarray(sample["ft"], dtype=np.float32).reshape(6)
    force_norm = float(np.linalg.norm(ft[:3]))
    torque_norm = float(np.linalg.norm(ft[3:]))

    np.save(debug_dir / f"{prefix}_rgb.npy", rgb)
    np.save(debug_dir / f"{prefix}_rgb_third.npy", rgb_third)
    np.save(debug_dir / f"{prefix}_ft.npy", ft)
    preview_path = _save_rgb_preview(debug_dir / f"{prefix}_rgb.png", rgb)
    third_preview_path = _save_rgb_preview(debug_dir / f"{prefix}_rgb_third.png", rgb_third)
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
        f"rgb_third_npy={debug_dir / f'{prefix}_rgb_third.npy'}\n"
        f"rgb_third_preview={third_preview_path}\n"
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
    if args_cli.labware_random_xy is not None:
        env_cfg.labware_pos_randomization_xy = tuple(args_cli.labware_random_xy)
    if args_cli.labware_random_yaw_degrees is not None:
        env_cfg.labware_yaw_randomization = float(np.deg2rad(args_cli.labware_random_yaw_degrees))
    if args_cli.break_force_threshold_n > 0.0:
        env_cfg.terminate_break_force_threshold_n = args_cli.break_force_threshold_n
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = LabPickEnv(env_cfg, render_mode="rgb_array")
    record_dir = _record_base_dir()
    record_dir.mkdir(parents=True, exist_ok=True)
    safe_fraction = (
        1.0
        if args_cli.success_only
        else 0.0
        if args_cli.failure_only
        else args_cli.safe_demo_fraction
    )
    position_failure_fraction = 0.0 if args_cli.success_only else args_cli.position_failure_demo_fraction
    demo_modes = _build_demo_modes(
        args_cli.num_demos,
        safe_fraction,
        position_failure_fraction,
        args_cli.seed,
    )
    collection_config = {
        "num_demos": args_cli.num_demos,
        "safe_demo_fraction": safe_fraction,
        "overforce_demo_fraction": 1.0 - safe_fraction - position_failure_fraction,
        "position_failure_demo_fraction": position_failure_fraction,
        "safe_close_width_m": args_cli.safe_close_width_m,
        "overforce_close_width_m": args_cli.overforce_close_width_m,
        "position_failure_offset_m": args_cli.position_failure_offset_m,
        "break_force_threshold_n": env_cfg.terminate_break_force_threshold_n,
        "labware_random_xy_m": list(env_cfg.labware_pos_randomization_xy),
        "labware_random_yaw_degrees": float(np.rad2deg(env_cfg.labware_yaw_randomization)),
        "seed": args_cli.seed,
    }
    (record_dir / "collection_config.json").write_text(
        json.dumps(collection_config, indent=2) + "\n", encoding="utf-8"
    )
    existing_record_indices = [
        int(path.name.removeprefix("record_"))
        for path in record_dir.glob("record_*")
        if path.is_dir() and path.name.removeprefix("record_").isdigit()
    ]
    next_record_index = max(existing_record_indices, default=-1) + 1
    recorded = 0
    attempted = 0
    successful = 0
    failure_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()

    try:
        while (
            simulation_app.is_running()
            and recorded < args_cli.num_demos
            and (args_cli.max_attempts <= 0 or attempted < args_cli.max_attempts)
        ):
            env.reset()
            demonstration_mode = demo_modes[recorded]
            slide_close_width_m = (
                args_cli.overforce_close_width_m
                if demonstration_mode == "overforce"
                else args_cli.safe_close_width_m
            )
            position_failure_sign = -1.0 if (next_record_index + recorded) % 2 else 1.0
            attempt_index = attempted
            attempted += 1
            writer = CafeRecordWriter(record_dir / f"record_{next_record_index + recorded:06d}")
            failure_debug_dir = record_dir / "failed_attempts" / f"attempt_{next_record_index + attempt_index:06d}"
            next_aligned_t = 0.0
            next_ft_t = 0.0
            next_tracker_t = 0.0
            next_encoder_t = 0.0
            next_xense_t = 0.0
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
            peak_break_force_n = 0.0

            for step in range(args_cli.max_episode_steps):
                timestamp = float(step * env.physics_dt)
                observation = _make_cafe_observation(env)
                env.command_pick_state_machine(slide_close_width_m=slide_close_width_m)
                if demonstration_mode == "position_failure":
                    position_offset = position_failure_sign * args_cli.position_failure_offset_m
                    env.ik_commands[:, 1] += position_offset
                    env.last_target_pos_b[:, 1] += position_offset
                action = env.get_cafe_action()
                sample = _make_cafe_sample(observation, action)
                last_sample = sample
                last_timestamp = timestamp
                last_step = step

                while _due(next_aligned_t, timestamp):
                    writer.append_aligned_sample(next_aligned_t, sample)
                    next_aligned_t += 1.0 / args_cli.aligned_hz
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

                env._pre_physics_step(None)
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
                env.sim.render()

                flags = env._get_termination_flags()
                peak_break_force_n = max(peak_break_force_n, float(flags["break_force_n"][0].item()))
                terminated_now = any(
                    _flag_is_set(flags, name)
                    for name in ("object_dropped", "object_too_far", "ee_outside_workspace", "object_broken")
                )
                if terminated_now and not episode_failed:
                    episode_failed = True
                    first_failure_step = step
                    failure_reason = "+".join(_failure_reasons(flags)) or "terminated"
                    failure_sample = _make_cafe_sample(_make_cafe_observation(env), action)
                    failure_timestamp = float((step + 1) * env.physics_dt)
                    failure_step = step
                if terminated_now:
                    break

                lift_delta = env.labware.data.root_pos_w[:, 2] - env.initial_object_height
                success = bool((lift_delta[0] > env.cfg.success_lift_height).item())
                if success and not episode_failed:
                    successful += 1
                    if args_cli.failure_only:
                        writer.clear_episode()
                        exported = True
                        print(
                            f"[INFO] skipped_success attempt={attempted} "
                            f"recorded={recorded}/{args_cli.num_demos} failure_only=True"
                        )
                    else:
                        exported = writer.flush_episode(
                            success=True,
                            labware_reset_pos_w=_to_numpy(env.labware_reset_pos_w).astype(np.float32),
                            labware_reset_quat_w=_to_numpy(env.labware_reset_quat_w).astype(np.float32),
                            demonstration_mode=demonstration_mode,
                            failure_reason="",
                            peak_break_force_n=peak_break_force_n,
                        )
                        if exported:
                            recorded += 1
                            mode_counts[demonstration_mode] += 1
                            print(f"[INFO] recorded_demo={recorded}/{args_cli.num_demos} success=True")
                    break

            if not exported:
                if not episode_failed:
                    failure_reason = (
                        "position_unreachable"
                        if demonstration_mode == "position_failure"
                        else "timeout_or_no_success"
                    )
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
                        demonstration_mode=demonstration_mode,
                        failure_reason=failure_reason,
                        peak_break_force_n=peak_break_force_n,
                    )
                if exported:
                    recorded += 1
                    mode_counts[demonstration_mode] += 1
                    failure_counts[failure_reason] += 1
                    print(f"[INFO] recorded_demo={recorded}/{args_cli.num_demos} success=False")
            _print_collection_stats(attempted=attempted, successful=successful, recorded=recorded)
    finally:
        _print_collection_stats(attempted=attempted, successful=successful, recorded=recorded, final=True)
        recorded_failures = int(sum(failure_counts.values()))
        summary = {
            **collection_config,
            "attempted": attempted,
            "recorded": recorded,
            "successful": successful,
            "success_rate": successful / max(attempted, 1),
            "recorded_failures": recorded_failures,
            "failure_counts": dict(failure_counts),
            "mode_counts": dict(mode_counts),
            "break_failure_fraction": failure_counts["break_force"] / max(recorded_failures, 1),
            "position_failure_fraction": failure_counts["position_unreachable"]
            / max(recorded_failures, 1),
        }
        (record_dir / "collection_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
