"""Collect and optionally film one tactile seal-tightening expert trajectory."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _reexec_with_cuda_libraries() -> None:
    py_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / py_dir / "site-packages"
    required = [site_packages / "torch" / "lib", site_packages / "nvidia" / "cudnn" / "lib"]
    current = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item]
    missing = [str(item) for item in required if item.is_dir() and str(item) not in current]
    if missing:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*missing, *current])
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def _bootstrap_isaaclab() -> None:
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        return
    source_root = Path(spec.origin).resolve().parent / "source"
    for package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
        package_root = source_root / package
        if (package_root / package).is_dir() and str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))


_reexec_with_cuda_libraries()
_bootstrap_isaaclab()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("datasets/seal_twist/record_000000"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=1050)
parser.add_argument("--record_stride", type=int, default=4)
parser.add_argument("--save_rgb", action="store_true")
parser.add_argument("--save_tactile", action="store_true")
parser.add_argument("--record_video", action="store_true")
parser.add_argument("--video_path", type=Path, default=Path("artifacts/seal_twist/seal_twist_expert.mp4"))
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--render_width", type=int, default=640)
parser.add_argument("--render_height", type=int, default=360)
parser.add_argument("--print_interval", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.record_stride <= 0:
    parser.error("--record_stride must be positive")
need_cameras = bool(args.save_rgb or args.save_tactile or args.record_video)
args.enable_cameras = need_cameras

launcher = AppLauncher(args)
simulation_app = launcher.app

import cv2
import imageio.v2 as imageio
import numpy as np

from tacex_tasks.seal_twist.dataset import SealTwistTrajectoryBuffer, validate_episode
from tacex_tasks.seal_twist.seal_twist_env import SealTwistEnv
from tacex_tasks.seal_twist.seal_twist_env_cfg import SealTwistEnvCfg


def _flag(flags: dict[str, object], name: str) -> bool:
    return bool(flags[name][0].item())


def _annotate(frame: np.ndarray, env: SealTwistEnv, flags: dict[str, object]) -> np.ndarray:
    output = np.ascontiguousarray(frame[..., :3].copy())
    phase = "SUCCESS" if _flag(flags, "success") else env.get_expert_phase_names()[0].upper()
    torque = float(flags["seal_torque_nm"][0].item())
    angle_deg = float(np.degrees(flags["turn_angle_rad"][0].item()))
    left = float(flags["left_finger_force_n"][0].item())
    right = float(flags["right_finger_force_n"][0].item())
    lower, upper = env.cfg.torque_window_nm
    safe = lower <= torque <= upper
    cv2.rectangle(output, (12, 12), (430, 142), (18, 20, 25), thickness=-1)
    cv2.putText(output, f"SEAL TWIST | {phase}", (25, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(output, f"angle  {angle_deg:6.1f} deg", (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (110, 210, 255), 1, cv2.LINE_AA)
    torque_color = (80, 225, 110) if safe else (70, 180, 255)
    cv2.putText(output, f"torque {torque:5.3f} Nm  window [{lower:.2f}, {upper:.2f}]", (25, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.54, torque_color, 1, cv2.LINE_AA)
    cv2.putText(output, f"touch L/R {left:4.2f} / {right:4.2f} N", (25, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (230, 210, 120), 1, cv2.LINE_AA)
    return output


def main() -> None:
    cfg = SealTwistEnvCfg()
    cfg.seed = args.seed
    cfg.enable_cameras = need_cameras
    cfg.enable_gelsight = args.save_tactile
    cfg.episode_length_s = max(args.max_steps * cfg.sim.dt, 1.0)
    cfg.third_person_camera.width = args.render_width
    cfg.third_person_camera.height = args.render_height
    if args.device:
        cfg.sim.device = args.device

    env = SealTwistEnv(cfg, render_mode="rgb_array" if need_cameras else None)
    buffer = SealTwistTrajectoryBuffer()
    writer = None
    if args.record_video:
        args.video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(args.video_path), fps=args.video_fps, codec="libx264", quality=8, macro_block_size=None
        )

    flags = None
    frames = 0
    terminal_reason = "timeout"
    try:
        env.reset(seed=args.seed)
        for _ in range(args.max_steps):
            if not simulation_app.is_running():
                break
            env.command_expert()
            action = env.get_cafe_action()
            env._pre_physics_step(None)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            flags = env.get_task_flags()
            step = int(env.step_count[0].item())
            sample_due = step % args.record_stride == 0
            if sample_due and need_cameras:
                env.sim.render()
            if sample_due:
                buffer.append(env, action, save_rgb=args.save_rgb, save_tactile=args.save_tactile)
            if writer is not None and sample_due:
                frame = env.get_camera_rgb("third")[0].cpu().numpy()
                writer.append_data(_annotate(frame, env, flags))
                frames += 1
            if args.print_interval > 0 and step % args.print_interval == 0:
                print(
                    f"[STATE] step={step:04d} phase={env.get_expert_phase_names()[0]} "
                    f"angle={flags['turn_angle_rad'][0].item():.3f}rad "
                    f"torque={flags['seal_torque_nm'][0].item():.3f}Nm "
                    f"touch=({flags['left_finger_force_n'][0].item():.2f},"
                    f"{flags['right_finger_force_n'][0].item():.2f})N "
                    f"success={_flag(flags, 'success')}",
                    flush=True,
                )
            for reason in ("success", "over_torqued", "slipped"):
                if _flag(flags, reason):
                    terminal_reason = reason
                    break
            if terminal_reason != "timeout":
                # Always preserve the exact terminal transition even when it
                # falls between regular 30 Hz samples.
                if not sample_due:
                    if need_cameras:
                        env.sim.render()
                    buffer.append(env, action, save_rgb=args.save_rgb, save_tactile=args.save_tactile)
                    if writer is not None:
                        frame = env.get_camera_rgb("third")[0].cpu().numpy()
                        writer.append_data(_annotate(frame, env, flags))
                        frames += 1
                break
    finally:
        if writer is not None:
            writer.close()

    if flags is None:
        flags = env.get_task_flags()
    saved = buffer.save(
        args.output,
        env,
        seed=args.seed,
        success=_flag(flags, "success"),
        sample_period_s=env.physics_dt * args.record_stride,
    )
    metadata = validate_episode(saved)
    summary = {
        "task": "TacEx-SealTwist-Torque-v0",
        "success": _flag(flags, "success"),
        "terminal_reason": terminal_reason,
        "steps": int(env.step_count[0].item()),
        "samples": int(metadata["length"]),
        "final_torque_nm": float(flags["seal_torque_nm"][0].item()),
        "peak_torque_nm": float(env.peak_torque_nm[0].item()),
        "final_angle_deg": float(np.degrees(env.turn_angle_rad[0].item())),
        "peak_left_finger_force_n": float(env.peak_left_force_n[0].item()),
        "peak_right_finger_force_n": float(env.peak_right_force_n[0].item()),
        "bilateral_contact": _flag(flags, "bilateral_contact"),
        "dataset": str(saved.resolve()),
        "video": str(args.video_path.resolve()) if writer is not None else None,
        "video_frames": frames,
    }
    summary_path = args.video_path.parent / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[RESULT] " + json.dumps(summary, ensure_ascii=False), flush=True)
    env.close()
    if not summary["success"]:
        raise RuntimeError(f"SealTwist expert did not succeed: {terminal_reason}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
