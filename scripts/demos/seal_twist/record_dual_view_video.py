#!/usr/bin/env python3
"""Record synchronized wrist, third-person, and side-by-side SealTwist videos."""

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
    required = [
        site_packages / "torch" / "lib",
        site_packages / "nvidia" / "cudnn" / "lib",
    ]
    current = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item]
    missing = [str(item) for item in required if item.is_dir() and str(item) not in current]
    if missing:
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = os.pathsep.join([*missing, *current])
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)


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
parser.add_argument("--seed", type=int, default=180700)
parser.add_argument("--max-steps", type=int, default=1050)
parser.add_argument("--video-stride", type=int, default=4)
parser.add_argument("--video-fps", type=float, default=30.0)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("exp_report/simulation_videos/seal_twist"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.max_steps < 1 or args.video_stride < 1 or args.video_fps <= 0.0:
    parser.error("max-steps, video-stride and video-fps must be positive")
args.enable_cameras = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import cv2
import imageio.v2 as imageio
import numpy as np

from tacex_tasks.seal_twist.seal_twist_env import SealTwistEnv
from tacex_tasks.seal_twist.seal_twist_env_cfg import SealTwistEnvCfg


def _writer(path: Path):
    return imageio.get_writer(
        str(path), fps=args.video_fps, codec="libx264", quality=9, macro_block_size=None
    )


def _rgb(env: SealTwistEnv, view: str) -> np.ndarray:
    return np.ascontiguousarray(
        env.get_camera_rgb(view)[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    )


def _combined(first: np.ndarray, third: np.ndarray) -> np.ndarray:
    if first.shape[:2] != third.shape[:2]:
        first = cv2.resize(first, (third.shape[1], third.shape[0]), interpolation=cv2.INTER_AREA)
    output = np.concatenate((first, third), axis=1)
    for x, label in ((0, "FIRST-PERSON WRIST"), (third.shape[1], "THIRD-PERSON")):
        cv2.rectangle(output, (x + 14, 14), (x + 236, 48), (12, 18, 24), -1)
        cv2.putText(
            output,
            label,
            (x + 25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return np.ascontiguousarray(output)


def _flag(flags: dict[str, object], name: str) -> bool:
    return bool(flags[name][0].item())


def main() -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    first_path = output_dir / "seal_twist_first_person.mp4"
    third_path = output_dir / "seal_twist_third_person.mp4"
    dual_path = output_dir / "seal_twist_dual_view.mp4"

    cfg = SealTwistEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.enable_cameras = True
    cfg.enable_gelsight = False
    cfg.randomize_labware_position = False
    cfg.randomize_force_thresholds = False
    cfg.randomize_expert_targets = False
    cfg.episode_length_s = max(args.max_steps * cfg.sim.dt, 1.0)
    cfg.wrist_camera.width = 640
    cfg.wrist_camera.height = 480
    cfg.third_person_camera.width = 640
    cfg.third_person_camera.height = 480
    if args.device:
        cfg.sim.device = args.device

    env = SealTwistEnv(cfg, render_mode="rgb_array")
    first_writer = _writer(first_path)
    third_writer = _writer(third_path)
    dual_writer = _writer(dual_path)
    frame_count = 0
    terminal_reason = "timeout"
    flags = None
    last_frames: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def append_frame() -> None:
        nonlocal frame_count, last_frames
        env.sim.render()
        first = _rgb(env, "wrist")
        third = _rgb(env, "third")
        dual = _combined(first, third)
        first_writer.append_data(first)
        third_writer.append_data(third)
        dual_writer.append_data(dual)
        last_frames = first, third, dual
        frame_count += 1

    try:
        env.reset(seed=args.seed)
        append_frame()
        for _ in range(args.max_steps):
            if not simulation_app.is_running():
                terminal_reason = "interrupted"
                break
            env.command_expert()
            env._pre_physics_step(None)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            flags = env.get_task_flags()
            step = int(env.step_count[0].item())
            terminal_reason = ""
            for reason in (
                "success",
                "over_torqued",
                "over_force",
                "under_force",
                "alignment_failure",
                "slipped",
            ):
                if _flag(flags, reason):
                    terminal_reason = reason
                    break
            if step % args.video_stride == 0 or terminal_reason:
                append_frame()
            if terminal_reason:
                break
        if not terminal_reason:
            terminal_reason = "timeout"
        if last_frames is not None:
            for _ in range(max(1, int(round(args.video_fps)))):
                first_writer.append_data(last_frames[0])
                third_writer.append_data(last_frames[1])
                dual_writer.append_data(last_frames[2])
                frame_count += 1
    finally:
        first_writer.close()
        third_writer.close()
        dual_writer.close()

    if flags is None:
        flags = env.get_task_flags()
    summary = {
        "task": "TacEx-SealTwist-Torque-v0",
        "seed": args.seed,
        "success": _flag(flags, "success"),
        "terminal_reason": terminal_reason,
        "steps": int(env.step_count[0].item()),
        "frames": frame_count,
        "fps": args.video_fps,
        "first_person_video": str(first_path),
        "third_person_video": str(third_path),
        "dual_view_video": str(dual_path),
        "final_torque_nm": float(flags["seal_torque_nm"][0].item()),
        "peak_torque_nm": float(env.peak_torque_nm[0].item()),
        "final_left_force_n": float(flags["left_finger_force_n"][0].item()),
        "final_right_force_n": float(flags["right_finger_force_n"][0].item()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("[SUMMARY] " + json.dumps(summary, ensure_ascii=False), flush=True)
    env.close()
    if not summary["success"]:
        raise RuntimeError(f"SealTwist video episode failed: {terminal_reason}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
