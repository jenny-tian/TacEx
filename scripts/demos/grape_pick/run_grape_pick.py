"""Run, inspect and optionally record the force-constrained grape-picking expert."""

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
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--disable_gelsight", action="store_true")
parser.add_argument("--physics_only", action="store_true")
parser.add_argument("--no_randomization", action="store_true")
parser.add_argument("--randomize_force_thresholds", action="store_true")
parser.add_argument("--record_video", action="store_true")
parser.add_argument("--video_path", type=Path, default=Path("artifacts/grape_pick/grape_pick_expert.mp4"))
parser.add_argument("--summary_path", type=Path, default=Path("artifacts/grape_pick/summary.json"))
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--video_stride", type=int, default=4)
parser.add_argument("--render_width", type=int, default=640)
parser.add_argument("--render_height", type=int, default=360)
parser.add_argument("--print_interval", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.physics_only

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np

from tacex_tasks.grape_pick.grape_pick_env import GrapePickEnv
from tacex_tasks.grape_pick.grape_pick_env_cfg import GrapePickEnvCfg


def _flag(flags: dict, name: str) -> bool:
    return bool(flags[name][0].item())


def _annotate(frame: np.ndarray, env: GrapePickEnv, flags: dict) -> np.ndarray:
    import cv2

    output = np.ascontiguousarray(frame[..., :3].copy())
    phase = "SUCCESS" if _flag(flags, "success") else env.get_expert_phase_names()[0]
    grip = float(flags["grip_force_n"][0].item())
    tension = float(flags["stem_tension_n"][0].item())
    detached = _flag(flags, "detached")
    obstacle = float(flags["obstacle_contact_force_n"][0].item())
    damage_limit = float(flags["damage_threshold_n"][0].item())
    detach_limit = float(flags["detach_threshold_n"][0].item())
    cv2.rectangle(output, (12, 12), (430, 130), (18, 20, 24), thickness=-1)
    cv2.putText(output, f"GRAPE PICK | {phase}", (25, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    grip_color = (80, 225, 110) if grip <= damage_limit else (50, 50, 245)
    obstacle_color = (80, 225, 110) if obstacle < env.cfg.obstacle_contact_threshold_n else (50, 50, 245)
    cv2.putText(output, f"grip {grip:4.2f} / {damage_limit:4.2f} N", (25, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.52, grip_color, 1, cv2.LINE_AA)
    cv2.putText(output, f"stem {tension:4.2f} / {detach_limit:4.2f} N  {'DETACHED' if detached else 'ATTACHED'}", (25, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (110, 210, 255), 1, cv2.LINE_AA)
    cv2.putText(output, f"vine contact {obstacle:4.2f} N", (25, 119), cv2.FONT_HERSHEY_SIMPLEX, 0.52, obstacle_color, 1, cv2.LINE_AA)
    return output


def main() -> None:
    cfg = GrapePickEnvCfg()
    cfg.seed = args.seed
    cfg.episode_length_s = args.duration
    cfg.enable_cameras = not args.physics_only
    cfg.enable_gelsight = not args.disable_gelsight and not args.physics_only
    cfg.randomize_grape_pose = not args.no_randomization
    cfg.randomize_force_thresholds = args.randomize_force_thresholds
    cfg.third_person_camera.width = args.render_width
    cfg.third_person_camera.height = args.render_height
    cfg.wrist_camera.width = max(args.render_width // 2, 160)
    cfg.wrist_camera.height = max(args.render_height // 2, 120)
    if args.device:
        cfg.sim.device = args.device

    env = GrapePickEnv(cfg, render_mode="rgb_array" if cfg.enable_cameras else None)
    writer = None
    frames = 0
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    if args.record_video:
        if not cfg.enable_cameras:
            raise ValueError("--record_video cannot be combined with --physics_only")
        import imageio.v2 as imageio

        args.video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(args.video_path),
            fps=args.video_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )

    terminal_reason = "timeout"
    flags = None
    minimum_center_distance_m = float("inf")
    try:
        env.reset(seed=args.seed)
        max_steps = int(args.duration / env.physics_dt)
        for _ in range(max_steps):
            if not simulation_app.is_running():
                break
            env.command_expert()
            env._pre_physics_step(None)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            flags = env.get_task_flags()
            step = int(env.step_count[0].item())
            grape_pos_b = env.grape.data.root_pos_w - env._robot.data.root_link_pos_w
            gripper_center_b = env._gripper_center_pos_b()
            center_distance_m = float(
                (grape_pos_b[0] - gripper_center_b[0]).norm().item()
            )
            minimum_center_distance_m = min(minimum_center_distance_m, center_distance_m)
            if args.print_interval > 0 and step % args.print_interval == 0:
                print(
                    f"[STATE] step={step:04d} phase={env.get_expert_phase_names()[0]} "
                    f"grip={flags['grip_force_n'][0].item():.3f}N "
                    f"stem={flags['stem_tension_n'][0].item():.3f}N "
                    f"obstacle={flags['obstacle_contact_force_n'][0].item():.3f}N "
                    f"center_error={center_distance_m:.4f}m "
                    f"attached={_flag(flags, 'stem_attached')} "
                    f"success={_flag(flags, 'success')}"
                )
            render_due = cfg.enable_cameras and step % max(args.video_stride, 1) == 0
            if render_due:
                env.sim.render()
            if writer is not None and render_due:
                frame = env.get_camera_rgb("third")[0].cpu().numpy()
                writer.append_data(_annotate(frame, env, flags))
                frames += 1
            reached_terminal = False
            for reason in ("success", "crushed", "slipped", "obstacle_collision", "workspace_failure"):
                if _flag(flags, reason):
                    terminal_reason = reason
                    reached_terminal = True
                    break
            if reached_terminal and writer is not None and not render_due:
                env.sim.render()
                frame = env.get_camera_rgb("third")[0].cpu().numpy()
                writer.append_data(_annotate(frame, env, flags))
                frames += 1
            if reached_terminal:
                break
    finally:
        if writer is not None:
            writer.close()

    if flags is None:
        flags = env.get_task_flags()
    summary = {
        "task": "TacEx-GrapePick-Force-v0",
        "seed": args.seed,
        "terminal_reason": terminal_reason,
        "success": _flag(flags, "success"),
        "detached": _flag(flags, "detached"),
        "crushed": _flag(flags, "crushed"),
        "slipped": _flag(flags, "slipped"),
        "obstacle_collision": _flag(flags, "obstacle_collision"),
        "steps": int(env.step_count[0].item()),
        "peak_grip_force_n": float(env.peak_grip_force_n[0].item()),
        "peak_stem_tension_n": float(env.peak_stem_tension_n[0].item()),
        "peak_obstacle_contact_force_n": float(env.peak_obstacle_contact_force_n[0].item()),
        "minimum_gripper_center_distance_m": minimum_center_distance_m,
        "grip_force_definition": "maximum left/right pad-normal contact load",
        "grip_limit_n": cfg.damage_grip_force_threshold_n,
        "detach_threshold_n": cfg.detach_force_threshold_n,
        "contact_force_source": "PhysX ContactSensor filtered to target_grape",
        "video": str(args.video_path.resolve()) if writer is not None else None,
        "video_frames": frames,
        "gelsight_enabled": cfg.enable_gelsight,
    }
    args.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[RESULT] " + json.dumps(summary, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
