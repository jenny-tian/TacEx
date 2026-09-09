#!/usr/bin/env python3
"""Collect one force-safe RaspberryPick trajectory and record an annotated MP4."""

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
    for package in (
        "isaaclab",
        "isaaclab_assets",
        "isaaclab_tasks",
        "isaaclab_rl",
        "isaaclab_mimic",
    ):
        package_root = source_root / package
        if (package_root / package).is_dir() and str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))


_reexec_with_cuda_libraries()
_bootstrap_isaaclab()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=930000)
parser.add_argument("--max-episode-steps", type=int, default=900)
parser.add_argument("--record-stride", type=int, default=4)
parser.add_argument("--video-stride", type=int, default=4)
parser.add_argument("--video-fps", type=float, default=30.0)
parser.add_argument("--no-randomization", action="store_true")
parser.add_argument("--randomize-force-thresholds", action="store_true")
parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("exp_report/raspberry_pick/data_collection"),
)
parser.add_argument(
    "--video-path",
    type=Path,
    default=Path("exp_report/raspberry_pick/raspberry_data_collection.mp4"),
)
parser.add_argument(
    "--first-video-path",
    type=Path,
    default=Path("exp_report/raspberry_pick/raspberry_data_collection_first_person.mp4"),
)
parser.add_argument(
    "--third-video-path",
    type=Path,
    default=Path("exp_report/raspberry_pick/raspberry_data_collection_third_person.mp4"),
)
parser.add_argument(
    "--paper-first-video-path",
    type=Path,
    default=Path("exp_report/raspberry_pick/raspberry_pick_paper_first_person.mp4"),
)
parser.add_argument(
    "--paper-third-video-path",
    type=Path,
    default=Path("exp_report/raspberry_pick/raspberry_pick_paper_third_person.mp4"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.max_episode_steps < 1 or args.record_stride < 1 or args.video_stride < 1:
    parser.error("step counts and strides must be positive")
if args.video_fps <= 0.0:
    parser.error("--video-fps must be positive")
args.enable_cameras = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from tacex_tasks.raspberry_pick.dataset import RaspberryTrajectoryBuffer, validate_episode
from tacex_tasks.raspberry_pick.raspberry_pick_env import RaspberryPickEnv
from tacex_tasks.raspberry_pick.raspberry_pick_env_cfg import RaspberryPickEnvCfg


def _flag(flags: dict[str, torch.Tensor], key: str) -> bool:
    return bool(flags[key][0].item())


def _rgb_224(image: torch.Tensor) -> torch.Tensor:
    image_chw = image[..., :3].permute(0, 3, 1, 2).float()
    resized = F.interpolate(image_chw, size=(224, 224), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1).clamp(0, 255).byte()


def _terminal_reason(flags: dict[str, torch.Tensor], timeout: bool) -> str:
    for key in ("success", "crushed", "slipped", "obstacle_collision", "workspace_failure"):
        if _flag(flags, key):
            return key
    if timeout:
        return "no_detach" if _flag(flags, "stem_attached") else "timeout_after_detach"
    return ""


def _annotate(
    env: RaspberryPickEnv,
    flags: dict[str, torch.Tensor],
    step: int,
    terminal_reason: str,
) -> np.ndarray:
    third = env.get_camera_rgb("third")[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    wrist = env.get_camera_rgb("wrist")[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    output = np.ascontiguousarray(third.copy())
    wrist_small = cv2.resize(np.ascontiguousarray(wrist), (288, 216), interpolation=cv2.INTER_AREA)
    inset_x = output.shape[1] - wrist_small.shape[1] - 18
    inset_y = output.shape[0] - wrist_small.shape[0] - 18
    cv2.rectangle(
        output,
        (inset_x - 4, inset_y - 28),
        (output.shape[1] - 14, output.shape[0] - 14),
        (245, 245, 245),
        thickness=3,
    )
    output[inset_y : inset_y + wrist_small.shape[0], inset_x : inset_x + wrist_small.shape[1]] = wrist_small
    cv2.rectangle(output, (inset_x, inset_y - 25), (inset_x + 158, inset_y), (18, 22, 18), -1)
    cv2.putText(
        output,
        "WRIST CAMERA",
        (inset_x + 8, inset_y - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

    phase = env.get_expert_phase_names()[0].upper()
    state = terminal_reason.replace("_", " ").upper() if terminal_reason else phase
    state_color = (70, 230, 105) if terminal_reason == "success" else (245, 245, 245)
    if terminal_reason and terminal_reason != "success":
        state_color = (255, 185, 65)
    grip = float(flags["grip_force_n"][0].item())
    tension = float(flags["stem_tension_n"][0].item())
    damage = float(flags["damage_threshold_n"][0].item())
    detach = float(flags["detach_threshold_n"][0].item())
    cv2.rectangle(output, (14, 14), (565, 154), (14, 22, 17), -1)
    cv2.putText(
        output,
        "RASPBERRY PICK | EXPERT DATA COLLECTION",
        (28, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"{state}  |  t={step / 120.0:4.2f}s  |  frame={step // args.record_stride}",
        (28, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        state_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"pad grip  {grip:4.2f} / {damage:4.2f} N  (damage limit)",
        (28, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (95, 230, 115) if grip < damage else (250, 75, 75),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"pedicle   {tension:4.2f} / {detach:4.2f} N  (detach target)",
        (28, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (95, 205, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def _annotate_single_view(
    image: np.ndarray,
    env: RaspberryPickEnv,
    flags: dict[str, torch.Tensor],
    step: int,
    terminal_reason: str,
    view_name: str,
) -> np.ndarray:
    """Add the same compact task telemetry to an independent camera stream."""
    output = np.ascontiguousarray(image.copy())
    phase = env.get_expert_phase_names()[0].upper()
    state = terminal_reason.replace("_", " ").upper() if terminal_reason else phase
    state_color = (70, 230, 105) if terminal_reason == "success" else (245, 245, 245)
    if terminal_reason and terminal_reason != "success":
        state_color = (255, 185, 65)
    grip = float(flags["grip_force_n"][0].item())
    tension = float(flags["stem_tension_n"][0].item())
    damage = float(flags["damage_threshold_n"][0].item())
    detach = float(flags["detach_threshold_n"][0].item())
    panel_width = min(output.shape[1] - 28, 545)
    cv2.rectangle(output, (14, 14), (14 + panel_width, 127), (14, 22, 17), -1)
    cv2.putText(
        output,
        f"RASPBERRY PICK | {view_name}",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"{state}  |  t={step / 120.0:4.2f}s",
        (28, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        state_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"grip {grip:4.2f}/{damage:4.2f} N   pedicle {tension:4.2f}/{detach:4.2f} N",
        (28, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (95, 230, 115) if grip < damage else (250, 75, 75),
        1,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    output_root = args.output_root.expanduser().resolve()
    dataset_root = output_root / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in dataset_root.glob("record_*") if path.is_dir())
    record_dir = dataset_root / f"record_{len(existing):06d}"
    video_path = args.video_path.expanduser().resolve()
    first_video_path = args.first_video_path.expanduser().resolve()
    third_video_path = args.third_video_path.expanduser().resolve()
    paper_first_video_path = args.paper_first_video_path.expanduser().resolve()
    paper_third_video_path = args.paper_third_video_path.expanduser().resolve()
    for path in (
        video_path,
        first_video_path,
        third_video_path,
        paper_first_video_path,
        paper_third_video_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    cfg = RaspberryPickEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.enable_cameras = True
    cfg.enable_gelsight = False
    cfg.randomize_grape_pose = not args.no_randomization
    cfg.randomize_force_thresholds = args.randomize_force_thresholds
    cfg.episode_length_s = max(args.max_episode_steps * cfg.sim.dt, 1.0)
    if args.device:
        cfg.sim.device = args.device

    env = RaspberryPickEnv(cfg, render_mode="rgb_array")
    buffer = RaspberryTrajectoryBuffer(1)
    writer = imageio.get_writer(
        str(video_path),
        fps=args.video_fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    first_writer = imageio.get_writer(
        str(first_video_path),
        fps=args.video_fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    third_writer = imageio.get_writer(
        str(third_video_path),
        fps=args.video_fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    paper_first_writer = imageio.get_writer(
        str(paper_first_video_path),
        fps=args.video_fps,
        codec="libx264",
        quality=9,
        macro_block_size=None,
    )
    paper_third_writer = imageio.get_writer(
        str(paper_third_video_path),
        fps=args.video_fps,
        codec="libx264",
        quality=9,
        macro_block_size=None,
    )
    active = torch.ones(1, dtype=torch.bool, device=env.device)
    flags: dict[str, torch.Tensor] | None = None
    reason = ""
    frame_count = 0
    try:
        env.reset(seed=args.seed)
        for step_index in range(args.max_episode_steps):
            if not simulation_app.is_running():
                break
            env.command_expert()
            action = env.get_cafe_action()
            if step_index % args.record_stride == 0:
                env.sim.render()
                buffer.append(
                    env,
                    action,
                    active,
                    rgb=_rgb_224(env.get_camera_rgb("wrist")),
                    rgb_third=_rgb_224(env.get_camera_rgb("third")),
                )
            env._pre_physics_step(None)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            flags = env.get_task_flags()
            step = int(env.step_count[0].item())
            timed_out = step_index == args.max_episode_steps - 1
            reason = _terminal_reason(flags, timed_out)
            render_due = step % args.video_stride == 0 or bool(reason)
            if render_due:
                env.sim.render()
                frame = _annotate(env, flags, step, reason)
                first_rgb = env.get_camera_rgb("wrist")[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
                third_rgb = env.get_camera_rgb("third")[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
                first_frame = _annotate_single_view(
                    first_rgb, env, flags, step, reason, "FIRST-PERSON WRIST CAMERA"
                )
                third_frame = _annotate_single_view(
                    third_rgb, env, flags, step, reason, "THIRD-PERSON CAMERA"
                )
                writer.append_data(frame)
                first_writer.append_data(first_frame)
                third_writer.append_data(third_frame)
                paper_first_writer.append_data(np.ascontiguousarray(first_rgb))
                paper_third_writer.append_data(np.ascontiguousarray(third_rgb))
                frame_count += 1
            if step % 120 == 0:
                print(
                    f"[DATA] step={step}/{args.max_episode_steps} "
                    f"phase={env.get_expert_phase_names()[0]} "
                    f"grip={flags['grip_force_n'][0].item():.3f}N "
                    f"stem={flags['stem_tension_n'][0].item():.3f}N",
                    flush=True,
                )
            if reason:
                active[:] = False
                for _ in range(max(1, int(round(args.video_fps)))):
                    writer.append_data(frame)
                    first_writer.append_data(first_frame)
                    third_writer.append_data(third_frame)
                    paper_first_writer.append_data(np.ascontiguousarray(first_rgb))
                    paper_third_writer.append_data(np.ascontiguousarray(third_rgb))
                    frame_count += 1
                break
    finally:
        writer.close()
        first_writer.close()
        third_writer.close()
        paper_first_writer.close()
        paper_third_writer.close()

    if flags is None:
        flags = env.get_task_flags()
    success = _flag(flags, "success")
    buffer.save_episode(
        0,
        record_dir,
        success=success,
        failure_reason="" if success else (reason or "interrupted"),
        env=env,
        seed=args.seed,
        sample_period_s=env.physics_dt * args.record_stride,
    )
    validation = validate_episode(record_dir)
    summary = {
        "task": "TacEx-RaspberryPick-Force-v0",
        "seed": args.seed,
        "success": success,
        "terminal_reason": reason or "interrupted",
        "steps": int(env.step_count[0].item()),
        "record": str(record_dir),
        "record_samples": int(validation["length"]),
        "video": str(video_path),
        "first_person_video": str(first_video_path),
        "third_person_video": str(third_video_path),
        "paper_first_person_video": str(paper_first_video_path),
        "paper_third_person_video": str(paper_third_video_path),
        "video_frames": frame_count,
        "video_fps": args.video_fps,
        "grip_damage_threshold_n": float(env.damage_threshold_n[0].item()),
        "pedicle_detach_threshold_n": float(env.detach_threshold_n[0].item()),
        "peak_grip_force_n": float(env.peak_grip_force_n[0].item()),
        "peak_pedicle_tension_n": float(env.peak_stem_tension_n[0].item()),
        "randomize_raspberry_pose": cfg.randomize_grape_pose,
        "raspberry_position_randomization_m": list(cfg.grape_position_randomization_m),
        "randomize_force_thresholds": cfg.randomize_force_thresholds,
        "reference_image": (
            "/home/limx/.codex/attachments/79a40e8c-1ee4-4125-b90f-0d1601d60a8c/image-1.png"
        ),
    }
    summary_path = output_root / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[SUMMARY] " + json.dumps(summary), flush=True)
    env.close()
    if not success:
        raise RuntimeError(f"Raspberry data-collection episode failed: {reason}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # SimulationApp.close() may terminate Kit before Python gets a chance
        # to emit the traceback, so print it explicitly first.
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
