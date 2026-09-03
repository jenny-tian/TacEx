#!/usr/bin/env python
"""Closed-loop Isaac Lab evaluation for DINOv3 ACT or Flow BC checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
    SCRIPT_DIR,
    REPO_ROOT / "source" / "tacex",
    REPO_ROOT / "source" / "tacex_assets",
    REPO_ROOT / "source" / "tacex_tasks",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _bootstrap_isaaclab_source_paths() -> None:
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        return
    source_root = Path(spec.origin).resolve().parent / "source"
    for package_name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
        package_source = source_root / package_name
        if (package_source / package_name).is_dir() and str(package_source) not in sys.path:
            sys.path.insert(0, str(package_source))


def _bootstrap_isaacsim_warp_path() -> None:
    spec = importlib.util.find_spec("isaacsim")
    if spec is None or spec.origin is None:
        return
    extcache_dir = Path(spec.origin).resolve().parent / "extscache"
    for path in sorted(extcache_dir.glob("omni.warp.core-*"), reverse=True):
        if (path / "warp" / "__init__.py").is_file():
            sys.path.insert(0, str(path))
            return


_bootstrap_isaaclab_source_paths()
_bootstrap_isaacsim_warp_path()

from isaaclab.app import AppLauncher  # noqa: E402


def _patch_isaaclab_missing_exports() -> None:
    import isaaclab.utils as isaaclab_utils
    from isaaclab.utils.buffers.circular_buffer import CircularBuffer
    from isaaclab.utils.buffers.delay_buffer import DelayBuffer
    from isaaclab.utils.buffers.timestamped_buffer import TimestampedBuffer

    isaaclab_utils.CircularBuffer = CircularBuffer
    isaaclab_utils.DelayBuffer = DelayBuffer
    isaaclab_utils.TimestampedBuffer = TimestampedBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DINOv3 ACT or Flow BC policy on LabPick")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--episode-steps", type=int, default=960)
    parser.add_argument("--action-repeat", type=int, default=2, help="Physics steps per 60 Hz policy action")
    parser.add_argument("--chunk-execute-steps", type=int, default=16)
    parser.add_argument("--camera-warmup-steps", type=int, default=8)
    parser.add_argument("--labware-random-xy", type=float, nargs=2, default=(0.10, 0.10))
    parser.add_argument("--labware-random-yaw-degrees", type=float, default=0.0)
    parser.add_argument(
        "--align-action-yaw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace predicted action rotation with simulator yaw alignment; disabled for oracle-free evaluation.",
    )
    parser.add_argument("--break-force-threshold-n", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-camera", choices=("third", "wrist", "viewer"), default="third")
    parser.add_argument("--video-every-n-steps", type=int, default=4)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--print-state-interval", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_trials < 1:
        parser.error("--num-trials must be positive")
    if args.action_repeat < 1 or args.chunk_execute_steps < 1:
        parser.error("--action-repeat and --chunk-execute-steps must be positive")
    args.enable_cameras = True
    return args


args_cli = parse_args()
_patch_isaaclab_missing_exports()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio.v2 as imageio  # noqa: E402

import dinov3_act as act  # noqa: E402
import dinov3_flow as flow  # noqa: E402
from tacex_tasks.lab_pick.lab_pick_env import LabPickEnv  # noqa: E402
from tacex_tasks.lab_pick.lab_pick_env_cfg import LabPickEnvCfg  # noqa: E402


def scalar(value: torch.Tensor) -> float:
    return float(value.reshape(-1)[0].item())


def flag(value: torch.Tensor) -> bool:
    return bool(value.reshape(-1)[0].item())


def camera_rgb_224(camera_rgb: torch.Tensor) -> np.ndarray:
    rgb = camera_rgb[0, :, :, :3].permute(2, 0, 1).unsqueeze(0).float()
    rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
    return rgb[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()


def policy_observation(
    env: LabPickEnv, *, include_force: bool
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cafe = env.get_cafe_observation()
    state = cafe["robot0_pos"][0].detach().cpu().numpy().astype(np.float32)
    if include_force:
        force = cafe["robot0_force"][0].detach().cpu().numpy().astype(np.float32)
        state = np.concatenate((state, force))
    wrist = camera_rgb_224(env.wrist_camera.data.output["rgb"])
    third = camera_rgb_224(env.third_person_camera.data.output["rgb"])
    images = {
        "rgb": wrist,
        "rgb_third": third,
        "robot0_image": wrist,
        "robot0_image_third": third,
    }
    return state, images


def physics_step(env: LabPickEnv, action: torch.Tensor, *, render: bool = True) -> None:
    env._pre_physics_step(action)
    env._apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    if render:
        env.sim.render()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(summary), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    checkpoint = args_cli.checkpoint.expanduser().resolve()
    output_path = args_cli.output.expanduser().resolve()
    video_dir = (args_cli.video_dir or output_path.parent / "videos").expanduser().resolve()
    if args_cli.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    cfg = LabPickEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args_cli.seed
    cfg.labware_name = "slide"
    cfg.randomize_labware_position = True
    cfg.labware_pos_randomization_xy = tuple(args_cli.labware_random_xy)
    cfg.labware_yaw_randomization = math.radians(args_cli.labware_random_yaw_degrees)
    cfg.terminate_break_force_threshold_n = args_cli.break_force_threshold_n
    cfg.rl_normalized_actions = False
    cfg.rl_align_cafe_action_yaw = args_cli.align_action_yaw
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device

    env = LabPickEnv(cfg, render_mode="rgb_array")
    checkpoint_header = torch.load(checkpoint, map_location="cpu", weights_only=False)
    runner_type = (
        flow.DINOv3FlowRunner
        if checkpoint_header.get("model_type") == flow.MODEL_TYPE
        else act.DINOv3ACTRunner
    )
    del checkpoint_header
    runner = runner_type(checkpoint, device=args_cli.device or "cuda")
    include_force = bool(runner.checkpoint.get("train_config", {}).get("include_force", False))
    expected_state_dim = 16 if include_force else 10
    if runner.config.state_dim != expected_state_dim:
        raise ValueError(
            f"Checkpoint force metadata expects state_dim={expected_state_dim}, "
            f"got {runner.config.state_dim}."
        )
    execute_steps = min(args_cli.chunk_execute_steps, runner.config.chunk_size)
    results: list[dict[str, Any]] = []
    print(
        "[INFO] DINOv3 policy evaluation "
        f"checkpoint={checkpoint} trials={args_cli.num_trials} seed={args_cli.seed} "
        f"xy={tuple(args_cli.labware_random_xy)} yaw_deg={args_cli.labware_random_yaw_degrees} "
        f"action_repeat={args_cli.action_repeat} chunk_execute_steps={execute_steps}",
        flush=True,
    )
    try:
        for trial in range(args_cli.num_trials):
            trial_seed = args_cli.seed + trial
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            torch.cuda.manual_seed_all(trial_seed)
            env.reset(seed=trial_seed)

            # The reset pose changes after the first rendered frame.  Holding
            # the initial robot pose avoids feeding a stale camera observation.
            hold_action = env.get_cafe_observation()["robot0_pos"].clone()
            for _ in range(args_cli.camera_warmup_steps):
                physics_step(env, hold_action)
            env.step_count.zero_()
            env.has_touched.zero_()
            runner.reset()
            if getattr(runner, "generator", None) is not None:
                runner.generator.manual_seed(trial_seed)
            state, images = policy_observation(env, include_force=include_force)
            runner.update(state, images, phase=0.0)

            reset_pos = env.labware_reset_pos_w[0].detach().cpu().tolist()
            reset_quat = env.labware_reset_quat_w[0].detach().cpu().tolist()
            physics_steps = 0
            policy_steps = 0
            success = False
            broken = False
            touched = False
            terminal_reason = ""
            max_lift_m = 0.0
            min_center_distance_m = float("inf")
            peak_break_force_n = 0.0
            first_contact_xy_error_m: float | None = None
            frames: list[np.ndarray] = []

            while simulation_app.is_running() and physics_steps < args_cli.episode_steps and not terminal_reason:
                action_chunk = runner.predict_action_chunk()
                for action_np in action_chunk[:execute_steps]:
                    action = torch.as_tensor(action_np, device=env.device, dtype=torch.float32).view(1, -1)
                    for repeat_index in range(args_cli.action_repeat):
                        physics_step(env, action, render=repeat_index == args_cli.action_repeat - 1)
                        physics_steps += 1
                        flags = env._get_termination_flags()
                        lift_m = scalar(env.labware.data.root_pos_w[:, 2] - env.initial_object_height)
                        object_pos_b = env.labware.data.root_pos_w - env._robot.data.root_link_pos_w
                        center_delta = object_pos_b - env._gripper_center_pos_b()
                        center_distance = scalar(torch.linalg.norm(center_delta, dim=1))
                        center_xy_error = scalar(torch.linalg.norm(center_delta[:, :2], dim=1))
                        left_touch, right_touch = env.tactile_contact_depths()
                        has_contact = scalar(left_touch) > env.cfg.tactile_threshold_mm or scalar(right_touch) > env.cfg.tactile_threshold_mm
                        touched = touched or has_contact
                        if has_contact and first_contact_xy_error_m is None:
                            first_contact_xy_error_m = center_xy_error
                        max_lift_m = max(max_lift_m, lift_m)
                        min_center_distance_m = min(min_center_distance_m, center_distance)
                        peak_break_force_n = max(peak_break_force_n, scalar(flags["break_force_n"]))
                        broken = broken or flag(flags["object_broken"])
                        success = (success or flag(flags["success"])) and not broken

                        if args_cli.record_video and physics_steps % args_cli.video_every_n_steps == 0:
                            frames.append(env.get_video_frame(args_cli.video_camera))
                        if args_cli.print_state_interval and physics_steps % args_cli.print_state_interval == 0:
                            env.print_state()
                        if broken:
                            terminal_reason = "object_broken"
                        elif success:
                            terminal_reason = "success"
                        elif flag(flags["object_dropped"]):
                            terminal_reason = "object_dropped"
                        elif flag(flags["object_too_far"]):
                            terminal_reason = "object_too_far"
                        elif flag(flags["ee_outside_workspace"]):
                            terminal_reason = "ee_outside_workspace"
                        if terminal_reason or physics_steps >= args_cli.episode_steps:
                            break
                    policy_steps += 1
                    phase = min(policy_steps / float(runner.config.phase_horizon_steps), 1.0)
                    state, images = policy_observation(env, include_force=include_force)
                    runner.update(state, images, phase=phase)
                    if terminal_reason or physics_steps >= args_cli.episode_steps:
                        break
            if not terminal_reason:
                terminal_reason = "time_limit"

            video_path = ""
            if frames:
                path = video_dir / f"slide_trial_{trial:03d}_{'success' if success else 'fail'}_{args_cli.video_camera}.mp4"
                with imageio.get_writer(str(path), fps=args_cli.video_fps, macro_block_size=1) as writer:
                    for frame in frames:
                        writer.append_data(frame)
                video_path = str(path)
            result = {
                "trial": trial,
                "seed": trial_seed,
                "success": success,
                "broken": broken,
                "touched": touched,
                "terminal_reason": terminal_reason,
                "physics_steps": physics_steps,
                "policy_steps": policy_steps,
                "max_lift_m": max_lift_m,
                "min_center_distance_m": min_center_distance_m,
                "first_contact_xy_error_m": first_contact_xy_error_m,
                "peak_break_force_n": peak_break_force_n,
                "reset_pos_w": reset_pos,
                "reset_quat_w": reset_quat,
                "video": video_path,
            }
            results.append(result)
            successes = sum(int(item["success"]) for item in results)
            partial = {
                "checkpoint": str(checkpoint),
                "completed_trials": len(results),
                "successes": successes,
                "success_rate": successes / len(results),
                "results": results,
            }
            write_summary(output_path, partial)
            print(
                f"[RESULT] trial={trial} seed={trial_seed} success={success} reason={terminal_reason} "
                f"max_lift={max_lift_m:.4f}m touched={touched} peak_break_force={peak_break_force_n:.3f}N "
                f"running_success={successes}/{len(results)}",
                flush=True,
            )
    finally:
        env.close()

    successes = sum(int(item["success"]) for item in results)
    broken = sum(int(item["broken"]) for item in results)
    touched = sum(int(item["touched"]) for item in results)
    failure_counts = Counter(item["terminal_reason"] for item in results if not item["success"])
    summary = {
        "checkpoint": str(checkpoint),
        "model_type": runner.checkpoint.get("model_type"),
        "checkpoint_epoch": runner.checkpoint.get("epoch"),
        "checkpoint_validation_loss": runner.checkpoint.get("validation_loss"),
        "include_force": include_force,
        "policy_state_dim": runner.config.state_dim,
        "num_trials": len(results),
        "successes": successes,
        "success_rate": successes / max(len(results), 1),
        "broken": broken,
        "broken_rate": broken / max(len(results), 1),
        "touched": touched,
        "touch_rate": touched / max(len(results), 1),
        "failure_counts": dict(failure_counts),
        "seed": args_cli.seed,
        "action_repeat": args_cli.action_repeat,
        "chunk_execute_steps": execute_steps,
        "phase_horizon_steps": runner.config.phase_horizon_steps,
        "camera_warmup_steps": args_cli.camera_warmup_steps,
        "break_force_threshold_n": args_cli.break_force_threshold_n,
        "labware_random_xy_m": list(args_cli.labware_random_xy),
        "labware_random_yaw_degrees": args_cli.labware_random_yaw_degrees,
        "rl_align_cafe_action_yaw": args_cli.align_action_yaw,
        "results": results,
    }
    write_summary(output_path, summary)
    print(
        f"[SUMMARY] success={successes}/{len(results)} ({summary['success_rate']:.2%}) "
        f"broken={broken}/{len(results)} touched={touched}/{len(results)} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
