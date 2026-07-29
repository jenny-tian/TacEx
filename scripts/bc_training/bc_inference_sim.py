#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "tacex_dinov3_fm_bc" / "best.pt"


def _bootstrap_repo_paths() -> None:
    for path in (
        SCRIPT_DIR,
        REPO_ROOT / "source" / "tacex",
        REPO_ROOT / "source" / "tacex_assets",
        REPO_ROOT / "source" / "tacex_tasks",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _bootstrap_isaaclab_source_paths() -> None:
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


def _bootstrap_isaacsim_warp_path() -> None:
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


_bootstrap_repo_paths()
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
    parser = argparse.ArgumentParser(description="Closed-loop LabPick simulation rollout for TacEx BC checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-prefer", "--checkpoint_prefer", dest="checkpoint_prefer", choices=("best", "latest"), default="best")
    parser.add_argument("--labware", choices=("slide", "coverslip", "cup"), default="slide")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--max_episode_steps", type=int, default=960)
    parser.add_argument("--aligned_hz", type=float, default=30.0)
    parser.add_argument("--break_force_threshold_n", type=float, default=6.0)
    parser.add_argument("--chunk_execute_steps", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--video_dir", "--video-dir", dest="video_dir", type=Path, default=None)
    parser.add_argument("--record_video", "--record-video", dest="record_video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--video_camera",
        "--video-camera",
        dest="video_camera",
        choices=("third", "wrist", "viewer", "tactile_left", "tactile_right"),
        default="third",
    )
    parser.add_argument("--video_every_n_steps", "--video-every-n-steps", dest="video_every_n_steps", type=int, default=4)
    parser.add_argument("--video_fps", "--video-fps", dest="video_fps", type=int, default=30)
    parser.add_argument("--success_lift_height", "--success-lift-height", dest="success_lift_height", type=float, default=0.20)
    parser.add_argument("--success_hold_steps", "--success-hold-steps", dest="success_hold_steps", type=int, default=60)
    parser.add_argument(
        "--success_gripper_distance",
        "--success-gripper-distance",
        dest="success_gripper_distance",
        type=float,
        default=0.08,
    )
    parser.add_argument("--no_ema", "--no-ema", dest="no_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reset_policy_seed_each_trial",
        "--reset-policy-seed-each-trial",
        dest="reset_policy_seed_each_trial",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--print_state_interval", "--print-state-interval", dest="print_state_interval", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 1:
        parser.error("bc_inference_sim.py currently supports --num_envs 1 only.")
    if args.aligned_hz <= 0.0:
        parser.error("--aligned_hz must be positive.")
    if args.max_episode_steps <= 0:
        parser.error("--max_episode_steps must be positive.")
    if args.chunk_execute_steps <= 0:
        parser.error("--chunk_execute_steps must be positive.")
    if args.num_trials <= 0:
        parser.error("--num_trials must be positive.")
    args.enable_cameras = True
    return args


args_cli = parse_args()
_patch_isaaclab_missing_exports()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio.v2 as imageio  # noqa: E402

import train_bc as bc  # noqa: E402
from tacex_tasks.lab_pick.lab_pick_env import LabPickEnv  # noqa: E402
from tacex_tasks.lab_pick.lab_pick_env_cfg import LabPickEnvCfg  # noqa: E402


@dataclass
class StableLiftSuccessTracker:
    lift_height_m: float
    hold_steps: int
    max_object_gripper_distance_m: float
    stable_steps: int = 0

    def update(self, lift_delta_m: float, has_touched: bool, object_gripper_distance_m: float) -> bool:
        stable = (
            has_touched
            and lift_delta_m >= self.lift_height_m
            and object_gripper_distance_m <= self.max_object_gripper_distance_m
        )
        self.stable_steps = self.stable_steps + 1 if stable else 0
        return self.stable_steps >= self.hold_steps


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def resolve_checkpoint(path: Path, prefer: str) -> Path:
    path = resolve_path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    preferred = path / f"{prefer}.pt"
    fallback = path / ("latest.pt" if prefer == "best" else "best.pt")
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No best.pt or latest.pt found under {path}")


def checkpoint_label(path: Path) -> str:
    if path.name in {"best.pt", "latest.pt"}:
        return f"{path.parent.name}_{path.stem}"
    return path.stem


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def _camera_rgb_224(camera_rgb: torch.Tensor) -> np.ndarray:
    rgb = camera_rgb[0, :, :, :3].permute(2, 0, 1).unsqueeze(0).float()
    rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
    rgb = rgb.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte()
    return rgb.detach().cpu().numpy()


def policy_state(env: LabPickEnv) -> np.ndarray:
    obs = env.get_cafe_observation()
    return obs["robot0_pos"][0].detach().cpu().numpy().astype(np.float32)


def policy_images(env: LabPickEnv) -> dict[str, np.ndarray]:
    return {
        "rgb": _camera_rgb_224(env.wrist_camera.data.output["rgb"]),
        "rgb_third": _camera_rgb_224(env.third_person_camera.data.output["rgb"]),
    }


def update_runner_from_env(runner: bc.FlowMatchingBCRunner, env: LabPickEnv) -> None:
    runner.update(policy_state(env), policy_images(env))


def _safe_normalize(vector: np.ndarray, eps: float = 1.0e-8) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < eps:
        return None
    return vector / norm


def _orthogonal_unit_vector(vector: np.ndarray) -> np.ndarray:
    basis = np.eye(3, dtype=np.float64)
    candidate = basis[int(np.argmin(np.abs(vector)))]
    orthogonal = candidate - vector * float(np.dot(vector, candidate))
    normalized = _safe_normalize(orthogonal)
    if normalized is None:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return normalized


def rot6d_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray | None:
    raw = np.asarray(rot6d, dtype=np.float64).reshape(3, 2)
    x_axis = _safe_normalize(raw[:, 0])
    if x_axis is None:
        return None

    y_axis = raw[:, 1] - x_axis * float(np.dot(x_axis, raw[:, 1]))
    y_axis = _safe_normalize(y_axis)
    if y_axis is None:
        y_axis = _orthogonal_unit_vector(x_axis)

    z_axis = _safe_normalize(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None
    y_axis = _safe_normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None

    matrix = np.stack((x_axis, y_axis, z_axis), axis=1)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        if diagonal[0] > diagonal[1] and diagonal[0] > diagonal[2]:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif diagonal[1] > diagonal[2]:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )

    quat = _safe_normalize(quat)
    if quat is None or not np.all(np.isfinite(quat)):
        return None
    return quat.astype(np.float32)


def apply_policy_action(env: LabPickEnv, action: np.ndarray) -> None:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < 10:
        raise ValueError(f"Expected at least 10 action dimensions, got {action.shape}.")

    target_pos = torch.as_tensor(action[:3], dtype=torch.float32, device=env.device).view(1, 3).repeat(env.num_envs, 1)
    target_pos = torch.minimum(torch.maximum(target_pos, env.workspace_min_b), env.workspace_max_b)

    quat = rot6d_to_quat_wxyz(action[3:9])
    if quat is None:
        target_quat = env.nominal_ee_quat_b.clone()
    else:
        target_quat = torch.as_tensor(quat, dtype=torch.float32, device=env.device).view(1, 4).repeat(env.num_envs, 1)

    target_width = float(np.clip(action[9], 0.0, 0.04))
    env.ik_commands[:, :3] = target_pos
    env.ik_commands[:, 3:7] = target_quat
    env.gripper_width[:] = target_width
    env.last_target_pos_b[:] = target_pos
    env.last_target_quat_b[:] = target_quat


def step_env(env: LabPickEnv) -> None:
    env._pre_physics_step(None)
    env._apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    env.sim.render()


def env_step(env: LabPickEnv) -> int:
    return int(env.step_count[0].item())


def lift_delta_m(env: LabPickEnv) -> float:
    lift_delta = env.labware.data.root_pos_w[:, 2] - env.initial_object_height
    return float(lift_delta[0].item())


def object_gripper_distance_m(env: LabPickEnv) -> float:
    object_pos_b = env.labware.data.root_pos_w - env._robot.data.root_link_pos_w
    tool_pos_b, _ = env._compute_frame_pose()
    return float(torch.linalg.norm(object_pos_b - tool_pos_b, dim=1)[0].item())


def failure_reasons(env: LabPickEnv) -> list[str]:
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


def make_output_dirs(checkpoint_path: Path) -> tuple[Path, Path]:
    output_dir = args_cli.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "outputs" / "bc_inference_sim" / f"{checkpoint_label(checkpoint_path)}_{args_cli.labware}"
    output_dir = resolve_path(output_dir)

    video_dir = args_cli.video_dir
    if video_dir is None:
        video_dir = output_dir / "videos"
    video_dir = resolve_path(video_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args_cli.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, video_dir


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")


def rollout_trial(
    env: LabPickEnv,
    runner: bc.FlowMatchingBCRunner,
    trial: int,
    *,
    control_interval_steps: int,
    chunk_execute_steps: int,
    video_dir: Path,
) -> dict[str, Any]:
    env.reset()
    runner.reset()
    if args_cli.reset_policy_seed_each_trial and runner.generator is not None:
        runner.generator.manual_seed(int(args_cli.seed))

    success_tracker = StableLiftSuccessTracker(
        lift_height_m=float(args_cli.success_lift_height),
        hold_steps=max(1, int(args_cli.success_hold_steps)),
        max_object_gripper_distance_m=float(args_cli.success_gripper_distance),
    )
    frames: list[np.ndarray] = []
    success = False
    failure_reason = "timeout"
    terminated = False

    update_runner_from_env(runner, env)

    while simulation_app.is_running() and env_step(env) < args_cli.max_episode_steps:
        action_chunk = runner.predict_action_chunk()
        for action in action_chunk[:chunk_execute_steps]:
            apply_policy_action(env, action)
            for _ in range(control_interval_steps):
                if not simulation_app.is_running() or env_step(env) >= args_cli.max_episode_steps:
                    break

                step_env(env)
                step = env_step(env)

                if args_cli.record_video and step % max(1, int(args_cli.video_every_n_steps)) == 0:
                    frames.append(env.get_video_frame(args_cli.video_camera))
                if args_cli.print_state_interval > 0 and step % args_cli.print_state_interval == 0:
                    env.print_state()

                dones, _ = env._get_dones()
                terminated = bool(dones[0].item())
                lift = lift_delta_m(env)
                touched = bool(env.has_touched[0].item())
                distance = object_gripper_distance_m(env)
                success = success_tracker.update(lift, touched, distance)

                if success:
                    failure_reason = ""
                    break
                if terminated:
                    reasons = failure_reasons(env)
                    failure_reason = "+".join(reasons) if reasons else "terminated"
                    break

            if success or terminated or env_step(env) >= args_cli.max_episode_steps or not simulation_app.is_running():
                break
            update_runner_from_env(runner, env)

        if success or terminated or env_step(env) >= args_cli.max_episode_steps:
            break

    if not simulation_app.is_running() and not success and not terminated:
        failure_reason = "simulation_stopped"
    elif env_step(env) >= args_cli.max_episode_steps and not success and not terminated:
        failure_reason = "timeout"

    video_path = ""
    if args_cli.record_video and frames:
        status = "success" if success else "fail"
        path = video_dir / f"{args_cli.labware}_trial_{trial:03d}_{status}_{args_cli.video_camera}.mp4"
        with imageio.get_writer(str(path), fps=int(args_cli.video_fps), macro_block_size=1) as video_writer:
            for frame in frames:
                video_writer.append_data(frame)
        video_path = str(path)

    row = {
        "trial": int(trial),
        "success": bool(success),
        "failure_reason": failure_reason,
        "steps": env_step(env),
        "duration_s": float(env_step(env) * env.physics_dt),
        "lift_delta_m": lift_delta_m(env),
        "has_touched": bool(env.has_touched[0].item()),
        "object_gripper_distance_m": object_gripper_distance_m(env),
        "stable_steps": int(success_tracker.stable_steps),
        "reset_pos_w": json.dumps(env.labware_reset_pos_w[0].detach().cpu().numpy().round(6).tolist()),
        "reset_quat_w": json.dumps(env.labware_reset_quat_w[0].detach().cpu().numpy().round(6).tolist()),
        "video_path": video_path,
    }
    print(
        "[RESULT] "
        f"trial={trial} success={success} steps={row['steps']} lift={row['lift_delta_m']:.4f}m "
        f"touched={row['has_touched']} object_gripper_distance={row['object_gripper_distance_m']:.4f}m "
        f"failure_reason={failure_reason or 'none'} video={video_path or 'none'}"
    )
    return row


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    checkpoint_path = resolve_checkpoint(args_cli.checkpoint, args_cli.checkpoint_prefer)
    output_dir, video_dir = make_output_dirs(checkpoint_path)
    policy_device = args_cli.device if getattr(args_cli, "device", None) is not None else "cuda"

    env_cfg = LabPickEnvCfg()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.labware_name = args_cli.labware
    env_cfg.seed = int(args_cli.seed)
    env_cfg.success_lift_height = float(args_cli.success_lift_height)
    env_cfg.terminate_break_force_threshold_n = float(args_cli.break_force_threshold_n)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env: LabPickEnv | None = None
    rows: list[dict[str, Any]] = []
    summary_path = output_dir / "summary.json"
    results_path = output_dir / "results.csv"

    try:
        env = LabPickEnv(env_cfg, render_mode="rgb_array")
        control_interval_steps = max(1, round((1.0 / float(args_cli.aligned_hz)) / float(env.physics_dt)))
        runner = bc.FlowMatchingBCRunner(
            checkpoint_path=checkpoint_path,
            device=policy_device,
            use_ema=not args_cli.no_ema,
            num_inference_steps=args_cli.num_inference_steps,
            seed=int(args_cli.seed),
        )
        chunk_execute_steps = min(max(1, int(args_cli.chunk_execute_steps)), int(runner.config.chunk_size))

        print(
            "[INFO] Closed-loop BC rollout: "
            f"trials={args_cli.num_trials}, checkpoint={checkpoint_path}, labware={args_cli.labware}, "
            f"control_interval_steps={control_interval_steps}, chunk_execute_steps={chunk_execute_steps}, "
            f"num_inference_steps={args_cli.num_inference_steps if args_cli.num_inference_steps is not None else runner.config.num_inference_steps}, "
            f"output_dir={output_dir}"
        )

        for trial in range(int(args_cli.num_trials)):
            if not simulation_app.is_running():
                break
            row = rollout_trial(
                env,
                runner,
                trial,
                control_interval_steps=control_interval_steps,
                chunk_execute_steps=chunk_execute_steps,
                video_dir=video_dir,
            )
            rows.append(row)
            write_results_csv(results_path, rows)

        successes = sum(1 for row in rows if row["success"])
        summary = {
            "checkpoint": str(checkpoint_path),
            "model_type": runner.checkpoint.get("model_type"),
            "epoch": runner.checkpoint.get("epoch"),
            "global_step": runner.checkpoint.get("global_step"),
            "use_ema": not args_cli.no_ema,
            "device": str(runner.device),
            "labware": args_cli.labware,
            "num_trials": int(args_cli.num_trials),
            "completed_trials": len(rows),
            "successes": int(successes),
            "success_rate": float(successes / max(len(rows), 1)),
            "aligned_hz": float(args_cli.aligned_hz),
            "control_interval_steps": int(control_interval_steps),
            "chunk_execute_steps": int(chunk_execute_steps),
            "max_episode_steps": int(args_cli.max_episode_steps),
            "break_force_threshold_n": float(args_cli.break_force_threshold_n),
            "success_lift_height": float(args_cli.success_lift_height),
            "success_hold_steps": int(args_cli.success_hold_steps),
            "success_gripper_distance": float(args_cli.success_gripper_distance),
            "record_video": bool(args_cli.record_video),
            "video_dir": str(video_dir),
            "results": rows,
        }
        write_summary(summary_path, summary)
        print(f"[SUMMARY] success_rate={successes}/{len(rows)} ({successes / max(len(rows), 1):.2%})")
        print(f"[INFO] wrote results={results_path} summary={summary_path}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
