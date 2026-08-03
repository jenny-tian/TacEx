from __future__ import annotations

import argparse
import json
import os
import random
import sys
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

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a sim_robot Flow Matching policy on canonical randomized LabPick.")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--policy_root", type=Path, default=Path("bc_policy"))
parser.add_argument("--num_trials", type=int, default=20)
parser.add_argument("--seed", type=int, default=0, help="Environment randomization seed base.")
parser.add_argument("--policy_seed", type=int, default=None, help="Optional fixed Flow Matching noise seed, independent of environment seed.")
parser.add_argument(
    "--reset_policy_noise_each_chunk",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Reuse the same Flow Matching noise template for every replanned action chunk.",
)
parser.add_argument("--num_inference_steps", type=int, default=20)
parser.add_argument("--chunk_execute_steps", type=int, default=16)
parser.add_argument("--action_repeat", type=int, default=2, help="Physics steps per 60 Hz policy action.")
parser.add_argument("--episode_steps", type=int, default=960, help="Maximum 120 Hz physics steps per trial.")
parser.add_argument("--phase_horizon_steps", type=int, default=383, help="60 Hz policy steps corresponding to phase=1.")
parser.add_argument("--camera_warmup_steps", type=int, default=8, help="Held physics steps after reset so camera frames match the new scene.")
parser.add_argument("--visual_xy_lock_phase", type=float, default=0.30, help="Freeze visual XY before gripper closing; negative disables locking.")
parser.add_argument(
    "--close_onset_width_m",
    type=float,
    default=0.038,
    help="Record close-onset error when the commanded finger width first crosses this value.",
)
parser.add_argument("--print_state_interval", type=int, default=60)
parser.add_argument("--video_dir", type=Path, default=Path("logs/lab_pick_flow_eval_canonical"))
parser.add_argument("--video_every_n_steps", type=int, default=4)
parser.add_argument("--video_fps", type=int, default=30)
parser.add_argument(
    "--video_camera",
    choices=("third", "wrist", "viewer", "tactile_left", "tactile_right"),
    default="third",
)
parser.add_argument(
    "--policy_camera",
    choices=("wrist", "third"),
    default="wrist",
    help="Camera stream used as the policy's robot0_image input.",
)
parser.add_argument("--output", type=Path, default=Path("logs/lab_pick_flow_eval_canonical/results.json"))
parser.add_argument("--no_randomize_labware", action="store_true")
parser.add_argument(
    "--labware_random_xy",
    type=float,
    nargs=2,
    metavar=("X", "Y"),
    default=None,
    help="Uniform reset half-range in meters for labware x/y.",
)
parser.add_argument(
    "--labware_random_yaw",
    type=float,
    default=None,
    help="Uniform reset yaw half-range in radians.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from tacex_tasks.lab_pick.lab_pick_env import LabPickEnv
from tacex_tasks.lab_pick.lab_pick_env_cfg import LabPickEnvCfg


def _policy_observation(
    env: LabPickEnv, camera_name: str, phase: float | None = None
) -> dict[str, np.ndarray | float]:
    def camera_image(camera) -> np.ndarray:
        rgb = camera.data.output["rgb"][:, :, :, :3].permute(0, 3, 1, 2).float()
        rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        return rgb[0].permute(1, 2, 0).clamp(0, 255).byte().detach().cpu().numpy()

    state = env.get_cafe_observation()["robot0_pos"][0].detach().cpu().numpy().astype(np.float32)
    wrist_image = camera_image(env.wrist_camera)
    third_image = camera_image(env.third_person_camera)
    selected_image = wrist_image if camera_name == "wrist" else third_image
    observation: dict[str, np.ndarray | float] = {
        "robot0_pos": state,
        "robot0_image": selected_image,
        "robot0_image_third": third_image,
    }
    if phase is not None:
        observation["phase"] = float(phase)
    return observation


def _step_physics(env: LabPickEnv, action: torch.Tensor) -> None:
    env._pre_physics_step(action)
    env._apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    env.sim.render()


def _scalar(value: torch.Tensor) -> float:
    return float(value.reshape(-1)[0].item())


def _bool(value: torch.Tensor) -> bool:
    return bool(value.reshape(-1)[0].item())


def main() -> None:
    if args_cli.num_trials < 1:
        raise ValueError("--num_trials must be at least 1")
    if args_cli.action_repeat < 1:
        raise ValueError("--action_repeat must be at least 1")
    if not 1 <= args_cli.chunk_execute_steps <= 32:
        raise ValueError("--chunk_execute_steps must be in [1, 32]")
    if args_cli.phase_horizon_steps < 1:
        raise ValueError("--phase_horizon_steps must be at least 1")
    if args_cli.camera_warmup_steps < 0:
        raise ValueError("--camera_warmup_steps must be non-negative")
    if not 0.0 < args_cli.close_onset_width_m < 0.04:
        raise ValueError("--close_onset_width_m must be in (0, 0.04)")

    policy_root = args_cli.policy_root.expanduser().resolve()
    if str(policy_root) not in sys.path:
        sys.path.insert(0, str(policy_root))
    from sim_robot.deployment.policy_runner import SimActionChunkPolicyRunner

    cfg = LabPickEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args_cli.seed
    cfg.randomize_labware_position = not args_cli.no_randomize_labware
    if args_cli.labware_random_xy is not None:
        cfg.randomize_labware_position = True
        cfg.labware_pos_randomization_xy = tuple(args_cli.labware_random_xy)
    if args_cli.labware_random_yaw is not None:
        cfg.randomize_labware_position = True
        cfg.labware_yaw_randomization = args_cli.labware_random_yaw
    cfg.rl_normalized_actions = False
    # Data collection uses the yaw-aligned scripted target quaternion. The
    # policy predicts the same rotation, while this setting executes the
    # canonical environment's numerically stable yaw-aligned quaternion.
    cfg.rl_align_cafe_action_yaw = True
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device

    env = LabPickEnv(cfg, render_mode="rgb_array")
    runner = SimActionChunkPolicyRunner(
        checkpoint_path=args_cli.checkpoint.expanduser().resolve(),
        device=args_cli.device or "cuda",
        use_ema=True,
        num_inference_steps=args_cli.num_inference_steps,
        seed=args_cli.seed,
        visual_xy_lock_phase=(None if args_cli.visual_xy_lock_phase < 0.0 else args_cli.visual_xy_lock_phase),
    )
    observation_camera = "wrist" if len(runner.image_keys) > 1 else args_cli.policy_camera

    video_dir = args_cli.video_dir.expanduser().resolve()
    output_path = args_cli.output.expanduser().resolve()
    video_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    print(
        "[INFO] Canonical Flow Matching BC evaluation "
        f"trials={args_cli.num_trials} seed={args_cli.seed} action_repeat={args_cli.action_repeat} "
        f"chunk_execute_steps={args_cli.chunk_execute_steps} "
        f"num_inference_steps={args_cli.num_inference_steps} episode_steps={args_cli.episode_steps} "
        f"policy_camera={args_cli.policy_camera}",
        flush=True,
    )

    try:
        for trial in range(args_cli.num_trials):
            trial_seed = args_cli.seed + trial
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(trial_seed)

            env.reset(seed=trial_seed)
            hold_action = env.get_cafe_observation()["robot0_pos"].clone()
            for _ in range(args_cli.camera_warmup_steps):
                _step_physics(env, hold_action)
            env.step_count.zero_()
            env.has_touched.zero_()
            runner.reset()
            policy_trial_seed = (
                args_cli.policy_seed if args_cli.policy_seed is not None else trial_seed
            )
            if runner.generator is not None:
                runner.generator.manual_seed(policy_trial_seed)
            runner.update(_policy_observation(env, observation_camera, phase=0.0))

            reset_pos = env.labware_reset_pos_w[0].detach().cpu().tolist()
            reset_quat = env.labware_reset_quat_w[0].detach().cpu().tolist()
            frames: list[np.ndarray] = []
            physics_steps = 0
            policy_steps = 0
            success = False
            broken = False
            touched = False
            terminal_reason = ""
            max_lift_m = 0.0
            min_center_distance_m = float("inf")
            close_onset_xy_error_m = None
            close_onset_center_distance_m = None
            close_onset_physics_step = None
            first_contact_xy_error_m = None
            first_contact_center_distance_m = None
            first_contact_physics_step = None
            bilateral_contact_xy_error_m = None
            bilateral_contact_center_distance_m = None
            bilateral_contact_physics_step = None
            peak_net_force_n = 0.0
            peak_left_finger_force_n = 0.0
            peak_right_finger_force_n = 0.0
            peak_grip_force_n = 0.0
            peak_break_force_n = 0.0

            while simulation_app.is_running() and physics_steps < args_cli.episode_steps:
                if args_cli.reset_policy_noise_each_chunk and runner.generator is not None:
                    runner.generator.manual_seed(policy_trial_seed)
                action_chunk = runner.predict_action_chunk()
                stop = False

                for action_np in action_chunk[: args_cli.chunk_execute_steps]:
                    action = torch.as_tensor(
                        action_np,
                        device=env.device,
                        dtype=torch.float32,
                    ).view(1, -1)
                    for _ in range(args_cli.action_repeat):
                        _step_physics(env, action)
                        physics_steps += 1

                        flags = env._get_termination_flags()
                        lift_m = _scalar(env.labware.data.root_pos_w[:, 2] - env.initial_object_height)
                        object_pos_b = env.labware.data.root_pos_w - env._robot.data.root_link_pos_w
                        gripper_center_pos_b = env._gripper_center_pos_b()
                        center_delta_b = object_pos_b - gripper_center_pos_b
                        center_xy_error_m = _scalar(
                            torch.linalg.norm(center_delta_b[:, :2], dim=1)
                        )
                        center_distance_m = _scalar(torch.linalg.norm(center_delta_b, dim=1))
                        left_touch, right_touch = env.tactile_contact_depths()
                        left_contact = _scalar(left_touch) > env.cfg.tactile_threshold_mm
                        right_contact = _scalar(right_touch) > env.cfg.tactile_threshold_mm
                        gripper_width_m = _scalar(env.gripper_width[:, :1])

                        if close_onset_xy_error_m is None and gripper_width_m <= args_cli.close_onset_width_m:
                            close_onset_xy_error_m = center_xy_error_m
                            close_onset_center_distance_m = center_distance_m
                            close_onset_physics_step = physics_steps
                        if first_contact_xy_error_m is None and (left_contact or right_contact):
                            first_contact_xy_error_m = center_xy_error_m
                            first_contact_center_distance_m = center_distance_m
                            first_contact_physics_step = physics_steps
                        if bilateral_contact_xy_error_m is None and left_contact and right_contact:
                            bilateral_contact_xy_error_m = center_xy_error_m
                            bilateral_contact_center_distance_m = center_distance_m
                            bilateral_contact_physics_step = physics_steps

                        net_force_n = _scalar(flags["net_force_n"])
                        left_finger_force_n = _scalar(flags["left_finger_force_n"])
                        right_finger_force_n = _scalar(flags["right_finger_force_n"])
                        grip_force_n = _scalar(flags["grip_force_n"])
                        break_force_n = _scalar(flags["break_force_n"])
                        max_lift_m = max(max_lift_m, lift_m)
                        min_center_distance_m = min(min_center_distance_m, center_distance_m)
                        peak_net_force_n = max(peak_net_force_n, net_force_n)
                        peak_left_finger_force_n = max(peak_left_finger_force_n, left_finger_force_n)
                        peak_right_finger_force_n = max(peak_right_finger_force_n, right_finger_force_n)
                        peak_grip_force_n = max(peak_grip_force_n, grip_force_n)
                        peak_break_force_n = max(peak_break_force_n, break_force_n)
                        broken = broken or _bool(flags["object_broken"])
                        success = (success or _bool(flags["success"])) and not broken
                        touched = touched or left_contact or right_contact

                        if (
                            args_cli.video_every_n_steps > 0
                            and physics_steps % args_cli.video_every_n_steps == 0
                        ):
                            frames.append(env.get_video_frame(args_cli.video_camera))
                        if (
                            args_cli.print_state_interval > 0
                            and physics_steps % args_cli.print_state_interval == 0
                        ):
                            env.print_state()

                        if broken:
                            terminal_reason = "object_broken"
                        elif success:
                            terminal_reason = "success"
                        elif _bool(flags["object_dropped"]):
                            terminal_reason = "object_dropped"
                        elif _bool(flags["object_too_far"]):
                            terminal_reason = "object_too_far"
                        elif _bool(flags["ee_outside_workspace"]):
                            terminal_reason = "ee_outside_workspace"

                        if terminal_reason or physics_steps >= args_cli.episode_steps:
                            stop = True
                            break

                    policy_steps += 1
                    phase = min(policy_steps / float(args_cli.phase_horizon_steps), 1.0)
                    runner.update(_policy_observation(env, observation_camera, phase=phase))
                    if stop:
                        break

                if stop:
                    break

            if not terminal_reason:
                terminal_reason = "time_limit"

            video_path = video_dir / (
                f"slide_trial_{trial:03d}_{'success' if success else 'fail'}_{args_cli.video_camera}.mp4"
            )
            if frames:
                with imageio.get_writer(str(video_path), fps=args_cli.video_fps, macro_block_size=1) as writer:
                    for frame in frames:
                        writer.append_data(frame)

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
                "min_grasp_distance_m": min_center_distance_m,
                "min_center_distance_m": min_center_distance_m,
                "close_onset_xy_error_m": close_onset_xy_error_m,
                "close_onset_center_distance_m": close_onset_center_distance_m,
                "close_onset_physics_step": close_onset_physics_step,
                "first_contact_xy_error_m": first_contact_xy_error_m,
                "first_contact_center_distance_m": first_contact_center_distance_m,
                "first_contact_physics_step": first_contact_physics_step,
                "bilateral_contact_xy_error_m": bilateral_contact_xy_error_m,
                "bilateral_contact_center_distance_m": bilateral_contact_center_distance_m,
                "bilateral_contact_physics_step": bilateral_contact_physics_step,
                "peak_force_n": peak_break_force_n,
                "peak_net_force_n": peak_net_force_n,
                "peak_left_finger_force_n": peak_left_finger_force_n,
                "peak_right_finger_force_n": peak_right_finger_force_n,
                "peak_grip_force_n": peak_grip_force_n,
                "peak_break_force_n": peak_break_force_n,
                "reset_pos_w": reset_pos,
                "reset_quat_w": reset_quat,
                "video": str(video_path),
            }
            results.append(result)
            print(
                "[RESULT] "
                f"trial={trial} seed={trial_seed} success={success} broken={broken} touched={touched} "
                f"reason={terminal_reason} max_lift={max_lift_m:.4f}m "
                f"first_contact_xy={(first_contact_xy_error_m if first_contact_xy_error_m is not None else float('nan')):.4f}m "
                f"peak_break_force={peak_break_force_n:.4f}N peak_net_force={peak_net_force_n:.4f}N "
                f"physics_steps={physics_steps} reset_xy={[round(value, 4) for value in reset_pos[:2]]} "
                f"video={video_path}",
                flush=True,
            )
    finally:
        env.close()

    successes = sum(int(item["success"]) for item in results)
    broken_count = sum(int(item["broken"]) for item in results)
    touched_count = sum(int(item["touched"]) for item in results)
    summary = {
        "checkpoint": str(args_cli.checkpoint.expanduser().resolve()),
        "num_trials": len(results),
        "successes": successes,
        "success_rate": successes / max(len(results), 1),
        "broken": broken_count,
        "broken_rate": broken_count / max(len(results), 1),
        "touched": touched_count,
        "touch_rate": touched_count / max(len(results), 1),
        "seed": args_cli.seed,
        "policy_seed": args_cli.policy_seed,
        "reset_policy_noise_each_chunk": args_cli.reset_policy_noise_each_chunk,
        "action_repeat": args_cli.action_repeat,
        "chunk_execute_steps": args_cli.chunk_execute_steps,
        "num_inference_steps": args_cli.num_inference_steps,
        "policy_camera": args_cli.policy_camera,
        "policy_image_keys": list(runner.image_keys),
        "phase_horizon_steps": args_cli.phase_horizon_steps,
        "camera_warmup_steps": args_cli.camera_warmup_steps,
        "visual_xy_lock_phase": args_cli.visual_xy_lock_phase,
        "labware_random_xy": args_cli.labware_random_xy,
        "labware_random_yaw": args_cli.labware_random_yaw,
        "close_onset_width_m": args_cli.close_onset_width_m,
        "results": results,
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[SUMMARY] success_rate={successes}/{len(results)} ({summary['success_rate']:.2%}) "
        f"touch_rate={touched_count}/{len(results)} ({summary['touch_rate']:.2%}) "
        f"broken_rate={broken_count}/{len(results)} ({summary['broken_rate']:.2%}) "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
