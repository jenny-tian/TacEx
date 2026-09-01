from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate native Flow BC or clean DSRL-SAC on LabPick.")
parser.add_argument("--bc_policy", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--task", default="TacEx-LabPick-Slide-Clean-DSRL-SAC-v0")
parser.add_argument("--num_trials", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mode", choices=("auto", "native_bc", "zero_noise", "deterministic", "stochastic"), default="auto")
parser.add_argument("--bc_device", type=str, default="cuda:0")
parser.add_argument("--learned_noise_steps", type=int, default=1)
parser.add_argument("--noise_padding_mode", choices=("repeat_last", "zeros"), default="repeat_last")
parser.add_argument("--chunk_execute_steps", type=int, default=32)
parser.add_argument("--chunk_discount", type=float, default=0.99)
parser.add_argument("--flow_num_inference_steps", type=int, default=20)
parser.add_argument("--phase_horizon_steps", type=int, default=383)
parser.add_argument("--camera_warmup_steps", type=int, default=8)
parser.add_argument(
    "--use_visual_xy_override",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Match the frozen BC visual-XY override used during training.",
)
parser.add_argument("--labware_random_xy_m", type=float, nargs=2, default=(0.10, 0.10))
parser.add_argument("--labware_random_yaw_deg", type=float, default=45.0)
parser.add_argument("--break_force_threshold_n", type=float, default=4.0)
parser.add_argument("--max_outer_steps", type=int, default=0)
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Optional directory for one MP4 per evaluation trial.",
)
parser.add_argument("--video_prefix", type=str, default="clean-dsrl")
parser.add_argument("--video_fps", type=int, default=30)
parser.add_argument(
    "--record_rotation_trace",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Store decoded action yaw versus the scripted object-aligned target yaw.",
)
parser.add_argument("--output", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0], *hydra_args]


import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import tacex_tasks  # noqa: F401
import tacex_tasks.lab_pick  # noqa: F401

from clean_dsrl_sac import (
    CLEAN_DSRL_CONTRACT_VERSION,
    CleanDSRLActor,
    validate_absolute_dsrl_policy_state,
)
from clean_dsrl_wrapper import CleanDSRLLabPickWrapper


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def _flag(info: dict[str, Any], key: str) -> bool:
    return _scalar(info.get("log", {}).get(key), 0.0) > 0.5


def _quat_yaw_degrees(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    return torch.rad2deg(
        torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y.square() + z.square()),
        )
    )


def _symmetric_yaw_error_degrees(
    predicted_yaw: torch.Tensor,
    target_yaw: torch.Tensor,
) -> torch.Tensor:
    """Return gripper-axis yaw error modulo the gripper's 180-degree symmetry."""

    return torch.remainder(predicted_yaw - target_yaw + 90.0, 180.0) - 90.0


def _resolve_mode() -> str:
    if args_cli.mode == "auto":
        return "native_bc" if args_cli.checkpoint is None else "deterministic"
    if args_cli.mode in {"deterministic", "stochastic"} and args_cli.checkpoint is None:
        raise ValueError(f"--mode {args_cli.mode} requires --checkpoint.")
    return args_cli.mode


@hydra_task_config(args_cli.task, "skrl_clean_dsrl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict,
) -> None:
    mode = _resolve_mode()
    if args_cli.num_trials < 1:
        raise ValueError("--num_trials must be positive.")
    if args_cli.video_fps < 1:
        raise ValueError("--video_fps must be positive.")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env_cfg.rl_align_cafe_action_yaw = False
    env_cfg.rl_action_penalty_scale = 0.0
    env_cfg.labware_pos_randomization_xy = tuple(args_cli.labware_random_xy_m)
    env_cfg.labware_yaw_randomization = math.radians(args_cli.labware_random_yaw_deg)
    env_cfg.terminate_break_force_threshold_n = args_cli.break_force_threshold_n

    video_dir = None
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video_dir else None,
    )
    if args_cli.video_dir:
        video_dir = Path(args_cli.video_dir).expanduser().resolve()
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: episode_id < args_cli.num_trials,
            video_length=0,
            name_prefix=args_cli.video_prefix,
            fps=args_cli.video_fps,
            disable_logger=True,
        )
    env = CleanDSRLLabPickWrapper(
        env,
        args_cli.bc_policy,
        device=args_cli.bc_device,
        learned_noise_steps=args_cli.learned_noise_steps,
        padding_mode=args_cli.noise_padding_mode,
        chunk_execute_steps=args_cli.chunk_execute_steps,
        chunk_discount=args_cli.chunk_discount,
        flow_num_inference_steps=args_cli.flow_num_inference_steps,
        phase_horizon_steps=args_cli.phase_horizon_steps,
        camera_warmup_steps=args_cli.camera_warmup_steps,
        use_visual_xy_override=args_cli.use_visual_xy_override,
        seed=args_cli.seed,
    )
    layout = env.layout
    base = env.unwrapped
    device = torch.device(base.device)
    actor = None
    checkpoint_path = None
    if args_cli.checkpoint is not None:
        checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DSRL checkpoint not found: {checkpoint_path}")
        network_cfg = agent_cfg.get("network", {})
        actor = CleanDSRLActor(
            env.observation_space,
            env.action_space,
            device,
            layout=layout,
            hidden_dims=network_cfg.get("actor_hidden_dims", (512, 512, 512)),
            initial_log_std=float(network_cfg.get("initial_log_std", 0.0)),
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "policy" not in checkpoint:
            raise KeyError(f"Checkpoint has no policy state: {checkpoint_path}")
        validate_absolute_dsrl_policy_state(checkpoint["policy"])
        actor.load_state_dict(checkpoint["policy"], strict=True)
        actor.eval()

    physical_steps_per_outer = args_cli.chunk_execute_steps * env.action_repeat
    default_limit = math.ceil(int(base.max_episode_length) / physical_steps_per_outer) + 1
    max_outer_steps = args_cli.max_outer_steps or default_limit
    output_path = Path(args_cli.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    print(
        f"[INFO] Clean DSRL evaluation mode={mode} trials={args_cli.num_trials} "
        f"seed={args_cli.seed} max_outer_steps={max_outer_steps}",
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
            observation, _ = env.reset(seed=trial_seed, flow_noise_seed=trial_seed)
            episode_reward = 0.0
            max_lift_m = float("-inf")
            min_grasp_distance_m = float("inf")
            peak_force_n = 0.0
            success = broken = dropped = too_far = outside = timeout = False
            outer_steps = 0
            noise_rms_sum = 0.0
            rotation_trace: list[dict[str, Any]] = []
            terminated = truncated = None

            while simulation_app.is_running() and outer_steps < max_outer_steps:
                target_yaw = _quat_yaw_degrees(base.scripted_target_quat_b).detach()
                if mode == "native_bc":
                    observation, reward, terminated, truncated, info = env.step_bc()
                    action_rms = 1.0
                else:
                    policy_observation = observation["policy"] if isinstance(observation, dict) else observation
                    policy_observation = torch.as_tensor(
                        policy_observation, dtype=torch.float32, device=device
                    ).reshape(1, -1)
                    with torch.inference_mode():
                        if mode == "zero_noise":
                            action = torch.zeros((1, layout.action_dim), device=device)
                        elif mode == "stochastic":
                            action = actor.act(
                                {"observations": policy_observation}, role="policy"
                            )[0]
                        else:
                            action = actor.deterministic_action(
                                {"observations": policy_observation}
                            )
                    action_rms = _scalar(action.square().mean().sqrt())
                    observation, reward, terminated, truncated, info = env.step(action)
                if args_cli.record_rotation_trace:
                    decoded = env.last_decoded_action_chunk
                    if decoded is None:
                        raise RuntimeError("The DSRL wrapper did not expose its decoded action chunk.")
                    executed = int(info["clean_dsrl/action_steps_executed"])
                    decoded = decoded[:executed]
                    predicted_quat = base._rot6d_to_quat(decoded[:, 3:9])
                    predicted_yaw = _quat_yaw_degrees(predicted_quat)
                    yaw_error = _symmetric_yaw_error_degrees(
                        predicted_yaw,
                        target_yaw.reshape(-1)[0],
                    )
                    rotation_trace.append(
                        {
                            "outer_step": outer_steps,
                            "target_yaw_deg": _scalar(target_yaw),
                            "predicted_yaw_deg": predicted_yaw.detach().cpu().tolist(),
                            "yaw_error_deg": yaw_error.detach().cpu().tolist(),
                            "mean_abs_yaw_error_deg": _scalar(yaw_error.abs().mean()),
                            "first_action_rot6d": decoded[0, 3:9].detach().cpu().tolist(),
                            "last_action_rot6d": decoded[-1, 3:9].detach().cpu().tolist(),
                        }
                    )
                outer_steps += 1
                noise_rms_sum += action_rms
                episode_reward += _scalar(reward)
                log = info.get("log", {})
                success |= _flag(info, "LabPick/success_terminal_step")
                broken |= _flag(info, "LabPick/broken_terminal_step")
                dropped |= _flag(info, "LabPick/object_dropped_terminal_step")
                too_far |= _flag(info, "LabPick/object_too_far_terminal_step")
                outside |= _flag(info, "LabPick/ee_outside_workspace_terminal_step")
                timeout |= _flag(info, "LabPick/timeout_terminal_step")
                max_lift_m = max(max_lift_m, _scalar(log.get("LabPick/lift_m"), float("-inf")))
                min_grasp_distance_m = min(
                    min_grasp_distance_m,
                    _scalar(log.get("LabPick/grasp_distance_m"), float("inf")),
                )
                peak_force_n = max(peak_force_n, _scalar(log.get("LabPick/contact_force_n")))
                if bool(torch.as_tensor(terminated | truncated).any().item()):
                    break

            if success:
                failure_reason = "success"
            elif broken:
                failure_reason = "object_broken"
            elif dropped:
                failure_reason = "object_dropped"
            elif too_far:
                failure_reason = "object_too_far"
            elif outside:
                failure_reason = "ee_outside_workspace"
            elif timeout:
                failure_reason = "time_limit"
            else:
                failure_reason = "evaluation_limit"
            result = {
                "trial": trial,
                "seed": trial_seed,
                "success": success,
                "failure_reason": failure_reason,
                "episode_reward": episode_reward,
                "max_lift_m": max_lift_m,
                "min_grasp_distance_m": min_grasp_distance_m,
                "peak_force_n": peak_force_n,
                "outer_steps": outer_steps,
                "mean_policy_noise_rms": noise_rms_sum / max(outer_steps, 1),
            }
            if args_cli.record_rotation_trace:
                result["rotation_trace"] = rotation_trace
            results.append(result)
            print(
                f"[RESULT] trial={trial} seed={trial_seed} success={success} "
                f"reason={failure_reason} reward={episode_reward:.4f} "
                f"lift={max_lift_m:.4f} force={peak_force_n:.4f} "
                f"noise_rms={result['mean_policy_noise_rms']:.4f}",
                flush=True,
            )
    finally:
        env.close()

    successes = sum(int(item["success"]) for item in results)
    failure_counts: dict[str, int] = {}
    for item in results:
        reason = str(item["failure_reason"])
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    summary = {
        "mode": mode,
        "bc_policy": str(Path(args_cli.bc_policy).expanduser().resolve()),
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "task": args_cli.task,
        "num_trials": len(results),
        "successes": successes,
        "success_rate": successes / max(len(results), 1),
        "failure_counts": failure_counts,
        "mean_episode_reward": sum(item["episode_reward"] for item in results) / max(len(results), 1),
        "mean_policy_noise_rms": sum(item["mean_policy_noise_rms"] for item in results) / max(len(results), 1),
        "seed": args_cli.seed,
        "contract": "flow_noise_dsrl_absolute_repeat_last_v3_tactile",
        "contract_version": CLEAN_DSRL_CONTRACT_VERSION,
        "learned_noise_steps": args_cli.learned_noise_steps,
        "noise_padding_mode": args_cli.noise_padding_mode,
        "noise_action_semantics": "absolute",
        "noise_action_bounds": [-1.0, 1.0],
        "rl_align_cafe_action_yaw": False,
        "use_visual_xy_override": args_cli.use_visual_xy_override,
        "policy_action_source": "flow_decoder_full_xyz_rot6d_width",
        "chunk_execute_steps": args_cli.chunk_execute_steps,
        "chunk_discount": args_cli.chunk_discount,
        "video_directory": None if video_dir is None else str(video_dir),
        "video_files": []
        if video_dir is None
        else sorted(path.name for path in video_dir.glob("*.mp4")),
        "results": results,
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[SUMMARY] success={successes}/{len(results)} ({summary['success_rate']:.2%}) "
        f"reward={summary['mean_episode_reward']:.4f} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
