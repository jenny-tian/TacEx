from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Evaluate zero-residual BC or a base-preserving Clean Residual SAC checkpoint."
)
parser.add_argument("--bc_policy", type=str, required=True)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="SKRL agent checkpoint. Omit it to evaluate the zero-residual BC baseline.",
)
parser.add_argument(
    "--task",
    default="TacEx-LabPick-Slide-Clean-Residual-SAC-v0",
)
parser.add_argument("--num_trials", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--flow_noise_seed",
    type=int,
    default=None,
    help="Fixed Flow seed for all trials; default uses each trial's environment seed.",
)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--residual_scale", type=float, default=0.01)
parser.add_argument("--residual_contact_gate", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--bc_device", type=str, default="cuda:0")
parser.add_argument("--flow_num_inference_steps", type=int, default=20)
parser.add_argument("--phase_horizon_steps", type=int, default=383)
parser.add_argument("--camera_warmup_steps", type=int, default=8)
parser.add_argument("--labware_random_xy_m", type=float, nargs=2, default=(0.10, 0.10))
parser.add_argument("--labware_random_yaw_deg", type=float, default=45.0)
parser.add_argument("--break_force_threshold_n", type=float, default=4.0)
parser.add_argument("--max_outer_steps", type=int, default=0)
parser.add_argument("--training_step", type=int, default=0)
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

from clean_residual_sac import CleanResidualActor, validate_tactile_residual_policy_state
from clean_residual_wrapper import CleanResidualLabPickWrapper


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def _bool(value: Any) -> bool:
    return bool(_scalar(value))


def _log_value(info: dict[str, Any], key: str, default: float = 0.0) -> float:
    log = info.get("log", {}) if isinstance(info, dict) else {}
    return _scalar(log.get(key), default)


def _terminal_reason(info: dict[str, Any], *, success: bool) -> str:
    if success:
        return "success"
    ordered_flags = (
        ("object_broken", "LabPick/broken_terminal_step"),
        ("object_dropped", "LabPick/object_dropped_terminal_step"),
        ("object_too_far", "LabPick/object_too_far_terminal_step"),
        ("ee_outside_workspace", "LabPick/ee_outside_workspace_terminal_step"),
        ("time_limit", "LabPick/timeout_terminal_step"),
    )
    for reason, key in ordered_flags:
        if _log_value(info, key) > 0.5:
            return reason
    return "unknown_terminal"


@hydra_task_config(args_cli.task, "skrl_clean_sac_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict,
) -> None:
    if args_cli.num_trials < 1:
        raise ValueError("--num_trials must be positive.")
    if args_cli.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args_cli.flow_noise_seed is not None and args_cli.flow_noise_seed < 0:
        raise ValueError("--flow_noise_seed must be non-negative.")
    if not math.isfinite(args_cli.residual_scale) or args_cli.residual_scale < 0.0:
        raise ValueError("--residual_scale must be finite and non-negative.")
    bc_path = Path(args_cli.bc_policy).expanduser().resolve()
    if not bc_path.is_file():
        raise FileNotFoundError(f"Flow BC checkpoint not found: {bc_path}")

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env_cfg.rl_align_cafe_action_yaw = False
    env_cfg.rl_action_penalty_scale = 0.0
    env_cfg.labware_pos_randomization_xy = tuple(args_cli.labware_random_xy_m)
    env_cfg.labware_yaw_randomization = math.radians(args_cli.labware_random_yaw_deg)
    env_cfg.terminate_break_force_threshold_n = args_cli.break_force_threshold_n

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    env = CleanResidualLabPickWrapper(
        base_env,
        bc_path,
        device=args_cli.bc_device,
        residual_scale=args_cli.residual_scale,
        flow_num_inference_steps=args_cli.flow_num_inference_steps,
        phase_horizon_steps=args_cli.phase_horizon_steps,
        camera_warmup_steps=args_cli.camera_warmup_steps,
        contact_gate=args_cli.residual_contact_gate,
        seed=args_cli.seed if args_cli.flow_noise_seed is None else args_cli.flow_noise_seed,
    )
    base = env.unwrapped
    device = torch.device(base.device)

    actor = None
    checkpoint_path = None
    if args_cli.checkpoint:
        checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "policy" not in checkpoint:
            raise KeyError(f"Checkpoint has no policy state: {checkpoint_path}")
        validate_tactile_residual_policy_state(checkpoint["policy"])
        initial_log_std = float(agent_cfg.get("network", {}).get("initial_log_std", -3.0))
        actor = CleanResidualActor(
            env.observation_space,
            env.action_space,
            device,
            layout=env.layout,
            initial_log_std=initial_log_std,
        ).to(device)
        actor.load_state_dict(checkpoint["policy"], strict=True)
        actor.eval()

    default_outer_steps = (int(base.max_episode_length) + env.action_repeat - 1) // env.action_repeat
    max_outer_steps = args_cli.max_outer_steps if args_cli.max_outer_steps > 0 else default_outer_steps + 1
    output_path = Path(args_cli.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "zero_residual_bc" if actor is None else (
        "stochastic_actor" if args_cli.stochastic else "deterministic_actor"
    )
    results: list[dict[str, Any]] = []
    print(
        "[INFO] Base-preserving Clean Residual evaluation "
        f"mode={mode} checkpoint={checkpoint_path} trials={args_cli.num_trials} "
        f"seed={args_cli.seed} chunk={env.replan_steps} training_step={args_cli.training_step}",
        flush=True,
    )

    try:
        for trial in range(args_cli.num_trials):
            trial_seed = args_cli.seed + trial
            flow_seed = trial_seed if args_cli.flow_noise_seed is None else args_cli.flow_noise_seed
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(trial_seed)
            observation, _ = env.reset(seed=trial_seed, flow_noise_seed=flow_seed)
            reset_pos = base.labware_reset_pos_w[0].detach().cpu().tolist()
            reset_quat = base.labware_reset_quat_w[0].detach().cpu().tolist()
            episode_reward = 0.0
            max_lift_m = 0.0
            min_grasp_distance_m = float("inf")
            peak_force_n = 0.0
            residual_rms_sum = 0.0
            success = False
            broken = False
            terminated = truncated = None
            info: dict[str, Any] = {}
            outer_steps = 0

            while simulation_app.is_running() and outer_steps < max_outer_steps:
                if actor is None:
                    residual = torch.zeros((1, env.layout.action), device=device)
                else:
                    inputs = {"observations": observation["policy"].to(device)}
                    with torch.inference_mode():
                        if args_cli.stochastic:
                            residual = actor.act(inputs, role="policy")[0]
                        else:
                            residual = actor.deterministic_action(inputs)
                observation, reward, terminated, truncated, info = env.step(residual)
                outer_steps += 1
                episode_reward += _scalar(reward)
                residual_rms_sum += _scalar(residual.square().mean().sqrt())
                success = success or _log_value(info, "LabPick/success_terminal_step") > 0.5
                broken = broken or _log_value(info, "LabPick/broken_terminal_step") > 0.5
                max_lift_m = max(max_lift_m, _log_value(info, "LabPick/lift_m"))
                min_grasp_distance_m = min(
                    min_grasp_distance_m,
                    _log_value(info, "LabPick/grasp_distance_m", float("inf")),
                )
                peak_force_n = max(
                    peak_force_n,
                    _log_value(info, "LabPick/contact_force_n"),
                )
                if _bool(terminated) or _bool(truncated):
                    break

            terminal_reason = _terminal_reason(info, success=success)
            result = {
                "trial": trial,
                "seed": trial_seed,
                "flow_noise_seed": flow_seed,
                "success": success,
                "broken": broken,
                "terminal_reason": terminal_reason,
                "episode_reward": episode_reward,
                "max_lift_m": max_lift_m,
                "min_grasp_distance_m": min_grasp_distance_m,
                "peak_force_n": peak_force_n,
                "mean_residual_rms": residual_rms_sum / max(outer_steps, 1),
                "outer_steps": outer_steps,
                "reset_pos_w": reset_pos,
                "reset_quat_w": reset_quat,
            }
            results.append(result)
            print(
                "[RESULT] "
                f"trial={trial} seed={trial_seed} flow_seed={flow_seed} "
                f"success={success} reason={terminal_reason} reward={episode_reward:.4f} "
                f"max_lift={max_lift_m:.4f}m peak_force={peak_force_n:.4f}N "
                f"residual_rms={result['mean_residual_rms']:.4f} steps={outer_steps}",
                flush=True,
            )
    finally:
        env.close()

    successes = sum(int(result["success"]) for result in results)
    broken_count = sum(int(result["broken"]) for result in results)
    failure_counts = Counter(
        result["terminal_reason"] for result in results if not result["success"]
    )
    summary = {
        "bc_policy": str(bc_path),
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "training_step": args_cli.training_step,
        "mode": mode,
        "num_trials": len(results),
        "successes": successes,
        "success_rate": successes / max(len(results), 1),
        "broken": broken_count,
        "broken_rate": broken_count / max(len(results), 1),
        "failures": len(results) - successes,
        "failure_counts": dict(failure_counts),
        "mean_episode_reward": sum(result["episode_reward"] for result in results) / max(len(results), 1),
        "mean_residual_rms": sum(result["mean_residual_rms"] for result in results) / max(len(results), 1),
        "seed": args_cli.seed,
        "flow_noise_seed": args_cli.flow_noise_seed,
        "flow_noise_semantics": "fixed_within_trial",
        "chunk_execute_steps": env.replan_steps,
        "action_repeat": env.action_repeat,
        "residual_scale": args_cli.residual_scale,
        "results": results,
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[SUMMARY] step={args_cli.training_step} success={successes}/{len(results)} "
        f"({summary['success_rate']:.2%}) broken={broken_count}/{len(results)} "
        f"({summary['broken_rate']:.2%}) failures={dict(failure_counts)} "
        f"mean_reward={summary['mean_episode_reward']:.4f} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
