from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a frozen LeRobot Diffusion Policy on randomized LabPick.")
parser.add_argument("--policy", type=str, required=True, help="Saved LeRobot Diffusion Policy directory.")
parser.add_argument("--task", type=str, default="TacEx-LabPick-Slide-DSRL-Base-v0")
parser.add_argument("--num_trials", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--action_repeat", type=int, default=2, help="Physics steps per 60 Hz policy action.")
parser.add_argument("--policy_device", type=str, default="cuda")
parser.add_argument("--max_outer_steps", type=int, default=0, help="0 uses the environment episode length.")
parser.add_argument(
    "--output",
    type=str,
    default="logs/lab_pick_bc_eval/frozen_diffusion_bc.json",
    help="JSON file for per-trial metrics.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import tacex_tasks  # noqa: F401
import tacex_tasks.lab_pick  # noqa: F401

from lab_pick_dsrl_wrapper import LabPickDSRLWrapper


def _scalar(value: torch.Tensor) -> float:
    return float(value.reshape(-1)[0].item())


def _bool(value: torch.Tensor) -> bool:
    return bool(value.reshape(-1)[0].item())


@hydra_task_config(args_cli.task, "skrl_sac_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg: dict) -> None:
    if args_cli.num_trials < 1:
        raise ValueError(f"num_trials must be at least 1, received {args_cli.num_trials}.")
    if args_cli.action_repeat < 1:
        raise ValueError(f"action_repeat must be at least 1, received {args_cli.action_repeat}.")

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    env = LabPickDSRLWrapper(
        base_env,
        args_cli.policy,
        device=args_cli.policy_device,
        action_repeat=args_cli.action_repeat,
    )
    base = env.unwrapped
    physics_hz = 1.0 / float(base.physics_dt)
    action_hz = physics_hz / args_cli.action_repeat
    physical_steps_per_outer = env.adapter.policy.config.n_action_steps * args_cli.action_repeat
    default_outer_steps = (int(base.max_episode_length) + physical_steps_per_outer - 1) // physical_steps_per_outer
    max_outer_steps = args_cli.max_outer_steps if args_cli.max_outer_steps > 0 else default_outer_steps + 1
    output_path = Path(args_cli.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    print(
        "[INFO] Frozen Diffusion BC evaluation "
        f"trials={args_cli.num_trials} seed={args_cli.seed} action_repeat={args_cli.action_repeat} "
        f"action_hz={action_hz:.2f} max_outer_steps={max_outer_steps}",
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
            reset_pos = base.labware_reset_pos_w[0].detach().cpu().tolist()
            reset_quat = base.labware_reset_quat_w[0].detach().cpu().tolist()
            episode_reward = 0.0
            max_lift_m = float("-inf")
            min_grasp_distance_m = float("inf")
            peak_force_n = 0.0
            success = False
            broken = False
            terminated = truncated = None
            outer_steps = 0

            while simulation_app.is_running() and outer_steps < max_outer_steps:
                _, reward, terminated, truncated, _ = env.step_bc()
                outer_steps += 1
                episode_reward += _scalar(reward)

                flags = base._get_termination_flags()
                lift_m = _scalar(base.labware.data.root_pos_w[:, 2] - base.initial_object_height)
                object_pos_b = base.labware.data.root_pos_w - base._robot.data.root_link_pos_w
                grasp_distance_m = _scalar(torch.linalg.norm(object_pos_b - base._gripper_center_pos_b(), dim=1))
                force_n = _scalar(flags["force_norm"])
                max_lift_m = max(max_lift_m, lift_m)
                min_grasp_distance_m = min(min_grasp_distance_m, grasp_distance_m)
                peak_force_n = max(peak_force_n, force_n)
                success = success or _bool(flags["success"])
                broken = broken or _bool(flags["object_broken"])
                if _bool(terminated | truncated):
                    break

            result = {
                "trial": trial,
                "seed": trial_seed,
                "success": success,
                "broken": broken,
                "episode_reward": episode_reward,
                "max_lift_m": max_lift_m,
                "min_grasp_distance_m": min_grasp_distance_m,
                "peak_force_n": peak_force_n,
                "outer_steps": outer_steps,
                "reset_pos_w": reset_pos,
                "reset_quat_w": reset_quat,
                "terminated": _bool(terminated) if terminated is not None else False,
                "truncated": _bool(truncated) if truncated is not None else False,
            }
            results.append(result)
            print(
                "[RESULT] "
                f"trial={trial} seed={trial_seed} success={success} broken={broken} "
                f"reward={episode_reward:.4f} max_lift={max_lift_m:.4f}m "
                f"min_grasp_distance={min_grasp_distance_m:.4f}m peak_force={peak_force_n:.4f}N "
                f"outer_steps={outer_steps} reset_xy={[round(value, 4) for value in reset_pos[:2]]}",
                flush=True,
            )
    finally:
        env.close()

    successes = sum(int(result["success"]) for result in results)
    broken_count = sum(int(result["broken"]) for result in results)
    summary = {
        "policy": str(Path(args_cli.policy).expanduser().resolve()),
        "task": args_cli.task,
        "num_trials": len(results),
        "successes": successes,
        "success_rate": successes / max(len(results), 1),
        "broken": broken_count,
        "broken_rate": broken_count / max(len(results), 1),
        "seed": args_cli.seed,
        "action_repeat": args_cli.action_repeat,
        "action_hz": action_hz,
        "physics_hz": physics_hz,
        "results": results,
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[SUMMARY] success_rate={successes}/{len(results)} ({summary['success_rate']:.2%}) "
        f"broken_rate={broken_count}/{len(results)} ({summary['broken_rate']:.2%}) output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
