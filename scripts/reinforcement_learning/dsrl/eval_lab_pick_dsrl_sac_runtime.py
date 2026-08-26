from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a BC-initialized DSRL-SAC checkpoint on randomized LabPick.")
parser.add_argument("--policy", type=str, required=True, help="Frozen Flow Matching BC checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Saved SKRL SAC agent checkpoint.")
parser.add_argument("--task", type=str, default="TacEx-LabPick-Slide-DSRL-Base-v0")
parser.add_argument("--num_trials", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--policy_seed", type=int, default=42)
parser.add_argument("--stochastic", action="store_true", help="Sample the actor instead of using its deterministic mean.")
parser.add_argument(
    "--residual_mode",
    choices=["latent", "physical"],
    default="physical",
    help="Residual interface used by the checkpoint.",
)
parser.add_argument("--base_noise_seed", type=int, default=42)
parser.add_argument("--policy_device", type=str, default="cuda")
parser.add_argument("--noise_magnitude", type=float, default=3.0)
parser.add_argument("--action_repeat", type=int, default=2)
parser.add_argument("--flow_num_inference_steps", type=int, default=20)
parser.add_argument("--flow_chunk_execute_steps", type=int, default=32)
parser.add_argument("--physical_residual_segments", type=int, default=4)
parser.add_argument("--flow_phase_horizon_steps", type=int, default=383)
parser.add_argument("--flow_camera_warmup_steps", type=int, default=8)
parser.add_argument("--gate", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--gate_temperature", type=float, default=0.5)
parser.add_argument("--gate_penalty", type=float, default=5.0)
parser.add_argument("--gate_min", type=float, default=0.02)
parser.add_argument("--gate_max", type=float, default=0.2)
parser.add_argument("--max_outer_steps", type=int, default=0, help="0 derives the limit from the environment horizon.")
parser.add_argument(
    "--labware_random_xy",
    type=float,
    nargs=2,
    metavar=("X", "Y"),
    default=None,
    help="Uniform reset range in meters for labware x/y.",
)
parser.add_argument(
    "--labware_random_yaw",
    type=float,
    default=None,
    help="Uniform reset yaw range in radians.",
)
parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 512, 512])
parser.add_argument(
    "--output",
    type=str,
    default="logs/lab_pick_dsrl_eval/dsrl_sac.json",
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
import torch.nn as nn

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from skrl.models.torch import GaussianMixin, Model

import tacex_tasks  # noqa: F401
import tacex_tasks.lab_pick  # noqa: F401

from lab_pick_dsrl_wrapper import LabPickDSRLWrapper


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU()))
        current_dim = hidden_dim
    layers.extend((nn.Linear(current_dim, output_dim), nn.Tanh()))
    return nn.Sequential(*layers)


class StochasticActor(GaussianMixin, Model):
    """Checkpoint-compatible actor used by train_sac.py."""

    def __init__(self, observation_space, action_space, device, hidden_dims: list[int]):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_mean_actions=True,
            clip_log_std=True,
            min_log_std=-5,
            max_log_std=2,
        )
        self.net = _build_mlp(self.num_observations, hidden_dims, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


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


def _policy_observation(observation: Any, device: torch.device) -> torch.Tensor:
    if isinstance(observation, dict):
        observation = observation["policy"]
    return torch.as_tensor(observation, dtype=torch.float32, device=device).reshape(1, -1)


def _log_value(info: dict[str, Any], key: str, default: float = 0.0) -> float:
    log = info.get("log", {}) if isinstance(info, dict) else {}
    return _scalar(log.get(key), default)


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
    if args_cli.labware_random_xy is not None:
        env_cfg.labware_pos_randomization_xy = tuple(args_cli.labware_random_xy)
    if args_cli.labware_random_yaw is not None:
        env_cfg.labware_yaw_randomization = args_cli.labware_random_yaw

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    env = LabPickDSRLWrapper(
        base_env,
        args_cli.policy,
        device=args_cli.policy_device,
        noise_magnitude=args_cli.noise_magnitude,
        action_repeat=args_cli.action_repeat,
        policy_type="flow_matching",
        residual_mode=args_cli.residual_mode,
        flow_num_inference_steps=args_cli.flow_num_inference_steps,
        flow_chunk_execute_steps=args_cli.flow_chunk_execute_steps,
        physical_residual_segments=args_cli.physical_residual_segments,
        flow_phase_horizon_steps=args_cli.flow_phase_horizon_steps,
        flow_camera_warmup_steps=args_cli.flow_camera_warmup_steps,
        gate_enabled=args_cli.gate,
        gate_temperature=args_cli.gate_temperature,
        gate_penalty=args_cli.gate_penalty,
        gate_min=args_cli.gate_min,
        gate_max=args_cli.gate_max,
        base_noise_seed=args_cli.base_noise_seed,
    )
    base = env.unwrapped
    device = torch.device(base.device)
    first_observation, _ = env.reset(seed=args_cli.seed)
    observation_dim = int(_policy_observation(first_observation, device).shape[-1])
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)
    actor = StochasticActor(observation_space, env.action_space, device, list(args_cli.hidden_dims)).to(device)
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "policy" not in checkpoint:
        raise KeyError(f"Checkpoint has no policy state: {checkpoint_path}")
    actor.load_state_dict(checkpoint["policy"], strict=True)
    actor.eval()

    physical_steps_per_outer = args_cli.flow_chunk_execute_steps * args_cli.action_repeat
    default_outer_steps = (int(base.max_episode_length) + physical_steps_per_outer - 1) // physical_steps_per_outer
    max_outer_steps = args_cli.max_outer_steps if args_cli.max_outer_steps > 0 else default_outer_steps + 1
    output_path = Path(args_cli.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    mode = "stochastic" if args_cli.stochastic else "deterministic_mean"
    print(
        "[INFO] DSRL-SAC episodic evaluation "
        f"checkpoint={checkpoint_path} trials={args_cli.num_trials} seed={args_cli.seed} "
        f"mode={mode} max_outer_steps={max_outer_steps}",
        flush=True,
    )

    try:
        for trial in range(args_cli.num_trials):
            trial_seed = args_cli.seed + trial
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            torch.manual_seed(args_cli.policy_seed + trial)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args_cli.policy_seed + trial)

            observation, _ = env.reset(seed=trial_seed)
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
            gate_sum = 0.0
            gated_steps = 0

            while simulation_app.is_running() and outer_steps < max_outer_steps:
                inputs = {"observations": _policy_observation(observation, device)}
                with torch.inference_mode():
                    if args_cli.stochastic:
                        action = actor.act(inputs, role="policy")[0]
                    else:
                        action = actor.compute(inputs, role="policy")[0]
                observation, reward, terminated, truncated, info = env.step(action)
                outer_steps += 1
                if "dsrl/gate" in info:
                    gate_sum += _scalar(info["dsrl/gate"])
                    gated_steps += 1
                episode_reward += _scalar(reward)
                success = success or max(
                    _log_value(info, "LabPick/success_terminal_step"),
                    _log_value(info, "LabPick/success_rate"),
                ) > 0.5
                broken = broken or max(
                    _log_value(info, "LabPick/broken_terminal_step"),
                    _log_value(info, "LabPick/broken_rate"),
                ) > 0.5
                max_lift_m = max(max_lift_m, _log_value(info, "LabPick/lift_m", float("-inf")))
                min_grasp_distance_m = min(
                    min_grasp_distance_m,
                    _log_value(info, "LabPick/grasp_distance_m", float("inf")),
                )
                peak_force_n = max(peak_force_n, _log_value(info, "LabPick/contact_force_n"))
                if _bool(terminated) or _bool(truncated):
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
                "mean_gate": gate_sum / max(gated_steps, 1),
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
                f"mean_gate={result['mean_gate']:.4f} outer_steps={outer_steps}",
                flush=True,
            )
    finally:
        env.close()

    successes = sum(int(result["success"]) for result in results)
    broken_count = sum(int(result["broken"]) for result in results)
    summary = {
        "policy": str(Path(args_cli.policy).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "task": args_cli.task,
        "mode": mode,
        "num_trials": len(results),
        "successes": successes,
        "success_rate": successes / max(len(results), 1),
        "broken": broken_count,
        "broken_rate": broken_count / max(len(results), 1),
        "seed": args_cli.seed,
        "policy_seed": args_cli.policy_seed,
        "gate_enabled": args_cli.gate,
        "gate_temperature": args_cli.gate_temperature,
        "gate_penalty": args_cli.gate_penalty,
        "gate_max": args_cli.gate_max,
        "mean_gate": sum(result["mean_gate"] for result in results) / max(len(results), 1),
        "results": results,
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[SUMMARY] success_rate={successes}/{len(results)} ({summary['success_rate']:.2%}) "
        f"broken_rate={broken_count}/{len(results)} ({summary['broken_rate']:.2%}) output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
