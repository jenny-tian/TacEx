"""Evaluate a trained GrapePick BC policy on independent randomized seeds."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter
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
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--seed", type=int, default=50000)
parser.add_argument("--max_episode_steps", type=int, default=790)
parser.add_argument("--action_smoothing", type=float, default=0.35)
parser.add_argument("--success_hold_steps", type=int, default=5)
parser.add_argument(
    "--randomize_force_thresholds",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Randomize conservative grape damage and stem-detachment thresholds per episode.",
)
parser.add_argument("--damage_range", type=float, nargs=2, default=(3.40, 5.00))
parser.add_argument("--detach_range", type=float, nargs=2, default=(2.80, 3.40))
parser.add_argument("--report", type=Path, default=Path("artifacts/grape_pick/eval_report.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch

from tacex_tasks.grape_pick.grape_pick_env import GrapePickEnv
from tacex_tasks.grape_pick.grape_pick_env_cfg import GrapePickEnvCfg
from tacex_tasks.grape_pick.policy import (
    GrapeBCPolicy,
    denormalize,
    environment_policy_observation,
    normalize,
)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_time_horizon = int(checkpoint.get("max_episode_steps", 1000))
    cfg = GrapePickEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.enable_cameras = False
    cfg.enable_gelsight = False
    cfg.randomize_force_thresholds = args.randomize_force_thresholds
    cfg.success_hold_steps = args.success_hold_steps
    cfg.damage_force_randomization_range_n = tuple(args.damage_range)
    cfg.detach_force_randomization_range_n = tuple(args.detach_range)
    cfg.episode_length_s = max(args.max_episode_steps * cfg.sim.dt, 1.0)
    if args.device:
        cfg.sim.device = args.device
    env = GrapePickEnv(cfg)
    policy_device = torch.device(cfg.sim.device)
    model = GrapeBCPolicy(
        hidden_dim=int(checkpoint["hidden_dim"]), depth=int(checkpoint["depth"])
    ).to(policy_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    obs_mean = checkpoint["obs_mean"].to(policy_device)
    obs_std = checkpoint["obs_std"].to(policy_device)
    action_mean = checkpoint["action_mean"].to(policy_device)
    action_std = checkpoint["action_std"].to(policy_device)

    outcomes = []
    failure_counts: Counter[str] = Counter()
    successes = 0
    evaluated = 0
    batch = 0
    try:
        while evaluated < args.episodes and simulation_app.is_running():
            batch_seed = args.seed + batch * args.num_envs
            env.reset(seed=batch_seed)
            active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
            previous_action = None
            batch_records: dict[int, dict] = {}
            for step in range(args.max_episode_steps):
                with torch.inference_mode():
                    observation = environment_policy_observation(
                        env, max_episode_steps=policy_time_horizon
                    )
                    prediction = model(normalize(observation, obs_mean, obs_std))
                    action = denormalize(prediction, action_mean, action_std)
                    if previous_action is not None:
                        alpha = float(args.action_smoothing)
                        action = alpha * action + (1.0 - alpha) * previous_action
                    previous_action = action
                env._pre_physics_step(action)
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
                flags = env.get_task_flags()
                terminal = (
                    flags["success"]
                    | flags["crushed"]
                    | flags["slipped"]
                    | flags["obstacle_collision"]
                    | flags["workspace_failure"]
                )
                if step == args.max_episode_steps - 1:
                    terminal |= active
                just_finished = active & terminal
                for index in torch.nonzero(just_finished, as_tuple=False).squeeze(-1).cpu().tolist():
                    if bool(flags["success"][index].item()):
                        outcome = "success"
                    elif bool(flags["crushed"][index].item()):
                        outcome = "crushed"
                    elif bool(flags["slipped"][index].item()):
                        outcome = "slipped"
                    elif bool(flags["obstacle_collision"][index].item()):
                        outcome = "obstacle_collision"
                    elif bool(flags["workspace_failure"][index].item()):
                        outcome = "workspace_failure"
                    elif bool(flags["stem_attached"][index].item()):
                        outcome = "no_detach"
                    else:
                        outcome = "timeout_after_detach"
                    batch_records[index] = {
                        "seed": batch_seed + index,
                        "outcome": outcome,
                        "steps": step + 1,
                        "peak_grip_force_n": float(env.peak_grip_force_n[index].item()),
                        "peak_stem_tension_n": float(env.peak_stem_tension_n[index].item()),
                        "damage_threshold_n": float(env.damage_threshold_n[index].item()),
                        "detach_threshold_n": float(env.detach_threshold_n[index].item()),
                    }
                active &= ~just_finished
                if not bool(active.any().item()):
                    break

            for index in sorted(batch_records):
                if evaluated >= args.episodes:
                    break
                record = batch_records[index]
                outcomes.append(record)
                evaluated += 1
                successes += int(record["outcome"] == "success")
                if record["outcome"] != "success":
                    failure_counts[record["outcome"]] += 1
            print(
                f"[EVAL] evaluated={evaluated}/{args.episodes} successes={successes} "
                f"rate={successes / max(evaluated, 1):.2%}",
                flush=True,
            )
            batch += 1
    finally:
        env.close()

    lower, upper = wilson_interval(successes, evaluated)
    report = {
        "task": "TacEx-GrapePick-Force-v0",
        "policy": "trajectory_conditioned_tactile_mlp_bc",
        "tactile_inputs": ["left_pad_normal_force_n", "right_pad_normal_force_n"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "seed_start": args.seed,
        "episodes": evaluated,
        "successes": successes,
        "success_rate": successes / max(evaluated, 1),
        "wilson_95_ci": [lower, upper],
        "failure_counts": dict(failure_counts),
        "grip_limit_n": cfg.damage_grip_force_threshold_n,
        "detach_threshold_n": cfg.detach_force_threshold_n,
        "randomize_force_thresholds": cfg.randomize_force_thresholds,
        "damage_force_randomization_range_n": list(cfg.damage_force_randomization_range_n),
        "detach_force_randomization_range_n": list(cfg.detach_force_randomization_range_n),
        "evaluation_horizon_steps": args.max_episode_steps,
        "policy_time_horizon_steps": policy_time_horizon,
        "success_hold_steps": cfg.success_hold_steps,
        "outcomes": outcomes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[SUMMARY] " + json.dumps({key: value for key, value in report.items() if key != "outcomes"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
