"""Evaluate SealTwist BC with randomized force limits and cap alignment."""

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
parser.add_argument(
    "--checkpoint", type=Path, default=Path("exp_report/bottle/checkpoints/state_bc.pt")
)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--seed", type=int, default=50000)
parser.add_argument("--max_episode_steps", type=int, default=900)
parser.add_argument("--action_smoothing", type=float, default=0.35)
parser.add_argument("--success_hold_steps", type=int, default=5)
parser.add_argument("--force_violation_dwell_steps", type=int, default=5)
parser.add_argument("--position_range", type=float, nargs=2, default=(0.010, 0.010))
parser.add_argument("--alignment_tolerance", type=float, default=0.012)
parser.add_argument("--minimum_force_range", type=float, nargs=2, default=(4.7, 6.1))
parser.add_argument("--maximum_force_range", type=float, nargs=2, default=(9.0, 11.0))
parser.add_argument(
    "--report", type=Path, default=Path("exp_report/bottle/evaluation/base/seed_50000.json")
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch

from tacex_tasks.seal_twist.policy import (
    SealTwistBCPolicy,
    denormalize,
    environment_policy_observation,
    normalize,
)
from tacex_tasks.seal_twist.seal_twist_env import SealTwistEnv
from tacex_tasks.seal_twist.seal_twist_env_cfg import SealTwistEnvCfg


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


def outcome_for(flags: dict[str, torch.Tensor], index: int) -> str:
    if bool(flags["success"][index].item()):
        return "success"
    if bool(flags["over_force"][index].item()) or bool(flags["over_torqued"][index].item()):
        return "over_force"
    if bool(flags["alignment_failure"][index].item()) or bool(flags["slipped"][index].item()):
        return "alignment_failure"
    if bool(flags["under_force"][index].item()) or bool(flags["under_torqued"][index].item()):
        return "under_force"
    if not bool(flags["aligned"][index].item()) or not bool(flags["bilateral_contact"][index].item()):
        return "alignment_failure"
    return "under_force"


def main() -> None:
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_time_horizon = int(checkpoint.get("max_episode_steps", 900))
    cfg = SealTwistEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.enable_cameras = False
    cfg.enable_gelsight = False
    cfg.randomize_labware_position = True
    cfg.labware_pos_randomization_xy = tuple(args.position_range)
    cfg.labware_yaw_randomization = 0.0
    cfg.randomize_expert_targets = False
    cfg.randomize_force_thresholds = True
    cfg.minimum_force_randomization_range_n = tuple(args.minimum_force_range)
    cfg.maximum_force_randomization_range_n = tuple(args.maximum_force_range)
    cfg.alignment_tolerance_m = args.alignment_tolerance
    cfg.success_hold_steps = args.success_hold_steps
    cfg.force_violation_dwell_steps = args.force_violation_dwell_steps
    cfg.alignment_failure_dwell_steps = args.force_violation_dwell_steps
    cfg.episode_length_s = max(args.max_episode_steps * cfg.sim.dt, 1.0)
    if args.device:
        cfg.sim.device = args.device
    env = SealTwistEnv(cfg)
    policy_device = torch.device(cfg.sim.device)
    model = SealTwistBCPolicy(
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
                    | flags["over_force"]
                    | flags["over_torqued"]
                    | flags["under_force"]
                    | flags["alignment_failure"]
                    | flags["slipped"]
                )
                if step == args.max_episode_steps - 1:
                    terminal |= active
                just_finished = active & terminal
                for index in (
                    torch.nonzero(just_finished, as_tuple=False).squeeze(-1).cpu().tolist()
                ):
                    outcome = outcome_for(flags, index)
                    batch_records[index] = {
                        "seed": batch_seed + index,
                        "outcome": outcome,
                        "steps": step + 1,
                        "final_torque_nm": float(env.seal_torque_nm[index].item()),
                        "peak_torque_nm": float(env.peak_torque_nm[index].item()),
                        "final_minimum_finger_force_n": float(
                            flags["minimum_finger_force_n"][index].item()
                        ),
                        "final_maximum_finger_force_n": float(
                            flags["maximum_finger_force_n"][index].item()
                        ),
                        "peak_left_finger_force_n": float(env.peak_left_force_n[index].item()),
                        "peak_right_finger_force_n": float(env.peak_right_force_n[index].item()),
                        "minimum_force_threshold_n": float(
                            env.minimum_force_threshold_n[index].item()
                        ),
                        "maximum_force_threshold_n": float(
                            env.maximum_force_threshold_n[index].item()
                        ),
                        "alignment_error_m": float(flags["alignment_error_m"][index].item()),
                        "alignment_tolerance_m": float(cfg.alignment_tolerance_m),
                        "cap_position_offset_m": env.sampled_cap_position_offset_m[index]
                        .detach()
                        .cpu()
                        .tolist(),
                        "bilateral_contact": bool(flags["bilateral_contact"][index].item()),
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
                f"rate={successes / max(evaluated, 1):.2%} failures={dict(failure_counts)}",
                flush=True,
            )
            batch += 1
    finally:
        env.close()

    failures = evaluated - successes
    lower, upper = wilson_interval(successes, evaluated)
    failure_percent = {
        key: count / max(failures, 1) for key, count in sorted(failure_counts.items())
    }
    report = {
        "task": "TacEx-SealTwist-Torque-v0",
        "policy": "state_tactile_mlp_bc",
        "tactile_inputs": ["left_finger_force_n", "right_finger_force_n"],
        "force_thresholds_hidden_from_policy": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "seed_start": args.seed,
        "episodes": evaluated,
        "successes": successes,
        "success_rate": successes / max(evaluated, 1),
        "wilson_95_ci": [lower, upper],
        "failures": failures,
        "failure_counts": dict(failure_counts),
        "failure_fraction_among_failures": failure_percent,
        "minimum_force_randomization_range_n": list(args.minimum_force_range),
        "maximum_force_randomization_range_n": list(args.maximum_force_range),
        "position_randomization_half_range_m": list(args.position_range),
        "alignment_tolerance_m": args.alignment_tolerance,
        "torque_window_nm": list(cfg.torque_window_nm),
        "evaluation_horizon_steps": args.max_episode_steps,
        "policy_time_horizon_steps": policy_time_horizon,
        "success_hold_steps": cfg.success_hold_steps,
        "outcomes": outcomes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "[SUMMARY] " + json.dumps({key: value for key, value in report.items() if key != "outcomes"}),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
