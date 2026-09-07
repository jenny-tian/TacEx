#!/usr/bin/env python3
"""Evaluate the frozen GrapePick Flow Matching BC on randomized episodes."""

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
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num-envs", type=int, default=20)
parser.add_argument("--seed", type=int, default=87000)
parser.add_argument("--max-episode-steps", type=int, default=790)
parser.add_argument("--action-repeat", type=int, default=4)
parser.add_argument("--chunk-execute-steps", type=int, default=4)
parser.add_argument("--integration-steps", type=int, default=16)
parser.add_argument("--action-smoothing", type=float, default=0.35)
parser.add_argument("--damage-range", type=float, nargs=2, default=(3.4, 4.6))
parser.add_argument("--detach-range", type=float, nargs=2, default=(3.0, 3.7))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch

from grape_flow_policy import FLOW_NOISE_SEED_OFFSET, load_flow_checkpoint
from tacex_tasks.grape_pick.grape_pick_env import GrapePickEnv
from tacex_tasks.grape_pick.grape_pick_env_cfg import GrapePickEnvCfg
from tacex_tasks.grape_pick.policy import environment_policy_observation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, total: int) -> list[float]:
    if total < 1:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def classify(flags: dict[str, torch.Tensor], index: int) -> str:
    if bool(flags["success"][index].item()):
        return "success"
    for name in ("crushed", "slipped", "obstacle_collision", "workspace_failure"):
        if bool(flags[name][index].item()):
            return name
    if bool(flags["stem_attached"][index].item()):
        return "no_detach"
    return "timeout_after_detach"


def main() -> None:
    if args.chunk_execute_steps > 8:
        parser.error("chunk-execute-steps cannot exceed the checkpoint horizon")
    config = GrapePickEnvCfg()
    config.scene.num_envs = args.num_envs
    config.seed = args.seed
    config.enable_cameras = False
    config.enable_gelsight = False
    config.randomize_force_thresholds = True
    config.damage_force_randomization_range_n = tuple(args.damage_range)
    config.detach_force_randomization_range_n = tuple(args.detach_range)
    config.episode_length_s = max(args.max_episode_steps * config.sim.dt, 1.0)
    if args.device:
        config.sim.device = args.device
    env = GrapePickEnv(config)
    device = torch.device(config.sim.device)
    loaded = load_flow_checkpoint(args.checkpoint, device=device)
    if args.chunk_execute_steps > loaded.model.config.horizon:
        raise ValueError("chunk-execute-steps exceeds Flow horizon")

    records: list[dict] = []
    batch = 0
    try:
        while len(records) < args.episodes and simulation_app.is_running():
            batch_seed = args.seed + batch * args.num_envs
            env.reset(seed=batch_seed)
            active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
            previous_action: torch.Tensor | None = None
            step = 0
            batch_records: dict[int, dict] = {}
            generator = torch.Generator(device=device).manual_seed(
                batch_seed + FLOW_NOISE_SEED_OFFSET
            )
            while step < args.max_episode_steps and bool(active.any().item()):
                observation = environment_policy_observation(
                    env, max_episode_steps=loaded.max_episode_steps
                )
                normalized = loaded.normalize_observation(observation)
                native_noise = torch.randn(
                    (
                        args.num_envs,
                        loaded.model.config.horizon,
                        loaded.model.config.action_dim,
                    ),
                    generator=generator,
                    device=device,
                )
                normalized_chunk = loaded.model.decode(
                    normalized,
                    native_noise,
                    integration_steps=args.integration_steps,
                    solver="heun",
                )
                chunk = loaded.denormalize_action(normalized_chunk)
                for chunk_index in range(args.chunk_execute_steps):
                    target = chunk[:, chunk_index]
                    for _ in range(args.action_repeat):
                        if previous_action is not None:
                            alpha = float(args.action_smoothing)
                            action = alpha * target + (1.0 - alpha) * previous_action
                        else:
                            action = target
                        previous_action = action
                        env._pre_physics_step(action)
                        env._apply_action()
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(dt=env.physics_dt)
                        step += 1
                        flags = env.get_task_flags()
                        terminal = (
                            flags["success"]
                            | flags["crushed"]
                            | flags["slipped"]
                            | flags["obstacle_collision"]
                            | flags["workspace_failure"]
                        )
                        if step >= args.max_episode_steps:
                            terminal |= active
                        just_finished = active & terminal
                        for index in torch.nonzero(just_finished, as_tuple=False).squeeze(-1).cpu().tolist():
                            batch_records[index] = {
                                "seed": batch_seed + index,
                                "outcome": classify(flags, index),
                                "steps": step,
                                "peak_grip_force_n": float(env.peak_grip_force_n[index].item()),
                                "peak_stem_tension_n": float(env.peak_stem_tension_n[index].item()),
                                "damage_threshold_n": float(env.damage_threshold_n[index].item()),
                                "detach_threshold_n": float(env.detach_threshold_n[index].item()),
                            }
                        active &= ~just_finished
                        if step >= args.max_episode_steps or not bool(active.any().item()):
                            break
                    if step >= args.max_episode_steps or not bool(active.any().item()):
                        break
            for index in sorted(batch_records):
                if len(records) >= args.episodes:
                    break
                records.append(batch_records[index])
            successes = sum(item["outcome"] == "success" for item in records)
            print(
                f"[FLOW EVAL] episodes={len(records)}/{args.episodes} "
                f"success={successes / max(len(records), 1):.1%}",
                flush=True,
            )
            batch += 1
    finally:
        env.close()

    successes = sum(item["outcome"] == "success" for item in records)
    failures = Counter(item["outcome"] for item in records if item["outcome"] != "success")
    report = {
        "task": "TacEx-GrapePick-Force-v0",
        "policy": "frozen_state_tactile_conditional_flow_matching",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "seed_start": args.seed,
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / max(len(records), 1),
        "wilson_95_ci": wilson(successes, len(records)),
        "failure_counts": dict(failures),
        "damage_force_randomization_range_n": list(args.damage_range),
        "detach_force_randomization_range_n": list(args.detach_range),
        "action_repeat": args.action_repeat,
        "chunk_execute_steps": args.chunk_execute_steps,
        "integration_steps": args.integration_steps,
        "native_flow_noise": "independent standard Gaussian per action chunk",
        "outcomes": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[SUMMARY] " + json.dumps({k: v for k, v in report.items() if k != "outcomes"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
