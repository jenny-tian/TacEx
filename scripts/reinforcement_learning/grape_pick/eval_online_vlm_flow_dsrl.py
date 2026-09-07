#!/usr/bin/env python3
"""Paired evaluation of Flow BC, force control, and trained Flow-noise DSRL."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path


def _reexec_with_cuda_libraries() -> None:
    py_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / py_dir / "site-packages"
    required = [site_packages / "torch" / "lib", site_packages / "nvidia" / "cudnn" / "lib"]
    current = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item]
    missing = [str(item) for item in required if item.is_dir() and str(item) not in current]
    if missing:
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = os.pathsep.join([*missing, *current])
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)


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


METHODS = ("flow_bc", "flow_bc_force_controller", "vlm_flow_dsrl")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dsrl-checkpoint", type=Path, required=True)
parser.add_argument("--flow-checkpoint", type=Path, default=None)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=97000)
parser.add_argument("--max-episode-steps", type=int, default=790)
parser.add_argument("--action-repeat", type=int, default=1)
parser.add_argument("--integration-steps", type=int, default=16)
parser.add_argument("--action-smoothing", type=float, default=0.35)
parser.add_argument("--damage-range", type=float, nargs=2, default=(3.4, 4.6))
parser.add_argument("--detach-range", type=float, nargs=2, default=(3.0, 3.7))
parser.add_argument("--physx-contact-capacity", type=int, default=2**20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch

from grape_flow_policy import (
    FLOW_NOISE_SEED_OFFSET,
    FlowNoiseDSRLLayout,
    load_flow_checkpoint,
    pull_phase_noise_gate,
)
from grape_force_control import ForceControllerConfig, ForceRange, GrapeForceController
from learning import SACResidualAgent
from tacex_tasks.grape_pick.grape_pick_env import GrapePickEnv
from tacex_tasks.grape_pick.grape_pick_env_cfg import GrapePickEnvCfg
from tacex_tasks.grape_pick.policy import environment_policy_observation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def wilson(successes: int, total: int) -> list[float]:
    if total < 1:
        return [0.0, 0.0]
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total
        + z * z / (4.0 * total * total)
    ) / denominator
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


def actor_observation(
    normalized: torch.Tensor,
    force_range: ForceRange,
    grip_force: torch.Tensor,
) -> torch.Tensor:
    low = torch.full(
        (len(normalized), 1),
        force_range.minimum_n / 4.0,
        device=normalized.device,
        dtype=normalized.dtype,
    )
    high = torch.full_like(low, force_range.maximum_n / 4.0)
    error = (force_range.center_n - grip_force[:, None]) / 4.0
    return torch.cat((normalized, low, high, error), dim=-1)


def evaluate(
    env: GrapePickEnv,
    *,
    method: str,
    loaded,
    layout: FlowNoiseDSRLLayout,
    agent: SACResidualAgent,
    final_range: ForceRange,
    dsrl_start_step: int,
    controller_config: ForceControllerConfig,
) -> dict:
    device = env.device
    records: list[dict] = []
    batch_index = 0
    use_actor = method == "vlm_flow_dsrl"
    use_controller = method in {"flow_bc_force_controller", "vlm_flow_dsrl"}
    while len(records) < args.episodes and simulation_app.is_running():
        batch_seed = args.seed + batch_index * args.num_envs
        env.reset(seed=batch_seed)
        active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
        previous_action: torch.Tensor | None = None
        controller = GrapeForceController(
            final_range,
            controller_config,
        )
        generator = torch.Generator(device=device).manual_seed(
            batch_seed + FLOW_NOISE_SEED_OFFSET
        )
        step = 0
        batch_records: dict[int, dict] = {}
        correction_abs_sum = torch.zeros(args.num_envs, device=device)
        correction_count = torch.zeros(args.num_envs, device=device)
        dsrl_active_decisions = torch.zeros(args.num_envs, device=device)
        controller_active_steps = torch.zeros(args.num_envs, device=device)
        safety_override_steps = torch.zeros(args.num_envs, device=device)
        while step < args.max_episode_steps and bool(active.any().item()):
            metrics = env.get_contact_force_metrics()
            raw_observation = environment_policy_observation(
                env, max_episode_steps=loaded.max_episode_steps
            )
            normalized = loaded.normalize_observation(raw_observation)
            if use_actor:
                dsrl_active = pull_phase_noise_gate(
                    step,
                    metrics["bilateral_contact"],
                    env.stem_attached,
                    minimum_step=dsrl_start_step,
                ) & active
                with torch.inference_mode():
                    proposed_correction = agent.act(
                        actor_observation(
                            normalized, final_range, metrics["grip_force_n"]
                        ),
                        deterministic=True,
                    )
                    correction = torch.where(
                        dsrl_active[:, None],
                        proposed_correction,
                        torch.zeros_like(proposed_correction),
                    )
            else:
                correction = torch.zeros(
                    (args.num_envs, layout.correction_dim), device=device
                )
                dsrl_active = torch.zeros_like(active)
            correction_abs_sum += correction.abs().mean(dim=-1) * dsrl_active.float()
            correction_count += dsrl_active.float()
            dsrl_active_decisions += dsrl_active.float()
            native_noise = torch.randn(
                (args.num_envs, layout.horizon, layout.action_dim),
                generator=generator,
                device=device,
            )
            decoder_noise = layout.compose_noise(native_noise, correction)
            with torch.inference_mode():
                normalized_chunk = loaded.model.decode(
                    normalized,
                    decoder_noise,
                    integration_steps=args.integration_steps,
                    solver="heun",
                )
                target = loaded.denormalize_action(normalized_chunk)[:, 0]
            for _ in range(args.action_repeat):
                if previous_action is None:
                    policy_action = target
                else:
                    alpha = float(args.action_smoothing)
                    policy_action = alpha * target + (1.0 - alpha) * previous_action
                previous_action = policy_action
                if use_controller:
                    metrics = env.get_contact_force_metrics()
                    executed_action, diagnostics = controller.control(
                        policy_action,
                        contact_force_n=metrics["grip_force_n"],
                        bilateral_contact=metrics["bilateral_contact"],
                        dt_s=float(env.physics_dt),
                    )
                    controller_active_steps += diagnostics.active.float() * active.float()
                    safety_override_steps += diagnostics.safety_override.float() * active.float()
                else:
                    executed_action = policy_action
                env._pre_physics_step(executed_action)
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
                        "controller_active_steps": int(controller_active_steps[index].item()),
                        "safety_override_steps": int(safety_override_steps[index].item()),
                        "mean_abs_noise_correction": float(
                            correction_abs_sum[index].item()
                            / max(correction_count[index].item(), 1.0)
                        ),
                        "dsrl_active_decisions": int(
                            dsrl_active_decisions[index].item()
                        ),
                    }
                active &= ~just_finished
                if step >= args.max_episode_steps or not bool(active.any().item()):
                    break
        for index in sorted(batch_records):
            if len(records) >= args.episodes:
                break
            records.append(batch_records[index])
            append_jsonl(
                args.report.with_suffix(args.report.suffix + f".{method}.jsonl"),
                batch_records[index],
            )
        successes = sum(item["outcome"] == "success" for item in records)
        print(
            f"[EVAL] method={method} episodes={len(records)}/{args.episodes} "
            f"success={successes / max(len(records), 1):.1%}",
            flush=True,
        )
        batch_index += 1

    successes = sum(item["outcome"] == "success" for item in records)
    failures = Counter(
        item["outcome"] for item in records if item["outcome"] != "success"
    )
    return {
        "method": method,
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / max(len(records), 1),
        "wilson_95_ci": wilson(successes, len(records)),
        "failure_counts": dict(failures),
        "mean_peak_grip_force_n": sum(item["peak_grip_force_n"] for item in records)
        / max(len(records), 1),
        "max_peak_grip_force_n": max(
            (item["peak_grip_force_n"] for item in records), default=0.0
        ),
        "mean_abs_noise_correction": sum(
            item["mean_abs_noise_correction"] for item in records
        )
        / max(len(records), 1),
        "outcomes": records,
    }


def main() -> None:
    checkpoint = torch.load(args.dsrl_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != "grape_flow_initial_noise_dsrl_sac_with_online_vlm_force_control":
        raise ValueError("checkpoint does not satisfy the Grape Flow-noise DSRL contract")
    if int(checkpoint.get("contract_version", -1)) != 2:
        raise ValueError("evaluation requires the pull-gated Flow-noise DSRL contract v2")
    execution_metadata_present = "flow_execution" in checkpoint
    execution = checkpoint.get(
        "flow_execution",
        {
            "action_repeat": 1,
            "integration_steps": 16,
            "action_smoothing": 0.35,
            "native_noise_distribution": "independent standard Gaussian per policy decision",
            "native_noise_seed_offset": FLOW_NOISE_SEED_OFFSET,
        },
    )
    expected_execution = {
        "action_repeat": args.action_repeat,
        "integration_steps": args.integration_steps,
        "action_smoothing": args.action_smoothing,
    }
    for key, actual in expected_execution.items():
        expected = execution[key]
        if actual != expected:
            raise ValueError(
                f"evaluation {key}={actual} differs from training checkpoint value {expected}"
            )
    flow_path = (
        Path(checkpoint["flow_checkpoint"])
        if args.flow_checkpoint is None
        else args.flow_checkpoint
    ).resolve()
    if sha256(flow_path) != checkpoint["flow_checkpoint_sha256"]:
        raise ValueError("Flow checkpoint hash does not match the DSRL training contract")

    config = GrapePickEnvCfg()
    config.scene.num_envs = args.num_envs
    config.seed = args.seed
    config.enable_cameras = False
    config.enable_gelsight = False
    config.randomize_force_thresholds = True
    config.sim.physx.gpu_max_rigid_contact_count = args.physx_contact_capacity
    config.sim.physx.gpu_max_rigid_patch_count = args.physx_contact_capacity
    config.damage_force_randomization_range_n = tuple(args.damage_range)
    config.detach_force_randomization_range_n = tuple(args.detach_range)
    config.episode_length_s = max(args.max_episode_steps * config.sim.dt, 1.0)
    if args.device:
        config.sim.device = args.device
    env = GrapePickEnv(config)
    device = torch.device(config.sim.device)
    loaded = load_flow_checkpoint(flow_path, device=device)
    loaded.model.requires_grad_(False).eval()
    layout = FlowNoiseDSRLLayout(
        horizon=int(checkpoint["flow_horizon"]),
        action_dim=int(checkpoint["flow_action_dim"]),
        learned_noise_steps=int(checkpoint["learned_noise_steps"]),
        residual_scale=float(checkpoint["noise_residual_scale"]),
        padding_mode=str(checkpoint["noise_padding_mode"]),
        active_action_indices=tuple(checkpoint["noise_action_indices"]),
    )
    agent = SACResidualAgent(14, layout.correction_dim, device)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.actor.eval()
    final_range = ForceRange(*map(float, checkpoint["current_force_range_n"]))
    controller_metadata_present = "force_controller_config" in checkpoint
    controller_config = ForceControllerConfig(
        **checkpoint.get(
            "force_controller_config",
            {
                "hard_force_limit_n": float(args.damage_range[0]) - 0.15,
                "free_space_close_rate_m_s": 0.006,
                "maximum_width_rate_m_s": 0.006,
                "safety_open_rate_m_s": 0.006,
                "release_hysteresis_steps": 120,
            },
        )
    )
    method_reports: list[dict] = []
    try:
        for method in args.methods:
            method_reports.append(
                evaluate(
                    env,
                    method=method,
                    loaded=loaded,
                    layout=layout,
                    agent=agent,
                    final_range=final_range,
                    dsrl_start_step=int(
                        checkpoint["dsrl_activation"]["minimum_physics_step"]
                    ),
                    controller_config=controller_config,
                )
            )
    finally:
        env.close()

    compact = {
        item["method"]: {
            key: value for key, value in item.items() if key != "outcomes"
        }
        for item in method_reports
    }
    report = {
        "experiment": "paired single-seed GrapePick Flow-noise DSRL evaluation",
        "evaluation_seed_start": args.seed,
        "episodes_per_method": args.episodes,
        "paired_environment_seeds": True,
        "paired_native_flow_noise": True,
        "damage_force_randomization_range_n": list(args.damage_range),
        "detach_force_randomization_range_n": list(args.detach_range),
        "flow_checkpoint": str(flow_path),
        "flow_checkpoint_sha256": sha256(flow_path),
        "dsrl_checkpoint": str(args.dsrl_checkpoint.resolve()),
        "dsrl_checkpoint_sha256": sha256(args.dsrl_checkpoint),
        "dsrl_action_semantics": checkpoint["rl_action_semantics"],
        "noise_action_indices": checkpoint["noise_action_indices"],
        "dsrl_activation": checkpoint["dsrl_activation"],
        "flow_execution": execution,
        "force_controller_config": asdict(controller_config),
        "checkpoint_metadata_fallbacks": {
            "flow_execution": not execution_metadata_present,
            "force_controller_config": not controller_metadata_present,
        },
        "final_vlm_force_range_n": final_range.as_list(),
        "methods": method_reports,
        "summary": compact,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[SUMMARY] " + json.dumps(compact), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
