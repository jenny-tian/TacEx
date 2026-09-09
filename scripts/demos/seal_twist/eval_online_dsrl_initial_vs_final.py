#!/usr/bin/env python3
"""Paired holdout evaluation of initial and trained SealTwist controllers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _reexec_with_cuda_libraries() -> None:
    py_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / py_dir / "site-packages"
    required = [
        site_packages / "torch" / "lib",
        site_packages / "nvidia" / "cudnn" / "lib",
    ]
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


ROOT = Path(__file__).resolve().parents[3]
GRAPE_HELPERS = ROOT / "scripts" / "reinforcement_learning" / "grape_pick"
if str(GRAPE_HELPERS) not in sys.path:
    sys.path.insert(0, str(GRAPE_HELPERS))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--flow-checkpoint",
    type=Path,
    default=ROOT / "exp_report" / "bottle" / "checkpoints" / "flow_bc_6d_force_v5.pt",
)
parser.add_argument(
    "--dsrl-checkpoint",
    type=Path,
    default=(
        ROOT
        / "exp_report"
        / "bottle"
        / "online_codex_vlm_dsrl"
        / "seed_907700_n100"
        / "checkpoints"
        / "final.pt"
    ),
)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num-envs", "--num_envs", type=int, default=20)
parser.add_argument("--seed", type=int, default=917700)
parser.add_argument(
    "--pair-order",
    choices=("initial-first", "final-first"),
    default="initial-first",
    help="Counterbalance simulator-history effects by choosing which controller runs first.",
)
parser.add_argument("--max-episode-steps", type=int, default=900)
parser.add_argument("--physical-force-bounds", type=float, nargs=2, default=(6.2, 8.8))
parser.add_argument("--initial-force-range", type=float, nargs=2, default=(7.2, 8.2))
parser.add_argument("--minimum-force-range", type=float, nargs=2, default=(4.7, 6.1))
parser.add_argument("--maximum-force-range", type=float, nargs=2, default=(9.0, 11.0))
parser.add_argument("--position-range", type=float, nargs=2, default=(0.010, 0.010))
parser.add_argument("--alignment-tolerance", type=float, default=0.012)
parser.add_argument("--normalized-observation-clip", type=float, default=6.0)
parser.add_argument(
    "--constant-final-correction",
    type=float,
    nargs="+",
    default=None,
    help="Diagnostic override for the final actor action; omit for learned policy evaluation.",
)
parser.add_argument("--physx-contact-capacity", type=int, default=2**20)
parser.add_argument("--policy-device", choices=("auto", "cpu", "cuda"), default="cuda")
parser.add_argument(
    "--report",
    type=Path,
    default=(
        ROOT
        / "exp_report"
        / "bottle"
        / "online_codex_vlm_dsrl"
        / "seed_907700_n100"
        / "evaluation"
        / "initial_vs_final_seed_917700_n100.json"
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch

from learning import SquashedGaussianActor
from online_rl_support import ACTOR_OBSERVATION_DIM, force_conditioned_observation
from tacex_tasks.seal_twist.flow_policy import (
    PREVIOUS_GRIPPER_WIDTH_OBSERVATION_INDEX,
    environment_flow_observation,
    load_flow_checkpoint,
)
from tacex_tasks.seal_twist.seal_twist_env import SealTwistEnv
from tacex_tasks.seal_twist.seal_twist_env_cfg import SealTwistEnvCfg


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int) -> list[float]:
    if total < 1:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def exact_mcnemar_pvalue(final_only: int, initial_only: int) -> float:
    discordant = final_only + initial_only
    if discordant == 0:
        return 1.0
    smaller = min(final_only, initial_only)
    one_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return min(1.0, 2.0 * one_tail)


def classify(flags: dict[str, torch.Tensor], index: int) -> str:
    if bool(flags["success"][index].item()):
        return "success"
    if bool(flags["over_force"][index].item()) or bool(flags["over_torqued"][index].item()):
        return "over_force"
    if bool(flags["alignment_failure"][index].item()) or bool(flags["slipped"][index].item()):
        return "alignment_failure"
    return "under_force"


def actor_observation(
    normalized_flow_observation: torch.Tensor,
    force_range: tuple[float, float],
    left_force_n: torch.Tensor,
    right_force_n: torch.Tensor,
    observation_dim: int,
) -> torch.Tensor:
    if observation_dim == ACTOR_OBSERVATION_DIM:
        return force_conditioned_observation(
            normalized_flow_observation,
            force_low_n=force_range[0],
            force_high_n=force_range[1],
            left_force_n=left_force_n,
            right_force_n=right_force_n,
        )
    if observation_dim != 31:
        raise ValueError(f"unsupported actor observation dimension: {observation_dim}")
    batch = len(normalized_flow_observation)
    low = torch.full(
        (batch, 1),
        force_range[0] / 10.0,
        dtype=normalized_flow_observation.dtype,
        device=normalized_flow_observation.device,
    )
    high = torch.full_like(low, force_range[1] / 10.0)
    mean_force_n = 0.5 * (left_force_n + right_force_n)
    error = (0.5 * (force_range[0] + force_range[1]) - mean_force_n.reshape(-1, 1)) / 10.0
    return torch.cat((normalized_flow_observation, low, high, error), dim=-1)


def expand_correction(
    correction: torch.Tensor,
    *,
    horizon: int,
    action_dim: int,
    learned_noise_steps: int,
    action_indices: tuple[int, ...],
    residual_scale: float,
) -> torch.Tensor:
    expected = learned_noise_steps * len(action_indices)
    if correction.ndim != 2 or correction.shape[-1] != expected:
        raise ValueError(f"expected correction shape [B,{expected}], got {tuple(correction.shape)}")
    learned = correction.reshape(len(correction), learned_noise_steps, len(action_indices))
    short = correction.new_zeros((len(correction), learned_noise_steps, action_dim))
    short[:, :, list(action_indices)] = learned
    if learned_noise_steps < horizon:
        padding = short[:, -1:, :].expand(-1, horizon - learned_noise_steps, -1)
        short = torch.cat((short, padding), dim=1)
    return residual_scale * short


def run_controller(
    env: SealTwistEnv,
    loaded: Any,
    *,
    batch_seed: int,
    controller: str,
    force_range: tuple[float, float],
    actor: SquashedGaussianActor | None,
    constant_correction: tuple[float, ...] | None,
    checkpoint: dict[str, Any],
    environment_device: torch.device,
    policy_device: torch.device,
) -> list[dict[str, Any]]:
    env.reset(seed=batch_seed)
    active = torch.ones(args.num_envs, dtype=torch.bool, device=environment_device)
    previous_action: torch.Tensor | None = None
    correction_dim = int(checkpoint["learned_noise_steps"]) * len(
        checkpoint["noise_action_indices"]
    )
    correction_sum = torch.zeros(
        (args.num_envs, correction_dim), device=policy_device
    )
    correction_count = torch.zeros(args.num_envs, dtype=torch.long, device=policy_device)
    records: dict[int, dict[str, Any]] = {}
    physics_step = 0

    while physics_step < args.max_episode_steps and bool(active.any().item()):
        flags_before = env.get_task_flags()
        left_before = flags_before["left_finger_force_n"]
        right_before = flags_before["right_finger_force_n"]
        mean_before = 0.5 * (left_before + right_before)
        raw_observation = environment_flow_observation(
            env, max_episode_steps=loaded.max_episode_steps
        ).to(policy_device)
        normalized = loaded.normalize_observation(raw_observation)
        if args.normalized_observation_clip > 0.0:
            normalized = normalized.clamp(
                -args.normalized_observation_clip, args.normalized_observation_clip
            )
        dsrl_active = (
            (physics_step >= int(checkpoint["dsrl_activation"]["minimum_physics_step"]))
            & flags_before["bilateral_contact"].to(environment_device)
            & active
        )
        with torch.inference_mode():
            if actor is None:
                correction = torch.zeros(
                    (args.num_envs, correction_dim),
                    device=policy_device,
                )
            else:
                policy_observation = actor_observation(
                    normalized,
                    force_range,
                    left_before.to(policy_device),
                    right_before.to(policy_device),
                    int(checkpoint["actor_observation_dim"]),
                )
                if constant_correction is None:
                    correction = actor.sample(policy_observation, deterministic=True)[0]
                else:
                    correction = torch.tensor(
                        constant_correction,
                        dtype=policy_observation.dtype,
                        device=policy_device,
                    ).expand(args.num_envs, -1)
                correction = torch.where(
                    dsrl_active.to(policy_device).unsqueeze(-1),
                    correction,
                    torch.zeros_like(correction),
                )
                correction_sum += correction.abs()
                correction_count += dsrl_active.to(policy_device).long()

            decoder_noise = expand_correction(
                (
                    correction.index_fill(
                        1,
                        torch.tensor(
                            [int(checkpoint["semantic_force_balance_residual"]["actor_coordinate"])],
                            device=correction.device,
                        ),
                        0.0,
                    )
                    if checkpoint.get("semantic_force_balance_residual", {}).get("enabled", False)
                    else correction
                ),
                horizon=int(checkpoint["flow_horizon"]),
                action_dim=int(checkpoint["flow_action_dim"]),
                learned_noise_steps=int(checkpoint["learned_noise_steps"]),
                action_indices=tuple(map(int, checkpoint["noise_action_indices"])),
                residual_scale=float(checkpoint["noise_residual_scale"]),
            )
            normalized_chunk = loaded.model.decode(
                normalized,
                decoder_noise,
                integration_steps=int(checkpoint["flow_execution"]["integration_steps"]),
                solver=str(checkpoint["flow_execution"]["solver"]),
            )
            action_chunk = loaded.denormalize_action(normalized_chunk)
            action_chunk[..., :3] += raw_observation[:, None, :3]
            semantic_residual = checkpoint.get("semantic_force_balance_residual", {})
            if semantic_residual.get("enabled", False):
                actor_coordinate = int(semantic_residual["actor_coordinate"])
                tool_action_index = int(semantic_residual["tool_action_index"])
                action_chunk[..., tool_action_index] += (
                    correction[:, actor_coordinate, None]
                    * float(semantic_residual["maximum_offset_m"])
                )
            action_chunk[..., 9] += raw_observation[
                :, None, PREVIOUS_GRIPPER_WIDTH_OBSERVATION_INDEX
            ]
            contact = torch.maximum(left_before, right_before) >= 0.05
            feedback_active = (
                physics_step >= int(checkpoint["force_projection"]["start_step"])
            ) & contact
            target_force = 0.5 * (force_range[0] + force_range[1])
            width_correction = (
                float(checkpoint["force_projection"]["gain_m_per_n"])
                * (mean_before.to(policy_device) - target_force)
            ).clamp(
                -float(checkpoint["force_projection"]["maximum_step_m"]),
                float(checkpoint["force_projection"]["maximum_step_m"]),
            )
            width_correction = torch.where(
                feedback_active.to(policy_device), width_correction, torch.zeros_like(width_correction)
            )
            action_chunk[..., 9] += width_correction[:, None]
            width_limits = tuple(map(float, checkpoint["force_projection"]["width_limits_m"]))
            action_chunk[..., 9].clamp_(width_limits[0], width_limits[1])
            action_chunk = action_chunk.to(environment_device)

        chunk_steps = int(checkpoint["flow_execution"]["chunk_execute_steps"])
        action_repeat = int(checkpoint["flow_execution"]["action_repeat"])
        smoothing = float(checkpoint["flow_execution"]["action_smoothing"])
        for chunk_index in range(chunk_steps):
            target_action = action_chunk[:, chunk_index]
            for _ in range(action_repeat):
                if previous_action is None:
                    action = target_action
                else:
                    action = smoothing * target_action + (1.0 - smoothing) * previous_action
                previous_action = action
                env._pre_physics_step(action)
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
                physics_step += 1
                flags = env.get_task_flags()
                terminal = (
                    flags["success"]
                    | flags["over_force"]
                    | flags["over_torqued"]
                    | flags["under_force"]
                    | flags["alignment_failure"]
                    | flags["slipped"]
                )
                if physics_step >= args.max_episode_steps:
                    terminal |= active
                just_finished = active & terminal
                for index in torch.nonzero(just_finished, as_tuple=False).squeeze(-1).cpu().tolist():
                    count = int(correction_count[index].item())
                    records[index] = {
                        "seed": batch_seed + index,
                        "controller": controller,
                        "success": bool(flags["success"][index].item()),
                        "outcome": classify(flags, index),
                        "steps": physics_step,
                        "force_range_n": list(force_range),
                        "minimum_force_threshold_n": float(
                            env.minimum_force_threshold_n[index].item()
                        ),
                        "maximum_force_threshold_n": float(
                            env.maximum_force_threshold_n[index].item()
                        ),
                        "cap_position_offset_m": env.sampled_cap_position_offset_m[index]
                        .detach()
                        .cpu()
                        .tolist(),
                        "peak_left_force_n": float(env.peak_left_force_n[index].item()),
                        "peak_right_force_n": float(env.peak_right_force_n[index].item()),
                        "peak_torque_nm": float(env.peak_torque_nm[index].item()),
                        "final_torque_nm": float(flags["seal_torque_nm"][index].item()),
                        "final_left_finger_force_n": float(
                            flags["left_finger_force_n"][index].item()
                        ),
                        "final_right_finger_force_n": float(
                            flags["right_finger_force_n"][index].item()
                        ),
                        "final_minimum_finger_force_n": float(
                            flags["minimum_finger_force_n"][index].item()
                        ),
                        "final_maximum_finger_force_n": float(
                            flags["maximum_finger_force_n"][index].item()
                        ),
                        "final_alignment_error_m": float(
                            flags["alignment_error_m"][index].item()
                        ),
                        "actor_active_decisions": count,
                        "mean_abs_actor_correction": (
                            (correction_sum[index] / max(count, 1)).detach().cpu().tolist()
                        ),
                    }
                active &= ~just_finished
                if physics_step >= args.max_episode_steps or not bool(active.any().item()):
                    break
            if physics_step >= args.max_episode_steps or not bool(active.any().item()):
                break
    return [records[index] for index in sorted(records)]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(record["success"] for record in records)
    failures = Counter(record["outcome"] for record in records if not record["success"])
    active_decisions = sum(record["actor_active_decisions"] for record in records)
    weighted_correction = [
        sum(
            record["mean_abs_actor_correction"][coordinate]
            * record["actor_active_decisions"]
            for record in records
        )
        / max(active_decisions, 1)
        for coordinate in range(len(records[0]["mean_abs_actor_correction"]))
    ]
    return {
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "wilson_95_ci": wilson_interval(successes, len(records)),
        "failure_counts": dict(failures),
        "actor_active_decisions": active_decisions,
        "mean_abs_actor_correction": weighted_correction,
    }


def main() -> None:
    if args.episodes < 1 or args.num_envs < 1:
        parser.error("episodes and num-envs must be positive")
    if args.physx_contact_capacity < 2**16:
        parser.error("physx-contact-capacity is implausibly small")
    flow_path = args.flow_checkpoint.expanduser().resolve()
    dsrl_path = args.dsrl_checkpoint.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {report_path}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch.load(dsrl_path, map_location="cpu", weights_only=False)
    required_contract = "seal_twist_zero_latent_flow_dsrl_sac_with_online_codex_vlm_force_adaptation"
    if checkpoint.get("contract") != required_contract:
        raise ValueError("DSRL checkpoint contract mismatch")
    if checkpoint.get("flow_checkpoint_sha256") != sha256(flow_path):
        raise ValueError("DSRL checkpoint does not reference the supplied Flow checkpoint")

    cfg = SealTwistEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.enable_cameras = False
    cfg.enable_gelsight = False
    cfg.randomize_labware_position = True
    cfg.labware_pos_randomization_xy = tuple(map(float, args.position_range))
    cfg.labware_yaw_randomization = 0.0
    cfg.randomize_expert_targets = False
    cfg.randomize_force_thresholds = True
    cfg.minimum_force_randomization_range_n = tuple(map(float, args.minimum_force_range))
    cfg.maximum_force_randomization_range_n = tuple(map(float, args.maximum_force_range))
    cfg.alignment_tolerance_m = float(args.alignment_tolerance)
    cfg.success_hold_steps = 5
    cfg.force_violation_dwell_steps = 5
    cfg.alignment_failure_dwell_steps = 5
    cfg.episode_length_s = max(args.max_episode_steps * cfg.sim.dt, 1.0)
    cfg.sim.physx.gpu_max_rigid_contact_count = args.physx_contact_capacity
    cfg.sim.physx.gpu_max_rigid_patch_count = args.physx_contact_capacity
    if args.device:
        cfg.sim.device = args.device

    env = SealTwistEnv(cfg)
    environment_device = torch.device(cfg.sim.device)
    policy_device = torch.device(
        "cuda"
        if args.policy_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.policy_device == "auto"
        else args.policy_device
    )
    loaded = load_flow_checkpoint(flow_path, device=policy_device)
    loaded.model.requires_grad_(False)
    loaded.model.eval()
    if loaded.model.config.horizon != int(checkpoint["flow_horizon"]):
        raise ValueError("Flow horizon mismatch")
    if loaded.model.config.action_dim != int(checkpoint["flow_action_dim"]):
        raise ValueError("Flow action dimension mismatch")

    action_dim = int(checkpoint["learned_noise_steps"]) * len(
        checkpoint["noise_action_indices"]
    )
    actor = SquashedGaussianActor(
        int(checkpoint["actor_observation_dim"]),
        action_dim,
        initial_log_std=-2.5,
    ).to(policy_device)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.eval()
    constant_final_correction = (
        None
        if args.constant_final_correction is None
        else tuple(map(float, args.constant_final_correction))
    )
    if constant_final_correction is not None and len(constant_final_correction) != action_dim:
        raise ValueError(
            f"constant-final-correction requires {action_dim} values, got "
            f"{len(constant_final_correction)}"
        )
    initial_range = tuple(map(float, args.initial_force_range))
    final_range = tuple(map(float, checkpoint["current_force_range_n"]))
    low, high = map(float, args.physical_force_bounds)
    if not (low <= initial_range[0] < initial_range[1] <= high):
        raise ValueError("initial force range violates physical bounds")
    if not (low <= final_range[0] < final_range[1] <= high):
        raise ValueError("final force range violates physical bounds")

    initial_records: list[dict[str, Any]] = []
    final_records: list[dict[str, Any]] = []
    try:
        batch = 0
        while len(initial_records) < args.episodes and simulation_app.is_running():
            batch_seed = args.seed + batch * args.num_envs
            controller_specs = {
                "initial": {
                    "controller": "initial_zero_actor",
                    "force_range": initial_range,
                    "actor": None,
                    "constant_correction": None,
                },
                "final": {
                    "controller": "final_deterministic_actor",
                    "force_range": final_range,
                    "actor": actor,
                    "constant_correction": constant_final_correction,
                },
            }
            execution_order = (
                ("initial", "final")
                if args.pair_order == "initial-first"
                else ("final", "initial")
            )
            batch_results: dict[str, list[dict[str, Any]]] = {}
            for name in execution_order:
                spec = controller_specs[name]
                batch_results[name] = run_controller(
                    env,
                    loaded,
                    batch_seed=batch_seed,
                    controller=str(spec["controller"]),
                    force_range=spec["force_range"],
                    actor=spec["actor"],
                    constant_correction=spec["constant_correction"],
                    checkpoint=checkpoint,
                    environment_device=environment_device,
                    policy_device=policy_device,
                )
            initial_batch = batch_results["initial"]
            final_batch = batch_results["final"]
            remaining = args.episodes - len(initial_records)
            initial_records.extend(initial_batch[:remaining])
            final_records.extend(final_batch[:remaining])
            initial_successes = sum(record["success"] for record in initial_records)
            final_successes = sum(record["success"] for record in final_records)
            print(
                f"[PAIRED EVAL] evaluated={len(initial_records)}/{args.episodes} "
                f"initial={initial_successes / len(initial_records):.1%} "
                f"final={final_successes / len(final_records):.1%}",
                flush=True,
            )
            batch += 1
    finally:
        env.close()

    if len(initial_records) != args.episodes or len(final_records) != args.episodes:
        raise RuntimeError("evaluation ended before the requested paired episode count")
    if [record["seed"] for record in initial_records] != [record["seed"] for record in final_records]:
        raise RuntimeError("initial and final seed sequences differ")

    paired_randomization_matches = []
    pairs = []
    for initial, final in zip(initial_records, final_records, strict=True):
        threshold_match = math.isclose(
            initial["minimum_force_threshold_n"], final["minimum_force_threshold_n"], abs_tol=1e-7
        ) and math.isclose(
            initial["maximum_force_threshold_n"], final["maximum_force_threshold_n"], abs_tol=1e-7
        )
        position_match = all(
            math.isclose(a, b, abs_tol=1e-7)
            for a, b in zip(
                initial["cap_position_offset_m"], final["cap_position_offset_m"], strict=True
            )
        )
        paired_randomization_matches.append(threshold_match and position_match)
        pairs.append(
            {
                "seed": initial["seed"],
                "initial": initial,
                "final": final,
                "randomization_matches": threshold_match and position_match,
            }
        )

    both_success = sum(pair["initial"]["success"] and pair["final"]["success"] for pair in pairs)
    final_only = sum((not pair["initial"]["success"]) and pair["final"]["success"] for pair in pairs)
    initial_only = sum(pair["initial"]["success"] and (not pair["final"]["success"]) for pair in pairs)
    neither = args.episodes - both_success - final_only - initial_only
    initial_summary = summarize(initial_records)
    final_summary = summarize(final_records)
    report = {
        "task": "TacEx-SealTwist-Torque-v0",
        "comparison": "paired holdout evaluation of initial versus final online controller",
        "training_seed": int(checkpoint["train_seed"]),
        "holdout_seed_start": args.seed,
        "holdout_seed_end": args.seed + args.episodes - 1,
        "episodes_per_controller": args.episodes,
        "pair_execution_order": args.pair_order,
        "paired_episode_seeds": True,
        "paired_randomization_matches": all(paired_randomization_matches),
        "evaluation_updates_policy": False,
        "evaluation_uses_vlm": False,
        "initial_controller": {
            "definition": "frozen zero-latent Flow BC, zero SAC correction, configured comparison force interval",
            "force_range_n": list(initial_range),
            **initial_summary,
        },
        "final_controller": {
            "definition": (
                "frozen Flow BC plus deterministic trained SAC correction and final tactile force interval"
                if constant_final_correction is None
                else "diagnostic constant residual plus final tactile force interval"
            ),
            "constant_correction_override": constant_final_correction,
            "force_range_n": list(final_range),
            **final_summary,
        },
        "paired_comparison": {
            "success_rate_difference_final_minus_initial": (
                final_summary["success_rate"] - initial_summary["success_rate"]
            ),
            "percentage_point_difference_final_minus_initial": 100.0
            * (final_summary["success_rate"] - initial_summary["success_rate"]),
            "both_success": both_success,
            "final_only_success": final_only,
            "initial_only_success": initial_only,
            "neither_success": neither,
            "exact_mcnemar_two_sided_p": exact_mcnemar_pvalue(final_only, initial_only),
        },
        "flow_checkpoint": str(flow_path),
        "flow_checkpoint_sha256": sha256(flow_path),
        "dsrl_checkpoint": str(dsrl_path),
        "dsrl_checkpoint_sha256": sha256(dsrl_path),
        "dsrl_updates_at_checkpoint": int(checkpoint["updates"]),
        "flow_execution": checkpoint["flow_execution"],
        "noise_action_indices": checkpoint["noise_action_indices"],
        "noise_residual_scale": checkpoint["noise_residual_scale"],
        "semantic_force_balance_residual": checkpoint.get(
            "semantic_force_balance_residual",
            {"enabled": False},
        ),
        "dsrl_activation": checkpoint["dsrl_activation"],
        "force_projection": checkpoint["force_projection"],
        "randomized_minimum_force_population_range_n": list(args.minimum_force_range),
        "randomized_maximum_force_population_range_n": list(args.maximum_force_range),
        "position_randomization_half_range_m": list(args.position_range),
        "alignment_tolerance_m": args.alignment_tolerance,
        "torque_window_nm": list(cfg.torque_window_nm),
        "max_episode_steps": args.max_episode_steps,
        "simulation_device": str(environment_device),
        "policy_device": str(policy_device),
        "pairs": pairs,
    }
    if not report["paired_randomization_matches"]:
        raise RuntimeError("paired environment randomization did not reproduce exactly")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "[SUMMARY] "
        + json.dumps({key: value for key, value in report.items() if key != "pairs"}, ensure_ascii=False),
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
