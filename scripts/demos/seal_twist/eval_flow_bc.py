#!/usr/bin/env python3
"""Evaluate 6D-wrench Flow Matching BC on the frozen SealTwist protocol."""

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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("exp_report/bottle/checkpoints/flow_bc_6d_force.pt"),
)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num-envs", "--num_envs", type=int, default=20)
parser.add_argument("--seed", type=int, default=80000)
parser.add_argument("--policy-seed-offset", type=int, default=631_337)
parser.add_argument("--max-episode-steps", "--max_episode_steps", type=int, default=900)
parser.add_argument("--chunk-execute-steps", type=int, default=4)
parser.add_argument(
    "--action-repeat",
    type=int,
    default=4,
    help="Physics steps per predicted action (the demonstrations were recorded every 4 steps).",
)
parser.add_argument("--integration-steps", type=int, default=16)
parser.add_argument("--solver", choices=("euler", "heun"), default="heun")
parser.add_argument(
    "--noise-mode",
    choices=("zero", "fixed_gaussian", "resampled_gaussian"),
    default="zero",
    help=(
        "Initial Flow latent. 'zero' is deterministic mean-latent inference for this "
        "unimodal expert; the Gaussian modes retain generative sampling."
    ),
)
parser.add_argument("--action-smoothing", "--action_smoothing", type=float, default=0.35)
parser.add_argument(
    "--gripper-width-offset-m",
    type=float,
    default=0.0,
    help="Optional calibration offset applied to the decoded absolute gripper width.",
)
parser.add_argument(
    "--force-feedback-target-n",
    type=float,
    default=0.0,
    help="Fixed observed mean fingertip-force target; <=0 disables tactile projection.",
)
parser.add_argument("--force-feedback-gain-m-per-n", type=float, default=2.0e-4)
parser.add_argument("--force-feedback-max-step-m", type=float, default=3.0e-4)
parser.add_argument("--force-feedback-start-step", type=int, default=350)
parser.add_argument(
    "--gripper-width-limits-m",
    type=float,
    nargs=2,
    default=(0.0200, 0.0400),
)
parser.add_argument(
    "--normalized-observation-clip",
    type=float,
    default=6.0,
    help="Clamp normalized observations to limit contact-sensor outliers; <=0 disables.",
)
parser.add_argument("--success-hold-steps", type=int, default=5)
parser.add_argument("--force-violation-dwell-steps", type=int, default=5)
parser.add_argument("--position-range", type=float, nargs=2, default=(0.010, 0.010))
parser.add_argument("--alignment-tolerance", type=float, default=0.012)
parser.add_argument("--minimum-force-range", type=float, nargs=2, default=(4.7, 6.1))
parser.add_argument("--maximum-force-range", type=float, nargs=2, default=(9.0, 11.0))
parser.add_argument(
    "--policy-device",
    choices=("auto", "cpu", "cuda"),
    default="auto",
    help="Flow network device; simulation can remain on CPU PhysX.",
)
parser.add_argument(
    "--report",
    type=Path,
    default=Path("exp_report/bottle/evaluation/flow_6d_force/seed_80000.json"),
)
parser.add_argument("--trace-output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch

from tacex_tasks.seal_twist.flow_policy import (
    FLOW_ACTION_REPRESENTATION,
    FLOW_OBSERVATION_FIELDS,
    PREVIOUS_GRIPPER_WIDTH_OBSERVATION_INDEX,
    WRENCH_OBSERVATION_SLICE,
    environment_flow_observation,
    load_flow_checkpoint,
)
from tacex_tasks.seal_twist.seal_twist_env import SealTwistEnv
from tacex_tasks.seal_twist.seal_twist_env_cfg import SealTwistEnvCfg


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(flags: dict[str, torch.Tensor], index: int) -> str:
    """Apply the same mutually exclusive failure priority as the MLP baseline."""

    if bool(flags["success"][index].item()):
        return "success"
    if bool(flags["over_force"][index].item()) or bool(flags["over_torqued"][index].item()):
        return "over_force"
    if bool(flags["alignment_failure"][index].item()) or bool(flags["slipped"][index].item()):
        return "alignment_failure"
    if bool(flags["under_force"][index].item()) or bool(flags["under_torqued"][index].item()):
        return "under_force"
    if not bool(flags["aligned"][index].item()) or not bool(
        flags["bilateral_contact"][index].item()
    ):
        return "alignment_failure"
    return "under_force"


def main() -> None:
    if args.episodes < 1 or args.num_envs < 1:
        parser.error("episodes and num-envs must be positive")
    if args.chunk_execute_steps < 1 or args.integration_steps < 1 or args.action_repeat < 1:
        parser.error("chunk-execute-steps, integration-steps, and action-repeat must be positive")
    if not 0.0 <= args.action_smoothing <= 1.0:
        parser.error("action-smoothing must lie in [0, 1]")
    if args.force_feedback_gain_m_per_n < 0.0 or args.force_feedback_max_step_m < 0.0:
        parser.error("force-feedback gain and max-step must be non-negative")
    if args.force_feedback_start_step < 0:
        parser.error("force-feedback-start-step must be non-negative")
    if not 0.0 <= args.gripper_width_limits_m[0] <= args.gripper_width_limits_m[1] <= 0.04:
        parser.error("gripper-width-limits-m must be ordered inside [0, 0.04]")

    checkpoint_path = args.checkpoint.resolve()
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
    environment_device = torch.device(cfg.sim.device)
    policy_device = torch.device(
        "cuda"
        if args.policy_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.policy_device == "auto"
        else args.policy_device
    )
    loaded = load_flow_checkpoint(checkpoint_path, device=policy_device)
    if args.chunk_execute_steps > loaded.model.config.horizon:
        raise ValueError(
            f"chunk-execute-steps={args.chunk_execute_steps} exceeds checkpoint horizon="
            f"{loaded.model.config.horizon}"
        )

    outcomes: list[dict[str, object]] = []
    failure_counts: Counter[str] = Counter()
    trace: list[dict[str, object]] = []
    batch = 0
    try:
        while len(outcomes) < args.episodes and simulation_app.is_running():
            batch_seed = args.seed + batch * args.num_envs
            env.reset(seed=batch_seed)
            active = torch.ones(args.num_envs, dtype=torch.bool, device=environment_device)
            previous_action: torch.Tensor | None = None
            batch_records: dict[int, dict[str, object]] = {}
            bilateral_contact_steps = torch.zeros(
                args.num_envs, dtype=torch.long, device=environment_device
            )
            bilateral_contact_force_sum_n = torch.zeros(
                args.num_envs, dtype=torch.float64, device=environment_device
            )
            policy_generator = torch.Generator(device=policy_device).manual_seed(
                batch_seed + args.policy_seed_offset
            )
            step = 0
            while step < args.max_episode_steps and bool(active.any().item()):
                with torch.inference_mode():
                    if args.noise_mode == "fixed_gaussian":
                        policy_generator.manual_seed(batch_seed + args.policy_seed_offset)
                    observation = environment_flow_observation(
                        env, max_episode_steps=loaded.max_episode_steps
                    ).to(policy_device)
                    normalized_observation = loaded.normalize_observation(observation)
                    if args.normalized_observation_clip > 0.0:
                        normalized_observation = normalized_observation.clamp(
                            -args.normalized_observation_clip,
                            args.normalized_observation_clip,
                        )
                    noise_shape = (
                        args.num_envs,
                        loaded.model.config.horizon,
                        loaded.model.config.action_dim,
                    )
                    if args.noise_mode == "zero":
                        initial_noise = torch.zeros(noise_shape, device=policy_device)
                    else:
                        initial_noise = torch.randn(
                            noise_shape,
                            generator=policy_generator,
                            device=policy_device,
                        )
                    normalized_chunk = loaded.model.decode(
                        normalized_observation,
                        initial_noise,
                        integration_steps=args.integration_steps,
                        solver=args.solver,
                    )
                    action_chunk = loaded.denormalize_action(normalized_chunk)
                    action_chunk[..., :3] += observation[:, None, :3]
                    action_chunk[..., 9] += observation[
                        :, None, PREVIOUS_GRIPPER_WIDTH_OBSERVATION_INDEX
                    ]
                    if args.force_feedback_target_n > 0.0:
                        finger_force_n = observation[:, -3:-1]
                        mean_finger_force_n = finger_force_n.mean(dim=-1)
                        contact = finger_force_n.max(dim=-1).values >= 0.05
                        feedback_active = (
                            observation[:, -1] * float(loaded.max_episode_steps)
                            >= float(args.force_feedback_start_step)
                        ) & contact
                        width_correction_m = (
                            args.force_feedback_gain_m_per_n
                            * (mean_finger_force_n - args.force_feedback_target_n)
                        ).clamp(
                            -args.force_feedback_max_step_m,
                            args.force_feedback_max_step_m,
                        )
                        width_correction_m = torch.where(
                            feedback_active,
                            width_correction_m,
                            torch.zeros_like(width_correction_m),
                        )
                        action_chunk[..., 9] += width_correction_m[:, None]
                        action_chunk[..., 9].clamp_(
                            args.gripper_width_limits_m[0],
                            args.gripper_width_limits_m[1],
                        )
                    action_chunk = action_chunk.to(environment_device)
                    action_chunk[..., 9] += args.gripper_width_offset_m
                    if args.trace_output is not None and batch == 0:
                        trace.append(
                            {
                                "step": step,
                                "observation": observation[0].detach().cpu().tolist(),
                                "normalized_observation": normalized_observation[0]
                                .detach()
                                .cpu()
                                .tolist(),
                                "decoded_action_0": action_chunk[0, 0]
                                .detach()
                                .cpu()
                                .tolist(),
                            }
                        )

                for chunk_index in range(args.chunk_execute_steps):
                    target_action = action_chunk[:, chunk_index]
                    for _ in range(args.action_repeat):
                        if previous_action is None:
                            action = target_action
                        else:
                            alpha = float(args.action_smoothing)
                            action = alpha * target_action + (1.0 - alpha) * previous_action
                        previous_action = action
                        env._pre_physics_step(action)
                        env._apply_action()
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(dt=env.physics_dt)
                        step += 1

                        flags = env.get_task_flags()
                        bilateral_active = active & flags["bilateral_contact"]
                        bilateral_mean_force_n = 0.5 * (
                            flags["left_finger_force_n"]
                            + flags["right_finger_force_n"]
                        )
                        bilateral_contact_steps += bilateral_active.to(torch.long)
                        bilateral_contact_force_sum_n += torch.where(
                            bilateral_active,
                            bilateral_mean_force_n.to(torch.float64),
                            torch.zeros_like(bilateral_contact_force_sum_n),
                        )
                        terminal = (
                            flags["success"]
                            | flags["over_force"]
                            | flags["over_torqued"]
                            | flags["under_force"]
                            | flags["alignment_failure"]
                            | flags["slipped"]
                        )
                        if step >= args.max_episode_steps:
                            terminal |= active
                        just_finished = active & terminal
                        finished_indices = (
                            torch.nonzero(just_finished, as_tuple=False)
                            .squeeze(-1)
                            .cpu()
                            .tolist()
                        )
                        for index in finished_indices:
                            outcome = classify(flags, index)
                            batch_records[index] = {
                                "seed": batch_seed + index,
                                "outcome": outcome,
                                "steps": step,
                                "final_torque_nm": float(env.seal_torque_nm[index].item()),
                                "peak_torque_nm": float(env.peak_torque_nm[index].item()),
                                "final_minimum_finger_force_n": float(
                                    flags["minimum_finger_force_n"][index].item()
                                ),
                                "final_maximum_finger_force_n": float(
                                    flags["maximum_finger_force_n"][index].item()
                                ),
                                "peak_left_finger_force_n": float(
                                    env.peak_left_force_n[index].item()
                                ),
                                "peak_right_finger_force_n": float(
                                    env.peak_right_force_n[index].item()
                                ),
                                "minimum_force_threshold_n": float(
                                    env.minimum_force_threshold_n[index].item()
                                ),
                                "maximum_force_threshold_n": float(
                                    env.maximum_force_threshold_n[index].item()
                                ),
                                "alignment_error_m": float(
                                    flags["alignment_error_m"][index].item()
                                ),
                                "alignment_tolerance_m": float(cfg.alignment_tolerance_m),
                                "cap_position_offset_m": env.sampled_cap_position_offset_m[index]
                                .detach()
                                .cpu()
                                .tolist(),
                                "bilateral_contact": bool(
                                    flags["bilateral_contact"][index].item()
                                ),
                                "bilateral_contact_steps": int(
                                    bilateral_contact_steps[index].item()
                                ),
                                "mean_bilateral_finger_force_n": float(
                                    bilateral_contact_force_sum_n[index].item()
                                    / max(
                                        int(bilateral_contact_steps[index].item()),
                                        1,
                                    )
                                ),
                            }
                        active &= ~just_finished
                        if step >= args.max_episode_steps or not bool(active.any().item()):
                            break
                    if step >= args.max_episode_steps or not bool(active.any().item()):
                        break

            for index in sorted(batch_records):
                if len(outcomes) >= args.episodes:
                    break
                record = batch_records[index]
                outcomes.append(record)
                if record["outcome"] != "success":
                    failure_counts[str(record["outcome"])] += 1
            successes = sum(record["outcome"] == "success" for record in outcomes)
            print(
                f"[FLOW EVAL] evaluated={len(outcomes)}/{args.episodes} "
                f"successes={successes} rate={successes / max(len(outcomes), 1):.2%} "
                f"failures={dict(failure_counts)}",
                flush=True,
            )
            batch += 1
    finally:
        env.close()

    successes = sum(record["outcome"] == "success" for record in outcomes)
    failures = len(outcomes) - successes
    total_bilateral_contact_steps = sum(
        int(record["bilateral_contact_steps"]) for record in outcomes
    )
    contact_phase_force_sum_n = sum(
        float(record["mean_bilateral_finger_force_n"])
        * int(record["bilateral_contact_steps"])
        for record in outcomes
    )
    lower, upper = wilson_interval(successes, len(outcomes))
    report = {
        "task": "TacEx-SealTwist-Torque-v0",
        "policy": "seal_twist_6d_wrench_conditional_flow_matching",
        "observation_fields": list(FLOW_OBSERVATION_FIELDS),
        "wrench_observation_indices": [
            WRENCH_OBSERVATION_SLICE.start,
            WRENCH_OBSERVATION_SLICE.stop,
        ],
        "six_dimensional_force_source": "environment.get_task_wrench(): Fx,Fy,Fz,Tx,Ty,Tz",
        "gripper_state_alignment": "current command from the preceding policy action",
        "action_representation": FLOW_ACTION_REPRESENTATION,
        "force_thresholds_hidden_from_policy": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "seed_start": args.seed,
        "policy_seed_offset": args.policy_seed_offset,
        "episodes": len(outcomes),
        "successes": successes,
        "success_rate": successes / max(len(outcomes), 1),
        "wilson_95_ci": [lower, upper],
        "failures": failures,
        "failure_counts": dict(failure_counts),
        "failure_fraction_among_failures": {
            name: failure_counts[name] / max(failures, 1)
            for name in ("under_force", "over_force", "alignment_failure")
        },
        "bilateral_contact_steps": total_bilateral_contact_steps,
        "contact_phase_mean_force_n": (
            contact_phase_force_sum_n / max(total_bilateral_contact_steps, 1)
        ),
        "contact_phase_force_definition": (
            "contact-step-weighted mean of (left_force_n + right_force_n) / 2 "
            "over physics steps with bilateral contact"
        ),
        "minimum_force_randomization_range_n": list(args.minimum_force_range),
        "maximum_force_randomization_range_n": list(args.maximum_force_range),
        "position_randomization_half_range_m": list(args.position_range),
        "alignment_tolerance_m": args.alignment_tolerance,
        "torque_window_nm": list(cfg.torque_window_nm),
        "evaluation_horizon_steps": args.max_episode_steps,
        "policy_time_horizon_steps": loaded.max_episode_steps,
        "success_hold_steps": cfg.success_hold_steps,
        "force_violation_dwell_steps": cfg.force_violation_dwell_steps,
        "action_horizon": loaded.model.config.horizon,
        "chunk_execute_steps": args.chunk_execute_steps,
        "action_repeat": args.action_repeat,
        "integration_steps": args.integration_steps,
        "ode_solver": args.solver,
        "action_smoothing": args.action_smoothing,
        "gripper_width_offset_m": args.gripper_width_offset_m,
        "force_feedback_target_n": args.force_feedback_target_n,
        "force_feedback_gain_m_per_n": args.force_feedback_gain_m_per_n,
        "force_feedback_max_step_m": args.force_feedback_max_step_m,
        "force_feedback_start_step": args.force_feedback_start_step,
        "gripper_width_limits_m": list(args.gripper_width_limits_m),
        "normalized_observation_clip": args.normalized_observation_clip,
        "noise_mode": args.noise_mode,
        "native_flow_noise": {
            "zero": "deterministic zero (mean) latent",
            "fixed_gaussian": "one standard Gaussian latent reused within each episode",
            "resampled_gaussian": "independent standard Gaussian per predicted action chunk",
        }[args.noise_mode],
        "simulation_device": str(environment_device),
        "policy_device": str(policy_device),
        "outcomes": outcomes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(
        "[SUMMARY] "
        + json.dumps({key: value for key, value in report.items() if key != "outcomes"}),
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
