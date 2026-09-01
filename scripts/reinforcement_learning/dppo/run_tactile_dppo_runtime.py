"""Train tactile DPPO for an exact number of complete LabPick episodes."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bc-policy", type=Path, required=True)
parser.add_argument("--bc-validation-metrics", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-episodes", type=int, default=100)
parser.add_argument("--seed", type=int, default=4200)
parser.add_argument("--break-force-threshold-n", type=float, default=4.5)
parser.add_argument("--labware-random-xy-m", type=float, nargs=2, default=(0.10, 0.10))
parser.add_argument("--labware-random-yaw-deg", type=float, default=0.0)
parser.add_argument("--camera-warmup-steps", type=int, default=8)
parser.add_argument("--chunk-discount", type=float, default=0.99)
parser.add_argument("--gpu-max-rigid-contact-count", type=int, default=2**20)
parser.add_argument("--gpu-max-rigid-patch-count", type=int, default=2**20)
parser.add_argument("--rollout-steps", type=int, default=32)
parser.add_argument("--learning-epochs", type=int, default=8)
parser.add_argument("--mini-batches", type=int, default=4)
parser.add_argument("--learning-rate", type=float, default=3.0e-5)
parser.add_argument("--ratio-clip", type=float, default=0.2)
parser.add_argument("--value-clip", type=float, default=0.2)
parser.add_argument("--grad-clip", type=float, default=0.5)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--fine-tune-denoising-steps", type=int, default=5)
parser.add_argument("--num-inference-steps", type=int, default=10)
parser.add_argument("--min-sampling-denoising-std", type=float, default=1.0e-3)
parser.add_argument("--min-logprob-denoising-std", type=float, default=0.1)
parser.add_argument("--checkpoint-interval", type=int, default=250)
parser.add_argument("--max-outer-interactions", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.num_episodes < 1:
    parser.error("--num-episodes must be positive.")
if args_cli.rollout_steps < 1 or args_cli.learning_epochs < 1 or args_cli.mini_batches < 1:
    parser.error("DPPO rollout and update counts must be positive.")
if args_cli.rollout_steps % args_cli.mini_batches:
    parser.error("--rollout-steps must be divisible by --mini-batches.")
if args_cli.learning_rate <= 0.0 or args_cli.grad_clip <= 0.0:
    parser.error("DPPO learning rate and gradient clip must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0], *hydra_args]


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import tacex_tasks  # noqa: E402,F401
import tacex_tasks.lab_pick  # noqa: E402,F401
from recording import OnlineEpisodeRecorder  # noqa: E402
from tactile_dppo import (  # noqa: E402
    TACTILE_DPPO_CONTRACT_VERSION,
    DPPORollout,
)
from tactile_dppo_wrapper import (  # noqa: E402
    DIFFUSION_BC_CAMERA_CONTRACT,
    TactileDPPOLabPickWrapper,
)
from tactile_observation import tactile_contract_metadata  # noqa: E402


TASK = "TacEx-LabPick-Slide-Clean-DSRL-SAC-v0"
DPPO_PAPER = "Ren et al., Diffusion Policy Policy Optimization, arXiv:2409.00588"
DPPO_REFERENCE_REVISION = "fa5847a9853aca9e8d5aaa3e2836e025ed8cbf97"
DPPO_REFERENCE_REMOTE = "git@github.com:ajwagen/dppo-dsrl.git"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _save_checkpoint(path: Path, wrapper: TactileDPPOLabPickWrapper) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(wrapper.agent.checkpoint_payload(), temporary)
    temporary.replace(path)


def _parameter_signature(module: torch.nn.Module) -> dict[str, float | int]:
    with torch.no_grad():
        parameters = [parameter.detach().double() for parameter in module.parameters()]
        return {
            "count": sum(parameter.numel() for parameter in parameters),
            "sum": sum(float(parameter.sum().cpu()) for parameter in parameters),
            "squared_sum": sum(float(parameter.square().sum().cpu()) for parameter in parameters),
        }


@hydra_task_config(TASK, "skrl_clean_dsrl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict[str, Any],
) -> None:
    print("[STARTUP] entered tactile DPPO main", flush=True)
    del agent_cfg
    checkpoint = args_cli.bc_policy.expanduser().resolve()
    validation_metrics = args_cli.bc_validation_metrics.expanduser().resolve()
    output_dir = args_cli.output_dir.expanduser().resolve()
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not validation_metrics.is_file():
        raise FileNotFoundError(validation_metrics)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[STARTUP] output directory prepared; seeding Python/NumPy/Torch", flush=True)
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    print("[STARTUP] seeding complete; applying environment configuration", flush=True)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env_cfg.rl_align_cafe_action_yaw = False
    env_cfg.rl_action_penalty_scale = 0.0
    env_cfg.labware_pos_randomization_xy = tuple(args_cli.labware_random_xy_m)
    env_cfg.labware_yaw_randomization = math.radians(args_cli.labware_random_yaw_deg)
    env_cfg.terminate_break_force_threshold_n = args_cli.break_force_threshold_n
    env_cfg.sim.physx.gpu_max_rigid_contact_count = args_cli.gpu_max_rigid_contact_count
    env_cfg.sim.physx.gpu_max_rigid_patch_count = args_cli.gpu_max_rigid_patch_count

    print("[STARTUP] creating physical environment", flush=True)
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        physical = gym.make(TASK, cfg=env_cfg)
    finally:
        faulthandler.cancel_dump_traceback_later()
    print("[STARTUP] physical environment ready; loading DPPO wrapper", flush=True)
    wrapper = TactileDPPOLabPickWrapper(
        physical,
        checkpoint,
        device=args_cli.device or "cuda:0",
        camera_warmup_steps=args_cli.camera_warmup_steps,
        chunk_discount=args_cli.chunk_discount,
        dppo_kwargs={
            "fine_tune_denoising_steps": args_cli.fine_tune_denoising_steps,
            "num_inference_steps": args_cli.num_inference_steps,
            "min_sampling_denoising_std": args_cli.min_sampling_denoising_std,
            "min_logprob_denoising_std": args_cli.min_logprob_denoising_std,
            "learning_rate": args_cli.learning_rate,
            "clip_range": args_cli.ratio_clip,
            "value_clip": args_cli.value_clip,
            "grad_clip": args_cli.grad_clip,
            "gae_lambda": args_cli.gae_lambda,
            "update_epochs": args_cli.learning_epochs,
            "minibatches": args_cli.mini_batches,
        },
    )
    print("[STARTUP] DPPO wrapper ready; creating episode recorder", flush=True)
    env = OnlineEpisodeRecorder(
        wrapper,
        output_dir=output_dir,
        mode="dppo_tactile",
        experiment_seed=args_cli.seed,
    )
    print("[STARTUP] episode recorder ready; writing resolved metadata", flush=True)
    resolved_dppo = {
        "contract_version": TACTILE_DPPO_CONTRACT_VERSION,
        "paper": DPPO_PAPER,
        "reference_code_revision": DPPO_REFERENCE_REVISION,
        "reference_code_remote": DPPO_REFERENCE_REMOTE,
        "backbone": "LeRobot DiffusionPolicy DDPM",
        "objective": "clipped_PPO_over_reverse_diffusion_transition_likelihoods",
        "single_initial_latent_surrogate": False,
        "rollout_environment_transitions": args_cli.rollout_steps,
        "learning_epochs": args_cli.learning_epochs,
        "mini_batches": args_cli.mini_batches,
        "learning_rate": args_cli.learning_rate,
        "ratio_clip": args_cli.ratio_clip,
        "value_clip": args_cli.value_clip,
        "gradient_clip": args_cli.grad_clip,
        "gae_lambda": args_cli.gae_lambda,
        "fine_tune_denoising_steps": args_cli.fine_tune_denoising_steps,
        "total_denoising_steps": wrapper.agent.num_inference_steps,
        "minimum_sampling_denoising_std": args_cli.min_sampling_denoising_std,
        "minimum_logprob_denoising_std": args_cli.min_logprob_denoising_std,
        "tactile_adapter_initialization": "exact_zero_linear_no_bias",
        "visual_encoder_frozen": True,
        "base_unet_frozen": True,
        "fine_tuned_module": "deep_copy_of_pretrained_unet_last_denoising_steps",
        "actor_trainable_parameters": sum(
            parameter.numel() for parameter in wrapper.agent.actor_ft.parameters()
        )
        + sum(parameter.numel() for parameter in wrapper.agent.tactile_adapter.parameters()),
        "value_trainable_parameters": sum(
            parameter.numel() for parameter in wrapper.agent.value.parameters()
        ),
        "initial_actor_parameter_signature": _parameter_signature(wrapper.agent.actor_ft),
    }
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "dppo_tactile",
        "phase": "online_training",
        "num_episodes": args_cli.num_episodes,
        "seed": args_cli.seed,
        "task": TASK,
        "bc_policy": str(checkpoint),
        "bc_model_sha256": _sha256(model_path),
        "bc_checkpoint_tree_sha256": _tree_sha256(checkpoint),
        "bc_validation_metrics": str(validation_metrics),
        "bc_validation_metrics_sha256": _sha256(validation_metrics),
        "bc_camera_contract": DIFFUSION_BC_CAMERA_CONTRACT,
        "break_force_threshold_n": args_cli.break_force_threshold_n,
        "labware_random_xy_m": list(args_cli.labware_random_xy_m),
        "labware_random_yaw_deg": args_cli.labware_random_yaw_deg,
        "action_repeat": wrapper.action_repeat,
        "action_horizon": int(wrapper.policy.config.horizon),
        "executed_action_steps": int(wrapper.policy.config.n_action_steps),
        "chunk_discount": args_cli.chunk_discount,
        "gpu_max_rigid_contact_count": env_cfg.sim.physx.gpu_max_rigid_contact_count,
        "gpu_max_rigid_patch_count": env_cfg.sim.physx.gpu_max_rigid_patch_count,
        "tactile_actor": tactile_contract_metadata(),
        "dppo": resolved_dppo,
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    _write_json(output_dir / "resolved_dppo.json", resolved_dppo)
    dump_yaml(str(output_dir / "resolved_env.yaml"), env_cfg)
    dump_pickle(str(output_dir / "resolved_env.pkl"), env_cfg)

    rollout = DPPORollout(capacity=args_cli.rollout_steps)
    diagnostics_path = output_dir / "optimizer_diagnostics.jsonl"
    interactions = 0
    total_denoising_steps = 0
    total_physics_steps = 0
    maximum = args_cli.max_outer_interactions or args_cli.num_episodes * 20
    generator = torch.Generator(device=wrapper.agent.device)
    generator.manual_seed(args_cli.seed)
    observation, _ = env.reset(seed=args_cli.seed)
    started = time.perf_counter()
    try:
        while simulation_app.is_running() and env.completed_episodes < args_cli.num_episodes:
            sample = wrapper.agent.sample(
                observation["global_condition"],
                observation["tactile_actor"],
                observation["visual_xy"],
                generator=generator,
            )
            next_observation, reward, terminated, truncated, info = env.step(sample["action"])
            done = bool(torch.as_tensor(terminated | truncated).any().item())
            rollout.add(
                global_condition=observation["global_condition"],
                tactile=observation["tactile_actor"],
                chain_previous=sample["chain_previous"],
                chain_next=sample["chain_next"],
                timesteps=sample["timesteps"],
                old_log_probs=sample["old_log_probs"],
                value=sample["value"],
                reward=reward,
                done=done,
            )
            interactions += 1
            total_denoising_steps += int(sample["denoising_steps"].item())
            total_physics_steps += int(info["dppo/physics_steps_executed"])
            observation = next_observation

            if rollout.full:
                with torch.no_grad():
                    bootstrap = (
                        torch.zeros((), device=wrapper.agent.device)
                        if done
                        else wrapper.agent.predict_value(
                            observation["global_condition"], observation["tactile_actor"]
                        )[0]
                    )
                diagnostics = wrapper.agent.update(rollout, bootstrap)
                _append_jsonl(
                    diagnostics_path,
                    {
                        "ending_outer_interaction": interactions,
                        "completed_episodes": env.completed_episodes,
                        "cumulative_optimizer_steps": wrapper.agent.optimizer_steps,
                        **diagnostics.to_dict(),
                    },
                )

            if done:
                result = env.complete_pending_episode(
                    dsrl_updates_completed=wrapper.agent.optimizer_steps
                )
                print(
                    "[EPISODE] "
                    f"mode=dppo_tactile episode={result['episode_index'] + 1}/{args_cli.num_episodes} "
                    f"success={result['success']} reason={result['terminal_reason']} "
                    f"force_peak={result['peak_contact_force_n']:.3f}N "
                    f"optimizer_steps={wrapper.agent.optimizer_steps}",
                    flush=True,
                )
                if env.completed_episodes < args_cli.num_episodes:
                    env.begin_auto_reset_episode()

            if args_cli.checkpoint_interval and interactions % args_cli.checkpoint_interval == 0:
                _save_checkpoint(output_dir / f"dppo_interaction_{interactions:06d}.pt", wrapper)
            if interactions >= maximum and env.completed_episodes < args_cli.num_episodes:
                raise RuntimeError(
                    f"Reached {maximum} interactions before completing {args_cli.num_episodes} episodes."
                )
    finally:
        elapsed_seconds = time.perf_counter() - started
        env.close()

    final_checkpoint = output_dir / "dppo_final.pt"
    _save_checkpoint(final_checkpoint, wrapper)
    successes = sum(int(item["success"]) for item in env.results)
    failures = Counter(
        item["terminal_reason"] for item in env.results if not item["success"]
    )
    summary = {
        **metadata,
        "completed_episodes": env.completed_episodes,
        "successes": successes,
        "success_rate": successes / max(env.completed_episodes, 1),
        "failure_counts": dict(failures),
        "outer_interactions": interactions,
        "physics_steps": total_physics_steps,
        "denoising_steps": total_denoising_steps,
        "gradient_updates": wrapper.agent.optimizer_steps,
        "wall_time_seconds": elapsed_seconds,
        "learned_checkpoint": str(final_checkpoint),
        "learned_checkpoint_sha256": _sha256(final_checkpoint),
        "final_actor_parameter_signature": _parameter_signature(wrapper.agent.actor_ft),
        "final_tactile_adapter_l2_norm": float(
            wrapper.agent.tactile_adapter.weight.detach().float().norm().cpu()
        ),
        "trajectory_count": len(list((output_dir / "trajectories").glob("*.jsonl.gz"))),
        "results": env.results,
    }
    _write_json(output_dir / "results.json", summary)
    print(
        f"[SUMMARY] mode=dppo_tactile success={successes}/{env.completed_episodes} "
        f"({summary['success_rate']:.1%}) interactions={interactions} "
        f"optimizer_steps={wrapper.agent.optimizer_steps} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
