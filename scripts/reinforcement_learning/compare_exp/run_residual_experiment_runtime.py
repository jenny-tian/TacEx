"""Train or independently evaluate the clean residual-RL baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", choices=("online_training", "evaluation"), required=True)
parser.add_argument("--bc-policy", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-episodes", type=int, default=50)
parser.add_argument("--training-interactions", type=int, default=0)
parser.add_argument("--seed", type=int, default=4200)
parser.add_argument("--break-force-threshold-n", type=float, default=4.5)
parser.add_argument("--labware-random-xy-m", type=float, nargs=2, default=(0.10, 0.10))
parser.add_argument("--labware-random-yaw-deg", type=float, default=0.0)
parser.add_argument("--residual-scale", type=float, default=0.15)
parser.add_argument("--flow-num-inference-steps", type=int, default=20)
parser.add_argument("--phase-horizon-steps", type=int, default=383)
parser.add_argument("--camera-warmup-steps", type=int, default=8)
parser.add_argument("--gpu-max-rigid-contact-count", type=int, default=0)
parser.add_argument("--gpu-max-rigid-patch-count", type=int, default=0)
parser.add_argument("--learning-starts", type=int, default=32)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--actor-lr", type=float, default=3.0e-5)
parser.add_argument("--critic-lr", type=float, default=1.0e-4)
parser.add_argument("--max-outer-interactions", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.num_episodes < 1:
    parser.error("--num-episodes must be positive.")
if args_cli.training_interactions < 0:
    parser.error("--training-interactions cannot be negative.")
if min(args_cli.gpu_max_rigid_contact_count, args_cli.gpu_max_rigid_patch_count) < 0:
    parser.error("PhysX GPU buffer capacities cannot be negative.")
if args_cli.phase == "evaluation" and args_cli.training_interactions:
    parser.error("--training-interactions is only valid during online training.")
if args_cli.phase == "evaluation" and args_cli.checkpoint is None:
    parser.error("Residual-RL evaluation requires --checkpoint.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0], *hydra_args]


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import skrl  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from skrl.agents.torch.sac import SAC_CFG  # noqa: E402
from skrl.memories.torch import RandomMemory  # noqa: E402
from skrl.utils import set_seed  # noqa: E402

import tacex_tasks  # noqa: E402,F401
import tacex_tasks.lab_pick  # noqa: E402,F401
from clean_alpha_zero_sac import CleanAlphaZeroSAC  # noqa: E402
from clean_residual_sac import (  # noqa: E402
    CLEAN_RESIDUAL_CONTRACT_VERSION,
    CleanResidualActor,
    build_clean_residual_sac_models,
    validate_tactile_residual_policy_state,
)
from clean_residual_wrapper import CleanResidualLabPickWrapper  # noqa: E402
from episode_trainer import EpisodeLimitedSequentialTrainer  # noqa: E402
from recording import OnlineEpisodeRecorder  # noqa: E402
from tactile_observation import tactile_contract_metadata  # noqa: E402


TASK = "TacEx-LabPick-Slide-Clean-Residual-SAC-v0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _agent_cfg(raw: dict[str, Any], output_dir: Path) -> SAC_CFG:
    cfg = copy.deepcopy(raw)
    cfg["learning_starts"] = args_cli.learning_starts
    cfg["batch_size"] = args_cli.batch_size
    cfg["learning_rate"] = [args_cli.actor_lr, args_cli.critic_lr, 0.0]
    cfg["experiment"]["directory"] = str(output_dir / "training")
    cfg["experiment"]["experiment_name"] = "skrl"
    cfg["experiment"]["write_interval"] = 25
    cfg["experiment"]["checkpoint_interval"] = 2500
    return SAC_CFG(**cfg)


def _train(env: OnlineEpisodeRecorder, agent_cfg: dict[str, Any], output_dir: Path):
    vec_env = SkrlVecEnvWrapper(env, ml_framework="torch")
    memory = RandomMemory(
        memory_size=int(agent_cfg["memory_size"]),
        num_envs=vec_env.num_envs,
        device=vec_env.device,
    )
    models = build_clean_residual_sac_models(
        vec_env.observation_space,
        vec_env.state_space,
        vec_env.action_space,
        vec_env.device,
        layout=env.env.layout,
        initial_log_std=float(agent_cfg.get("network", {}).get("initial_log_std", -3.0)),
    )
    cfg = _agent_cfg(agent_cfg["agent"], output_dir)
    agent = CleanAlphaZeroSAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=vec_env.observation_space,
        state_space=vec_env.state_space,
        action_space=vec_env.action_space,
        device=vec_env.device,
    )
    interaction_limited = args_cli.training_interactions > 0
    maximum = (
        args_cli.training_interactions
        if interaction_limited
        else args_cli.max_outer_interactions or args_cli.num_episodes * 600
    )
    trainer = EpisodeLimitedSequentialTrainer(
        episode_env=env,
        num_episodes=None if interaction_limited else args_cli.num_episodes,
        max_interactions=maximum,
        stop_at_interaction_budget=interaction_limited,
        cfg={
            "timesteps": maximum,
            "headless": True,
            "environment_info": agent_cfg["trainer"].get("environment_info", "log"),
            "close_environment_at_exit": False,
            "disable_progressbar": True,
        },
        env=vec_env,
        agents=agent,
    )
    trainer.train()
    checkpoint = output_dir / "residual_final.pt"
    agent.save(str(checkpoint))
    return trainer.interactions_completed, trainer.gradient_updates_completed, checkpoint


def _evaluate(env: OnlineEpisodeRecorder, agent_cfg: dict[str, Any]):
    assert args_cli.checkpoint is not None
    checkpoint = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    inner = env.env
    device = torch.device(inner.device)
    actor = CleanResidualActor(
        inner.observation_space,
        inner.action_space,
        device,
        layout=inner.layout,
        initial_log_std=float(agent_cfg.get("network", {}).get("initial_log_std", -3.0)),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if "policy" not in payload:
        raise KeyError(f"Checkpoint has no policy state: {checkpoint}")
    validate_tactile_residual_policy_state(payload["policy"])
    actor.load_state_dict(payload["policy"], strict=True)
    actor.eval()
    maximum = args_cli.max_outer_interactions or 600
    interactions = 0
    for episode in range(args_cli.num_episodes):
        seed = args_cli.seed + episode
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        observation, _ = env.reset(seed=seed, flow_noise_seed=seed)
        for _ in range(maximum):
            with torch.inference_mode():
                residual = actor.deterministic_action(
                    {"observations": observation["policy"].to(device)}
                )
            observation, _, terminated, truncated, _ = env.step(residual)
            interactions += 1
            if bool(torch.as_tensor(terminated | truncated).any().item()):
                result = env.complete_pending_episode(dsrl_updates_completed=0)
                print(
                    "[EVALUATION] "
                    f"mode=residual_rl episode={episode + 1}/{args_cli.num_episodes} "
                    f"seed={seed} success={result['success']} "
                    f"reason={result['terminal_reason']}",
                    flush=True,
                )
                break
        else:
            raise RuntimeError(f"Residual evaluation seed {seed} exceeded its step limit.")
    return interactions, 0, checkpoint


@hydra_task_config(TASK, "skrl_clean_sac_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict[str, Any],
) -> None:
    checkpoint = args_cli.bc_policy.expanduser().resolve()
    output_dir = args_cli.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    set_seed(args_cli.seed)

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    # This established residual baseline uses reset-time simulator yaw and is
    # therefore an intentionally strong (privileged) comparator.
    env_cfg.rl_align_cafe_action_yaw = True
    env_cfg.rl_action_penalty_scale = 0.0
    env_cfg.labware_pos_randomization_xy = tuple(args_cli.labware_random_xy_m)
    env_cfg.labware_yaw_randomization = math.radians(args_cli.labware_random_yaw_deg)
    env_cfg.terminate_break_force_threshold_n = args_cli.break_force_threshold_n
    if args_cli.gpu_max_rigid_contact_count:
        env_cfg.sim.physx.gpu_max_rigid_contact_count = (
            args_cli.gpu_max_rigid_contact_count
        )
    if args_cli.gpu_max_rigid_patch_count:
        env_cfg.sim.physx.gpu_max_rigid_patch_count = args_cli.gpu_max_rigid_patch_count

    physical = gym.make(TASK, cfg=env_cfg)
    residual = CleanResidualLabPickWrapper(
        physical,
        checkpoint,
        device=args_cli.device or "cuda:0",
        residual_scale=args_cli.residual_scale,
        flow_num_inference_steps=args_cli.flow_num_inference_steps,
        phase_horizon_steps=args_cli.phase_horizon_steps,
        camera_warmup_steps=args_cli.camera_warmup_steps,
        seed=args_cli.seed,
    )
    env = OnlineEpisodeRecorder(
        residual,
        output_dir=output_dir,
        mode="residual_rl_tactile",
        experiment_seed=args_cli.seed,
    )
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "residual_rl_tactile",
        "phase": args_cli.phase,
        "algorithm": "alpha_zero_sac",
        "num_episodes": args_cli.num_episodes,
        "training_interaction_budget": args_cli.training_interactions or None,
        "seed": args_cli.seed,
        "task": TASK,
        "bc_policy": str(checkpoint),
        "bc_checkpoint_sha256": _sha256(checkpoint),
        "input_checkpoint": None if args_cli.checkpoint is None else str(args_cli.checkpoint),
        "break_force_threshold_n": args_cli.break_force_threshold_n,
        "labware_random_xy_m": list(args_cli.labware_random_xy_m),
        "labware_random_yaw_deg": args_cli.labware_random_yaw_deg,
        "residual_scale": args_cli.residual_scale,
        "residual_indices": [0, 1, 2, 9],
        "residual_contract_version": CLEAN_RESIDUAL_CONTRACT_VERSION,
        "actor_observation_dim": residual.layout.policy_dim,
        "critic_state_dim": residual.layout.state_dim,
        "critic_input_dim": residual.layout.critic_input_dim,
        "tactile_actor": tactile_contract_metadata(),
        "gpu_max_rigid_contact_count": env_cfg.sim.physx.gpu_max_rigid_contact_count,
        "gpu_max_rigid_patch_count": env_cfg.sim.physx.gpu_max_rigid_patch_count,
        "oracle_yaw": True,
        "learning_starts": args_cli.learning_starts,
        "batch_size": args_cli.batch_size,
        "learning_rates": [args_cli.actor_lr, args_cli.critic_lr],
        "skrl_version": skrl.__version__,
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    dump_yaml(str(output_dir / "resolved_env.yaml"), env_cfg)
    dump_pickle(str(output_dir / "resolved_env.pkl"), env_cfg)
    _write_json(output_dir / "resolved_agent.json", agent_cfg)
    learned_checkpoint = None
    try:
        if args_cli.phase == "online_training":
            interactions, updates, learned_checkpoint = _train(env, agent_cfg, output_dir)
        else:
            interactions, updates, learned_checkpoint = _evaluate(env, agent_cfg)
    finally:
        env.close()

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
        "gradient_updates": updates,
        "learned_checkpoint": None if learned_checkpoint is None else str(learned_checkpoint),
        "trajectory_count": len(list((output_dir / "trajectories").glob("*.jsonl.gz"))),
        "results": env.results,
    }
    _write_json(output_dir / "results.json", summary)
    print(
        f"[SUMMARY] mode=residual_rl_tactile phase={args_cli.phase} "
        f"success={successes}/{env.completed_episodes} "
        f"({summary['success_rate']:.1%}) output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
