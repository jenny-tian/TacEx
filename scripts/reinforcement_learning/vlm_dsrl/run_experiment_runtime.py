"""Run an exact episode-budget online VLM/DSRL experiment in Isaac Lab."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--mode",
    choices=(
        "ours_tactile",
        "dsrl_tactile",
        "joint",
        "joint_bilateral",
        "guarded_joint",
        "dsrl",
        "vlm",
        "base",
        "flow_rwr",
        "flow_ppo",
    ),
    required=True,
)
parser.add_argument(
    "--phase",
    choices=("online_training", "evaluation"),
    default="online_training",
    help="Evaluation freezes learned weights and uses one explicit seed per episode.",
)
parser.add_argument(
    "--evaluation-policy",
    choices=("deterministic", "stochastic"),
    default="deterministic",
    help=(
        "How to act from a frozen SAC checkpoint. Stochastic evaluation samples "
        "the learned squashed-Gaussian policy after reseeding every episode."
    ),
)
parser.add_argument(
    "--dsrl-checkpoint",
    type=Path,
    default=None,
    help="Required for evaluation of a DSRL-bearing mode.",
)
parser.add_argument("--bc-policy", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-episodes", type=int, default=50)
parser.add_argument(
    "--training-interactions",
    type=int,
    default=0,
    help="If positive, online training stops after this many outer policy decisions instead of an episode budget.",
)
parser.add_argument("--seed", type=int, default=4200)
parser.add_argument("--break-force-threshold-n", type=float, required=True)
parser.add_argument(
    "--physical-force-range-n", type=float, nargs=2, default=(0.25, 3.25)
)
parser.add_argument("--initial-force-range-n", type=float, nargs=2, default=(1.0, 3.0))
parser.add_argument("--minimum-range-width-n", type=float, default=0.30)
parser.add_argument(
    "--advisor", choices=("deterministic", "openai"), default="deterministic"
)
parser.add_argument(
    "--vlm-model", default=os.environ.get("LAB_PICK_VLM_MODEL", "gpt-4.1-mini")
)
parser.add_argument(
    "--api-base", default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
)
parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
parser.add_argument(
    "--api-mode", choices=("responses", "chat_completions"), default="responses"
)
parser.add_argument(
    "--save-advisor-images", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument(
    "--record-videos", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument("--video-dir", type=Path, default=None)
parser.add_argument(
    "--video-camera",
    choices=("third", "wrist", "viewer", "tactile_left", "tactile_right"),
    default="third",
)
parser.add_argument("--video-every-n-physics-steps", type=int, default=4)
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--learned-noise-steps", type=int, default=1)
parser.add_argument(
    "--noise-padding-mode", choices=("repeat_last", "zeros"), default="repeat_last"
)
parser.add_argument("--dsrl-noise-residual-scale", type=float, default=0.25)
parser.add_argument("--dsrl-action-l2-weight", type=float, default=10.0)
parser.add_argument("--dsrl-max-gradient-updates", type=int, default=500)
parser.add_argument("--chunk-execute-steps", type=int, default=32)
parser.add_argument("--chunk-discount", type=float, default=0.99)
parser.add_argument("--flow-num-inference-steps", type=int, default=20)
parser.add_argument("--phase-horizon-steps", type=int, default=383)
parser.add_argument("--camera-warmup-steps", type=int, default=8)
parser.add_argument(
    "--gpu-max-rigid-contact-count",
    type=int,
    default=0,
    help="Optional lower PhysX contact-buffer capacity for one-environment runs.",
)
parser.add_argument(
    "--gpu-max-rigid-patch-count",
    type=int,
    default=0,
    help="Optional lower PhysX contact-patch capacity for one-environment runs.",
)
parser.add_argument("--labware-random-xy-m", type=float, nargs=2, default=(0.10, 0.10))
parser.add_argument("--labware-random-yaw-deg", type=float, default=0.0)
parser.add_argument("--learning-starts", type=int, default=32)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--actor-lr", type=float, default=3.0e-5)
parser.add_argument("--critic-lr", type=float, default=3.0e-5)
parser.add_argument("--alpha-lr", type=float, default=3.0e-5)
parser.add_argument(
    "--initial-log-std",
    type=float,
    default=None,
    help="Optional DSRL actor log-standard-deviation override.",
)
parser.add_argument("--checkpoint-interval", type=int, default=250)
parser.add_argument("--write-interval", type=int, default=25)
parser.add_argument("--max-outer-interactions", type=int, default=0)
parser.add_argument("--kp", type=float, default=0.006)
parser.add_argument("--ki", type=float, default=0.018)
parser.add_argument("--kd", type=float, default=0.00008)
parser.add_argument("--maximum-width-rate-m-s", type=float, default=0.018)
parser.add_argument("--guarded-minimum-width-m", type=float, default=0.0075)
parser.add_argument("--guarded-alignment-rate-m-s", type=float, default=0.020)
parser.add_argument("--guarded-maximum-alignment-offset-m", type=float, default=0.018)
parser.add_argument("--guarded-bilateral-confirm-steps", type=int, default=6)
parser.add_argument("--guarded-secure-dwell-steps", type=int, default=30)
parser.add_argument("--guarded-lift-rate-m-s", type=float, default=0.14)
parser.add_argument("--guarded-lift-height-m", type=float, default=0.225)
parser.add_argument("--guarded-hard-force-margin-n", type=float, default=0.30)
parser.add_argument("--flow-rwr-lr", type=float, default=1.0e-5)
parser.add_argument("--flow-rwr-weight-decay", type=float, default=1.0e-4)
parser.add_argument("--flow-rwr-batch-size", type=int, default=8)
parser.add_argument("--flow-rwr-updates-per-success", type=int, default=4)
parser.add_argument("--flow-rwr-replay-capacity", type=int, default=512)
parser.add_argument("--flow-rwr-grad-clip", type=float, default=1.0)
parser.add_argument("--flow-ppo-rollouts", type=int, default=32)
parser.add_argument("--flow-ppo-learning-epochs", type=int, default=8)
parser.add_argument("--flow-ppo-mini-batches", type=int, default=4)
parser.add_argument("--flow-ppo-learning-rate", type=float, default=3.0e-5)
parser.add_argument("--flow-ppo-gae-lambda", type=float, default=0.95)
parser.add_argument("--flow-ppo-ratio-clip", type=float, default=0.2)
parser.add_argument("--flow-ppo-value-clip", type=float, default=0.2)
parser.add_argument("--flow-ppo-entropy-scale", type=float, default=0.001)
parser.add_argument("--flow-ppo-grad-clip", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.num_episodes < 1:
    parser.error("--num-episodes must be positive.")
if args_cli.training_interactions < 0:
    parser.error("--training-interactions cannot be negative.")
if min(args_cli.gpu_max_rigid_contact_count, args_cli.gpu_max_rigid_patch_count) < 0:
    parser.error("PhysX GPU buffer capacities cannot be negative.")
if args_cli.dsrl_noise_residual_scale < 0.0 or args_cli.dsrl_action_l2_weight < 0.0:
    parser.error("DSRL residual scale and action-L2 weight must be non-negative.")
if args_cli.dsrl_max_gradient_updates < 1:
    parser.error("--dsrl-max-gradient-updates must be positive.")
if min(
    args_cli.flow_ppo_rollouts,
    args_cli.flow_ppo_learning_epochs,
    args_cli.flow_ppo_mini_batches,
) < 1:
    parser.error("Flow-PPO rollout and optimization counts must be positive.")
if args_cli.flow_ppo_rollouts % args_cli.flow_ppo_mini_batches:
    parser.error("--flow-ppo-rollouts must be divisible by --flow-ppo-mini-batches.")
if args_cli.flow_ppo_learning_rate <= 0.0 or args_cli.flow_ppo_grad_clip <= 0.0:
    parser.error("Flow-PPO learning rate and gradient clip must be positive.")
if args_cli.phase == "evaluation" and args_cli.training_interactions:
    parser.error("--training-interactions is only valid during online training.")
if args_cli.video_every_n_physics_steps < 1 or args_cli.video_fps < 1:
    parser.error("Video sampling interval and FPS must be positive.")
if args_cli.advisor == "openai" and args_cli.mode in {
    "ours_tactile",
    "joint",
    "joint_bilateral",
    "guarded_joint",
    "vlm",
}:
    if not os.environ.get(args_cli.api_key_env):
        parser.error(f"Set {args_cli.api_key_env} for --advisor openai.")
if args_cli.phase == "evaluation":
    if args_cli.mode in {
        "ours_tactile",
        "dsrl_tactile",
        "joint",
        "joint_bilateral",
        "guarded_joint",
        "dsrl",
        "flow_ppo",
    }:
        if args_cli.dsrl_checkpoint is None:
            parser.error("DSRL-bearing evaluation modes require --dsrl-checkpoint.")
    elif args_cli.dsrl_checkpoint is not None:
        parser.error("--dsrl-checkpoint is only valid for DSRL-bearing modes.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0], *hydra_args]


import gymnasium as gym  # noqa: E402
import skrl  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from skrl.agents.torch.ppo import PPO, PPO_CFG  # noqa: E402
from skrl.agents.torch.sac import SAC_CFG  # noqa: E402
from skrl.memories.torch import RandomMemory  # noqa: E402
from skrl.utils import set_seed  # noqa: E402

import tacex_tasks  # noqa: E402,F401
import tacex_tasks.lab_pick  # noqa: E402,F401
from clean_dsrl_agent import CleanDSRLSAC  # noqa: E402
from clean_dsrl_sac import (  # noqa: E402
    CLEAN_DSRL_CONTRACT_VERSION,
    CleanDSRLActor,
    build_clean_dsrl_sac_models,
    validate_absolute_dsrl_policy_state,
)
from episode_trainer import EpisodeLimitedSequentialTrainer  # noqa: E402
from hybrid_wrapper import VLMDSRLLabPickWrapper  # noqa: E402
from guarded_controller import GuardedControllerConfig  # noqa: E402
from guarded_wrapper import GuardedVLMDSRLLabPickWrapper  # noqa: E402
from flow_rwr import (  # noqa: E402
    FlowRWRLabPickWrapper,
    write_flow_rwr_metadata,
)
from flow_ppo import build_flow_ppo_models  # noqa: E402
from tactile_observation import tactile_contract_metadata  # noqa: E402
from vlm_force import (  # noqa: E402
    ConvergentForceEstimator,
    DeterministicVLMAdvisor,
    EpisodeForceAdaptationLoop,
    ForceControllerConfig,
    ForceEstimatorConfig,
    ForceRange,
    OpenAICompatibleVLMAdvisor,
)


TASK = "TacEx-LabPick-Slide-Clean-DSRL-SAC-v0"


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


def _make_adaptation(output_dir: Path):
    if args_cli.mode in {"dsrl_tactile", "dsrl", "base", "flow_rwr", "flow_ppo"}:
        return None
    physical_range = ForceRange(*args_cli.physical_force_range_n)
    initial_range = ForceRange(*args_cli.initial_force_range_n)
    if physical_range.maximum_n >= args_cli.break_force_threshold_n:
        raise ValueError(
            "physical force maximum must stay below the break-force threshold."
        )
    estimator = ConvergentForceEstimator(
        ForceEstimatorConfig(
            physical_range_n=physical_range,
            initial_range_n=initial_range,
            minimum_range_width_n=args_cli.minimum_range_width_n,
        )
    )
    if args_cli.advisor == "openai":
        advisor = OpenAICompatibleVLMAdvisor(
            model=args_cli.vlm_model,
            api_key=os.environ[args_cli.api_key_env],
            physical_range_n=physical_range,
            break_force_threshold_n=args_cli.break_force_threshold_n,
            api_base=args_cli.api_base,
            api_mode=args_cli.api_mode,
        )
    else:
        advisor = DeterministicVLMAdvisor(
            physical_range_n=physical_range,
            break_force_threshold_n=args_cli.break_force_threshold_n,
        )
    return EpisodeForceAdaptationLoop(
        advisor=advisor,
        estimator=estimator,
        log_path=output_dir / "vlm_episode_interactions.jsonl",
    )


def _make_agent_cfg(agent_cfg: dict[str, Any], output_dir: Path) -> SAC_CFG:
    cfg = copy.deepcopy(agent_cfg)
    cfg["batch_size"] = args_cli.batch_size
    cfg["learning_starts"] = args_cli.learning_starts
    cfg["learning_rate"] = [args_cli.actor_lr, args_cli.critic_lr, args_cli.alpha_lr]
    cfg["target_entropy"] = -0.5 * args_cli.learned_noise_steps * 10
    cfg["experiment"]["directory"] = str(output_dir / "training")
    cfg["experiment"]["experiment_name"] = "skrl"
    cfg["experiment"]["checkpoint_interval"] = args_cli.checkpoint_interval
    cfg["experiment"]["write_interval"] = args_cli.write_interval
    return SAC_CFG(**cfg)


def _run_native_bc(env: VLMDSRLLabPickWrapper) -> tuple[int, int, None]:
    env.reset(seed=args_cli.seed, flow_noise_seed=args_cli.seed)
    interactions = 0
    interaction_budget = args_cli.training_interactions or None
    max_interactions = (
        interaction_budget
        or args_cli.max_outer_interactions
        or args_cli.num_episodes * 20
    )
    while simulation_app.is_running() and (
        interactions < interaction_budget
        if interaction_budget is not None
        else env.completed_episodes < args_cli.num_episodes
    ):
        _, _, terminated, truncated, _ = env.step_bc()
        interactions += 1
        if bool((terminated | truncated).any().item()):
            result = env.complete_pending_episode(dsrl_updates_completed=0)
            print(
                "[EPISODE] "
                f"mode={env.mode} episode={result['episode_index'] + 1}/{args_cli.num_episodes} "
                f"success={result['success']} "
                f"reason={result['terminal_reason']}/{result['diagnosed_failure_reason']} "
                f"force_peak={result['peak_contact_force_n']:.3f}N",
                flush=True,
            )
            if (
                interactions < interaction_budget
                if interaction_budget is not None
                else env.completed_episodes < args_cli.num_episodes
            ):
                env.begin_auto_reset_episode()
        if (
            interactions >= max_interactions
            and interaction_budget is None
            and env.completed_episodes < args_cli.num_episodes
        ):
            raise RuntimeError(
                f"Reached {max_interactions} interactions before completing the episode budget."
            )
    return interactions, 0, None


def _run_frozen_evaluation(
    env: VLMDSRLLabPickWrapper,
    agent_cfg: dict[str, Any],
) -> tuple[int, int, Path | None]:
    """Evaluate on explicit paired seeds without any gradient updates."""

    actor = None
    checkpoint_path: Path | None = None
    if args_cli.mode in {
        "ours_tactile",
        "dsrl_tactile",
        "joint",
        "joint_bilateral",
        "guarded_joint",
        "dsrl",
        "flow_ppo",
    }:
        assert args_cli.dsrl_checkpoint is not None
        checkpoint_path = args_cli.dsrl_checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DSRL checkpoint not found: {checkpoint_path}")
        network_cfg = agent_cfg.get("network", {})
        actor = CleanDSRLActor(
            env.observation_space,
            env.action_space,
            torch.device(env.device),
            layout=env.layout,
            hidden_dims=network_cfg.get("actor_hidden_dims", (512, 512, 512)),
            initial_log_std=float(network_cfg.get("initial_log_std", 0.0)),
        ).to(env.device)
        checkpoint = torch.load(
            checkpoint_path, map_location=env.device, weights_only=False
        )
        if "policy" not in checkpoint:
            raise KeyError(f"Checkpoint has no policy state: {checkpoint_path}")
        validate_absolute_dsrl_policy_state(checkpoint["policy"])
        actor.load_state_dict(checkpoint["policy"], strict=True)
        actor.eval()

    interactions = 0
    max_per_episode = args_cli.max_outer_interactions or 20
    for episode in range(args_cli.num_episodes):
        episode_seed = args_cli.seed + episode
        random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)
        observation, _ = env.reset(
            seed=episode_seed,
            flow_noise_seed=episode_seed,
        )
        terminal = False
        episode_interactions = 0
        while simulation_app.is_running() and not terminal:
            if actor is None:
                observation, _, terminated, truncated, _ = env.step_bc()
            else:
                policy_observation = torch.as_tensor(
                    observation["policy"],
                    dtype=torch.float32,
                    device=env.device,
                ).reshape(1, -1)
                with torch.inference_mode():
                    inputs = {"observations": policy_observation}
                    action = (
                        actor.act(inputs, role="policy")[0]
                        if args_cli.evaluation_policy == "stochastic"
                        else actor.deterministic_action(inputs)
                    )
                observation, _, terminated, truncated, _ = env.step(action)
            interactions += 1
            episode_interactions += 1
            terminal = bool(torch.as_tensor(terminated | truncated).any().item())
            if episode_interactions >= max_per_episode and not terminal:
                raise RuntimeError(
                    f"Evaluation seed {episode_seed} exceeded "
                    f"{max_per_episode} outer interactions."
                )
        result = env.complete_pending_episode(dsrl_updates_completed=0)
        print(
            "[EVALUATION] "
            f"mode={env.mode} episode={episode + 1}/{args_cli.num_episodes} "
            f"seed={episode_seed} success={result['success']} "
            f"reason={result['terminal_reason']}/"
            f"{result['diagnosed_failure_reason']}",
            flush=True,
        )
    return interactions, 0, checkpoint_path


def _run_online_dsrl(
    env: VLMDSRLLabPickWrapper,
    agent_cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[int, int, Path]:
    vec_env = SkrlVecEnvWrapper(env, ml_framework="torch")
    cfg = _make_agent_cfg(agent_cfg["agent"], output_dir)
    cfg.discount_factor = env.outer_discount_factor
    memory = RandomMemory(
        memory_size=int(agent_cfg["memory_size"]),
        num_envs=vec_env.num_envs,
        device=vec_env.device,
    )
    network_cfg = agent_cfg.get("network", {})
    initial_log_std = (
        float(network_cfg.get("initial_log_std", 0.0))
        if args_cli.initial_log_std is None
        else args_cli.initial_log_std
    )
    models = build_clean_dsrl_sac_models(
        vec_env.observation_space,
        vec_env.state_space,
        vec_env.action_space,
        vec_env.device,
        layout=env.layout,
        actor_hidden_dims=network_cfg.get("actor_hidden_dims", (512, 512, 512)),
        critic_hidden_dims=network_cfg.get("critic_hidden_dims", (512, 512, 512)),
        initial_log_std=initial_log_std,
    )
    agent = CleanDSRLSAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=vec_env.observation_space,
        state_space=vec_env.state_space,
        action_space=vec_env.action_space,
        device=vec_env.device,
        backup_entropy=False,
        minimum_entropy_value=1.0e-3,
        action_l2_weight=args_cli.dsrl_action_l2_weight,
        max_gradient_updates=args_cli.dsrl_max_gradient_updates,
    )
    interaction_limited = args_cli.training_interactions > 0
    max_interactions = (
        args_cli.training_interactions
        if interaction_limited
        else args_cli.max_outer_interactions or args_cli.num_episodes * 20
    )
    trainer = EpisodeLimitedSequentialTrainer(
        episode_env=env,
        num_episodes=None if interaction_limited else args_cli.num_episodes,
        max_interactions=max_interactions,
        stop_at_interaction_budget=interaction_limited,
        cfg={
            "timesteps": max_interactions,
            "headless": True,
            "environment_info": agent_cfg["trainer"].get("environment_info", "log"),
            "close_environment_at_exit": False,
            "disable_progressbar": True,
        },
        env=vec_env,
        agents=agent,
    )
    trainer.train()
    checkpoint = output_dir / "dsrl_final.pt"
    agent.save(str(checkpoint))
    return (
        trainer.interactions_completed,
        trainer.gradient_updates_completed,
        checkpoint,
    )


def _run_online_flow_ppo(
    env: VLMDSRLLabPickWrapper,
    agent_cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[int, int, Path]:
    """Train PPO on the bounded latent noise supplied to the frozen Flow decoder."""

    vec_env = SkrlVecEnvWrapper(env, ml_framework="torch")
    memory = RandomMemory(
        memory_size=args_cli.flow_ppo_rollouts,
        num_envs=vec_env.num_envs,
        device=vec_env.device,
    )
    network_cfg = agent_cfg.get("network", {})
    initial_log_std = (
        float(network_cfg.get("initial_log_std", -2.0))
        if args_cli.initial_log_std is None
        else args_cli.initial_log_std
    )
    models = build_flow_ppo_models(
        vec_env.observation_space,
        vec_env.state_space,
        vec_env.action_space,
        vec_env.device,
        layout=env.layout,
        actor_hidden_dims=network_cfg.get("actor_hidden_dims", (512, 512, 512)),
        value_hidden_dims=network_cfg.get("critic_hidden_dims", (512, 512, 512)),
        initial_log_std=initial_log_std,
    )
    cfg = PPO_CFG(
        rollouts=args_cli.flow_ppo_rollouts,
        learning_epochs=args_cli.flow_ppo_learning_epochs,
        mini_batches=args_cli.flow_ppo_mini_batches,
        discount_factor=env.outer_discount_factor,
        gae_lambda=args_cli.flow_ppo_gae_lambda,
        learning_rate=args_cli.flow_ppo_learning_rate,
        grad_norm_clip=args_cli.flow_ppo_grad_clip,
        ratio_clip=args_cli.flow_ppo_ratio_clip,
        value_clip=args_cli.flow_ppo_value_clip,
        entropy_loss_scale=args_cli.flow_ppo_entropy_scale,
        value_loss_scale=1.0,
        experiment={
            "directory": str(output_dir / "training"),
            "experiment_name": "skrl_flow_ppo",
            "checkpoint_interval": args_cli.checkpoint_interval,
            "write_interval": args_cli.write_interval,
        },
    )
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=vec_env.observation_space,
        state_space=vec_env.state_space,
        action_space=vec_env.action_space,
        device=vec_env.device,
    )
    interaction_limited = args_cli.training_interactions > 0
    max_interactions = (
        args_cli.training_interactions
        if interaction_limited
        else args_cli.max_outer_interactions or args_cli.num_episodes * 20
    )
    trainer = EpisodeLimitedSequentialTrainer(
        episode_env=env,
        num_episodes=None if interaction_limited else args_cli.num_episodes,
        max_interactions=max_interactions,
        stop_at_interaction_budget=interaction_limited,
        cfg={
            "timesteps": max_interactions,
            "headless": True,
            "environment_info": agent_cfg["trainer"].get("environment_info", "log"),
            "close_environment_at_exit": False,
            "disable_progressbar": True,
        },
        env=vec_env,
        agents=agent,
    )
    trainer.train()
    checkpoint = output_dir / "flow_ppo_final.pt"
    agent.save(str(checkpoint))
    return (
        trainer.interactions_completed,
        trainer.gradient_updates_completed,
        checkpoint,
    )


@hydra_task_config(TASK, "skrl_clean_dsrl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict[str, Any],
) -> None:
    checkpoint = args_cli.bc_policy.expanduser().resolve()
    output_dir = args_cli.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {checkpoint}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    set_seed(args_cli.seed)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env_cfg.rl_align_cafe_action_yaw = False
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

    adaptation = _make_adaptation(output_dir)
    physical_env = gym.make(
        TASK,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.record_videos else None,
    )
    wrapper_type = (
        GuardedVLMDSRLLabPickWrapper
        if args_cli.mode in {"guarded_joint", "ours_tactile"}
        else (
            FlowRWRLabPickWrapper
            if args_cli.mode == "flow_rwr" and args_cli.phase == "online_training"
            else VLMDSRLLabPickWrapper
        )
    )
    wrapper_kwargs = {}
    if args_cli.mode in {"guarded_joint", "ours_tactile"}:
        wrapper_kwargs["guarded_config"] = GuardedControllerConfig(
            minimum_width_m=args_cli.guarded_minimum_width_m,
            hard_force_limit_n=args_cli.break_force_threshold_n,
            hard_force_margin_n=args_cli.guarded_hard_force_margin_n,
            alignment_rate_m_s=args_cli.guarded_alignment_rate_m_s,
            maximum_alignment_offset_m=args_cli.guarded_maximum_alignment_offset_m,
            bilateral_confirm_steps=args_cli.guarded_bilateral_confirm_steps,
            secure_dwell_steps=args_cli.guarded_secure_dwell_steps,
            lift_rate_m_s=args_cli.guarded_lift_rate_m_s,
            lift_height_m=args_cli.guarded_lift_height_m,
        )
    common_wrapper_kwargs = dict(
        output_dir=output_dir,
        **(
            wrapper_kwargs
            if args_cli.mode in {"guarded_joint", "ours_tactile"}
            else {
                "controller_config": ForceControllerConfig(
                    kp_width_rate_per_n=args_cli.kp,
                    ki_width_rate_per_n_s=args_cli.ki,
                    kd_width_per_n=args_cli.kd,
                    maximum_width_rate_m_s=args_cli.maximum_width_rate_m_s,
                    require_contact_mask_for_activation=args_cli.mode == "joint_bilateral",
                    hard_force_limit_n=args_cli.break_force_threshold_n,
                )
            }
        ),
        save_advisor_images=args_cli.save_advisor_images,
        record_videos=args_cli.record_videos,
        video_dir=args_cli.video_dir,
        video_camera=args_cli.video_camera,
        video_every_n_physics_steps=args_cli.video_every_n_physics_steps,
        video_fps=args_cli.video_fps,
        device=args_cli.device or "cuda:0",
        learned_noise_steps=args_cli.learned_noise_steps,
        padding_mode=args_cli.noise_padding_mode,
        noise_residual_scale=args_cli.dsrl_noise_residual_scale,
        chunk_execute_steps=args_cli.chunk_execute_steps,
        chunk_discount=args_cli.chunk_discount,
        flow_num_inference_steps=args_cli.flow_num_inference_steps,
        phase_horizon_steps=args_cli.phase_horizon_steps,
        camera_warmup_steps=args_cli.camera_warmup_steps,
        use_visual_xy_override=True,
        online_metrics_dir=output_dir / "online_metrics",
        seed=args_cli.seed,
    )
    if wrapper_type is FlowRWRLabPickWrapper:
        env = wrapper_type(
            physical_env,
            checkpoint,
            fine_tuner_kwargs={
                "learning_rate": args_cli.flow_rwr_lr,
                "weight_decay": args_cli.flow_rwr_weight_decay,
                "batch_size": args_cli.flow_rwr_batch_size,
                "gradient_steps_per_success": args_cli.flow_rwr_updates_per_success,
                "replay_capacity": args_cli.flow_rwr_replay_capacity,
                "grad_clip": args_cli.flow_rwr_grad_clip,
                "seed": args_cli.seed,
            },
            **common_wrapper_kwargs,
        )
    else:
        env = wrapper_type(
            physical_env,
            checkpoint,
            mode=args_cli.mode,
            adaptation=adaptation,
            **common_wrapper_kwargs,
        )
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args_cli.mode,
        "phase": args_cli.phase,
        "evaluation_policy": (
            args_cli.evaluation_policy if args_cli.phase == "evaluation" else None
        ),
        "num_episodes": args_cli.num_episodes,
        "training_interaction_budget": (
            args_cli.training_interactions or None
        ),
        "seed": args_cli.seed,
        "task": TASK,
        "bc_policy": str(checkpoint),
        "bc_checkpoint_sha256": _sha256(checkpoint),
        "advisor": None if adaptation is None else args_cli.advisor,
        "advisor_is_real_vlm": args_cli.advisor == "openai" and adaptation is not None,
        "break_force_threshold_n": args_cli.break_force_threshold_n,
        "physical_force_range_n": list(args_cli.physical_force_range_n),
        "initial_force_range_n": list(args_cli.initial_force_range_n),
        "labware_random_xy_m": list(args_cli.labware_random_xy_m),
        "labware_random_yaw_deg": args_cli.labware_random_yaw_deg,
        "learned_noise_steps": args_cli.learned_noise_steps,
        "noise_residual_scale": args_cli.dsrl_noise_residual_scale,
        "dsrl_action_l2_weight": args_cli.dsrl_action_l2_weight,
        "dsrl_max_gradient_updates": args_cli.dsrl_max_gradient_updates,
        "chunk_execute_steps": args_cli.chunk_execute_steps,
        "action_repeat": env.action_repeat,
        "chunk_discount": args_cli.chunk_discount,
        "flow_num_inference_steps": args_cli.flow_num_inference_steps,
        "gpu_max_rigid_contact_count": env_cfg.sim.physx.gpu_max_rigid_contact_count,
        "gpu_max_rigid_patch_count": env_cfg.sim.physx.gpu_max_rigid_patch_count,
        "use_visual_xy_override": True,
        "flow_condition_embedding_dim": env.layout.flow_condition_dim,
        "actor_observation_dim": env.layout.policy_dim,
        "critic_state_dim": env.layout.state_dim,
        "critic_input_dim": env.layout.critic_input_dim,
        "tactile_actor": tactile_contract_metadata(),
        "outer_interaction_log": str(
            output_dir / "online_metrics" / "online_interactions.jsonl"
        ),
        "learning_starts": (
            args_cli.learning_starts
            if args_cli.mode in {"ours_tactile", "dsrl_tactile", "joint", "joint_bilateral", "guarded_joint", "dsrl"}
            else None
        ),
        "batch_size": (
            args_cli.batch_size
            if args_cli.mode in {"ours_tactile", "dsrl_tactile", "joint", "joint_bilateral", "guarded_joint", "dsrl"}
            else None
        ),
        "learning_rates": (
            [args_cli.actor_lr, args_cli.critic_lr, args_cli.alpha_lr]
            if args_cli.mode in {"ours_tactile", "dsrl_tactile", "joint", "joint_bilateral", "guarded_joint", "dsrl"}
            else None
        ),
        "initial_log_std": args_cli.initial_log_std,
        "force_controller": (
            None
            if args_cli.mode in {"dsrl_tactile", "dsrl", "base", "flow_rwr", "flow_ppo"}
            else {
                "controlled_action_index": 9,
                "kp_width_rate_per_n": (
                    None if args_cli.mode in {"guarded_joint", "ours_tactile"} else args_cli.kp
                ),
                "ki_width_rate_per_n_s": (
                    None if args_cli.mode in {"guarded_joint", "ours_tactile"} else args_cli.ki
                ),
                "kd_width_per_n": (
                    None if args_cli.mode in {"guarded_joint", "ours_tactile"} else args_cli.kd
                ),
                "maximum_width_rate_m_s": (
                    None
                    if args_cli.mode in {"guarded_joint", "ours_tactile"}
                    else args_cli.maximum_width_rate_m_s
                ),
                "contact_gated": True,
                "contact_on_force_n": env.controller.config.contact_on_force_n,
                "contact_off_force_n": env.controller.config.contact_off_force_n,
                "release_hysteresis_steps": (
                    env.controller.config.release_hysteresis_steps
                ),
                "activation_gate": (
                    "bilateral_tactile_and_per_finger_force"
                    if args_cli.mode == "joint_bilateral"
                    else (
                        "tactile_contact_mode_supervisor"
                        if args_cli.mode in {"guarded_joint", "ours_tactile"}
                        else "unilateral_tactile_or_average_force"
                    )
                ),
                "require_contact_mask_for_activation": (
                    args_cli.mode in {"joint_bilateral", "guarded_joint", "ours_tactile"}
                ),
                "guarded_supervisor": (
                    None
                    if args_cli.mode not in {"guarded_joint", "ours_tactile"}
                    else {
                        "minimum_width_m": args_cli.guarded_minimum_width_m,
                        "hard_force_margin_n": args_cli.guarded_hard_force_margin_n,
                        "alignment_rate_m_s": args_cli.guarded_alignment_rate_m_s,
                        "maximum_alignment_offset_m": args_cli.guarded_maximum_alignment_offset_m,
                        "bilateral_confirm_steps": args_cli.guarded_bilateral_confirm_steps,
                        "secure_dwell_steps": args_cli.guarded_secure_dwell_steps,
                        "lift_rate_m_s": args_cli.guarded_lift_rate_m_s,
                        "lift_height_m": args_cli.guarded_lift_height_m,
                    }
                ),
            }
        ),
        "video_recording": (
            None
            if not args_cli.record_videos
            else {
                "directory": str(env.video_dir),
                "camera": args_cli.video_camera,
                "every_n_physics_steps": args_cli.video_every_n_physics_steps,
                "fps": args_cli.video_fps,
            }
        ),
        "dsrl_contract_version": CLEAN_DSRL_CONTRACT_VERSION,
        "input_dsrl_checkpoint": (
            None
            if args_cli.dsrl_checkpoint is None
            else str(args_cli.dsrl_checkpoint.expanduser().resolve())
        ),
        "flow_rwr": (
            None
            if args_cli.mode != "flow_rwr"
            else {
                "algorithm": "success_filtered_reward_weighted_regression",
                "direct_policy_update": True,
                "updated_module": "velocity_net",
                "learning_rate": args_cli.flow_rwr_lr,
                "weight_decay": args_cli.flow_rwr_weight_decay,
                "batch_size": args_cli.flow_rwr_batch_size,
                "updates_per_success": args_cli.flow_rwr_updates_per_success,
                "replay_capacity": args_cli.flow_rwr_replay_capacity,
                "grad_clip": args_cli.flow_rwr_grad_clip,
            }
        ),
        "flow_ppo": (
            None
            if args_cli.mode != "flow_ppo"
            else {
                "algorithm": "ppo_over_frozen_flow_initial_noise",
                "direct_flow_update": False,
                "rollouts": args_cli.flow_ppo_rollouts,
                "learning_epochs": args_cli.flow_ppo_learning_epochs,
                "mini_batches": args_cli.flow_ppo_mini_batches,
                "learning_rate": args_cli.flow_ppo_learning_rate,
                "gae_lambda": args_cli.flow_ppo_gae_lambda,
                "ratio_clip": args_cli.flow_ppo_ratio_clip,
                "value_clip": args_cli.flow_ppo_value_clip,
                "entropy_loss_scale": args_cli.flow_ppo_entropy_scale,
                "grad_clip": args_cli.flow_ppo_grad_clip,
            }
        ),
        "skrl_version": skrl.__version__,
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    dump_yaml(str(output_dir / "resolved_env.yaml"), env_cfg)
    dump_pickle(str(output_dir / "resolved_env.pkl"), env_cfg)
    _write_json(output_dir / "resolved_agent.json", agent_cfg)

    checkpoint_path: Path | None = None
    try:
        if args_cli.phase == "evaluation":
            interactions, updates, checkpoint_path = _run_frozen_evaluation(
                env, agent_cfg
            )
        elif args_cli.mode in {"vlm", "base", "flow_rwr"}:
            interactions, updates, checkpoint_path = _run_native_bc(env)
            if args_cli.mode == "flow_rwr":
                checkpoint_path = env.fine_tuner.save_checkpoint(
                    output_dir / "flow_rwr_final.pt",
                    source_checkpoint=checkpoint,
                )
                write_flow_rwr_metadata(
                    output_dir / "flow_rwr_metadata.json", env.fine_tuner
                )
                updates = env.fine_tuner.gradient_updates
        elif args_cli.mode == "flow_ppo":
            interactions, updates, checkpoint_path = _run_online_flow_ppo(
                env, agent_cfg, output_dir
            )
        else:
            interactions, updates, checkpoint_path = _run_online_dsrl(
                env, agent_cfg, output_dir
            )
    finally:
        env.close()

    successes = sum(int(item["success"]) for item in env.results)
    raw_failures = Counter(
        item["terminal_reason"] for item in env.results if not item["success"]
    )
    diagnosed_failures = Counter(
        item["diagnosed_failure_reason"] for item in env.results if not item["success"]
    )
    vlm_ranges = [
        item["vlm_recommended_force_range_n"]
        for item in env.results
        if item["vlm_recommended_force_range_n"] is not None
    ]
    summary = {
        **metadata,
        "completed_episodes": env.completed_episodes,
        "successes": successes,
        "success_rate": successes / max(env.completed_episodes, 1),
        "broken": int(raw_failures.get("object_broken", 0)),
        "broken_rate": raw_failures.get("object_broken", 0)
        / max(env.completed_episodes, 1),
        "failure_counts": dict(raw_failures),
        "diagnosed_failure_counts": dict(diagnosed_failures),
        "outer_interactions": interactions,
        "dsrl_gradient_updates": (
            updates
            if args_cli.mode in {"ours_tactile", "dsrl_tactile", "joint", "joint_bilateral", "guarded_joint", "dsrl"}
            else 0
        ),
        "flow_rwr_gradient_updates": (
            updates if args_cli.mode == "flow_rwr" else 0
        ),
        "flow_ppo_optimizer_steps": (
            updates if args_cli.mode == "flow_ppo" else 0
        ),
        "advisor_calls": env.advisor_calls,
        "final_force_range_n": (
            None if adaptation is None else adaptation.current_range_n.as_list()
        ),
        "vlm_recommended_force_range_envelope_n": (
            None
            if not vlm_ranges
            else [
                min(pair[0] for pair in vlm_ranges),
                max(pair[1] for pair in vlm_ranges),
            ]
        ),
        "maximum_non_gripper_action_delta": max(
            (item["max_non_gripper_action_delta"] for item in env.results),
            default=0.0,
        ),
        "trajectory_count": len(list((output_dir / "trajectories").glob("*.jsonl.gz"))),
        "video_count": (
            len(list(env.video_dir.glob("*.mp4"))) if args_cli.record_videos else 0
        ),
        "bilateral_gate_violations": sum(
            int(item.get("bilateral_activation_contract_satisfied") is False)
            for item in env.results
        ),
        "learned_checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "dsrl_checkpoint": (
            str(checkpoint_path)
            if checkpoint_path is not None
            and args_cli.mode in {"ours_tactile", "dsrl_tactile", "joint", "joint_bilateral", "guarded_joint", "dsrl"}
            else None
        ),
        "results": env.results,
    }
    _write_json(output_dir / "results.json", summary)
    print(
        f"[SUMMARY] mode={args_cli.mode} success={successes}/{env.completed_episodes} "
        f"({summary['success_rate']:.1%}) broken={summary['broken']} "
        f"advisor_calls={summary['advisor_calls']} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
