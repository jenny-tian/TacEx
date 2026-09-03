#!/usr/bin/env python3
"""Train the isolated clean 4-D residual SAC baseline for LabPick."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import random
import sys
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="TacEx-LabPick-Slide-Clean-Residual-SAC-v0",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timesteps", type=int, default=None)
parser.add_argument("--learning_starts", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--checkpoint_interval", type=int, default=None)
parser.add_argument("--write_interval", type=int, default=None)
parser.add_argument("--bc_policy", type=str, default=None, help="Required Flow BC checkpoint.")
parser.add_argument(
    "--resume_checkpoint",
    type=str,
    default=None,
    help="Optional SKRL agent checkpoint used to continue actor, critics, targets, and optimizers.",
)
parser.add_argument(
    "--resume_step",
    type=int,
    default=0,
    help="Completed training steps represented by --resume_checkpoint (metadata and run naming only).",
)
parser.add_argument("--bc_device", type=str, default="cuda:0")
parser.add_argument("--residual_scale", type=float, default=0.01)
parser.add_argument("--residual_contact_gate", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--action_l2_weight", type=float, default=1.0)
parser.add_argument("--flow_num_inference_steps", type=int, default=20)
parser.add_argument(
    "--flow_noise_seed",
    type=int,
    default=None,
    help="Base seed for the per-episode Flow noise template; defaults to --seed.",
)
parser.add_argument("--phase_horizon_steps", type=int, default=383)
parser.add_argument("--camera_warmup_steps", type=int, default=8)
parser.add_argument("--labware_random_xy_m", type=float, nargs=2, default=(0.10, 0.10))
parser.add_argument("--labware_random_yaw_deg", type=float, default=45.0)
parser.add_argument("--break_force_threshold_n", type=float, default=4.0)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=480)
parser.add_argument("--video_interval", type=int, default=5000)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0], *hydra_args]


import gymnasium as gym
import skrl
import torch

from skrl.agents.torch.sac import SAC_CFG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import tacex_tasks  # noqa: F401
import tacex_tasks.lab_pick  # noqa: F401

dsrl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dsrl"))
if dsrl_dir not in sys.path:
    sys.path.insert(0, dsrl_dir)

from clean_alpha_zero_sac import CleanAlphaZeroSAC
from clean_residual_sac import CleanResidualLayout, build_clean_residual_sac_models
from clean_residual_wrapper import CleanResidualLabPickWrapper


def _validate_cli() -> None:
    if args_cli.task != "TacEx-LabPick-Slide-Clean-Residual-SAC-v0":
        raise ValueError("The clean entrypoint only supports its dedicated LabPick task.")
    if args_cli.num_envs != 1:
        raise ValueError("The frozen Flow policy currently requires --num_envs 1.")
    if not args_cli.bc_policy:
        raise ValueError("--bc_policy is required.")
    if not os.path.isfile(args_cli.bc_policy):
        raise FileNotFoundError(f"Frozen Flow checkpoint not found: {args_cli.bc_policy}")
    if args_cli.resume_checkpoint is not None and not os.path.isfile(args_cli.resume_checkpoint):
        raise FileNotFoundError(f"Resume checkpoint not found: {args_cli.resume_checkpoint}")
    if args_cli.resume_checkpoint is not None and args_cli.resume_step < 1:
        raise ValueError("--resume_step must be positive when --resume_checkpoint is provided.")
    if args_cli.resume_checkpoint is None and args_cli.resume_step != 0:
        raise ValueError("--resume_step requires --resume_checkpoint.")
    if not math.isfinite(args_cli.residual_scale) or args_cli.residual_scale < 0:
        raise ValueError("--residual_scale must be finite and non-negative.")
    if not math.isfinite(args_cli.action_l2_weight) or args_cli.action_l2_weight < 0:
        raise ValueError("--action_l2_weight must be finite and non-negative.")
    if args_cli.flow_num_inference_steps < 1:
        raise ValueError("--flow_num_inference_steps must be positive.")
    if args_cli.flow_noise_seed is not None and args_cli.flow_noise_seed < 0:
        raise ValueError("--flow_noise_seed must be non-negative.")
    if args_cli.phase_horizon_steps < 1:
        raise ValueError("--phase_horizon_steps must be positive.")
    if args_cli.camera_warmup_steps < 1:
        raise ValueError(
            "--camera_warmup_steps must be at least 1 so reset observations "
            "cannot reuse a terminal camera frame."
        )


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_agent_cfg(agent_cfg: dict) -> SAC_CFG:
    cfg = copy.deepcopy(agent_cfg)
    if cfg.get("learn_entropy") is not False:
        raise ValueError("Clean residual SAC requires learn_entropy: false.")
    if float(cfg.get("initial_entropy_value", 0.0)) != 0.0:
        raise ValueError("Clean residual SAC requires initial_entropy_value: 0.0.")
    if int(cfg.get("random_timesteps", 0)) != 0:
        raise ValueError("Clean residual SAC requires random_timesteps: 0.")
    if cfg.get("rewards_shaper") is not None:
        raise ValueError("Clean residual SAC uses the environment reward without a reward shaper.")
    return SAC_CFG(**cfg)


@hydra_task_config(args_cli.task, "skrl_clean_sac_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict,
) -> None:
    _validate_cli()
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env_cfg.rl_align_cafe_action_yaw = False
    env_cfg.rl_action_penalty_scale = 0.0
    env_cfg.labware_pos_randomization_xy = tuple(args_cli.labware_random_xy_m)
    env_cfg.labware_yaw_randomization = math.radians(args_cli.labware_random_yaw_deg)
    env_cfg.terminate_break_force_threshold_n = args_cli.break_force_threshold_n

    agent_cfg["seed"] = args_cli.seed
    set_seed(args_cli.seed)
    if args_cli.timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.timesteps
    if args_cli.learning_starts is not None:
        agent_cfg["agent"]["learning_starts"] = args_cli.learning_starts
    if args_cli.batch_size is not None:
        agent_cfg["agent"]["batch_size"] = args_cli.batch_size
    if args_cli.checkpoint_interval is not None:
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = args_cli.checkpoint_interval
    if args_cli.write_interval is not None:
        agent_cfg["agent"]["experiment"]["write_interval"] = args_cli.write_interval

    flow_noise_seed = (
        args_cli.seed if args_cli.flow_noise_seed is None else args_cli.flow_noise_seed
    )
    scale_tag = str(args_cli.residual_scale).replace(".", "p")
    agent_cfg["agent"]["experiment"]["experiment_name"] = (
        f"clean_residual_sac_s{scale_tag}_basepreserving_chunk32_v3"
    )
    if args_cli.resume_checkpoint is not None:
        agent_cfg["agent"]["experiment"]["experiment_name"] += (
            f"_resume{args_cli.resume_step}"
        )
    log_root = os.path.abspath(
        os.path.join(
            "logs",
            "skrl",
            agent_cfg["agent"]["experiment"]["directory"],
        )
    )
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    log_dir = os.path.join(log_root, run_name)
    agent_cfg["agent"]["experiment"]["directory"] = log_root
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_name

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    dump_yaml(
        os.path.join(log_dir, "params", "clean_residual_sac.yaml"),
        {
            "contract": "clean_residual_sac_basepreserving_xyz_width_chunk32_v3",
            "algorithm_name": "SAC",
            "objective": "alpha_zero_one_step_twin_q",
            "entropy_loss": False,
            "entropy_backup": False,
            "target_actor": False,
            "skrl_version": skrl.__version__,
            "policy": "single_tanh_squashed_gaussian_from_timestep_zero",
            "external_behavior_noise": False,
            "random_action_timesteps": 0,
            "initial_log_std": agent_cfg["network"]["initial_log_std"],
            "min_log_std": -5.0,
            "max_log_std": 2.0,
            "log_std_trainable": True,
            "target": "reward_plus_gamma_not_terminated_min_target_q",
            "actor_loss": "negative_mean_min_online_q_plus_action_l2",
            "action_l2_weight": args_cli.action_l2_weight,
            "policy_observation": "proprio10_relative_position3_object_rot6d6_bc_action10_tactile5",
            "policy_observation_dim": 34,
            "critic_state_dim": 19,
            "critic_input": "critic_state19_full_bc_action10_raw_residual4",
            "critic_input_dim": 38,
            "residual_action_dim": 4,
            "residual_indices": [0, 1, 2, 9],
            "residual_scale": args_cli.residual_scale,
            "rotation_source": "frozen_bc_rot6d",
            "contact_gate": args_cli.residual_contact_gate,
            "bc_checkpoint": os.path.abspath(args_cli.bc_policy),
            "bc_checkpoint_bytes": os.path.getsize(args_cli.bc_policy),
            "bc_checkpoint_sha256": _sha256(args_cli.bc_policy),
            "flow_num_inference_steps": args_cli.flow_num_inference_steps,
            "flow_noise_seed_base": flow_noise_seed,
            "flow_noise_semantics": "native_gaussian_stream",
            "flow_phase_horizon_steps": args_cli.phase_horizon_steps,
            "flow_camera_warmup_steps": args_cli.camera_warmup_steps,
            "flow_visual_xy_override": True,
            "flow_visual_xy_lock_phase": 0.30,
            "bc_replan_actions": 32,
            "action_repeat": 2,
            "environment_reward_only": True,
            "environment_reward": "lab_pick_dense_shaped_reward",
            "contact_reward_semantics": "first_any_and_first_bilateral_contact_once_per_episode",
            "agent_side_potential_shaping": False,
            "timeout_is_terminal_for_bootstrap": True,
            "discount_factor": agent_cfg["agent"]["discount_factor"],
            "polyak": agent_cfg["agent"]["polyak"],
            "learning_rate": agent_cfg["agent"]["learning_rate"],
            "learning_starts": agent_cfg["agent"]["learning_starts"],
            "checkpoint_semantics": (
                "fresh_training_only"
                if args_cli.resume_checkpoint is None
                else "continued_model_and_optimizers_with_fresh_replay"
            ),
            "resume_checkpoint": (
                None
                if args_cli.resume_checkpoint is None
                else os.path.abspath(args_cli.resume_checkpoint)
            ),
            "resume_step": args_cli.resume_step,
            "additional_timesteps": agent_cfg["trainer"]["timesteps"],
            "replay_buffer_restored": False,
        },
    )

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    env = CleanResidualLabPickWrapper(
        env,
        args_cli.bc_policy,
        device=args_cli.bc_device,
        residual_scale=args_cli.residual_scale,
        flow_num_inference_steps=args_cli.flow_num_inference_steps,
        phase_horizon_steps=args_cli.phase_horizon_steps,
        camera_warmup_steps=args_cli.camera_warmup_steps,
        contact_gate=args_cli.residual_contact_gate,
        seed=flow_noise_seed,
    )
    layout = env.layout

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    device = env.device
    memory = RandomMemory(
        memory_size=int(agent_cfg["memory_size"]),
        num_envs=env.num_envs,
        device=device,
    )

    network_cfg = agent_cfg.get("network", {})
    expected_actor_dims = tuple(network_cfg.get("actor_hidden_dims", (256, 256)))
    expected_critic_dims = tuple(network_cfg.get("critic_hidden_dims", (256, 256)))
    if expected_actor_dims != (256, 256) or expected_critic_dims != (256, 256):
        raise ValueError("The clean v1 contract fixes both hidden networks to [256, 256].")
    models = build_clean_residual_sac_models(
        env.observation_space,
        env.state_space,
        env.action_space,
        device,
        layout=layout,
        initial_log_std=float(network_cfg.get("initial_log_std", -3.0)),
    )
    cfg = _make_agent_cfg(agent_cfg["agent"])
    agent = CleanAlphaZeroSAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
        action_l2_weight=args_cli.action_l2_weight,
    )
    print(f"[INFO] Log directory: {log_dir}")
    print(
        "[INFO] Clean residual SAC: obs=34 state=19 action=4 critic_input=38 "
        f"scale={layout.scale:g} contact_gate={args_cli.residual_contact_gate} "
        f"action_l2={args_cli.action_l2_weight:g} entropy=off target_actor=none"
    )
    trainer = SequentialTrainer(
        cfg={
            "timesteps": int(agent_cfg["trainer"]["timesteps"]),
            "headless": True,
            "environment_info": agent_cfg["trainer"].get("environment_info", "log"),
            "close_environment_at_exit": False,
        },
        env=env,
        agents=agent,
    )
    if args_cli.resume_checkpoint is not None:
        agent.load(args_cli.resume_checkpoint)
        print(
            "[INFO] Restored actor, critics, target critics, and optimizers from "
            f"{args_cli.resume_checkpoint}; replay memory starts empty."
        )
    trainer.train()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
