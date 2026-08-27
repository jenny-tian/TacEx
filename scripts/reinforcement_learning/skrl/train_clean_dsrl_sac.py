#!/usr/bin/env python3
"""Train reference-style low-dimensional Flow-noise DSRL-SAC for LabPick."""

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
parser.add_argument("--task", default="TacEx-LabPick-Slide-Clean-DSRL-SAC-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timesteps", type=int, default=None)
parser.add_argument("--learning_starts", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--checkpoint_interval", type=int, default=None)
parser.add_argument("--write_interval", type=int, default=None)
parser.add_argument("--actor_lr", type=float, default=None)
parser.add_argument("--critic_lr", type=float, default=None)
parser.add_argument("--alpha_lr", type=float, default=None)
parser.add_argument("--initial_log_std", type=float, default=None)
parser.add_argument("--initial_entropy_value", type=float, default=None)
parser.add_argument("--bc_policy", type=str, default=None, help="Required frozen Flow checkpoint.")
parser.add_argument("--resume_checkpoint", type=str, default=None)
parser.add_argument("--bc_device", type=str, default="cuda:0")
parser.add_argument("--learned_noise_steps", type=int, default=1)
parser.add_argument("--noise_padding_mode", choices=("repeat_last", "zeros"), default="repeat_last")
parser.add_argument("--chunk_execute_steps", type=int, default=32)
parser.add_argument("--chunk_discount", type=float, default=0.99)
parser.add_argument("--flow_num_inference_steps", type=int, default=20)
parser.add_argument("--phase_horizon_steps", type=int, default=383)
parser.add_argument("--camera_warmup_steps", type=int, default=8)
parser.add_argument("--labware_random_xy_m", type=float, nargs=2, default=(0.10, 0.10))
parser.add_argument("--labware_random_yaw_deg", type=float, default=45.0)
parser.add_argument("--break_force_threshold_n", type=float, default=4.0)
parser.add_argument("--backup_entropy", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--minimum_entropy_value", type=float, default=1.0e-3)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=480)
parser.add_argument("--video_interval", type=int, default=500)
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

from clean_dsrl_agent import CleanDSRLSAC
from clean_dsrl_sac import (
    CLEAN_DSRL_CONTRACT_VERSION,
    build_clean_dsrl_sac_models,
    validate_absolute_dsrl_policy_state,
)
from clean_dsrl_wrapper import CleanDSRLLabPickWrapper


def _validate_cli() -> None:
    if args_cli.task != "TacEx-LabPick-Slide-Clean-DSRL-SAC-v0":
        raise ValueError("The clean DSRL entrypoint only supports its dedicated task.")
    if args_cli.num_envs != 1:
        raise ValueError("The frozen Flow policy currently requires --num_envs 1.")
    if not args_cli.bc_policy:
        raise ValueError("--bc_policy is required.")
    if not os.path.isfile(args_cli.bc_policy):
        raise FileNotFoundError(f"Frozen Flow checkpoint not found: {args_cli.bc_policy}")
    if args_cli.resume_checkpoint is not None and not os.path.isfile(args_cli.resume_checkpoint):
        raise FileNotFoundError(f"Resume checkpoint not found: {args_cli.resume_checkpoint}")
    for value, name in (
        (args_cli.learned_noise_steps, "--learned_noise_steps"),
        (args_cli.chunk_execute_steps, "--chunk_execute_steps"),
        (args_cli.flow_num_inference_steps, "--flow_num_inference_steps"),
        (args_cli.phase_horizon_steps, "--phase_horizon_steps"),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive.")
    if args_cli.learned_noise_steps > 32 or args_cli.chunk_execute_steps > 32:
        raise ValueError("Learned and executed horizons cannot exceed the 32-step Flow horizon.")
    if args_cli.camera_warmup_steps < 1:
        raise ValueError("--camera_warmup_steps must be at least 1.")
    if not 0.0 < args_cli.chunk_discount <= 1.0:
        raise ValueError("--chunk_discount must lie in (0, 1].")
    if args_cli.minimum_entropy_value <= 0.0:
        raise ValueError("--minimum_entropy_value must be positive.")
    for value, name in (
        (args_cli.actor_lr, "--actor_lr"),
        (args_cli.critic_lr, "--critic_lr"),
        (args_cli.alpha_lr, "--alpha_lr"),
        (args_cli.initial_entropy_value, "--initial_entropy_value"),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be finite and positive.")
    if args_cli.initial_log_std is not None and not math.isfinite(args_cli.initial_log_std):
        raise ValueError("--initial_log_std must be finite.")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_agent_cfg(agent_cfg: dict) -> SAC_CFG:
    cfg = copy.deepcopy(agent_cfg)
    if cfg.get("learn_entropy") is not True:
        raise ValueError("Clean DSRL-SAC requires learn_entropy: true.")
    if int(cfg.get("random_timesteps", 0)) != 0:
        raise ValueError("Clean DSRL-SAC samples its policy from timestep zero.")
    if cfg.get("rewards_shaper") is not None:
        raise ValueError("Clean DSRL-SAC uses the wrapper's chunk return directly.")
    return SAC_CFG(**cfg)


@hydra_task_config(args_cli.task, "skrl_clean_dsrl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict,
) -> None:
    _validate_cli()
    if args_cli.resume_checkpoint is not None:
        resume_state = torch.load(
            args_cli.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if "policy" not in resume_state:
            raise KeyError(
                f"Resume checkpoint has no policy state: {args_cli.resume_checkpoint}"
            )
        validate_absolute_dsrl_policy_state(resume_state["policy"])
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
    for argument, section, key in (
        (args_cli.timesteps, "trainer", "timesteps"),
        (args_cli.learning_starts, "agent", "learning_starts"),
        (args_cli.batch_size, "agent", "batch_size"),
        (args_cli.checkpoint_interval, "experiment", "checkpoint_interval"),
        (args_cli.write_interval, "experiment", "write_interval"),
    ):
        if argument is None:
            continue
        if section == "experiment":
            agent_cfg["agent"]["experiment"][key] = argument
        else:
            agent_cfg[section][key] = argument
    learning_rates = list(agent_cfg["agent"]["learning_rate"])
    for index, override in enumerate(
        (args_cli.actor_lr, args_cli.critic_lr, args_cli.alpha_lr)
    ):
        if override is not None:
            learning_rates[index] = override
    agent_cfg["agent"]["learning_rate"] = learning_rates
    if args_cli.initial_log_std is not None:
        agent_cfg["network"]["initial_log_std"] = args_cli.initial_log_std
    if args_cli.initial_entropy_value is not None:
        agent_cfg["agent"]["initial_entropy_value"] = args_cli.initial_entropy_value

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    env = CleanDSRLLabPickWrapper(
        env,
        args_cli.bc_policy,
        device=args_cli.bc_device,
        learned_noise_steps=args_cli.learned_noise_steps,
        padding_mode=args_cli.noise_padding_mode,
        chunk_execute_steps=args_cli.chunk_execute_steps,
        chunk_discount=args_cli.chunk_discount,
        flow_num_inference_steps=args_cli.flow_num_inference_steps,
        phase_horizon_steps=args_cli.phase_horizon_steps,
        camera_warmup_steps=args_cli.camera_warmup_steps,
        seed=args_cli.seed,
    )
    layout = env.layout
    agent_cfg["agent"]["discount_factor"] = env.outer_discount_factor
    agent_cfg["agent"]["target_entropy"] = -0.5 * layout.action_dim
    agent_cfg["agent"]["experiment"]["experiment_name"] = (
        f"clean_dsrl_sac_absolute_l{layout.learned_noise_steps}"
        f"_e{args_cli.chunk_execute_steps}_v2"
    )
    log_root = os.path.abspath(
        os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
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
        os.path.join(log_dir, "params", "clean_dsrl_sac.yaml"),
        {
            "contract": "flow_noise_dsrl_absolute_repeat_last_v2",
            "contract_version": CLEAN_DSRL_CONTRACT_VERSION,
            "reference": "/home/limx/vlarl/src/dsrl",
            "algorithm": "SAC",
            "backup_entropy": args_cli.backup_entropy,
            "learn_entropy": True,
            "minimum_entropy_value": args_cli.minimum_entropy_value,
            "target_entropy": agent_cfg["agent"]["target_entropy"],
            "learning_rate": agent_cfg["agent"]["learning_rate"],
            "initial_log_std": agent_cfg["network"]["initial_log_std"],
            "initial_entropy_value": agent_cfg["agent"]["initial_entropy_value"],
            "actor_observation": "frozen_flow_global_cond",
            "actor_observation_dim": layout.policy_dim,
            "critic_state": "normalized_proprio10_relative_position3_object_rot6d6",
            "critic_state_dim": layout.state_dim,
            "critic_input_dim": layout.critic_input_dim,
            "learned_noise_shape": [layout.learned_noise_steps, layout.noise_dim],
            "decoder_noise_shape": [layout.flow_horizon, layout.noise_dim],
            "noise_padding_mode": layout.padding_mode,
            "noise_action_semantics": "absolute",
            "noise_action_bounds": [-1.0, 1.0],
            "chunk_execute_steps": args_cli.chunk_execute_steps,
            "action_repeat": env.action_repeat,
            "chunk_discount": args_cli.chunk_discount,
            "outer_discount_factor": env.outer_discount_factor,
            "bc_checkpoint": os.path.abspath(args_cli.bc_policy),
            "bc_checkpoint_bytes": os.path.getsize(args_cli.bc_policy),
            "bc_checkpoint_sha256": _sha256(args_cli.bc_policy),
            "flow_num_inference_steps": args_cli.flow_num_inference_steps,
            "environment_reward_only": True,
            "timeout_is_terminal_for_bootstrap": True,
            "skrl_version": skrl.__version__,
        },
    )

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
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
    models = build_clean_dsrl_sac_models(
        env.observation_space,
        env.state_space,
        env.action_space,
        device,
        layout=layout,
        actor_hidden_dims=network_cfg.get("actor_hidden_dims", (512, 512, 512)),
        critic_hidden_dims=network_cfg.get("critic_hidden_dims", (512, 512, 512)),
        initial_log_std=float(network_cfg.get("initial_log_std", 0.0)),
    )
    agent = CleanDSRLSAC(
        models=models,
        memory=memory,
        cfg=_make_agent_cfg(agent_cfg["agent"]),
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
        backup_entropy=args_cli.backup_entropy,
        minimum_entropy_value=args_cli.minimum_entropy_value,
    )
    if args_cli.resume_checkpoint is not None:
        agent.load(args_cli.resume_checkpoint)

    print(f"[INFO] Log directory: {log_dir}")
    print(
        "[INFO] Clean DSRL-SAC: "
        f"obs={layout.policy_dim} state={layout.state_dim} "
        f"learned_noise={layout.learned_noise_steps}x{layout.noise_dim} "
        f"decoder_noise={layout.flow_horizon}x{layout.noise_dim} "
        f"semantics=absolute execute={args_cli.chunk_execute_steps} "
        f"gamma_outer={env.outer_discount_factor:.8f}"
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
    trainer.train()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
