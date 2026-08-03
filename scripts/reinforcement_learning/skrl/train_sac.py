# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""
import argparse
import copy
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--timesteps", type=int, default=None, help="Override the configured number of environment steps.")
parser.add_argument("--sac_learning_starts", type=int, default=None)
parser.add_argument("--sac_batch_size", type=int, default=None)
parser.add_argument("--sac_checkpoint_interval", type=int, default=None)
parser.add_argument("--sac_write_interval", type=int, default=None)
parser.add_argument("--dsrl_policy", type=str, default=None, help="Frozen Diffusion or Flow Matching BC checkpoint for DSRL-SAC.")
parser.add_argument("--dsrl_device", type=str, default="cuda", help="Device used by the frozen BC policy.")
parser.add_argument("--dsrl_noise_magnitude", type=float, default=1.5)
parser.add_argument(
    "--dsrl_action_mode",
    choices=["noise", "residual"],
    default="noise",
    help="SAC controls the full BC noise tensor or a 4-D xyz/width residual. Noise preserves old checkpoints.",
)
parser.add_argument(
    "--dsrl_residual_position_scale_m",
    type=float,
    nargs=3,
    default=(0.03, 0.03, 0.01),
    metavar=("X", "Y", "Z"),
)
parser.add_argument("--dsrl_residual_width_scale_m", type=float, default=0.002)
parser.add_argument(
    "--dsrl_residual_penalty_scale",
    type=float,
    default=0.0,
    help="L2 reward penalty on normalized residual actions; zero preserves legacy behavior.",
)
parser.add_argument("--dsrl_curriculum_steps", type=int, default=0)
parser.add_argument("--dsrl_curriculum_start_step", type=int, default=0)
parser.add_argument(
    "--dsrl_curriculum_start_xy_m",
    type=float,
    nargs=2,
    default=(0.05, 0.05),
    metavar=("X", "Y"),
)
parser.add_argument(
    "--dsrl_curriculum_end_xy_m",
    type=float,
    nargs=2,
    default=(0.10, 0.10),
    metavar=("X", "Y"),
)
parser.add_argument("--dsrl_curriculum_start_yaw_deg", type=float, default=30.0)
parser.add_argument("--dsrl_curriculum_end_yaw_deg", type=float, default=45.0)
parser.add_argument("--dsrl_chunk_discount", type=float, default=0.99)
parser.add_argument(
    "--dsrl_action_repeat",
    type=int,
    default=2,
    help="Physics steps per decoded action. Use 2 for a 60 Hz BC dataset in the 120 Hz LabPick simulator.",
)
parser.add_argument(
    "--dsrl_policy_type",
    choices=["auto", "diffusion", "flow_matching"],
    default="auto",
    help="Frozen BC family. Auto detects .pt files as sim_robot Flow Matching.",
)
parser.add_argument("--dsrl_flow_num_inference_steps", type=int, default=20)
parser.add_argument(
    "--dsrl_flow_chunk_execute_steps",
    type=int,
    default=32,
)
parser.add_argument("--dsrl_flow_phase_horizon_steps", type=int, default=383)
parser.add_argument("--dsrl_flow_camera_warmup_steps", type=int, default=8)
parser.add_argument(
    "--dsrl_bc_prior_init",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Initialize the SAC actor as the frozen BC's zero-mean unit-Gaussian prior.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="SAC",
    choices=["SAC"],
    help="The RL algorithm used for training the skrl agent.",
)

parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to model.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

"""Rest everything follows."""

from isaacsim.core.utils.extensions import enable_extension

enable_extension("omni.isaac.debug_draw")

import gymnasium as gym
import math
import os
import random
import torch
import torch.nn as nn
from datetime import datetime

import skrl
from packaging import version

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_CFG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer

# check for minimum supported skrl version
SKRL_VERSION = "2.0.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch") or args_cli.ml_framework.startswith("jax"):
    pass

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import tacex_tasks  # noqa: F401
import tacex_tasks.lab_pick  # noqa: F401


# define models (stochastic and deterministic models) using mixins
def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, final_activation=None) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.ELU()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    if final_activation is not None:
        layers.append(final_activation())
    return nn.Sequential(*layers)


class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_dims: list[int]):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_mean_actions=True,
            clip_log_std=True,
            min_log_std=-5,
            max_log_std=2,
        )
        self.net = _build_mlp(self.num_observations, hidden_dims, self.num_actions, nn.Tanh)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def initialize_bc_prior(self, *, log_std: float = 0.0) -> None:
        output_layer = next(module for module in reversed(self.net) if isinstance(module, nn.Linear))
        with torch.no_grad():
            output_layer.weight.zero_()
            output_layer.bias.zero_()
            self.log_std_parameter.fill_(log_std)

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_dims: list[int]):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = _build_mlp(self.num_observations + self.num_actions, hidden_dims, 1)

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["observations"], inputs["taken_actions"]], dim=1)), {}


# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


def _process_cfg(cfg: dict) -> dict:
    """Convert simple types to skrl classes/components
    :param cfg: A configuration dictionary
    :return: Updated dictionary
    """
    _direct_eval = [
        "learning_rate_scheduler",
        "shared_state_preprocessor",
        "state_preprocessor",
        "value_preprocessor",
    ]

    def reward_shaper_function(scale):
        def reward_shaper(rewards, *args, **kwargs):
            return rewards * scale

        return reward_shaper

    def update_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in _direct_eval:
                    if type(d[key]) is str:
                        d[key] = eval(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
                elif key in ["rewards_shaper_scale"]:
                    d["rewards_shaper"] = reward_shaper_function(value)
        return d

    return update_dict(copy.deepcopy(cfg))


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.dsrl_policy and args_cli.dsrl_curriculum_steps > 0:
        env_cfg.randomize_labware_position = True
        env_cfg.labware_pos_randomization_xy = tuple(args_cli.dsrl_curriculum_start_xy_m)
        env_cfg.labware_yaw_randomization = math.radians(args_cli.dsrl_curriculum_start_yaw_deg)

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # SAC is updated every environment step, so the legacy max_iterations
    # option is treated as a step override as well.
    if args_cli.timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.timesteps
    elif args_cli.max_iterations is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations
    if args_cli.sac_learning_starts is not None:
        agent_cfg["agent"]["learning_starts"] = args_cli.sac_learning_starts
    if args_cli.sac_batch_size is not None:
        agent_cfg["agent"]["batch_size"] = args_cli.sac_batch_size
    if args_cli.sac_checkpoint_interval is not None:
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = args_cli.sac_checkpoint_interval
    if args_cli.sac_write_interval is not None:
        agent_cfg["agent"]["experiment"]["write_interval"] = args_cli.sac_write_interval
    if args_cli.dsrl_policy and (
        args_cli.dsrl_policy_type == "flow_matching"
        or (args_cli.dsrl_policy_type == "auto" and os.path.isfile(args_cli.dsrl_policy))
    ):
        suffix = "residual" if args_cli.dsrl_action_mode == "residual" else "noise"
        agent_cfg["agent"]["experiment"]["experiment_name"] = f"dsrl_sac_flow_bc_{suffix}"
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
        # log_dir = os.path.dirname(os.path.dirname(resume_path))

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.dsrl_policy:
        if args_cli.task != "TacEx-LabPick-Slide-DSRL-Base-v0":
            raise ValueError(
                "--dsrl_policy currently requires --task TacEx-LabPick-Slide-DSRL-Base-v0"
            )
        dsrl_module_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dsrl")
        )
        if dsrl_module_dir not in sys.path:
            sys.path.insert(0, dsrl_module_dir)
        from lab_pick_dsrl_wrapper import LabPickDSRLWrapper

        env = LabPickDSRLWrapper(
            env,
            args_cli.dsrl_policy,
            device=args_cli.dsrl_device,
            noise_magnitude=args_cli.dsrl_noise_magnitude,
            chunk_discount=args_cli.dsrl_chunk_discount,
            action_repeat=args_cli.dsrl_action_repeat,
            policy_type=args_cli.dsrl_policy_type,
            action_mode=args_cli.dsrl_action_mode,
            residual_position_scale_m=tuple(args_cli.dsrl_residual_position_scale_m),
            residual_width_scale_m=args_cli.dsrl_residual_width_scale_m,
            residual_penalty_scale=args_cli.dsrl_residual_penalty_scale,
            curriculum_steps=args_cli.dsrl_curriculum_steps,
            curriculum_start_step=args_cli.dsrl_curriculum_start_step,
            curriculum_start_xy_m=tuple(args_cli.dsrl_curriculum_start_xy_m),
            curriculum_end_xy_m=tuple(args_cli.dsrl_curriculum_end_xy_m),
            curriculum_start_yaw_rad=math.radians(args_cli.dsrl_curriculum_start_yaw_deg),
            curriculum_end_yaw_rad=math.radians(args_cli.dsrl_curriculum_end_yaw_deg),
            flow_num_inference_steps=args_cli.dsrl_flow_num_inference_steps,
            flow_chunk_execute_steps=args_cli.dsrl_flow_chunk_execute_steps,
            flow_phase_horizon_steps=args_cli.dsrl_flow_phase_horizon_steps,
            flow_camera_warmup_steps=args_cli.dsrl_flow_camera_warmup_steps,
        )
    # wrap for video recording
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

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # # configure and instantiate the skrl runner
    # # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    # runner = Runner(env, agent_cfg)

    device = env.device

    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=agent_cfg["memory_size"], num_envs=env.num_envs, device=device)

    # instantiate the agent's models (function approximators).
    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
    hidden_dims = [int(value) for value in agent_cfg.get("network", {}).get("hidden_dims", [256, 256])]
    print(f"[INFO] SAC spaces: observation={env.observation_space}, action={env.action_space}")
    print(f"[INFO] SAC hidden dimensions: {hidden_dims}")
    models = {
        "policy": StochasticActor(env.observation_space, env.action_space, device, hidden_dims),
        "critic_1": Critic(env.observation_space, env.action_space, device, hidden_dims),
        "critic_2": Critic(env.observation_space, env.action_space, device, hidden_dims),
        "target_critic_1": Critic(env.observation_space, env.action_space, device, hidden_dims),
        "target_critic_2": Critic(env.observation_space, env.action_space, device, hidden_dims),
    }

    if args_cli.dsrl_policy and args_cli.dsrl_bc_prior_init and not args_cli.checkpoint:
        prior_log_std = -2.0 if args_cli.dsrl_action_mode == "residual" else 0.0
        models["policy"].initialize_bc_prior(log_std=prior_log_std)
        agent_cfg["agent"]["random_timesteps"] = 0
        print(
            "[INFO] Initialized SAC actor as frozen-BC prior "
            f"action_mode={args_cli.dsrl_action_mode} log_std={prior_log_std:.1f} "
            f"residual_penalty={args_cli.dsrl_residual_penalty_scale:.3g}."
        )

    cfg = SAC_CFG(**_process_cfg(agent_cfg["agent"]))

    agent = SAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    if args_cli.checkpoint:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)
    # configure and instantiate the RL trainer
    cfg_trainer = {
        "timesteps": agent_cfg["trainer"]["timesteps"],
        "headless": True,
        "environment_info": agent_cfg["trainer"].get("environment_info", "episode"),
        "close_environment_at_exit": False,
    }  # headless command gets overridden by IsaacLab argument
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    # SKRL 2.1's trainer handles single-environment reset and environment_info logging.
    trainer.train()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
