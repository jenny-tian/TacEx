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
import inspect
import itertools
import sys
import textwrap

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
parser.add_argument("--sac_discount_factor", type=float, default=None)
parser.add_argument("--sac_critic_learning_rate", type=float, default=None)
parser.add_argument("--sac_reward_scale", type=float, default=None)
parser.add_argument("--sac_target_entropy", type=float, default=None)
parser.add_argument(
    "--sac_terminal_sample_fraction",
    type=float,
    default=None,
    help="Minimum terminal-transition fraction in each replay batch. Defaults to 0.25 for DSRL.",
)
parser.add_argument(
    "--sac_backup_entropy",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Include alpha*log pi(a|s) in the SAC target-Q backup.",
)
parser.add_argument(
    "--sac_terminal_timeouts",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Treat time-limit truncations as terminal failures in replay. Defaults to enabled for DSRL.",
)
parser.add_argument("--rl_success_reward", type=float, default=None)
parser.add_argument("--rl_failure_penalty", type=float, default=None)
parser.add_argument("--rl_timeout_penalty", type=float, default=None)
parser.add_argument("--rl_late_no_progress_penalty", type=float, default=None)
parser.add_argument("--rl_late_no_progress_onset", type=float, default=None)
parser.add_argument("--dsrl_policy", type=str, default=None, help="Frozen Diffusion or Flow Matching BC checkpoint for DSRL-SAC.")
parser.add_argument("--dsrl_visual_pose_probe", type=str, default=None, help="Independent image pose probe checkpoint.")
parser.add_argument("--dsrl_device", type=str, default="cuda", help="Device used by the frozen BC policy.")
parser.add_argument("--dsrl_noise_magnitude", type=float, default=1.5)
parser.add_argument("--dsrl_chunk_discount", type=float, default=1.0)
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
parser.add_argument(
    "--dsrl_residual_mode",
    choices=["latent", "physical"],
    default="latent",
    help="Residual action space: frozen-BC latent noise or low-dimensional physical action deltas.",
)
parser.add_argument("--dsrl_flow_num_inference_steps", type=int, default=20)
parser.add_argument(
    "--dsrl_flow_chunk_execute_steps",
    type=int,
    default=16,
)
parser.add_argument("--dsrl_physical_residual_segments", type=int, default=4)
parser.add_argument("--dsrl_flow_phase_horizon_steps", type=int, default=383)
parser.add_argument("--dsrl_flow_camera_warmup_steps", type=int, default=8)
parser.add_argument(
    "--dsrl_gate",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Learn a state-dependent soft gate between native BC noise and SAC DSRL noise.",
)
parser.add_argument("--dsrl_gate_init", type=float, default=0.025)
parser.add_argument("--dsrl_gate_temperature", type=float, default=0.5)
parser.add_argument("--dsrl_gate_penalty", type=float, default=5.0)
parser.add_argument("--dsrl_gate_min", type=float, default=0.02)
parser.add_argument("--dsrl_gate_max", type=float, default=0.2)
parser.add_argument("--dsrl_exploration_log_std", type=float, default=-2.5)
parser.add_argument("--dsrl_base_noise_seed", type=int, default=42)
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
from skrl.utils import ScopedTimer

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
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU()))
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

    def initialize_bc_prior(
        self,
        *,
        gated: bool = False,
        gate_init: float = 0.1,
        gate_temperature: float = 0.5,
        gate_min: float = 0.0,
        gate_max: float = 0.3,
        residual_log_std: float = 0.0,
    ) -> None:
        output_layer = next(module for module in reversed(self.net) if isinstance(module, nn.Linear))
        with torch.no_grad():
            output_layer.weight.zero_()
            output_layer.bias.zero_()
            self.log_std_parameter.fill_(float(residual_log_std))
            if gated:
                if not 0.0 <= gate_min < gate_max <= 1.0:
                    raise ValueError("--dsrl_gate_min and --dsrl_gate_max must satisfy 0 <= min < max <= 1.")
                if not gate_min < gate_init < gate_max:
                    raise ValueError("--dsrl_gate_init must be strictly between --dsrl_gate_min and --dsrl_gate_max.")
                if gate_temperature <= 0.0:
                    raise ValueError("--dsrl_gate_temperature must be positive.")
                gate_logit = gate_temperature * torch.logit(
                    torch.tensor(
                        (gate_init - gate_min) / (gate_max - gate_min),
                        device=output_layer.bias.device,
                    )
                )
                gate_action = torch.tanh(gate_logit).clamp(-0.999999, 0.999999)
                output_layer.bias[-1] = torch.atanh(gate_action)
            self.log_std_parameter[-1] = -3.0

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


def _sac_agent_class(backup_entropy: bool, terminal_timeouts: bool):
    """Return skrl SAC with task-safe target backup and timeout handling."""
    if backup_entropy:
        base_class = SAC
    else:
        source = textwrap.dedent(inspect.getsource(SAC.update))
        old = "target_q_values = (\n                    torch.min(target_q1_values, target_q2_values) - self._entropy_coefficient * next_log_prob\n                )"
        new = "target_q_values = torch.min(target_q1_values, target_q2_values)"
        if old not in source:
            raise RuntimeError("Installed skrl SAC.update does not match expected entropy-backup implementation.")
        namespace = {
            "torch": torch,
            "F": torch.nn.functional,
            "nn": nn,
            "itertools": itertools,
            "config": skrl.config,
            "ScopedTimer": ScopedTimer,
        }
        exec(source.replace(old, new), namespace)

        class NoBackupEntropySAC(SAC):
            pass

        NoBackupEntropySAC.update = namespace["update"]
        base_class = NoBackupEntropySAC

    class TaskSAC(base_class):
        def record_transition(self, **transition):
            if terminal_timeouts:
                # Isaac Lab auto-resets at a time limit. Bootstrapping from the
                # returned reset observation makes an unsuccessful timeout look
                # like a valuable transition into a fresh episode.
                transition["terminated"] = transition["terminated"] | transition["truncated"]
            return super().record_transition(**transition)

    return TaskSAC


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
        for key, value in list(d.items()):
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
                    # ``rewards_shaper_scale`` is a convenience input for
                    # this script, not a SAC_CFG constructor argument.
                    d.pop(key, None)
        return d

    return update_dict(copy.deepcopy(cfg))


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.rl_success_reward is not None:
        env_cfg.rl_success_reward = args_cli.rl_success_reward
    if args_cli.rl_failure_penalty is not None:
        env_cfg.rl_failure_penalty = args_cli.rl_failure_penalty
    if args_cli.rl_timeout_penalty is not None:
        env_cfg.rl_timeout_penalty = args_cli.rl_timeout_penalty
    if args_cli.rl_late_no_progress_penalty is not None:
        env_cfg.rl_late_no_progress_penalty_scale = args_cli.rl_late_no_progress_penalty
    if args_cli.rl_late_no_progress_onset is not None:
        env_cfg.rl_late_no_progress_onset = args_cli.rl_late_no_progress_onset

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
    if args_cli.sac_discount_factor is not None:
        agent_cfg["agent"]["discount_factor"] = args_cli.sac_discount_factor
    if args_cli.sac_critic_learning_rate is not None:
        learning_rates = list(agent_cfg["agent"].get("learning_rate", []))
        if len(learning_rates) != 3:
            raise ValueError("SAC learning_rate must contain policy, critic, and entropy rates")
        learning_rates[1] = args_cli.sac_critic_learning_rate
        agent_cfg["agent"]["learning_rate"] = learning_rates
    if args_cli.sac_reward_scale is not None:
        if args_cli.sac_reward_scale <= 0.0:
            raise ValueError("--sac_reward_scale must be positive")
        agent_cfg["agent"]["rewards_shaper_scale"] = args_cli.sac_reward_scale
    if args_cli.sac_target_entropy is not None:
        agent_cfg["agent"]["target_entropy"] = args_cli.sac_target_entropy
    if args_cli.sac_checkpoint_interval is not None:
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = args_cli.sac_checkpoint_interval
    if args_cli.sac_write_interval is not None:
        agent_cfg["agent"]["experiment"]["write_interval"] = args_cli.sac_write_interval
    if args_cli.dsrl_policy and (
        args_cli.dsrl_policy_type == "flow_matching"
        or (args_cli.dsrl_policy_type == "auto" and os.path.isfile(args_cli.dsrl_policy))
    ):
        gate_suffix = "_gated" if args_cli.dsrl_gate else ""
        residual_suffix = f"_{args_cli.dsrl_residual_mode}"
        entropy_suffix = "_no_backup_entropy_ln" if not args_cli.sac_backup_entropy else ""
        timeout_suffix = "_terminal_timeouts" if args_cli.sac_terminal_timeouts is not False else ""
        balanced_suffix = "_balanced_terminal" if args_cli.sac_terminal_sample_fraction not in (None, 0.0) else ""
        agent_cfg["agent"]["experiment"]["experiment_name"] = (
            f"dsrl_sac_flow_bc_encoded{residual_suffix}{gate_suffix}{entropy_suffix}{timeout_suffix}{balanced_suffix}"
        )
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
        # Make the vendored BC implementations available when this script is
        # launched directly (the DSRL helper launcher normally sets these
        # paths in PYTHONPATH for us).
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        for dependency_path in (
            os.path.join(repo_root, "bc_policy"),
            os.path.join(repo_root, "scripts", "lerobot", "src"),
        ):
            if dependency_path not in sys.path:
                sys.path.insert(0, dependency_path)
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
            residual_mode=args_cli.dsrl_residual_mode,
            flow_num_inference_steps=args_cli.dsrl_flow_num_inference_steps,
            flow_chunk_execute_steps=args_cli.dsrl_flow_chunk_execute_steps,
            physical_residual_segments=args_cli.dsrl_physical_residual_segments,
            flow_phase_horizon_steps=args_cli.dsrl_flow_phase_horizon_steps,
            flow_camera_warmup_steps=args_cli.dsrl_flow_camera_warmup_steps,
            gate_enabled=args_cli.dsrl_gate,
            gate_temperature=args_cli.dsrl_gate_temperature,
            gate_penalty=args_cli.dsrl_gate_penalty,
            gate_min=args_cli.dsrl_gate_min,
            gate_max=args_cli.dsrl_gate_max,
            base_noise_seed=args_cli.dsrl_base_noise_seed,
            visual_pose_probe=args_cli.dsrl_visual_pose_probe,
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
    terminal_sample_fraction = args_cli.sac_terminal_sample_fraction
    if terminal_sample_fraction is None:
        terminal_sample_fraction = 0.25 if args_cli.dsrl_policy else 0.0
    if not 0.0 <= terminal_sample_fraction <= 1.0:
        raise ValueError("--sac_terminal_sample_fraction must be in [0, 1]")
    if args_cli.dsrl_policy:
        from terminal_balanced_memory import TerminalBalancedRandomMemory

        memory = TerminalBalancedRandomMemory(
            memory_size=agent_cfg["memory_size"],
            num_envs=env.num_envs,
            device=device,
            terminal_fraction=terminal_sample_fraction,
        )
    else:
        memory = RandomMemory(memory_size=agent_cfg["memory_size"], num_envs=env.num_envs, device=device)
    print(f"[INFO] Replay terminal sample fraction: {terminal_sample_fraction:.3f}")

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
        models["policy"].initialize_bc_prior(
            gated=args_cli.dsrl_gate,
            gate_init=args_cli.dsrl_gate_init,
            gate_temperature=args_cli.dsrl_gate_temperature,
            gate_min=args_cli.dsrl_gate_min,
            gate_max=args_cli.dsrl_gate_max,
            # A physical residual should start as an almost deterministic
            # BC-equivalent policy. SAC can expand this variance after the
            # replay buffer contains meaningful success/failure transitions.
            residual_log_std=(
                args_cli.dsrl_exploration_log_std
                if args_cli.dsrl_residual_mode == "physical"
                else 0.0
            ),
        )
        agent_cfg["agent"]["random_timesteps"] = 0
        print(
            "[INFO] Initialized SAC actor as a frozen-BC prior "
            f"with residual_log_std={args_cli.dsrl_exploration_log_std if args_cli.dsrl_residual_mode == 'physical' else 0.0:.1f}, "
            f"gate_enabled={args_cli.dsrl_gate} gate_init={args_cli.dsrl_gate_init:.3f}."
        )

    if args_cli.checkpoint:
        # A SKRL checkpoint restores the policy, critics, and optimizers, but
        # not replay memory or the trainer timestep. Collect the replay warmup
        # with the restored policy instead of switching back to random actions.
        agent_cfg["agent"]["random_timesteps"] = 0

    cfg = SAC_CFG(**_process_cfg(agent_cfg["agent"]))

    terminal_timeouts = args_cli.sac_terminal_timeouts
    if terminal_timeouts is None:
        terminal_timeouts = bool(args_cli.dsrl_policy)
    print(f"[INFO] Treating replay timeouts as terminal: {terminal_timeouts}")
    sac_agent_class = _sac_agent_class(args_cli.sac_backup_entropy, terminal_timeouts)
    agent = sac_agent_class(
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
        # torch optimizer state dicts contain their original learning rates.
        # Reapply an explicit CLI override after loading a checkpoint.
        if args_cli.sac_critic_learning_rate is not None:
            for param_group in agent.critic_optimizer.param_groups:
                param_group["lr"] = args_cli.sac_critic_learning_rate
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
