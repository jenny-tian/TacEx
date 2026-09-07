#!/usr/bin/env python3
"""Train and compare tactile residual methods on force-constrained GrapePick.

The frozen 10-D BC action is shared by every method. Learned policies observe
the same 11-D vector, including the two live pad-normal contact loads, and emit
bounded residuals for tool X/Z and gripper width. DPPO uses an explicit
stochastic denoising chain; DSRL applies SAC in the normalized BC-action latent
coordinates. VLM+DSRL can either reproduce the legacy runtime force guard or,
in ``no_guard`` mode, use the VLM range only inside critic conditioning and
training losses while leaving every executed action to the learned actor.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _reexec_with_cuda_libraries() -> None:
    py_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / py_dir / "site-packages"
    required = [site_packages / "torch" / "lib", site_packages / "nvidia" / "cudnn" / "lib"]
    current = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item]
    missing = [str(item) for item in required if item.is_dir() and str(item) not in current]
    if missing:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*missing, *current])
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


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


METHODS = ("base", "residual_rl", "awr", "dppo", "dsrl", "vlm_dsrl", "vlm_guard")
VLM_METHODS = ("vlm_dsrl", "vlm_guard")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bc-checkpoint", type=Path, required=True)
parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("exp_report/compare_exp_100_with_tactile/grape_pick"),
)
parser.add_argument("--vlm-force-range", type=Path, default=None)
parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
parser.add_argument("--train-episodes", type=int, default=100)
parser.add_argument("--eval-episodes", type=int, default=100)
parser.add_argument("--eval-seeds", type=int, nargs="+", default=(50000, 60000, 70000))
parser.add_argument("--train-seed", type=int, default=100000)
parser.add_argument("--validation-seed", type=int, default=900000)
parser.add_argument("--validation-episodes", type=int, default=40)
parser.add_argument("--num-envs", type=int, default=20)
parser.add_argument("--max-episode-steps", type=int, default=790)
parser.add_argument("--action-repeat", type=int, default=4)
parser.add_argument("--action-smoothing", type=float, default=0.35)
parser.add_argument("--damage-range", type=float, nargs=2, default=(3.40, 5.00))
parser.add_argument("--detach-range", type=float, nargs=2, default=(2.80, 3.40))
parser.add_argument("--sac-updates-per-iteration", type=int, default=50)
parser.add_argument(
    "--vlm-controller-mode",
    choices=("legacy_guarded_pull", "rl_owned_pull", "no_guard"),
    default="legacy_guarded_pull",
    help=(
        "legacy_guarded_pull reproduces the original controller. rl_owned_pull makes the "
        "VLM layer a force-only guard and leaves X/Z/width residual control to RL. "
        "no_guard removes all rule-based VLM action edits: VLM information is used only "
        "as policy/Q conditioning and reward shaping."
    ),
)
parser.add_argument(
    "--sac-exploration-scale",
    type=float,
    default=1.0,
    help=(
        "Initial multiplier on SAC behavior-policy standard deviation during replay "
        "collection. Retained as a constant for backward compatibility unless a decay "
        "schedule and final scale are supplied."
    ),
)
parser.add_argument(
    "--sac-exploration-final-scale",
    type=float,
    default=None,
    help="Final SAC behavior-noise multiplier; defaults to the initial scale.",
)
parser.add_argument(
    "--sac-exploration-schedule",
    choices=("constant", "linear", "cosine", "exponential"),
    default="constant",
    help="Schedule for the external SAC behavior-noise multiplier across rollout batches.",
)
parser.add_argument(
    "--vlm-rl-algorithm",
    choices=("sac", "safe_pull_bandit"),
    default="sac",
    help="Residual learner used by VLM+DSRL in rl_owned_pull mode.",
)
parser.add_argument(
    "--vlm-q-force-conditioning",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Condition the VLM+DSRL critic on the VLM force interval and live "
        "controller action terms. The actor still emits only the learned residual."
    ),
)
parser.add_argument(
    "--vlm-force-band-reward",
    type=float,
    default=0.0,
    help="Per-step in-band reward used only by VLM+DSRL no_guard.",
)
parser.add_argument(
    "--vlm-force-violation-penalty",
    type=float,
    default=0.02,
    help="Per-step force-band violation penalty used only by VLM+DSRL no_guard.",
)
parser.add_argument(
    "--vlm-action-prior-weight",
    type=float,
    default=1.0,
    help=(
        "Weight of the VLM force-error actor prior in no_guard mode. The prior is "
        "used only in the training loss and never edits an executed action."
    ),
)
parser.add_argument(
    "--vlm-action-prior-gain",
    type=float,
    default=0.20,
    help="Residual-width correction per normalized VLM force error for the actor prior.",
)
parser.add_argument("--bandit-epsilon-start", type=float, default=1.0)
parser.add_argument("--bandit-epsilon-end", type=float, default=0.10)
parser.add_argument(
    "--min-validation-gain",
    type=float,
    default=None,
    help="Require the selected trained policy to beat its untrained validation score by this amount.",
)
parser.add_argument(
    "--vlm-guard-residual",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Z", "WIDTH"),
    help="Fixed normalized residual for the no-learning VLM guard diagnostic.",
)
parser.add_argument(
    "--vlm-guard-pull-residual",
    type=float,
    default=0.0,
    help=(
        "Fixed normalized pull residual for the no-learning guard diagnostic in "
        "rl_owned_pull mode; zero is the formal guard-only ablation."
    ),
)
parser.add_argument("--skip-evaluation", action="store_true")
parser.add_argument("--learner-device", choices=("auto", "cpu", "cuda"), default="auto")
parser.add_argument("--skip-training", action="store_true")
parser.add_argument(
    "--require-trained-checkpoint",
    action="store_true",
    help="Exclude the untrained initialization from validation checkpoint selection.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.sac_exploration_scale < 0.0:
    parser.error("--sac-exploration-scale must be non-negative")
if args.sac_exploration_final_scale is not None and args.sac_exploration_final_scale < 0.0:
    parser.error("--sac-exploration-final-scale must be non-negative")
if (
    args.sac_exploration_schedule == "exponential"
    and (
        args.sac_exploration_scale <= 0.0
        or (
            args.sac_exploration_final_scale is not None
            and args.sac_exploration_final_scale <= 0.0
        )
    )
):
    parser.error("exponential SAC exploration decay requires positive endpoint scales")
args.enable_cameras = False

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import torch

from learning import (
    AWRAgent,
    DiffusionPPOAgent,
    ReplayBuffer,
    SACBatch,
    SACResidualAgent,
    SafePullBanditAgent,
    discounted_returns,
    exploration_scale_schedule,
)
from tacex_tasks.grape_pick.grape_pick_env import GrapePickEnv
from tacex_tasks.grape_pick.grape_pick_env_cfg import GrapePickEnvCfg
from tacex_tasks.grape_pick.policy import (
    GrapeBCPolicy,
    denormalize,
    environment_policy_observation,
    normalize,
)


OBSERVATION_DIM = 11
VLM_CONTROLLER_STATE_DIM = 5
RESIDUAL_DIM = 3
RESIDUAL_ACTION_INDICES = (0, 2, 9)
PHYSICAL_RESIDUAL_SCALE = torch.tensor((0.010, 0.012, 0.0025))
LATENT_RESIDUAL_SCALE = torch.tensor((0.30, 0.30, 0.30))

# The VLM recommends a semantic operating interval, while this high-rate
# barrier accounts for one-step PhysX/PD transients around first contact.
VLM_FORCE_HEADROOM_N = 0.10
VLM_FREE_CLOSE_RATE_M_PER_STEP = 1.0e-4
VLM_CONTACT_CLOSE_RATE_M_PER_STEP = 1.5e-5
VLM_TRANSLATION_RATE_M_PER_STEP = 5.0e-4
VLM_CONTACT_ONSET_N = 0.10
VLM_RL_PULL_SCALE_M = 0.008
VLM_RL_START_STEP = 500
VLM_Q_ACTION_CONTEXT_DIM = 4
VLM_Q_ACTION_CONTEXT_COMPONENTS = (
    "vlm_force_low_div_5n",
    "vlm_control_high_div_5n",
    "force_integral_div_5mm",
    "guarded_pull_offset_div_6mm",
)
VLM_NO_GUARD_Q_CONTEXT_DIM = 2
VLM_NO_GUARD_Q_CONTEXT_COMPONENTS = (
    "vlm_force_low_div_5n",
    "vlm_force_high_div_5n",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


@dataclass
class Rollout:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    done: torch.Tensor
    valid: torch.Tensor
    outcomes: list[dict[str, Any]]
    action_context: torch.Tensor | None = None
    next_action_context: torch.Tensor | None = None
    old_log_prob: torch.Tensor | None = None
    diffusion_trace: torch.Tensor | None = None


class Comparison:
    def __init__(self) -> None:
        random.seed(args.train_seed)
        np.random.seed(args.train_seed)
        torch.manual_seed(args.train_seed)
        self.output_root = args.output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.learner_device = torch.device(
            "cuda"
            if args.learner_device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.learner_device == "auto"
            else args.learner_device
        )

        checkpoint_path = args.bc_checkpoint.resolve()
        self.bc_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.policy_horizon = int(self.bc_checkpoint.get("max_episode_steps", 1000))
        self.bc = GrapeBCPolicy(
            hidden_dim=int(self.bc_checkpoint["hidden_dim"]),
            depth=int(self.bc_checkpoint["depth"]),
        )
        self.bc.load_state_dict(self.bc_checkpoint["model_state_dict"])
        self.bc.eval()
        self.obs_mean = self.bc_checkpoint["obs_mean"].float()
        self.obs_std = self.bc_checkpoint["obs_std"].float()
        self.action_mean = self.bc_checkpoint["action_mean"].float()
        self.action_std = self.bc_checkpoint["action_std"].float()
        self.bc_sha256 = _sha256(checkpoint_path)

        cfg = GrapePickEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed = args.train_seed
        cfg.enable_cameras = False
        cfg.enable_gelsight = False
        cfg.randomize_force_thresholds = True
        cfg.damage_force_randomization_range_n = tuple(args.damage_range)
        cfg.detach_force_randomization_range_n = tuple(args.detach_range)
        cfg.episode_length_s = max(args.max_episode_steps * cfg.sim.dt, 1.0)
        if args.device:
            cfg.sim.device = args.device
        self.env = GrapePickEnv(cfg)
        self.policy_device = torch.device(cfg.sim.device)
        self.obs_mean = self.obs_mean.to(self.policy_device)
        self.obs_std = self.obs_std.to(self.policy_device)
        self.action_mean = self.action_mean.to(self.policy_device)
        self.action_std = self.action_std.to(self.policy_device)
        self.bc = self.bc.to(self.policy_device)
        self.agents: dict[str, Any] = {}
        self.training_reports: dict[str, Any] = {}
        self.vlm_record = self._load_vlm_record()

    def _load_vlm_record(self) -> dict[str, Any]:
        if args.vlm_force_range is None:
            return {
                "advisor": "calibrated_fallback",
                "force_range_n": [3.05, 3.30],
                "warning": "No VLM force-range artifact supplied.",
            }
        record = json.loads(args.vlm_force_range.read_text(encoding="utf-8"))
        force_range = record.get("force_range_n")
        if not isinstance(force_range, list) or len(force_range) != 2:
            raise ValueError("VLM force-range artifact must contain force_range_n=[low, high].")
        low, high = map(float, force_range)
        if not (0.0 < low < high < float(args.damage_range[0])):
            raise ValueError(
                f"Unsafe VLM force range {force_range}; expected 0 < low < high < {args.damage_range[0]} N."
            )
        record["force_range_n"] = [low, high]
        record["artifact"] = str(args.vlm_force_range.resolve())
        return record

    def _observation(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw = environment_policy_observation(self.env, max_episode_steps=self.policy_horizon)
        return raw, normalize(raw, self.obs_mean, self.obs_std)

    @torch.inference_mode()
    def _base_action(self, normalized_observation: torch.Tensor) -> torch.Tensor:
        prediction = self.bc(normalized_observation)
        return denormalize(prediction, self.action_mean, self.action_std)

    def _learner_observation_dim(self, method: str) -> int:
        if method == "vlm_dsrl" and args.vlm_controller_mode == "rl_owned_pull":
            return OBSERVATION_DIM + VLM_CONTROLLER_STATE_DIM
        if method == "vlm_dsrl" and args.vlm_controller_mode == "no_guard":
            # Keep the actor input identical to DSRL. The two measured pad
            # forces are already observation entries 6 and 7; duplicating
            # threshold-derived versions changed the actor architecture and
            # confounded the paired comparison.
            return OBSERVATION_DIM
        return OBSERVATION_DIM

    @staticmethod
    def _training_seed_method_index(method: str) -> int:
        """Pair DSRL and guard-free VLM+DSRL on all training randomness."""

        if method == "vlm_dsrl" and args.vlm_controller_mode == "no_guard":
            return METHODS.index("dsrl")
        return METHODS.index(method)

    @staticmethod
    def _critic_action_context_dim(method: str) -> int:
        if method == "vlm_dsrl" and args.vlm_q_force_conditioning:
            if args.vlm_controller_mode == "no_guard":
                return VLM_NO_GUARD_Q_CONTEXT_DIM
            return VLM_Q_ACTION_CONTEXT_DIM
        return 0

    @staticmethod
    def _critic_action_context_components(method: str) -> tuple[str, ...]:
        if method != "vlm_dsrl" or not args.vlm_q_force_conditioning:
            return ()
        if args.vlm_controller_mode == "no_guard":
            return VLM_NO_GUARD_Q_CONTEXT_COMPONENTS
        return VLM_Q_ACTION_CONTEXT_COMPONENTS

    def _critic_action_context(
        self,
        method: str,
        force_integral: torch.Tensor,
        guarded_pull_offset: torch.Tensor,
    ) -> torch.Tensor | None:
        if self._critic_action_context_dim(method) == 0:
            return None
        low, high = self.vlm_record["force_range_n"]
        if args.vlm_controller_mode == "no_guard":
            return torch.stack(
                (
                    torch.full_like(force_integral, low / 5.0),
                    torch.full_like(force_integral, high / 5.0),
                ),
                dim=-1,
            )
        control_high = max(low, high - VLM_FORCE_HEADROOM_N)
        return torch.stack(
            (
                torch.full_like(force_integral, low / 5.0),
                torch.full_like(force_integral, control_high / 5.0),
                force_integral / 0.005,
                guarded_pull_offset / 0.006,
            ),
            dim=-1,
        )

    @staticmethod
    def _residual_dim(method: str) -> int:
        if method in VLM_METHODS and args.vlm_controller_mode == "rl_owned_pull":
            return 1
        return RESIDUAL_DIM

    def _learner_observation(
        self,
        method: str,
        normalized_observation: torch.Tensor,
        force_integral: torch.Tensor,
    ) -> torch.Tensor:
        if method != "vlm_dsrl":
            return normalized_observation
        if args.vlm_controller_mode == "no_guard":
            return normalized_observation
        if args.vlm_controller_mode != "rl_owned_pull":
            return normalized_observation
        position_scale = self.action_std[:3].clamp_min(1.0e-4)
        normalized_last_target = (
            self.env.last_target_pos_b - self.action_mean[:3]
        ) / position_scale
        normalized_width = (
            self.env.gripper_width[:, 0] - self.action_mean[9]
        ) / self.action_std[9].clamp_min(1.0e-4)
        normalized_integral = force_integral / 0.005
        return torch.cat(
            (
                normalized_observation,
                normalized_last_target,
                normalized_width.unsqueeze(-1),
                normalized_integral.unsqueeze(-1),
            ),
            dim=-1,
        )

    def _agent_action(
        self,
        method: str,
        observation: torch.Tensor,
        deterministic: bool,
        exploration_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if method == "base":
            return torch.zeros(
                (len(observation), self._residual_dim(method)), device=self.policy_device
            ), None, None
        if method == "vlm_guard":
            if args.vlm_controller_mode == "rl_owned_pull":
                residual = torch.full(
                    (len(observation), 1),
                    float(args.vlm_guard_pull_residual),
                    dtype=observation.dtype,
                    device=self.policy_device,
                )
            else:
                residual = torch.as_tensor(
                    args.vlm_guard_residual,
                    dtype=observation.dtype,
                    device=self.policy_device,
                ).expand(len(observation), -1)
            return residual, None, None
        agent_name = method
        agent = self.agents[agent_name]
        learner_obs = observation.to(self.learner_device)
        if agent_name == "dppo":
            action, log_prob, trace = agent.act(learner_obs, deterministic)
            return action.to(self.policy_device), log_prob.to(self.policy_device), trace.to(self.policy_device)
        if isinstance(agent, SACResidualAgent):
            action = agent.act(
                learner_obs,
                deterministic,
                exploration_scale=exploration_scale,
            )
        elif isinstance(agent, SafePullBanditAgent):
            action = agent.act(
                learner_obs,
                deterministic,
                exploration_scale=exploration_scale,
            )
        else:
            action = agent.act(learner_obs, deterministic)
        log_prob = None
        if not deterministic and hasattr(agent, "actor"):
            with torch.no_grad():
                log_prob = agent.actor.log_prob(learner_obs, action).to(self.policy_device)
        return action.to(self.policy_device), log_prob, None

    def _apply_residual(
        self,
        base_action: torch.Tensor,
        residual: torch.Tensor,
        method: str,
    ) -> torch.Tensor:
        action = base_action.clone()
        if method in VLM_METHODS and args.vlm_controller_mode == "rl_owned_pull":
            metrics = self.env.get_contact_force_metrics()
            low, high = self.vlm_record["force_range_n"]
            control_high = max(low, high - VLM_FORCE_HEADROOM_N)
            pull_ready = (
                self.env.stem_attached
                & metrics["bilateral_contact"]
                & (self.env.step_count >= VLM_RL_START_STEP)
                & (metrics["grip_force_n"] >= low)
                & (metrics["grip_force_n"] <= control_high)
            )
            pull_offset = residual[:, :1] * VLM_RL_PULL_SCALE_M
            action[pull_ready, :3] += (
                self.env.pull_direction_b[pull_ready] * pull_offset[pull_ready]
            )
            return action
        if method in ("dsrl", "vlm_dsrl", "vlm_guard"):
            scale = self.action_std[list(RESIDUAL_ACTION_INDICES)] * LATENT_RESIDUAL_SCALE.to(
                self.policy_device
            )
        elif method == "base":
            return action
        else:
            scale = PHYSICAL_RESIDUAL_SCALE.to(self.policy_device)
        action[:, list(RESIDUAL_ACTION_INDICES)] += residual * scale
        return action

    def _reward(
        self,
        method: str,
        flags: dict[str, torch.Tensor],
        terminal: torch.Tensor,
    ) -> torch.Tensor:
        force_ratio = flags["grip_force_n"] / self.env.damage_threshold_n.clamp_min(1.0e-4)
        tension_ratio = flags["stem_tension_n"] / self.env.detach_threshold_n.clamp_min(1.0e-4)
        reward = 0.01 * flags["bilateral_contact"].float()
        reward += 0.015 * tension_ratio.clamp(0.0, 1.2)
        reward += 0.03 * flags["detached"].float()
        reward -= 0.005 * force_ratio.square().clamp(max=4.0)
        reward += 8.0 * flags["success"].float()
        reward -= 7.0 * flags["crushed"].float()
        reward -= 4.0 * flags["slipped"].float()
        reward -= 8.0 * flags["obstacle_collision"].float()
        reward -= 4.0 * flags["workspace_failure"].float()
        reward -= 2.0 * (terminal & flags["stem_attached"]).float()
        if method == "vlm_dsrl" and args.vlm_controller_mode == "no_guard":
            low, high = self.vlm_record["force_range_n"]
            grip_force = flags["grip_force_n"]
            bilateral = flags["bilateral_contact"].float()
            in_band = ((grip_force >= low) & (grip_force <= high)).float()
            below = ((low - grip_force) / max(high - low, 1.0e-6)).clamp(0.0, 2.0)
            above = ((grip_force - high) / max(high - low, 1.0e-6)).clamp(0.0, 2.0)
            # VLM supplies semantics for the desirable force interval, but it
            # never edits the command in no_guard mode. The learned policy must
            # discover the residual that reaches and maintains this interval.
            reward += float(args.vlm_force_band_reward) * in_band
            reward -= float(args.vlm_force_violation_penalty) * bilateral * (
                below.square() + 2.0 * above.square()
            )
        return reward

    def _vlm_no_guard_action_prior(
        self,
        observation: torch.Tensor,
        replay_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a VLM semantic policy-improvement target without runtime control.

        The target regularizes only the learned gripper-width residual.  It is
        absolute rather than relative to the replay action: the latter contains
        SAC exploration noise and must not be distilled into the deterministic
        actor. Tool X/Z remain fully governed by SAC. The mask activates only
        after bilateral contact in the grasp/pull portion of the trajectory.
        """

        raw_force = (
            observation[:, 6:8] * self.obs_std[6:8].to(observation.device)
            + self.obs_mean[6:8].to(observation.device)
        )
        raw_step_fraction = (
            observation[:, 10] * self.obs_std[10].to(observation.device)
            + self.obs_mean[10].to(observation.device)
        )
        raw_stem_attached = (
            observation[:, 9] * self.obs_std[9].to(observation.device)
            + self.obs_mean[9].to(observation.device)
        )
        grip_force = raw_force.max(dim=-1).values
        bilateral = raw_force.min(dim=-1).values >= VLM_CONTACT_ONSET_N
        low, high = self.vlm_record["force_range_n"]
        # The frozen BC policy already closes the gripper.  Asking the residual
        # actor to close further below the VLM lower bound caused first-contact
        # overshoot in the original symmetric-error prior.  Instead, increase
        # the learned opening residual as the measured force consumes the VLM
        # safe interval, with extra pressure above the upper threshold.  This is
        # a training target only; rollout always executes the actor output.
        interval = max(high - low, 1.0e-6)
        consumed_margin = ((grip_force - low) / interval).clamp(0.0, 1.5)
        target = torch.zeros_like(replay_action)
        target[:, 2] = (
            float(args.vlm_action_prior_gain) * consumed_margin
        ).clamp(0.0, 1.0)
        active = bilateral & (raw_step_fraction >= 0.35) & (raw_stem_attached >= 0.5)
        mask = torch.zeros_like(target)
        mask[:, 2] = active.float()
        return target, mask

    @staticmethod
    def _outcome(flags: dict[str, torch.Tensor], index: int) -> str:
        if bool(flags["success"][index].item()):
            return "success"
        for name in ("crushed", "slipped", "obstacle_collision", "workspace_failure"):
            if bool(flags[name][index].item()):
                return name
        if bool(flags["stem_attached"][index].item()):
            return "no_detach"
        return "timeout_after_detach"

    def rollout(
        self,
        method: str,
        seed: int,
        *,
        deterministic: bool,
        exploration_scale: float = 1.0,
    ) -> Rollout:
        self.env.reset(seed=seed)
        num_envs = self.env.num_envs
        active = torch.ones(num_envs, dtype=torch.bool, device=self.policy_device)
        previous_action = None
        held_residual = torch.zeros((num_envs, RESIDUAL_DIM), device=self.policy_device)
        episode_pull_residual: torch.Tensor | None = None
        force_integral = torch.zeros(num_envs, device=self.policy_device)
        guarded_pull_offset = torch.zeros(num_envs, device=self.policy_device)
        guarded_pull_started = torch.zeros(
            num_envs, dtype=torch.bool, device=self.policy_device
        )
        pending = None
        accumulated_reward = torch.zeros(num_envs, device=self.policy_device)
        done_since_decision = torch.zeros(num_envs, dtype=torch.bool, device=self.policy_device)
        outcomes: dict[int, dict[str, Any]] = {}
        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        next_observations: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        valids: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        traces: list[torch.Tensor] = []
        action_contexts: list[torch.Tensor] = []
        next_action_contexts: list[torch.Tensor] = []
        residual_sum = torch.zeros(
            (num_envs, self._residual_dim(method)), device=self.policy_device
        )
        residual_abs_sum = torch.zeros_like(residual_sum)
        residual_decisions = torch.zeros(num_envs, device=self.policy_device)

        def close_pending(
            next_normalized_observation: torch.Tensor,
            next_action_context: torch.Tensor | None,
        ) -> None:
            nonlocal pending, accumulated_reward, done_since_decision
            if pending is None:
                return
            observations.append(pending["observation"].detach().cpu())
            actions.append(pending["action"].detach().cpu())
            rewards.append(accumulated_reward.detach().cpu())
            next_observations.append(next_normalized_observation.detach().cpu())
            dones.append(done_since_decision.float().detach().cpu())
            valids.append(pending["valid"].detach().cpu())
            if pending["log_prob"] is not None:
                log_probs.append(pending["log_prob"].detach().cpu())
            if pending["trace"] is not None:
                traces.append(pending["trace"].detach().cpu())
            if pending["action_context"] is not None:
                if next_action_context is None:
                    raise RuntimeError("Missing next VLM critic action context.")
                action_contexts.append(pending["action_context"].detach().cpu())
                next_action_contexts.append(next_action_context.detach().cpu())
            pending = None
            accumulated_reward = torch.zeros(num_envs, device=self.policy_device)
            done_since_decision = torch.zeros(
                num_envs, dtype=torch.bool, device=self.policy_device
            )

        for step in range(args.max_episode_steps):
            _, normalized_observation = self._observation()
            learner_observation = self._learner_observation(
                method, normalized_observation, force_integral
            )
            critic_action_context = self._critic_action_context(
                method, force_integral, guarded_pull_offset
            )
            if step % args.action_repeat == 0:
                close_pending(learner_observation, critic_action_context)
                if not bool(active.any().item()):
                    break
                use_episode_pull = (
                    method == "vlm_dsrl"
                    and args.vlm_controller_mode == "rl_owned_pull"
                    and args.vlm_rl_algorithm == "safe_pull_bandit"
                )
                if use_episode_pull and episode_pull_residual is not None:
                    held_residual = episode_pull_residual
                    log_prob = None
                    trace = None
                else:
                    held_residual, log_prob, trace = self._agent_action(
                        method,
                        learner_observation,
                        deterministic,
                        exploration_scale,
                    )
                    if use_episode_pull:
                        episode_pull_residual = held_residual
                pending = {
                    "observation": learner_observation,
                    "action": held_residual,
                    "valid": active.clone(),
                    "log_prob": log_prob,
                    "trace": trace,
                    "action_context": critic_action_context,
                }
                residual_sum += held_residual * active.unsqueeze(-1)
                residual_abs_sum += held_residual.abs() * active.unsqueeze(-1)
                residual_decisions += active.float()

            base_action = self._base_action(normalized_observation)
            applied_residual = held_residual
            if method == "vlm_dsrl" and args.vlm_controller_mode == "legacy_guarded_pull":
                # The VLM force supervisor owns the pull coordinate; retaining
                # a separately learned Z residual can produce impulsive target
                # changes when the contact state switches.
                applied_residual = held_residual.clone()
                applied_residual[:, 1] = 0.0
            action = self._apply_residual(base_action, applied_residual, method)

            if method in VLM_METHODS and args.vlm_controller_mode != "no_guard":
                metrics = self.env.get_contact_force_metrics()
                grip_force = metrics["grip_force_n"]
                bilateral = metrics["bilateral_contact"]
                low, high = self.vlm_record["force_range_n"]
                control_high = max(low, high - VLM_FORCE_HEADROOM_N)
                target = 0.5 * (low + control_high)
                force_integral += torch.where(
                    bilateral,
                    1.0e-4 * (grip_force - target),
                    torch.where(
                        (self.env.step_count >= 480) & self.env.stem_attached,
                        torch.full_like(force_integral, -1.0e-5),
                        torch.zeros_like(force_integral),
                    ),
                )
                force_integral.clamp_(-0.0008, 0.0050)
                action[:, 9] += force_integral
                if args.vlm_controller_mode == "legacy_guarded_pull":
                    guarded_pull = (
                        self.env.stem_attached
                        & bilateral
                        & (self.env.step_count >= 610)
                        & (grip_force >= low)
                        & (grip_force <= control_high)
                    )
                    guarded_pull_started |= guarded_pull
                    guarded_pull_offset += guarded_pull.float() * 2.0e-5
                    guarded_pull_offset -= (grip_force > control_high).float() * 5.0e-4
                    guarded_pull_offset.clamp_(0.0, 0.006)
                    # This branch is retained only for exact reproduction of
                    # the original experiment. New causal ablations use
                    # rl_owned_pull, in which the learner owns pull control.
                    apply_pull = self.env.stem_attached & guarded_pull_started
                    action[apply_pull, :3] += (
                        self.env.pull_direction_b[apply_pull]
                        * guarded_pull_offset[apply_pull].unsqueeze(-1)
                    )

            if (
                previous_action is not None
                and not (
                    method in VLM_METHODS
                    and args.vlm_controller_mode == "rl_owned_pull"
                )
            ):
                alpha = float(args.action_smoothing)
                action = alpha * action + (1.0 - alpha) * previous_action
            if method in VLM_METHODS and args.vlm_controller_mode != "no_guard":
                # Bound absolute target motion after smoothing.  The policy
                # emits pose and width targets, so action smoothing alone does
                # not limit the physical increment when a contact-state
                # transition changes the network prediction discontinuously.
                last_target = self.env.last_target_pos_b
                translation_delta = (action[:, :3] - last_target).clamp(
                    -VLM_TRANSLATION_RATE_M_PER_STEP,
                    VLM_TRANSLATION_RATE_M_PER_STEP,
                )
                action[:, :3] = last_target + translation_delta

                current_width = self.env.gripper_width[:, 0]
                force_margin = (control_high - grip_force).clamp_min(0.0)
                close_rate = torch.where(
                    grip_force >= VLM_CONTACT_ONSET_N,
                    torch.full_like(grip_force, VLM_CONTACT_CLOSE_RATE_M_PER_STEP),
                    torch.full_like(grip_force, VLM_FREE_CLOSE_RATE_M_PER_STEP),
                )
                # Taper closing continuously to zero over the last 0.5 N of
                # force margin.  This prevents first-contact target jumps from
                # crossing the minimum randomized 3.4 N damage threshold.
                close_rate *= (force_margin / 0.50).clamp(0.0, 1.0)
                action[:, 9] = torch.maximum(action[:, 9], current_width - close_rate)

                # Apply the instantaneous safety opening after smoothing and
                # rate limiting so a stale closing command cannot mask it.
                safe_open_width = self.env.gripper_width[:, 0] + 2.0e-3 * (
                    grip_force - control_high
                ).clamp_min(0.0)
                over_high = grip_force > control_high
                action[:, 9] = torch.where(
                    over_high,
                    torch.maximum(action[:, 9], safe_open_width),
                    action[:, 9],
                )
            previous_action = action.detach()

            active_before = active.clone()
            self.env._pre_physics_step(action)
            self.env._apply_action()
            self.env.scene.write_data_to_sim()
            self.env.sim.step(render=False)
            self.env.scene.update(dt=self.env.physics_dt)
            flags = self.env.get_task_flags()
            terminal = (
                flags["success"]
                | flags["crushed"]
                | flags["slipped"]
                | flags["obstacle_collision"]
                | flags["workspace_failure"]
            )
            if step == args.max_episode_steps - 1:
                terminal |= active
            just_finished = active & terminal
            accumulated_reward += self._reward(method, flags, terminal) * active_before.float()
            done_since_decision |= just_finished
            obstacle_details = (
                self.env.get_obstacle_contact_force_details()
                if bool((just_finished & flags["obstacle_collision"]).any().item())
                else {}
            )
            for index in torch.nonzero(just_finished, as_tuple=False).squeeze(-1).cpu().tolist():
                outcome = self._outcome(flags, index)
                record = {
                    "seed": seed + index,
                    "outcome": outcome,
                    "steps": step + 1,
                    "peak_grip_force_n": float(self.env.peak_grip_force_n[index].item()),
                    "peak_stem_tension_n": float(self.env.peak_stem_tension_n[index].item()),
                    "peak_obstacle_contact_force_n": float(
                        self.env.peak_obstacle_contact_force_n[index].item()
                    ),
                    "damage_threshold_n": float(self.env.damage_threshold_n[index].item()),
                    "detach_threshold_n": float(self.env.detach_threshold_n[index].item()),
                    "mean_residual_action": (
                        residual_sum[index]
                        / residual_decisions[index].clamp_min(1.0)
                    ).detach().cpu().tolist(),
                    "mean_abs_residual_action": (
                        residual_abs_sum[index]
                        / residual_decisions[index].clamp_min(1.0)
                    ).detach().cpu().tolist(),
                }
                if outcome == "obstacle_collision":
                    record["obstacle_contact_details_n"] = {
                        name: float(value[index].item())
                        for name, value in obstacle_details.items()
                        if float(value[index].item()) >= self.env.cfg.obstacle_contact_threshold_n
                    }
                outcomes[index] = record
            active &= ~just_finished

        _, final_observation = self._observation()
        final_learner_observation = self._learner_observation(
            method, final_observation, force_integral
        )
        close_pending(
            final_learner_observation,
            self._critic_action_context(method, force_integral, guarded_pull_offset),
        )
        if len(outcomes) != num_envs:
            raise RuntimeError(f"Only {len(outcomes)}/{num_envs} environments reached an outcome.")
        return Rollout(
            observation=torch.stack(observations),
            action=torch.stack(actions),
            reward=torch.stack(rewards),
            next_observation=torch.stack(next_observations),
            done=torch.stack(dones),
            valid=torch.stack(valids),
            outcomes=[outcomes[index] for index in sorted(outcomes)],
            action_context=torch.stack(action_contexts) if action_contexts else None,
            next_action_context=(
                torch.stack(next_action_contexts) if next_action_contexts else None
            ),
            old_log_prob=torch.stack(log_probs) if log_probs else None,
            diffusion_trace=torch.stack(traces) if traces else None,
        )

    def _summarize(self, method: str, outcomes: list[dict[str, Any]], seed: int) -> dict[str, Any]:
        counts = Counter(record["outcome"] for record in outcomes)
        successes = counts.get("success", 0)
        peak_force = np.asarray([record["peak_grip_force_n"] for record in outcomes])
        peak_obstacle = np.asarray(
            [record["peak_obstacle_contact_force_n"] for record in outcomes]
        )
        report = {
            "task": "TacEx-GrapePick-Force-v0",
            "method": method,
            "seed_start": seed,
            "episodes": len(outcomes),
            "successes": successes,
            "success_rate": successes / max(len(outcomes), 1),
            "wilson_95_ci": _wilson(successes, len(outcomes)),
            "failure_counts": {key: value for key, value in counts.items() if key != "success"},
            "peak_grip_force_n": {
                "mean": float(peak_force.mean()),
                "max": float(peak_force.max()),
            },
            "peak_obstacle_contact_force_n": float(peak_obstacle.max()),
            "damage_force_randomization_range_n": list(args.damage_range),
            "detach_force_randomization_range_n": list(args.detach_range),
            "bc_checkpoint_sha256": self.bc_sha256,
            "tactile_observation_indices": [6, 7],
            "tactile_observation_fields": [
                "left_pad_normal_force_n",
                "right_pad_normal_force_n",
            ],
            "outcomes": outcomes,
        }
        if method in VLM_METHODS:
            report["vlm_force_advisor"] = self.vlm_record
            low, high = self.vlm_record["force_range_n"]
            report["vlm_tactile_safety_controller"] = {
                "advisor_range_n": [low, high],
                "control_high_n": max(low, high - VLM_FORCE_HEADROOM_N),
                "force_headroom_n": VLM_FORCE_HEADROOM_N,
                "free_close_rate_m_per_step": VLM_FREE_CLOSE_RATE_M_PER_STEP,
                "contact_close_rate_m_per_step": VLM_CONTACT_CLOSE_RATE_M_PER_STEP,
                "translation_rate_m_per_step": VLM_TRANSLATION_RATE_M_PER_STEP,
                "pull_requires_force_at_least_n": low,
                "controller_mode": args.vlm_controller_mode,
                "fixed_guard_residual": list(args.vlm_guard_residual)
                if method == "vlm_guard"
                else None,
                "fixed_guard_pull_residual": args.vlm_guard_pull_residual
                if method == "vlm_guard"
                else None,
                "runtime_guard_enabled": args.vlm_controller_mode != "no_guard",
            }
            report["vlm_q_action_encoding"] = {
                "enabled": bool(args.vlm_q_force_conditioning and method == "vlm_dsrl"),
                "components": list(self._critic_action_context_components(method)),
                "note": (
                    "The critic receives the effective learned residual plus the applicable "
                    "VLM force thresholds; guarded modes additionally include live "
                    "controller-owned action terms."
                ),
            }
        return report

    def train(self, method: str) -> None:
        if method in ("base", "vlm_guard"):
            return
        seed_method_index = self._training_seed_method_index(method)
        torch.manual_seed(args.train_seed + seed_method_index * 1000)
        learner_observation_dim = self._learner_observation_dim(method)
        residual_dim = self._residual_dim(method)
        corrected_vlm = method == "vlm_dsrl" and args.vlm_controller_mode == "rl_owned_pull"
        safe_bandit = corrected_vlm and args.vlm_rl_algorithm == "safe_pull_bandit"
        critic_action_context_dim = self._critic_action_context_dim(method)
        if safe_bandit:
            agent = SafePullBanditAgent(self.learner_device)
            replay = None
        elif method in ("residual_rl", "dsrl", "vlm_dsrl"):
            agent: Any = SACResidualAgent(
                learner_observation_dim,
                residual_dim,
                self.learner_device,
                learning_rate=3.0e-4,
                gamma=0.999,
                residual_l2_weight=0.1 if corrected_vlm else 0.5,
                initial_log_std=-1.0 if corrected_vlm else -2.0,
                initial_alpha=0.02 if corrected_vlm else 0.05,
                target_entropy=-0.5 if corrected_vlm else None,
                critic_action_context_dim=critic_action_context_dim,
                critic_context_mode=(
                    "bias_adapter"
                    if method == "vlm_dsrl"
                    and args.vlm_controller_mode == "no_guard"
                    and critic_action_context_dim > 0
                    else "concat"
                ),
                executed_action_mask=(
                    (1.0, 0.0, 1.0)
                    if method == "vlm_dsrl"
                    and args.vlm_controller_mode == "legacy_guarded_pull"
                    else None
                ),
            )
            replay = ReplayBuffer(
                300_000,
                learner_observation_dim,
                residual_dim,
                self.learner_device,
                action_context_dim=critic_action_context_dim,
            )
        elif method == "awr":
            agent = AWRAgent(OBSERVATION_DIM, RESIDUAL_DIM, self.learner_device)
            replay = None
        elif method == "dppo":
            agent = DiffusionPPOAgent(OBSERVATION_DIM, RESIDUAL_DIM, self.learner_device)
            replay = None
        else:
            raise ValueError(method)
        self.agents[method] = agent

        def validate_candidate(stage: str) -> tuple[dict[str, Any], float]:
            validation_outcomes: list[dict[str, Any]] = []
            batches = math.ceil(args.validation_episodes / args.num_envs)
            for batch in range(batches):
                validation_outcomes.extend(
                    self.rollout(
                        method,
                        args.validation_seed + batch * args.num_envs,
                        deterministic=True,
                    ).outcomes
                )
            validation_outcomes = validation_outcomes[: args.validation_episodes]
            validation_report = self._summarize(
                method, validation_outcomes, args.validation_seed
            )
            failures = validation_report["failure_counts"]
            score = float(validation_report["success_rate"])
            score -= 0.5 * failures.get("crushed", 0) / max(args.validation_episodes, 1)
            score -= failures.get("obstacle_collision", 0) / max(args.validation_episodes, 1)
            validation_report["selection_stage"] = stage
            validation_report["selection_score"] = score
            return validation_report, score

        selection_history: list[dict[str, Any]] = []
        initial_validation, initial_score = validate_candidate("initial_untrained_policy")
        selection_history.append(initial_validation)
        if args.require_trained_checkpoint:
            best_score = -math.inf
            best_checkpoint = None
            best_stage = None
        else:
            best_score = initial_score
            best_checkpoint = copy.deepcopy(agent.checkpoint())
            best_stage = "initial_untrained_policy"

        all_outcomes: list[dict[str, Any]] = []
        iterations = math.ceil(args.train_episodes / args.num_envs)
        diagnostics: list[dict[str, Any]] = []
        sac_final_exploration_scale = (
            args.sac_exploration_scale
            if args.sac_exploration_final_scale is None
            else args.sac_exploration_final_scale
        )
        for iteration in range(iterations):
            seed = args.train_seed + seed_method_index * 10_000 + iteration * args.num_envs
            if safe_bandit and iterations > 1:
                exploration_scale = args.bandit_epsilon_start + (
                    args.bandit_epsilon_end - args.bandit_epsilon_start
                ) * iteration / (iterations - 1)
            elif safe_bandit:
                exploration_scale = args.bandit_epsilon_end
            else:
                exploration_scale = exploration_scale_schedule(
                    args.sac_exploration_scale,
                    sac_final_exploration_scale,
                    iteration,
                    iterations,
                    args.sac_exploration_schedule,
                )
            rollout = self.rollout(
                method,
                seed,
                deterministic=False,
                exploration_scale=exploration_scale,
            )
            all_outcomes.extend(rollout.outcomes)
            mask = rollout.valid.bool()
            if safe_bandit:
                outcome_reward = []
                for outcome in rollout.outcomes:
                    label = outcome["outcome"]
                    terminal_score = (
                        1.0
                        if label == "success"
                        else 0.25
                        if label == "timeout_after_detach"
                        else -1.0
                        if label in ("crushed", "slipped", "obstacle_collision", "workspace_failure")
                        else 0.0
                    )
                    tension_ratio = min(
                        outcome["peak_stem_tension_n"]
                        / max(outcome["detach_threshold_n"], 1.0e-6),
                        1.1,
                    )
                    outcome_reward.append(terminal_score + 0.10 * tension_ratio)
                episode_return = torch.tensor(
                    outcome_reward, dtype=torch.float32, device=self.learner_device
                )
                first_action = rollout.action[0].to(self.learner_device)
                last = agent.update(first_action, episode_return)
                diagnostics.append(
                    {
                        "iteration": iteration + 1,
                        "behavior_epsilon": exploration_scale,
                        **last,
                    }
                )
            elif method in ("residual_rl", "dsrl", "vlm_dsrl"):
                assert replay is not None
                if critic_action_context_dim > 0 and (
                    rollout.action_context is None
                    or rollout.next_action_context is None
                ):
                    raise RuntimeError("VLM force-conditioned rollout omitted Q action context.")
                replay.add(
                    SACBatch(
                        rollout.observation[mask],
                        rollout.action[mask],
                        (rollout.reward[mask] / 8.0).clamp(-1.0, 1.0),
                        rollout.next_observation[mask],
                        rollout.done[mask],
                        (
                            rollout.action_context[mask]
                            if rollout.action_context is not None
                            else None
                        ),
                        (
                            rollout.next_action_context[mask]
                            if rollout.next_action_context is not None
                            else None
                        ),
                    )
                )
                last = {}
                if replay.size >= 1024:
                    for _ in range(args.sac_updates_per_iteration):
                        sampled_batch = replay.sample(512)
                        if (
                            method == "vlm_dsrl"
                            and args.vlm_controller_mode == "no_guard"
                            and args.vlm_action_prior_weight > 0.0
                        ):
                            action_prior, action_prior_mask = (
                                self._vlm_no_guard_action_prior(
                                    sampled_batch.observation,
                                    sampled_batch.action,
                                )
                            )
                            last = agent.update(
                                sampled_batch,
                                action_prior=action_prior,
                                action_prior_mask=action_prior_mask,
                                action_prior_weight=args.vlm_action_prior_weight,
                            )
                        else:
                            last = agent.update(sampled_batch)
                diagnostics.append(
                    {
                        "iteration": iteration + 1,
                        "replay_size": replay.size,
                        "behavior_exploration_scale": exploration_scale,
                        **last,
                    }
                )
            else:
                returns = discounted_returns(rollout.reward, rollout.done, rollout.valid, gamma=0.999)
                flat_observation = rollout.observation[mask]
                flat_returns = returns[mask]
                if method == "awr":
                    last = agent.update(
                        flat_observation,
                        rollout.action[mask],
                        flat_returns,
                        epochs=6,
                    )
                else:
                    assert rollout.old_log_prob is not None
                    assert rollout.diffusion_trace is not None
                    with torch.no_grad():
                        values = agent.value(flat_observation.to(self.learner_device)).cpu()
                    last = agent.update(
                        flat_observation,
                        rollout.diffusion_trace[mask],
                        rollout.old_log_prob[mask],
                        flat_returns,
                        flat_returns - values,
                        epochs=4,
                    )
                diagnostics.append({"iteration": iteration + 1, **last})
            validation_report, score = validate_candidate(f"iteration_{iteration + 1}")
            selection_history.append(validation_report)
            improved = score > best_score + 1.0e-9
            # VLM safety supervision can make the untrained initialization
            # perfect on a small validation set.  In that tie, retain the
            # first genuinely updated DSRL checkpoint rather than presenting
            # an untrained residual network as VLM+DSRL.
            first_trained_vlm_tie = (
                not args.require_trained_checkpoint
                and method == "vlm_dsrl"
                and best_stage == "initial_untrained_policy"
                and score >= best_score - 1.0e-9
            )
            if improved or first_trained_vlm_tie:
                best_score = score
                best_checkpoint = copy.deepcopy(agent.checkpoint())
                best_stage = f"iteration_{iteration + 1}"
            partial = self._summarize(method, all_outcomes, args.train_seed)
            print(
                f"[TRAIN] method={method} iteration={iteration + 1}/{iterations} "
                f"episodes={len(all_outcomes)} success={partial['success_rate']:.1%} "
                f"validation={validation_report['success_rate']:.1%} best={best_stage}",
                flush=True,
            )

        if best_checkpoint is None or best_stage is None:
            raise RuntimeError("No trained checkpoint was eligible for selection.")
        validation_gain = best_score - initial_score
        if (
            args.min_validation_gain is not None
            and validation_gain + 1.0e-9 < args.min_validation_gain
        ):
            raise RuntimeError(
                f"Best trained {method} validation score improved by only "
                f"{validation_gain:.3f}; required {args.min_validation_gain:.3f}."
            )
        checkpoint = {
            **best_checkpoint,
            "method": method,
            "observation_dim": learner_observation_dim,
            "residual_dim": residual_dim,
            "tactile_observation_indices": [6, 7],
            "bc_checkpoint_sha256": self.bc_sha256,
            "train_seed": args.train_seed,
            "training_seed_method_index": seed_method_index,
            "paired_with_dsrl_training_randomness": bool(
                method == "vlm_dsrl" and args.vlm_controller_mode == "no_guard"
            ),
            "train_episodes": len(all_outcomes),
            "selection_seed": args.validation_seed,
            "selection_episodes": args.validation_episodes,
            "selected_stage": best_stage,
            "selected_score": best_score,
            "require_trained_checkpoint": args.require_trained_checkpoint,
            "vlm_controller_mode": args.vlm_controller_mode,
            "initial_validation_score": initial_score,
            "validation_gain": validation_gain,
            "sac_exploration_scale": args.sac_exploration_scale,
            "sac_exploration_scale_start": args.sac_exploration_scale,
            "sac_exploration_scale_end": sac_final_exploration_scale,
            "sac_exploration_schedule": args.sac_exploration_schedule,
            "sac_exploration_scale_history": [
                float(item.get("behavior_exploration_scale"))
                for item in diagnostics
                if item.get("behavior_exploration_scale") is not None
            ],
            "vlm_rl_algorithm": args.vlm_rl_algorithm,
            "vlm_q_force_conditioning": bool(
                method == "vlm_dsrl" and args.vlm_q_force_conditioning
            ),
            "critic_action_context_dim": critic_action_context_dim,
            "critic_action_context_components": list(
                self._critic_action_context_components(method)
            ),
            "vlm_runtime_guard_enabled": bool(
                method == "vlm_dsrl" and args.vlm_controller_mode != "no_guard"
            ),
            "vlm_force_band_reward": args.vlm_force_band_reward
            if method == "vlm_dsrl" and args.vlm_controller_mode == "no_guard"
            else None,
            "vlm_force_violation_penalty": args.vlm_force_violation_penalty
            if method == "vlm_dsrl" and args.vlm_controller_mode == "no_guard"
            else None,
            "vlm_action_prior_weight": args.vlm_action_prior_weight
            if method == "vlm_dsrl" and args.vlm_controller_mode == "no_guard"
            else None,
            "vlm_action_prior_gain": args.vlm_action_prior_gain
            if method == "vlm_dsrl" and args.vlm_controller_mode == "no_guard"
            else None,
            "bandit_epsilon_start": args.bandit_epsilon_start if safe_bandit else None,
            "bandit_epsilon_end": args.bandit_epsilon_end if safe_bandit else None,
            "vlm_rl_pull_scale_m": VLM_RL_PULL_SCALE_M
            if method == "vlm_dsrl" and args.vlm_controller_mode == "rl_owned_pull"
            else None,
            "vlm_rl_start_step": VLM_RL_START_STEP
            if method == "vlm_dsrl" and args.vlm_controller_mode == "rl_owned_pull"
            else None,
        }
        torch.save(checkpoint, self.checkpoint_dir / f"{method}.pt")
        report = self._summarize(method, all_outcomes, args.train_seed)
        report["diagnostics"] = diagnostics
        report["selection_history"] = selection_history
        report["selected_stage"] = best_stage
        report["selected_score"] = best_score
        report["initial_validation_score"] = initial_score
        report["validation_gain"] = validation_gain
        report["vlm_controller_mode"] = args.vlm_controller_mode
        report["vlm_rl_algorithm"] = args.vlm_rl_algorithm
        report["vlm_q_force_conditioning"] = bool(
            method == "vlm_dsrl" and args.vlm_q_force_conditioning
        )
        report["vlm_runtime_guard_enabled"] = bool(
            method == "vlm_dsrl" and args.vlm_controller_mode != "no_guard"
        )
        report["vlm_force_band_reward"] = (
            args.vlm_force_band_reward
            if method == "vlm_dsrl" and args.vlm_controller_mode == "no_guard"
            else None
        )
        report["vlm_force_violation_penalty"] = (
            args.vlm_force_violation_penalty
            if method == "vlm_dsrl" and args.vlm_controller_mode == "no_guard"
            else None
        )
        report["vlm_action_prior_weight"] = (
            args.vlm_action_prior_weight
            if method == "vlm_dsrl" and args.vlm_controller_mode == "no_guard"
            else None
        )
        report["vlm_action_prior_gain"] = (
            args.vlm_action_prior_gain
            if method == "vlm_dsrl" and args.vlm_controller_mode == "no_guard"
            else None
        )
        report["sac_exploration_scale"] = args.sac_exploration_scale
        report["sac_exploration_scale_start"] = args.sac_exploration_scale
        report["sac_exploration_scale_end"] = sac_final_exploration_scale
        report["sac_exploration_schedule"] = args.sac_exploration_schedule
        report["sac_exploration_scale_history"] = [
            float(item.get("behavior_exploration_scale"))
            for item in diagnostics
            if item.get("behavior_exploration_scale") is not None
        ]
        report["training_seed_method_index"] = seed_method_index
        report["paired_with_dsrl_training_randomness"] = bool(
            method == "vlm_dsrl" and args.vlm_controller_mode == "no_guard"
        )
        report["require_trained_checkpoint"] = args.require_trained_checkpoint
        report["checkpoint"] = str((self.checkpoint_dir / f"{method}.pt").resolve())
        _write_json(self.output_root / "training" / f"{method}.json", report)
        self.training_reports[method] = report

    def evaluate(self, method: str, seed: int) -> dict[str, Any]:
        outcomes: list[dict[str, Any]] = []
        batches = math.ceil(args.eval_episodes / args.num_envs)
        for batch in range(batches):
            rollout = self.rollout(
                method,
                seed + batch * args.num_envs,
                deterministic=True,
            )
            outcomes.extend(rollout.outcomes)
            outcomes = outcomes[: args.eval_episodes]
            print(
                f"[EVAL] method={method} seed={seed} episodes={len(outcomes)}/{args.eval_episodes} "
                f"success={sum(x['outcome'] == 'success' for x in outcomes) / len(outcomes):.1%}",
                flush=True,
            )
        report = self._summarize(method, outcomes, seed)
        _write_json(self.output_root / "evaluation" / method / f"seed_{seed}.json", report)
        return report

    def load_agents(self) -> None:
        for method in ("residual_rl", "awr", "dppo", "dsrl", "vlm_dsrl"):
            if method not in args.methods:
                continue
            path = self.checkpoint_dir / f"{method}.pt"
            checkpoint = torch.load(path, map_location=self.learner_device, weights_only=False)
            if checkpoint.get("bc_checkpoint_sha256") != self.bc_sha256:
                raise ValueError(f"{path} was trained against a different BC checkpoint.")
            checkpoint_mode = checkpoint.get("vlm_controller_mode", "legacy_guarded_pull")
            if method == "vlm_dsrl" and checkpoint_mode != args.vlm_controller_mode:
                raise ValueError(
                    f"{path} uses VLM controller mode {checkpoint_mode!r}, "
                    f"not requested mode {args.vlm_controller_mode!r}."
                )
            checkpoint_force_conditioning = bool(
                checkpoint.get("vlm_q_force_conditioning", False)
            )
            if method == "vlm_dsrl" and checkpoint_force_conditioning != bool(
                args.vlm_q_force_conditioning
            ):
                raise ValueError(
                    f"{path} force-conditioning setting is "
                    f"{checkpoint_force_conditioning}, but the requested setting is "
                    f"{args.vlm_q_force_conditioning}."
                )
            if method in ("residual_rl", "dsrl", "vlm_dsrl"):
                observation_dim = int(checkpoint.get("observation_dim", OBSERVATION_DIM))
                residual_dim = int(checkpoint.get("residual_dim", RESIDUAL_DIM))
                if checkpoint.get("algorithm") == "safe_pull_bandit_q_learning":
                    agent = SafePullBanditAgent(self.learner_device)
                    agent.load_checkpoint(checkpoint)
                else:
                    agent = SACResidualAgent(
                        observation_dim,
                        residual_dim,
                        self.learner_device,
                        critic_action_context_dim=int(
                            checkpoint.get("critic_action_context_dim", 0)
                        ),
                        critic_context_mode=str(
                            checkpoint.get(
                                "critic_context_mode",
                                "bias_adapter"
                                if method == "vlm_dsrl"
                                and checkpoint_mode == "no_guard"
                                else "concat",
                            )
                        ),
                        executed_action_mask=tuple(
                            float(value)
                            for value in checkpoint.get(
                                "executed_action_mask",
                                torch.ones(residual_dim),
                            )
                        ),
                    )
                    agent.actor.load_state_dict(checkpoint["actor"])
                    agent.q1.load_state_dict(checkpoint["q1"])
                    agent.q2.load_state_dict(checkpoint["q2"])
            elif method == "awr":
                agent = AWRAgent(OBSERVATION_DIM, RESIDUAL_DIM, self.learner_device)
                agent.actor.load_state_dict(checkpoint["actor"])
                agent.value.load_state_dict(checkpoint["value"])
            else:
                agent = DiffusionPPOAgent(OBSERVATION_DIM, RESIDUAL_DIM, self.learner_device)
                agent.policy.load_state_dict(checkpoint["policy"])
                agent.value.load_state_dict(checkpoint["value"])
            self.agents[method] = agent

    def write_summary(self, reports: list[dict[str, Any]], wall_time_s: float) -> None:
        pooled = []
        for method in args.methods:
            selected = [report for report in reports if report["method"] == method]
            episodes = sum(int(report["episodes"]) for report in selected)
            successes = sum(int(report["successes"]) for report in selected)
            failure_counts: Counter[str] = Counter()
            for report in selected:
                failure_counts.update(report["failure_counts"])
            pooled.append(
                {
                    "method": method,
                    "episodes": episodes,
                    "successes": successes,
                    "success_rate": successes / max(episodes, 1),
                    "wilson_95_ci": _wilson(successes, episodes),
                    "failure_counts": dict(failure_counts),
                }
            )
        summary = {
            "task": "TacEx-GrapePick-Force-v0",
            "protocol": "paired independent evaluation seeds",
            "bc_checkpoint": str(args.bc_checkpoint.resolve()),
            "bc_checkpoint_sha256": self.bc_sha256,
            "train_episodes_per_method": args.train_episodes,
            "require_trained_checkpoint": args.require_trained_checkpoint,
            "eval_episodes_per_repeat": args.eval_episodes,
            "eval_seeds": list(args.eval_seeds),
            "damage_force_randomization_range_n": list(args.damage_range),
            "detach_force_randomization_range_n": list(args.detach_range),
            "tactile_conditioning": {
                "fields": ["left_pad_normal_force_n", "right_pad_normal_force_n"],
                "observation_indices": [6, 7],
            },
            "vlm_force_advisor": self.vlm_record,
            "vlm_q_force_conditioning": bool(args.vlm_q_force_conditioning),
            "vlm_q_action_context_components": list(
                self._critic_action_context_components("vlm_dsrl")
            ),
            "vlm_runtime_guard_enabled": args.vlm_controller_mode != "no_guard",
            "wall_time_seconds": wall_time_s,
            "methods": pooled,
        }
        _write_json(self.output_root / "comparison_summary.json", summary)
        with (self.output_root / "comparison_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("method", "episodes", "successes", "success_rate", "failure_counts"),
            )
            writer.writeheader()
            for row in pooled:
                writer.writerow(
                    {
                        **{key: row[key] for key in ("method", "episodes", "successes", "success_rate")},
                        "failure_counts": json.dumps(row["failure_counts"], sort_keys=True),
                    }
                )
        print("[SUMMARY] " + json.dumps(summary, ensure_ascii=False), flush=True)

    def run(self) -> None:
        start = time.monotonic()
        if not args.skip_training:
            for method in args.methods:
                self.train(method)
        if args.skip_evaluation:
            return
        self.load_agents()
        reports = []
        for seed in args.eval_seeds:
            for method in args.methods:
                reports.append(self.evaluate(method, seed))
        self.write_summary(reports, time.monotonic() - start)

    def close(self) -> None:
        self.env.close()


def main() -> None:
    comparison = Comparison()
    try:
        comparison.run()
    finally:
        comparison.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
