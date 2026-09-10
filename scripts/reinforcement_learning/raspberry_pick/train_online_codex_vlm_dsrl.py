#!/usr/bin/env python3
"""Post-train RaspberryPick DINOv3 Flow BC with DSRL and online Codex VLM force inference.

The frozen policy and its visual-XY head remain unchanged. SAC learns a bounded
correction to selected channels of the zero initial Flow latent. After every
episode, a schema-constrained Codex vision call reads a source-grounded force
and pedicle-tension diagnostic image and proposes the next safe force interval.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
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
BC_TRAINING_DIR = ROOT / "scripts" / "bc_training"
GRAPE_HELPERS = ROOT / "scripts" / "reinforcement_learning" / "grape_pick"
for helper_dir in (BC_TRAINING_DIR, GRAPE_HELPERS):
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))

DEFAULT_CHECKPOINT = (
    ROOT
    / "exp_report"
    / "raspberry_pick"
    / "bc"
    / "dinov3_flow"
    / "checkpoints"
    / "raspberry_dinov3_flow_bc.pt"
)
DEFAULT_SCHEMA = Path(__file__).with_name("codex_vlm_force_schema.json")
FORMAL_BASE_CHECKPOINT_SHA256 = (
    "f752dcd5ffddbf1b7f55559cb51d5c88233467ccf365c4d4824e95b565af6eed"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
parser.add_argument(
    "--protocol-name",
    default="raspberry_0909_v1",
    help="Protocol identifier persisted in manifests and checkpoints.",
)
parser.add_argument(
    "--agent-checkpoint",
    type=Path,
    default=None,
    help="Load a trained residual SAC checkpoint before running episodes.",
)
parser.add_argument(
    "--evaluation-mode",
    action="store_true",
    help="Use the loaded residual actor deterministically without replay updates.",
)
parser.add_argument(
    "--allow-dsrl-checkpoint-for-controller-only-oracle",
    action="store_true",
    help=(
        "In evaluation mode only, load the exact dsrl_only actor into oracle_force_dsrl "
        "so the paired ablation changes only the force controller."
    ),
)
parser.add_argument(
    "--branch-from-dsrl-checkpoint",
    action="store_true",
    help=(
        "Resume replay, optimizers and actor from a DSRL checkpoint, then branch into "
        "oracle_force_dsrl at --resume-episode."
    ),
)
parser.add_argument(
    "--resume-episode",
    type=int,
    default=0,
    help="Zero-based first episode to execute when branching from a saved training history.",
)
parser.add_argument("--output-root", type=Path, default=None)
parser.add_argument(
    "--resume",
    action="store_true",
    help=(
        "Resume an interrupted run in --output-root from its newest periodic checkpoint. "
        "Completed episodes and advisor calls are not repeated."
    ),
)
parser.add_argument("--train-episodes", type=int, default=100)
parser.add_argument(
    "--episodes-this-invocation",
    type=int,
    default=None,
    help="Run at most this many new episodes, leaving a resumable incomplete run.",
)
parser.add_argument("--train-seed", type=int, default=1_010_000)
parser.add_argument(
    "--fixed-episode-seed",
    action="store_true",
    help="Reset every episode to --train-seed for a single-scene overfit/replay experiment.",
)
parser.add_argument("--max-episode-steps", type=int, default=900)
parser.add_argument("--action-repeat", type=int, default=4)
parser.add_argument(
    "--chunk-execute-steps",
    type=int,
    default=1,
    help=(
        "Number of consecutive actions to execute from each predicted Flow chunk before "
        "running policy inference again; observation history still refreshes per action."
    ),
)
parser.add_argument("--action-smoothing", type=float, default=0.35)
parser.add_argument("--num-inference-steps", type=int, default=20)
parser.add_argument("--position-range", type=float, nargs=3, default=(0.008, 0.010, 0.012))
parser.add_argument("--damage-threshold-n", type=float, default=3.2)
parser.add_argument("--detach-threshold-n", type=float, default=0.8)
parser.add_argument("--position-command-offset-m", type=float, nargs=3, default=(0.010, 0.0, 0.0))
parser.add_argument("--gripper-command-offset-m", type=float, default=-0.001)
parser.add_argument("--pull-z-offset-m", type=float, default=-0.020)
parser.add_argument("--pull-z-start-phase", type=float, default=0.72)
parser.add_argument("--pull-z-ramp-end-phase", type=float, default=0.88)
parser.add_argument("--noise-action-indices", type=int, nargs="+", default=(2, 9))
parser.add_argument("--noise-residual-scale", type=float, default=0.35)
parser.add_argument("--learned-noise-steps", type=int, default=1)
parser.add_argument("--dsrl-start-step", type=int, default=360)
parser.add_argument("--initial-log-std", type=float, default=-2.5)
parser.add_argument("--learning-starts", type=int, default=256)
parser.add_argument("--replay-capacity", type=int, default=50_000)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--updates-per-decision", type=int, default=1)
parser.add_argument("--learning-rate", type=float, default=3.0e-4)
parser.add_argument(
    "--freeze-dsrl",
    action="store_true",
    help="Use the exact zero-residual frozen BC policy (for force/profile calibration).",
)
parser.add_argument("--exploration-start", type=float, default=1.0)
parser.add_argument("--exploration-end", type=float, default=0.15)
parser.add_argument(
    "--exploration-anneal-episodes",
    type=int,
    default=30,
    help="Anneal exploration over this many episodes, then hold exploration-end.",
)
parser.add_argument(
    "--bottle-style-lateral-dsrl",
    action="store_true",
    help=(
        "Use the SealTwist-style contact-gated 1-D physical closing-axis residual "
        "instead of the legacy Z/gripper Flow-latent residual."
    ),
)
parser.add_argument("--lateral-residual-scale-m", type=float, default=0.006)
parser.add_argument("--force-balance-prior-gain", type=float, default=0.25)
parser.add_argument("--force-balance-prior-weight", type=float, default=2.0)
parser.add_argument("--force-balance-prior-limit", type=float, default=0.20)
parser.add_argument("--force-balance-prior-smoothing", type=float, default=0.25)
parser.add_argument("--initial-force-range", type=float, nargs=2, default=(2.20, 2.80))
parser.add_argument("--physical-force-bounds", type=float, nargs=2, default=(1.20, 3.10))
parser.add_argument(
    "--force-range-source",
    choices=("fixed", "vlm", "oracle"),
    default=None,
    help="Explicit force-range provenance; required by the raspberry_0909 formal protocol.",
)
parser.add_argument(
    "--controller-mode",
    choices=("single_stage", "two_stage"),
    default=None,
    help="Explicit force-controller topology; required by the raspberry_0909 formal protocol.",
)
parser.add_argument("--public-grip-safety-limit-n", type=float, default=3.10)
parser.add_argument("--public-tension-target-n", type=float, default=0.90)
parser.add_argument(
    "--allow-force-bound-above-damage-threshold",
    action="store_true",
    help=(
        "Allow the controller's admissible force bound to exceed the fixed damage threshold. "
        "This is only for threshold-ablation baselines where over-force failures must remain possible."
    ),
)
parser.add_argument("--force-range-minimum-width", type=float, default=0.30)
parser.add_argument(
    "--oracle-force-width-n",
    type=float,
    default=0.60,
    help="Width of the privileged force interval used only by oracle_force_dsrl.",
)
parser.add_argument(
    "--oracle-damage-margin-n",
    type=float,
    default=0.20,
    help=(
        "Margin below the true damage threshold used by oracle_force_dsrl, either "
        "for the fixed interval upper edge or for the per-step safety shield."
    ),
)
parser.add_argument(
    "--oracle-control-mode",
    choices=("fixed_interval", "safety_shield", "tension_assist", "two_stage"),
    default="fixed_interval",
    help=(
        "Use either the legacy fixed ground-truth force interval or preserve the DSRL "
        "force target and intervene using the true damage threshold or late stem tension."
    ),
)
parser.add_argument(
    "--oracle-tension-assist-start-phase",
    type=float,
    default=0.80,
    help="Earliest normalized phase for privileged low-tension pull assistance.",
)
parser.add_argument(
    "--oracle-tension-assist-m",
    type=float,
    default=0.010,
    help="Additional downward pull applied only by the ground-truth tension assistant.",
)
parser.add_argument(
    "--oracle-two-stage-contact-confirm-steps",
    type=int,
    default=3,
    help="Consecutive bilateral-force samples required before two-stage stabilization.",
)
parser.add_argument(
    "--oracle-two-stage-grip-stable-steps",
    type=int,
    default=6,
    help="In-band grip-force samples required before tension-regulated pulling.",
)
parser.add_argument(
    "--oracle-two-stage-tension-kp",
    type=float,
    default=0.012,
    help="Two-stage pull velocity gain in m/s/N.",
)
parser.add_argument(
    "--oracle-two-stage-max-pull-velocity-m-s",
    type=float,
    default=0.010,
    help="Maximum two-stage pulling velocity.",
)
parser.add_argument(
    "--oracle-force-balance-alignment",
    action="store_true",
    help=(
        "Use unilateral ground-truth pad forces to center the gripper along its "
        "closing axis before the two-stage grip/tension controller takes over."
    ),
)
parser.add_argument(
    "--oracle-force-balance-velocity-m-s",
    type=float,
    default=0.030,
    help="Maximum force-balance alignment velocity.",
)
parser.add_argument(
    "--oracle-force-balance-max-offset-m",
    type=float,
    default=0.012,
    help="Maximum absolute closing-axis alignment offset.",
)
parser.add_argument(
    "--oracle-start-episode",
    type=int,
    default=0,
    help=(
        "Zero-based episode at which oracle_force_dsrl switches from the baseline force "
        "range to its ground-truth range. Use a positive value for a shared DSRL warm-up."
    ),
)
parser.add_argument(
    "--force-range-smoothing",
    type=float,
    default=1.0,
    help="EMA applied after the multi-episode evidence guard; v2 uses the guarded step directly.",
)
parser.add_argument("--vlm-max-center-step-n", type=float, default=0.05)
parser.add_argument("--guard-confirmations", type=int, default=2)
parser.add_argument(
    "--method",
    choices=(
        "dsrl_only",
        "vlm_two_stage_dsrl",
        "oracle_force_dsrl",
        "dsrl_semantic_v2",
        "dsrl_semantic_v3",
        "dsrl_semantic_xy_v4",
        "dsrl_semantic_xy_v5",
        "vlm_noop",
        "vlm_force_v2",
        "vlm_semantic_v2",
        "vlm_semantic_v3",
        "vlm_semantic_xy_v4",
        "vlm_semantic_xy_v5",
    ),
    default=None,
    help="Repaired paired-experiment arm; --disable-vlm remains a compatibility alias.",
)
parser.add_argument(
    "--berry-profile",
    choices=("fixed", "delicate", "firm"),
    default="fixed",
    help="Persistent hidden berry/pedicle profile for the complete run.",
)
parser.add_argument(
    "--semantic-translation-scale-m", type=float, nargs=3, default=(0.006, 0.006, 0.008)
)
parser.add_argument("--semantic-gripper-scale-m", type=float, default=0.0015)
parser.add_argument("--checkpoint-every", type=int, default=10)
parser.add_argument("--codex-cli", type=Path, default=None)
parser.add_argument("--codex-model", default="gpt-5.6-sol")
parser.add_argument(
    "--codex-reasoning-effort",
    choices=("low", "medium", "high", "xhigh"),
    default="high",
)
parser.add_argument("--codex-output-schema", type=Path, default=DEFAULT_SCHEMA)
parser.add_argument("--codex-timeout-s", type=float, default=300.0)
parser.add_argument("--codex-max-attempts", type=int, default=3)
parser.add_argument(
    "--vlm-call-policy",
    choices=("every_episode", "failure_only"),
    default="every_episode",
    help="Call the VLM after every episode or only after failed episodes.",
)
parser.add_argument("--disable-vlm", action="store_true")
parser.add_argument("--physx-contact-capacity", type=int, default=2**20)
parser.add_argument(
    "--enhanced-determinism",
    action="store_true",
    help="Enable PhysX enhanced determinism for paired single-seed ablations.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output_root is None:
    parser.error("--output-root is required")
if args.method is None:
    args.method = "dsrl_only" if args.disable_vlm else "vlm_force_v2"
if args.force_range_source is None:
    args.force_range_source = (
        "oracle"
        if args.method == "oracle_force_dsrl"
        else "fixed"
        if args.method in {
            "dsrl_only",
            "dsrl_semantic_v2",
            "dsrl_semantic_v3",
            "dsrl_semantic_xy_v4",
            "dsrl_semantic_xy_v5",
        }
        else "vlm"
    )
if args.controller_mode is None:
    args.controller_mode = (
        "two_stage"
        if args.method == "vlm_two_stage_dsrl"
        or (args.method == "oracle_force_dsrl" and args.oracle_control_mode == "two_stage")
        else "single_stage"
    )
if min(
    args.train_episodes,
    args.action_repeat,
    args.chunk_execute_steps,
    args.num_inference_steps,
) < 1:
    parser.error("episode, action-repeat, chunk and inference-step counts must be positive")
if not args.protocol_name.strip():
    parser.error("--protocol-name must not be empty")
if args.episodes_this_invocation is not None and args.episodes_this_invocation < 1:
    parser.error("--episodes-this-invocation must be positive")
if args.exploration_anneal_episodes < 1:
    parser.error("--exploration-anneal-episodes must be positive")
if min(
    args.lateral_residual_scale_m,
    args.force_balance_prior_gain,
    args.force_balance_prior_weight,
    args.force_balance_prior_limit,
) <= 0.0:
    parser.error("bottle-style DSRL scales, gain, weight and limit must be positive")
if not 0.0 <= args.force_balance_prior_smoothing <= 1.0:
    parser.error("--force-balance-prior-smoothing must be in [0, 1]")
if not 0 <= args.dsrl_start_step < args.max_episode_steps:
    parser.error("dsrl-start-step must lie inside the episode")
if not 0.0 <= args.pull_z_start_phase < args.pull_z_ramp_end_phase <= 1.0:
    parser.error("pull-Z phases must satisfy 0 <= start < end <= 1")
if len(set(args.noise_action_indices)) != len(args.noise_action_indices):
    parser.error("noise-action-indices must be unique")
if min(args.noise_action_indices) < 0 or max(args.noise_action_indices) >= 10:
    parser.error("noise-action-indices must be in [0, 10)")
if args.noise_residual_scale < 0.0 or args.learned_noise_steps < 1:
    parser.error("DSRL latent dimensions and scale are invalid")
if not args.physical_force_bounds[0] < args.physical_force_bounds[1]:
    parser.error("physical force bounds must be strictly ordered")
if not (
    args.physical_force_bounds[0]
    < args.public_grip_safety_limit_n
    <= args.physical_force_bounds[1]
):
    parser.error("public grip safety limit must lie inside the physical force bounds")
if args.public_tension_target_n <= 0.0:
    parser.error("public tension target must be positive")
if (
    args.physical_force_bounds[1] >= args.damage_threshold_n
    and not args.allow_force_bound_above_damage_threshold
):
    parser.error(
        "physical force upper bound must be below the damage threshold unless the explicit "
        "threshold-ablation override is enabled"
    )
if args.codex_model != "gpt-5.6-sol" or args.codex_reasoning_effort != "high":
    parser.error("the formal experiment contract requires gpt-5.6-sol at high reasoning")
if args.guard_confirmations < 1:
    parser.error("guard-confirmations must be positive")
if min(args.semantic_translation_scale_m) < 0.0 or args.semantic_gripper_scale_m < 0.0:
    parser.error("semantic residual scales must be non-negative")
if args.oracle_force_width_n <= 0.0 or args.oracle_damage_margin_n <= 0.0:
    parser.error("oracle force width and damage margin must be positive")
if not 0.0 <= args.oracle_tension_assist_start_phase <= 1.0:
    parser.error("oracle-tension-assist-start-phase must lie in [0, 1]")
if args.oracle_tension_assist_m < 0.0:
    parser.error("oracle-tension-assist-m must be non-negative")
if min(
    args.oracle_two_stage_contact_confirm_steps,
    args.oracle_two_stage_grip_stable_steps,
) < 1:
    parser.error("oracle two-stage dwell counts must be positive")
if min(
    args.oracle_two_stage_tension_kp,
    args.oracle_two_stage_max_pull_velocity_m_s,
) <= 0.0:
    parser.error("oracle two-stage pull gains must be positive")
if min(
    args.oracle_force_balance_velocity_m_s,
    args.oracle_force_balance_max_offset_m,
) <= 0.0:
    parser.error("oracle force-balance limits must be positive")
if args.oracle_force_balance_alignment and args.oracle_control_mode != "two_stage":
    parser.error("oracle force-balance alignment requires --oracle-control-mode two_stage")
if not 0 <= args.oracle_start_episode < args.train_episodes:
    parser.error("oracle-start-episode must lie in [0, train-episodes)")
if args.method != "oracle_force_dsrl" and args.oracle_start_episode != 0:
    parser.error("oracle-start-episode is only valid with --method oracle_force_dsrl")
if args.method != "oracle_force_dsrl" and args.oracle_control_mode != "fixed_interval":
    parser.error("oracle-control-mode is only valid with --method oracle_force_dsrl")
if args.bottle_style_lateral_dsrl and args.method not in {
    "dsrl_only",
    "vlm_two_stage_dsrl",
    "oracle_force_dsrl",
}:
    parser.error(
        "--bottle-style-lateral-dsrl is supported by dsrl_only, "
        "vlm_two_stage_dsrl and oracle_force_dsrl"
    )
if args.controller_mode == "two_stage" and not args.bottle_style_lateral_dsrl:
    parser.error("--controller-mode two_stage requires --bottle-style-lateral-dsrl")
if args.method == "dsrl_only" and (
    args.force_range_source != "fixed" or args.controller_mode != "single_stage"
):
    parser.error("dsrl_only requires fixed force range and single-stage control")
if args.method == "vlm_two_stage_dsrl" and (
    args.force_range_source != "vlm" or args.controller_mode != "two_stage"
):
    parser.error("vlm_two_stage_dsrl requires VLM force range and two-stage control")
if args.method == "oracle_force_dsrl" and args.force_range_source != "oracle":
    parser.error("oracle_force_dsrl requires --force-range-source oracle")
if args.force_range_source == "oracle" and args.method != "oracle_force_dsrl":
    parser.error("oracle force-range source is exclusive to oracle_force_dsrl")
if args.disable_vlm and args.method != "dsrl_only":
    parser.error("--disable-vlm cannot be combined with a VLM method")
if args.resume and (
    args.agent_checkpoint is not None
    or args.branch_from_dsrl_checkpoint
    or args.evaluation_mode
):
    parser.error("--resume cannot be combined with checkpoint branching or evaluation")
if args.evaluation_mode and args.agent_checkpoint is None:
    parser.error("--evaluation-mode requires --agent-checkpoint")
if args.allow_dsrl_checkpoint_for_controller_only_oracle and (
    not args.evaluation_mode
    or args.agent_checkpoint is None
    or args.method != "oracle_force_dsrl"
):
    parser.error(
        "--allow-dsrl-checkpoint-for-controller-only-oracle requires "
        "--evaluation-mode, --agent-checkpoint and --method oracle_force_dsrl"
    )
if args.branch_from_dsrl_checkpoint and (
    args.evaluation_mode
    or args.agent_checkpoint is None
    or args.method != "oracle_force_dsrl"
    or not 0 < args.resume_episode < args.train_episodes
    or args.oracle_start_episode != args.resume_episode
):
    parser.error(
        "--branch-from-dsrl-checkpoint requires --agent-checkpoint, "
        "--method oracle_force_dsrl, 0 < --resume-episode < --train-episodes, "
        "and --oracle-start-episode equal to --resume-episode"
    )
if args.resume_episode != 0 and not args.branch_from_dsrl_checkpoint:
    parser.error("--resume-episode requires --branch-from-dsrl-checkpoint")
args.enable_cameras = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import dinov3_flow as flow
import train_bc as common
from grape_force_control import (
    ForceControllerConfig,
    ForceRange,
    ForceRangeAdapterConfig,
    GrapeForceController,
    OnlineForceRangeAdapter,
)
from learning import ReplayBuffer, SACBatch, SACResidualAgent
from two_stage_force_control import (
    ForceBalanceAligner,
    ForceBalanceAlignmentConfig,
    TwoStageBerryForceController,
    TwoStageForceControllerConfig,
)
from online_rl_support import (
    BottleStyleLateralResidualLayout,
    RASPBERRY_PROFILES,
    GuardedForceProposal,
    SemanticResidualLayout,
    SemanticXYResidualLayout,
    environment_shaped_reward,
    exploration_scale_for_episode,
    force_balance_prior_target,
    guard_force_proposal,
    residual_is_active,
    should_call_vlm,
    theoretical_oracle_force_range,
    terminal_outcome_reward,
    validate_agent_checkpoint_contract,
    validate_training_branch_checkpoint,
)
from tacex_tasks.raspberry_pick.raspberry_pick_env import RaspberryPickEnv
from tacex_tasks.raspberry_pick.raspberry_pick_env_cfg import RaspberryPickEnvCfg


METHOD_CONTRACTS = {
    "dsrl_only": "raspberry_dinov3_flow_bc_zero_latent_dsrl_sac_fixed_force_v2",
    "vlm_two_stage_dsrl": (
        "raspberry_dinov3_flow_bc_bottle_lateral_dsrl_public_two_stage_vlm_v1"
    ),
    "oracle_force_dsrl": (
        "raspberry_dinov3_flow_bc_zero_latent_dsrl_sac_controller_only_oracle_force_v2"
    ),
    "dsrl_semantic_v2": "raspberry_dinov3_flow_bc_semantic_dsrl_sac_fixed_force_v2",
    "dsrl_semantic_v3": "raspberry_dinov3_flow_bc_semantic_dsrl_sac_prepull_fixed_force_v3",
    "dsrl_semantic_xy_v4": "raspberry_dinov3_flow_bc_semantic_xy_dsrl_sac_prepull_fixed_force_v4",
    "dsrl_semantic_xy_v5": "raspberry_dinov3_flow_bc_semantic_xy_dsrl_sac_prepull_delayed_credit_fixed_force_v5",
    "vlm_noop": "raspberry_dinov3_flow_bc_zero_latent_dsrl_sac_vlm_noop_v2",
    "vlm_force_v2": "raspberry_dinov3_flow_bc_zero_latent_dsrl_sac_guarded_vlm_force_v2",
    "vlm_semantic_v2": "raspberry_dinov3_flow_bc_semantic_dsrl_sac_guarded_vlm_force_v2",
    "vlm_semantic_v3": "raspberry_dinov3_flow_bc_semantic_dsrl_sac_prepull_guarded_vlm_force_v3",
    "vlm_semantic_xy_v4": "raspberry_dinov3_flow_bc_semantic_xy_dsrl_sac_prepull_guarded_vlm_force_v4",
    "vlm_semantic_xy_v5": "raspberry_dinov3_flow_bc_semantic_xy_dsrl_sac_prepull_delayed_credit_guarded_vlm_force_v5",
}
CONTRACT = METHOD_CONTRACTS[args.method]
CONTRACT_VERSION = int(CONTRACT.rsplit("_v", 1)[-1])
SEMANTIC_METHODS = {
    "dsrl_semantic_v2",
    "dsrl_semantic_v3",
    "dsrl_semantic_xy_v4",
    "dsrl_semantic_xy_v5",
    "vlm_semantic_v2",
    "vlm_semantic_v3",
    "vlm_semantic_xy_v4",
    "vlm_semantic_xy_v5",
}
PREPULL_SEMANTIC_METHODS = {
    "dsrl_semantic_v3",
    "dsrl_semantic_xy_v4",
    "dsrl_semantic_xy_v5",
    "vlm_semantic_v3",
    "vlm_semantic_xy_v4",
    "vlm_semantic_xy_v5",
}
SEMANTIC_XY_METHODS = {
    "dsrl_semantic_xy_v4",
    "dsrl_semantic_xy_v5",
    "vlm_semantic_xy_v4",
    "vlm_semantic_xy_v5",
}
DELAYED_CREDIT_METHODS = {"dsrl_semantic_xy_v5", "vlm_semantic_xy_v5"}
NO_VLM_METHODS = {
    "dsrl_only",
    "oracle_force_dsrl",
    "dsrl_semantic_v2",
    "dsrl_semantic_v3",
    "dsrl_semantic_xy_v4",
    "dsrl_semantic_xy_v5",
}
BASE_CONTRACT = "raspberry_dinov3_tactile_flow_matching_bc_v1"
TASK = "TacEx-RaspberryPick-Force-v0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_revision() -> dict[str, Any]:
    """Return immutable provenance for the exact source tree used by a run."""

    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ("git", "diff", "--binary", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "git_commit": commit,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "worktree_dirty": bool(diff),
        "trainer_sha256": sha256(Path(__file__).resolve()),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def replace_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@dataclass(frozen=True)
class FlowNoiseDSRLLayout:
    horizon: int
    action_dim: int
    learned_noise_steps: int
    residual_scale: float
    active_action_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if min(self.horizon, self.action_dim, self.learned_noise_steps) < 1:
            raise ValueError("Flow latent dimensions must be positive")
        if self.learned_noise_steps > self.horizon:
            raise ValueError("learned-noise-steps exceeds the Flow horizon")
        if not self.active_action_indices:
            raise ValueError("at least one Flow channel must be active")
        if min(self.active_action_indices) < 0 or max(self.active_action_indices) >= self.action_dim:
            raise ValueError("active Flow channel is outside the action dimension")

    @property
    def correction_dim(self) -> int:
        return self.learned_noise_steps * len(self.active_action_indices)

    def compose_zero_anchored_noise(self, correction: torch.Tensor) -> torch.Tensor:
        if correction.ndim != 2 or correction.shape[-1] != self.correction_dim:
            raise ValueError(
                f"correction must have shape [B,{self.correction_dim}], got {tuple(correction.shape)}"
            )
        learned = correction.reshape(
            len(correction), self.learned_noise_steps, len(self.active_action_indices)
        )
        short = correction.new_zeros((len(correction), self.learned_noise_steps, self.action_dim))
        short[:, :, list(self.active_action_indices)] = learned
        if self.learned_noise_steps < self.horizon:
            short = torch.cat(
                (short, short[:, -1:, :].expand(-1, self.horizon - self.learned_noise_steps, -1)),
                dim=1,
            )
        return self.residual_scale * short


class CodexVisualForceAdvisor:
    """Make one fresh, schema-constrained Codex vision call per episode."""

    def __init__(self, output_root: Path) -> None:
        executable = str(args.codex_cli) if args.codex_cli is not None else shutil.which("codex")
        if not executable:
            raise FileNotFoundError("Codex CLI was not found")
        self.executable = str(Path(executable).expanduser().resolve())
        self.schema = args.codex_output_schema.expanduser().resolve()
        if not self.schema.is_file():
            raise FileNotFoundError(f"Codex output schema not found: {self.schema}")
        self.output_dir = output_root / "codex_vlm"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = output_root / "codex_vlm_calls.jsonl"
        self.calls = 0
        self.successful_calls = 0
        version = subprocess.run(
            [self.executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.version = version.stdout.strip()

    @staticmethod
    def _prompt(request: dict[str, Any]) -> str:
        return (
            "You are the visual-tactile force advisor in an online robotic raspberry-picking "
            "experiment. Inspect the attached composite containing the wrist RGB view, the "
            "third-person RGB view, and source-grounded force/tension traces, together with the "
            "structured episode data. Recommend the safe bilateral pad-force interval for the next episode. "
            "The fruit properties persist across episodes. A crushed outcome or force near the "
            "safety envelope supports lowering the interval. A force change is valid only when "
            "the structured bilateral-force evidence supports that direction across consecutive "
            "episodes. Low pedicle tension by itself, no contact, alignment failure, bilateral "
            "imbalance, or a successful episode is not evidence for a force change; hold the "
            "interval in those cases because DSRL must correct the trajectory. Keep "
            "changes conservative, judge force separately from alignment, and keep both endpoints "
            "inside physical_force_bounds_n. Do not use tools or inspect repository files. Return "
            "only schema-conforming JSON.\n\n"
            + json.dumps(request, ensure_ascii=False, sort_keys=True)
        )

    def recommend(
        self,
        *,
        episode_index: int,
        image_paths: list[Path],
        request: dict[str, Any],
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        prompt = self._prompt(request)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        image_sha256 = {str(path.resolve()): sha256(path) for path in image_paths}
        for attempt in range(1, args.codex_max_attempts + 1):
            stem = f"episode_{episode_index:03d}_attempt_{attempt:02d}"
            response_path = self.output_dir / f"{stem}_response.json"
            log_path = self.output_dir / f"{stem}_cli.log"
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--ignore-rules",
                "--color",
                "never",
                "-m",
                args.codex_model,
                "-c",
                f'model_reasoning_effort="{args.codex_reasoning_effort}"',
            ]
            for image_path in image_paths:
                command.extend(("--image", str(image_path.resolve())))
            command.extend(
                (
                    "--output-schema",
                    str(self.schema),
                    "-o",
                    str(response_path),
                    "-",
                )
            )
            self.calls += 1
            started = time.time()
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=args.codex_timeout_s,
                )
                combined = completed.stdout + "\n" + completed.stderr
                log_path.write_text(combined, encoding="utf-8")
                if completed.returncode != 0:
                    raise RuntimeError(f"Codex VLM exited with code {completed.returncode}")
                if f"model: {args.codex_model}" not in combined:
                    raise RuntimeError("Codex audit did not confirm the requested model")
                if f"reasoning effort: {args.codex_reasoning_effort}" not in combined:
                    raise RuntimeError("Codex audit did not confirm the requested reasoning effort")
                response = json.loads(response_path.read_text(encoding="utf-8"))
                raw_response = response_path.read_text(encoding="utf-8")
                low, high = map(float, response["force_range_n"])
                physical_low, physical_high = map(float, request["physical_force_bounds_n"])
                if not physical_low <= low < high <= physical_high:
                    raise ValueError("Codex force interval violates the physical bounds")
                audit = {
                    "advisor_call_index": self.calls,
                    "episode_index": episode_index,
                    "attempt": attempt,
                    "model": args.codex_model,
                    "reasoning_effort": args.codex_reasoning_effort,
                    "codex_cli_version": self.version,
                    "images": [str(path.resolve()) for path in image_paths],
                    "image_sha256": image_sha256,
                    "prompt_sha256": prompt_sha256,
                    "request": request,
                    "response": response,
                    "raw_response": raw_response,
                    "elapsed_seconds": time.time() - started,
                    "success": True,
                    "model_audit_confirmed": True,
                    "reasoning_effort_audit_confirmed": True,
                    "cli_log": str(log_path.resolve()),
                }
                append_jsonl(self.audit_path, audit)
                self.successful_calls += 1
                return response
            except Exception as exc:
                last_error = exc
                append_jsonl(
                    self.audit_path,
                    {
                        "advisor_call_index": self.calls,
                        "episode_index": episode_index,
                        "attempt": attempt,
                        "model": args.codex_model,
                        "reasoning_effort": args.codex_reasoning_effort,
                        "images": [str(path.resolve()) for path in image_paths],
                        "image_sha256": image_sha256,
                        "prompt_sha256": prompt_sha256,
                        "request": request,
                        "elapsed_seconds": time.time() - started,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "cli_log": str(log_path.resolve()),
                    },
                )
        raise RuntimeError(f"Codex VLM failed after {args.codex_max_attempts} attempts") from last_error


def classify(flags: dict[str, torch.Tensor]) -> str:
    if bool(flags["success"][0].item()):
        return "success"
    for name in ("crushed", "slipped", "obstacle_collision", "workspace_failure"):
        if bool(flags[name][0].item()):
            return name
    if bool(flags["stem_attached"][0].item()):
        return "no_detach"
    return "timeout_after_detach"


def is_terminal(flags: dict[str, torch.Tensor], physics_step: int) -> bool:
    terminal = (
        flags["success"]
        | flags["crushed"]
        | flags["slipped"]
        | flags["obstacle_collision"]
        | flags["workspace_failure"]
    )
    return bool(terminal[0].item()) or physics_step >= args.max_episode_steps


def camera_rgb_224(env: RaspberryPickEnv, camera: str) -> torch.Tensor:
    image = env.get_camera_rgb(camera)[..., :3].permute(0, 3, 1, 2).float()
    image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
    return image.clamp(0.0, 255.0).div_(255.0)


def policy_state(env: RaspberryPickEnv) -> torch.Tensor:
    cafe = env.get_cafe_observation()["robot0_pos"]
    metrics = env.get_contact_force_metrics()
    pad_force = torch.stack(
        (metrics["left_normal_force_n"], metrics["right_normal_force_n"]), dim=-1
    )
    return torch.cat((cafe, pad_force), dim=-1)


def load_policy(
    checkpoint_path: Path, device: torch.device
) -> tuple[flow.DINOv3FlowPolicy, common.LinearNormalizer, dict[str, Any]]:
    checkpoint = common.load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_type") != flow.MODEL_TYPE:
        raise ValueError(f"unexpected model type: {checkpoint.get('model_type')!r}")
    if checkpoint.get("contract") != BASE_CONTRACT or checkpoint.get("task") != TASK:
        raise ValueError("checkpoint contract/task does not match RaspberryPick")
    config = flow.DINOv3FlowConfig.from_dict(checkpoint["policy_config"])
    dino, _ = common.load_dino_for_config(Path(config.dino_path) if config.dino_path else None)
    model = flow.DINOv3FlowPolicy(config, dino)
    flow.load_policy_state_dict(model, checkpoint["model"])
    model.to(device).requires_grad_(False).eval()
    normalizer = common.LinearNormalizer()
    normalizer.load_state_dict(checkpoint["normalizer"])
    normalizer.stats["robot0_pos"] = normalizer.stats[common.STATE_KEY]
    return model, normalizer, checkpoint


def model_observation(
    state_history: deque[torch.Tensor],
    image_history: deque[torch.Tensor],
    normalizer: common.LinearNormalizer,
    phase: float,
) -> dict[str, torch.Tensor]:
    state = normalizer.normalize_tensor(common.STATE_KEY, torch.stack(tuple(state_history), dim=1))
    return {
        common.STATE_KEY: state,
        "images": torch.stack(tuple(image_history), dim=1),
        "phase": torch.full((len(state), 1), phase, dtype=state.dtype, device=state.device),
    }


@torch.inference_mode()
def decode_from_condition(
    model: flow.DINOv3FlowPolicy,
    cond_tokens: torch.Tensor,
    initial_noise: torch.Tensor,
) -> torch.Tensor:
    action = initial_noise.clone()
    steps = int(args.num_inference_steps)
    dt = 1.0 / steps
    for index in range(steps):
        t0 = torch.full((len(action),), index / steps, device=action.device, dtype=action.dtype)
        velocity = model._velocity(action, t0, cond_tokens)
        if model.config.ode_solver == "heun" and index < steps - 1:
            proposal = action + dt * velocity
            t1 = torch.full(
                (len(action),), (index + 1) / steps, device=action.device, dtype=action.dtype
            )
            action = action + 0.5 * dt * (velocity + model._velocity(proposal, t1, cond_tokens))
        else:
            action = action + dt * velocity
        if model.config.clip_sample:
            action.clamp_(-1.0, 1.0)
    return action


def actor_observation(
    embedding: torch.Tensor,
    *,
    visual_xy: torch.Tensor,
    normalized_state: torch.Tensor,
    explicit_spatial_features: bool,
    phase: float,
    metrics: dict[str, torch.Tensor],
    flags: dict[str, torch.Tensor],
    force_range: ForceRange,
    bottle_style_force_features: bool = False,
) -> torch.Tensor:
    if visual_xy.ndim != 2 or visual_xy.shape[-1] != 2:
        raise ValueError("visual_xy must have shape [B, 2]")
    if normalized_state.ndim != 2 or normalized_state.shape[-1] != 12:
        raise ValueError("normalized_state must have shape [B, 12]")
    # Formal actors receive only public normalization constants and measurable
    # signals.  Environment damage/detachment thresholds remain unavailable.
    scale = float(args.public_grip_safety_limit_n)
    grip = metrics["grip_force_n"].reshape(-1, 1)
    scalars = torch.cat(
        (
            torch.full_like(grip, phase),
            grip / scale,
            metrics["left_normal_force_n"].reshape(-1, 1) / scale,
            metrics["right_normal_force_n"].reshape(-1, 1) / scale,
            flags["stem_tension_n"].reshape(-1, 1)
            / float(args.public_tension_target_n),
            metrics["bilateral_contact"].float().reshape(-1, 1),
            torch.full_like(grip, force_range.minimum_n / scale),
            torch.full_like(grip, force_range.maximum_n / scale),
            (force_range.center_n - grip) / scale,
        ),
        dim=-1,
    )
    # Mean-pooling the condition tokens is intentionally retained for broad
    # scene context, but it is not a sufficient statistic for spatial
    # residual control: averaging largely discards token position and dilutes
    # the robot-state tokens among the image tokens.  The frozen BC model's
    # supervised XY prediction and its normalized current state are therefore
    # exposed explicitly to both DSRL arms.  These are deployment observations,
    # not simulator-only privileged state.
    pieces = [embedding]
    if explicit_spatial_features:
        pieces.extend((visual_xy, normalized_state))
    pieces.append(scalars)
    if bottle_style_force_features:
        left = metrics["left_normal_force_n"].reshape(-1, 1)
        right = metrics["right_normal_force_n"].reshape(-1, 1)
        center = torch.full_like(left, force_range.center_n)
        pieces.append(
            torch.cat(
                (
                    (center - left) / scale,
                    (center - right) / scale,
                    (left - right) / scale,
                ),
                dim=-1,
            )
        )
    return torch.cat(tuple(pieces), dim=-1)


def bottle_style_critic_context(
    reference: torch.Tensor,
    *,
    force_range: ForceRange,
    grip_force_n: torch.Tensor,
    pedicle_tension_n: torch.Tensor,
) -> torch.Tensor:
    """Build public/measurable critic context without hidden task thresholds."""

    batch = len(reference)
    grip = grip_force_n.to(reference).reshape(batch, 1)
    tension = pedicle_tension_n.to(reference).reshape(batch, 1)
    force_scale = float(args.public_grip_safety_limit_n)
    tension_scale = float(args.public_tension_target_n)
    low = torch.full_like(grip, force_range.minimum_n) / force_scale
    high = torch.full_like(grip, force_range.maximum_n) / force_scale
    return torch.cat((low, high, grip / force_scale, tension / tension_scale), dim=-1)


def shaped_reward(
    flags: dict[str, torch.Tensor], *, terminal: bool
) -> float:
    return environment_shaped_reward(
        left_force_n=float(flags["left_normal_force_n"][0].item()),
        right_force_n=float(flags["right_normal_force_n"][0].item()),
        bilateral_contact=bool(flags["bilateral_contact"][0].item()),
        pedicle_tension_n=float(flags["stem_tension_n"][0].item()),
        detach_threshold_n=float(flags["detach_threshold_n"][0].item()),
        damage_threshold_n=float(flags["damage_threshold_n"][0].item()),
        terminal_outcome=classify(flags) if terminal else None,
    )


def capture_advisor_visual(
    path: Path,
    *,
    trace: list[dict[str, float | int | bool]],
    outcome: str,
    force_range: ForceRange,
    public_safety_limit_n: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1000, 620
    image = Image.new("RGB", (width, height), (250, 251, 253))
    draw = ImageDraw.Draw(image)
    draw.text((32, 18), f"RaspberryPick episode: {outcome}", fill=(20, 25, 35))
    draw.text(
        (32, 43),
        f"applied grip target: {force_range.minimum_n:.2f}-{force_range.maximum_n:.2f} N",
        fill=(28, 115, 70),
    )
    box = (78, 90, 950, 540)
    draw.rectangle(box, outline=(85, 95, 110), width=2)
    steps = [int(item["physics_step"]) for item in trace]
    grip = [float(item["grip_force_n"]) for item in trace]
    tension = [float(item["pedicle_tension_n"]) for item in trace]
    x_max = max(steps, default=1)
    y_max = max(3.5, max(grip, default=0.0), max(tension, default=0.0))

    def x_coord(step: int) -> float:
        return box[0] + (box[2] - box[0]) * step / max(x_max, 1)

    def y_coord(value: float) -> float:
        return box[3] - (box[3] - box[1]) * value / y_max

    draw.rectangle(
        (box[0], y_coord(force_range.maximum_n), box[2], y_coord(force_range.minimum_n)),
        fill=(224, 244, 229),
    )
    draw.line(
        (box[0], y_coord(float(public_safety_limit_n)), box[2], y_coord(float(public_safety_limit_n))),
        fill=(205, 55, 55),
        width=3,
    )
    if len(steps) >= 2:
        draw.line(
            [(x_coord(step), y_coord(value)) for step, value in zip(steps, grip)],
            fill=(37, 99, 190),
            width=3,
        )
        draw.line(
            [(x_coord(step), y_coord(value)) for step, value in zip(steps, tension)],
            fill=(220, 135, 20),
            width=3,
        )
    for tick in range(int(math.floor(y_max)) + 1):
        y = y_coord(float(tick))
        draw.line((box[0] - 5, y, box[0], y), fill=(80, 90, 105), width=1)
        draw.text((29, y - 7), f"{tick} N", fill=(60, 70, 85))
    draw.text((82, 565), "blue: pad grip", fill=(37, 99, 190))
    draw.text((250, 565), "orange: pedicle tension", fill=(190, 105, 10))
    draw.text((600, 565), "green: VLM target band", fill=(28, 115, 70))
    draw.text((805, 565), "red: public safety limit", fill=(180, 35, 35))
    image.save(path)


def save_camera_frame(env: RaspberryPickEnv, camera: str, path: Path) -> None:
    frame = env.get_camera_rgb(camera)[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(frame).save(path)


def compose_advisor_visual(
    path: Path,
    *,
    wrist_path: Path,
    third_path: Path,
    diagnostic_path: Path,
) -> None:
    """Compose the two true RGB views and the diagnostic plot into one image."""

    camera_size = (640, 480)
    chart_size = (1280, 794)
    canvas = Image.new("RGB", (1280, 1318), (246, 248, 251))
    draw = ImageDraw.Draw(canvas)
    with Image.open(wrist_path) as wrist_source:
        wrist = wrist_source.convert("RGB").resize(camera_size, Image.Resampling.LANCZOS)
    with Image.open(third_path) as third_source:
        third = third_source.convert("RGB").resize(camera_size, Image.Resampling.LANCZOS)
    with Image.open(diagnostic_path) as diagnostic_source:
        diagnostic = diagnostic_source.convert("RGB").resize(chart_size, Image.Resampling.LANCZOS)
    canvas.paste(wrist, (0, 30))
    canvas.paste(third, (640, 30))
    canvas.paste(diagnostic, (0, 524))
    draw.rectangle((0, 0, 640, 30), fill=(28, 34, 44))
    draw.rectangle((640, 0, 1280, 30), fill=(28, 34, 44))
    draw.text((16, 8), "WRIST RGB", fill=(255, 255, 255))
    draw.text((656, 8), "THIRD-PERSON RGB", fill=(255, 255, 255))
    canvas.save(path)


def save_training_checkpoint(
    path: Path,
    *,
    agent: SACResidualAgent,
    layout: FlowNoiseDSRLLayout,
    semantic_layout: SemanticResidualLayout | None,
    bottle_layout: BottleStyleLateralResidualLayout | None,
    replay: ReplayBuffer,
    adapter: OnlineForceRangeAdapter,
    controller: GrapeForceController,
    base_hash: str,
    completed_episodes: int,
    replay_transitions: int,
    balance_prior_value: float,
    recent_history: deque[dict[str, Any]],
    recent_candidate_directions: deque[str],
    advisor: CodexVisualForceAdvisor | None,
    episode_records: list[dict[str, Any]],
) -> None:
    payload = {
        **agent.checkpoint(),
        "protocol": args.protocol_name,
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "task": TASK,
        "method": args.method,
        "force_range_source": args.force_range_source,
        "controller_mode": args.controller_mode,
        "flow_num_inference_steps": int(args.num_inference_steps),
        "chunk_execute_steps": int(args.chunk_execute_steps),
        "action_repeat": int(args.action_repeat),
        "vlm_call_policy": args.vlm_call_policy,
        "public_grip_safety_limit_n": float(args.public_grip_safety_limit_n),
        "public_tension_target_n": float(args.public_tension_target_n),
        "berry_profile": args.berry_profile,
        "base_checkpoint": str(args.checkpoint.resolve()),
        "base_checkpoint_sha256": base_hash,
        "base_checkpoint_weights": "raw_best_epoch",
        "residual_mode": (
            "bottle_style_lateral"
            if bottle_layout is not None
            else "semantic_xy"
            if args.method in SEMANTIC_XY_METHODS
            else "semantic"
            if semantic_layout is not None
            else "flow_latent"
        ),
        "dsrl_frozen": bool(args.freeze_dsrl),
        "flow_initial_latent": (
            "zero"
            if semantic_layout is not None or bottle_layout is not None
            else "zero_plus_bounded_dsrl_correction"
        ),
        "flow_horizon": layout.horizon,
        "flow_action_dim": layout.action_dim,
        "learned_noise_steps": layout.learned_noise_steps,
        "noise_action_indices": list(layout.active_action_indices),
        "noise_residual_scale": layout.residual_scale,
        "semantic_residual_layout": (
            None if semantic_layout is None else asdict(semantic_layout)
        ),
        "bottle_style_lateral_residual": (
            None
            if bottle_layout is None
            else {
                **asdict(bottle_layout),
                "axis": "gripper_closing_axis_b",
                "activation": "bilateral_contact_after_dsrl_start_step",
                "critic_context": [
                    "force_target_low",
                    "force_target_high",
                    "measured_grip_force",
                    "measured_pedicle_tension",
                ],
                "force_balance_prior_gain": args.force_balance_prior_gain,
                "force_balance_prior_weight": args.force_balance_prior_weight,
                "force_balance_prior_limit": args.force_balance_prior_limit,
                "force_balance_prior_smoothing": args.force_balance_prior_smoothing,
                "current_prior_value": balance_prior_value,
            }
        ),
        "actor_observation_dim": agent.actor.network[0].in_features,
        "actor_observation_features": (
            "pooled_condition+visual_xy+normalized_state+physics+per_finger_target_errors"
            if bottle_layout is not None
            else "pooled_condition+visual_xy+normalized_state+physics"
            if args.method in PREPULL_SEMANTIC_METHODS
            else "pooled_condition+physics"
        ),
        "dsrl_start_step": args.dsrl_start_step,
        "exploration_anneal_episodes": args.exploration_anneal_episodes,
        "residual_active_until_phase": (
            float(args.pull_z_start_phase)
            if args.method in PREPULL_SEMANTIC_METHODS
            else None
        ),
        "force_controller_config": asdict(controller.config),
        "oracle_control_mode": (
            args.oracle_control_mode if args.method == "oracle_force_dsrl" else None
        ),
        "current_force_range_n": adapter.current_range.as_list(),
        "force_range_decisions": adapter.decisions,
        "completed_episodes": completed_episodes,
        "replay_transitions": replay_transitions,
        "replay_state": replay.state_dict(),
        "training_seed": args.train_seed,
        "fixed_episode_seed": bool(args.fixed_episode_seed),
        "stem_lock_rotation_while_attached": True,
        "recent_history": list(recent_history),
        "recent_candidate_directions": list(recent_candidate_directions),
        "episode_records": episode_records,
        "vlm_calls": 0 if advisor is None else int(advisor.calls),
        "vlm_successful_calls": 0 if advisor is None else int(advisor.successful_calls),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "torch_cuda_random_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    output_root = args.output_root.expanduser().resolve()
    resume_checkpoint_path: Path | None = None
    if args.resume:
        if not output_root.is_dir():
            raise FileNotFoundError(f"resume output root does not exist: {output_root}")
        latest = output_root / "checkpoints" / "latest.pt"
        if latest.is_file():
            resume_checkpoint_path = latest
        else:
            periodic = sorted((output_root / "checkpoints").glob("episode_*.pt"))
            if not periodic:
                raise FileNotFoundError("resume requested but no valid checkpoint exists")
            resume_checkpoint_path = periodic[-1]
    elif output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    frame_dir = output_root / "advisor_frames"
    trajectory_dir = output_root / "trajectories"
    checkpoint_dir = output_root / "checkpoints"
    for directory in (frame_dir, trajectory_dir, checkpoint_dir):
        directory.mkdir(exist_ok=args.resume)

    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)

    profile = RASPBERRY_PROFILES[args.berry_profile]
    effective_profile = (
        replace(
            profile,
            damage_threshold_n=float(args.damage_threshold_n),
            detach_threshold_n=float(args.detach_threshold_n),
        )
        if args.berry_profile == "fixed"
        else profile
    )
    oracle_enabled = args.method == "oracle_force_dsrl"
    oracle_force = None
    oracle_safety_limit_n: float | None = None
    if oracle_enabled and args.oracle_control_mode in {"fixed_interval", "two_stage"}:
        oracle_force = theoretical_oracle_force_range(
            effective_profile,
            interval_width_n=args.oracle_force_width_n,
            damage_margin_n=args.oracle_damage_margin_n,
        )
        physical_low, physical_high = map(float, args.physical_force_bounds)
        oracle_low, oracle_high = oracle_force.as_tuple()
        if not physical_low <= oracle_low < oracle_high <= physical_high:
            raise ValueError("oracle force interval violates the physical force bounds")
        if args.oracle_control_mode == "two_stage":
            oracle_safety_limit_n = (
                float(effective_profile.damage_threshold_n)
                - 0.5 * float(args.oracle_damage_margin_n)
            )
    elif oracle_enabled and args.oracle_control_mode == "safety_shield":
        oracle_safety_limit_n = (
            float(effective_profile.damage_threshold_n)
            - float(args.oracle_damage_margin_n)
        )
        if oracle_safety_limit_n <= float(effective_profile.detach_threshold_n):
            raise ValueError("oracle safety limit must clear the detach threshold")
    config = RaspberryPickEnvCfg()
    config.scene.num_envs = 1
    config.seed = args.train_seed
    config.enable_cameras = True
    config.enable_gelsight = False
    config.randomize_grape_pose = True
    config.grape_position_randomization_m = tuple(map(float, args.position_range))
    config.randomize_force_thresholds = False
    if args.berry_profile == "fixed":
        config.damage_grip_force_threshold_n = float(args.damage_threshold_n)
        config.detach_force_threshold_n = float(args.detach_threshold_n)
    else:
        config.damage_grip_force_threshold_n = profile.damage_threshold_n
        config.detach_force_threshold_n = profile.detach_threshold_n
    config.stem_stiffness_n_per_m = profile.stem_stiffness_n_per_m
    config.stem_damping_n_s_per_m = profile.stem_damping_n_s_per_m
    config.sim.physx.gpu_max_rigid_contact_count = args.physx_contact_capacity
    config.sim.physx.gpu_max_rigid_patch_count = args.physx_contact_capacity
    config.sim.physx.enable_enhanced_determinism = bool(args.enhanced_determinism)
    config.episode_length_s = max(args.max_episode_steps * config.sim.dt, 1.0)
    if args.device:
        config.sim.device = args.device
    env = RaspberryPickEnv(config, render_mode="rgb_array")
    device = torch.device(config.sim.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    base_hash = sha256(checkpoint_path)
    if args.method in {"dsrl_only", "vlm_two_stage_dsrl"} and args.bottle_style_lateral_dsrl:
        if base_hash != FORMAL_BASE_CHECKPOINT_SHA256:
            raise ValueError(
                "formal RaspberryPick run requires the preregistered frozen BC checkpoint"
            )
    manifest_path = output_root / "protocol_manifest.json"
    manifest = {
        "protocol": args.protocol_name,
        "method": args.method,
        "force_range_source": args.force_range_source,
        "controller_mode": args.controller_mode,
        "flow_num_inference_steps": int(args.num_inference_steps),
        "flow_chunk_size": None,
        "chunk_execute_steps": int(args.chunk_execute_steps),
        "action_repeat": int(args.action_repeat),
        "vlm_call_policy": args.vlm_call_policy,
        "training_seed": int(args.train_seed),
        "train_episodes": int(args.train_episodes),
        "episode_seed_mode": "fixed" if args.fixed_episode_seed else "sequential",
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha256": base_hash,
        "argv": list(sys.argv),
        "codex_model": args.codex_model if args.force_range_source == "vlm" else None,
        "codex_reasoning_effort": (
            args.codex_reasoning_effort if args.force_range_source == "vlm" else None
        ),
        "public_grip_safety_limit_n": float(args.public_grip_safety_limit_n),
        "public_tension_target_n": float(args.public_tension_target_n),
        "initial_force_range_n": list(map(float, args.initial_force_range)),
        "physical_force_bounds_n": list(map(float, args.physical_force_bounds)),
        "position_randomization_m": list(map(float, args.position_range)),
        "bottle_lateral_residual_maximum_m": float(args.lateral_residual_scale_m),
        "enhanced_determinism": bool(args.enhanced_determinism),
        **repository_revision(),
    }
    if args.resume:
        recorded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        legacy_protocol_defaults = {
            "protocol": "raspberry_0909_v1",
            "flow_num_inference_steps": 20,
            "chunk_execute_steps": 1,
            "action_repeat": 4,
            "vlm_call_policy": "every_episode",
        }
        for key in (
            "protocol",
            "method",
            "force_range_source",
            "controller_mode",
            "flow_num_inference_steps",
            "chunk_execute_steps",
            "action_repeat",
            "vlm_call_policy",
            "training_seed",
            "train_episodes",
            "base_checkpoint_sha256",
        ):
            recorded_value = recorded_manifest.get(key, legacy_protocol_defaults.get(key))
            if recorded_value != manifest.get(key):
                raise ValueError(f"resume protocol mismatch for {key}")
    model, normalizer, base_checkpoint = load_policy(checkpoint_path, device)
    if model.config.state_dim != 12 or model.config.num_cameras != 2:
        raise ValueError("RaspberryPick policy must use the 12-D tactile state and two cameras")
    if args.chunk_execute_steps > model.config.chunk_size:
        raise ValueError(
            f"chunk-execute-steps ({args.chunk_execute_steps}) exceeds the Flow chunk size "
            f"({model.config.chunk_size})"
        )
    manifest["flow_chunk_size"] = int(model.config.chunk_size)
    if not args.resume:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    layout = FlowNoiseDSRLLayout(
        horizon=model.config.chunk_size,
        action_dim=model.config.action_dim,
        learned_noise_steps=args.learned_noise_steps,
        residual_scale=args.noise_residual_scale,
        active_action_indices=tuple(args.noise_action_indices),
    )
    semantic_layout = None
    bottle_layout = (
        BottleStyleLateralResidualLayout(
            maximum_offset_m=float(args.lateral_residual_scale_m)
        )
        if args.bottle_style_lateral_dsrl
        else None
    )
    if args.method in SEMANTIC_XY_METHODS:
        semantic_layout = SemanticXYResidualLayout(
            translation_scale_m=tuple(map(float, args.semantic_translation_scale_m[:2]))
        )
    elif args.method in SEMANTIC_METHODS:
        semantic_layout = SemanticResidualLayout(
            translation_scale_m=tuple(map(float, args.semantic_translation_scale_m)),
            gripper_scale_m=float(args.semantic_gripper_scale_m),
        )
    correction_dim = (
        bottle_layout.correction_dim
        if bottle_layout is not None
        else semantic_layout.correction_dim
        if semantic_layout is not None
        else layout.correction_dim
    )
    actor_observation_dim = model.config.cond_dim + 9
    if args.method in PREPULL_SEMANTIC_METHODS or bottle_layout is not None:
        actor_observation_dim += 2 + model.config.state_dim
    if bottle_layout is not None:
        actor_observation_dim += 3
    critic_action_context_dim = 4 if bottle_layout is not None else 0
    agent = SACResidualAgent(
        observation_dim=actor_observation_dim,
        action_dim=correction_dim,
        device=device,
        learning_rate=args.learning_rate,
        gamma=0.995,
        tau=0.01,
        residual_l2_weight=0.10,
        initial_log_std=args.initial_log_std,
        initial_alpha=0.03,
        critic_action_context_dim=critic_action_context_dim,
        critic_context_mode="concat",
    )
    replay = ReplayBuffer(
        args.replay_capacity,
        actor_observation_dim,
        correction_dim,
        device,
        action_context_dim=critic_action_context_dim,
    )
    loaded_agent_checkpoint: Path | None = None
    loaded_agent_checkpoint_mode: str | None = None
    loaded_agent_checkpoint_source_method: str | None = None
    loaded_agent_checkpoint_source_contract: str | None = None
    loaded_agent_payload: dict[str, Any] | None = None
    branch_source_run: Path | None = None
    branch_source_episode_log: Path | None = None
    branch_source_episode_log_sha256: str | None = None
    checkpoint_to_load = (
        resume_checkpoint_path
        if resume_checkpoint_path is not None
        else args.agent_checkpoint.expanduser().resolve()
        if args.agent_checkpoint is not None
        else None
    )
    run_start_episode = 0
    if checkpoint_to_load is not None:
        loaded_agent_checkpoint = checkpoint_to_load
        payload = torch.load(loaded_agent_checkpoint, map_location="cpu", weights_only=False)
        loaded_agent_payload = payload
        loaded_agent_checkpoint_mode = validate_agent_checkpoint_contract(
            payload,
            target_method=args.method,
            target_contract=CONTRACT,
            dsrl_contract=METHOD_CONTRACTS["dsrl_only"],
            base_checkpoint_sha256=base_hash,
            expected_observation_dim=actor_observation_dim,
            expected_residual_mode=(
                "bottle_style_lateral"
                if bottle_layout is not None
                else "semantic_xy"
                if args.method in SEMANTIC_XY_METHODS
                else "semantic"
                if semantic_layout is not None
                else "flow_latent"
            ),
            evaluation_mode=bool(args.evaluation_mode),
            allow_dsrl_for_controller_only_oracle=bool(
                args.allow_dsrl_checkpoint_for_controller_only_oracle
            ),
            allow_dsrl_for_oracle_training_branch=bool(
                args.branch_from_dsrl_checkpoint
            ),
        )
        loaded_agent_checkpoint_source_method = str(payload.get("method"))
        loaded_agent_checkpoint_source_contract = str(payload.get("contract"))
        agent.load_checkpoint(payload)
        if args.resume:
            run_start_episode = int(payload.get("completed_episodes", -1))
            total_transitions_at_branch = validate_training_branch_checkpoint(
                payload,
                resume_episode=run_start_episode,
                training_seed=args.train_seed,
            )
            legacy_checkpoint_defaults = {
                "protocol": "raspberry_0909_v1",
                "flow_num_inference_steps": 20,
                "chunk_execute_steps": 1,
                "action_repeat": 4,
                "vlm_call_policy": "every_episode",
            }
            for key, expected in (
                ("protocol", args.protocol_name),
                ("force_range_source", args.force_range_source),
                ("controller_mode", args.controller_mode),
                ("flow_num_inference_steps", int(args.num_inference_steps)),
                ("chunk_execute_steps", int(args.chunk_execute_steps)),
                ("action_repeat", int(args.action_repeat)),
                ("vlm_call_policy", args.vlm_call_policy),
            ):
                actual = payload.get(key, legacy_checkpoint_defaults.get(key))
                if actual != expected:
                    raise ValueError(f"resume checkpoint mismatch for {key}")
            replay_state = payload["replay_state"]
            if int(replay_state["capacity"]) != int(args.replay_capacity):
                raise ValueError("checkpoint replay capacity does not match resumed run")
            replay.load_state_dict(replay_state)
        elif args.branch_from_dsrl_checkpoint:
            run_start_episode = int(args.resume_episode)
            total_transitions_at_branch = validate_training_branch_checkpoint(
                payload,
                resume_episode=args.resume_episode,
                training_seed=args.train_seed,
            )
            replay_state = payload["replay_state"]
            if int(replay_state["capacity"]) != int(args.replay_capacity):
                raise ValueError("checkpoint replay capacity does not match branch run")
            replay.load_state_dict(replay_state)
        else:
            total_transitions_at_branch = 0
    else:
        total_transitions_at_branch = 0
    starting_agent_updates = agent.updates
    baseline_force_range = tuple(map(float, args.initial_force_range))
    effective_initial_force_range = (
        oracle_force.as_tuple()
        if oracle_force is not None and args.oracle_start_episode == 0
        else baseline_force_range
    )
    # Legacy oracle ablations keep the residual actor blind to the privileged
    # interval.  Bottle-style actors receive the actual controller interval;
    # their critic context contains only public targets and measured forces.
    actor_force_context = ForceRange(*baseline_force_range)
    adapter = OnlineForceRangeAdapter(
        ForceRange(*effective_initial_force_range),
        ForceRangeAdapterConfig(
            physical_minimum_n=float(args.physical_force_bounds[0]),
            physical_maximum_n=float(args.physical_force_bounds[1]),
            minimum_width_n=float(args.force_range_minimum_width),
            smoothing=float(args.force_range_smoothing),
        ),
    )
    if args.branch_from_dsrl_checkpoint or args.resume:
        if loaded_agent_payload is None:
            raise AssertionError("training checkpoint was validated above")
        checkpoint_force_range = ForceRange(
            *map(float, loaded_agent_payload["current_force_range_n"])
        )
        if args.branch_from_dsrl_checkpoint and not np.allclose(
            checkpoint_force_range.as_list(),
            baseline_force_range,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError(
                "checkpoint current force range does not match the requested baseline range"
            )
        adapter.current_range = checkpoint_force_range
        adapter.decisions = [
            dict(item) for item in loaded_agent_payload["force_range_decisions"]
        ]
    controller = GrapeForceController(
        adapter.current_range,
        ForceControllerConfig(
            minimum_width_m=0.006,
            maximum_width_m=0.040,
            maximum_width_rate_m_s=0.006,
            free_space_close_rate_m_s=0.012,
            safety_open_rate_m_s=0.008,
            release_hysteresis_steps=30,
            hard_force_limit_n=float(args.physical_force_bounds[1]),
        ),
    )
    two_stage_controller = (
        TwoStageBerryForceController(
            adapter.current_range,
            TwoStageForceControllerConfig(
                contact_confirm_steps=args.oracle_two_stage_contact_confirm_steps,
                grip_stable_steps=args.oracle_two_stage_grip_stable_steps,
                tension_kp_velocity_m_s_per_n=args.oracle_two_stage_tension_kp,
                maximum_pull_velocity_m_s=args.oracle_two_stage_max_pull_velocity_m_s,
            ),
            grip_controller_config=controller.config,
        )
        if args.controller_mode == "two_stage"
        else None
    )
    force_balance_aligner = (
        ForceBalanceAligner(
            ForceBalanceAlignmentConfig(
                alignment_velocity_m_s=args.oracle_force_balance_velocity_m_s,
                maximum_alignment_offset_m=args.oracle_force_balance_max_offset_m,
            )
        )
        if args.oracle_force_balance_alignment
        else None
    )
    if args.branch_from_dsrl_checkpoint or args.resume:
        if loaded_agent_payload is None:
            raise AssertionError("training checkpoint was validated above")
        if loaded_agent_payload.get("force_controller_config") != asdict(controller.config):
            raise ValueError("checkpoint force-controller configuration does not match run")
    advisor = (
        CodexVisualForceAdvisor(output_root)
        if args.force_range_source == "vlm"
        else None
    )
    position_offset = torch.tensor(
        args.position_command_offset_m, dtype=torch.float32, device=device
    )
    records: list[dict[str, Any]] = []
    if args.branch_from_dsrl_checkpoint:
        if loaded_agent_checkpoint is None:
            raise AssertionError("branch checkpoint path was validated above")
        branch_source_run = loaded_agent_checkpoint.parent.parent
        branch_source_episode_log = branch_source_run / "training_episodes.jsonl"
        if not branch_source_episode_log.is_file():
            raise FileNotFoundError(
                f"branch source episode log is missing: {branch_source_episode_log}"
            )
        source_records = [
            json.loads(line)
            for line in branch_source_episode_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(source_records) < args.resume_episode:
            raise ValueError("branch source episode log is shorter than resume episode")
        records = [dict(item) for item in source_records[: args.resume_episode]]
        expected_indices = list(range(args.resume_episode))
        if [int(item["episode_index"]) for item in records] != expected_indices:
            raise ValueError("branch source episode prefix is non-contiguous")
        expected_seeds = (
            [args.train_seed for _ in expected_indices]
            if args.fixed_episode_seed
            else [args.train_seed + index for index in expected_indices]
        )
        if [int(item["seed"]) for item in records] != expected_seeds:
            raise ValueError("branch source episode seeds do not match the requested run")
        if any(
            not math.isclose(
                float(item["damage_threshold_n"]),
                float(args.damage_threshold_n),
                rel_tol=0.0,
                abs_tol=1.0e-5,
            )
            or not math.isclose(
                float(item["detach_threshold_n"]),
                float(args.detach_threshold_n),
                rel_tol=0.0,
                abs_tol=1.0e-5,
            )
            for item in records
        ):
            raise ValueError("branch source episode thresholds do not match the branch run")
        source_decision_path = branch_source_run / "force_range_decisions.jsonl"
        if not source_decision_path.is_file():
            raise FileNotFoundError(
                f"branch source force-decision log is missing: {source_decision_path}"
            )
        source_decisions = [
            json.loads(line)
            for line in source_decision_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][: args.resume_episode]
        if source_decisions != adapter.decisions:
            raise ValueError("checkpoint force decisions do not match the source run log")
        branch_source_episode_log_sha256 = sha256(branch_source_episode_log)
        for item in records:
            append_jsonl(output_root / "training_episodes.jsonl", item)
        for item in adapter.decisions:
            append_jsonl(output_root / "force_range_decisions.jsonl", item)
    elif args.resume:
        if loaded_agent_payload is None:
            raise AssertionError("resume checkpoint was validated above")
        episode_path = output_root / "training_episodes.jsonl"
        decision_path = output_root / "force_range_decisions.jsonl"
        logged_records = [
            json.loads(line)
            for line in (
                episode_path.read_text(encoding="utf-8").splitlines()
                if episode_path.is_file()
                else []
            )
            if line.strip()
        ]
        logged_decisions = [
            json.loads(line)
            for line in (
                decision_path.read_text(encoding="utf-8").splitlines()
                if decision_path.is_file()
                else []
            )
            if line.strip()
        ]
        records = [dict(item) for item in loaded_agent_payload.get("episode_records", [])]
        if len(records) != run_start_episode:
            raise ValueError("resume checkpoint episode history is incomplete")
        if logged_records != records:
            replace_jsonl(episode_path, records)
        if logged_decisions != adapter.decisions:
            replace_jsonl(decision_path, adapter.decisions)
        expected_indices = list(range(run_start_episode))
        if [int(item["episode_index"]) for item in records] != expected_indices:
            raise ValueError("resume episode log is non-contiguous")
    recent_history: deque[dict[str, Any]] = deque(
        (
            loaded_agent_payload.get("recent_history", [])
            if args.resume and loaded_agent_payload is not None
            else []
        ),
        maxlen=10,
    )
    recent_candidate_directions: deque[str] = deque(
        (
            loaded_agent_payload.get("recent_candidate_directions", [])
            if args.resume and loaded_agent_payload is not None
            else []
        ),
        maxlen=max(args.guard_confirmations, 3),
    )
    if args.resume and loaded_agent_payload is not None:
        if advisor is not None:
            advisor.calls = int(loaded_agent_payload.get("vlm_calls", 0))
            advisor.successful_calls = int(
                loaded_agent_payload.get("vlm_successful_calls", 0)
            )
        random.setstate(loaded_agent_payload["python_random_state"])
        np.random.set_state(loaded_agent_payload["numpy_random_state"])
        torch.random.set_rng_state(loaded_agent_payload["torch_random_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                loaded_agent_payload["torch_cuda_random_state_all"]
            )
    total_transitions = total_transitions_at_branch
    last_update: dict[str, float] = {}
    balance_prior_value = (
        0.0
        if loaded_agent_payload is None or bottle_layout is None
        else float(
            (loaded_agent_payload.get("bottle_style_lateral_residual") or {}).get(
                "current_prior_value", 0.0
            )
        )
    )
    started = time.time()

    try:
        invocation_stop_episode = min(
            args.train_episodes,
            run_start_episode + (
                args.episodes_this_invocation
                if args.episodes_this_invocation is not None
                else args.train_episodes
            ),
        )
        for episode_index in range(run_start_episode, invocation_stop_episode):
            episode_seed = (
                args.train_seed if args.fixed_episode_seed else args.train_seed + episode_index
            )
            env.reset(seed=episode_seed)
            oracle_active = (
                oracle_enabled and episode_index >= args.oracle_start_episode
            )
            if (
                oracle_active
                and oracle_force is not None
                and adapter.current_range.as_list() != list(oracle_force.as_tuple())
            ):
                adapter.current_range = ForceRange(*oracle_force.as_tuple())
            controller.set_target_range(adapter.current_range)
            controller.reset()
            if two_stage_controller is not None:
                two_stage_controller.set_grip_target_range(adapter.current_range)
                two_stage_controller.reset()
            if force_balance_aligner is not None:
                force_balance_aligner.reset()
            attempted_range = adapter.current_range
            previous_action: torch.Tensor | None = None
            physics_step = 0
            episode_return = 0.0
            policy_decisions = 0
            executed_chunk_actions = 0
            dsrl_active_decisions = 0
            correction_abs_sum = torch.zeros(correction_dim, device=device)
            max_abs_lateral_residual_m = 0.0
            pending_terminal_credit_index: int | None = None
            controller_active_steps = 0
            safety_override_steps = 0
            oracle_tension_assist_steps = 0
            bilateral_steps = 0
            force_sum = 0.0
            force_error_sum = 0.0
            left_force_sum = 0.0
            right_force_sum = 0.0
            damage_risk_steps = 0
            public_safety_risk_steps = 0
            force_trace: list[dict[str, float | int | bool]] = []

            env.sim.render()
            state_now = policy_state(env)
            image_now = torch.stack(
                (camera_rgb_224(env, "wrist"), camera_rgb_224(env, "third")), dim=1
            )
            state_history: deque[torch.Tensor] = deque(
                (state_now.clone() for _ in range(model.config.state_obs_steps)),
                maxlen=model.config.state_obs_steps,
            )
            image_history: deque[torch.Tensor] = deque(
                (image_now.clone() for _ in range(model.config.image_obs_steps)),
                maxlen=model.config.image_obs_steps,
            )
            model_obs = model_observation(state_history, image_history, normalizer, 0.0)
            with torch.inference_mode():
                cond_tokens, visual_xy = model.encode_observation(model_obs)
            normalized_state = model_obs[common.STATE_KEY][:, -1]

            trajectory_path = trajectory_dir / f"episode_{episode_index:03d}.jsonl.gz"
            with gzip.open(trajectory_path, "xt", encoding="utf-8") as trajectory_stream:
                while physics_step < args.max_episode_steps:
                    policy_decisions += 1
                    phase = min(physics_step / float(args.max_episode_steps), 1.0)
                    metrics = env.get_contact_force_metrics()
                    flags_before = env.get_task_flags()
                    embedding = model.dsrl_observation(cond_tokens)
                    policy_observation = actor_observation(
                        embedding,
                        visual_xy=visual_xy,
                        normalized_state=normalized_state,
                        explicit_spatial_features=(
                            args.method in PREPULL_SEMANTIC_METHODS
                            or bottle_layout is not None
                        ),
                        phase=phase,
                        metrics=metrics,
                        flags=flags_before,
                        force_range=(
                            attempted_range
                            if bottle_layout is not None or not oracle_enabled
                            else actor_force_context
                        ),
                        bottle_style_force_features=bottle_layout is not None,
                    )
                    dsrl_active = residual_is_active(
                        physics_step=physics_step,
                        start_step=args.dsrl_start_step,
                        phase=phase,
                        pull_start_phase=args.pull_z_start_phase,
                        stem_attached=bool(flags_before["stem_attached"][0].item()),
                        semantic_residual=args.method in PREPULL_SEMANTIC_METHODS,
                    ) and (
                        bottle_layout is None
                        or bool(metrics["bilateral_contact"][0].item())
                    )
                    exploration = exploration_scale_for_episode(
                        episode_index=episode_index,
                        anneal_episodes=args.exploration_anneal_episodes,
                        exploration_start=args.exploration_start,
                        exploration_end=args.exploration_end,
                    )
                    action_context = (
                        bottle_style_critic_context(
                            policy_observation,
                            force_range=attempted_range,
                            grip_force_n=metrics["grip_force_n"],
                            pedicle_tension_n=flags_before["stem_tension_n"],
                        )
                        if bottle_layout is not None
                        else None
                    )
                    with torch.inference_mode():
                        if dsrl_active and not args.freeze_dsrl:
                            correction = agent.act(
                                policy_observation,
                                deterministic=args.evaluation_mode,
                                exploration_scale=exploration,
                            )
                        else:
                            correction = torch.zeros(
                                (1, correction_dim), dtype=embedding.dtype, device=device
                            )
                        if semantic_layout is None and bottle_layout is None:
                            initial_noise = layout.compose_zero_anchored_noise(correction)
                        else:
                            initial_noise = correction.new_zeros(
                                (1, layout.horizon, layout.action_dim)
                            )
                        normalized_chunk = decode_from_condition(model, cond_tokens, initial_noise)
                        normalized_chunk = normalized_chunk.clone()
                        normalized_chunk[:, :, :2] = visual_xy[:, None, :]
                        chunk = normalizer.unnormalize_tensor(common.ACTION_KEY, normalized_chunk)
                        chunk[:, :, :3] += position_offset.view(1, 1, 3)
                        pull_progress = min(
                            max(
                                (phase - args.pull_z_start_phase)
                                / (args.pull_z_ramp_end_phase - args.pull_z_start_phase),
                                0.0,
                            ),
                            1.0,
                        )
                        chunk[:, :, 2] += float(args.pull_z_offset_m) * pull_progress
                        chunk[:, :, 9] += float(args.gripper_command_offset_m)
                        if semantic_layout is not None:
                            chunk = semantic_layout.apply(chunk, correction)
                        if bottle_layout is not None:
                            chunk = bottle_layout.apply(
                                chunk, correction, metrics["closing_axis_b"]
                            )
                    if dsrl_active and not args.freeze_dsrl:
                        correction_abs_sum += correction[0].abs()
                        dsrl_active_decisions += 1
                        if bottle_layout is not None:
                            max_abs_lateral_residual_m = max(
                                max_abs_lateral_residual_m,
                                abs(float(correction[0, 0].item()))
                                * bottle_layout.maximum_offset_m,
                            )

                    transition_reward = 0.0
                    terminal = False
                    final_flags = flags_before
                    for chunk_physics_index in range(
                        args.chunk_execute_steps * args.action_repeat
                    ):
                        chunk_action_index = chunk_physics_index // args.action_repeat
                        if chunk_physics_index % args.action_repeat == 0:
                            executed_chunk_actions += 1
                        target = chunk[:, chunk_action_index]
                        if previous_action is None:
                            policy_action = target
                        else:
                            alpha = float(args.action_smoothing)
                            policy_action = alpha * target + (1.0 - alpha) * previous_action
                        previous_action = policy_action
                        metrics = env.get_contact_force_metrics()
                        tension_assist_active = torch.zeros_like(
                            metrics["bilateral_contact"], dtype=torch.bool
                        )
                        two_stage_diagnostics = None
                        force_balance_diagnostics = None
                        if oracle_active and force_balance_aligner is not None:
                            policy_action, force_balance_diagnostics = (
                                force_balance_aligner.control(
                                    policy_action,
                                    left_force_n=metrics["left_normal_force_n"],
                                    right_force_n=metrics["right_normal_force_n"],
                                    bilateral_contact=metrics["bilateral_contact"],
                                    closing_axis_b=metrics["closing_axis_b"],
                                    dt_s=float(env.physics_dt),
                                )
                            )
                        if (
                            oracle_active
                            and args.oracle_control_mode == "tension_assist"
                            and phase >= float(args.oracle_tension_assist_start_phase)
                        ):
                            control_flags = env.get_task_flags()
                            tension_assist_active = (
                                metrics["bilateral_contact"]
                                & control_flags["stem_attached"]
                                & (
                                    control_flags["stem_tension_n"]
                                    < control_flags["detach_threshold_n"]
                                )
                                & (
                                    metrics["grip_force_n"]
                                    < control_flags["damage_threshold_n"]
                                    - float(args.oracle_damage_margin_n)
                                )
                            )
                            if bool(tension_assist_active.any()):
                                policy_action = policy_action.clone()
                                policy_action[tension_assist_active, 2] -= float(
                                    args.oracle_tension_assist_m
                                )
                        if two_stage_controller is not None:
                            control_flags = env.get_task_flags()
                            executed_action, two_stage_diagnostics = two_stage_controller.control(
                                policy_action,
                                grip_force_n=metrics["grip_force_n"],
                                bilateral_contact=metrics["bilateral_contact"],
                                pedicle_tension_n=control_flags["stem_tension_n"],
                                stem_attached=control_flags["stem_attached"],
                                pull_direction=env.pull_direction_b,
                                target_tension_n=float(args.public_tension_target_n),
                                dt_s=float(env.physics_dt),
                                grip_safety_limit_n=float(
                                    args.public_grip_safety_limit_n
                                ),
                            )
                            diagnostics = two_stage_diagnostics.grip
                            tension_assist_active = (
                                two_stage_diagnostics.pull_velocity_m_s.abs() > 0.0
                            )
                        else:
                            executed_action, diagnostics = controller.control(
                                policy_action,
                                contact_force_n=metrics["grip_force_n"],
                                bilateral_contact=metrics["bilateral_contact"],
                                dt_s=float(env.physics_dt),
                                safety_force_limit_n=(
                                    oracle_safety_limit_n
                                    if oracle_active and oracle_safety_limit_n is not None
                                    else None
                                ),
                            )
                        if two_stage_diagnostics is None:
                            if float(
                                (executed_action[:, :9] - policy_action[:, :9])
                                .abs()
                                .max()
                                .item()
                            ) > 1e-8:
                                raise RuntimeError(
                                    "force controller modified a non-gripper action"
                                )
                        elif float(
                            (executed_action[:, 3:9] - policy_action[:, 3:9])
                            .abs()
                            .max()
                            .item()
                        ) > 1e-8:
                            raise RuntimeError(
                                "two-stage force controller modified orientation"
                            )
                        env._pre_physics_step(executed_action)
                        env._apply_action()
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(dt=env.physics_dt)
                        physics_step += 1
                        final_flags = env.get_task_flags()
                        terminal = is_terminal(final_flags, physics_step)
                        reward = shaped_reward(final_flags, terminal=terminal)
                        transition_reward += reward
                        episode_return += reward
                        grip = float(final_flags["grip_force_n"][0].item())
                        bilateral = bool(final_flags["bilateral_contact"][0].item())
                        bilateral_steps += int(bilateral)
                        if bilateral:
                            left_force = float(final_flags["left_normal_force_n"][0].item())
                            right_force = float(final_flags["right_normal_force_n"][0].item())
                            force_sum += grip
                            force_error_sum += (grip - attempted_range.center_n) ** 2
                            left_force_sum += left_force
                            right_force_sum += right_force
                            damage_risk_steps += int(
                                grip >= 0.85 * float(final_flags["damage_threshold_n"][0].item())
                            )
                            public_safety_risk_steps += int(
                                grip >= 0.85 * float(args.public_grip_safety_limit_n)
                            )
                        controller_active_steps += int(diagnostics.active[0].item())
                        safety_override_steps += int(diagnostics.safety_override[0].item())
                        oracle_tension_assist_steps += int(
                            tension_assist_active[0].item()
                        )
                        if physics_step == 1 or physics_step % 4 == 0 or terminal:
                            row = {
                                "physics_step": physics_step,
                                "chunk_action_index": chunk_action_index,
                                "grip_force_n": grip,
                                "left_force_n": float(final_flags["left_normal_force_n"][0].item()),
                                "right_force_n": float(final_flags["right_normal_force_n"][0].item()),
                                "pedicle_tension_n": float(final_flags["stem_tension_n"][0].item()),
                                "bilateral_contact": bilateral,
                                "controller_active": bool(diagnostics.active[0].item()),
                                "safety_override": bool(diagnostics.safety_override[0].item()),
                                "safety_threshold_n": float(
                                    diagnostics.safety_threshold_n[0].item()
                                ),
                                "oracle_tension_assist": bool(
                                    tension_assist_active[0].item()
                                ),
                                "two_stage_controller": two_stage_diagnostics is not None,
                                "two_stage_pull_offset_m": (
                                    0.0
                                    if two_stage_diagnostics is None
                                    else float(
                                        two_stage_diagnostics.pull_offset_m[0].item()
                                    )
                                ),
                                "oracle_force_balance_active": (
                                    False
                                    if force_balance_diagnostics is None
                                    else bool(force_balance_diagnostics.active[0].item())
                                ),
                                "oracle_force_balance_offset_m": (
                                    0.0
                                    if force_balance_diagnostics is None
                                    else float(
                                        force_balance_diagnostics.alignment_offset_m[0].item()
                                    )
                                ),
                                "dsrl_active": dsrl_active,
                                "residual_mode": (
                                    "bottle_style_lateral"
                                    if bottle_layout is not None
                                    else "semantic_xy"
                                    if args.method in SEMANTIC_XY_METHODS
                                    else "semantic"
                                    if semantic_layout is not None
                                    else "flow_latent"
                                ),
                                "residual_correction": correction[0].detach().cpu().tolist(),
                                "noise_correction": correction[0].detach().cpu().tolist(),
                                "lateral_residual_offset_m": (
                                    float(
                                        correction[0, 0].item()
                                        * bottle_layout.maximum_offset_m
                                    )
                                    if bottle_layout is not None
                                    else 0.0
                                ),
                                "force_balance_prior_used": (
                                    balance_prior_value
                                    if bottle_layout is not None
                                    else 0.0
                                ),
                            }
                            force_trace.append(row)
                            trajectory_stream.write(json.dumps(row, allow_nan=False) + "\n")
                        if terminal:
                            break
                        if (chunk_physics_index + 1) % args.action_repeat == 0:
                            # The BC demonstrations are sampled once per action
                            # (four physics steps). Preserve that observation
                            # history while deferring DINO/Flow inference until
                            # the full selected chunk has been executed.
                            env.sim.render()
                            state_history.append(policy_state(env))
                            image_history.append(
                                torch.stack(
                                    (
                                        camera_rgb_224(env, "wrist"),
                                        camera_rgb_224(env, "third"),
                                    ),
                                    dim=1,
                                )
                            )

                    if terminal:
                        next_policy_observation = policy_observation
                        next_dsrl_active = False
                        next_action_context = action_context
                    else:
                        next_phase = min(physics_step / float(args.max_episode_steps), 1.0)
                        next_model_obs = model_observation(
                            state_history, image_history, normalizer, next_phase
                        )
                        with torch.inference_mode():
                            next_cond_tokens, next_visual_xy = model.encode_observation(next_model_obs)
                        next_metrics = env.get_contact_force_metrics()
                        next_flags = env.get_task_flags()
                        next_policy_observation = actor_observation(
                            model.dsrl_observation(next_cond_tokens),
                            visual_xy=next_visual_xy,
                            normalized_state=next_model_obs[common.STATE_KEY][:, -1],
                            explicit_spatial_features=(
                                args.method in PREPULL_SEMANTIC_METHODS
                                or bottle_layout is not None
                            ),
                            phase=next_phase,
                            metrics=next_metrics,
                            flags=next_flags,
                            force_range=(
                                attempted_range
                                if bottle_layout is not None or not oracle_enabled
                                else actor_force_context
                            ),
                            bottle_style_force_features=bottle_layout is not None,
                        )
                        next_dsrl_active = residual_is_active(
                            physics_step=physics_step,
                            start_step=args.dsrl_start_step,
                            phase=next_phase,
                            pull_start_phase=args.pull_z_start_phase,
                            stem_attached=bool(next_flags["stem_attached"][0].item()),
                            semantic_residual=args.method in PREPULL_SEMANTIC_METHODS,
                        ) and (
                            bottle_layout is None
                            or bool(next_metrics["bilateral_contact"][0].item())
                        )
                        next_action_context = (
                            bottle_style_critic_context(
                                next_policy_observation,
                                force_range=attempted_range,
                                grip_force_n=next_metrics["grip_force_n"],
                                pedicle_tension_n=next_flags["stem_tension_n"],
                            )
                            if bottle_layout is not None
                            else None
                        )
                    if dsrl_active and not args.freeze_dsrl and not args.evaluation_mode:
                        replay_index = replay.add(
                            SACBatch(
                                observation=policy_observation.detach().cpu(),
                                action=correction.detach().cpu(),
                                reward=torch.tensor(
                                    [transition_reward / 10.0], dtype=torch.float32
                                ),
                                next_observation=next_policy_observation.detach().cpu(),
                                done=torch.tensor(
                                    [float(terminal or not next_dsrl_active)], dtype=torch.float32
                                ),
                                action_context=(
                                    action_context.detach().cpu()
                                    if action_context is not None
                                    else None
                                ),
                                next_action_context=(
                                    next_action_context.detach().cpu()
                                    if next_action_context is not None
                                    else None
                                ),
                            )
                        )
                        if (
                            args.method in DELAYED_CREDIT_METHODS
                            and not terminal
                            and not next_dsrl_active
                        ):
                            pending_terminal_credit_index = replay_index
                        total_transitions += 1
                        if replay.size >= args.learning_starts:
                            for _ in range(args.updates_per_decision):
                                sampled_batch = replay.sample(args.batch_size)
                                if bottle_layout is not None:
                                    action_prior = torch.full_like(
                                        sampled_batch.action, balance_prior_value
                                    )
                                    last_update = agent.update(
                                        sampled_batch,
                                        action_prior=action_prior,
                                        action_prior_mask=torch.ones_like(action_prior),
                                        action_prior_weight=args.force_balance_prior_weight,
                                    )
                                else:
                                    last_update = agent.update(sampled_batch)
                    if terminal:
                        break
                    cond_tokens, visual_xy = next_cond_tokens, next_visual_xy
                    normalized_state = next_model_obs[common.STATE_KEY][:, -1]

            raw_outcome = classify(final_flags)
            outcome = (
                "no_contact"
                if raw_outcome == "no_detach" and bilateral_steps == 0
                else "no_detach_with_contact"
                if raw_outcome == "no_detach"
                else raw_outcome
            )
            if (
                args.method in DELAYED_CREDIT_METHODS
                and pending_terminal_credit_index is not None
            ):
                replay.add_reward(
                    pending_terminal_credit_index,
                    terminal_outcome_reward(raw_outcome) / 10.0,
                )
            contact_count = max(bilateral_steps, 1)
            mean_force = force_sum / contact_count
            mean_left_force = left_force_sum / contact_count
            mean_right_force = right_force_sum / contact_count
            force_rmse = math.sqrt(force_error_sum / contact_count)
            damage_risk_fraction = damage_risk_steps / contact_count
            public_safety_risk_fraction = public_safety_risk_steps / contact_count
            balance_prior_used = balance_prior_value
            if bottle_layout is not None:
                terminal_left_force = float(
                    final_flags["left_normal_force_n"][0].item()
                )
                terminal_right_force = float(
                    final_flags["right_normal_force_n"][0].item()
                )
                balance_prior_target = force_balance_prior_target(
                    left_force_n=terminal_left_force,
                    right_force_n=terminal_right_force,
                    force_scale_n=float(args.public_grip_safety_limit_n),
                    gain=args.force_balance_prior_gain,
                    limit=args.force_balance_prior_limit,
                )
                balance_prior_value = (
                    (1.0 - args.force_balance_prior_smoothing) * balance_prior_value
                    + args.force_balance_prior_smoothing * balance_prior_target
                )
            else:
                balance_prior_used = 0.0
                balance_prior_target = 0.0
            frame_stem = f"episode_{episode_index:03d}_{outcome}"
            diagnostic_path = frame_dir / f"{frame_stem}_diagnostic.png"
            wrist_path = frame_dir / f"{frame_stem}_wrist.png"
            third_path = frame_dir / f"{frame_stem}_third.png"
            frame_path = frame_dir / f"{frame_stem}_advisor.png"
            capture_advisor_visual(
                diagnostic_path,
                trace=force_trace,
                outcome=outcome,
                force_range=attempted_range,
                public_safety_limit_n=(
                    oracle_safety_limit_n
                    if oracle_active and oracle_safety_limit_n is not None
                    else float(args.physical_force_bounds[1])
                ),
            )
            env.sim.render()
            save_camera_frame(env, "wrist", wrist_path)
            save_camera_frame(env, "third", third_path)
            compose_advisor_visual(
                frame_path,
                wrist_path=wrist_path,
                third_path=third_path,
                diagnostic_path=diagnostic_path,
            )
            root_pos_w = env._robot.data.root_link_pos_w[0]
            raspberry_pos_b = env.grape.data.root_pos_w[0] - root_pos_w
            gripper_offset_m = (
                raspberry_pos_b - env._gripper_center_pos_b()[0]
            ).detach().cpu().tolist()
            episode_record: dict[str, Any] = {
                "episode_index": episode_index,
                "seed": episode_seed,
                "method": args.method,
                "force_range_source": args.force_range_source,
                "controller_mode": args.controller_mode,
                "berry_profile": args.berry_profile,
                "outcome": outcome,
                "success": outcome == "success",
                "steps": physics_step,
                "policy_decisions": policy_decisions,
                "executed_chunk_actions": executed_chunk_actions,
                "chunk_execute_steps": int(args.chunk_execute_steps),
                "action_repeat": int(args.action_repeat),
                "flow_num_inference_steps": int(args.num_inference_steps),
                "return": episode_return,
                "attempted_force_range_n": attempted_range.as_list(),
                "target_force_n": attempted_range.center_n,
                "oracle_active": oracle_active,
                "actor_force_context_n": (
                    attempted_range.as_list()
                    if bottle_layout is not None or not oracle_enabled
                    else actor_force_context.as_list()
                ),
                "oracle_control_mode": (
                    args.oracle_control_mode if oracle_enabled else None
                ),
                "oracle_safety_limit_n": (
                    oracle_safety_limit_n if oracle_active else None
                ),
                "mean_bilateral_grip_force_n": mean_force,
                "mean_bilateral_left_force_n": mean_left_force,
                "mean_bilateral_right_force_n": mean_right_force,
                "force_rmse_n": force_rmse,
                "peak_grip_force_n": float(env.peak_grip_force_n[0].item()),
                "peak_pedicle_tension_n": float(env.peak_stem_tension_n[0].item()),
                "bilateral_contact_fraction": bilateral_steps / max(physics_step, 1),
                "damage_risk_fraction": damage_risk_fraction,
                "public_safety_risk_fraction": public_safety_risk_fraction,
                "raspberry_relative_to_gripper_m": gripper_offset_m,
                "damage_threshold_n": float(env.damage_threshold_n[0].item()),
                "detach_threshold_n": float(env.detach_threshold_n[0].item()),
                "controller_active_steps": controller_active_steps,
                "safety_override_steps": safety_override_steps,
                "oracle_tension_assist_steps": oracle_tension_assist_steps,
                "two_stage_pull_steps": (
                    oracle_tension_assist_steps
                    if args.controller_mode == "two_stage"
                    else 0
                ),
                "residual_mode": (
                    "bottle_style_lateral"
                    if bottle_layout is not None
                    else "semantic_xy"
                    if args.method in SEMANTIC_XY_METHODS
                    else "semantic"
                    if semantic_layout is not None
                    else "flow_latent"
                ),
                "mean_abs_residual_correction": (
                    correction_abs_sum / max(dsrl_active_decisions, 1)
                ).detach().cpu().tolist(),
                "mean_abs_lateral_residual_m": (
                    float(
                        correction_abs_sum[0].item()
                        / max(dsrl_active_decisions, 1)
                        * bottle_layout.maximum_offset_m
                    )
                    if bottle_layout is not None
                    else 0.0
                ),
                "max_abs_lateral_residual_m": max_abs_lateral_residual_m,
                "force_balance_prior_used": balance_prior_used,
                "force_balance_prior_target": balance_prior_target,
                "next_force_balance_prior": balance_prior_value,
                "exploration_scale": exploration,
                "dsrl_active_decisions": dsrl_active_decisions,
                "replay_size": replay.size,
                "sac_updates": agent.updates,
                "delayed_terminal_credit": pending_terminal_credit_index is not None,
                "last_update": last_update,
                "trajectory": str(trajectory_path.resolve()),
                "advisor_frame": str(frame_path.resolve()),
                "wrist_frame": str(wrist_path.resolve()),
                "third_person_frame": str(third_path.resolve()),
                "diagnostic_frame": str(diagnostic_path.resolve()),
            }
            sampled_trace = force_trace[:: max(1, len(force_trace) // 40)][:40]
            request = {
                "physical_force_bounds_n": list(map(float, args.physical_force_bounds)),
                "current_force_range_n": attempted_range.as_list(),
                "profile_contract": (
                    "berry/pedicle properties persist for the full run; their numeric values are hidden"
                ),
                "episode": {
                    "episode_index": episode_index,
                    "outcome": outcome,
                    "failure_subtype": None if outcome == "success" else outcome,
                    "success": outcome == "success",
                    "mean_bilateral_grip_force_n": mean_force,
                    "mean_bilateral_left_force_n": mean_left_force,
                    "mean_bilateral_right_force_n": mean_right_force,
                    "force_rmse_n": force_rmse,
                    "peak_grip_force_n": episode_record["peak_grip_force_n"],
                    "peak_pedicle_tension_n": episode_record["peak_pedicle_tension_n"],
                    "bilateral_contact_fraction": episode_record["bilateral_contact_fraction"],
                    "public_safety_risk_fraction": public_safety_risk_fraction,
                    "raspberry_relative_to_gripper_m": gripper_offset_m,
                    "force_trace_sample": sampled_trace,
                },
                "recent_history": list(recent_history),
            }
            response: dict[str, Any]
            vlm_error: str | None = None
            vlm_called = False
            if args.force_range_source == "oracle":
                response = {
                    "force_range_n": attempted_range.as_list(),
                    "failure_cause": "not_applicable",
                    "force_assessment": (
                        "simulator_ground_truth" if oracle_active else "shared_dsrl_warmup"
                    ),
                    "rationale": (
                        (
                            "Fixed oracle interval computed from the true profile damage threshold; "
                            "no VLM was called"
                            if args.oracle_control_mode == "fixed_interval"
                            else "Ground-truth damage threshold supplied only to the per-step safety "
                            "shield while the baseline DSRL force target was retained; no VLM was called"
                            if args.oracle_control_mode == "safety_shield"
                            else "Ground-truth grip and stem forces drove staged grip stabilization "
                            "and tension-regulated pulling; no VLM was called"
                            if args.oracle_control_mode == "two_stage"
                            else "Ground-truth grip and stem forces enabled late low-tension pull "
                            "assistance while the baseline DSRL force target was retained; no VLM was called"
                        )
                        if oracle_active
                        else "Baseline force interval retained during the shared DSRL warm-up; "
                        "no VLM was called"
                    ),
                    "confidence": 1.0,
                }
                vlm_error = None
            elif args.force_range_source == "fixed":
                response = {
                    "force_range_n": attempted_range.as_list(),
                    "failure_cause": "not_applicable",
                    "force_assessment": "fixed_protocol_range",
                    "rationale": "Fixed-range protocol arm; no VLM was called",
                    "confidence": 1.0,
                }
                vlm_error = None
            elif not should_call_vlm(
                force_range_source=args.force_range_source,
                call_policy=args.vlm_call_policy,
                outcome=outcome,
            ):
                response = {
                    "force_range_n": attempted_range.as_list(),
                    "failure_cause": "success",
                    "force_assessment": "not_evaluated",
                    "rationale": (
                        "Successful episode; failure-only VLM policy retained the current range "
                        "without an advisor call"
                    ),
                    "confidence": 1.0,
                }
            else:
                vlm_called = True
                try:
                    response = advisor.recommend(
                        episode_index=episode_index,
                        image_paths=[frame_path],
                        request=request,
                    )
                except Exception as exc:
                    response = {
                        "force_range_n": attempted_range.as_list(),
                        "failure_cause": "unknown",
                        "force_assessment": "unknown",
                        "rationale": f"advisor failed: {type(exc).__name__}",
                        "confidence": 0.0,
                    }
                    vlm_error = f"{type(exc).__name__}: {exc}"
            raw_proposal = ForceRange(*map(float, response["force_range_n"]))
            if args.force_range_source == "oracle":
                guarded = GuardedForceProposal(
                    tuple(attempted_range.as_list()),
                    (
                        "simulator_ground_truth_force"
                        if oracle_active
                        else "shared_dsrl_warmup_force"
                    ),
                    (
                        "hold_oracle_force"
                        if oracle_active and args.oracle_control_mode == "fixed_interval"
                        else "hold_oracle_safety_shield_target"
                        if oracle_active and args.oracle_control_mode == "safety_shield"
                        else "hold_oracle_two_stage_force_target"
                        if oracle_active and args.oracle_control_mode == "two_stage"
                        else "hold_oracle_tension_assist_target"
                        if oracle_active
                        else "hold_shared_dsrl_warmup"
                    ),
                    "hold",
                    0,
                )
            elif args.method == "vlm_noop":
                guarded = GuardedForceProposal(
                    tuple(attempted_range.as_list()),
                    "forced_noop_control",
                    "hold_vlm_noop",
                    "hold",
                    0,
                )
            elif args.force_range_source == "fixed":
                guarded = GuardedForceProposal(
                    tuple(attempted_range.as_list()),
                    "fixed_protocol_range",
                    "hold_fixed_range",
                    "hold",
                    0,
                )
            elif not vlm_called:
                guarded = GuardedForceProposal(
                    tuple(attempted_range.as_list()),
                    "successful_episode_without_vlm_call",
                    "hold_success_without_vlm_call",
                    "hold",
                    0,
                )
            elif vlm_error is not None or advisor is None:
                guarded = GuardedForceProposal(
                    tuple(attempted_range.as_list()),
                    "advisor_unavailable",
                    "hold_advisor_error",
                    "hold",
                    0,
                )
            else:
                guarded = guard_force_proposal(
                    current_range_n=attempted_range.as_list(),
                    proposed_range_n=raw_proposal.as_list(),
                    physical_bounds_n=args.physical_force_bounds,
                    outcome=outcome,
                    bilateral_contact_fraction=episode_record["bilateral_contact_fraction"],
                    mean_left_force_n=mean_left_force,
                    mean_right_force_n=mean_right_force,
                    peak_grip_force_n=episode_record["peak_grip_force_n"],
                    damage_threshold_n=float(args.public_grip_safety_limit_n),
                    damage_risk_fraction=public_safety_risk_fraction,
                    recent_candidate_directions=tuple(recent_candidate_directions),
                    confirmations_required=args.guard_confirmations,
                    maximum_center_step_n=args.vlm_max_center_step_n,
                )
            recent_candidate_directions.append(guarded.candidate_direction)
            proposal = ForceRange(*guarded.force_range_n)
            decision = adapter.update_after_episode(
                episode_index=episode_index,
                proposed_range=(
                    proposal
                    if vlm_error is None
                    and (args.force_range_source != "vlm" or vlm_called)
                    else None
                ),
                raw_response=json.dumps(response, ensure_ascii=False),
                error=vlm_error,
            )
            if args.force_range_source == "vlm" and not vlm_called:
                decision["update_kind"] = "vlm_skipped_success_hold"
            decision.update(
                {
                    "method": args.method,
                    "force_range_source": args.force_range_source,
                    "controller_mode": args.controller_mode,
                    "vlm_call_policy": args.vlm_call_policy,
                    "vlm_called": vlm_called,
                    "raw_proposal_n": (
                        raw_proposal.as_list()
                        if args.force_range_source != "vlm" or vlm_called
                        else None
                    ),
                    "guard_evidence": guarded.evidence,
                    "guard_decision": guarded.decision,
                    "candidate_direction": guarded.candidate_direction,
                    "confirmation_streak": guarded.confirmation_streak,
                }
            )
            episode_record.update(
                {
                    "vlm_called": vlm_called,
                    "vlm_call_policy": args.vlm_call_policy,
                    "vlm_call_index": (
                        advisor.calls if vlm_called and advisor is not None else None
                    ),
                    "vlm_call_succeeded": (
                        vlm_called and vlm_error is None and advisor is not None
                    ),
                    "vlm_response": response,
                    "guarded_vlm_force_range_n": proposal.as_list(),
                    "guard_evidence": guarded.evidence,
                    "guard_decision": guarded.decision,
                    "candidate_direction": guarded.candidate_direction,
                    "confirmation_streak": guarded.confirmation_streak,
                    "next_force_range_n": decision["applied_range_n"],
                }
            )
            records.append(episode_record)
            recent_history.append(
                {
                    "episode_index": episode_index,
                    "outcome": outcome,
                    "attempted_range_n": attempted_range.as_list(),
                    "mean_force_n": mean_force,
                    "peak_force_n": episode_record["peak_grip_force_n"],
                    "peak_pedicle_tension_n": episode_record["peak_pedicle_tension_n"],
                    "bilateral_contact_fraction": episode_record["bilateral_contact_fraction"],
                    "mean_left_force_n": mean_left_force,
                    "mean_right_force_n": mean_right_force,
                    "public_safety_risk_fraction": public_safety_risk_fraction,
                    "vlm_called": vlm_called,
                    "guard_evidence": guarded.evidence,
                    "guard_decision": guarded.decision,
                    "next_range_n": decision["applied_range_n"],
                }
            )
            latest_checkpoint = checkpoint_dir / "latest.pt"
            save_training_checkpoint(
                latest_checkpoint,
                agent=agent,
                layout=layout,
                semantic_layout=semantic_layout,
                bottle_layout=bottle_layout,
                replay=replay,
                adapter=adapter,
                controller=controller,
                base_hash=base_hash,
                completed_episodes=episode_index + 1,
                replay_transitions=total_transitions,
                balance_prior_value=balance_prior_value,
                recent_history=recent_history,
                recent_candidate_directions=recent_candidate_directions,
                advisor=advisor,
                episode_records=records,
            )
            append_jsonl(output_root / "training_episodes.jsonl", episode_record)
            append_jsonl(output_root / "force_range_decisions.jsonl", decision)
            if (episode_index + 1) % args.checkpoint_every == 0:
                shutil.copy2(
                    latest_checkpoint,
                    checkpoint_dir / f"episode_{episode_index + 1:03d}.pt",
                )
            rolling = sum(int(item["success"]) for item in records[-10:]) / min(10, len(records))
            print(
                f"[ONLINE] episode={episode_index + 1}/{args.train_episodes} "
                f"outcome={outcome} rolling10={rolling:.1%} "
                f"force={episode_record['peak_grip_force_n']:.3f}N "
                f"range={attempted_range.as_list()}->{decision['applied_range_n']} "
                f"replay={replay.size} updates={agent.updates} "
                f"codex={'ok' if episode_record['vlm_call_succeeded'] else 'failed' if vlm_called else 'skipped'}",
                flush=True,
            )
    finally:
        env.close()

    if len(records) < args.train_episodes:
        incomplete = {
            "status": "incomplete_resumable",
            "completed_episodes": len(records),
            "target_episodes": args.train_episodes,
            "next_episode_seed": (
                args.train_seed
                if args.fixed_episode_seed
                else args.train_seed + len(records)
            ),
            "latest_checkpoint": str((checkpoint_dir / "latest.pt").resolve()),
            "vlm_calls": 0 if advisor is None else advisor.calls,
            "vlm_successful_calls": 0 if advisor is None else advisor.successful_calls,
        }
        (output_root / "interrupted_summary.json").write_text(
            json.dumps(incomplete, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("[INCOMPLETE] " + json.dumps(incomplete, ensure_ascii=False), flush=True)
        return

    final_checkpoint = checkpoint_dir / "final.pt"
    save_training_checkpoint(
        final_checkpoint,
        agent=agent,
        layout=layout,
        semantic_layout=semantic_layout,
        bottle_layout=bottle_layout,
        replay=replay,
        adapter=adapter,
        controller=controller,
        base_hash=base_hash,
        completed_episodes=len(records),
        replay_transitions=total_transitions,
        balance_prior_value=balance_prior_value,
        recent_history=recent_history,
        recent_candidate_directions=recent_candidate_directions,
        advisor=advisor,
        episode_records=records,
    )
    successes = sum(int(item["success"]) for item in records)
    failures = Counter(item["outcome"] for item in records if not item["success"])
    summary = {
        "protocol": args.protocol_name,
        "experiment": (
            "RaspberryPick privileged ground-truth force plus DSRL ablation"
            if oracle_enabled
            else "Repaired paired RaspberryPick online DSRL/VLM experiment"
        ),
        "post_training_method": args.method,
        "force_range_source": args.force_range_source,
        "controller_mode": args.controller_mode,
        "flow_num_inference_steps": int(args.num_inference_steps),
        "flow_chunk_size": int(model.config.chunk_size),
        "chunk_execute_steps": int(args.chunk_execute_steps),
        "action_repeat": int(args.action_repeat),
        "vlm_call_policy": args.vlm_call_policy,
        "force_adaptation": (
            (
                "oracle_fixed_from_simulator_ground_truth"
                if args.oracle_control_mode == "fixed_interval"
                else "oracle_safety_shield_from_simulator_ground_truth"
                if args.oracle_control_mode == "safety_shield"
                else "oracle_two_stage_grip_and_tension_from_simulator_ground_truth"
                if args.oracle_control_mode == "two_stage"
                else "oracle_tension_assist_from_simulator_ground_truth"
            )
            if oracle_enabled
            else "fixed"
            if args.method in NO_VLM_METHODS
            else args.method
        ),
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "task": TASK,
        "berry_profile": args.berry_profile,
        "profile_persistence": "entire_run",
        "profile_values_hidden_from_actor_and_vlm": True,
        "profile_values_exposed_only_to_force_oracle": oracle_enabled,
        "oracle_is_controller_only_intervention": (
            oracle_enabled
            and args.oracle_control_mode in {"fixed_interval", "safety_shield"}
        ),
        "oracle_control_mode": args.oracle_control_mode if oracle_enabled else None,
        "oracle_safety_limit_n": oracle_safety_limit_n,
        "oracle_tension_assist_start_phase": (
            float(args.oracle_tension_assist_start_phase)
            if oracle_enabled and args.oracle_control_mode in {"tension_assist", "two_stage"}
            else None
        ),
        "oracle_tension_assist_m": (
            float(args.oracle_tension_assist_m)
            if oracle_enabled and args.oracle_control_mode in {"tension_assist", "two_stage"}
            else None
        ),
        "oracle_two_stage_force_controller_config": (
            asdict(two_stage_controller.config)
            if two_stage_controller is not None
            else None
        ),
        "oracle_force_balance_alignment_config": (
            asdict(force_balance_aligner.config)
            if force_balance_aligner is not None
            else None
        ),
        "actor_force_context_n": (
            (
                list(effective_initial_force_range)
                if bottle_layout is not None
                else actor_force_context.as_list()
            )
            if oracle_enabled
            else None
        ),
        "force_advisor_source": (
            "simulator_ground_truth"
            if args.force_range_source == "oracle"
            else "codex"
            if args.force_range_source == "vlm"
            else "fixed"
        ),
        "training_seed": args.train_seed,
        "episode_seed_mode": "fixed" if args.fixed_episode_seed else "sequential",
        "completed_episodes": len(records),
        "successes": successes,
        "online_training_success_rate": successes / max(len(records), 1),
        "failure_counts": dict(failures),
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha256": base_hash,
        "base_checkpoint_weights": "raw_best_epoch",
        "dino_model": base_checkpoint["policy_config"]["dino_path"],
        "dino_frozen": True,
        "residual_mode": (
            "bottle_style_lateral"
            if bottle_layout is not None
            else "semantic_xy"
            if args.method in SEMANTIC_XY_METHODS
            else "semantic"
            if semantic_layout is not None
            else "flow_latent"
        ),
        "dsrl_frozen": bool(args.freeze_dsrl),
        "evaluation_mode": bool(args.evaluation_mode),
        "loaded_agent_checkpoint": (
            None if loaded_agent_checkpoint is None else str(loaded_agent_checkpoint)
        ),
        "loaded_agent_checkpoint_mode": loaded_agent_checkpoint_mode,
        "loaded_agent_checkpoint_source_method": loaded_agent_checkpoint_source_method,
        "loaded_agent_checkpoint_source_contract": loaded_agent_checkpoint_source_contract,
        "branch_from_dsrl_checkpoint": bool(args.branch_from_dsrl_checkpoint),
        "branch_resume_episode": (
            int(args.resume_episode) if args.branch_from_dsrl_checkpoint else None
        ),
        "branch_prefix_successes": (
            sum(int(item["success"]) for item in records[: args.resume_episode])
            if args.branch_from_dsrl_checkpoint
            else None
        ),
        "branch_continuation_episodes": (
            len(records) - args.resume_episode
            if args.branch_from_dsrl_checkpoint
            else None
        ),
        "branch_source_run": (
            str(branch_source_run) if branch_source_run is not None else None
        ),
        "branch_source_episode_log_sha256": branch_source_episode_log_sha256,
        "flow_initial_latent": (
            "zero"
            if semantic_layout is not None or bottle_layout is not None
            else "zero_plus_bounded_dsrl_correction"
        ),
        "noise_action_indices": list(layout.active_action_indices),
        "noise_residual_scale": layout.residual_scale,
        "dsrl_start_step": args.dsrl_start_step,
        "actor_observation_features": (
            "pooled_condition+visual_xy+normalized_state+physics+per_finger_target_errors"
            if bottle_layout is not None
            else "pooled_condition+visual_xy+normalized_state+physics"
            if args.method in PREPULL_SEMANTIC_METHODS
            else "pooled_condition+physics"
        ),
        "residual_active_until_phase": (
            float(args.pull_z_start_phase)
            if args.method in PREPULL_SEMANTIC_METHODS
            else None
        ),
        "bottle_style_lateral_residual": (
            None
            if bottle_layout is None
            else {
                **asdict(bottle_layout),
                "axis": "gripper_closing_axis_b",
                "activation": "bilateral_contact_after_dsrl_start_step",
                "critic_context": [
                    "force_target_low",
                    "force_target_high",
                    "measured_grip_force",
                    "measured_pedicle_tension",
                ],
                "force_balance_prior_gain": args.force_balance_prior_gain,
                "force_balance_prior_weight": args.force_balance_prior_weight,
                "force_balance_prior_limit": args.force_balance_prior_limit,
                "force_balance_prior_smoothing": args.force_balance_prior_smoothing,
                "final_prior_value": balance_prior_value,
            }
        ),
        "exploration_anneal_episodes": args.exploration_anneal_episodes,
        "sac_optimizer_updates": agent.updates,
        "sac_optimizer_updates_this_run": agent.updates - starting_agent_updates,
        "replay_transitions": total_transitions,
        "vlm_advisor": None if advisor is None else args.codex_model,
        "vlm_reasoning_effort": None if advisor is None else args.codex_reasoning_effort,
        "vlm_is_codex": advisor is not None,
        "vlm_calls": 0 if advisor is None else advisor.calls,
        "vlm_successful_calls": 0 if advisor is None else advisor.successful_calls,
        "requested_initial_force_range_n": list(map(float, args.initial_force_range)),
        "initial_force_range_n": list(effective_initial_force_range),
        "oracle_force_spec": None if oracle_force is None else asdict(oracle_force),
        "oracle_start_episode": (
            None if not oracle_enabled else int(args.oracle_start_episode)
        ),
        "final_force_range_n": adapter.current_range.as_list(),
        "physical_force_bounds_n": list(map(float, args.physical_force_bounds)),
        "public_grip_safety_limit_n": float(args.public_grip_safety_limit_n),
        "public_tension_target_n": float(args.public_tension_target_n),
        "force_bound_above_damage_threshold_allowed": bool(
            args.allow_force_bound_above_damage_threshold
        ),
        "damage_threshold_n": float(config.damage_grip_force_threshold_n),
        "detach_threshold_n": float(config.detach_force_threshold_n),
        "stem_stiffness_n_per_m": float(config.stem_stiffness_n_per_m),
        "stem_damping_n_s_per_m": float(config.stem_damping_n_s_per_m),
        "stem_lock_rotation_while_attached": bool(
            config.stem_lock_rotation_while_attached
        ),
        "reward_reference": "physical_environment_constraints_not_vlm_target",
        "position_randomization_m": list(map(float, args.position_range)),
        "physx_enhanced_determinism": bool(args.enhanced_determinism),
        "protocol_manifest": str(manifest_path.resolve()),
        "protocol_manifest_sha256": sha256(manifest_path),
        "resumed": bool(args.resume),
        "resume_start_episode": run_start_episode if args.resume else None,
        "final_checkpoint": str(final_checkpoint.resolve()),
        "elapsed_seconds": time.time() - started,
    }
    (output_root / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    incomplete_path = output_root / "interrupted_summary.json"
    if incomplete_path.exists():
        incomplete_path.unlink()
    artifact_hashes = {
        str(path.relative_to(output_root)): sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "file_hashes.json"
    }
    (output_root / "file_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[SUMMARY] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
