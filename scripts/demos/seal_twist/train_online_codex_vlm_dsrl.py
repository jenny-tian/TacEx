#!/usr/bin/env python3
"""Train SealTwist tactile DSRL with an online Codex visual advisor.

The SAC actor learns a bounded semantic tool-Y correction on top of the frozen
Flow Matching policy.  When enabled by the call policy, Codex reads a
source-grounded force/torque diagnostic image and structured trajectory summary,
then proposes the interval used by the next episode.  Torque adaptation changes
only the policy's maximum turn angle and keeps the true success window hidden
from the advisor.
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
from dataclasses import asdict, dataclass
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
GRAPE_HELPERS = ROOT / "scripts" / "reinforcement_learning" / "grape_pick"
if str(GRAPE_HELPERS) not in sys.path:
    sys.path.insert(0, str(GRAPE_HELPERS))

DEFAULT_FLOW_CHECKPOINT = ROOT / "exp_report" / "bottle" / "checkpoints" / "flow_bc_6d_force_v5.pt"
DEFAULT_SCHEMA = Path(__file__).with_name("codex_vlm_force_schema.json")
DEFAULT_TORQUE_SCHEMA = Path(__file__).with_name("codex_vlm_torque_schema.json")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--flow-checkpoint", type=Path, default=DEFAULT_FLOW_CHECKPOINT)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument(
    "--resume-run",
    type=Path,
    default=None,
    help=(
        "Continue a completed shorter run into a new output root. Existing episode logs, "
        "force-adapter state, networks and counters are restored; legacy checkpoints "
        "without replay/optimizer state use a short replay warm-up."
    ),
)
parser.add_argument("--train-episodes", type=int, default=100)
parser.add_argument("--train-seed", type=int, default=907700)
parser.add_argument(
    "--online-algorithm",
    choices=("dsrl", "residual_rl", "awr", "dppo"),
    default="dsrl",
    help=(
        "Online residual learner. All methods share the same observable state, "
        "reward, contact-phase data and semantic tool-Y action; only DSRL uses "
        "the privileged critic context and force-balance actor prior."
    ),
)
parser.add_argument("--max-episode-steps", type=int, default=900)
parser.add_argument("--chunk-execute-steps", type=int, default=10)
parser.add_argument("--action-repeat", type=int, default=4)
parser.add_argument("--integration-steps", type=int, default=10)
parser.add_argument("--action-smoothing", type=float, default=0.35)
parser.add_argument("--normalized-observation-clip", type=float, default=6.0)
parser.add_argument("--noise-residual-scale", type=float, default=0.50)
parser.add_argument("--learned-noise-steps", type=int, default=1)
parser.add_argument(
    "--noise-action-indices",
    type=int,
    nargs="+",
    default=(4,),
    help=(
        "Legacy Flow latent channel used to size the actor. Its route is disabled "
        "when the semantic force-balance residual is enabled."
    ),
)
parser.add_argument("--dsrl-start-step", type=int, default=350)
parser.add_argument("--initial-log-std", type=float, default=-2.5)
parser.add_argument(
    "--force-balance-y-scale-m",
    type=float,
    default=0.006,
    help=(
        "Maximum learned tool-Y residual. One actor coordinate is routed here "
        "because the frozen Flow latent has only micrometre-scale XY authority."
    ),
)
parser.add_argument("--force-balance-actor-coordinate", type=int, default=0)
parser.add_argument("--force-balance-prior-gain", type=float, default=0.25)
parser.add_argument("--force-balance-prior-weight", type=float, default=2.0)
parser.add_argument("--force-balance-prior-limit", type=float, default=0.20)
parser.add_argument("--force-balance-prior-smoothing", type=float, default=0.25)
parser.add_argument("--minimum-force-range", type=float, nargs=2, default=(4.7, 6.1))
parser.add_argument("--maximum-force-range", type=float, nargs=2, default=(9.0, 11.0))
parser.add_argument(
    "--force-threshold-schedule",
    choices=("iid", "persistent", "block"),
    default="persistent",
    help=(
        "Temporal process for the hidden force window. Persistent is the online-"
        "adaptation protocol; iid preserves the legacy non-identifiable benchmark."
    ),
)
parser.add_argument("--force-threshold-block-episodes", type=int, default=25)
parser.add_argument(
    "--force-threshold-profiles",
    type=float,
    nargs="*",
    default=(4.0, 9.0, 5.8, 11.0),
    metavar="MIN_MAX",
    help=(
        "Flattened coupled hidden (minimum, maximum) profiles. The calibrated default "
        "models soft [4,9] N and stiff [5.8,11] N seal batches. Pass the option with "
        "no values to use the independent minimum/maximum ranges instead."
    ),
)
parser.add_argument(
    "--force-threshold-profile-index",
    type=int,
    default=-1,
    help=(
        "Select one hidden profile deterministically for paired experiments; negative "
        "samples from the configured candidates. The selected index is not sent to the VLM."
    ),
)
parser.add_argument("--position-range", type=float, nargs=2, default=(0.010, 0.010))
parser.add_argument("--alignment-tolerance", type=float, default=0.012)
parser.add_argument("--initial-force-range", type=float, nargs=2, default=(7.2, 8.2))
parser.add_argument("--physical-force-bounds", type=float, nargs=2, default=(6.2, 8.8))
parser.add_argument("--force-range-minimum-width", type=float, default=0.40)
parser.add_argument("--force-range-smoothing", type=float, default=0.50)
parser.add_argument(
    "--adaptation-target",
    choices=("force", "torque"),
    default="force",
    help="Quantity inferred by the episode-level VLM; force preserves the legacy protocol.",
)
parser.add_argument(
    "--initial-torque-range-nm", type=float, nargs=2, default=(0.35, 0.50)
)
parser.add_argument(
    "--physical-torque-bounds-nm", type=float, nargs=2, default=(0.10, 0.60)
)
parser.add_argument("--torque-range-minimum-width-nm", type=float, default=0.08)
parser.add_argument("--torque-range-smoothing", type=float, default=1.0)
parser.add_argument("--vlm-max-torque-center-step-nm", type=float, default=0.08)
parser.add_argument(
    "--vlm-max-center-step-n",
    type=float,
    default=0.30,
    help=(
        "Maximum VLM target-center change before adapter smoothing; changes require "
        "symmetric tracking evidence or a persistent directional constraint label."
    ),
)
parser.add_argument(
    "--force-reward-reference",
    choices=("environment", "target"),
    default="environment",
    help=(
        "Use the actual task force constraint for reward shaping (recommended), or "
        "the VLM set-point for legacy reproduction."
    ),
)
parser.add_argument("--force-feedback-gain-m-per-n", type=float, default=3.0e-4)
parser.add_argument("--force-feedback-max-step-m", type=float, default=5.0e-4)
parser.add_argument("--gripper-width-limits-m", type=float, nargs=2, default=(0.019, 0.040))
parser.add_argument("--learning-starts", type=int, default=256)
parser.add_argument(
    "--resume-learning-starts",
    type=int,
    default=64,
    help="Replay transitions required after resuming a legacy checkpoint without replay state.",
)
parser.add_argument("--replay-capacity", type=int, default=50000)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--updates-per-decision", type=int, default=1)
parser.add_argument("--learning-rate", type=float, default=3.0e-4)
parser.add_argument("--exploration-start", type=float, default=1.0)
parser.add_argument("--exploration-end", type=float, default=0.15)
parser.add_argument(
    "--exploration-anneal-episodes",
    type=int,
    default=None,
    help=(
        "Episodes used to anneal from exploration-start to exploration-end. "
        "Defaults to train-episodes; paired continuation baselines use 30."
    ),
)
parser.add_argument("--checkpoint-every", type=int, default=10)
parser.add_argument("--policy-device", choices=("auto", "cpu", "cuda"), default="cuda")
parser.add_argument("--codex-cli", type=Path, default=None)
parser.add_argument("--codex-model", default="gpt-5.6-sol")
parser.add_argument("--codex-reasoning-effort", choices=("low", "medium", "high", "xhigh"), default="high")
parser.add_argument("--codex-output-schema", type=Path, default=DEFAULT_SCHEMA)
parser.add_argument(
    "--codex-torque-output-schema", type=Path, default=DEFAULT_TORQUE_SCHEMA
)
parser.add_argument("--codex-timeout-s", type=float, default=300.0)
parser.add_argument("--codex-max-attempts", type=int, default=3)
parser.add_argument(
    "--vlm-call-policy",
    choices=("every_episode", "failure_only"),
    default="every_episode",
)
parser.add_argument("--disable-vlm", action="store_true")
parser.add_argument("--physx-contact-capacity", type=int, default=2**20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.train_episodes < 1:
    parser.error("train-episodes must be positive")
if args.online_algorithm != "dsrl" and not args.disable_vlm:
    parser.error("non-DSRL baselines require --disable-vlm")
if args.online_algorithm != "dsrl" and args.resume_run is not None:
    parser.error("resume-run is currently supported only for DSRL checkpoints")
if args.resume_learning_starts < 1:
    parser.error("resume-learning-starts must be positive")
if args.exploration_anneal_episodes is not None and args.exploration_anneal_episodes < 1:
    parser.error("exploration-anneal-episodes must be positive")
if min(args.action_repeat, args.integration_steps, args.chunk_execute_steps) < 1:
    parser.error("action-repeat, integration-steps and chunk-execute-steps must be positive")
if args.dsrl_start_step < 0:
    parser.error("dsrl-start-step must be non-negative")
if args.physx_contact_capacity < 2**16:
    parser.error("physx-contact-capacity is implausibly small")
if args.codex_max_attempts < 1:
    parser.error("codex-max-attempts must be positive")
if args.vlm_max_center_step_n < 0.0:
    parser.error("vlm-max-center-step-n must be non-negative")
if args.vlm_max_torque_center_step_nm < 0.0:
    parser.error("vlm-max-torque-center-step-nm must be non-negative")
if args.adaptation_target == "torque" and args.resume_run is not None:
    parser.error("torque adaptation does not yet support --resume-run")
if args.force_threshold_block_episodes < 1:
    parser.error("force-threshold-block-episodes must be positive")
if len(args.force_threshold_profiles) % 2:
    parser.error("force-threshold-profiles must contain MIN MAX pairs")
force_threshold_profiles = tuple(
    (float(args.force_threshold_profiles[index]), float(args.force_threshold_profiles[index + 1]))
    for index in range(0, len(args.force_threshold_profiles), 2)
)
if any(low >= high for low, high in force_threshold_profiles):
    parser.error("every force-threshold profile must satisfy MIN < MAX")
if args.force_threshold_profile_index >= len(force_threshold_profiles):
    parser.error("force-threshold-profile-index is outside force-threshold-profiles")
if args.force_threshold_profile_index >= 0 and not force_threshold_profiles:
    parser.error("force-threshold-profile-index requires force-threshold-profiles")
if args.force_threshold_schedule == "block" and args.force_threshold_profile_index >= 0:
    parser.error("block schedule cannot use a fixed force-threshold-profile-index")
if min(
    args.force_balance_y_scale_m,
    args.force_balance_prior_gain,
    args.force_balance_prior_weight,
    args.force_balance_prior_limit,
) < 0.0:
    parser.error("force-balance scale, gain and prior weight must be non-negative")
if not 0.0 <= args.force_balance_prior_smoothing <= 1.0:
    parser.error("force-balance-prior-smoothing must be in [0, 1]")
if args.codex_model != "gpt-5.6-sol" or args.codex_reasoning_effort != "high":
    parser.error("This experiment contract requires gpt-5.6-sol with high reasoning effort")
args.enable_cameras = False

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import torch
from PIL import Image, ImageDraw

from grape_force_control import ForceRange, ForceRangeAdapterConfig, OnlineForceRangeAdapter
from learning import (
    AWRAgent,
    DiffusionPPOAgent,
    ReplayBuffer,
    SACBatch,
    SACResidualAgent,
    discounted_returns,
)
from online_rl_support import (
    ACTOR_OBSERVATION_DIM,
    CRITIC_FORCE_CONTEXT_DIM,
    OnlineTorqueRangeAdapter,
    TorqueRange,
    TorqueRangeAdapterConfig,
    critic_force_context,
    exploration_scale_for_episode,
    force_balance_action_prior,
    force_conditioned_observation,
    force_evidence,
    force_shaping_term,
    guard_vlm_force_proposal,
    guard_vlm_torque_proposal,
)
from tacex_tasks.seal_twist.flow_policy import (
    PREVIOUS_GRIPPER_WIDTH_OBSERVATION_INDEX,
    environment_flow_observation,
    load_flow_checkpoint,
)
from tacex_tasks.seal_twist.seal_twist_env import SealTwistEnv
from tacex_tasks.seal_twist.seal_twist_env_cfg import SealTwistEnvCfg, target_angle_for_torque


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def recent_history_entry(record: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the exact compact VLM history representation from an episode log."""

    return {
        "episode_index": int(record["episode_index"]),
        "outcome": str(record["outcome"]),
        "attempted_range_n": list(record["attempted_force_range_n"]),
        "mean_force_n": float(record["mean_bilateral_finger_force_n"]),
        "peak_force_n": max(
            float(record["peak_left_force_n"]),
            float(record["peak_right_force_n"]),
        ),
        "peak_torque_nm": float(record["peak_torque_nm"]),
        "final_left_force_n": float(record["final_left_finger_force_n"]),
        "final_right_force_n": float(record["final_right_finger_force_n"]),
        "force_evidence": str(record["terminal_force_evidence"]),
        "force_constraint_violation": str(
            record.get("force_constraint_violation", "none")
        ),
        "next_range_n": list(record["next_force_range_n"]),
    }


@dataclass(frozen=True)
class FlowNoiseDSRLLayout:
    horizon: int
    action_dim: int
    learned_noise_steps: int
    residual_scale: float
    active_action_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.action_dim < 1 or self.learned_noise_steps < 1:
            raise ValueError("Flow-noise dimensions must be positive")
        if self.learned_noise_steps > self.horizon:
            raise ValueError("learned-noise-steps exceeds the Flow horizon")
        if not self.active_action_indices:
            raise ValueError("at least one active Flow latent channel is required")
        if len(set(self.active_action_indices)) != len(self.active_action_indices):
            raise ValueError("noise-action-indices must be unique")
        if min(self.active_action_indices) < 0 or max(self.active_action_indices) >= self.action_dim:
            raise ValueError("noise-action-indices are outside the Flow action width")
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0.0:
            raise ValueError("noise-residual-scale must be finite and non-negative")

    @property
    def correction_dim(self) -> int:
        return self.learned_noise_steps * len(self.active_action_indices)

    def expand(self, correction: torch.Tensor) -> torch.Tensor:
        if correction.ndim != 2 or correction.shape[-1] != self.correction_dim:
            raise ValueError(
                f"correction must have shape [B, {self.correction_dim}], got {tuple(correction.shape)}"
            )
        learned = correction.reshape(
            len(correction), self.learned_noise_steps, len(self.active_action_indices)
        )
        short = correction.new_zeros((len(correction), self.learned_noise_steps, self.action_dim))
        short[:, :, list(self.active_action_indices)] = learned
        if self.learned_noise_steps == self.horizon:
            return short
        padding = short[:, -1:, :].expand(-1, self.horizon - self.learned_noise_steps, -1)
        return torch.cat((short, padding), dim=1)

    def compose_zero_anchored_noise(self, correction: torch.Tensor) -> torch.Tensor:
        return self.residual_scale * self.expand(correction)


class CodexVisualForceAdvisor:
    """Schema-constrained Codex vision advisor for force or torque adaptation."""

    def __init__(self, output_root: Path) -> None:
        executable = str(args.codex_cli) if args.codex_cli is not None else shutil.which("codex")
        if not executable:
            raise FileNotFoundError("Codex CLI was not found")
        self.executable = str(Path(executable).expanduser().resolve())
        schema_path = (
            args.codex_torque_output_schema
            if args.adaptation_target == "torque"
            else args.codex_output_schema
        )
        self.schema = schema_path.expanduser().resolve()
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
        if args.adaptation_target == "torque":
            return (
                "You are the visual-tactile torque advisor inside an online robotic "
                "bottle-cap tightening experiment. The acceptable seal-torque window is "
                "hidden and persistent for this run. Infer the target torque interval for "
                "the next episode from the attached diagnostic image, the directional "
                "torque-constraint label, and recent history. An under-torque label supports "
                "increasing the interval centre; an over-torque label supports decreasing it. "
                "Force-constraint, bilateral-imbalance, alignment, trajectory, and no-contact "
                "failures are not evidence for changing torque. Preserve approximately the "
                "current interval width and prefer a decisive 0.08–0.12 N m centre change after "
                "a directional torque failure. Keep both endpoints inside "
                "physical_torque_bounds_nm. Do not infer or state an unobserved ground-truth "
                "window, do not use tools, and do not inspect repository files. Return only "
                "the schema-conforming JSON.\n\n"
                + json.dumps(request, ensure_ascii=False, sort_keys=True)
            )
        schedule = str(request["force_threshold_schedule"]["mode"])
        if schedule == "iid":
            temporal_guidance = (
                "The hidden limits are sampled independently every episode, so a completed "
                "episode cannot reveal the next episode's limits. Use only the population "
                "ranges. Only uniformly low force may justify a small increase, and only "
                "uniformly high force may justify a decrease."
            )
        else:
            persists = bool(
                request["force_threshold_schedule"]["persists_next_episode"]
            )
            temporal_guidance = (
                "The hidden force window represents a persistent bottle/seal material batch. "
                "Completed episodes are evidence about the same hidden window used next, so "
                "use the directional force-constraint label and recent history to adapt the "
                "set-point. A reported under-force constraint supports increasing the centre; "
                "a reported over-force constraint supports decreasing it. Prefer a 0.15-0.30 N "
                "centre change after a force-constraint failure, and hold after success or a "
                "trajectory/alignment failure."
                if persists
                else
                "The material block changes before the next episode. Hold the force interval "
                "because the just-completed episode does not identify the next block's window."
            )
        return (
            "You are the visual-tactile force advisor inside an online robotic bottle-cap "
            "tightening experiment. Inspect the attached source-grounded diagnostic image "
            "and the structured episode data below. Recommend the fingertip-force interval "
            "for the next episode. Success requires bilateral contact, a final seal torque "
            "inside the stated torque window, and both fingertip forces inside the hidden "
            "material-dependent limits. "
            + temporal_guidance
            + " "
            "Judge the two fingertips separately: one low and the other high is bilateral force "
            "imbalance and must also be handled by the DSRL lateral-balancing policy. "
            "Alignment failures are not evidence that force is too low. "
            "If force tracks its target but torque remains low, classify the cause as trajectory. "
            "Keep changes conservative and keep both endpoints inside physical_force_bounds_n. "
            "Do not use tools or inspect repository files. Return only the schema-conforming JSON.\n\n"
            + json.dumps(request, ensure_ascii=False, sort_keys=True)
        )

    def recommend(self, *, episode_index: int, image_path: Path, request: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, args.codex_max_attempts + 1):
            stem = f"episode_{episode_index:03d}_attempt_{attempt:02d}"
            response_path = self.output_dir / f"{stem}_response.json"
            log_path = self.output_dir / f"{stem}_cli.log"
            if response_path.exists() or log_path.exists():
                raise FileExistsError(
                    f"Codex VLM output already exists for episode {episode_index}, attempt {attempt}"
                )
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
                "--image",
                str(image_path.resolve()),
                "--output-schema",
                str(self.schema),
                "-o",
                str(response_path),
                "-",
            ]
            started = time.time()
            self.calls += 1
            try:
                completed = subprocess.run(
                    command,
                    input=self._prompt(request),
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=args.codex_timeout_s,
                )
                elapsed = time.time() - started
                combined = completed.stdout + "\n" + completed.stderr
                log_path.write_text(combined, encoding="utf-8")
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Codex VLM call exited with code {completed.returncode}"
                    )
                if f"model: {args.codex_model}" not in combined:
                    raise RuntimeError("Codex CLI audit did not confirm the requested model")
                if f"reasoning effort: {args.codex_reasoning_effort}" not in combined:
                    raise RuntimeError(
                        "Codex CLI audit did not confirm the requested reasoning effort"
                    )
                response = json.loads(response_path.read_text(encoding="utf-8"))
                response_key = (
                    "torque_range_nm"
                    if args.adaptation_target == "torque"
                    else "force_range_n"
                )
                bounds_key = (
                    "physical_torque_bounds_nm"
                    if args.adaptation_target == "torque"
                    else "physical_force_bounds_n"
                )
                low, high = map(float, response[response_key])
                physical_low, physical_high = map(float, request[bounds_key])
                if not physical_low <= low < high <= physical_high:
                    raise ValueError(
                        f"Codex {args.adaptation_target} range {[low, high]} violates physical bounds "
                        f"{[physical_low, physical_high]}"
                    )
                record = {
                    "advisor_call_index": self.calls,
                    "episode_index": episode_index,
                    "attempt": attempt,
                    "model": args.codex_model,
                    "reasoning_effort": args.codex_reasoning_effort,
                    "codex_cli_version": self.version,
                    "image": str(image_path.resolve()),
                    "request": request,
                    "response": response,
                    "elapsed_seconds": elapsed,
                    "success": True,
                    "model_audit_confirmed": True,
                    "reasoning_effort_audit_confirmed": True,
                    "cli_log": str(log_path.resolve()),
                }
                append_jsonl(self.audit_path, record)
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
                        "elapsed_seconds": time.time() - started,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "cli_log": str(log_path.resolve()),
                    },
                )
        raise RuntimeError(
            f"Codex VLM failed after {args.codex_max_attempts} attempts"
        ) from last_error


def classify(flags: dict[str, torch.Tensor]) -> str:
    if bool(flags["success"][0].item()):
        return "success"
    if bool(flags["over_force"][0].item()) or bool(flags["over_torqued"][0].item()):
        return "over_force"
    if bool(flags["alignment_failure"][0].item()) or bool(flags["slipped"][0].item()):
        return "alignment_failure"
    return "under_force"


def force_constraint_violation(flags: dict[str, torch.Tensor]) -> str:
    """Return directional fingertip feedback without exposing threshold values."""

    if bool(flags["above_maximum_force_threshold"][0].item()):
        return "over"
    if bool(flags["below_minimum_force_threshold"][0].item()):
        return "under"
    return "none"


def torque_constraint_violation(flags: dict[str, torch.Tensor]) -> str:
    """Return only the directional torque failure label, never its hidden bounds."""

    if bool(flags["over_torqued"][0].item()):
        return "over"
    if bool(flags["under_torqued"][0].item()):
        return "under"
    return "none"


def threshold_persists_next_episode(episode_index: int) -> bool:
    if args.force_threshold_schedule == "persistent":
        return True
    if args.force_threshold_schedule == "iid":
        return False
    return (episode_index + 1) % args.force_threshold_block_episodes != 0


def is_terminal(flags: dict[str, torch.Tensor], physics_step: int) -> bool:
    terminal = (
        flags["success"]
        | flags["over_force"]
        | flags["over_torqued"]
        | flags["under_force"]
        | flags["alignment_failure"]
        | flags["slipped"]
    )
    return bool(terminal[0].item()) or physics_step >= args.max_episode_steps


def shaped_reward(
    flags: dict[str, torch.Tensor], *, terminal: bool, force_range: ForceRange
) -> float:
    torque = float(flags["seal_torque_nm"][0].item())
    bilateral = bool(flags["bilateral_contact"][0].item())
    reward = 0.002 * int(bilateral)
    reward += 0.004 * int(bool(flags["within_torque_window"][0].item()))
    reward -= 0.002 * abs(torque - 0.425) / 0.15
    if args.force_reward_reference == "environment":
        reward_low_n = float(flags["minimum_force_threshold_n"][0].item())
        reward_high_n = float(flags["maximum_force_threshold_n"][0].item())
    else:
        reward_low_n = force_range.minimum_n
        reward_high_n = force_range.maximum_n
    reward += float(
        force_shaping_term(
            left_force_n=flags["left_finger_force_n"],
            right_force_n=flags["right_finger_force_n"],
            force_low_n=reward_low_n,
            force_high_n=reward_high_n,
            bilateral_contact=flags["bilateral_contact"],
        )[0].item()
    )
    if terminal:
        reward += {
            "success": 10.0,
            "under_force": -3.0,
            "over_force": -10.0,
            "alignment_failure": -6.0,
        }[classify(flags)]
    return reward


def actor_observation(
    normalized_flow_observation: torch.Tensor,
    force_range: ForceRange,
    left_force_n: torch.Tensor,
    right_force_n: torch.Tensor,
) -> torch.Tensor:
    return force_conditioned_observation(
        normalized_flow_observation,
        force_low_n=force_range.minimum_n,
        force_high_n=force_range.maximum_n,
        left_force_n=left_force_n,
        right_force_n=right_force_n,
    )


def capture_advisor_visual(
    path: Path,
    *,
    trace: list[dict[str, float | int | bool]],
    outcome: str,
    attempted_range: ForceRange,
    attempted_torque_range: TorqueRange | None = None,
) -> None:
    """Render only observed trajectory values into a two-panel visual card."""

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1000, 660
    image = Image.new("RGB", (width, height), (250, 251, 253))
    draw = ImageDraw.Draw(image)
    draw.text((32, 18), f"SealTwist episode: {outcome}", fill=(20, 25, 35))
    if attempted_torque_range is None:
        subtitle = (
            f"applied force target: {attempted_range.minimum_n:.2f}-"
            f"{attempted_range.maximum_n:.2f} N"
        )
    else:
        subtitle = (
            f"applied torque target: {attempted_torque_range.minimum_nm:.2f}-"
            f"{attempted_torque_range.maximum_nm:.2f} N m"
        )
    draw.text((32, 42), subtitle, fill=(28, 115, 70))
    force_box = (78, 90, 950, 350)
    torque_box = (78, 400, 950, 610)
    draw.rectangle(force_box, outline=(85, 95, 110), width=2)
    draw.rectangle(torque_box, outline=(85, 95, 110), width=2)
    steps = [int(item["physics_step"]) for item in trace]
    left_force = [float(item["left_force_n"]) for item in trace]
    right_force = [float(item["right_force_n"]) for item in trace]
    torque = [float(item["seal_torque_nm"]) for item in trace]
    x_max = max(steps, default=1)
    force_max = max(12.0, max(left_force, default=0.0), max(right_force, default=0.0))
    torque_max = max(0.60, max(torque, default=0.0))

    def x_coord(step: int, box: tuple[int, int, int, int]) -> float:
        return box[0] + (box[2] - box[0]) * step / max(x_max, 1)

    def y_coord(value: float, maximum: float, box: tuple[int, int, int, int]) -> float:
        return box[3] - (box[3] - box[1]) * value / maximum

    force_band_top = y_coord(attempted_range.maximum_n, force_max, force_box)
    force_band_bottom = y_coord(attempted_range.minimum_n, force_max, force_box)
    draw.rectangle(
        (force_box[0], force_band_top, force_box[2], force_band_bottom),
        fill=(224, 244, 229),
    )
    torque_band_range = (
        (0.35, 0.50)
        if attempted_torque_range is None
        else (
            attempted_torque_range.minimum_nm,
            attempted_torque_range.maximum_nm,
        )
    )
    torque_band_top = y_coord(torque_band_range[1], torque_max, torque_box)
    torque_band_bottom = y_coord(torque_band_range[0], torque_max, torque_box)
    draw.rectangle(
        (torque_box[0], torque_band_top, torque_box[2], torque_band_bottom),
        fill=(224, 244, 229) if attempted_torque_range is not None else (253, 226, 226),
    )
    if len(steps) >= 2:
        draw.line(
            [(x_coord(step, force_box), y_coord(value, force_max, force_box)) for step, value in zip(steps, left_force)],
            fill=(37, 99, 190),
            width=3,
        )
        draw.line(
            [(x_coord(step, force_box), y_coord(value, force_max, force_box)) for step, value in zip(steps, right_force)],
            fill=(0, 145, 155),
            width=3,
        )
        draw.line(
            [(x_coord(step, torque_box), y_coord(value, torque_max, torque_box)) for step, value in zip(steps, torque)],
            fill=(224, 126, 28),
            width=3,
        )
    for force_tick in range(0, int(math.ceil(force_max)) + 1, 2):
        y = y_coord(float(force_tick), force_max, force_box)
        draw.line((force_box[0] - 5, y, force_box[0], y), fill=(80, 90, 105), width=1)
        draw.text((35, y - 7), f"{force_tick}N", fill=(60, 70, 85))
    torque_ticks = (
        (0.0, 0.15, 0.30, 0.45, 0.60)
        if attempted_torque_range is not None
        else (0.0, 0.2, 0.35, 0.50, 0.6)
    )
    for torque_tick in torque_ticks:
        y = y_coord(torque_tick, torque_max, torque_box)
        draw.line((torque_box[0] - 5, y, torque_box[0], y), fill=(80, 90, 105), width=1)
        draw.text((24, y - 7), f"{torque_tick:.2f}", fill=(60, 70, 85))
    draw.text((80, 360), "blue: left fingertip", fill=(37, 99, 190))
    draw.text((260, 360), "cyan: right fingertip", fill=(0, 125, 135))
    draw.text((485, 360), "green: applied target band", fill=(28, 115, 70))
    draw.text((80, 618), "orange: seal torque (N m)", fill=(190, 95, 10))
    torque_band_label = (
        "green band: current torque estimate"
        if attempted_torque_range is not None
        else "red band: success torque window"
    )
    draw.text(
        (335, 618),
        torque_band_label,
        fill=(28, 115, 70) if attempted_torque_range is not None else (170, 45, 45),
    )
    image.save(path)


def save_training_checkpoint(
    path: Path,
    *,
    agent: SACResidualAgent | AWRAgent | DiffusionPPOAgent,
    replay: ReplayBuffer,
    completed_episodes: int,
    layout: FlowNoiseDSRLLayout,
    adapter: OnlineForceRangeAdapter,
    torque_adapter: OnlineTorqueRangeAdapter | None,
    flow_hash: str,
    total_transitions: int,
    balance_prior_value: float,
    continuation: dict[str, Any] | None,
) -> None:
    payload = {
        **agent.checkpoint(),
        "contract": (
            "seal_twist_zero_latent_flow_dsrl_sac_with_online_codex_vlm_"
            f"{args.adaptation_target}_adaptation"
            if args.online_algorithm == "dsrl"
            else f"seal_twist_online_{args.online_algorithm}_semantic_residual"
        ),
        "contract_version": 5 if args.online_algorithm == "dsrl" else 1,
        "online_algorithm": args.online_algorithm,
        "flow_checkpoint": str(args.flow_checkpoint.resolve()),
        "flow_checkpoint_sha256": flow_hash,
        "flow_horizon": layout.horizon,
        "flow_action_dim": layout.action_dim,
        "learned_noise_steps": layout.learned_noise_steps,
        "noise_residual_scale": layout.residual_scale,
        "noise_action_indices": list(layout.active_action_indices),
        "rl_action_semantics": (
            "one bounded tool-Y residual shared by all online baselines; the frozen "
            "Flow decoder remains zero-latent"
        ),
        "semantic_force_balance_residual": {
            "enabled": args.force_balance_y_scale_m > 0.0,
            "actor_coordinate": args.force_balance_actor_coordinate,
            "tool_action_index": 1,
            "maximum_offset_m": args.force_balance_y_scale_m,
            "latent_route_for_actor_coordinate": "disabled",
            "prior_gain": (
                args.force_balance_prior_gain if args.online_algorithm == "dsrl" else 0.0
            ),
            "prior_weight": (
                args.force_balance_prior_weight if args.online_algorithm == "dsrl" else 0.0
            ),
            "prior_limit": args.force_balance_prior_limit,
            "prior_smoothing": args.force_balance_prior_smoothing,
            "current_prior_value": balance_prior_value,
            "prior_signal": "smoothed_terminal_left_minus_right_finger_force_div_10n",
        },
        "zero_actor_semantics": "exact frozen zero-latent Flow BC before tactile width projection",
        "dsrl_activation": {
            "minimum_physics_step": args.dsrl_start_step,
            "requires_bilateral_contact": True,
            "pre_activation_correction": "exact_zero",
            "replay_scope": "active_contact_twist_phase_only",
        },
        "actor_observation_dim": ACTOR_OBSERVATION_DIM,
        "actor_observation_components": [
            "normalized_flow_observation_28d",
            "codex_force_low_div_10n",
            "codex_force_high_div_10n",
            "codex_force_center_div_10n",
            "target_minus_left_finger_force_div_10n",
            "target_minus_right_finger_force_div_10n",
            "left_minus_right_finger_force_div_10n",
        ],
        "critic_action_context_dim": (
            CRITIC_FORCE_CONTEXT_DIM if args.online_algorithm == "dsrl" else 0
        ),
        "critic_action_context_components": (
            [
                "codex_force_low_div_10n",
                "codex_force_high_div_10n",
                "privileged_episode_minimum_force_threshold_div_10n",
                "privileged_episode_maximum_force_threshold_div_10n",
                "executed_width_correction_div_maximum_step",
            ]
            if args.online_algorithm == "dsrl"
            else []
        ),
        "force_threshold_process": {
            "schedule": args.force_threshold_schedule,
            "block_episodes": args.force_threshold_block_episodes,
            "sampling_source": "coupled_profiles" if force_threshold_profiles else "independent_ranges",
            "configured_profile_index": args.force_threshold_profile_index,
            "coupled_profiles_n": [list(profile) for profile in force_threshold_profiles],
            "independent_minimum_range_n": list(map(float, args.minimum_force_range)),
            "independent_maximum_range_n": list(map(float, args.maximum_force_range)),
        },
        "force_reward_reference": args.force_reward_reference,
        "flow_execution": {
            "base_initial_latent": "zeros",
            "integration_steps": args.integration_steps,
            "solver": "heun",
            "chunk_execute_steps": args.chunk_execute_steps,
            "action_repeat": args.action_repeat,
            "action_smoothing": args.action_smoothing,
        },
        "exploration_schedule": {
            "start": args.exploration_start,
            "end": args.exploration_end,
            "anneal_episodes": (
                args.train_episodes
                if args.exploration_anneal_episodes is None
                else args.exploration_anneal_episodes
            ),
            "resume_policy": "hold_at_exploration_end",
        },
        "codex_vlm": {
            "model": args.codex_model,
            "reasoning_effort": args.codex_reasoning_effort,
            "call_policy": args.vlm_call_policy,
            "input": "source-grounded force/torque diagnostic image plus structured trajectory summary",
            "output_schema": str(
                (
                    args.codex_torque_output_schema
                    if args.adaptation_target == "torque"
                    else args.codex_output_schema
                ).resolve()
            ),
        },
        "force_projection": {
            "action_index": 9,
            "gain_m_per_n": args.force_feedback_gain_m_per_n,
            "maximum_step_m": args.force_feedback_max_step_m,
            "start_step": args.dsrl_start_step,
            "width_limits_m": list(map(float, args.gripper_width_limits_m)),
        },
        "completed_episodes": completed_episodes,
        "total_transitions": total_transitions,
        "train_seed": args.train_seed,
        "adaptation_target": args.adaptation_target,
        "vlm_call_policy": args.vlm_call_policy,
        "current_force_range_n": adapter.current_range.as_list(),
        "force_range_decisions": adapter.decisions,
        "current_torque_range_nm": (
            None if torque_adapter is None else torque_adapter.current_range.as_list()
        ),
        "torque_range_decisions": (
            [] if torque_adapter is None else torque_adapter.decisions
        ),
        "replay_state": (
            replay.state_dict() if isinstance(agent, SACResidualAgent) else None
        ),
        "continuation": continuation,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    output_root = args.output_root.resolve()
    flow_hash = sha256(args.flow_checkpoint.resolve())
    resume_root = None if args.resume_run is None else args.resume_run.expanduser().resolve()
    resume_summary: dict[str, Any] | None = None
    resume_checkpoint: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    start_episode = 0
    if resume_root is not None:
        if resume_root == output_root:
            raise ValueError("resume-run and output-root must be different directories")
        summary_path = resume_root / "training_summary.json"
        episode_path = resume_root / "training_episodes.jsonl"
        checkpoint_path = resume_root / "checkpoints" / "final.pt"
        for required in (summary_path, episode_path, checkpoint_path):
            if not required.is_file():
                raise FileNotFoundError(f"resume artifact not found: {required}")
        resume_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records = load_jsonl(episode_path)
        records.sort(key=lambda row: int(row["episode_index"]))
        start_episode = len(records)
        if [int(row["episode_index"]) for row in records] != list(range(start_episode)):
            raise ValueError("resume episode log is not contiguous from zero")
        if int(resume_summary["completed_episodes"]) != start_episode:
            raise ValueError("resume summary and episode log disagree")
        if args.train_episodes <= start_episode:
            raise ValueError(
                f"train-episodes={args.train_episodes} must exceed resumed count {start_episode}"
            )
        if int(resume_summary["training_seed"]) != args.train_seed:
            raise ValueError("resume training seed does not match --train-seed")
        if bool(resume_summary.get("vlm_is_codex")) == bool(args.disable_vlm):
            raise ValueError("resume VLM enable/disable mode does not match the source run")
        if str(resume_summary["flow_checkpoint_sha256"]) != flow_hash:
            raise ValueError("resume run uses a different Flow checkpoint")
        resume_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(resume_checkpoint["completed_episodes"]) != start_episode:
            raise ValueError("resume checkpoint and episode log disagree")
        if str(resume_checkpoint["flow_checkpoint_sha256"]) != flow_hash:
            raise ValueError("resume checkpoint uses a different Flow checkpoint")
        process = dict(resume_checkpoint.get("force_threshold_process", {}))
        if process:
            if str(process["schedule"]) != args.force_threshold_schedule:
                raise ValueError("resume force-threshold schedule mismatch")
            stored_profiles = [list(map(float, pair)) for pair in process["coupled_profiles_n"]]
            requested_profiles = [list(map(float, pair)) for pair in force_threshold_profiles]
            if stored_profiles != requested_profiles:
                raise ValueError("resume force-threshold profiles mismatch")
            stored_index = int(process.get("configured_profile_index", -1))
            if stored_index != args.force_threshold_profile_index:
                raise ValueError("resume force-threshold profile index mismatch")
        if str(resume_checkpoint.get("force_reward_reference", "target")) != args.force_reward_reference:
            raise ValueError("resume force-reward reference mismatch")

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    frame_dir = output_root / "advisor_frames"
    trajectory_dir = output_root / "trajectories"
    checkpoint_dir = output_root / "checkpoints"
    for directory in (frame_dir, trajectory_dir, checkpoint_dir):
        directory.mkdir()
    training_log = output_root / "training_episodes.jsonl"
    decision_log = output_root / (
        "torque_range_decisions.jsonl"
        if args.adaptation_target == "torque"
        else "force_range_decisions.jsonl"
    )
    if resume_root is not None:
        shutil.copy2(resume_root / "training_episodes.jsonl", training_log)
        source_decisions = resume_root / "force_range_decisions.jsonl"
        if source_decisions.is_file():
            shutil.copy2(source_decisions, decision_log)
        source_audit = resume_root / "codex_vlm_calls.jsonl"
        if source_audit.is_file():
            shutil.copy2(source_audit, output_root / "codex_vlm_calls.jsonl")

    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)

    config = SealTwistEnvCfg()
    config.scene.num_envs = 1
    config.seed = args.train_seed
    config.enable_cameras = False
    config.enable_gelsight = False
    config.randomize_labware_position = True
    config.labware_pos_randomization_xy = tuple(map(float, args.position_range))
    config.labware_yaw_randomization = 0.0
    config.randomize_expert_targets = False
    config.randomize_force_thresholds = True
    config.force_threshold_schedule = args.force_threshold_schedule
    config.force_threshold_block_episodes = args.force_threshold_block_episodes
    config.force_threshold_profiles_n = force_threshold_profiles
    config.force_threshold_profile_index = args.force_threshold_profile_index
    config.minimum_force_randomization_range_n = tuple(map(float, args.minimum_force_range))
    config.maximum_force_randomization_range_n = tuple(map(float, args.maximum_force_range))
    config.alignment_tolerance_m = float(args.alignment_tolerance)
    config.success_hold_steps = 5
    config.force_violation_dwell_steps = 5
    config.alignment_failure_dwell_steps = 5
    config.episode_length_s = max(args.max_episode_steps * config.sim.dt, 1.0)
    config.sim.physx.gpu_max_rigid_contact_count = args.physx_contact_capacity
    config.sim.physx.gpu_max_rigid_patch_count = args.physx_contact_capacity
    if args.device:
        config.sim.device = args.device
    env = SealTwistEnv(config)
    environment_device = torch.device(config.sim.device)
    policy_device = torch.device(
        "cuda"
        if args.policy_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.policy_device == "auto"
        else args.policy_device
    )
    loaded = load_flow_checkpoint(args.flow_checkpoint.resolve(), device=policy_device)
    loaded.model.requires_grad_(False)
    loaded.model.eval()
    if args.chunk_execute_steps > loaded.model.config.horizon:
        raise ValueError("chunk-execute-steps exceeds the checkpoint horizon")
    layout = FlowNoiseDSRLLayout(
        horizon=loaded.model.config.horizon,
        action_dim=loaded.model.config.action_dim,
        learned_noise_steps=args.learned_noise_steps,
        residual_scale=args.noise_residual_scale,
        active_action_indices=tuple(args.noise_action_indices),
    )
    if not 0 <= args.force_balance_actor_coordinate < layout.correction_dim:
        raise ValueError("force-balance-actor-coordinate is outside the DSRL action")
    critic_context_dim = (
        CRITIC_FORCE_CONTEXT_DIM if args.online_algorithm == "dsrl" else 0
    )
    if args.online_algorithm in ("dsrl", "residual_rl"):
        agent: SACResidualAgent | AWRAgent | DiffusionPPOAgent = SACResidualAgent(
            observation_dim=ACTOR_OBSERVATION_DIM,
            action_dim=layout.correction_dim,
            device=policy_device,
            learning_rate=args.learning_rate,
            gamma=0.995,
            tau=0.01,
            residual_l2_weight=0.10,
            initial_log_std=args.initial_log_std,
            initial_alpha=0.03,
            critic_action_context_dim=critic_context_dim,
            critic_context_mode="concat",
        )
    elif args.online_algorithm == "awr":
        agent = AWRAgent(
            ACTOR_OBSERVATION_DIM,
            layout.correction_dim,
            policy_device,
        )
    else:
        agent = DiffusionPPOAgent(
            ACTOR_OBSERVATION_DIM,
            layout.correction_dim,
            policy_device,
        )
    replay = ReplayBuffer(
        args.replay_capacity,
        ACTOR_OBSERVATION_DIM,
        layout.correction_dim,
        policy_device,
        action_context_dim=critic_context_dim,
    )
    adapter = OnlineForceRangeAdapter(
        ForceRange(*map(float, args.initial_force_range)),
        ForceRangeAdapterConfig(
            physical_minimum_n=float(args.physical_force_bounds[0]),
            physical_maximum_n=float(args.physical_force_bounds[1]),
            minimum_width_n=float(args.force_range_minimum_width),
            smoothing=float(args.force_range_smoothing),
        ),
    )
    torque_adapter = (
        OnlineTorqueRangeAdapter(
            TorqueRange(*map(float, args.initial_torque_range_nm)),
            TorqueRangeAdapterConfig(
                physical_minimum_nm=float(args.physical_torque_bounds_nm[0]),
                physical_maximum_nm=float(args.physical_torque_bounds_nm[1]),
                minimum_width_nm=float(args.torque_range_minimum_width_nm),
                smoothing=float(args.torque_range_smoothing),
            ),
        )
        if args.adaptation_target == "torque"
        else None
    )
    optimizer_restored = False
    replay_restored = False
    if resume_checkpoint is not None:
        assert isinstance(agent, SACResidualAgent)
        agent.load_checkpoint(resume_checkpoint)
        optimizer_restored = all(
            name in resume_checkpoint
            for name in ("actor_optimizer", "q_optimizer", "alpha_optimizer")
        )
        if "replay_state" in resume_checkpoint:
            replay.load_state_dict(resume_checkpoint["replay_state"])
            replay_restored = True
        adapter.current_range = ForceRange(
            *map(float, resume_checkpoint["current_force_range_n"])
        )
        adapter.decisions = list(resume_checkpoint["force_range_decisions"])
        if len(adapter.decisions) != start_episode:
            raise ValueError("resume force-range decision count mismatch")
    advisor = None if args.disable_vlm else CodexVisualForceAdvisor(output_root)
    if advisor is not None and resume_summary is not None:
        advisor.calls = int(resume_summary["vlm_calls"])
        advisor.successful_calls = int(resume_summary["vlm_successful_calls"])
    recent_history: deque[dict[str, Any]] = deque(
        (recent_history_entry(record) for record in records[-10:]), maxlen=10
    )
    total_transitions = (
        0
        if resume_checkpoint is None
        else int(resume_checkpoint.get("total_transitions", 0))
    )
    last_update: dict[str, float] = {}
    balance_prior_value = (
        0.0
        if resume_checkpoint is None
        else float(
            resume_checkpoint.get("semantic_force_balance_residual", {}).get(
                "current_prior_value", 0.0
            )
        )
    )
    inherited_learning_starts = (
        args.learning_starts
        if resume_checkpoint is None
        else int(
            (resume_checkpoint.get("continuation") or {}).get(
                "effective_learning_starts", args.learning_starts
            )
        )
    )
    effective_learning_starts = (
        inherited_learning_starts
        if resume_checkpoint is None or replay_restored
        else args.resume_learning_starts
    )
    upstream_legacy_note = (
        None
        if resume_checkpoint is None
        else (resume_checkpoint.get("continuation") or {}).get("legacy_resume_note")
    )
    continuation = (
        None
        if resume_root is None
        else {
            "source_run": str(resume_root),
            "source_completed_episodes": start_episode,
            "optimizer_state_restored": optimizer_restored,
            "replay_state_restored": replay_restored,
            "effective_learning_starts": effective_learning_starts,
            "exploration_policy": "hold_at_exploration_end",
            "exploration_scale": args.exploration_end,
            "legacy_resume_note": (
                upstream_legacy_note
                or (
                    None
                    if optimizer_restored and replay_restored
                    else "legacy checkpoint lacked optimizer/replay state; networks, Q targets, "
                    "alpha, force adapter and counters were restored, followed by replay warm-up"
                )
            ),
        }
    )
    started = time.time()

    try:
        for episode_index in range(start_episode, args.train_episodes):
            episode_seed = args.train_seed + episode_index
            attempted_torque_range = (
                None if torque_adapter is None else torque_adapter.current_range
            )
            if attempted_torque_range is not None:
                env.cfg.policy_max_turn_angle_rad = target_angle_for_torque(
                    env.cfg, attempted_torque_range.center_nm
                )
            env.reset(seed=episode_seed)
            attempted_range = adapter.current_range
            previous_action: torch.Tensor | None = None
            physics_step = 0
            episode_return = 0.0
            bilateral_steps = 0
            force_sum = 0.0
            force_squared_error_sum = 0.0
            in_band_steps = 0
            force_projection_decisions = 0
            correction_abs_sum = torch.zeros(layout.correction_dim, device=policy_device)
            semantic_y_abs_sum = 0.0
            correction_decisions = 0
            active_decisions = 0
            zero_correction_decisions = 0
            first_dsrl_active_step: int | None = None
            trace: list[dict[str, float | int | bool]] = []
            online_observations: list[torch.Tensor] = []
            online_actions: list[torch.Tensor] = []
            online_rewards: list[float] = []
            online_dones: list[float] = []
            online_old_log_probs: list[torch.Tensor] = []
            online_diffusion_traces: list[torch.Tensor] = []
            final_flags: dict[str, torch.Tensor] | None = None
            outcome = "under_force"
            trajectory_path = trajectory_dir / f"episode_{episode_index:03d}.jsonl.gz"

            with gzip.open(trajectory_path, "xt", encoding="utf-8") as trajectory_stream:
                while physics_step < args.max_episode_steps and simulation_app.is_running():
                    flags_before = env.get_task_flags()
                    left_before = flags_before["left_finger_force_n"]
                    right_before = flags_before["right_finger_force_n"]
                    mean_before = 0.5 * (left_before + right_before)
                    raw_observation = environment_flow_observation(
                        env, max_episode_steps=loaded.max_episode_steps
                    ).to(policy_device)
                    normalized = loaded.normalize_observation(raw_observation)
                    if args.normalized_observation_clip > 0.0:
                        normalized = normalized.clamp(
                            -args.normalized_observation_clip,
                            args.normalized_observation_clip,
                        )
                    policy_observation = actor_observation(
                        normalized,
                        attempted_range,
                        left_before.to(policy_device),
                        right_before.to(policy_device),
                    )
                    exploration = exploration_scale_for_episode(
                        episode_index=episode_index,
                        total_episodes=args.train_episodes,
                        exploration_start=args.exploration_start,
                        exploration_end=args.exploration_end,
                        resumed=resume_checkpoint is not None,
                        anneal_episodes=args.exploration_anneal_episodes,
                    )
                    dsrl_active = physics_step >= args.dsrl_start_step and bool(
                        flags_before["bilateral_contact"][0].item()
                    )
                    with torch.inference_mode():
                        old_log_prob: torch.Tensor | None = None
                        diffusion_trace: torch.Tensor | None = None
                        if dsrl_active:
                            if isinstance(agent, SACResidualAgent):
                                correction = agent.act(
                                    policy_observation,
                                    deterministic=False,
                                    exploration_scale=exploration,
                                )
                            elif isinstance(agent, AWRAgent):
                                correction = agent.act(
                                    policy_observation,
                                    deterministic=False,
                                )
                            else:
                                correction, old_log_prob, diffusion_trace = agent.act(
                                    policy_observation,
                                    deterministic=False,
                                )
                        else:
                            correction = torch.zeros(
                                (1, layout.correction_dim), device=policy_device
                            )
                        latent_correction = correction.clone()
                        if args.force_balance_y_scale_m > 0.0:
                            latent_correction[:, args.force_balance_actor_coordinate] = 0.0
                        decoder_noise = layout.compose_zero_anchored_noise(latent_correction)
                        normalized_chunk = loaded.model.decode(
                            normalized,
                            decoder_noise,
                            integration_steps=args.integration_steps,
                            solver="heun",
                        )
                        action_chunk = loaded.denormalize_action(normalized_chunk)
                        action_chunk[..., :3] += raw_observation[:, None, :3]
                        semantic_y_correction = (
                            correction[:, args.force_balance_actor_coordinate]
                            * args.force_balance_y_scale_m
                        )
                        action_chunk[..., 1] += semantic_y_correction[:, None]
                        action_chunk[..., 9] += raw_observation[
                            :, None, PREVIOUS_GRIPPER_WIDTH_OBSERVATION_INDEX
                        ]
                        contact = torch.maximum(left_before, right_before) >= 0.05
                        feedback_active = (
                            physics_step >= args.dsrl_start_step
                        ) & contact
                        width_correction = (
                            args.force_feedback_gain_m_per_n
                            * (mean_before.to(policy_device) - attempted_range.center_n)
                        ).clamp(
                            -args.force_feedback_max_step_m,
                            args.force_feedback_max_step_m,
                        )
                        width_correction = torch.where(
                            feedback_active.to(policy_device),
                            width_correction,
                            torch.zeros_like(width_correction),
                        )
                        action_chunk[..., 9] += width_correction[:, None]
                        action_chunk[..., 9].clamp_(
                            args.gripper_width_limits_m[0],
                            args.gripper_width_limits_m[1],
                        )
                        action_context = critic_force_context(
                            reference=policy_observation,
                            force_low_n=attempted_range.minimum_n,
                            force_high_n=attempted_range.maximum_n,
                            hidden_minimum_n=env.minimum_force_threshold_n.to(policy_device),
                            hidden_maximum_n=env.maximum_force_threshold_n.to(policy_device),
                            width_correction_m=width_correction,
                            maximum_width_correction_m=args.force_feedback_max_step_m,
                        )
                        action_chunk = action_chunk.to(environment_device)
                    force_projection_decisions += int(bool(feedback_active[0].item()))
                    if dsrl_active:
                        correction_abs_sum += correction[0].abs()
                        semantic_y_abs_sum += abs(float(semantic_y_correction[0].item()))
                        correction_decisions += 1
                        active_decisions += 1
                        if first_dsrl_active_step is None:
                            first_dsrl_active_step = physics_step
                    else:
                        zero_correction_decisions += 1

                    transition_reward = 0.0
                    terminal = False
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
                            physics_step += 1
                            flags = env.get_task_flags()
                            terminal = is_terminal(flags, physics_step)
                            reward = shaped_reward(
                                flags,
                                terminal=terminal,
                                force_range=attempted_range,
                            )
                            transition_reward += reward
                            episode_return += reward
                            left = float(flags["left_finger_force_n"][0].item())
                            right = float(flags["right_finger_force_n"][0].item())
                            mean_force = 0.5 * (left + right)
                            bilateral = bool(flags["bilateral_contact"][0].item())
                            bilateral_steps += int(bilateral)
                            if bilateral:
                                force_sum += mean_force
                                force_squared_error_sum += (
                                    mean_force - attempted_range.center_n
                                ) ** 2
                                in_band_steps += int(
                                    attempted_range.minimum_n <= left <= attempted_range.maximum_n
                                    and attempted_range.minimum_n
                                    <= right
                                    <= attempted_range.maximum_n
                                )
                            if physics_step == 1 or physics_step % 4 == 0 or terminal:
                                trace_row = {
                                    "physics_step": physics_step,
                                    "left_force_n": left,
                                    "right_force_n": right,
                                    "mean_finger_force_n": mean_force,
                                    "seal_torque_nm": float(flags["seal_torque_nm"][0].item()),
                                    "turn_angle_rad": float(flags["turn_angle_rad"][0].item()),
                                    "alignment_error_m": float(flags["alignment_error_m"][0].item()),
                                    "bilateral_contact": bilateral,
                                    "dsrl_active": dsrl_active,
                                    "noise_correction": correction[0].detach().cpu().tolist(),
                                    "policy_width_m": float(action[0, 9].item()),
                                }
                                trace.append(trace_row)
                                trajectory_stream.write(
                                    json.dumps(trace_row, allow_nan=False) + "\n"
                                )
                            final_flags = flags
                            if terminal:
                                break
                        if terminal:
                            break

                    assert final_flags is not None
                    next_raw = environment_flow_observation(
                        env, max_episode_steps=loaded.max_episode_steps
                    ).to(policy_device)
                    next_normalized = loaded.normalize_observation(next_raw)
                    if args.normalized_observation_clip > 0.0:
                        next_normalized = next_normalized.clamp(
                            -args.normalized_observation_clip,
                            args.normalized_observation_clip,
                        )
                    next_left = final_flags["left_finger_force_n"]
                    next_right = final_flags["right_finger_force_n"]
                    next_mean = 0.5 * (next_left + next_right)
                    next_policy_observation = actor_observation(
                        next_normalized,
                        attempted_range,
                        next_left.to(policy_device),
                        next_right.to(policy_device),
                    )
                    next_dsrl_active = physics_step >= args.dsrl_start_step and bool(
                        final_flags["bilateral_contact"][0].item()
                    )
                    next_contact = torch.maximum(next_left, next_right) >= 0.05
                    next_feedback_active = (
                        physics_step >= args.dsrl_start_step
                    ) & next_contact
                    next_width_correction = (
                        args.force_feedback_gain_m_per_n
                        * (next_mean.to(policy_device) - attempted_range.center_n)
                    ).clamp(
                        -args.force_feedback_max_step_m,
                        args.force_feedback_max_step_m,
                    )
                    next_width_correction = torch.where(
                        next_feedback_active.to(policy_device),
                        next_width_correction,
                        torch.zeros_like(next_width_correction),
                    )
                    next_action_context = critic_force_context(
                        reference=next_policy_observation,
                        force_low_n=attempted_range.minimum_n,
                        force_high_n=attempted_range.maximum_n,
                        hidden_minimum_n=env.minimum_force_threshold_n.to(policy_device),
                        hidden_maximum_n=env.maximum_force_threshold_n.to(policy_device),
                        width_correction_m=next_width_correction,
                        maximum_width_correction_m=args.force_feedback_max_step_m,
                    )
                    if dsrl_active:
                        total_transitions += 1
                        transition_done = float(terminal or not next_dsrl_active)
                        if isinstance(agent, SACResidualAgent):
                            use_dsrl_context = args.online_algorithm == "dsrl"
                            replay.add(
                                SACBatch(
                                    observation=policy_observation.detach().cpu(),
                                    action=correction.detach().cpu(),
                                    reward=torch.tensor(
                                        [transition_reward / 10.0], dtype=torch.float32
                                    ),
                                    next_observation=next_policy_observation.detach().cpu(),
                                    done=torch.tensor(
                                        [transition_done], dtype=torch.float32
                                    ),
                                    action_context=(
                                        action_context.detach().cpu()
                                        if use_dsrl_context
                                        else None
                                    ),
                                    next_action_context=(
                                        next_action_context.detach().cpu()
                                        if use_dsrl_context
                                        else None
                                    ),
                                )
                            )
                            if replay.size >= effective_learning_starts:
                                for _ in range(args.updates_per_decision):
                                    sampled_batch = replay.sample(args.batch_size)
                                    if args.online_algorithm == "dsrl":
                                        action_prior, action_prior_mask = (
                                            force_balance_action_prior(
                                                sampled_batch.action,
                                                action_dim=layout.correction_dim,
                                                actor_coordinate=(
                                                    args.force_balance_actor_coordinate
                                                ),
                                                target_value=balance_prior_value,
                                                limit=args.force_balance_prior_limit,
                                            )
                                        )
                                        last_update = agent.update(
                                            sampled_batch,
                                            action_prior=action_prior,
                                            action_prior_mask=action_prior_mask,
                                            action_prior_weight=(
                                                args.force_balance_prior_weight
                                            ),
                                        )
                                    else:
                                        last_update = agent.update(sampled_batch)
                        else:
                            online_observations.append(
                                policy_observation.detach().cpu()
                            )
                            online_actions.append(correction.detach().cpu())
                            online_rewards.append(transition_reward / 10.0)
                            online_dones.append(transition_done)
                            if isinstance(agent, DiffusionPPOAgent):
                                assert old_log_prob is not None
                                assert diffusion_trace is not None
                                online_old_log_probs.append(
                                    old_log_prob.detach().cpu()
                                )
                                online_diffusion_traces.append(
                                    diffusion_trace.detach().cpu()
                                )
                    if terminal:
                        outcome = classify(final_flags)
                        break

            if final_flags is None:
                raise RuntimeError("episode produced no task flags")
            if online_observations:
                observations = torch.cat(online_observations, dim=0)
                actions = torch.cat(online_actions, dim=0)
                rewards = torch.tensor(online_rewards, dtype=torch.float32).unsqueeze(1)
                dones = torch.tensor(online_dones, dtype=torch.float32).unsqueeze(1)
                returns = discounted_returns(
                    rewards,
                    dones,
                    torch.ones_like(dones, dtype=torch.bool),
                    gamma=0.995,
                )[:, 0]
                if isinstance(agent, AWRAgent):
                    last_update = agent.update(
                        observations,
                        actions,
                        returns,
                        epochs=6,
                        batch_size=args.batch_size,
                    )
                elif isinstance(agent, DiffusionPPOAgent):
                    diffusion_traces = torch.cat(online_diffusion_traces, dim=0)
                    old_log_probs = torch.cat(online_old_log_probs, dim=0)
                    with torch.no_grad():
                        values = agent.value(observations.to(policy_device)).cpu()
                    last_update = agent.update(
                        observations,
                        diffusion_traces,
                        old_log_probs,
                        returns,
                        returns - values,
                        epochs=4,
                        batch_size=args.batch_size,
                    )
            contact_count = max(bilateral_steps, 1)
            mean_force = force_sum / contact_count
            force_rmse = math.sqrt(force_squared_error_sum / contact_count)
            terminal_left_force = float(final_flags["left_finger_force_n"][0].item())
            terminal_right_force = float(final_flags["right_finger_force_n"][0].item())
            terminal_constraint_violation = force_constraint_violation(final_flags)
            terminal_torque_constraint_violation = torque_constraint_violation(final_flags)
            threshold_stays = threshold_persists_next_episode(episode_index)
            terminal_force_evidence = force_evidence(
                left_force_n=terminal_left_force,
                right_force_n=terminal_right_force,
                force_low_n=attempted_range.minimum_n,
                force_high_n=attempted_range.maximum_n,
            )
            balance_prior_used = balance_prior_value
            if args.online_algorithm == "dsrl":
                terminal_normalized_imbalance = (
                    terminal_left_force - terminal_right_force
                ) / 10.0
                balance_prior_target = max(
                    -args.force_balance_prior_limit,
                    min(
                        args.force_balance_prior_limit,
                        -args.force_balance_prior_gain * terminal_normalized_imbalance,
                    ),
                )
                balance_prior_value = (
                    (1.0 - args.force_balance_prior_smoothing)
                    * balance_prior_value
                    + args.force_balance_prior_smoothing * balance_prior_target
                )
            else:
                balance_prior_used = 0.0
                balance_prior_target = 0.0
                balance_prior_value = 0.0
            frame_path = frame_dir / f"episode_{episode_index:03d}_{outcome}.png"
            capture_advisor_visual(
                frame_path,
                trace=trace,
                outcome=outcome,
                attempted_range=attempted_range,
                attempted_torque_range=attempted_torque_range,
            )
            episode_record: dict[str, Any] = {
                "online_algorithm": args.online_algorithm,
                "episode_index": episode_index,
                "seed": episode_seed,
                "outcome": outcome,
                "success": outcome == "success",
                "steps": physics_step,
                "return": episode_return,
                "attempted_force_range_n": attempted_range.as_list(),
                "target_force_n": attempted_range.center_n,
                "attempted_torque_range_nm": (
                    None
                    if attempted_torque_range is None
                    else attempted_torque_range.as_list()
                ),
                "target_torque_nm": (
                    None
                    if attempted_torque_range is None
                    else attempted_torque_range.center_nm
                ),
                "policy_max_turn_angle_rad": float(env.cfg.policy_max_turn_angle_rad),
                "mean_bilateral_finger_force_n": mean_force,
                "force_rmse_n": force_rmse,
                "peak_left_force_n": float(env.peak_left_force_n[0].item()),
                "peak_right_force_n": float(env.peak_right_force_n[0].item()),
                "peak_torque_nm": float(env.peak_torque_nm[0].item()),
                "final_torque_nm": float(final_flags["seal_torque_nm"][0].item()),
                "final_left_finger_force_n": terminal_left_force,
                "final_right_finger_force_n": terminal_right_force,
                "final_minimum_finger_force_n": min(
                    terminal_left_force, terminal_right_force
                ),
                "final_maximum_finger_force_n": max(
                    terminal_left_force, terminal_right_force
                ),
                "terminal_force_evidence": terminal_force_evidence,
                "force_constraint_violation": terminal_constraint_violation,
                "torque_constraint_violation": terminal_torque_constraint_violation,
                "force_threshold_schedule": args.force_threshold_schedule,
                "force_threshold_profile_index": int(
                    env.force_threshold_profile_index[0].item()
                ),
                "force_threshold_persists_next_episode": threshold_stays,
                "force_balance_prior_used": balance_prior_used,
                "force_balance_prior_target": balance_prior_target,
                "next_force_balance_prior": balance_prior_value,
                "final_alignment_error_m": float(final_flags["alignment_error_m"][0].item()),
                "minimum_force_threshold_n": float(env.minimum_force_threshold_n[0].item()),
                "maximum_force_threshold_n": float(env.maximum_force_threshold_n[0].item()),
                "bilateral_contact_fraction": bilateral_steps / max(physics_step, 1),
                "bilateral_contact_steps": bilateral_steps,
                "in_force_band_fraction_given_bilateral": in_band_steps / contact_count,
                "force_projection_decisions": force_projection_decisions,
                "mean_abs_noise_correction": (
                    correction_abs_sum / max(correction_decisions, 1)
                ).detach().cpu().tolist(),
                "mean_abs_semantic_y_correction_m": (
                    semantic_y_abs_sum / max(correction_decisions, 1)
                ),
                "noise_correction_decisions": correction_decisions,
                "dsrl_active_decisions": active_decisions,
                "pre_activation_zero_correction_decisions": zero_correction_decisions,
                "first_dsrl_active_step": first_dsrl_active_step,
                "noise_action_indices": list(layout.active_action_indices),
                "sampled_cap_position_offset_m": env.sampled_cap_position_offset_m[0]
                .detach()
                .cpu()
                .tolist(),
                "base_flow_initial_latent": "zeros",
                "trajectory": str(trajectory_path.resolve()),
                "advisor_frame": str(frame_path.resolve()),
                "replay_size": replay.size,
                "optimizer_updates": agent.updates,
                "sac_updates": (
                    agent.updates if isinstance(agent, SACResidualAgent) else None
                ),
                "last_update": last_update,
            }
            sampled_trace = trace[:: max(1, len(trace) // 40)][:40]
            if args.adaptation_target == "torque":
                assert attempted_torque_range is not None
                request = {
                    "task": "SealTwist bottle-cap tightening",
                    "visual_type": "source-grounded force/torque diagnostic",
                    "physical_torque_bounds_nm": list(
                        map(float, args.physical_torque_bounds_nm)
                    ),
                    "current_range_nm": attempted_torque_range.as_list(),
                    "episode": {
                        "episode_index": episode_index,
                        "outcome": outcome,
                        "success": outcome == "success",
                        "attempted_torque_range_nm": attempted_torque_range.as_list(),
                        "peak_torque_nm": episode_record["peak_torque_nm"],
                        "final_torque_nm": episode_record["final_torque_nm"],
                        "torque_constraint_violation": (
                            terminal_torque_constraint_violation
                        ),
                        "force_constraint_violation": terminal_constraint_violation,
                        "mean_bilateral_finger_force_n": mean_force,
                        "peak_left_force_n": episode_record["peak_left_force_n"],
                        "peak_right_force_n": episode_record["peak_right_force_n"],
                        "final_left_finger_force_n": terminal_left_force,
                        "final_right_finger_force_n": terminal_right_force,
                        "terminal_force_evidence": terminal_force_evidence,
                        "final_alignment_error_m": episode_record[
                            "final_alignment_error_m"
                        ],
                        "bilateral_contact_fraction": episode_record[
                            "bilateral_contact_fraction"
                        ],
                        "force_trace_sample": sampled_trace,
                    },
                    "recent_history": list(recent_history),
                }
            else:
                request = {
                    "task": "SealTwist bottle-cap tightening",
                    "visual_type": "source-grounded force/torque diagnostic",
                    "physical_force_bounds_n": list(
                        map(float, args.physical_force_bounds)
                    ),
                    "force_threshold_schedule": {
                        "mode": args.force_threshold_schedule,
                        "block_episodes": args.force_threshold_block_episodes,
                        "persists_next_episode": threshold_stays,
                    },
                    "candidate_hidden_force_profiles_n": [
                        list(profile) for profile in force_threshold_profiles
                    ],
                    "independent_minimum_force_population_range_n": (
                        None
                        if force_threshold_profiles
                        else list(map(float, args.minimum_force_range))
                    ),
                    "independent_maximum_force_population_range_n": (
                        None
                        if force_threshold_profiles
                        else list(map(float, args.maximum_force_range))
                    ),
                    "torque_window_nm": [0.35, 0.50],
                    "current_range_n": attempted_range.as_list(),
                    "episode": {
                        "episode_index": episode_index,
                        "outcome": outcome,
                        "success": outcome == "success",
                        "attempted_force_range_n": attempted_range.as_list(),
                        "mean_bilateral_finger_force_n": mean_force,
                        "force_rmse_n": force_rmse,
                        "peak_left_force_n": episode_record["peak_left_force_n"],
                        "peak_right_force_n": episode_record["peak_right_force_n"],
                        "peak_torque_nm": episode_record["peak_torque_nm"],
                        "final_torque_nm": episode_record["final_torque_nm"],
                        "final_left_finger_force_n": terminal_left_force,
                        "final_right_finger_force_n": terminal_right_force,
                        "terminal_force_evidence": terminal_force_evidence,
                        "force_constraint_violation": terminal_constraint_violation,
                        "final_alignment_error_m": episode_record[
                            "final_alignment_error_m"
                        ],
                        "bilateral_contact_fraction": episode_record[
                            "bilateral_contact_fraction"
                        ],
                        "in_force_band_fraction_given_bilateral": episode_record[
                            "in_force_band_fraction_given_bilateral"
                        ],
                        "force_trace_sample": sampled_trace,
                    },
                    "recent_history": list(recent_history),
                }

            vlm_should_call = advisor is not None and (
                args.vlm_call_policy == "every_episode" or outcome != "success"
            )
            vlm_called = False
            vlm_ok = False
            if advisor is None:
                response = {
                    (
                        "torque_range_nm"
                        if args.adaptation_target == "torque"
                        else "force_range_n"
                    ): (
                        attempted_torque_range.as_list()
                        if attempted_torque_range is not None
                        else attempted_range.as_list()
                    ),
                    "failure_cause": "unknown",
                    (
                        "torque_assessment"
                        if args.adaptation_target == "torque"
                        else "force_assessment"
                    ): "unknown",
                    "rationale": "Codex VLM explicitly disabled",
                    "confidence": 0.0,
                }
                proposal = None
                adapter_error = "Codex VLM disabled"
            elif not vlm_should_call:
                response = {
                    (
                        "torque_range_nm"
                        if args.adaptation_target == "torque"
                        else "force_range_n"
                    ): (
                        attempted_torque_range.as_list()
                        if attempted_torque_range is not None
                        else attempted_range.as_list()
                    ),
                    "failure_cause": "unknown",
                    (
                        "torque_assessment"
                        if args.adaptation_target == "torque"
                        else "force_assessment"
                    ): "safe",
                    "rationale": "VLM call skipped after a successful episode",
                    "confidence": 1.0,
                }
                proposal = (
                    attempted_torque_range
                    if attempted_torque_range is not None
                    else attempted_range
                )
                adapter_error = None
            else:
                vlm_called = True
                response = advisor.recommend(
                    episode_index=episode_index,
                    image_path=frame_path,
                    request=request,
                )
                vlm_ok = True
                adapter_error = None
                if args.adaptation_target == "torque":
                    assert attempted_torque_range is not None
                    raw_proposal = TorqueRange(
                        *map(float, response["torque_range_nm"])
                    )
                    guarded = guard_vlm_torque_proposal(
                        current_range_nm=(
                            attempted_torque_range.minimum_nm,
                            attempted_torque_range.maximum_nm,
                        ),
                        proposed_range_nm=(
                            raw_proposal.minimum_nm,
                            raw_proposal.maximum_nm,
                        ),
                        physical_bounds_nm=tuple(
                            map(float, args.physical_torque_bounds_nm)
                        ),
                        torque_constraint_violation=(
                            terminal_torque_constraint_violation
                        ),
                        outcome=outcome,
                        maximum_center_step_nm=float(
                            args.vlm_max_torque_center_step_nm
                        ),
                    )
                    proposal = TorqueRange(*guarded.torque_range_nm)
                    response["proposal_guard"] = {
                        "raw_vlm_range_nm": raw_proposal.as_list(),
                        "guarded_range_nm": proposal.as_list(),
                        "torque_constraint_violation": (
                            terminal_torque_constraint_violation
                        ),
                        "decision": guarded.decision,
                    }
                else:
                    raw_proposal = ForceRange(
                        *map(float, response["force_range_n"])
                    )
                    guarded = guard_vlm_force_proposal(
                        current_range_n=(
                            attempted_range.minimum_n,
                            attempted_range.maximum_n,
                        ),
                        proposed_range_n=(
                            raw_proposal.minimum_n,
                            raw_proposal.maximum_n,
                        ),
                        physical_bounds_n=tuple(map(float, args.physical_force_bounds)),
                        terminal_left_force_n=terminal_left_force,
                        terminal_right_force_n=terminal_right_force,
                        outcome=outcome,
                        maximum_center_step_n=float(args.vlm_max_center_step_n),
                        threshold_schedule=args.force_threshold_schedule,
                        force_constraint_violation=terminal_constraint_violation,
                        threshold_persists_next_episode=threshold_stays,
                    )
                    proposal = ForceRange(*guarded.force_range_n)
                    response["proposal_guard"] = {
                        "raw_vlm_range_n": raw_proposal.as_list(),
                        "guarded_range_n": proposal.as_list(),
                        "terminal_force_evidence": guarded.evidence,
                        "decision": guarded.decision,
                    }

            if args.adaptation_target == "torque":
                assert torque_adapter is not None
                decision = torque_adapter.update_after_episode(
                    episode_index=episode_index,
                    proposed_range=proposal,
                    raw_response=json.dumps(response, ensure_ascii=False),
                    error=adapter_error,
                    vlm_called=vlm_called,
                )
                next_force_range = attempted_range.as_list()
                next_torque_range = decision["applied_range_nm"]
            else:
                decision = adapter.update_after_episode(
                    episode_index=episode_index,
                    proposed_range=proposal,
                    raw_response=json.dumps(response, ensure_ascii=False),
                    error=adapter_error,
                )
                decision["vlm_called"] = vlm_called
                next_force_range = decision["applied_range_n"]
                next_torque_range = None
            episode_record.update(
                {
                    "vlm_model": args.codex_model if vlm_ok else None,
                    "vlm_reasoning_effort": args.codex_reasoning_effort if vlm_ok else None,
                    "vlm_called": vlm_called,
                    "vlm_call_succeeded": vlm_ok if vlm_called else None,
                    "vlm_response": response,
                    "next_force_range_n": next_force_range,
                    "next_torque_range_nm": next_torque_range,
                }
            )
            records.append(episode_record)
            if args.adaptation_target == "torque":
                recent_history.append(
                    {
                        "episode_index": episode_index,
                        "outcome": outcome,
                        "attempted_range_nm": attempted_torque_range.as_list(),
                        "peak_torque_nm": episode_record["peak_torque_nm"],
                        "final_torque_nm": episode_record["final_torque_nm"],
                        "torque_constraint_violation": (
                            terminal_torque_constraint_violation
                        ),
                        "force_constraint_violation": terminal_constraint_violation,
                        "next_range_nm": next_torque_range,
                    }
                )
            else:
                recent_history.append({
                    "episode_index": episode_index,
                    "outcome": outcome,
                    "attempted_range_n": attempted_range.as_list(),
                    "mean_force_n": mean_force,
                    "peak_force_n": max(
                        episode_record["peak_left_force_n"],
                        episode_record["peak_right_force_n"],
                    ),
                    "peak_torque_nm": episode_record["peak_torque_nm"],
                    "final_left_force_n": terminal_left_force,
                    "final_right_force_n": terminal_right_force,
                    "force_evidence": terminal_force_evidence,
                    "force_constraint_violation": terminal_constraint_violation,
                    "next_range_n": next_force_range,
                })
            append_jsonl(training_log, episode_record)
            append_jsonl(decision_log, decision)
            if (episode_index + 1) % args.checkpoint_every == 0:
                save_training_checkpoint(
                    checkpoint_dir / f"episode_{episode_index + 1:03d}.pt",
                    agent=agent,
                    replay=replay,
                    completed_episodes=episode_index + 1,
                    layout=layout,
                    adapter=adapter,
                    torque_adapter=torque_adapter,
                    flow_hash=flow_hash,
                    total_transitions=total_transitions,
                    balance_prior_value=balance_prior_value,
                    continuation=continuation,
                )
            recent_successes = sum(int(item["success"]) for item in records[-10:])
            if args.adaptation_target == "torque":
                range_transition = (
                    f"{attempted_torque_range.as_list()}->{next_torque_range} Nm"
                )
            else:
                range_transition = f"{attempted_range.as_list()}->{next_force_range} N"
            print(
                f"[ONLINE] episode={episode_index + 1}/{args.train_episodes} "
                f"outcome={outcome} rolling10={recent_successes / min(10, len(records)):.1%} "
                f"force={mean_force:.3f}N torque={episode_record['peak_torque_nm']:.3f}Nm "
                f"range={range_transition} replay={replay.size} updates={agent.updates} "
                f"vlm_called={vlm_called} codex_ok={vlm_ok}",
                flush=True,
            )
    finally:
        env.close()

    final_checkpoint = checkpoint_dir / "final.pt"
    save_training_checkpoint(
        final_checkpoint,
        agent=agent,
        replay=replay,
        completed_episodes=len(records),
        layout=layout,
        adapter=adapter,
        torque_adapter=torque_adapter,
        flow_hash=flow_hash,
        total_transitions=total_transitions,
        balance_prior_value=balance_prior_value,
        continuation=continuation,
    )
    successes = sum(int(item["success"]) for item in records)
    failures = Counter(item["outcome"] for item in records if not item["success"])
    summary = {
        "experiment": "SealTwist online semantic residual learning",
        "online_algorithm": args.online_algorithm,
        "adaptation_target": args.adaptation_target,
        "training_seed": args.train_seed,
        "completed_episodes": len(records),
        "new_episodes_this_process": len(records) - start_episode,
        "continuation": continuation,
        "successes": successes,
        "online_training_success_rate": successes / max(len(records), 1),
        "failure_counts": dict(failures),
        "force_threshold_process": {
            "schedule": args.force_threshold_schedule,
            "block_episodes": args.force_threshold_block_episodes,
            "sampling_source": "coupled_profiles" if force_threshold_profiles else "independent_ranges",
            "configured_profile_index": args.force_threshold_profile_index,
            "coupled_profiles_n": [list(profile) for profile in force_threshold_profiles],
            "independent_minimum_range_n": list(map(float, args.minimum_force_range)),
            "independent_maximum_range_n": list(map(float, args.maximum_force_range)),
        },
        "sampled_force_threshold_profile_index": int(
            env.force_threshold_profile_index[0].item()
        ),
        "sampled_force_threshold_n": [
            float(env.minimum_force_threshold_n[0].item()),
            float(env.maximum_force_threshold_n[0].item()),
        ],
        "force_reward_reference": args.force_reward_reference,
        "exploration_schedule": {
            "start": args.exploration_start,
            "end": args.exploration_end,
            "anneal_episodes": (
                args.train_episodes
                if args.exploration_anneal_episodes is None
                else args.exploration_anneal_episodes
            ),
            "resume_policy": "hold_at_exploration_end",
        },
        "flow_checkpoint": str(args.flow_checkpoint.resolve()),
        "flow_checkpoint_sha256": flow_hash,
        "dsrl_action_space": (
            f"bounded {layout.correction_dim}-D semantic tool-Y correction on the "
            f"zero-latent [{layout.horizon},{layout.action_dim}] frozen Flow policy"
        ),
        "rl_action_space": (
            f"bounded {layout.correction_dim}-D semantic tool-Y correction on the "
            f"zero-latent [{layout.horizon},{layout.action_dim}] frozen Flow policy"
        ),
        "noise_action_indices": list(layout.active_action_indices),
        "noise_residual_scale": layout.residual_scale,
        "semantic_force_balance_residual": {
            "actor_coordinate": args.force_balance_actor_coordinate,
            "tool_action_index": 1,
            "maximum_offset_m": args.force_balance_y_scale_m,
            "prior_gain": (
                args.force_balance_prior_gain if args.online_algorithm == "dsrl" else 0.0
            ),
            "prior_weight": (
                args.force_balance_prior_weight if args.online_algorithm == "dsrl" else 0.0
            ),
            "prior_limit": args.force_balance_prior_limit,
            "prior_smoothing": args.force_balance_prior_smoothing,
            "final_prior_value": balance_prior_value,
        },
        "optimizer_updates": agent.updates,
        "sac_optimizer_updates": (
            agent.updates if isinstance(agent, SACResidualAgent) else None
        ),
        "replay_transitions": total_transitions,
        "replay_size": replay.size,
        "vlm_advisor": None if advisor is None else args.codex_model,
        "vlm_reasoning_effort": None if advisor is None else args.codex_reasoning_effort,
        "vlm_is_codex": advisor is not None,
        "vlm_calls": 0 if advisor is None else advisor.calls,
        "vlm_successful_calls": 0 if advisor is None else advisor.successful_calls,
        "vlm_call_policy": args.vlm_call_policy,
        "failure_episodes": sum(int(not item["success"]) for item in records),
        "episodes_with_vlm_call": sum(int(item["vlm_called"]) for item in records),
        "vlm_input": "source-grounded diagnostic image plus structured force/torque trace",
        "initial_force_range_n": list(map(float, args.initial_force_range)),
        "final_force_range_n": adapter.current_range.as_list(),
        "physical_force_bounds_n": list(map(float, args.physical_force_bounds)),
        "initial_torque_range_nm": (
            list(map(float, args.initial_torque_range_nm))
            if torque_adapter is not None
            else None
        ),
        "final_torque_range_nm": (
            torque_adapter.current_range.as_list()
            if torque_adapter is not None
            else None
        ),
        "physical_torque_bounds_nm": (
            list(map(float, args.physical_torque_bounds_nm))
            if torque_adapter is not None
            else None
        ),
        "flow_execution": {
            "integration_steps": args.integration_steps,
            "chunk_execute_steps": args.chunk_execute_steps,
            "action_repeat": args.action_repeat,
        },
        "final_checkpoint": str(final_checkpoint.resolve()),
        "elapsed_seconds": time.time() - started,
        "cumulative_elapsed_seconds": (
            time.time() - started
            + (0.0 if resume_summary is None else float(resume_summary["elapsed_seconds"]))
        ),
    }
    (output_root / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[SUMMARY] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
