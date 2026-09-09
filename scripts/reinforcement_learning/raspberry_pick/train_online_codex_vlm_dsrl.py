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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
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
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--train-episodes", type=int, default=100)
parser.add_argument("--train-seed", type=int, default=1_010_000)
parser.add_argument("--max-episode-steps", type=int, default=900)
parser.add_argument("--action-repeat", type=int, default=4)
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
parser.add_argument("--initial-force-range", type=float, nargs=2, default=(2.20, 2.80))
parser.add_argument("--physical-force-bounds", type=float, nargs=2, default=(1.20, 3.10))
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
    help="Margin below the true damage threshold used only by oracle_force_dsrl.",
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
parser.add_argument("--disable-vlm", action="store_true")
parser.add_argument("--physx-contact-capacity", type=int, default=2**20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.train_episodes, args.action_repeat, args.num_inference_steps) < 1:
    parser.error("episode, action-repeat and inference-step counts must be positive")
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
if not args.physical_force_bounds[0] < args.physical_force_bounds[1] < args.damage_threshold_n:
    parser.error("physical force bounds must be ordered and below the damage threshold")
if args.codex_model != "gpt-5.6-sol" or args.codex_reasoning_effort != "high":
    parser.error("the formal experiment contract requires gpt-5.6-sol at high reasoning")
if args.guard_confirmations < 1:
    parser.error("guard-confirmations must be positive")
if min(args.semantic_translation_scale_m) < 0.0 or args.semantic_gripper_scale_m < 0.0:
    parser.error("semantic residual scales must be non-negative")
if args.oracle_force_width_n <= 0.0 or args.oracle_damage_margin_n <= 0.0:
    parser.error("oracle force width and damage margin must be positive")
if args.method is None:
    args.method = "dsrl_only" if args.disable_vlm else "vlm_force_v2"
if args.disable_vlm and args.method != "dsrl_only":
    parser.error("--disable-vlm cannot be combined with a VLM method")
if args.evaluation_mode and args.agent_checkpoint is None:
    parser.error("--evaluation-mode requires --agent-checkpoint")
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
from online_rl_support import (
    RASPBERRY_PROFILES,
    GuardedForceProposal,
    SemanticResidualLayout,
    SemanticXYResidualLayout,
    environment_shaped_reward,
    guard_force_proposal,
    residual_is_active,
    theoretical_oracle_force_range,
    terminal_outcome_reward,
)
from tacex_tasks.raspberry_pick.raspberry_pick_env import RaspberryPickEnv
from tacex_tasks.raspberry_pick.raspberry_pick_env_cfg import RaspberryPickEnvCfg


METHOD_CONTRACTS = {
    "dsrl_only": "raspberry_dinov3_flow_bc_zero_latent_dsrl_sac_fixed_force_v2",
    "oracle_force_dsrl": "raspberry_dinov3_flow_bc_zero_latent_dsrl_sac_oracle_force_v1",
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


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
                    input=self._prompt(request),
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
                    "request": request,
                    "response": response,
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
) -> torch.Tensor:
    if visual_xy.ndim != 2 or visual_xy.shape[-1] != 2:
        raise ValueError("visual_xy must have shape [B, 2]")
    if normalized_state.ndim != 2 or normalized_state.shape[-1] != 12:
        raise ValueError("normalized_state must have shape [B, 12]")
    scale = float(args.damage_threshold_n)
    grip = metrics["grip_force_n"].reshape(-1, 1)
    scalars = torch.cat(
        (
            torch.full_like(grip, phase),
            grip / scale,
            metrics["left_normal_force_n"].reshape(-1, 1) / scale,
            metrics["right_normal_force_n"].reshape(-1, 1) / scale,
            flags["stem_tension_n"].reshape(-1, 1) / float(args.detach_threshold_n),
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
    if explicit_spatial_features:
        return torch.cat((embedding, visual_xy, normalized_state, scalars), dim=-1)
    return torch.cat((embedding, scalars), dim=-1)


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
    replay: ReplayBuffer,
    adapter: OnlineForceRangeAdapter,
    controller: GrapeForceController,
    base_hash: str,
    completed_episodes: int,
    replay_transitions: int,
) -> None:
    payload = {
        **agent.checkpoint(),
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "task": TASK,
        "method": args.method,
        "berry_profile": args.berry_profile,
        "base_checkpoint": str(args.checkpoint.resolve()),
        "base_checkpoint_sha256": base_hash,
        "base_checkpoint_weights": "raw_best_epoch",
        "residual_mode": (
            "semantic_xy"
            if args.method in SEMANTIC_XY_METHODS
            else "semantic"
            if semantic_layout is not None
            else "flow_latent"
        ),
        "dsrl_frozen": bool(args.freeze_dsrl),
        "flow_initial_latent": (
            "zero" if semantic_layout is not None else "zero_plus_bounded_dsrl_correction"
        ),
        "flow_horizon": layout.horizon,
        "flow_action_dim": layout.action_dim,
        "learned_noise_steps": layout.learned_noise_steps,
        "noise_action_indices": list(layout.active_action_indices),
        "noise_residual_scale": layout.residual_scale,
        "semantic_residual_layout": (
            None if semantic_layout is None else asdict(semantic_layout)
        ),
        "actor_observation_dim": agent.actor.network[0].in_features,
        "actor_observation_features": (
            "pooled_condition+visual_xy+normalized_state+physics"
            if args.method in PREPULL_SEMANTIC_METHODS
            else "pooled_condition+physics"
        ),
        "dsrl_start_step": args.dsrl_start_step,
        "residual_active_until_phase": (
            float(args.pull_z_start_phase)
            if args.method in PREPULL_SEMANTIC_METHODS
            else None
        ),
        "force_controller_config": asdict(controller.config),
        "current_force_range_n": adapter.current_range.as_list(),
        "force_range_decisions": adapter.decisions,
        "completed_episodes": completed_episodes,
        "replay_transitions": replay_transitions,
        "replay_state": replay.state_dict(),
        "training_seed": args.train_seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    frame_dir = output_root / "advisor_frames"
    trajectory_dir = output_root / "trajectories"
    checkpoint_dir = output_root / "checkpoints"
    for directory in (frame_dir, trajectory_dir, checkpoint_dir):
        directory.mkdir()

    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)

    profile = RASPBERRY_PROFILES[args.berry_profile]
    oracle_force = None
    if args.method == "oracle_force_dsrl":
        oracle_force = theoretical_oracle_force_range(
            profile,
            interval_width_n=args.oracle_force_width_n,
            damage_margin_n=args.oracle_damage_margin_n,
        )
        physical_low, physical_high = map(float, args.physical_force_bounds)
        oracle_low, oracle_high = oracle_force.as_tuple()
        if not physical_low <= oracle_low < oracle_high <= physical_high:
            raise ValueError("oracle force interval violates the physical force bounds")
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
    config.episode_length_s = max(args.max_episode_steps * config.sim.dt, 1.0)
    if args.device:
        config.sim.device = args.device
    env = RaspberryPickEnv(config, render_mode="rgb_array")
    device = torch.device(config.sim.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    model, normalizer, base_checkpoint = load_policy(checkpoint_path, device)
    if model.config.state_dim != 12 or model.config.num_cameras != 2:
        raise ValueError("RaspberryPick policy must use the 12-D tactile state and two cameras")

    layout = FlowNoiseDSRLLayout(
        horizon=model.config.chunk_size,
        action_dim=model.config.action_dim,
        learned_noise_steps=args.learned_noise_steps,
        residual_scale=args.noise_residual_scale,
        active_action_indices=tuple(args.noise_action_indices),
    )
    semantic_layout = None
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
        semantic_layout.correction_dim if semantic_layout is not None else layout.correction_dim
    )
    actor_observation_dim = model.config.cond_dim + 9
    if args.method in PREPULL_SEMANTIC_METHODS:
        actor_observation_dim += 2 + model.config.state_dim
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
    )
    replay = ReplayBuffer(
        args.replay_capacity, actor_observation_dim, correction_dim, device
    )
    loaded_agent_checkpoint: Path | None = None
    if args.agent_checkpoint is not None:
        loaded_agent_checkpoint = args.agent_checkpoint.expanduser().resolve()
        payload = torch.load(loaded_agent_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("contract") != CONTRACT or payload.get("method") != args.method:
            raise ValueError("residual checkpoint contract/method does not match this run")
        if payload.get("base_checkpoint_sha256") != sha256(checkpoint_path):
            raise ValueError("residual checkpoint was trained with a different base policy")
        agent.load_checkpoint(payload)
    starting_agent_updates = agent.updates
    effective_initial_force_range = (
        oracle_force.as_tuple()
        if oracle_force is not None
        else tuple(map(float, args.initial_force_range))
    )
    adapter = OnlineForceRangeAdapter(
        ForceRange(*effective_initial_force_range),
        ForceRangeAdapterConfig(
            physical_minimum_n=float(args.physical_force_bounds[0]),
            physical_maximum_n=float(args.physical_force_bounds[1]),
            minimum_width_n=float(args.force_range_minimum_width),
            smoothing=float(args.force_range_smoothing),
        ),
    )
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
    advisor = (
        None
        if args.method in NO_VLM_METHODS
        else CodexVisualForceAdvisor(output_root)
    )
    base_hash = sha256(checkpoint_path)
    position_offset = torch.tensor(
        args.position_command_offset_m, dtype=torch.float32, device=device
    )
    records: list[dict[str, Any]] = []
    recent_history: deque[dict[str, Any]] = deque(maxlen=10)
    recent_candidate_directions: deque[str] = deque(maxlen=max(args.guard_confirmations, 3))
    total_transitions = 0
    last_update: dict[str, float] = {}
    started = time.time()

    try:
        for episode_index in range(args.train_episodes):
            episode_seed = args.train_seed + episode_index
            env.reset(seed=episode_seed)
            controller.set_target_range(adapter.current_range)
            controller.reset()
            attempted_range = adapter.current_range
            previous_action: torch.Tensor | None = None
            physics_step = 0
            episode_return = 0.0
            dsrl_active_decisions = 0
            correction_abs_sum = torch.zeros(correction_dim, device=device)
            pending_terminal_credit_index: int | None = None
            controller_active_steps = 0
            safety_override_steps = 0
            bilateral_steps = 0
            force_sum = 0.0
            force_error_sum = 0.0
            left_force_sum = 0.0
            right_force_sum = 0.0
            damage_risk_steps = 0
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
                    phase = min(physics_step / float(args.max_episode_steps), 1.0)
                    metrics = env.get_contact_force_metrics()
                    flags_before = env.get_task_flags()
                    embedding = model.dsrl_observation(cond_tokens)
                    policy_observation = actor_observation(
                        embedding,
                        visual_xy=visual_xy,
                        normalized_state=normalized_state,
                        explicit_spatial_features=args.method in PREPULL_SEMANTIC_METHODS,
                        phase=phase,
                        metrics=metrics,
                        flags=flags_before,
                        force_range=attempted_range,
                    )
                    dsrl_active = residual_is_active(
                        physics_step=physics_step,
                        start_step=args.dsrl_start_step,
                        phase=phase,
                        pull_start_phase=args.pull_z_start_phase,
                        stem_attached=bool(flags_before["stem_attached"][0].item()),
                        semantic_residual=args.method in PREPULL_SEMANTIC_METHODS,
                    )
                    progress = episode_index / max(args.train_episodes - 1, 1)
                    exploration = args.exploration_start + progress * (
                        args.exploration_end - args.exploration_start
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
                        if semantic_layout is None:
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
                        target = chunk[:, 0]
                    if dsrl_active and not args.freeze_dsrl:
                        correction_abs_sum += correction[0].abs()
                        dsrl_active_decisions += 1

                    transition_reward = 0.0
                    terminal = False
                    final_flags = flags_before
                    for _ in range(args.action_repeat):
                        if previous_action is None:
                            policy_action = target
                        else:
                            alpha = float(args.action_smoothing)
                            policy_action = alpha * target + (1.0 - alpha) * previous_action
                        previous_action = policy_action
                        metrics = env.get_contact_force_metrics()
                        executed_action, diagnostics = controller.control(
                            policy_action,
                            contact_force_n=metrics["grip_force_n"],
                            bilateral_contact=metrics["bilateral_contact"],
                            dt_s=float(env.physics_dt),
                        )
                        if float((executed_action[:, :9] - policy_action[:, :9]).abs().max().item()) > 1e-8:
                            raise RuntimeError("force controller modified a non-gripper action")
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
                        controller_active_steps += int(diagnostics.active[0].item())
                        safety_override_steps += int(diagnostics.safety_override[0].item())
                        if physics_step == 1 or physics_step % 4 == 0 or terminal:
                            row = {
                                "physics_step": physics_step,
                                "grip_force_n": grip,
                                "left_force_n": float(final_flags["left_normal_force_n"][0].item()),
                                "right_force_n": float(final_flags["right_normal_force_n"][0].item()),
                                "pedicle_tension_n": float(final_flags["stem_tension_n"][0].item()),
                                "bilateral_contact": bilateral,
                                "controller_active": bool(diagnostics.active[0].item()),
                                "safety_override": bool(diagnostics.safety_override[0].item()),
                                "dsrl_active": dsrl_active,
                                "residual_mode": (
                                    "semantic_xy"
                                    if args.method in SEMANTIC_XY_METHODS
                                    else "semantic"
                                    if semantic_layout is not None
                                    else "flow_latent"
                                ),
                                "residual_correction": correction[0].detach().cpu().tolist(),
                                "noise_correction": correction[0].detach().cpu().tolist(),
                            }
                            force_trace.append(row)
                            trajectory_stream.write(json.dumps(row, allow_nan=False) + "\n")
                        if terminal:
                            break

                    if terminal:
                        next_policy_observation = policy_observation
                        next_dsrl_active = False
                    else:
                        env.sim.render()
                        state_history.append(policy_state(env))
                        image_history.append(
                            torch.stack(
                                (camera_rgb_224(env, "wrist"), camera_rgb_224(env, "third")),
                                dim=1,
                            )
                        )
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
                            explicit_spatial_features=args.method in PREPULL_SEMANTIC_METHODS,
                            phase=next_phase,
                            metrics=next_metrics,
                            flags=next_flags,
                            force_range=attempted_range,
                        )
                        next_dsrl_active = residual_is_active(
                            physics_step=physics_step,
                            start_step=args.dsrl_start_step,
                            phase=next_phase,
                            pull_start_phase=args.pull_z_start_phase,
                            stem_attached=bool(next_flags["stem_attached"][0].item()),
                            semantic_residual=args.method in PREPULL_SEMANTIC_METHODS,
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
                                last_update = agent.update(replay.sample(args.batch_size))
                    if terminal:
                        break
                    cond_tokens, visual_xy = next_cond_tokens, next_visual_xy
                    normalized_state = next_model_obs[common.STATE_KEY][:, -1]

            outcome = classify(final_flags)
            if (
                args.method in DELAYED_CREDIT_METHODS
                and pending_terminal_credit_index is not None
            ):
                replay.add_reward(
                    pending_terminal_credit_index,
                    terminal_outcome_reward(outcome) / 10.0,
                )
            contact_count = max(bilateral_steps, 1)
            mean_force = force_sum / contact_count
            mean_left_force = left_force_sum / contact_count
            mean_right_force = right_force_sum / contact_count
            force_rmse = math.sqrt(force_error_sum / contact_count)
            damage_risk_fraction = damage_risk_steps / contact_count
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
                public_safety_limit_n=float(args.physical_force_bounds[1]),
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
                "berry_profile": args.berry_profile,
                "outcome": outcome,
                "success": outcome == "success",
                "steps": physics_step,
                "return": episode_return,
                "attempted_force_range_n": attempted_range.as_list(),
                "target_force_n": attempted_range.center_n,
                "mean_bilateral_grip_force_n": mean_force,
                "mean_bilateral_left_force_n": mean_left_force,
                "mean_bilateral_right_force_n": mean_right_force,
                "force_rmse_n": force_rmse,
                "peak_grip_force_n": float(env.peak_grip_force_n[0].item()),
                "peak_pedicle_tension_n": float(env.peak_stem_tension_n[0].item()),
                "bilateral_contact_fraction": bilateral_steps / max(physics_step, 1),
                "damage_risk_fraction": damage_risk_fraction,
                "raspberry_relative_to_gripper_m": gripper_offset_m,
                "damage_threshold_n": float(env.damage_threshold_n[0].item()),
                "detach_threshold_n": float(env.detach_threshold_n[0].item()),
                "controller_active_steps": controller_active_steps,
                "safety_override_steps": safety_override_steps,
                "residual_mode": (
                    "semantic_xy"
                    if args.method in SEMANTIC_XY_METHODS
                    else "semantic"
                    if semantic_layout is not None
                    else "flow_latent"
                ),
                "mean_abs_residual_correction": (
                    correction_abs_sum / max(dsrl_active_decisions, 1)
                ).detach().cpu().tolist(),
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
                    "success": outcome == "success",
                    "mean_bilateral_grip_force_n": mean_force,
                    "mean_bilateral_left_force_n": mean_left_force,
                    "mean_bilateral_right_force_n": mean_right_force,
                    "force_rmse_n": force_rmse,
                    "peak_grip_force_n": episode_record["peak_grip_force_n"],
                    "peak_pedicle_tension_n": episode_record["peak_pedicle_tension_n"],
                    "bilateral_contact_fraction": episode_record["bilateral_contact_fraction"],
                    "damage_risk_fraction": damage_risk_fraction,
                    "raspberry_relative_to_gripper_m": gripper_offset_m,
                    "force_trace_sample": sampled_trace,
                },
                "recent_history": list(recent_history),
            }
            response: dict[str, Any]
            vlm_error: str | None = None
            if oracle_force is not None:
                response = {
                    "force_range_n": attempted_range.as_list(),
                    "failure_cause": "not_applicable",
                    "force_assessment": "simulator_ground_truth",
                    "rationale": (
                        "Fixed oracle interval computed from the true profile damage threshold; "
                        "no VLM was called"
                    ),
                    "confidence": 1.0,
                }
                vlm_error = None
            elif advisor is None:
                response = {
                    "force_range_n": attempted_range.as_list(),
                    "failure_cause": "unknown",
                    "force_assessment": "unknown",
                    "rationale": "Codex VLM explicitly disabled",
                    "confidence": 0.0,
                }
                vlm_error = "Codex VLM explicitly disabled"
            else:
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
            if oracle_force is not None:
                guarded = GuardedForceProposal(
                    tuple(attempted_range.as_list()),
                    "simulator_ground_truth_force",
                    "hold_oracle_force",
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
                    damage_threshold_n=episode_record["damage_threshold_n"],
                    damage_risk_fraction=damage_risk_fraction,
                    recent_candidate_directions=tuple(recent_candidate_directions),
                    confirmations_required=args.guard_confirmations,
                    maximum_center_step_n=args.vlm_max_center_step_n,
                )
            recent_candidate_directions.append(guarded.candidate_direction)
            proposal = ForceRange(*guarded.force_range_n)
            decision = adapter.update_after_episode(
                episode_index=episode_index,
                proposed_range=proposal if vlm_error is None else None,
                raw_response=json.dumps(response, ensure_ascii=False),
                error=vlm_error,
            )
            decision.update(
                {
                    "method": args.method,
                    "raw_proposal_n": raw_proposal.as_list(),
                    "guard_evidence": guarded.evidence,
                    "guard_decision": guarded.decision,
                    "candidate_direction": guarded.candidate_direction,
                    "confirmation_streak": guarded.confirmation_streak,
                }
            )
            episode_record.update(
                {
                    "vlm_call_index": None if advisor is None else advisor.calls,
                    "vlm_call_succeeded": vlm_error is None and advisor is not None,
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
                    "damage_risk_fraction": damage_risk_fraction,
                    "guard_evidence": guarded.evidence,
                    "guard_decision": guarded.decision,
                    "next_range_n": decision["applied_range_n"],
                }
            )
            append_jsonl(output_root / "training_episodes.jsonl", episode_record)
            append_jsonl(output_root / "force_range_decisions.jsonl", decision)
            if (episode_index + 1) % args.checkpoint_every == 0:
                save_training_checkpoint(
                    checkpoint_dir / f"episode_{episode_index + 1:03d}.pt",
                    agent=agent,
                    layout=layout,
                    semantic_layout=semantic_layout,
                    replay=replay,
                    adapter=adapter,
                    controller=controller,
                    base_hash=base_hash,
                    completed_episodes=episode_index + 1,
                    replay_transitions=total_transitions,
                )
            rolling = sum(int(item["success"]) for item in records[-10:]) / min(10, len(records))
            print(
                f"[ONLINE] episode={episode_index + 1}/{args.train_episodes} "
                f"outcome={outcome} rolling10={rolling:.1%} "
                f"force={episode_record['peak_grip_force_n']:.3f}N "
                f"range={attempted_range.as_list()}->{decision['applied_range_n']} "
                f"replay={replay.size} updates={agent.updates} "
                f"codex_ok={episode_record['vlm_call_succeeded']}",
                flush=True,
            )
    finally:
        env.close()

    final_checkpoint = checkpoint_dir / "final.pt"
    save_training_checkpoint(
        final_checkpoint,
        agent=agent,
        layout=layout,
        semantic_layout=semantic_layout,
        replay=replay,
        adapter=adapter,
        controller=controller,
        base_hash=base_hash,
        completed_episodes=len(records),
        replay_transitions=total_transitions,
    )
    successes = sum(int(item["success"]) for item in records)
    failures = Counter(item["outcome"] for item in records if not item["success"])
    summary = {
        "experiment": (
            "RaspberryPick privileged ground-truth force plus DSRL ablation"
            if oracle_force is not None
            else "Repaired paired RaspberryPick online DSRL/VLM experiment"
        ),
        "post_training_method": args.method,
        "force_adaptation": (
            "oracle_fixed_from_simulator_ground_truth"
            if oracle_force is not None
            else "fixed"
            if args.method in NO_VLM_METHODS
            else args.method
        ),
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "task": TASK,
        "berry_profile": args.berry_profile,
        "profile_persistence": "entire_run",
        "profile_values_hidden_from_actor_and_vlm": oracle_force is None,
        "profile_values_exposed_only_to_force_oracle": oracle_force is not None,
        "force_advisor_source": (
            "simulator_ground_truth" if oracle_force is not None else "codex_or_none"
        ),
        "training_seed": args.train_seed,
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
            "semantic_xy"
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
        "flow_initial_latent": (
            "zero" if semantic_layout is not None else "zero_plus_bounded_dsrl_correction"
        ),
        "noise_action_indices": list(layout.active_action_indices),
        "noise_residual_scale": layout.residual_scale,
        "dsrl_start_step": args.dsrl_start_step,
        "actor_observation_features": (
            "pooled_condition+visual_xy+normalized_state+physics"
            if args.method in PREPULL_SEMANTIC_METHODS
            else "pooled_condition+physics"
        ),
        "residual_active_until_phase": (
            float(args.pull_z_start_phase)
            if args.method in PREPULL_SEMANTIC_METHODS
            else None
        ),
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
        "final_force_range_n": adapter.current_range.as_list(),
        "physical_force_bounds_n": list(map(float, args.physical_force_bounds)),
        "damage_threshold_n": float(config.damage_grip_force_threshold_n),
        "detach_threshold_n": float(config.detach_force_threshold_n),
        "stem_stiffness_n_per_m": float(config.stem_stiffness_n_per_m),
        "stem_damping_n_s_per_m": float(config.stem_damping_n_s_per_m),
        "reward_reference": "physical_environment_constraints_not_vlm_target",
        "position_randomization_m": list(map(float, args.position_range)),
        "final_checkpoint": str(final_checkpoint.resolve()),
        "elapsed_seconds": time.time() - started,
    }
    (output_root / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("[SUMMARY] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
