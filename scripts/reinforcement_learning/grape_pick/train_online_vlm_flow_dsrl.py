#!/usr/bin/env python3
"""Train genuine Flow-noise DSRL with episodic VLM force adaptation.

The SAC action is a bounded correction to the frozen Flow policy's initial
Gaussian noise.  A local VLM proposes a force interval after every completed
episode, and a 120 Hz gripper-width controller applies that interval only to
CAFE action index 9 during the next episode.
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
import select
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _reexec_with_cuda_libraries() -> None:
    py_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / py_dir / "site-packages"
    required = [site_packages / "torch" / "lib", site_packages / "nvidia" / "cudnn" / "lib"]
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
DEFAULT_QWEN = Path(
    "/home/limx/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
    "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--flow-checkpoint", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--train-episodes", type=int, default=100)
parser.add_argument("--train-seed", type=int, default=907300)
parser.add_argument("--max-episode-steps", type=int, default=790)
parser.add_argument("--action-repeat", type=int, default=1)
parser.add_argument("--integration-steps", type=int, default=16)
parser.add_argument("--action-smoothing", type=float, default=0.35)
parser.add_argument("--noise-residual-scale", type=float, default=5.0)
parser.add_argument("--learned-noise-steps", type=int, default=1)
parser.add_argument(
    "--noise-action-index",
    type=int,
    default=2,
    help="CAFE action channel whose initial Flow noise is corrected (2 is tool Z).",
)
parser.add_argument(
    "--dsrl-start-step",
    type=int,
    default=624,
    help="Earliest physics step at which bilateral-contact Flow-noise DSRL may act.",
)
parser.add_argument("--initial-log-std", type=float, default=-2.5)
parser.add_argument("--damage-range", type=float, nargs=2, default=(3.4, 4.6))
parser.add_argument("--detach-range", type=float, nargs=2, default=(3.0, 3.7))
parser.add_argument("--initial-force-range", type=float, nargs=2, default=(2.90, 3.20))
parser.add_argument("--force-safety-margin", type=float, default=0.15)
parser.add_argument("--force-range-minimum", type=float, default=2.60)
parser.add_argument("--force-range-smoothing", type=float, default=0.35)
parser.add_argument("--learning-starts", type=int, default=512)
parser.add_argument("--replay-capacity", type=int, default=100000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--updates-per-decision", type=int, default=1)
parser.add_argument("--learning-rate", type=float, default=3.0e-4)
parser.add_argument("--exploration-start", type=float, default=1.0)
parser.add_argument("--exploration-end", type=float, default=0.15)
parser.add_argument(
    "--vlm-python",
    type=Path,
    default=Path("/home/limx/anaconda3/envs/vlarl2/bin/python"),
)
parser.add_argument("--vlm-model", type=Path, default=DEFAULT_QWEN)
parser.add_argument("--vlm-timeout-s", type=float, default=180.0)
parser.add_argument("--disable-vlm", action="store_true")
parser.add_argument(
    "--advisor-visual",
    choices=("diagnostic", "rgb"),
    default="diagnostic",
    help=(
        "diagnostic renders the real episode force/tension trace without loading the "
        "memory-heavy RTX camera stack; rgb captures the simulator third-person camera"
    ),
)
parser.add_argument("--checkpoint-every", type=int, default=10)
parser.add_argument(
    "--physx-contact-capacity",
    type=int,
    default=2**20,
    help="GPU rigid-contact and patch buffer capacity for this single environment.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.train_episodes < 1:
    parser.error("train-episodes must be positive")
if args.action_repeat < 1 or args.integration_steps < 1:
    parser.error("action-repeat and integration-steps must be positive")
if args.dsrl_start_step < 0:
    parser.error("dsrl-start-step must be non-negative")
if args.physx_contact_capacity < 2**16:
    parser.error("physx-contact-capacity is implausibly small")
if args.damage_range[0] - args.force_safety_margin <= args.force_range_minimum:
    parser.error("force safety interval is empty")
args.enable_cameras = args.advisor_visual == "rgb"

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import torch
from PIL import Image, ImageDraw

from grape_flow_policy import (
    FLOW_NOISE_SEED_OFFSET,
    FlowNoiseDSRLLayout,
    load_flow_checkpoint,
    pull_phase_noise_gate,
)
from grape_force_control import (
    ForceControllerConfig,
    ForceRange,
    ForceRangeAdapterConfig,
    GrapeForceController,
    OnlineForceRangeAdapter,
)
from learning import ReplayBuffer, SACBatch, SACResidualAgent
from tacex_tasks.grape_pick.grape_pick_env import GrapePickEnv
from tacex_tasks.grape_pick.grape_pick_env_cfg import GrapePickEnvCfg
from tacex_tasks.grape_pick.policy import environment_policy_observation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class LocalQwenClient:
    def __init__(self, stderr_path: Path) -> None:
        worker = Path(__file__).with_name("local_qwen_force_worker.py")
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TRANSFORMERS_VERBOSITY": "error",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        # The Isaac process re-executes with env_isaaclab's CUDA libraries at
        # the front of LD_LIBRARY_PATH.  A child from vlarl2 must not inherit
        # that cuDNN: its torch build otherwise resolves an incompatible
        # libcudnn_graph.so and exits on the first generation call.
        vlm_prefix = args.vlm_python.expanduser().resolve().parent.parent
        vlm_library_dirs: list[str] = []
        for site_packages in sorted((vlm_prefix / "lib").glob("python*/site-packages")):
            for relative in ("torch/lib", "nvidia/cudnn/lib"):
                candidate = site_packages / relative
                if candidate.is_dir():
                    vlm_library_dirs.append(str(candidate))
        inherited = [
            item
            for item in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if item and "env_isaaclab" not in item
        ]
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*vlm_library_dirs, *inherited]
        )
        environment["PATH"] = os.pathsep.join(
            [str(vlm_prefix / "bin"), environment.get("PATH", "")]
        )
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(args.vlm_python),
                str(worker),
                "--model",
                str(args.vlm_model),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
            env=environment,
        )
        ready = self._read(timeout_s=args.vlm_timeout_s)
        if not ready.get("ready") or not ready.get("advisor_is_real_vlm"):
            raise RuntimeError(f"local VLM worker failed readiness check: {ready}")
        self.ready = ready
        self.calls = 0

    def _read(self, *, timeout_s: float) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("VLM stdout pipe is unavailable")
        readable, _, _ = select.select([self.process.stdout], [], [], timeout_s)
        if not readable:
            raise TimeoutError(f"local VLM did not respond within {timeout_s:.1f}s")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"local VLM worker exited with code {self.process.poll()}"
            )
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("local VLM response is not a JSON object")
        return value

    def recommend(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None:
            raise RuntimeError("VLM stdin pipe is unavailable")
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        self.calls += 1
        return self._read(timeout_s=args.vlm_timeout_s)

    def close(self) -> None:
        try:
            if self.process.poll() is None and self.process.stdin is not None:
                self.process.stdin.write('{"command":"shutdown"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=15)
        except Exception:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self._stderr.close()


def classify(flags: dict[str, torch.Tensor]) -> str:
    if bool(flags["success"][0].item()):
        return "success"
    for name in ("crushed", "slipped", "obstacle_collision", "workspace_failure"):
        if bool(flags[name][0].item()):
            return name
    if bool(flags["stem_attached"][0].item()):
        return "no_detach"
    return "timeout_after_detach"


def terminal_reward(
    flags: dict[str, torch.Tensor], *, terminal: bool, force_target_n: float
) -> float:
    grip = float(flags["grip_force_n"][0].item())
    tension = float(flags["stem_tension_n"][0].item())
    detach_threshold = max(float(flags["detach_threshold_n"][0].item()), 1.0e-6)
    bilateral = bool(flags["bilateral_contact"][0].item())
    reward = 0.002 * int(bilateral)
    reward += 0.004 * min(tension / detach_threshold, 1.2)
    if bilateral:
        reward -= 0.001 * min(((grip - force_target_n) / 0.30) ** 2, 4.0)
    if terminal:
        outcome = classify(flags)
        reward += {
            "success": 10.0,
            "crushed": -10.0,
            "slipped": -6.0,
            "obstacle_collision": -10.0,
            "workspace_failure": -6.0,
            "no_detach": -3.0,
            "timeout_after_detach": 1.0,
        }[outcome]
    return reward


def actor_observation(
    normalized_flow_observation: torch.Tensor,
    force_range: ForceRange,
    grip_force_n: torch.Tensor,
) -> torch.Tensor:
    batch = len(normalized_flow_observation)
    low = torch.full(
        (batch, 1),
        force_range.minimum_n / 4.0,
        dtype=normalized_flow_observation.dtype,
        device=normalized_flow_observation.device,
    )
    high = torch.full_like(low, force_range.maximum_n / 4.0)
    error = (force_range.center_n - grip_force_n.reshape(-1, 1)) / 4.0
    return torch.cat((normalized_flow_observation, low, high, error), dim=-1)


def capture_advisor_visual(
    env: GrapePickEnv,
    path: Path,
    *,
    force_trace: list[dict[str, float | int | bool]],
    outcome: str,
    attempted_range: ForceRange,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.advisor_visual == "rgb":
        env.sim.render()
        env.sim.render()
        frame = env.get_camera_rgb("third")[0].detach().cpu().numpy()
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        Image.fromarray(frame).save(path)
        return

    # A source-grounded visual episode card avoids loading Isaac's RTX camera
    # stack while still giving the VLM a visual force/tension trajectory.  No
    # values are synthesized: every point comes from the simulator trace.
    width, height = 960, 540
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), f"GrapePick episode {outcome}", fill=(20, 25, 35))
    draw.text(
        (28, 44),
        f"target grip band: {attempted_range.minimum_n:.2f}-{attempted_range.maximum_n:.2f} N",
        fill=(20, 80, 45),
    )
    draw.text(
        (600, 20),
        f"damage floor: {float(args.damage_range[0]):.2f} N",
        fill=(170, 35, 35),
    )
    left, top, right, bottom = 70, 90, 910, 470
    draw.rectangle((left, top, right, bottom), outline=(80, 90, 105), width=2)
    values = [float(item["grip_force_n"]) for item in force_trace]
    tensions = [float(item["stem_tension_n"]) for item in force_trace]
    steps = [int(item["physics_step"]) for item in force_trace]
    y_max = max(5.0, max(values, default=0.0), max(tensions, default=0.0))
    x_max = max(steps, default=1)

    def x_coord(step: int) -> float:
        return left + (right - left) * step / max(x_max, 1)

    def y_coord(value: float) -> float:
        return bottom - (bottom - top) * value / y_max

    band_top = y_coord(attempted_range.maximum_n)
    band_bottom = y_coord(attempted_range.minimum_n)
    draw.rectangle(
        (left, band_top, right, band_bottom), fill=(218, 241, 223)
    )
    damage_y = y_coord(float(args.damage_range[0]))
    draw.line((left, damage_y, right, damage_y), fill=(210, 55, 55), width=3)
    if len(steps) >= 2:
        grip_points = [
            (x_coord(step), y_coord(value)) for step, value in zip(steps, values)
        ]
        tension_points = [
            (x_coord(step), y_coord(value))
            for step, value in zip(steps, tensions)
        ]
        draw.line(grip_points, fill=(25, 90, 200), width=3)
        draw.line(tension_points, fill=(220, 135, 20), width=3)
    for force_tick in range(0, int(math.ceil(y_max)) + 1):
        y = y_coord(float(force_tick))
        draw.line((left - 5, y, left, y), fill=(80, 90, 105), width=1)
        draw.text((24, y - 7), f"{force_tick}N", fill=(60, 70, 85))
    draw.text((72, 486), "blue: pad grip force", fill=(25, 90, 200))
    draw.text((280, 486), "orange: stem tension", fill=(190, 105, 10))
    draw.text((510, 486), "green: target band", fill=(20, 110, 55))
    draw.text((720, 486), "red: damage floor", fill=(180, 35, 35))
    image.save(path)


def save_training_checkpoint(
    path: Path,
    *,
    agent: SACResidualAgent,
    completed_episodes: int,
    layout: FlowNoiseDSRLLayout,
    adapter: OnlineForceRangeAdapter,
    controller: GrapeForceController,
    flow_hash: str,
    total_transitions: int,
) -> None:
    payload = {
        **agent.checkpoint(),
        "contract": "grape_flow_initial_noise_dsrl_sac_with_online_vlm_force_control",
        "contract_version": 2,
        "rl_action_semantics": (
            "pull-gated bounded correction to selected native Flow initial Gaussian "
            "noise channels"
        ),
        "zero_actor_semantics": (
            "exact frozen Flow decoder output for identical native noise before the "
            "independent index-9 force controller"
        ),
        "flow_checkpoint": str(args.flow_checkpoint.resolve()),
        "flow_checkpoint_sha256": flow_hash,
        "flow_horizon": layout.horizon,
        "flow_action_dim": layout.action_dim,
        "learned_noise_steps": layout.learned_noise_steps,
        "noise_residual_scale": layout.residual_scale,
        "noise_padding_mode": layout.padding_mode,
        "noise_action_indices": list(layout.active_action_indices or ()),
        "dsrl_activation": {
            "minimum_physics_step": args.dsrl_start_step,
            "requires_bilateral_contact": True,
            "requires_stem_attached": True,
            "pre_activation_correction": "exact_zero",
            "replay_scope": "active_pull_phase_only",
        },
        "flow_execution": {
            "action_repeat": args.action_repeat,
            "integration_steps": args.integration_steps,
            "action_smoothing": args.action_smoothing,
            "native_noise_distribution": "independent standard Gaussian per policy decision",
            "native_noise_seed_offset": FLOW_NOISE_SEED_OFFSET,
            "physx_contact_capacity": args.physx_contact_capacity,
        },
        "initial_actor_log_std": args.initial_log_std,
        "actor_observation_dim": 14,
        "actor_observation_components": [
            "normalized_flow_observation_11d",
            "vlm_force_low_div_4n",
            "vlm_force_high_div_4n",
            "target_minus_grip_force_div_4n",
        ],
        "force_controller_action_index": 9,
        "force_controller_config": asdict(controller.config),
        "completed_episodes": completed_episodes,
        "total_transitions": total_transitions,
        "train_seed": args.train_seed,
        "current_force_range_n": adapter.current_range.as_list(),
        "force_range_decisions": adapter.decisions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    frame_dir = output_root / "advisor_frames"
    trajectory_dir = output_root / "trajectories"
    checkpoint_dir = output_root / "checkpoints"
    frame_dir.mkdir()
    trajectory_dir.mkdir()
    checkpoint_dir.mkdir()
    training_log = output_root / "training_episodes.jsonl"
    vlm_log = output_root / "vlm_force_updates.jsonl"

    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)

    config = GrapePickEnvCfg()
    config.scene.num_envs = 1
    config.seed = args.train_seed
    config.enable_cameras = args.advisor_visual == "rgb"
    config.enable_gelsight = False
    config.randomize_force_thresholds = True
    config.sim.physx.gpu_max_rigid_contact_count = args.physx_contact_capacity
    config.sim.physx.gpu_max_rigid_patch_count = args.physx_contact_capacity
    config.damage_force_randomization_range_n = tuple(args.damage_range)
    config.detach_force_randomization_range_n = tuple(args.detach_range)
    config.episode_length_s = max(args.max_episode_steps * config.sim.dt, 1.0)
    if args.device:
        config.sim.device = args.device
    env = GrapePickEnv(
        config, render_mode="rgb_array" if config.enable_cameras else None
    )
    device = torch.device(config.sim.device)
    loaded = load_flow_checkpoint(args.flow_checkpoint, device=device)
    loaded.model.requires_grad_(False)
    loaded.model.eval()
    layout = FlowNoiseDSRLLayout(
        horizon=loaded.model.config.horizon,
        action_dim=loaded.model.config.action_dim,
        learned_noise_steps=args.learned_noise_steps,
        residual_scale=args.noise_residual_scale,
        padding_mode="repeat_last",
        active_action_indices=(args.noise_action_index,),
    )
    agent = SACResidualAgent(
        observation_dim=14,
        action_dim=layout.correction_dim,
        device=device,
        learning_rate=args.learning_rate,
        gamma=0.995,
        tau=0.01,
        residual_l2_weight=0.10,
        initial_log_std=args.initial_log_std,
        initial_alpha=0.03,
    )
    replay = ReplayBuffer(
        args.replay_capacity, 14, layout.correction_dim, device
    )
    safe_maximum = float(args.damage_range[0]) - args.force_safety_margin
    adapter = OnlineForceRangeAdapter(
        ForceRange(*map(float, args.initial_force_range)),
        ForceRangeAdapterConfig(
            physical_minimum_n=args.force_range_minimum,
            physical_maximum_n=safe_maximum,
            minimum_width_n=0.15,
            smoothing=args.force_range_smoothing,
        ),
    )
    controller = GrapeForceController(
        adapter.current_range,
        ForceControllerConfig(
            hard_force_limit_n=safe_maximum,
            free_space_close_rate_m_s=0.006,
            maximum_width_rate_m_s=0.006,
            safety_open_rate_m_s=0.006,
            release_hysteresis_steps=120,
        ),
    )
    flow_hash = sha256(args.flow_checkpoint)
    vlm: LocalQwenClient | None = None
    if not args.disable_vlm:
        vlm = LocalQwenClient(output_root / "vlm_worker_stderr.log")
        (output_root / "vlm_worker_ready.json").write_text(
            json.dumps(vlm.ready, indent=2) + "\n", encoding="utf-8"
        )

    records: list[dict[str, Any]] = []
    recent_history: deque[dict[str, Any]] = deque(maxlen=10)
    total_transitions = 0
    last_update: dict[str, float] = {}
    start_time = time.time()
    try:
        for episode_index in range(args.train_episodes):
            episode_seed = args.train_seed + episode_index
            env.reset(seed=episode_seed)
            controller.set_target_range(adapter.current_range)
            controller.reset()
            attempted_range = adapter.current_range
            generator = torch.Generator(device=device).manual_seed(
                episode_seed + FLOW_NOISE_SEED_OFFSET
            )
            previous_action: torch.Tensor | None = None
            physics_step = 0
            episode_return = 0.0
            controller_active_steps = 0
            safety_override_steps = 0
            in_band_steps = 0
            bilateral_steps = 0
            force_sum = 0.0
            force_squared_error_sum = 0.0
            correction_abs_sum = torch.zeros(layout.correction_dim, device=device)
            correction_decisions = 0
            dsrl_active_decisions = 0
            pre_activation_decisions = 0
            first_dsrl_active_step: int | None = None
            force_trace: list[dict[str, float | int | bool]] = []
            outcome = "no_detach"

            trajectory_path = trajectory_dir / f"episode_{episode_index:03d}.jsonl.gz"
            with gzip.open(trajectory_path, "xt", encoding="utf-8") as trajectory_stream:
                while physics_step < args.max_episode_steps:
                    metrics_before = env.get_contact_force_metrics()
                    raw_observation = environment_policy_observation(
                        env, max_episode_steps=loaded.max_episode_steps
                    )
                    normalized = loaded.normalize_observation(raw_observation)
                    policy_observation = actor_observation(
                        normalized, attempted_range, metrics_before["grip_force_n"]
                    )
                    progress = episode_index / max(args.train_episodes - 1, 1)
                    exploration = args.exploration_start + progress * (
                        args.exploration_end - args.exploration_start
                    )
                    dsrl_active = bool(
                        pull_phase_noise_gate(
                            physics_step,
                            metrics_before["bilateral_contact"],
                            env.stem_attached,
                            minimum_step=args.dsrl_start_step,
                        )[0].item()
                    )
                    with torch.inference_mode():
                        if dsrl_active:
                            correction = agent.act(
                                policy_observation,
                                deterministic=False,
                                exploration_scale=exploration,
                            )
                        else:
                            correction = torch.zeros(
                                (1, layout.correction_dim), device=device
                            )
                        native_noise = torch.randn(
                            (
                                1,
                                loaded.model.config.horizon,
                                loaded.model.config.action_dim,
                            ),
                            generator=generator,
                            device=device,
                        )
                        decoder_noise = layout.compose_noise(native_noise, correction)
                        normalized_chunk = loaded.model.decode(
                            normalized,
                            decoder_noise,
                            integration_steps=args.integration_steps,
                            solver="heun",
                        )
                        target = loaded.denormalize_action(normalized_chunk)[:, 0]
                    if dsrl_active:
                        correction_abs_sum += correction[0].abs()
                        correction_decisions += 1
                        dsrl_active_decisions += 1
                        if first_dsrl_active_step is None:
                            first_dsrl_active_step = physics_step
                    else:
                        pre_activation_decisions += 1
                    transition_reward = 0.0
                    terminal = False
                    final_flags: dict[str, torch.Tensor] | None = None

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
                        non_gripper_delta = (
                            executed_action[:, :9] - policy_action[:, :9]
                        ).abs().max()
                        if float(non_gripper_delta.item()) > 1.0e-8:
                            raise RuntimeError("force controller changed a non-gripper action")
                        env._pre_physics_step(executed_action)
                        env._apply_action()
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(dt=env.physics_dt)
                        physics_step += 1
                        flags = env.get_task_flags()
                        terminal = bool(
                            (
                                flags["success"]
                                | flags["crushed"]
                                | flags["slipped"]
                                | flags["obstacle_collision"]
                                | flags["workspace_failure"]
                            )[0].item()
                        ) or physics_step >= args.max_episode_steps
                        step_reward = terminal_reward(
                            flags,
                            terminal=terminal,
                            force_target_n=attempted_range.center_n,
                        )
                        transition_reward += step_reward
                        episode_return += step_reward
                        grip = float(flags["grip_force_n"][0].item())
                        bilateral = bool(flags["bilateral_contact"][0].item())
                        bilateral_steps += int(bilateral)
                        if bilateral:
                            force_sum += grip
                            force_squared_error_sum += (
                                grip - attempted_range.center_n
                            ) ** 2
                            in_band_steps += int(
                                attempted_range.minimum_n
                                <= grip
                                <= attempted_range.maximum_n
                            )
                        controller_active_steps += int(diagnostics.active[0].item())
                        safety_override_steps += int(
                            diagnostics.safety_override[0].item()
                        )
                        if physics_step == 1 or physics_step % 4 == 0 or terminal:
                            trace_row = {
                                "physics_step": physics_step,
                                "grip_force_n": grip,
                                "left_force_n": float(
                                    flags["left_normal_force_n"][0].item()
                                ),
                                "right_force_n": float(
                                    flags["right_normal_force_n"][0].item()
                                ),
                                "stem_tension_n": float(
                                    flags["stem_tension_n"][0].item()
                                ),
                                "bilateral_contact": bilateral,
                                "controller_active": bool(
                                    diagnostics.active[0].item()
                                ),
                                "dsrl_active": dsrl_active,
                                "noise_correction": correction[0].detach().cpu().tolist(),
                                "safety_override": bool(
                                    diagnostics.safety_override[0].item()
                                ),
                                "policy_width_m": float(policy_action[0, 9].item()),
                                "executed_width_m": float(
                                    executed_action[0, 9].item()
                                ),
                            }
                            force_trace.append(trace_row)
                            trajectory_stream.write(
                                json.dumps(trace_row, allow_nan=False) + "\n"
                            )
                        final_flags = flags
                        if terminal:
                            break

                    next_metrics = env.get_contact_force_metrics()
                    next_raw_observation = environment_policy_observation(
                        env, max_episode_steps=loaded.max_episode_steps
                    )
                    next_normalized = loaded.normalize_observation(next_raw_observation)
                    next_policy_observation = actor_observation(
                        next_normalized, attempted_range, next_metrics["grip_force_n"]
                    )
                    next_dsrl_active = bool(
                        pull_phase_noise_gate(
                            physics_step,
                            next_metrics["bilateral_contact"],
                            env.stem_attached,
                            minimum_step=args.dsrl_start_step,
                        )[0].item()
                    )
                    if dsrl_active:
                        replay.add(
                            SACBatch(
                                observation=policy_observation.detach().cpu(),
                                action=correction.detach().cpu(),
                                reward=torch.tensor(
                                    [transition_reward / 10.0], dtype=torch.float32
                                ),
                                next_observation=next_policy_observation.detach().cpu(),
                                done=torch.tensor(
                                    [float(terminal or not next_dsrl_active)],
                                    dtype=torch.float32,
                                ),
                            )
                        )
                        total_transitions += 1
                        if replay.size >= args.learning_starts:
                            for _ in range(args.updates_per_decision):
                                last_update = agent.update(replay.sample(args.batch_size))
                    if terminal:
                        assert final_flags is not None
                        outcome = classify(final_flags)
                        break

            assert final_flags is not None
            contact_count = max(bilateral_steps, 1)
            mean_force = force_sum / contact_count
            force_rmse = math.sqrt(force_squared_error_sum / contact_count)
            frame_path = frame_dir / f"episode_{episode_index:03d}_{outcome}.png"
            capture_advisor_visual(
                env,
                frame_path,
                force_trace=force_trace,
                outcome=outcome,
                attempted_range=attempted_range,
            )
            episode_record: dict[str, Any] = {
                "episode_index": episode_index,
                "seed": episode_seed,
                "outcome": outcome,
                "success": outcome == "success",
                "steps": physics_step,
                "return": episode_return,
                "attempted_force_range_n": attempted_range.as_list(),
                "target_force_n": attempted_range.center_n,
                "mean_bilateral_grip_force_n": mean_force,
                "force_rmse_n": force_rmse,
                "peak_grip_force_n": float(env.peak_grip_force_n[0].item()),
                "peak_stem_tension_n": float(env.peak_stem_tension_n[0].item()),
                "damage_threshold_n": float(env.damage_threshold_n[0].item()),
                "detach_threshold_n": float(env.detach_threshold_n[0].item()),
                "bilateral_contact_fraction": bilateral_steps / max(physics_step, 1),
                "in_force_band_fraction_given_bilateral": in_band_steps / contact_count,
                "controller_active_steps": controller_active_steps,
                "safety_override_steps": safety_override_steps,
                "mean_abs_noise_correction": (
                    correction_abs_sum / max(correction_decisions, 1)
                ).detach().cpu().tolist(),
                "noise_correction_decisions": correction_decisions,
                "dsrl_active_decisions": dsrl_active_decisions,
                "pre_activation_zero_correction_decisions": pre_activation_decisions,
                "first_dsrl_active_step": first_dsrl_active_step,
                "noise_action_indices": list(layout.active_action_indices or ()),
                "flow_native_noise_distribution": "independent standard Gaussian",
                "trajectory": str(trajectory_path),
                "advisor_frame": str(frame_path),
                "replay_size": replay.size,
                "sac_updates": agent.updates,
                "last_update": last_update,
            }

            sampled_trace = force_trace[:: max(1, len(force_trace) // 40)][:40]
            request = {
                "image_path": str(frame_path),
                "visual_type": args.advisor_visual,
                "physical_force_bounds_n": [args.force_range_minimum, safe_maximum],
                "minimum_damage_force_n": float(args.damage_range[0]),
                "current_range_n": attempted_range.as_list(),
                "episode": {
                    "episode_index": episode_index,
                    "outcome": outcome,
                    "success": outcome == "success",
                    "attempted_force_range_n": attempted_range.as_list(),
                    "mean_bilateral_grip_force_n": mean_force,
                    "force_rmse_n": force_rmse,
                    "peak_grip_force_n": episode_record["peak_grip_force_n"],
                    "peak_stem_tension_n": episode_record["peak_stem_tension_n"],
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
            if vlm is None:
                vlm_response = {
                    "ok": False,
                    "advisor_is_real_vlm": False,
                    "raw_response": "",
                    "error": "VLM explicitly disabled",
                }
            else:
                try:
                    vlm_response = vlm.recommend(request)
                except Exception as exc:
                    vlm_response = {
                        "ok": False,
                        "advisor_is_real_vlm": True,
                        "raw_response": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            proposal: ForceRange | None = None
            if vlm_response.get("ok"):
                try:
                    proposal = ForceRange(*map(float, vlm_response["force_range_n"]))
                except Exception as exc:
                    vlm_response["ok"] = False
                    vlm_response["error"] = f"invalid parsed range: {exc}"
            decision = adapter.update_after_episode(
                episode_index=episode_index,
                proposed_range=proposal,
                raw_response=str(vlm_response.get("raw_response", "")),
                error=vlm_response.get("error"),
            )
            episode_record["vlm_call_index"] = episode_index + 1
            episode_record["vlm_call_succeeded"] = bool(vlm_response.get("ok"))
            episode_record["next_force_range_n"] = decision["applied_range_n"]
            records.append(episode_record)
            recent_history.append(
                {
                    "episode_index": episode_index,
                    "outcome": outcome,
                    "attempted_range_n": attempted_range.as_list(),
                    "mean_force_n": mean_force,
                    "peak_force_n": episode_record["peak_grip_force_n"],
                    "force_rmse_n": force_rmse,
                    "next_range_n": decision["applied_range_n"],
                }
            )
            append_jsonl(training_log, episode_record)
            append_jsonl(
                vlm_log,
                {
                    "advisor_call_index": episode_index + 1,
                    "history_size_before_call": len(recent_history) - 1,
                    "request": request,
                    "response": vlm_response,
                    "decision": decision,
                },
            )
            if (episode_index + 1) % args.checkpoint_every == 0:
                save_training_checkpoint(
                    checkpoint_dir / f"episode_{episode_index + 1:03d}.pt",
                    agent=agent,
                    completed_episodes=episode_index + 1,
                    layout=layout,
                    adapter=adapter,
                    controller=controller,
                    flow_hash=flow_hash,
                    total_transitions=total_transitions,
                )
            recent_successes = sum(item["success"] for item in records[-10:])
            print(
                f"[ONLINE] episode={episode_index + 1}/{args.train_episodes} "
                f"outcome={outcome} rolling10={recent_successes / min(10, len(records)):.1%} "
                f"force={episode_record['peak_grip_force_n']:.3f}N "
                f"range={attempted_range.as_list()}->{decision['applied_range_n']} "
                f"replay={replay.size} updates={agent.updates} vlm_ok={vlm_response.get('ok')}",
                flush=True,
            )
    finally:
        if vlm is not None:
            vlm.close()
        env.close()

    final_checkpoint = checkpoint_dir / "final.pt"
    save_training_checkpoint(
        final_checkpoint,
        agent=agent,
        completed_episodes=len(records),
        layout=layout,
        adapter=adapter,
        controller=controller,
        flow_hash=flow_hash,
        total_transitions=total_transitions,
    )
    successes = sum(item["success"] for item in records)
    failures = Counter(item["outcome"] for item in records if not item["success"])
    summary = {
        "experiment": "GrapePick genuine Flow-noise DSRL + online local VLM + 120Hz force control",
        "training_seed": args.train_seed,
        "completed_episodes": len(records),
        "successes": successes,
        "online_training_success_rate": successes / max(len(records), 1),
        "failure_counts": dict(failures),
        "flow_checkpoint": str(args.flow_checkpoint.resolve()),
        "flow_checkpoint_sha256": flow_hash,
        "dsrl_action_space": (
            f"bounded {layout.correction_dim}-D correction to native "
            f"[{layout.horizon},{layout.action_dim}] Flow Gaussian noise"
        ),
        "noise_action_indices": list(layout.active_action_indices or ()),
        "dsrl_activation": {
            "minimum_physics_step": args.dsrl_start_step,
            "requires_bilateral_contact": True,
            "requires_stem_attached": True,
            "pre_activation_correction": "exact_zero",
            "replay_scope": "active_pull_phase_only",
        },
        "noise_residual_scale": args.noise_residual_scale,
        "flow_execution": {
            "action_repeat": args.action_repeat,
            "integration_steps": args.integration_steps,
            "action_smoothing": args.action_smoothing,
            "native_noise_distribution": "independent standard Gaussian per policy decision",
            "native_noise_seed_offset": FLOW_NOISE_SEED_OFFSET,
            "physx_contact_capacity": args.physx_contact_capacity,
        },
        "sac_optimizer_updates": agent.updates,
        "replay_transitions": total_transitions,
        "vlm_advisor": "Qwen3-VL-4B-Instruct" if vlm is not None else "disabled",
        "vlm_is_real": vlm is not None,
        "vlm_visual_input": args.advisor_visual,
        "vlm_calls": 0 if vlm is None else vlm.calls,
        "vlm_successful_calls": sum(item["vlm_call_succeeded"] for item in records),
        "initial_force_range_n": list(map(float, args.initial_force_range)),
        "final_force_range_n": adapter.current_range.as_list(),
        "force_safety_bounds_n": [args.force_range_minimum, safe_maximum],
        "force_controller_rate_hz": 1.0 / float(config.sim.dt),
        "force_controller_owned_action_indices": [9],
        "force_controller_config": asdict(controller.config),
        "damage_force_randomization_range_n": list(args.damage_range),
        "detach_force_randomization_range_n": list(args.detach_range),
        "final_checkpoint": str(final_checkpoint),
        "elapsed_seconds": time.time() - start_time,
    }
    (output_root / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("[SUMMARY] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
