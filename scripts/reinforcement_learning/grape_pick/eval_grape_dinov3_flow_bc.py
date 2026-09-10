#!/usr/bin/env python3
"""Closed-loop evaluation for a trained GrapePick DINOv3 Flow Matching BC."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PICK_VARIANT = os.environ.get("TACEX_PICK_VARIANT", "grape").strip().lower()
if PICK_VARIANT not in {"grape", "raspberry"}:
    raise ValueError(
        "TACEX_PICK_VARIANT must be either 'grape' or 'raspberry', "
        f"got {PICK_VARIANT!r}"
    )
IS_RASPBERRY = PICK_VARIANT == "raspberry"


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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=910000)
parser.add_argument(
    "--fixed-episode-seed",
    action="store_true",
    help="Reset every episode to --seed; requires --num-envs 1 and is intended for replay tests.",
)
parser.add_argument(
    "--enhanced-determinism",
    action="store_true",
    help="Enable PhysX enhanced determinism for repeated single-seed evaluation.",
)
parser.add_argument("--max-episode-steps", type=int, default=790)
parser.add_argument("--action-repeat", type=int, default=4)
parser.add_argument("--chunk-execute-steps", type=int, default=1)
parser.add_argument("--num-inference-steps", type=int, default=None)
parser.add_argument("--action-smoothing", type=float, default=0.35)
parser.add_argument(
    "--position-command-offset-m",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(0.0, 0.0, 0.0),
    help="Constant non-privileged XYZ calibration added to every BC command.",
)
parser.add_argument(
    "--gripper-command-offset-m",
    type=float,
    default=0.0,
    help="Constant calibration added to the BC gripper-width command.",
)
parser.add_argument(
    "--late-gripper-max-width-m",
    type=float,
    default=None,
    help="Optional maximum gripper width after the close phase begins.",
)
parser.add_argument("--late-gripper-start-phase", type=float, default=0.48)
parser.add_argument(
    "--pull-z-offset-m",
    type=float,
    default=0.0,
    help="Additional Z pull offset, ramped in late in the episode.",
)
parser.add_argument("--pull-z-start-phase", type=float, default=0.72)
parser.add_argument("--pull-z-ramp-end-phase", type=float, default=0.88)
parser.add_argument(
    "--noise-mode",
    choices=("zero", "gaussian"),
    default="zero",
    help="Initial Flow latent. Zero is deterministic and matches the bottle protocol.",
)
parser.add_argument(
    "--use-ema",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Deploy EMA weights instead of the best epoch's raw model weights.",
)
parser.add_argument(
    "--visual-xy-override", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument(
    "--visual-xy-lock-phase",
    type=float,
    default=None,
    help=(
        "Optionally freeze the visual XY prediction after this normalized phase; "
        "the default keeps updating it throughout the rollout."
    ),
)
parser.add_argument(
    "--randomize-force-thresholds", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--randomize-grape-pose",
    "--randomize-raspberry-pose",
    dest="randomize_grape_pose",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--grape-position-range",
    "--raspberry-position-range",
    dest="grape_position_range",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help="Override the symmetric grape-position randomization range in metres.",
)
parser.add_argument(
    "--damage-range",
    type=float,
    nargs=2,
    default=(2.4, 3.1) if IS_RASPBERRY else (3.4, 4.6),
)
parser.add_argument(
    "--detach-range",
    type=float,
    nargs=2,
    default=(1.2, 1.6) if IS_RASPBERRY else (3.0, 3.7),
)
parser.add_argument("--damage-threshold-n", type=float, default=None)
parser.add_argument("--detach-threshold-n", type=float, default=None)
parser.add_argument("--video-path", type=Path, default=None)
parser.add_argument("--video-camera", choices=("third", "wrist"), default="third")
parser.add_argument("--video-env-index", type=int, default=0)
parser.add_argument("--video-fps", type=float, default=30.0)
parser.add_argument("--video-stride", type=int, default=4)
parser.add_argument(
    "--two-stage-force-control",
    action="store_true",
    help=(
        "Overlay oracle grip-force and pedicle-tension feedback on the frozen base policy. "
        "The simulator thresholds are used only by the controller."
    ),
)
parser.add_argument("--oracle-grip-damage-margin-n", type=float, default=0.20)
parser.add_argument("--oracle-grip-range-width-n", type=float, default=0.60)
parser.add_argument("--oracle-tension-margin-n", type=float, default=0.08)
parser.add_argument("--contact-confirm-steps", type=int, default=3)
parser.add_argument("--grip-stable-steps", type=int, default=6)
parser.add_argument("--tension-kp-velocity-m-s-per-n", type=float, default=0.012)
parser.add_argument("--maximum-pull-velocity-m-s", type=float, default=0.010)
parser.add_argument("--maximum-pull-offset-m", type=float, default=0.060)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1 or args.num_envs < 1:
    parser.error("--episodes and --num-envs must be positive")
if args.fixed_episode_seed and args.num_envs != 1:
    parser.error("--fixed-episode-seed requires --num-envs 1")
if args.action_repeat < 1 or args.chunk_execute_steps < 1:
    parser.error("--action-repeat and --chunk-execute-steps must be positive")
if args.video_env_index < 0 or args.video_env_index >= args.num_envs:
    parser.error("--video-env-index must be in [0, num-envs)")
if args.video_fps <= 0.0 or args.video_stride < 1:
    parser.error("--video-fps and --video-stride must be positive")
if not 0.0 <= args.pull_z_start_phase < args.pull_z_ramp_end_phase <= 1.0:
    parser.error("pull Z phases must satisfy 0 <= start < ramp end <= 1")
if not 0.0 <= args.late_gripper_start_phase <= 1.0:
    parser.error("--late-gripper-start-phase must be in [0, 1]")
if args.two_stage_force_control:
    if args.oracle_grip_damage_margin_n <= 0.0:
        parser.error("--oracle-grip-damage-margin-n must be positive")
    if args.oracle_grip_range_width_n <= 0.0:
        parser.error("--oracle-grip-range-width-n must be positive")
    if args.oracle_tension_margin_n <= 0.0:
        parser.error("--oracle-tension-margin-n must be positive")
    if args.grip_stable_steps < 1:
        parser.error("--grip-stable-steps must be positive")
    if args.contact_confirm_steps < 1:
        parser.error("--contact-confirm-steps must be positive")
    if args.randomize_force_thresholds and args.num_envs != 1:
        parser.error(
            "oracle two-stage control with randomized thresholds currently requires --num-envs 1"
        )
args.enable_cameras = True

launcher = AppLauncher(args)
simulation_app = launcher.app


REPO_ROOT = Path(__file__).resolve().parents[3]
BC_TRAINING_DIR = REPO_ROOT / "scripts" / "bc_training"
if str(BC_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(BC_TRAINING_DIR))

import dinov3_flow as flow  # noqa: E402
import train_bc as common  # noqa: E402
from grape_force_control import ForceRange  # noqa: E402
from two_stage_force_control import (  # noqa: E402
    STAGE_NAMES,
    TwoStageBerryForceController,
    TwoStageForceControllerConfig,
)
if IS_RASPBERRY:
    from tacex_tasks.raspberry_pick.raspberry_pick_env import (  # noqa: E402
        RaspberryPickEnv as GrapePickEnv,
    )
    from tacex_tasks.raspberry_pick.raspberry_pick_env_cfg import (  # noqa: E402
        RaspberryPickEnvCfg as GrapePickEnvCfg,
    )

    CONTRACT = "raspberry_dinov3_tactile_flow_matching_bc_v1"
    TASK = "TacEx-RaspberryPick-Force-v0"
    OBJECT_NAME = "raspberry"
    FORCE_ELEMENT = "pedicle"
else:
    from tacex_tasks.grape_pick.grape_pick_env import GrapePickEnv  # noqa: E402
    from tacex_tasks.grape_pick.grape_pick_env_cfg import GrapePickEnvCfg  # noqa: E402

    CONTRACT = "grape_dinov3_tactile_flow_matching_bc_v1"
    TASK = "TacEx-GrapePick-Force-v0"
    OBJECT_NAME = "grape"
    FORCE_ELEMENT = "stem"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, total: int) -> list[float]:
    if total < 1:
        return [0.0, 0.0]
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def classify(flags: dict[str, torch.Tensor], index: int) -> str:
    if bool(flags["success"][index].item()):
        return "success"
    for name in ("crushed", "slipped", "obstacle_collision", "workspace_failure"):
        if bool(flags[name][index].item()):
            return name
    if bool(flags["stem_attached"][index].item()):
        return "no_detach"
    return "timeout_after_detach"


def camera_rgb_224(env: GrapePickEnv, camera: str) -> torch.Tensor:
    image = env.get_camera_rgb(camera)[..., :3].permute(0, 3, 1, 2).float()
    image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
    return image.clamp(0.0, 255.0).div_(255.0)


def annotated_video_frame(
    env: GrapePickEnv,
    camera: str,
    env_index: int,
    physics_step: int,
    flags: dict[str, torch.Tensor],
    outcome_override: str | None = None,
) -> np.ndarray:
    """Return an RGB deployment frame with policy and force diagnostics."""

    import cv2

    frame = (
        env.get_camera_rgb(camera)[env_index, ..., :3]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8, copy=False)
    )
    output = np.ascontiguousarray(frame.copy())
    outcome = "RUNNING"
    color = (245, 245, 245)
    if bool(flags["success"][env_index].item()):
        outcome, color = "SUCCESS", (70, 230, 100)
    elif bool(flags["crushed"][env_index].item()):
        outcome, color = "CRUSHED", (250, 80, 80)
    elif bool(flags["slipped"][env_index].item()):
        outcome, color = "SLIPPED", (255, 190, 70)
    elif bool(flags["obstacle_collision"][env_index].item()):
        outcome, color = "COLLISION", (250, 80, 80)
    elif bool(flags["workspace_failure"][env_index].item()):
        outcome, color = "WORKSPACE FAILURE", (250, 80, 80)
    elif not bool(flags["stem_attached"][env_index].item()):
        outcome, color = "DETACHED", (90, 210, 255)
    if outcome_override is not None:
        outcome = outcome_override.replace("_", " ").upper()
        color = (70, 230, 100) if outcome_override == "success" else (255, 190, 70)

    grip = float(flags["grip_force_n"][env_index].item())
    tension = float(flags["stem_tension_n"][env_index].item())
    damage = float(flags["damage_threshold_n"][env_index].item())
    detach = float(flags["detach_threshold_n"][env_index].item())
    cv2.rectangle(output, (12, 12), (510, 134), (18, 20, 24), thickness=-1)
    cv2.putText(
        output,
        "DINOv3 FLOW MATCHING BC",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"{outcome} | t={physics_step / 120.0:4.2f}s | step={physics_step}",
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"grip  {grip:4.2f} / {damage:4.2f} N",
        (25, 97),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (100, 230, 120) if grip < damage else (250, 80, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"{FORCE_ELEMENT:<7}{tension:4.2f} / {detach:4.2f} N",
        (25, 123),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (100, 205, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def policy_state(env: GrapePickEnv) -> torch.Tensor:
    cafe = env.get_cafe_observation()["robot0_pos"]
    force_metrics = env.get_contact_force_metrics()
    pad_force = torch.stack(
        (
            force_metrics["left_normal_force_n"],
            force_metrics["right_normal_force_n"],
        ),
        dim=-1,
    )
    return torch.cat((cafe, pad_force), dim=-1)


def load_grape_policy(
    checkpoint_path: Path, device: torch.device, *, use_ema: bool
) -> tuple[flow.DINOv3FlowPolicy, common.LinearNormalizer, dict[str, Any]]:
    """Load on CPU first so Isaac RTX initialization does not stall CUDA deserialization."""

    started = time.time()
    checkpoint = common.load_checkpoint(checkpoint_path, map_location="cpu")
    print(f"[LOAD] checkpoint_cpu seconds={time.time() - started:.2f}", flush=True)
    if checkpoint.get("model_type") != flow.MODEL_TYPE:
        raise ValueError(
            f"Expected model_type {flow.MODEL_TYPE!r}, got {checkpoint.get('model_type')!r}"
        )
    policy_config = flow.DINOv3FlowConfig.from_dict(checkpoint["policy_config"])
    started = time.time()
    dino, _ = common.load_dino_for_config(
        Path(policy_config.dino_path) if policy_config.dino_path else None
    )
    print(f"[LOAD] dinov3_cpu seconds={time.time() - started:.2f}", flush=True)
    started = time.time()
    model = flow.DINOv3FlowPolicy(policy_config, dino)
    if use_ema and "ema" not in checkpoint:
        raise ValueError("--use-ema was requested but the checkpoint has no EMA weights")
    state = checkpoint["ema"]["averaged_model"] if use_ema else checkpoint["model"]
    flow.load_policy_state_dict(model, state)
    model.to(device).eval()
    print(f"[LOAD] policy_cuda seconds={time.time() - started:.2f}", flush=True)
    normalizer = common.LinearNormalizer()
    normalizer.load_state_dict(checkpoint["normalizer"])
    normalizer.stats["robot0_pos"] = normalizer.stats[common.STATE_KEY]
    return model, normalizer, checkpoint


def main() -> None:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    print(f"[INFO] Constructing {PICK_VARIANT.title()}Pick scene", flush=True)
    config = GrapePickEnvCfg()
    config.scene.num_envs = args.num_envs
    config.seed = args.seed
    config.enable_cameras = True
    config.enable_gelsight = False
    if IS_RASPBERRY:
        config.wrist_camera.width = 640
        config.wrist_camera.height = 480
        config.third_person_camera.width = 960
        config.third_person_camera.height = 540
    config.randomize_grape_pose = args.randomize_grape_pose
    if args.grape_position_range is not None:
        config.grape_position_randomization_m = tuple(args.grape_position_range)
    config.randomize_force_thresholds = args.randomize_force_thresholds
    config.sim.physx.enable_enhanced_determinism = bool(args.enhanced_determinism)
    config.damage_force_randomization_range_n = tuple(args.damage_range)
    config.detach_force_randomization_range_n = tuple(args.detach_range)
    if args.damage_threshold_n is not None:
        config.damage_grip_force_threshold_n = float(args.damage_threshold_n)
    if args.detach_threshold_n is not None:
        config.detach_force_threshold_n = float(args.detach_threshold_n)
    config.episode_length_s = max(args.max_episode_steps * config.sim.dt, 1.0)
    if args.device:
        config.sim.device = args.device
    env = GrapePickEnv(config, render_mode="rgb_array")
    device = torch.device(config.sim.device)
    print(f"[INFO] Scene ready; loading checkpoint on {device}: {checkpoint_path}", flush=True)
    model, normalizer, checkpoint = load_grape_policy(
        checkpoint_path, device, use_ema=args.use_ema
    )
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError(
            f"Expected checkpoint contract {CONTRACT!r}, got {checkpoint.get('contract')!r}"
        )
    if checkpoint.get("task") != TASK:
        raise ValueError(f"Checkpoint task mismatch: {checkpoint.get('task')!r}")
    if model.config.state_dim != 12 or model.config.num_cameras != 2:
        raise ValueError(
            f"{PICK_VARIANT.title()} DINOv3 policy must use 12-D robot/tactile "
            "state and two cameras"
        )
    if args.chunk_execute_steps > model.config.chunk_size:
        raise ValueError("--chunk-execute-steps exceeds checkpoint chunk size")

    two_stage_controller: TwoStageBerryForceController | None = None
    two_stage_config: TwoStageForceControllerConfig | None = None
    if args.two_stage_force_control:
        two_stage_config = TwoStageForceControllerConfig(
            contact_confirm_steps=args.contact_confirm_steps,
            grip_stable_steps=args.grip_stable_steps,
            tension_kp_velocity_m_s_per_n=args.tension_kp_velocity_m_s_per_n,
            maximum_pull_velocity_m_s=args.maximum_pull_velocity_m_s,
            maximum_pull_offset_m=args.maximum_pull_offset_m,
        )
        # Reset below replaces this placeholder with the current episode's
        # simulator-ground-truth target interval before any action is applied.
        two_stage_controller = TwoStageBerryForceController(
            ForceRange(1.0, 1.1), two_stage_config
        )

    print("[INFO] Policy ready; starting closed-loop evaluation", flush=True)
    position_command_offset = torch.tensor(
        args.position_command_offset_m, dtype=torch.float32, device=device
    )
    records: list[dict[str, Any]] = []
    batch = 0
    video_writer = None
    video_last_frame: np.ndarray | None = None
    video_path: Path | None = None
    video_finished = False
    if args.video_path is not None:
        import imageio.v2 as imageio

        video_path = args.video_path.expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(
            str(video_path),
            fps=args.video_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        print(f"[VIDEO] Recording {args.video_camera} camera to {video_path}", flush=True)
    try:
        while len(records) < args.episodes and simulation_app.is_running():
            batch_seed = (
                args.seed
                if args.fixed_episode_seed
                else args.seed + batch * args.num_envs
            )
            env.reset(seed=batch_seed)
            oracle_grip_range: ForceRange | None = None
            oracle_tension_target_n: torch.Tensor | None = None
            oracle_grip_safety_limit_n: torch.Tensor | None = None
            if two_stage_controller is not None:
                damage_threshold = env.damage_threshold_n.detach().clone()
                detach_threshold = env.detach_threshold_n.detach().clone()
                grip_high = float(damage_threshold[0].item()) - float(
                    args.oracle_grip_damage_margin_n
                )
                grip_low = grip_high - float(args.oracle_grip_range_width_n)
                if grip_low <= 0.0:
                    raise ValueError("oracle grip range is not positive")
                oracle_grip_range = ForceRange(grip_low, grip_high)
                two_stage_controller.set_grip_target_range(oracle_grip_range)
                two_stage_controller.reset()
                oracle_tension_target_n = detach_threshold + float(
                    args.oracle_tension_margin_n
                )
                oracle_grip_safety_limit_n = damage_threshold - 0.5 * float(
                    args.oracle_grip_damage_margin_n
                )
            active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
            previous_action: torch.Tensor | None = None
            state_history: list[torch.Tensor] = []
            image_history: list[torch.Tensor] = []
            locked_visual_xy: torch.Tensor | None = None
            generator = torch.Generator(device=device).manual_seed(batch_seed + 909_731)
            physics_step = 0
            batch_records: dict[int, dict[str, Any]] = {}
            force_stage_steps = torch.zeros(
                (args.num_envs, len(STAGE_NAMES)), dtype=torch.long, device=device
            )
            force_grip_active_steps = torch.zeros(
                args.num_envs, dtype=torch.long, device=device
            )
            force_safety_override_steps = torch.zeros(
                args.num_envs, dtype=torch.long, device=device
            )
            force_pull_command_steps = torch.zeros(
                args.num_envs, dtype=torch.long, device=device
            )
            maximum_controlled_pull_offset_m = torch.zeros(
                args.num_envs, dtype=torch.float32, device=device
            )
            maximum_controlled_pull_velocity_m_s = torch.zeros(
                args.num_envs, dtype=torch.float32, device=device
            )
            min_gripper_object_distance_m = torch.full(
                (args.num_envs,), float("inf"), device=device
            )
            commanded_position_min_m = torch.full(
                (args.num_envs, 3), float("inf"), device=device
            )
            commanded_position_max_m = torch.full(
                (args.num_envs, 3), -float("inf"), device=device
            )
            env.sim.render()
            state_now = policy_state(env)
            image_now = torch.stack(
                (camera_rgb_224(env, "wrist"), camera_rgb_224(env, "third")), dim=1
            )
            state_history = [
                state_now.clone() for _ in range(model.config.state_obs_steps)
            ]
            image_history = [
                image_now.clone() for _ in range(model.config.image_obs_steps)
            ]
            if video_writer is not None and batch == 0:
                initial_flags = env.get_task_flags()
                video_last_frame = annotated_video_frame(
                    env,
                    args.video_camera,
                    args.video_env_index,
                    physics_step,
                    initial_flags,
                )
                video_writer.append_data(video_last_frame)
            while physics_step < args.max_episode_steps and bool(active.any().item()):
                if physics_step % 120 == 0:
                    print(
                        f"[PROGRESS] batch={batch} physics_step={physics_step}/"
                        f"{args.max_episode_steps} active={int(active.sum().item())}",
                        flush=True,
                    )
                normalized_state = normalizer.normalize_tensor(
                    common.STATE_KEY, torch.stack(state_history, dim=1)
                )
                observation = {
                    common.STATE_KEY: normalized_state,
                    "images": torch.stack(image_history, dim=1),
                    "phase": torch.full(
                        (args.num_envs, 1),
                        min(physics_step / float(args.max_episode_steps), 1.0),
                        dtype=torch.float32,
                        device=device,
                    ),
                }
                result = model.predict_action(
                    observation,
                    generator=generator,
                    num_inference_steps=args.num_inference_steps,
                    initial_noise=(
                        torch.zeros(
                            (
                                args.num_envs,
                                model.config.chunk_size,
                                model.config.action_dim,
                            ),
                            dtype=normalized_state.dtype,
                            device=device,
                        )
                        if args.noise_mode == "zero"
                        else None
                    ),
                )
                normalized_chunk = result[common.ACTION_KEY]
                if args.visual_xy_override:
                    visual_xy = result["visual_xy"]
                    if (
                        args.visual_xy_lock_phase is not None
                        and locked_visual_xy is None
                        and physics_step
                        >= int(args.visual_xy_lock_phase * args.max_episode_steps)
                    ):
                        locked_visual_xy = visual_xy.clone()
                    selected_xy = visual_xy if locked_visual_xy is None else locked_visual_xy
                    normalized_chunk = normalized_chunk.clone()
                    normalized_chunk[:, :, :2] = selected_xy[:, None, :]
                chunk = normalizer.unnormalize_tensor(common.ACTION_KEY, normalized_chunk)
                chunk = chunk.clone()
                chunk[:, :, :3] += position_command_offset.view(1, 1, 3)
                pull_progress = min(
                    max(
                        (
                            physics_step / float(args.max_episode_steps)
                            - args.pull_z_start_phase
                        )
                        / (args.pull_z_ramp_end_phase - args.pull_z_start_phase),
                        0.0,
                    ),
                    1.0,
                )
                chunk[:, :, 2] += float(args.pull_z_offset_m) * pull_progress
                chunk[:, :, 9] += float(args.gripper_command_offset_m)
                if (
                    args.late_gripper_max_width_m is not None
                    and physics_step
                    >= int(args.late_gripper_start_phase * args.max_episode_steps)
                ):
                    chunk[:, :, 9].clamp_(max=args.late_gripper_max_width_m)
                for chunk_index in range(args.chunk_execute_steps):
                    target = chunk[:, chunk_index]
                    for _ in range(args.action_repeat):
                        if previous_action is None:
                            action = target
                        else:
                            alpha = float(args.action_smoothing)
                            action = alpha * target + (1.0 - alpha) * previous_action
                        previous_action = action
                        force_diagnostics = None
                        if two_stage_controller is not None:
                            assert oracle_tension_target_n is not None
                            assert oracle_grip_safety_limit_n is not None
                            force_metrics = env.get_contact_force_metrics()
                            flags_before = env.get_task_flags()
                            action, force_diagnostics = two_stage_controller.control(
                                action,
                                grip_force_n=force_metrics["grip_force_n"],
                                bilateral_contact=force_metrics["bilateral_contact"],
                                pedicle_tension_n=flags_before["stem_tension_n"],
                                stem_attached=flags_before["stem_attached"],
                                pull_direction=env.pull_direction_b,
                                target_tension_n=oracle_tension_target_n,
                                dt_s=float(env.physics_dt),
                                grip_safety_limit_n=oracle_grip_safety_limit_n,
                            )
                            force_stage_steps.scatter_add_(
                                1,
                                force_diagnostics.stage[:, None],
                                active.long()[:, None],
                            )
                            force_grip_active_steps += (
                                force_diagnostics.grip.active & active
                            ).long()
                            force_safety_override_steps += (
                                force_diagnostics.grip.safety_override & active
                            ).long()
                            force_pull_command_steps += (
                                (force_diagnostics.pull_velocity_m_s.abs() > 1.0e-9)
                                & active
                            ).long()
                            maximum_controlled_pull_offset_m = torch.maximum(
                                maximum_controlled_pull_offset_m,
                                force_diagnostics.pull_offset_m,
                            )
                            maximum_controlled_pull_velocity_m_s = torch.maximum(
                                maximum_controlled_pull_velocity_m_s,
                                force_diagnostics.pull_velocity_m_s.abs(),
                            )
                        commanded_position_min_m = torch.minimum(
                            commanded_position_min_m, action[:, :3]
                        )
                        commanded_position_max_m = torch.maximum(
                            commanded_position_max_m, action[:, :3]
                        )
                        env._pre_physics_step(action)
                        env._apply_action()
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(dt=env.physics_dt)
                        gripper_center_b = env._gripper_center_pos_b()
                        object_pos_b = (
                            env.grape.data.root_pos_w - env._robot.data.root_link_pos_w
                        )
                        center_distance = torch.linalg.vector_norm(
                            gripper_center_b - object_pos_b, dim=-1
                        )
                        min_gripper_object_distance_m = torch.minimum(
                            min_gripper_object_distance_m, center_distance
                        )
                        physics_step += 1
                        flags = env.get_task_flags()
                        terminal = (
                            flags["success"]
                            | flags["crushed"]
                            | flags["slipped"]
                            | flags["obstacle_collision"]
                            | flags["workspace_failure"]
                        )
                        if physics_step >= args.max_episode_steps:
                            terminal |= active
                        just_finished = active & terminal
                        for index in torch.nonzero(
                            just_finished, as_tuple=False
                        ).squeeze(-1).cpu().tolist():
                            batch_records[index] = {
                                "seed": batch_seed + index,
                                "outcome": classify(flags, index),
                                "steps": physics_step,
                                f"{OBJECT_NAME}_position_offset_m": (
                                    env.grape_reset_pos_w[index]
                                    - env.scene.env_origins[index]
                                    - torch.tensor(
                                        config.grape_initial_pos,
                                        dtype=env.grape_reset_pos_w.dtype,
                                        device=device,
                                    )
                                )
                                .detach()
                                .cpu()
                                .tolist(),
                                "peak_grip_force_n": float(
                                    env.peak_grip_force_n[index].item()
                                ),
                                "peak_stem_tension_n": float(
                                    env.peak_stem_tension_n[index].item()
                                ),
                                "peak_stem_rotation_error_deg": math.degrees(
                                    float(env.peak_stem_rotation_error_rad[index].item())
                                ),
                                "peak_stem_restoring_torque_nm": float(
                                    env.peak_stem_restoring_torque_nm[index].item()
                                ),
                                "final_stem_rotation_error_deg": math.degrees(
                                    float(env.stem_rotation_error_rad[index].item())
                                ),
                                "min_gripper_object_distance_m": float(
                                    min_gripper_object_distance_m[index].item()
                                ),
                                "commanded_position_min_m": commanded_position_min_m[
                                    index
                                ]
                                .detach()
                                .cpu()
                                .tolist(),
                                "commanded_position_max_m": commanded_position_max_m[
                                    index
                                ]
                                .detach()
                                .cpu()
                                .tolist(),
                                "final_gripper_center_pos_b_m": gripper_center_b[index]
                                .detach()
                                .cpu()
                                .tolist(),
                                "final_object_pos_b_m": object_pos_b[index]
                                .detach()
                                .cpu()
                                .tolist(),
                                "final_gripper_width_m": float(
                                    env.gripper_width[index, 0].item()
                                ),
                                "damage_threshold_n": float(
                                    env.damage_threshold_n[index].item()
                                ),
                                "detach_threshold_n": float(
                                    env.detach_threshold_n[index].item()
                                ),
                                "two_stage_force_control": bool(
                                    two_stage_controller is not None
                                ),
                                "oracle_grip_force_range_n": (
                                    None
                                    if oracle_grip_range is None
                                    else oracle_grip_range.as_list()
                                ),
                                "oracle_tension_target_n": (
                                    None
                                    if oracle_tension_target_n is None
                                    else float(oracle_tension_target_n[index].item())
                                ),
                                "force_stage_steps": {
                                    name: int(force_stage_steps[index, stage].item())
                                    for stage, name in enumerate(STAGE_NAMES)
                                },
                                "force_grip_active_steps": int(
                                    force_grip_active_steps[index].item()
                                ),
                                "force_safety_override_steps": int(
                                    force_safety_override_steps[index].item()
                                ),
                                "force_pull_command_steps": int(
                                    force_pull_command_steps[index].item()
                                ),
                                "maximum_controlled_pull_offset_m": float(
                                    maximum_controlled_pull_offset_m[index].item()
                                ),
                                "maximum_controlled_pull_velocity_m_s": float(
                                    maximum_controlled_pull_velocity_m_s[index].item()
                                ),
                            }
                        active &= ~just_finished
                        if video_writer is not None and batch == 0 and not video_finished:
                            video_terminal = bool(
                                just_finished[args.video_env_index].item()
                            )
                            if physics_step % args.video_stride == 0 or video_terminal:
                                env.sim.render()
                                video_last_frame = annotated_video_frame(
                                    env,
                                    args.video_camera,
                                    args.video_env_index,
                                    physics_step,
                                    flags,
                                    (
                                        classify(flags, args.video_env_index)
                                        if video_terminal
                                        else None
                                    ),
                                )
                                video_writer.append_data(video_last_frame)
                            video_finished = video_terminal
                        if physics_step >= args.max_episode_steps or not bool(active.any().item()):
                            break
                    if physics_step < args.max_episode_steps and bool(active.any().item()):
                        # The demonstrations were sampled every four physics steps.
                        # Refresh history at the same 30 Hz rate for every executed
                        # chunk action, even when several actions are open-loop.
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
                        state_history = state_history[-model.config.state_obs_steps :]
                        image_history = image_history[-model.config.image_obs_steps :]
                    if physics_step >= args.max_episode_steps or not bool(active.any().item()):
                        break
            if video_writer is not None and batch == 0 and video_last_frame is not None:
                for _ in range(max(1, int(round(args.video_fps)))):
                    video_writer.append_data(video_last_frame)
            for index in sorted(batch_records):
                if len(records) >= args.episodes:
                    break
                records.append(batch_records[index])
            successes = sum(record["outcome"] == "success" for record in records)
            print(
                f"[DINOv3 FLOW EVAL] episodes={len(records)}/{args.episodes} "
                f"success={successes / max(len(records), 1):.1%}",
                flush=True,
            )
            batch += 1
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()

    successes = sum(record["outcome"] == "success" for record in records)
    failures = Counter(
        record["outcome"] for record in records if record["outcome"] != "success"
    )
    report = {
        "task": TASK,
        "policy": "frozen_dinov3_tactile_conditional_flow_matching_bc",
        "control": (
            "oracle_two_stage_grip_and_pedicle_tension"
            if two_stage_controller is not None
            else "base_policy_only"
        ),
        "two_stage_force_controller_config": (
            None if two_stage_config is None else asdict(two_stage_config)
        ),
        "oracle_grip_damage_margin_n": (
            args.oracle_grip_damage_margin_n if two_stage_controller is not None else None
        ),
        "oracle_grip_range_width_n": (
            args.oracle_grip_range_width_n if two_stage_controller is not None else None
        ),
        "oracle_tension_margin_n": (
            args.oracle_tension_margin_n if two_stage_controller is not None else None
        ),
        "contract": CONTRACT,
        "model_type": checkpoint["model_type"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "dino_model": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "dino_frozen": True,
        "state_fields": ["robot0_pos[10]", "finger_normal_force_n[2]"],
        "camera_views": ["wrist", "third"],
        "seed_start": args.seed,
        "episode_seed_mode": "fixed" if args.fixed_episode_seed else "sequential",
        "enhanced_determinism": bool(args.enhanced_determinism),
        "num_envs": args.num_envs,
        "episodes": len(records),
        "max_episode_steps": args.max_episode_steps,
        "successes": successes,
        "success_rate": successes / max(len(records), 1),
        "wilson_95_ci": wilson(successes, len(records)),
        "failure_counts": dict(failures),
        "damage_force_randomization_range_n": list(args.damage_range),
        "detach_force_randomization_range_n": list(args.detach_range),
        "randomize_force_thresholds": args.randomize_force_thresholds,
        "fixed_damage_threshold_n": config.damage_grip_force_threshold_n,
        "fixed_detach_threshold_n": config.detach_force_threshold_n,
        "stem_angular_stiffness_nm_per_rad": config.stem_angular_stiffness_nm_per_rad,
        "stem_angular_damping_nm_s_per_rad": config.stem_angular_damping_nm_s_per_rad,
        "stem_max_restoring_torque_nm": config.stem_max_restoring_torque_nm,
        "stem_lock_rotation_while_attached": config.stem_lock_rotation_while_attached,
        f"randomize_{OBJECT_NAME}_pose": args.randomize_grape_pose,
        f"{OBJECT_NAME}_position_randomization_m": list(
            config.grape_position_randomization_m
        ),
        "action_repeat": args.action_repeat,
        "chunk_execute_steps": args.chunk_execute_steps,
        "action_smoothing": args.action_smoothing,
        "position_command_offset_m": list(args.position_command_offset_m),
        "gripper_command_offset_m": args.gripper_command_offset_m,
        "late_gripper_max_width_m": args.late_gripper_max_width_m,
        "late_gripper_start_phase": args.late_gripper_start_phase,
        "pull_z_offset_m": args.pull_z_offset_m,
        "pull_z_start_phase": args.pull_z_start_phase,
        "pull_z_ramp_end_phase": args.pull_z_ramp_end_phase,
        "num_inference_steps": (
            model.config.num_inference_steps
            if args.num_inference_steps is None
            else args.num_inference_steps
        ),
        "visual_xy_override": args.visual_xy_override,
        "visual_xy_lock_phase": args.visual_xy_lock_phase,
        "checkpoint_weights": "ema" if args.use_ema else "raw_best_epoch",
        "noise_mode": args.noise_mode,
        "native_flow_noise": (
            "zero latent"
            if args.noise_mode == "zero"
            else "independent standard Gaussian per action chunk"
        ),
        "video_path": None if video_path is None else str(video_path),
        "video_camera": None if video_path is None else args.video_camera,
        "video_env_index": None if video_path is None else args.video_env_index,
        "video_fps": None if video_path is None else args.video_fps,
        "outcomes": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "[SUMMARY] "
        + json.dumps({key: value for key, value in report.items() if key != "outcomes"}),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
