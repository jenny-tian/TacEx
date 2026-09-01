"""Closed-loop DINOv3-Flow BC + VLM force-control evaluation in Isaac Lab."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
    REPO_ROOT / "scripts" / "bc_training",
    REPO_ROOT / "source" / "tacex",
    REPO_ROOT / "source" / "tacex_assets",
    REPO_ROOT / "source" / "tacex_tasks",
    REPO_ROOT / "source" / "tacex_tasks" / "tacex_tasks" / "lab_pick",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _reexec_with_torch_cuda_library_path() -> None:
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / python_dir / "site-packages"
    required_dirs = [
        site_packages / "torch" / "lib",
        site_packages / "nvidia" / "cudnn" / "lib",
    ]
    current_paths = [
        value
        for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if value
    ]
    missing = [
        str(path)
        for path in required_dirs
        if path.is_dir() and str(path) not in current_paths
    ]
    if not missing:
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join([*missing, *current_paths])
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


_reexec_with_torch_cuda_library_path()


def _bootstrap_isaaclab_source_paths() -> None:
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        return
    source_root = Path(spec.origin).resolve().parent / "source"
    for package_name in (
        "isaaclab",
        "isaaclab_assets",
        "isaaclab_tasks",
        "isaaclab_rl",
        "isaaclab_mimic",
    ):
        package_source = source_root / package_name
        if (package_source / package_name).is_dir() and str(
            package_source
        ) not in sys.path:
            sys.path.insert(0, str(package_source))


def _bootstrap_isaacsim_warp_path() -> None:
    spec = importlib.util.find_spec("isaacsim")
    if spec is None or spec.origin is None:
        return
    extcache_dir = Path(spec.origin).resolve().parent / "extscache"
    for path in sorted(extcache_dir.glob("omni.warp.core-*"), reverse=True):
        if (path / "warp" / "__init__.py").is_file():
            sys.path.insert(0, str(path))
            return


_bootstrap_isaaclab_source_paths()
_bootstrap_isaacsim_warp_path()

from isaaclab.app import AppLauncher  # noqa: E402


def _patch_isaaclab_missing_exports() -> None:
    import isaaclab.utils as isaaclab_utils
    from isaaclab.utils.buffers.circular_buffer import CircularBuffer
    from isaaclab.utils.buffers.delay_buffer import DelayBuffer
    from isaaclab.utils.buffers.timestamped_buffer import TimestampedBuffer

    isaaclab_utils.CircularBuffer = CircularBuffer
    isaaclab_utils.DelayBuffer = DelayBuffer
    isaaclab_utils.TimestampedBuffer = TimestampedBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate episode-level VLM target-force adaptation on LabPick"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-steps", type=int, default=960)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--chunk-execute-steps", type=int, default=32)
    parser.add_argument("--camera-warmup-steps", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument(
        "--labware-random-xy", type=float, nargs=2, default=(0.10, 0.10)
    )
    parser.add_argument("--labware-random-yaw-degrees", type=float, default=0.0)
    parser.add_argument("--break-force-threshold-n", type=float, default=3.5)
    parser.add_argument(
        "--physical-force-range-n", type=float, nargs=2, default=(0.25, 3.25)
    )
    parser.add_argument(
        "--initial-force-range-n", type=float, nargs=2, default=(1.0, 3.0)
    )
    parser.add_argument("--minimum-range-width-n", type=float, default=0.30)
    parser.add_argument(
        "--advisor", choices=("deterministic", "openai"), default="deterministic"
    )
    parser.add_argument(
        "--vlm-model", default=os.environ.get("LAB_PICK_VLM_MODEL", "gpt-4.1-mini")
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--api-mode", choices=("responses", "chat_completions"), default="responses"
    )
    parser.add_argument(
        "--force-control", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save-advisor-images", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--kp", type=float, default=0.006)
    parser.add_argument("--ki", type=float, default=0.018)
    parser.add_argument("--kd", type=float, default=0.00008)
    parser.add_argument("--maximum-width-rate-m-s", type=float, default=0.018)
    parser.add_argument("--print-state-interval", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_trials < 1 or args.action_repeat < 1 or args.chunk_execute_steps < 1:
        parser.error("Trial and step counts must be positive.")
    if args.advisor == "openai" and not os.environ.get(args.api_key_env):
        parser.error(f"Set {args.api_key_env} for --advisor openai.")
    args.enable_cameras = True
    return args


args_cli = parse_args()
_patch_isaaclab_missing_exports()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import dinov3_flow as flow  # noqa: E402
from tacex_tasks.lab_pick.lab_pick_env import LabPickEnv  # noqa: E402
from tacex_tasks.lab_pick.lab_pick_env_cfg import LabPickEnvCfg  # noqa: E402
from vlm_force import (  # noqa: E402
    ConvergentForceEstimator,
    DeterministicVLMAdvisor,
    EpisodeFeedback,
    EpisodeForceAdaptationLoop,
    ForceControllerConfig,
    ForceEstimatorConfig,
    ForceRange,
    OpenAICompatibleVLMAdvisor,
    TactileForceController,
    diagnose_episode_failure,
)


def scalar(value: torch.Tensor) -> float:
    return float(value.reshape(-1)[0].item())


def flag(value: torch.Tensor) -> bool:
    return bool(value.reshape(-1)[0].item())


def camera_rgb_224(camera_rgb: torch.Tensor) -> np.ndarray:
    rgb = camera_rgb[0, :, :, :3].permute(2, 0, 1).unsqueeze(0).float()
    rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
    return rgb[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()


def policy_observation(env: LabPickEnv) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    state = (
        env.get_cafe_observation()["robot0_pos"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    wrist = camera_rgb_224(env.wrist_camera.data.output["rgb"])
    third = camera_rgb_224(env.third_person_camera.data.output["rgb"])
    return state, {
        "rgb": wrist,
        "rgb_third": third,
        "robot0_image": wrist,
        "robot0_image_third": third,
    }


def physics_step(env: LabPickEnv, action: torch.Tensor, *, render: bool = True) -> None:
    env._pre_physics_step(action)
    env._apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    if render:
        env.sim.render()


def terminal_reason(
    flags: dict[str, torch.Tensor], *, success: bool, broken: bool
) -> str:
    if broken:
        return "object_broken"
    if success:
        return "success"
    for key in ("object_dropped", "object_too_far", "ee_outside_workspace"):
        if flag(flags[key]):
            return key
    return ""


def save_advisor_images(
    env: LabPickEnv, directory: Path, episode_index: int
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in ("third", "wrist", "tactile_left", "tactile_right"):
        path = directory / f"episode_{episode_index:03d}_{name}.png"
        Image.fromarray(env.get_video_frame(name)[:, :, :3]).save(path)
        paths.append(path)
    return paths


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_force_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    checkpoint = args_cli.checkpoint.expanduser().resolve()
    output_dir = args_cli.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if (output_dir / "results.json").exists() or (
        output_dir / "vlm_episode_interactions.jsonl"
    ).exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing experiment in {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    physical_range = ForceRange(*args_cli.physical_force_range_n)
    initial_range = ForceRange(*args_cli.initial_force_range_n)
    if physical_range.maximum_n >= args_cli.break_force_threshold_n:
        raise ValueError(
            "physical-force-range maximum must be below the break threshold."
        )
    estimator = ConvergentForceEstimator(
        ForceEstimatorConfig(
            physical_range_n=physical_range,
            initial_range_n=initial_range,
            minimum_range_width_n=args_cli.minimum_range_width_n,
        )
    )
    if args_cli.advisor == "openai":
        advisor = OpenAICompatibleVLMAdvisor(
            model=args_cli.vlm_model,
            api_key=os.environ[args_cli.api_key_env],
            physical_range_n=physical_range,
            break_force_threshold_n=args_cli.break_force_threshold_n,
            api_base=args_cli.api_base,
            api_mode=args_cli.api_mode,
        )
    else:
        advisor = DeterministicVLMAdvisor(
            physical_range_n=physical_range,
            break_force_threshold_n=args_cli.break_force_threshold_n,
        )
    adaptation = EpisodeForceAdaptationLoop(
        advisor=advisor,
        estimator=estimator,
        log_path=output_dir / "vlm_episode_interactions.jsonl",
    )
    controller = TactileForceController(
        adaptation.current_range_n,
        ForceControllerConfig(
            kp_width_rate_per_n=args_cli.kp,
            ki_width_rate_per_n_s=args_cli.ki,
            kd_width_per_n=args_cli.kd,
            maximum_width_rate_m_s=args_cli.maximum_width_rate_m_s,
            hard_force_limit_n=args_cli.break_force_threshold_n,
        ),
    )

    cfg = LabPickEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args_cli.seed
    cfg.labware_name = "slide"
    cfg.randomize_labware_position = True
    cfg.labware_pos_randomization_xy = tuple(args_cli.labware_random_xy)
    cfg.labware_yaw_randomization = math.radians(args_cli.labware_random_yaw_degrees)
    cfg.terminate_break_force_threshold_n = args_cli.break_force_threshold_n
    cfg.rl_normalized_actions = False
    cfg.rl_align_cafe_action_yaw = False
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    env = LabPickEnv(cfg, render_mode="rgb_array")
    header = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if header.get("model_type") != flow.MODEL_TYPE:
        raise ValueError(
            f"This evaluator requires {flow.MODEL_TYPE}, got {header.get('model_type')!r}."
        )
    del header
    runner = flow.DINOv3FlowRunner(
        checkpoint,
        device=args_cli.device or "cuda",
        num_inference_steps=args_cli.num_inference_steps,
        visual_xy_lock_phase=0.30,
        use_visual_xy_override=True,
        seed=args_cli.seed,
    )
    execute_steps = min(args_cli.chunk_execute_steps, runner.config.chunk_size)
    results: list[dict[str, Any]] = []
    print(
        "[INFO] VLM-force evaluation "
        f"advisor={args_cli.advisor} force_control={args_cli.force_control} "
        f"trials={args_cli.num_trials} seed={args_cli.seed} initial_range={initial_range.as_list()}",
        flush=True,
    )

    try:
        for trial in range(args_cli.num_trials):
            trial_seed = args_cli.seed + trial
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(trial_seed)
            env.reset(seed=trial_seed)
            hold_action = env.get_cafe_observation()["robot0_pos"].clone()
            for _ in range(args_cli.camera_warmup_steps):
                physics_step(env, hold_action)
            env.step_count.zero_()
            env.has_touched.zero_()
            runner.reset()
            if runner.generator is not None:
                runner.generator.manual_seed(trial_seed)
            state, images = policy_observation(env)
            runner.update(state, images, phase=0.0)
            attempted_range = adaptation.current_range_n
            controller.reset()
            controller.set_target_range(attempted_range)

            physics_steps = 0
            policy_steps = 0
            success = False
            broken = False
            touched = False
            raw_reason = ""
            max_lift_m = 0.0
            peak_break_force_n = 0.0
            contact_force_sum = 0.0
            contact_force_samples = 0
            tactile_contact_samples = 0
            bilateral_contact_samples = 0
            squared_force_error_sum = 0.0
            controlled_steps = 0
            max_uncontrolled_action_delta = 0.0
            force_trace: list[dict[str, Any]] = []

            while (
                simulation_app.is_running()
                and physics_steps < args_cli.episode_steps
                and not raw_reason
            ):
                action_chunk = runner.predict_action_chunk()
                for action_np in action_chunk[:execute_steps]:
                    policy_action = torch.as_tensor(
                        action_np, device=env.device, dtype=torch.float32
                    ).view(1, -1)
                    for repeat_index in range(args_cli.action_repeat):
                        force_metrics = env.get_contact_force_metrics()
                        left_touch, right_touch = env.tactile_contact_depths()
                        tactile_contact = (
                            left_touch > env.cfg.tactile_threshold_mm
                        ) | (right_touch > env.cfg.tactile_threshold_mm)
                        grip_force = force_metrics["grip_force_n"]
                        break_force = force_metrics["max_finger_force_n"]
                        if args_cli.force_control:
                            executed_action, diagnostics = controller.control(
                                policy_action,
                                contact_force_n=grip_force,
                                safety_force_n=break_force,
                                contact_mask=tactile_contact
                                & (grip_force >= controller.config.contact_on_force_n),
                                dt_s=float(env.physics_dt),
                            )
                            controlled = bool(diagnostics.active[0].item())
                            filtered_force = scalar(diagnostics.filtered_force_n)
                            force_error = scalar(diagnostics.force_error_n)
                        else:
                            executed_action = policy_action.clone()
                            controlled = False
                            filtered_force = scalar(grip_force)
                            force_error = controller.target_force_n - filtered_force
                        uncontrolled_delta = (
                            torch.cat(
                                (executed_action[:, :9] - policy_action[:, :9],), dim=-1
                            )
                            .abs()
                            .max()
                        )
                        max_uncontrolled_action_delta = max(
                            max_uncontrolled_action_delta, scalar(uncontrolled_delta)
                        )
                        physics_step(
                            env,
                            executed_action,
                            render=repeat_index == args_cli.action_repeat - 1,
                        )
                        physics_steps += 1

                        flags = env._get_termination_flags()
                        lift_m = scalar(
                            env.labware.data.root_pos_w[:, 2]
                            - env.initial_object_height
                        )
                        left_touch_after, right_touch_after = (
                            env.tactile_contact_depths()
                        )
                        any_contact = (
                            scalar(left_touch_after) > env.cfg.tactile_threshold_mm
                            or scalar(right_touch_after) > env.cfg.tactile_threshold_mm
                        )
                        both_contact = (
                            scalar(left_touch_after) > env.cfg.tactile_threshold_mm
                            and scalar(right_touch_after) > env.cfg.tactile_threshold_mm
                        )
                        touched = touched or any_contact
                        current_grip_force = scalar(flags["grip_force_n"])
                        current_break_force = scalar(flags["break_force_n"])
                        tactile_contact_samples += int(any_contact)
                        bilateral_contact_samples += int(both_contact)
                        loaded_contact = (
                            current_grip_force >= controller.config.contact_on_force_n
                        )
                        if loaded_contact:
                            contact_force_sum += current_grip_force
                            contact_force_samples += 1
                            squared_force_error_sum += (
                                current_grip_force - controller.target_force_n
                            ) ** 2
                        controlled_steps += int(controlled)
                        max_lift_m = max(max_lift_m, lift_m)
                        peak_break_force_n = max(
                            peak_break_force_n, current_break_force
                        )
                        broken = broken or flag(flags["object_broken"])
                        success = (success or flag(flags["success"])) and not broken
                        force_trace.append(
                            {
                                "physics_step": physics_steps,
                                "policy_width_m": scalar(policy_action[:, 9]),
                                "executed_width_m": scalar(executed_action[:, 9]),
                                "grip_force_n": current_grip_force,
                                "max_finger_force_n": current_break_force,
                                "filtered_force_n": filtered_force,
                                "target_force_n": controller.target_force_n,
                                "force_error_n": force_error,
                                "contact": int(any_contact),
                                "bilateral_contact": int(both_contact),
                                "loaded_contact": int(loaded_contact),
                                "controller_active": int(controlled),
                            }
                        )
                        if (
                            args_cli.print_state_interval
                            and physics_steps % args_cli.print_state_interval == 0
                        ):
                            env.print_state()
                        raw_reason = terminal_reason(
                            flags, success=success, broken=broken
                        )
                        if raw_reason or physics_steps >= args_cli.episode_steps:
                            break
                    policy_steps += 1
                    phase = min(
                        policy_steps / float(runner.config.phase_horizon_steps), 1.0
                    )
                    state, images = policy_observation(env)
                    runner.update(state, images, phase=phase)
                    if raw_reason or physics_steps >= args_cli.episode_steps:
                        break
            if not raw_reason:
                raw_reason = "time_limit"

            mean_force_n = contact_force_sum / max(contact_force_samples, 1)
            force_rmse_n = math.sqrt(
                squared_force_error_sum / max(contact_force_samples, 1)
            )
            contact_fraction = contact_force_samples / max(physics_steps, 1)
            tactile_contact_fraction = tactile_contact_samples / max(physics_steps, 1)
            bilateral_contact_fraction = bilateral_contact_samples / max(
                physics_steps, 1
            )
            diagnosed_reason = diagnose_episode_failure(
                raw_reason,
                touched=touched,
                contact_fraction=contact_fraction,
                bilateral_contact_fraction=bilateral_contact_fraction,
                mean_force_n=mean_force_n,
                attempted_range_n=attempted_range,
            )
            episode_feedback = EpisodeFeedback(
                episode_index=trial,
                success=success,
                failure_reason=diagnosed_reason,
                attempted_range_n=attempted_range,
                target_force_n=attempted_range.center_n,
                mean_contact_force_n=mean_force_n,
                peak_contact_force_n=peak_break_force_n,
                force_rmse_n=force_rmse_n,
                contact_fraction=contact_fraction,
                max_lift_m=max_lift_m,
                metadata={
                    "terminal_reason": raw_reason,
                    "seed": trial_seed,
                    "tactile_contact_fraction": tactile_contact_fraction,
                    "bilateral_contact_fraction": bilateral_contact_fraction,
                },
            )
            need_images = args_cli.save_advisor_images or args_cli.advisor == "openai"
            image_paths = (
                save_advisor_images(env, output_dir / "advisor_images", trial)
                if need_images
                else []
            )
            decision = adaptation.complete_episode(
                episode_feedback, image_paths=image_paths
            )
            write_force_trace(
                output_dir / "force_traces" / f"episode_{trial:03d}.csv", force_trace
            )

            result = {
                "episode_index": trial,
                "seed": trial_seed,
                "success": success,
                "terminal_reason": raw_reason,
                "diagnosed_failure_reason": diagnosed_reason,
                "touched": touched,
                "physics_steps": physics_steps,
                "policy_steps": policy_steps,
                "max_lift_m": max_lift_m,
                "peak_contact_force_n": peak_break_force_n,
                "mean_contact_force_n": mean_force_n,
                "force_rmse_n": force_rmse_n,
                "contact_fraction": contact_fraction,
                "tactile_contact_fraction": tactile_contact_fraction,
                "bilateral_contact_fraction": bilateral_contact_fraction,
                "controller_active_fraction": controlled_steps / max(physics_steps, 1),
                "attempted_force_range_n": attempted_range.as_list(),
                "attempted_target_force_n": attempted_range.center_n,
                "next_force_range_n": decision.target_range_n.as_list(),
                "next_target_force_n": decision.target_range_n.center_n,
                "update_kind": decision.update_kind,
                "max_uncontrolled_action_delta": max_uncontrolled_action_delta,
                "force_trace": str(
                    output_dir / "force_traces" / f"episode_{trial:03d}.csv"
                ),
            }
            results.append(result)
            write_json(
                output_dir / "results.partial.json",
                {"completed_trials": len(results), "results": results},
            )
            print(
                "[RESULT] "
                f"episode={trial} seed={trial_seed} success={success} reason={raw_reason}/{diagnosed_reason} "
                f"force_mean={mean_force_n:.3f}N force_peak={peak_break_force_n:.3f}N "
                f"range={attempted_range.as_list()} -> {decision.target_range_n.as_list()}",
                flush=True,
            )
    finally:
        env.close()

    successes = sum(int(item["success"]) for item in results)
    broken_count = sum(
        int(item["terminal_reason"] == "object_broken") for item in results
    )
    summary = {
        "schema_version": 1,
        "experiment": (
            "DINOv3 Flow BC with episode-level VLM force adaptation and 120 Hz tactile force control"
            if args_cli.force_control
            else "DINOv3 Flow BC policy-only control (force-controller ablation)"
        ),
        "advisor": args_cli.advisor,
        "advisor_is_real_vlm": args_cli.advisor == "openai",
        "force_control": args_cli.force_control,
        "checkpoint": str(checkpoint),
        "num_trials": len(results),
        "successes": successes,
        "success_rate": successes / max(len(results), 1),
        "broken": broken_count,
        "broken_rate": broken_count / max(len(results), 1),
        "failure_counts": dict(
            Counter(item["terminal_reason"] for item in results if not item["success"])
        ),
        "seed": args_cli.seed,
        "episode_steps": args_cli.episode_steps,
        "action_repeat": args_cli.action_repeat,
        "chunk_execute_steps": execute_steps,
        "camera_warmup_steps": args_cli.camera_warmup_steps,
        "num_inference_steps": args_cli.num_inference_steps,
        "labware_random_xy_m": list(args_cli.labware_random_xy),
        "labware_random_yaw_degrees": args_cli.labware_random_yaw_degrees,
        "physics_rate_hz": round(1.0 / float(env.physics_dt)),
        "policy_rate_hz": round(1.0 / (float(env.physics_dt) * args_cli.action_repeat)),
        "break_force_threshold_n": args_cli.break_force_threshold_n,
        "initial_force_range_n": initial_range.as_list(),
        "controller_gains": {
            "kp_width_rate_per_n": args_cli.kp,
            "ki_width_rate_per_n_s": args_cli.ki,
            "kd_width_per_n": args_cli.kd,
            "maximum_width_rate_m_s": args_cli.maximum_width_rate_m_s,
        },
        "final_force_range_n": adaptation.current_range_n.as_list(),
        "advisor_calls": int(getattr(advisor, "call_count", len(results))),
        "maximum_uncontrolled_action_delta": max(
            item["max_uncontrolled_action_delta"] for item in results
        ),
        "estimator": estimator.state_dict(),
        "results": results,
    }
    write_json(output_dir / "results.json", summary)
    print(
        f"[SUMMARY] success={successes}/{len(results)} ({summary['success_rate']:.1%}) "
        f"broken={broken_count}/{len(results)} final_range={adaptation.current_range_n.as_list()} "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
