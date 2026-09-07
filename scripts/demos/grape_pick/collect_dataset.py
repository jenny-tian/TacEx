"""Collect vectorized force-safe grape-picking demonstrations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path


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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, default=Path("datasets/grape_pick"))
parser.add_argument("--record_stride", type=int, default=2, help="Save one sample every N physics steps")
parser.add_argument("--max_episode_steps", type=int, default=1100)
parser.add_argument("--max_batches", type=int, default=100)
parser.add_argument("--include_failures", action="store_true")
parser.add_argument("--save_rgb", action="store_true")
parser.add_argument("--save_gelsight", action="store_true")
parser.add_argument("--randomize_force_thresholds", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_episodes <= 0 or args.num_envs <= 0:
    parser.error("--num_episodes and --num_envs must be positive")
if args.record_stride <= 0:
    parser.error("--record_stride must be positive")
need_cameras = bool(args.save_rgb or args.save_gelsight)
args.enable_cameras = need_cameras

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
import torch.nn.functional as F

from tacex_tasks.grape_pick.dataset import GrapeTrajectoryBuffer, validate_episode
from tacex_tasks.grape_pick.grape_pick_env import GrapePickEnv
from tacex_tasks.grape_pick.grape_pick_env_cfg import GrapePickEnvCfg


def _rgb_224(image: torch.Tensor) -> torch.Tensor:
    image_chw = image[..., :3].permute(0, 3, 1, 2).float()
    resized = F.interpolate(image_chw, size=(224, 224), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1).clamp(0, 255).byte()


def _failure_reason(flags: dict[str, torch.Tensor], index: int) -> str:
    for name in ("crushed", "slipped", "obstacle_collision", "workspace_failure", "no_detach"):
        if bool(flags[name][index].item()):
            return name
    return "timeout"


def main() -> None:
    print("[INFO] Initializing GrapePick dataset collector", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = GrapePickEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.enable_cameras = need_cameras
    cfg.enable_gelsight = args.save_gelsight
    cfg.randomize_force_thresholds = args.randomize_force_thresholds
    cfg.episode_length_s = max(args.max_episode_steps * cfg.sim.dt, 1.0)
    if args.device:
        cfg.sim.device = args.device
    print("[INFO] Configuration ready; creating simulation scene", flush=True)
    env = GrapePickEnv(cfg, render_mode="rgb_array" if cfg.enable_cameras else None)
    print(f"[INFO] Environment ready with {args.num_envs} parallel instances", flush=True)

    existing = [
        int(path.name.removeprefix("record_"))
        for path in args.output.glob("record_*")
        if path.is_dir() and path.name.removeprefix("record_").isdigit()
    ]
    next_index = max(existing, default=-1) + 1
    written = 0
    attempted = 0
    successes = 0
    failures: Counter[str] = Counter()

    try:
        for batch in range(args.max_batches):
            if written >= args.num_episodes or not simulation_app.is_running():
                break
            batch_seed = args.seed + batch * args.num_envs
            env.reset(seed=batch_seed)
            buffer = GrapeTrajectoryBuffer(args.num_envs)
            active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)

            for step in range(args.max_episode_steps):
                if step % 120 == 0:
                    print(f"[PROGRESS] batch={batch} step={step}/{args.max_episode_steps}", flush=True)
                env.command_expert()
                action = env.get_cafe_action()
                if step % args.record_stride == 0:
                    rgb = rgb_third = tactile = None
                    if cfg.enable_cameras:
                        env.sim.render()
                    if args.save_rgb:
                        rgb = _rgb_224(env.get_camera_rgb("wrist"))
                        rgb_third = _rgb_224(env.get_camera_rgb("third"))
                    if args.save_gelsight:
                        tactile = (
                            env.get_tactile_rgb("left"),
                            env.get_tactile_rgb("right"),
                            env.get_tactile_height_map("left"),
                            env.get_tactile_height_map("right"),
                        )
                    buffer.append(
                        env,
                        action,
                        active,
                        rgb=rgb,
                        rgb_third=rgb_third,
                        tactile=tactile,
                    )

                env._pre_physics_step(None)
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
                flags = env.get_task_flags()
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
                for env_index in torch.nonzero(just_finished, as_tuple=False).squeeze(-1).cpu().tolist():
                    attempted += 1
                    success = bool(flags["success"][env_index].item())
                    reason = "" if success else _failure_reason(flags, env_index)
                    successes += int(success)
                    if not success:
                        failures[reason] += 1
                    if success or args.include_failures:
                        episode_path = args.output / f"record_{next_index + written:06d}"
                        buffer.save_episode(
                            env_index,
                            episode_path,
                            success=success,
                            failure_reason=reason,
                            env=env,
                            seed=batch_seed + env_index,
                            sample_period_s=env.physics_dt * args.record_stride,
                        )
                        validate_episode(episode_path)
                        written += 1
                        print(
                            f"[EPISODE] written={written}/{args.num_episodes} success={success} "
                            f"reason={reason or 'none'} seed={batch_seed + env_index}",
                            flush=True,
                        )
                        if written >= args.num_episodes:
                            break
                active &= ~just_finished
                if written >= args.num_episodes or not bool(active.any().item()):
                    break
    finally:
        env.close()

    summary = {
        "task": "TacEx-GrapePick-Force-v0",
        "requested": args.num_episodes,
        "written": written,
        "attempted": attempted,
        "successes": successes,
        "online_success_rate": successes / max(attempted, 1),
        "failure_counts": dict(failures),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "save_rgb": args.save_rgb,
        "save_gelsight": args.save_gelsight,
        "contact_force_source": "PhysX ContactSensor filtered to target_grape",
    }
    (args.output / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("[SUMMARY] " + json.dumps(summary), flush=True)
    if written < args.num_episodes:
        raise RuntimeError(f"Collected only {written}/{args.num_episodes} requested episodes")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
