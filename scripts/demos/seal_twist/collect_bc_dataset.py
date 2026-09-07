"""Collect vectorized, randomized SealTwist expert demonstrations for BC."""

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
parser.add_argument("--num_episodes", type=int, default=48)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--output", type=Path, default=Path("exp_report/bottle/dataset"))
parser.add_argument("--record_stride", type=int, default=4)
parser.add_argument("--max_episode_steps", type=int, default=900)
parser.add_argument("--max_batches", type=int, default=30)
parser.add_argument("--include_failures", action="store_true")
parser.add_argument("--position_range", type=float, nargs=2, default=(0.008, 0.008))
parser.add_argument("--target_torque_range", type=float, nargs=2, default=(0.38, 0.46))
parser.add_argument("--grasp_width_range", type=float, nargs=2, default=(0.0205, 0.0225))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_episodes <= 0 or args.num_envs <= 0:
    parser.error("--num_episodes and --num_envs must be positive")
if args.record_stride <= 0:
    parser.error("--record_stride must be positive")
args.enable_cameras = False

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch

from tacex_tasks.seal_twist.dataset import SealTwistTrajectoryBuffer, validate_episode
from tacex_tasks.seal_twist.seal_twist_env import SealTwistEnv
from tacex_tasks.seal_twist.seal_twist_env_cfg import SealTwistEnvCfg


def _failure_reason(flags: dict[str, torch.Tensor], index: int) -> str:
    for name in ("over_force", "over_torqued", "alignment_failure", "under_force", "slipped"):
        if bool(flags[name][index].item()):
            if name == "over_torqued":
                return "over_force"
            return name
    return "timeout"


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = SealTwistEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.enable_cameras = False
    cfg.enable_gelsight = False
    cfg.randomize_labware_position = True
    cfg.labware_pos_randomization_xy = tuple(args.position_range)
    cfg.labware_yaw_randomization = 0.0
    cfg.randomize_expert_targets = True
    cfg.expert_target_torque_randomization_range_nm = tuple(args.target_torque_range)
    cfg.expert_grasp_width_randomization_range_m = tuple(args.grasp_width_range)
    cfg.randomize_force_thresholds = False
    cfg.minimum_finger_force_n = 0.05
    cfg.maximum_finger_force_n = 20.0
    cfg.success_hold_steps = 8
    cfg.episode_length_s = max(args.max_episode_steps * cfg.sim.dt, 1.0)
    if args.device:
        cfg.sim.device = args.device
    print(f"[INFO] Creating {args.num_envs} parallel SealTwist environments", flush=True)
    env = SealTwistEnv(cfg)

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
            buffer = SealTwistTrajectoryBuffer()
            active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
            for step in range(args.max_episode_steps):
                if step % 180 == 0:
                    print(
                        f"[PROGRESS] batch={batch} step={step}/{args.max_episode_steps}",
                        flush=True,
                    )
                env.command_expert()
                action = env.get_cafe_action()
                if step % args.record_stride == 0:
                    buffer.append(env, action, save_rgb=False, save_tactile=False)
                env._pre_physics_step(None)
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
                flags = env.get_task_flags()
                terminal = (
                    flags["success"]
                    | flags["over_force"]
                    | flags["over_torqued"]
                    | flags["under_force"]
                    | flags["alignment_failure"]
                    | flags["slipped"]
                )
                if step == args.max_episode_steps - 1:
                    terminal |= active
                just_finished = active & terminal
                for env_index in (
                    torch.nonzero(just_finished, as_tuple=False).squeeze(-1).cpu().tolist()
                ):
                    attempted += 1
                    success = bool(flags["success"][env_index].item())
                    reason = "" if success else _failure_reason(flags, env_index)
                    successes += int(success)
                    if not success:
                        failures[reason] += 1
                    if success or args.include_failures:
                        episode_path = args.output / f"record_{next_index + written:06d}"
                        buffer.save(
                            episode_path,
                            env,
                            seed=batch_seed + env_index,
                            success=success,
                            failure_reason=reason,
                            sample_period_s=env.physics_dt * args.record_stride,
                            env_index=env_index,
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
        "task": "TacEx-SealTwist-Torque-v0",
        "requested": args.num_episodes,
        "written": written,
        "attempted": attempted,
        "successes": successes,
        "online_success_rate": successes / max(attempted, 1),
        "failure_counts": dict(failures),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "position_randomization_half_range_m": list(args.position_range),
        "expert_target_torque_range_nm": list(args.target_torque_range),
        "expert_grasp_width_range_m": list(args.grasp_width_range),
        "contact_force_source": "PhysX ContactSensor filtered to cap",
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
