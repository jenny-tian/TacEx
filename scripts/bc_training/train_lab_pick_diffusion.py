#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDIM LeRobot Diffusion Policy for TacEx LabPick.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/tacex_lab_pick_slide")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action-steps", type=int, default=8)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", choices=["torchcodec", "pyav"], default="pyav")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_virtualgl_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env.pop("VGL_ISACTIVE", None)
    env.pop("VGL_DISPLAY", None)
    return env


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset root; missing {info_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")

    executable = shutil.which("lerobot-train")
    if executable is None:
        raise RuntimeError("lerobot-train is not on PATH. Activate the tacex_lerobot environment.")

    command = [
        executable,
        "--policy.type=diffusion",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        "--policy.noise_scheduler_type=DDIM",
        f"--policy.num_inference_steps={args.inference_steps}",
        f"--policy.horizon={args.horizon}",
        f"--policy.n_action_steps={args.action_steps}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.video_backend={args.video_backend}",
        f"--output_dir={output_dir}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--save_freq={args.save_freq}",
        f"--policy.down_dims=[{','.join(str(value) for value in args.down_dims)}]",
        f"--seed={args.seed}",
        "--wandb.enable=false",
    ]
    print("[COMMAND]", " ".join(command))
    if args.dry_run:
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, env=clean_virtualgl_env())


if __name__ == "__main__":
    main()
