#!/usr/bin/env python3
"""Run the staged, paired RaspberryPick VLM+DSRL retraining protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TRAINER = Path(__file__).with_name("train_online_codex_vlm_dsrl.py")
DEFAULT_OUTPUT = ROOT / "exp_report" / "raspberry_pick" / "online_vlm_dsrl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "paired30", "formal100"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profiles", nargs="+", choices=("fixed", "delicate", "firm"), default=("delicate", "firm"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(1_120_000, 1_130_000, 1_140_000))
    parser.add_argument("--calibration-centers", type=float, nargs="+", default=(2.10, 2.40, 2.70))
    parser.add_argument("--calibration-width", type=float, default=0.60)
    parser.add_argument("--calibration-episodes", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("dsrl_only", "vlm_noop", "vlm_force_v2", "vlm_semantic_v2"),
        default=("dsrl_only", "vlm_noop", "vlm_force_v2", "vlm_semantic_v2"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-trainer-arg", action="append", default=[])
    return parser.parse_args()


def completed(run_dir: Path, expected_episodes: int) -> bool:
    summary_path = run_dir / "training_summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return int(summary.get("completed_episodes", -1)) == expected_episodes


def launch(run_dir: Path, arguments: list[str], expected_episodes: int, *, dry_run: bool) -> None:
    command = [sys.executable, str(TRAINER), "--headless", "--output-root", str(run_dir), *arguments]
    if completed(run_dir, expected_episodes):
        print(f"[SKIP] complete: {run_dir}", flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"incomplete non-empty run directory requires inspection: {run_dir}")
    print("[RUN] " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if args.stage == "calibration":
        episodes = args.calibration_episodes if args.episodes is None else args.episodes
        half_width = 0.5 * args.calibration_width
        for profile in args.profiles:
            for seed in args.seeds:
                for center in args.calibration_centers:
                    low, high = center - half_width, center + half_width
                    label = f"center{center:.2f}_seed{seed}_n{episodes}".replace(".", "p")
                    run_dir = output_root / "calibration" / profile / label
                    launch(
                        run_dir,
                        [
                            "--method",
                            "dsrl_only",
                            "--freeze-dsrl",
                            "--berry-profile",
                            profile,
                            "--initial-force-range",
                            f"{low:.6f}",
                            f"{high:.6f}",
                            "--train-seed",
                            str(seed),
                            "--train-episodes",
                            str(episodes),
                            *args.extra_trainer_arg,
                        ],
                        episodes,
                        dry_run=args.dry_run,
                    )
        return

    episodes = (30 if args.stage == "paired30" else 100) if args.episodes is None else args.episodes
    stage_dir = "paired_n30" if args.stage == "paired30" else "paired_n100"
    for profile in args.profiles:
        for seed in args.seeds:
            for method in args.methods:
                run_dir = output_root / stage_dir / profile / method / f"seed{seed}_n{episodes}"
                launch(
                    run_dir,
                    [
                        "--method",
                        method,
                        "--berry-profile",
                        profile,
                        "--train-seed",
                        str(seed),
                        "--train-episodes",
                        str(episodes),
                        *args.extra_trainer_arg,
                    ],
                    episodes,
                    dry_run=args.dry_run,
                )


if __name__ == "__main__":
    main()
