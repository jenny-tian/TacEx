#!/usr/bin/env python3
"""Run a paired 5-method, multi-threshold online experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def threshold_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bc-policy",
        type=Path,
        default=Path("outputs/lab_pick_dinov3_flow_bc200_yaw0/best.pt"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("exp_report/vlm_with_dsrl"),
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=(3.5, 4.0, 4.5))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("joint", "joint_bilateral", "dsrl", "vlm", "base"),
        default=("joint", "joint_bilateral", "dsrl", "vlm", "base"),
    )
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument(
        "--advisor", choices=("deterministic", "openai"), default="deterministic"
    )
    parser.add_argument(
        "--physical-force-range-n",
        type=float,
        nargs=2,
        default=(0.25, 3.25),
        metavar=("MIN", "MAX"),
        help="Allowed target-force range; MAX must remain below every break threshold.",
    )
    parser.add_argument(
        "--initial-force-range-n",
        type=float,
        nargs=2,
        default=(1.0, 3.0),
        metavar=("MIN", "MAX"),
        help="Initial target-force range, contained in --physical-force-range-n.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def valid_complete_result(path: Path, *, episodes: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(payload.get("completed_episodes", -1)) == episodes


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    output_root = (repo_root / args.output_root).resolve()
    checkpoint = (repo_root / args.bc_policy).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_root.mkdir(parents=True, exist_ok=True)
    console_dir = output_root / "console_logs"
    console_dir.mkdir(exist_ok=True)
    launcher = Path(__file__).with_name("run_experiment.py")
    runs: list[dict[str, object]] = []

    physical_min, physical_max = args.physical_force_range_n
    initial_min, initial_max = args.initial_force_range_n
    if not 0.0 <= physical_min < physical_max:
        raise ValueError(
            "--physical-force-range-n must be non-negative and increasing."
        )
    if not physical_min <= initial_min < initial_max <= physical_max:
        raise ValueError(
            "--initial-force-range-n must be increasing and contained in the physical range."
        )
    if any(method in {"joint", "joint_bilateral", "vlm"} for method in args.methods):
        unsafe = [
            threshold for threshold in args.thresholds if threshold <= physical_max
        ]
        if unsafe:
            raise ValueError(
                "Every force-controlled break threshold must exceed the physical force "
                f"maximum {physical_max:g} N; incompatible thresholds: {unsafe}."
            )

    for threshold in args.thresholds:
        if threshold <= 0.0:
            raise ValueError("All break-force thresholds must be positive.")
        for method in args.methods:
            run_name = (
                f"threshold_{threshold_tag(threshold)}_"
                f"{method}_seed{args.seed}_n{args.num_episodes}"
            )
            run_dir = output_root / "runs" / run_name
            result_path = run_dir / "results.json"
            command = [
                sys.executable,
                str(launcher),
                "--mode",
                method,
                "--bc-policy",
                str(checkpoint),
                "--output-dir",
                str(run_dir),
                "--num-episodes",
                str(args.num_episodes),
                "--seed",
                str(args.seed),
                "--break-force-threshold-n",
                str(threshold),
                "--advisor",
                args.advisor,
                "--physical-force-range-n",
                str(physical_min),
                str(physical_max),
                "--initial-force-range-n",
                str(initial_min),
                str(initial_max),
            ]
            if args.resume and valid_complete_result(
                result_path, episodes=args.num_episodes
            ):
                print(f"[SKIP] complete: {run_name}", flush=True)
                runs.append(
                    {"run": run_name, "status": "existing", "result": str(result_path)}
                )
                continue
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(
                    f"Incomplete non-empty run requires manual inspection: {run_dir}"
                )
            print("[RUN]", " ".join(command), flush=True)
            if args.dry_run:
                runs.append(
                    {"run": run_name, "status": "dry_run", "result": str(result_path)}
                )
                continue
            log_path = console_dir / f"{run_name}.log"
            env = os.environ.copy()
            env.setdefault(
                "TACEX_ISAAC_PYTHON",
                "/home/limx/anaconda3/envs/env_isaaclab/bin/python",
            )
            with log_path.open("x", encoding="utf-8", buffering=1) as log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    log_stream.write(line)
                    if line.startswith(("[EPISODE]", "[SUMMARY]")):
                        print(f"[{run_name}] {line}", end="", flush=True)
                return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"Experiment {run_name} failed with exit code {return_code}; "
                    f"see {log_path}."
                )
            if not valid_complete_result(result_path, episodes=args.num_episodes):
                raise RuntimeError(
                    f"Experiment {run_name} did not produce a complete result."
                )
            runs.append(
                {"run": run_name, "status": "completed", "result": str(result_path)}
            )

    manifest = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "thresholds_n": list(args.thresholds),
        "methods": list(args.methods),
        "num_episodes_per_condition": args.num_episodes,
        "seed": args.seed,
        "advisor": args.advisor,
        "runs": runs,
    }
    (output_root / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] matrix manifest: {output_root / 'matrix_manifest.json'}")


if __name__ == "__main__":
    main()
