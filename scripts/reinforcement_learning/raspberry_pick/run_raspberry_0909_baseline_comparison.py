#!/usr/bin/env python3
"""Run the matched RaspberryPick Residual RL, AWR, and DPPO comparison."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
TRAINER = ROOT / "scripts/reinforcement_learning/raspberry_pick/train_online_codex_vlm_dsrl.py"
ANALYZER = ROOT / "scripts/reinforcement_learning/raspberry_pick/analyze_raspberry_0909_baselines.py"
DEFAULT_PYTHON = Path("/home/limx/anaconda3/envs/env_isaaclab/bin/python")
REPORT_ROOT = ROOT / "exp_report/raspberry_0909/scenario_revision_zero_pull"
BASELINE_ROOT = REPORT_ROOT / "baseline_comparison_n100"
METHODS = ("residual_rl", "awr", "dppo")
FORMAL_SEEDS = (970_000, 980_000, 990_000)
SMOKE_SEEDS = {
    "residual_rl": 969_810,
    "awr": 969_820,
    "dppo": 969_830,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assert_finite(value: Any, location: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{location}[{index}]")


def command_for(
    python: Path,
    *,
    method: str,
    output_root: Path,
    seed: int,
    episodes: int,
) -> list[str]:
    if method not in METHODS:
        raise ValueError(method)
    command = [
        str(python),
        str(TRAINER),
        "--output-root",
        str(output_root),
        "--method",
        "dsrl_only",
        "--online-algorithm",
        method,
        "--force-range-source",
        "fixed",
        "--controller-mode",
        "single_stage",
        "--bottle-style-lateral-dsrl",
        "--initial-force-range",
        "2.2",
        "2.8",
        "--train-seed",
        str(seed),
        "--train-episodes",
        str(episodes),
        "--exploration-anneal-episodes",
        "30",
        "--checkpoint-every",
        "10" if episodes == 100 else "1",
        "--pull-z-offset-m",
        "0.0",
        "--disable-vlm",
        "--enhanced-determinism",
        "--headless",
    ]
    if episodes < 100:
        command.extend(("--learning-starts", "64", "--batch-size", "64"))
    return command


def is_complete(output_root: Path, method: str, episodes: int) -> bool:
    summary_path = output_root / "training_summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return (
        int(summary.get("completed_episodes", -1)) == episodes
        and summary.get("online_algorithm") == method
        and (output_root / "checkpoints/final.pt").is_file()
        and (output_root / "file_hashes.json").is_file()
    )


def execute(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("\n[COMMAND] " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"run failed with exit code {result.returncode}; see {log_path}")


def run_one(
    python: Path,
    *,
    method: str,
    output_root: Path,
    seed: int,
    episodes: int,
) -> None:
    if is_complete(output_root, method, episodes):
        print(f"[SKIP] complete: {output_root}", flush=True)
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            f"incomplete baseline output requires inspection before rerun: {output_root}"
        )
    command = command_for(
        python,
        method=method,
        output_root=output_root,
        seed=seed,
        episodes=episodes,
    )
    execute(command, log_path=output_root.parent / f"{output_root.name}.log")


def validate_run(
    output_root: Path, *, method: str, seed: int, episodes: int
) -> dict[str, Any]:
    if not is_complete(output_root, method, episodes):
        raise ValueError(f"incomplete run: {output_root}")
    rows = read_jsonl(output_root / "training_episodes.jsonl")
    decisions = read_jsonl(output_root / "force_range_decisions.jsonl")
    summary = json.loads((output_root / "training_summary.json").read_text(encoding="utf-8"))
    if len(rows) != episodes or len(decisions) != episodes:
        raise ValueError(f"episode/decision count mismatch: {output_root}")
    if [int(row["episode_index"]) for row in rows] != list(range(episodes)):
        raise ValueError(f"non-contiguous episode indices: {output_root}")
    if [int(row["seed"]) for row in rows] != list(range(seed, seed + episodes)):
        raise ValueError(f"non-contiguous episode seeds: {output_root}")
    assert_finite(rows)
    if summary["online_algorithm"] != method:
        raise ValueError(f"algorithm mismatch: {output_root}")
    if summary["force_range_source"] != "fixed" or summary["controller_mode"] != "single_stage":
        raise ValueError(f"matched protocol mismatch: {output_root}")
    if int(summary["vlm_calls"]) != 0 or int(summary["vlm_successful_calls"]) != 0:
        raise ValueError(f"baseline unexpectedly called the VLM: {output_root}")
    if int(summary["optimizer_updates"]) <= 0 or int(summary["replay_transitions"]) <= 0:
        raise ValueError(f"online learner did not update: {output_root}")
    if any(float(row["max_abs_lateral_residual_m"]) > 0.0060001 for row in rows):
        raise ValueError(f"lateral residual exceeded 6 mm: {output_root}")
    if any(row["attempted_force_range_n"] != [2.2, 2.8] for row in rows):
        raise ValueError(f"fixed force interval changed: {output_root}")
    return {
        "method": method,
        "seed": seed,
        "episodes": episodes,
        "successes": sum(int(bool(row["success"])) for row in rows),
        "last20_success_rate": sum(int(bool(row["success"])) for row in rows[-20:])
        / min(20, episodes),
        "optimizer_updates": int(summary["optimizer_updates"]),
        "replay_transitions": int(summary["replay_transitions"]),
        "output_root": str(output_root),
    }


def run_jobs(
    python: Path,
    jobs: list[tuple[str, int, int, Path]],
    *,
    workers: int,
) -> None:
    if workers <= 1:
        for method, seed, episodes, output_root in jobs:
            run_one(
                python,
                method=method,
                seed=seed,
                episodes=episodes,
                output_root=output_root,
            )
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_one,
                python,
                method=method,
                seed=seed,
                episodes=episodes,
                output_root=output_root,
            ): (method, seed)
            for method, seed, episodes, output_root in jobs
        }
        for future in as_completed(futures):
            method, seed = futures[future]
            future.result()
            print(f"[DONE] {method} seed={seed}", flush=True)


def smoke(python: Path, *, workers: int) -> None:
    jobs = [
        (
            method,
            SMOKE_SEEDS[method],
            3,
            BASELINE_ROOT / "smoke" / method / f"seed{SMOKE_SEEDS[method]}_n3",
        )
        for method in METHODS
    ]
    run_jobs(python, jobs, workers=workers)
    qa = [
        validate_run(root, method=method, seed=seed, episodes=episodes)
        for method, seed, episodes, root in jobs
    ]
    path = BASELINE_ROOT / "SMOKE_QA.json"
    path.write_text(json.dumps({"passed": True, "runs": qa}, indent=2) + "\n")
    print(f"[PASS] baseline smoke gate: {path}", flush=True)


def formal(python: Path, *, workers: int) -> None:
    smoke_path = BASELINE_ROOT / "SMOKE_QA.json"
    if not smoke_path.is_file() or not json.loads(smoke_path.read_text())["passed"]:
        raise RuntimeError("formal comparison requires a passing baseline smoke gate")
    jobs = [
        (
            method,
            seed,
            100,
            BASELINE_ROOT / method / f"seed{seed}_n100",
        )
        for method in METHODS
        for seed in FORMAL_SEEDS
    ]
    run_jobs(python, jobs, workers=workers)
    qa = [
        validate_run(root, method=method, seed=seed, episodes=episodes)
        for method, seed, episodes, root in jobs
    ]
    path = BASELINE_ROOT / "FORMAL_QA.json"
    path.write_text(json.dumps({"passed": True, "runs": qa}, indent=2) + "\n")
    subprocess.run(
        (str(python), str(ANALYZER), "--report-root", str(REPORT_ROOT)),
        cwd=ROOT,
        check=True,
    )


def analyze(python: Path) -> None:
    subprocess.run(
        (str(python), str(ANALYZER), "--report-root", str(REPORT_ROOT)),
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal", "all", "analyze"), default="all")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.mode in {"smoke", "all"}:
        smoke(args.python, workers=args.workers)
    if args.mode in {"formal", "all"}:
        formal(args.python, workers=args.workers)
    if args.mode == "analyze":
        analyze(args.python)


if __name__ == "__main__":
    main()
