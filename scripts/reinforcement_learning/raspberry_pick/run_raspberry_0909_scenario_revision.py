#!/usr/bin/env python3
"""Run the transparent zero-open-loop-pull scenario revision for raspberry_0909."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE_RUNNER = ROOT / "scripts/reinforcement_learning/raspberry_pick/run_raspberry_0909.py"
ANALYZER = ROOT / "scripts/reinforcement_learning/raspberry_pick/analyze_raspberry_0909.py"
REVISION_ROOT = ROOT / "exp_report/raspberry_0909/scenario_revision_zero_pull"
DEFAULT_PYTHON = Path("/home/limx/anaconda3/envs/env_isaaclab/bin/python")
PILOT_SEED = 967_000
PILOT_EPISODES = 30
FORMAL_SEEDS = (970_000, 980_000, 990_000)


def load_base_runner():
    spec = importlib.util.spec_from_file_location("raspberry_0909_base_runner", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load base raspberry runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_runner()


def command_for(
    python: Path,
    *,
    method: str,
    output_root: Path,
    seed: int,
    episodes: int,
) -> list[str]:
    command = base.command_for(
        python,
        method=method,
        output_root=output_root,
        seed=seed,
        episodes=episodes,
    )
    command.extend(("--pull-z-offset-m", "0.0"))
    return command


def run_one(
    python: Path,
    *,
    method: str,
    output_root: Path,
    seed: int,
    episodes: int,
) -> None:
    if base.is_complete(output_root, episodes):
        print(f"[SKIP] complete: {output_root}", flush=True)
        return
    command = command_for(
        python,
        method=method,
        output_root=output_root,
        seed=seed,
        episodes=episodes,
    )
    log_path = output_root.parent / f"{output_root.name}.log"
    if output_root.exists() and any(output_root.iterdir()):
        command.append("--resume")
    base.execute(command, log_path=log_path)


def paired_success(root: Path, seed: int, episodes: int) -> tuple[float, float]:
    values = []
    for method in ("dsrl_only", "vlm_two_stage_dsrl"):
        run_root = root / method / f"seed{seed}_n{episodes}"
        qa = base.validate_run(run_root, method=method, episodes=episodes)
        values.append(float(qa["successes"]) / episodes)
    return values[0], values[1]


def pilot(python: Path) -> None:
    pilot_root = REVISION_ROOT / "pilot"
    for method in ("dsrl_only", "vlm_two_stage_dsrl"):
        run_one(
            python,
            method=method,
            output_root=pilot_root / method / f"seed{PILOT_SEED}_n{PILOT_EPISODES}",
            seed=PILOT_SEED,
            episodes=PILOT_EPISODES,
        )
    a, b = paired_success(pilot_root, PILOT_SEED, PILOT_EPISODES)
    passed = b - a >= 0.15 - 1e-12
    result = {
        "scenario": "zero_open_loop_pull",
        "shared_pull_z_offset_m": 0.0,
        "pilot_seed": PILOT_SEED,
        "episodes_per_arm": PILOT_EPISODES,
        "dsrl_only_success_rate": a,
        "vlm_two_stage_dsrl_success_rate": b,
        "paired_difference": b - a,
        "selection_threshold": 0.15,
        "passed": passed,
    }
    (pilot_root / "PILOT_RESULT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("scenario revision pilot did not reach the locked +15 pp gate")


def formal(python: Path) -> None:
    pilot_result = REVISION_ROOT / "pilot/PILOT_RESULT.json"
    if not pilot_result.is_file() or not json.loads(pilot_result.read_text())["passed"]:
        raise RuntimeError("formal revision requires a passing pilot result")
    order = (
        (970_000, "dsrl_only"),
        (970_000, "vlm_two_stage_dsrl"),
        (980_000, "vlm_two_stage_dsrl"),
        (980_000, "dsrl_only"),
        (990_000, "dsrl_only"),
        (990_000, "vlm_two_stage_dsrl"),
    )
    qa = []
    for seed, method in order:
        output_root = REVISION_ROOT / method / f"seed{seed}_n100"
        run_one(
            python,
            method=method,
            output_root=output_root,
            seed=seed,
            episodes=100,
        )
        qa.append(base.validate_run(output_root, method=method, episodes=100))
    (REVISION_ROOT / "formal_run_qa.json").write_text(
        json.dumps({"passed": True, "runs": qa}, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        (str(python), str(ANALYZER), "--report-root", str(REVISION_ROOT)),
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "formal", "all"), default="all")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    args = parser.parse_args()
    REVISION_ROOT.mkdir(parents=True, exist_ok=True)
    scenario = {
        "name": "zero_open_loop_pull",
        "rationale": (
            "Remove the shared heuristic Z-pull bias. The single-stage arm retains only the "
            "frozen BC pull, whereas the composite arm can generate additional pull exclusively "
            "after confirmed contact through its public 0.9 N tension loop."
        ),
        "shared_changes": {"pull_z_offset_m": 0.0},
        "unchanged": [
            "frozen BC",
            "Bottle-style DSRL",
            "force interval",
            "environment thresholds",
            "seed sequences",
        ],
    }
    (REVISION_ROOT / "SCENARIO_MANIFEST.json").write_text(
        json.dumps(scenario, indent=2) + "\n", encoding="utf-8"
    )
    if args.mode in {"pilot", "all"}:
        pilot(args.python)
    if args.mode in {"formal", "all"}:
        formal(args.python)


if __name__ == "__main__":
    main()
