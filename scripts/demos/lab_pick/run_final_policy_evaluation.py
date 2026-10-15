#!/usr/bin/env python3
"""Evaluate every LabPick paper-table policy on paired held-out episodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROOT = ROOT / "exp_report/lab_pick/runs"
DEFAULT_OUTPUT_ROOT = ROOT / "exp_report/lab_pick/final_evaluation_n100"
DEFAULT_EXPORT_ROOT = Path("/home/limx/Documents/tactile_rl/data/labpick/final_evaluation_n100")
FLOW_LAUNCHER = ROOT / "scripts/reinforcement_learning/vlm_dsrl/run_experiment.py"
RESIDUAL_LAUNCHER = ROOT / "scripts/reinforcement_learning/compare_exp/run_residual_experiment.py"
DPPO_LAUNCHER = ROOT / "scripts/reinforcement_learning/dppo/run_tactile_dppo.py"
ANALYZER = Path(__file__).with_name("analyze_final_policy_evaluation.py")
METHODS = ("base", "residual_rl", "awr", "dppo", "dsrl", "rapid_tactile")
TRAINING_SEEDS = (5200, 6200, 7200)
EVALUATION_SEED_STARTS = {5200: 45200, 6200: 45300, 7200: 45400}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_run(source_root: Path, method: str, training_seed: int) -> Path:
    repeat = TRAINING_SEEDS.index(training_seed) + 1
    return source_root / method / f"repeat_{repeat:02d}_seed_{training_seed}"


def result_path(output_root: Path, method: str, training_seed: int, episodes: int) -> Path:
    eval_seed = EVALUATION_SEED_STARTS[training_seed]
    return (
        output_root
        / method
        / f"trainseed_{training_seed}_evalseed_{eval_seed}_n{episodes}"
        / "results.json"
    )


def _common_flow_args(
    *, mode: str, bc_policy: Path, output_dir: Path, episodes: int, eval_seed: int
) -> list[str]:
    return [
        sys.executable,
        str(FLOW_LAUNCHER),
        "--mode",
        mode,
        "--phase",
        "evaluation",
        "--evaluation-policy",
        "deterministic",
        "--bc-policy",
        str(bc_policy),
        "--output-dir",
        str(output_dir),
        "--num-episodes",
        str(episodes),
        "--seed",
        str(eval_seed),
        "--break-force-threshold-n",
        "4.5",
        "--labware-random-xy-m",
        "0.10",
        "0.10",
        "--labware-random-yaw-deg",
        "0",
        "--physical-force-range-n",
        "0.5",
        "3.8",
        "--initial-force-range-n",
        "0.8",
        "1.4",
        "--learned-noise-steps",
        "1",
        "--dsrl-noise-residual-scale",
        "0.25",
        "--chunk-execute-steps",
        "32",
        "--flow-num-inference-steps",
        "20",
        "--initial-log-std",
        "-2.0",
        "--gpu-max-rigid-contact-count",
        "1048576",
        "--gpu-max-rigid-patch-count",
        "1048576",
        "--max-outer-interactions",
        "20",
    ]


def command_for(
    source_root: Path,
    output_root: Path,
    *,
    method: str,
    training_seed: int,
    episodes: int,
) -> list[str]:
    source = source_run(source_root, method, training_seed)
    metadata = read_json(source / "run_metadata.json")
    output_dir = result_path(output_root, method, training_seed, episodes).parent
    eval_seed = EVALUATION_SEED_STARTS[training_seed]
    bc_policy = Path(metadata["bc_policy"]).expanduser().resolve()
    if method == "base":
        return _common_flow_args(
            mode="base",
            bc_policy=bc_policy,
            output_dir=output_dir,
            episodes=episodes,
            eval_seed=eval_seed,
        )
    if method == "awr":
        return _common_flow_args(
            mode="flow_rwr",
            bc_policy=(source / "flow_rwr_final.pt").resolve(),
            output_dir=output_dir,
            episodes=episodes,
            eval_seed=eval_seed,
        )
    if method in {"dsrl", "rapid_tactile"}:
        mode = "dsrl_tactile" if method == "dsrl" else "ours_tactile"
        command = _common_flow_args(
            mode=mode,
            bc_policy=bc_policy,
            output_dir=output_dir,
            episodes=episodes,
            eval_seed=eval_seed,
        )
        command.extend(("--dsrl-checkpoint", str((source / "dsrl_final.pt").resolve())))
        command.extend(
            ("--dsrl-contract-version", str(int(metadata["dsrl_contract_version"])))
        )
        if method == "rapid_tactile":
            final_range = read_json(source / "results.json")["final_force_range_n"]
            start = command.index("--initial-force-range-n") + 1
            command[start : start + 2] = [str(final_range[0]), str(final_range[1])]
            command.extend(("--advisor", "deterministic", "--freeze-force-adaptation"))
        return command
    if method == "residual_rl":
        return [
            sys.executable,
            str(RESIDUAL_LAUNCHER),
            "--phase",
            "evaluation",
            "--bc-policy",
            str(bc_policy),
            "--checkpoint",
            str((source / "residual_final.pt").resolve()),
            "--output-dir",
            str(output_dir),
            "--num-episodes",
            str(episodes),
            "--seed",
            str(eval_seed),
            "--break-force-threshold-n",
            "4.5",
            "--labware-random-xy-m",
            "0.10",
            "0.10",
            "--labware-random-yaw-deg",
            "0",
            "--residual-scale",
            str(metadata["residual_scale"]),
            "--gpu-max-rigid-contact-count",
            "1048576",
            "--gpu-max-rigid-patch-count",
            "1048576",
            "--max-outer-interactions",
            "600",
        ]
    if method == "dppo":
        return [
            sys.executable,
            str(DPPO_LAUNCHER),
            "--phase",
            "evaluation",
            "--bc-policy",
            str(bc_policy),
            "--bc-validation-metrics",
            str(Path(metadata["bc_validation_metrics"]).expanduser().resolve()),
            "--checkpoint",
            str((source / "dppo_final.pt").resolve()),
            "--output-dir",
            str(output_dir),
            "--num-episodes",
            str(episodes),
            "--seed",
            str(eval_seed),
            "--break-force-threshold-n",
            "4.5",
            "--labware-random-xy-m",
            "0.10",
            "0.10",
            "--labware-random-yaw-deg",
            "0",
            "--rollout-steps",
            "32",
            "--learning-epochs",
            "8",
            "--mini-batches",
            "4",
            "--learning-rate",
            "3e-05",
            "--ratio-clip",
            "0.2",
            "--value-clip",
            "0.2",
            "--grad-clip",
            "0.5",
            "--gae-lambda",
            "0.95",
            "--fine-tune-denoising-steps",
            "5",
            "--num-inference-steps",
            "10",
            "--min-sampling-denoising-std",
            "0.001",
            "--min-logprob-denoising-std",
            "0.1",
            "--gpu-max-rigid-contact-count",
            "1048576",
            "--gpu-max-rigid-patch-count",
            "1048576",
            "--max-outer-interactions",
            "20",
        ]
    raise ValueError(f"unsupported method: {method}")


def is_complete(path: Path, *, episodes: int, eval_seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    rows = payload.get("results", [])
    return (
        payload.get("phase") == "evaluation"
        and int(payload.get("completed_episodes", -1)) == episodes
        and len(rows) == episodes
        and [int(row.get("reset_seed", -1)) for row in rows]
        == list(range(eval_seed, eval_seed + episodes))
    )


def run_one(command: list[str], result: Path, log_path: Path, *, episodes: int) -> None:
    eval_seed = int(command[command.index("--seed") + 1])
    if is_complete(result, episodes=episodes, eval_seed=eval_seed):
        print(f"[SKIP] complete: {result.parent}", flush=True)
        return
    if result.parent.exists() and any(result.parent.iterdir()):
        raise RuntimeError(f"Incomplete output requires inspection: {result.parent}")
    result.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault(
        "TACEX_ISAAC_PYTHON", "/home/limx/anaconda3/envs/env_isaaclab/bin/python"
    )
    with log_path.open("x", encoding="utf-8", buffering=1) as stream:
        stream.write("[COMMAND] " + " ".join(command) + "\n")
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if process.returncode:
        raise RuntimeError(f"Evaluation failed ({process.returncode}); see {log_path}")
    if not is_complete(result, episodes=episodes, eval_seed=eval_seed):
        raise RuntimeError(f"Evaluation did not produce a complete result: {result}")
    print(f"[DONE] {result.parent}", flush=True)


def run_jobs(
    source_root: Path,
    output_root: Path,
    *,
    episodes: int,
    workers: int,
    smoke: bool,
) -> None:
    jobs: list[tuple[list[str], Path, Path]] = []
    seeds = (TRAINING_SEEDS[0],) if smoke else TRAINING_SEEDS
    methods = METHODS
    target_root = output_root / "smoke" if smoke else output_root
    actual_episodes = 1 if smoke else episodes
    for training_seed in seeds:
        for method in methods:
            command = command_for(
                source_root,
                target_root,
                method=method,
                training_seed=training_seed,
                episodes=actual_episodes,
            )
            if smoke:
                seed_index = command.index("--seed") + 1
                command[seed_index] = "45100"
            result = Path(command[command.index("--output-dir") + 1]) / "results.json"
            log = output_root / "logs" / f"{method}_trainseed{training_seed}_n{actual_episodes}.log"
            jobs.append((command, result, log))
    if workers == 1:
        for command, result, log in jobs:
            run_one(command, result, log, episodes=actual_episodes)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one, command, result, log, episodes=actual_episodes): result
            for command, result, log in jobs
        }
        for future in as_completed(futures):
            future.result()


def analyze(output_root: Path, export_root: Path, episodes: int) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--evaluation-root",
            str(output_root),
            "--export-root",
            str(export_root),
            "--episodes",
            str(episodes),
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal", "analyze", "all"), default="all")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1:
        parser.error("--episodes and --workers must be positive")
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    export_root = args.export_root.expanduser().resolve()
    if args.mode in {"smoke", "all"}:
        run_jobs(source_root, output_root, episodes=args.episodes, workers=args.workers, smoke=True)
    if args.mode in {"formal", "all"}:
        run_jobs(source_root, output_root, episodes=args.episodes, workers=args.workers, smoke=False)
    if args.mode in {"analyze", "all"}:
        analyze(output_root, export_root, args.episodes)


if __name__ == "__main__":
    main()
