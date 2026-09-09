#!/usr/bin/env python3
"""Curate the six formal LabPick comparison methods into one raw-data tree."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


METHOD_SOURCES = {
    "base": {
        "dataset": "compare_exp_100_episode",
        "source_method": "base",
        "reason": "Frozen Flow base-policy reference used by the comparison.",
    },
    "residual_rl": {
        "dataset": "compare_exp_100_with_tactile",
        "source_method": "residual_rl_tactile",
        "reason": "Tactile-conditioned action-space residual baseline.",
    },
    "awr": {
        "dataset": "compare_exp_100_episode",
        "source_method": "flow_rwr",
        "reason": "Direct Flow reward-weighted regression implementation of AWR.",
    },
    "dppo": {
        "dataset": "compare_exp_100_with_tactile",
        "source_method": "dppo_tactile",
        "reason": "True diffusion-policy PPO; excludes the Flow-PPO surrogate.",
    },
    "dsrl": {
        "dataset": "compare_exp_100_with_tactile",
        "source_method": "dsrl_tactile",
        "reason": "Tactile-conditioned latent-noise DSRL baseline.",
    },
    "rapid_tactile": {
        "dataset": "compare_exp_100_with_tactile",
        "source_method": "ours_tactile",
        "reason": "Guarded tactile DSRL with episode-level force adaptation.",
    },
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "exp_report" / "lab_pick",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Use independent copies by default; hardlink saves disk space.",
    )
    parser.add_argument(
        "--residual-source-root",
        type=Path,
        default=None,
        help=(
            "Optional experiment root containing runs/residual_rl_tactile and "
            "console_logs. Use this to curate a validated isolated rerun."
        ),
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "Replace only the generated runs, console logs, legacy manifests, "
            "and source-selection manifest under the output root."
        ),
    )
    return parser.parse_args()


def copy_function(mode: str) -> Callable[[str, str], str]:
    if mode == "copy":
        return shutil.copy2

    def hardlink(source: str, destination: str) -> str:
        os.link(source, destination)
        return destination

    return hardlink


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_residual_contract(metadata: dict[str, Any], run_dir: Path) -> None:
    expected_scalars = {
        "residual_contract_version": 3,
        "residual_scale": 0.01,
        "residual_action_l2_weight": 1.0,
        "initial_log_std": -3.0,
        "learning_starts": 32,
        "batch_size": 32,
    }
    for name, expected in expected_scalars.items():
        actual = metadata.get(name)
        if isinstance(expected, float):
            valid = actual is not None and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12
            )
        else:
            valid = actual == expected
        if not valid:
            raise ValueError(
                f"{run_dir}: Residual RL {name}={actual!r}; expected {expected!r}"
            )
    if metadata.get("residual_contact_gate") is not True:
        raise ValueError(f"{run_dir}: Residual RL contact gate is not enabled")
    rates = [float(value) for value in metadata.get("learning_rates", [])]
    if rates != [3.0e-5, 3.0e-5]:
        raise ValueError(
            f"{run_dir}: Residual RL learning rates are {rates}; expected [3e-5, 3e-5]"
        )


def run_summary(run_dir: Path, method: str) -> dict[str, Any]:
    result = read_json(run_dir / "results.json")
    metadata = read_json(run_dir / "run_metadata.json")
    if method == "residual_rl":
        validate_residual_contract(metadata, run_dir)
        if not (run_dir / "residual_final.pt").is_file():
            raise ValueError(f"{run_dir}: missing final Residual RL checkpoint")
        trajectory_count = len(list((run_dir / "trajectories").glob("episode_*.jsonl.gz")))
        if trajectory_count != 100:
            raise ValueError(
                f"{run_dir}: expected 100 Residual RL trajectories, found {trajectory_count}"
            )
    return {
        "seed": int(metadata["seed"]),
        "task": metadata["task"],
        "completed_episodes": int(result["completed_episodes"]),
        "successes": int(result["successes"]),
        "success_rate": float(result["success_rate"]),
        "failure_counts": result["failure_counts"],
    }


def log_name(dataset: str, source_method: str, repeat: int, seed: int) -> str:
    return f"{source_method}_repeat{repeat:02d}_seed{seed}_n100.log"


def copy_small_source_files(
    repo_root: Path,
    output_root: Path,
    residual_source_root: Path | None,
) -> None:
    destination = output_root / "legacy_manifests"
    destination.mkdir(parents=True, exist_ok=True)
    for dataset in ("compare_exp_100_episode", "compare_exp_100_with_tactile"):
        source_root = repo_root / "exp_report" / dataset
        for filename in (
            "manifest.json",
            "analysis_summary.json",
            "final_summary.csv",
            "per_run_summary.csv",
            "report.md",
        ):
            source = source_root / filename
            if source.exists():
                shutil.copy2(source, destination / f"{dataset}_{filename}")
    if residual_source_root is not None:
        source = residual_source_root / "manifest.json"
        if source.is_file():
            shutil.copy2(source, destination / "residual_rl_rerun_v3_manifest.json")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    residual_source_root = (
        None
        if args.residual_source_root is None
        else args.residual_source_root.expanduser().resolve()
    )
    runs_root = output_root / "runs"
    logs_root = output_root / "console_logs"

    source_plan: dict[
        str, tuple[str, str, Path, list[Path]]
    ] = {}
    for method, selection in METHOD_SOURCES.items():
        dataset = selection["dataset"]
        source_method = selection["source_method"]
        dataset_root = (
            residual_source_root
            if method == "residual_rl" and residual_source_root is not None
            else repo_root / "exp_report" / dataset
        )
        source_method_root = dataset_root / "runs" / source_method
        if not source_method_root.is_dir():
            raise FileNotFoundError(source_method_root)
        source_runs = sorted(source_method_root.glob("repeat_*_seed_*"))
        if len(source_runs) != 3:
            raise ValueError(
                f"{source_method_root}: expected three formal runs, found {len(source_runs)}"
            )
        summaries = [run_summary(source_run, method) for source_run in source_runs]
        if [summary["seed"] for summary in summaries] != [5200, 6200, 7200]:
            raise ValueError(
                f"{source_method_root}: expected ordered seeds 5200, 6200, 7200"
            )
        if any(summary["completed_episodes"] != 100 for summary in summaries):
            raise ValueError(f"{source_method_root}: contains an incomplete run")
        source_plan[method] = (dataset, source_method, dataset_root, source_runs)

    if args.replace_existing:
        for generated_directory in (
            runs_root,
            logs_root,
            output_root / "legacy_manifests",
        ):
            if generated_directory.exists():
                shutil.rmtree(generated_directory)
        (output_root / "source_selection.json").unlink(missing_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    copier = copy_function(args.copy_mode)

    manifest_methods: list[dict[str, Any]] = []
    for method, selection in METHOD_SOURCES.items():
        dataset, source_method, dataset_root, source_runs = source_plan[method]
        source_method_root = dataset_root / "runs" / source_method
        target_method_root = runs_root / method
        if target_method_root.exists():
            raise FileExistsError(
                f"refusing to overwrite existing curated method: {target_method_root}"
            )
        target_method_root.mkdir()

        copied_runs: list[dict[str, Any]] = []
        for repeat, source_run in enumerate(source_runs, start=1):
            target_run = target_method_root / source_run.name
            shutil.copytree(source_run, target_run, copy_function=copier)
            summary = run_summary(target_run, method)
            if summary["completed_episodes"] != 100:
                raise ValueError(f"{target_run}: incomplete run")
            source_log = (
                dataset_root
                / "console_logs"
                / log_name(dataset, source_method, repeat, summary["seed"])
            )
            copied_log: str | None = None
            if source_log.exists():
                target_log = logs_root / f"{method}_{source_log.name}"
                shutil.copy2(source_log, target_log)
                copied_log = str(target_log.relative_to(output_root))
            copied_runs.append(
                {
                    "repeat": repeat,
                    **summary,
                    "raw_directory": str(target_run.relative_to(output_root)),
                    "source_directory": str(source_run.relative_to(repo_root)),
                    "console_log": copied_log,
                }
            )
        manifest_methods.append(
            {
                "method": method,
                "source_dataset": (
                    dataset
                    if dataset_root == repo_root / "exp_report" / dataset
                    else dataset_root.name
                ),
                "source_root": str(dataset_root),
                "source_method": source_method,
                "selection_reason": selection["reason"],
                "runs": copied_runs,
            }
        )

    copy_small_source_files(repo_root, output_root, residual_source_root)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "TacEx-LabPick-Slide-Clean",
        "paper_task_name": "Thin Sheet Transfer",
        "purpose": "Raw evidence for the six-method online comparison table.",
        "copy_mode": args.copy_mode,
        "training_seeds": [5200, 6200, 7200],
        "episodes_per_run": 100,
        "runs_per_method": 3,
        "methods": manifest_methods,
        "excluded_historical_methods": {
            "flow_ppo": "Not DPPO; it optimizes a bounded Flow initial-noise surrogate.",
            "joint": "Historical unguarded VLM+DSRL diagnostic, not a paper-table method.",
            "vlm": "Ablation, not one of the six paper-table methods.",
            "guarded_joint_old": "Superseded by the tactile-audited RAPID-Tactile runs.",
        },
    }
    (output_root / "source_selection.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
