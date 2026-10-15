#!/usr/bin/env python3
"""Audit and summarize frozen final-policy LabPick evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
METHODS = ("base", "residual_rl", "awr", "dppo", "dsrl", "rapid_tactile")
DISPLAY_NAMES = {
    "base": "Base Policy",
    "residual_rl": "Residual RL",
    "awr": "AWR",
    "dppo": "DPPO",
    "dsrl": "DSRL",
    "rapid_tactile": "RAPID-Tactile",
}
TRAINING_SEEDS = (5200, 6200, 7200)
EVALUATION_SEED_STARTS = {5200: 45200, 6200: 45300, 7200: 45400}
UNDER_FORCE = "object_dropped"
OVER_FORCE = "object_broken"
T_975_DF2 = 4.302652729911275


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_path(root: Path, method: str, training_seed: int, episodes: int) -> Path:
    eval_seed = EVALUATION_SEED_STARTS[training_seed]
    return (
        root
        / method
        / f"trainseed_{training_seed}_evalseed_{eval_seed}_n{episodes}"
        / "results.json"
    )


def checkpoint_path(payload: dict[str, Any], method: str) -> Path:
    metadata_path = Path(payload["bc_policy"])
    if method in {"base", "awr"}:
        return metadata_path
    if method == "residual_rl":
        return Path(payload["input_checkpoint"])
    if method == "dppo":
        return Path(payload["input_checkpoint"])
    return Path(payload["input_dsrl_checkpoint"])


def load_run(root: Path, method: str, training_seed: int, episodes: int) -> dict[str, Any]:
    path = result_path(root, method, training_seed, episodes)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    rows = payload.get("results", [])
    eval_seed = EVALUATION_SEED_STARTS[training_seed]
    if payload.get("phase") != "evaluation":
        raise ValueError(f"not an evaluation result: {path}")
    if int(payload.get("completed_episodes", -1)) != episodes or len(rows) != episodes:
        raise ValueError(f"incomplete evaluation: {path}")
    observed_seeds = [int(row.get("reset_seed", -1)) for row in rows]
    if observed_seeds != list(range(eval_seed, eval_seed + episodes)):
        raise ValueError(f"unpaired or non-contiguous evaluation seeds: {path}")
    update_fields = (
        "gradient_updates",
        "dsrl_gradient_updates",
        "flow_rwr_gradient_updates",
        "flow_ppo_optimizer_steps",
    )
    if any(int(payload.get(field, 0) or 0) != 0 for field in update_fields):
        raise ValueError(f"evaluation performed parameter updates: {path}")
    if method == "rapid_tactile":
        if not payload.get("force_adaptation_frozen"):
            raise ValueError(f"RAPID-Tactile force adaptation was not frozen: {path}")
        if int(payload.get("advisor_calls", -1)) != 0:
            raise ValueError(f"RAPID-Tactile called the advisor during evaluation: {path}")
        if payload.get("initial_force_range_n") != payload.get("final_force_range_n"):
            raise ValueError(f"RAPID-Tactile force range changed during evaluation: {path}")
    source_checkpoint = checkpoint_path(payload, method).expanduser().resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    outcomes = [str(row["terminal_reason"]) for row in rows]
    successes = sum(int(bool(row["success"])) for row in rows)
    failures = Counter(outcome for outcome in outcomes if outcome != "success")
    return {
        "method": method,
        "display_name": DISPLAY_NAMES[method],
        "training_seed": training_seed,
        "evaluation_seed_start": eval_seed,
        "evaluation_seed_end": eval_seed + episodes - 1,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "under_force_failures": failures.get(UNDER_FORCE, 0),
        "over_force_failures": failures.get(OVER_FORCE, 0),
        "force_related_failures": failures.get(UNDER_FORCE, 0) + failures.get(OVER_FORCE, 0),
        "force_related_failure_rate": (
            failures.get(UNDER_FORCE, 0) + failures.get(OVER_FORCE, 0)
        )
        / episodes,
        "failure_counts": json.dumps(dict(failures), ensure_ascii=False, sort_keys=True),
        "evaluation_policy": payload.get("evaluation_policy"),
        "force_adaptation_frozen": bool(payload.get("force_adaptation_frozen", False)),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "result_json": str(path.resolve()),
        "result_json_sha256": sha256(path),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        rates = [float(row["success_rate"]) for row in selected]
        force_rates = [float(row["force_related_failure_rate"]) for row in selected]
        successes = sum(int(row["successes"]) for row in selected)
        total = sum(int(row["episodes"]) for row in selected)
        rate_mean = statistics.fmean(rates)
        rate_sd = statistics.stdev(rates)
        half_width = T_975_DF2 * rate_sd / math.sqrt(len(rates))
        wilson_low, wilson_high = wilson(successes, total)
        summaries.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "independent_runs": len(selected),
                "episodes_per_run": selected[0]["episodes"],
                "total_episodes": total,
                "successes": successes,
                "final_success_rate_mean_pct": 100.0 * rate_mean,
                "final_success_rate_sample_sd_pct": 100.0 * rate_sd,
                "seed_level_t95_ci_low_pct": 100.0 * max(0.0, rate_mean - half_width),
                "seed_level_t95_ci_high_pct": 100.0 * min(1.0, rate_mean + half_width),
                "pooled_success_rate_pct": 100.0 * successes / total,
                "pooled_wilson_95_ci_low_pct": 100.0 * wilson_low,
                "pooled_wilson_95_ci_high_pct": 100.0 * wilson_high,
                "force_related_failure_rate_mean_pct": 100.0 * statistics.fmean(force_rates),
                "force_related_failure_rate_sample_sd_pct": 100.0 * statistics.stdev(force_rates),
                "under_force_failures": sum(int(row["under_force_failures"]) for row in selected),
                "over_force_failures": sum(int(row["over_force_failures"]) for row in selected),
            }
        )
    return summaries


def combined_rows(summary_rows: list[dict[str, Any]], online_table: Path) -> list[dict[str, Any]]:
    with online_table.open(newline="", encoding="utf-8") as stream:
        online = {row["method"]: row for row in csv.DictReader(stream)}
    combined = []
    for final in summary_rows:
        row = online[final["method"]]
        combined.append(
            {
                "method": final["method"],
                "display_name": final["display_name"],
                "online_sr_mean_pct": row["success_rate_mean_pct"],
                "online_sr_sample_sd_pct": row["success_rate_sample_sd_pct"],
                "online_nauc_pct": row["nauc_pct"],
                "online_e80_episode": row["t80_episode"],
                "online_force_failure_rate_pct": row["force_related_failure_rate_pct"],
                "final_sr_mean_pct": final["final_success_rate_mean_pct"],
                "final_sr_sample_sd_pct": final["final_success_rate_sample_sd_pct"],
                "final_force_failure_rate_mean_pct": final[
                    "force_related_failure_rate_mean_pct"
                ],
            }
        )
    return combined


def write_report(path: Path, summary_rows: list[dict[str, Any]], combined: list[dict[str, Any]]) -> None:
    lines = [
        "# LabPick final-policy evaluation",
        "",
        (
            "Each row summarizes three frozen-policy evaluations. Each training seed is "
            "evaluated for 100 episodes on a disjoint held-out seed block, paired across "
            "methods. Mean and sample SD use the three seed-level success rates as the "
            "independent units. DPPO retains its seeded stochastic DDPM sampling; the other "
            "learned residual actors are evaluated deterministically on top of seeded Flow noise."
        ),
        "",
        "| Method | Final SR mean ± sample SD | Seed-level 95% t CI | Successes / 300 | Force failure mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['display_name']} | {row['final_success_rate_mean_pct']:.1f}% ± "
            f"{row['final_success_rate_sample_sd_pct']:.1f}% | "
            f"[{row['seed_level_t95_ci_low_pct']:.1f}%, "
            f"{row['seed_level_t95_ci_high_pct']:.1f}%] | "
            f"{row['successes']} / {row['total_episodes']} | "
            f"{row['force_related_failure_rate_mean_pct']:.1f}% ± "
            f"{row['force_related_failure_rate_sample_sd_pct']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Manuscript-table metrics versus final-policy metrics",
            "",
            (
                "The current main comparison table defines SR, nAUC, E80, and force failure "
                "over the 100 online adaptation episodes. Final SR below is a separate "
                "post-training endpoint and must not silently replace the table SR without "
                "changing the table definition."
            ),
            "",
            "| Method | Online SR | nAUC | E80 | Online force failure | Final SR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in combined:
        e80 = row["online_e80_episode"] or "/"
        lines.append(
            f"| {row['display_name']} | {float(row['online_sr_mean_pct']):.1f}% ± "
            f"{float(row['online_sr_sample_sd_pct']):.1f}% | "
            f"{float(row['online_nauc_pct']):.1f}% | {e80} | "
            f"{float(row['online_force_failure_rate_pct']):.1f}% | "
            f"{row['final_sr_mean_pct']:.1f}% ± {row['final_sr_sample_sd_pct']:.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=ROOT / "exp_report/lab_pick/final_evaluation_n100",
    )
    parser.add_argument(
        "--online-data-root",
        type=Path,
        default=Path("/home/limx/Documents/tactile_rl/data/labpick"),
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path("/home/limx/Documents/tactile_rl/data/labpick/final_evaluation_n100"),
    )
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    evaluation_root = args.evaluation_root.expanduser().resolve()
    export_root = args.export_root.expanduser().resolve()
    export_root.mkdir(parents=True, exist_ok=True)
    per_run = [
        load_run(evaluation_root, method, seed, args.episodes)
        for method in METHODS
        for seed in TRAINING_SEEDS
    ]
    summaries = summarize(per_run)
    combined = combined_rows(
        summaries,
        args.online_data_root.expanduser().resolve() / "paper_table_values.csv",
    )
    write_csv(export_root / "final_evaluation_per_seed.csv", per_run)
    write_csv(export_root / "final_evaluation_summary.csv", summaries)
    write_csv(export_root / "online_and_final_metrics.csv", combined)
    payload = {
        "schema_version": 1,
        "independent_unit": "training seed paired with one held-out 100-episode seed block",
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seed_starts": EVALUATION_SEED_STARTS,
        "episodes_per_run": args.episodes,
        "force_related_failure_definition": {
            "under_force": UNDER_FORCE,
            "over_force": OVER_FORCE,
        },
        "per_run": per_run,
        "summary": summaries,
        "online_and_final_metrics": combined,
    }
    (export_root / "final_evaluation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(export_root / "FINAL_POLICY_RESULTS.md", summaries, combined)
    source_manifest = args.online_data_root.expanduser().resolve() / "comparison_manifest.json"
    shutil.copy2(source_manifest, export_root / "online_comparison_manifest.json")
    print(f"[PASS] wrote audited final-policy results to {export_root}")


if __name__ == "__main__":
    main()
