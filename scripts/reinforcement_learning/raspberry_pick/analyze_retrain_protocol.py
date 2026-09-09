#!/usr/bin/env python3
"""Audit calibration or paired RaspberryPick retraining and render source-backed curves.

Figure contract
---------------
Core conclusion: repaired VLM+DSRL must exceed DSRL-only across paired seeds,
not only in a selected run or a single smoothed endpoint.
Archetype: quantitative grid.
Hero evidence: all-seed online trailing-window success curves.
Validation: paired per-seed overall/last-20 changes and failure decomposition.
Reviewer risks: small seed count, autocorrelated online outcomes, contact
non-determinism, post-hoc profile/method selection, and missing holdout evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = ROOT / "exp_report" / "raspberry_pick" / "online_vlm_dsrl"
METHOD_LABELS = {
    "dsrl_only": "DSRL only",
    "vlm_noop": "VLM no-op + DSRL",
    "vlm_force_v2": "Guarded VLM + latent DSRL",
    "vlm_semantic_v2": "Guarded VLM + semantic DSRL",
}
COLORS = {
    "dsrl_only": "#68707a",
    "vlm_noop": "#a2a8ae",
    "vlm_force_v2": "#5c88b2",
    "vlm_semantic_v2": "#c34f52",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", choices=("calibration", "paired_n30", "paired_n100"), required=True)
    parser.add_argument("--profile", choices=("fixed", "delicate", "firm"), default="delicate")
    parser.add_argument("--vlm-method", choices=tuple(METHOD_LABELS)[1:], default="vlm_semantic_v2")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--minimum-gain-pp", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_runs(root: Path, stage: str, profile: str) -> list[dict[str, Any]]:
    base = root / stage / profile
    runs: list[dict[str, Any]] = []
    if not base.is_dir():
        return runs
    for summary_path in sorted(base.glob("**/training_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records_path = summary_path.with_name("training_episodes.jsonl")
        if not records_path.is_file():
            continue
        records = load_jsonl(records_path)
        runs.append({"summary": summary, "records": records, "directory": summary_path.parent})
    return runs


def trailing(values: list[int], window: int) -> list[float]:
    return [sum(values[max(0, index - window + 1) : index + 1]) / min(index + 1, window) for index in range(len(values))]


def failure_class(record: dict[str, Any]) -> str:
    if record["success"]:
        return "success"
    if record["outcome"] == "no_detach" and float(record["bilateral_contact_fraction"]) < 0.10:
        return "low_contact"
    if record["outcome"] in {"crushed", "slipped"}:
        return str(record["outcome"])
    return "other_failure"


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    records = run["records"]
    outcomes = [int(row["success"]) for row in records]
    failures = Counter(failure_class(row) for row in records)
    unsafe_changes = sum(
        row["attempted_force_range_n"] != row["next_force_range_n"]
        for row in records
        if row["success"] or failure_class(row) == "low_contact"
    )
    return {
        "method": run["summary"]["post_training_method"],
        "profile": run["summary"]["berry_profile"],
        "seed": int(run["summary"]["training_seed"]),
        "episodes": len(records),
        "overall": mean(outcomes),
        "last20": mean(outcomes[-20:]),
        "failures": dict(failures),
        "unsafe_guard_changes": unsafe_changes,
        "outcomes": outcomes,
        "rolling": trailing(outcomes, 10),
    }


def calibration_report(runs: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    excluded_short_pilots: list[dict[str, Any]] = []
    maximum_episodes = max((len(run["records"]) for run in runs), default=0)
    for run in runs:
        if len(run["records"]) < maximum_episodes:
            excluded_short_pilots.append(
                {
                    "profile": run["summary"]["berry_profile"],
                    "seed": int(run["summary"]["training_seed"]),
                    "episodes": len(run["records"]),
                    "reason": f"short pilot excluded; formal calibration length is {maximum_episodes}",
                }
            )
            continue
        low, high = run["summary"]["initial_force_range_n"]
        center = round(0.5 * (float(low) + float(high)), 6)
        grouped[(run["summary"]["berry_profile"], center)].append(summarize_run(run))

    rows: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {}
    for (profile, center), values in sorted(grouped.items()):
        row = {
            "profile": profile,
            "force_center_n": center,
            "runs": len(values),
            "episodes": sum(item["episodes"] for item in values),
            "mean_success_rate": mean(item["overall"] for item in values),
            "low_contact": sum(item["failures"].get("low_contact", 0) for item in values),
            "crushed": sum(item["failures"].get("crushed", 0) for item in values),
            "slipped": sum(item["failures"].get("slipped", 0) for item in values),
            "other_failure": sum(item["failures"].get("other_failure", 0) for item in values),
        }
        rows.append(row)
        if profile not in best or row["mean_success_rate"] > best[profile]["mean_success_rate"]:
            best[profile] = row

    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "calibration_source_data.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    distinct_best = len({item["force_center_n"] for item in best.values()}) >= 2
    report = {
        "input_runs": len(runs),
        "included_runs": sum(len(values) for values in grouped.values()),
        "included_episode_length": maximum_episodes,
        "excluded_short_pilots": excluded_short_pilots,
        "rows": rows,
        "best_by_profile": best,
        "distinct_best_centers": distinct_best,
    }
    (output_dir / "calibration_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def exact_sign_test_two_sided(differences: list[float]) -> float | None:
    nonzero = [value for value in differences if value != 0.0]
    if not nonzero:
        return None
    positives = sum(value > 0.0 for value in nonzero)
    tail = min(positives, len(nonzero) - positives)
    probability = sum(math.comb(len(nonzero), k) for k in range(tail + 1)) / (2 ** len(nonzero))
    return min(1.0, 2.0 * probability)


def paired_report(
    runs: list[dict[str, Any]],
    *,
    vlm_method: str,
    window: int,
    minimum_gain_pp: float,
    output_dir: Path,
    allow_incomplete: bool,
) -> dict[str, Any]:
    summaries = [summarize_run(run) for run in runs]
    by_method_seed = {(item["method"], item["seed"]): item for item in summaries}
    seeds = sorted(
        {seed for method, seed in by_method_seed if method == "dsrl_only"}
        & {seed for method, seed in by_method_seed if method == vlm_method}
    )
    if not seeds:
        raise RuntimeError("no paired DSRL/VLM seeds were found")
    baseline = [by_method_seed[("dsrl_only", seed)] for seed in seeds]
    vlm = [by_method_seed[(vlm_method, seed)] for seed in seeds]
    episode_counts = {item["episodes"] for item in [*baseline, *vlm]}
    if len(episode_counts) != 1 and not allow_incomplete:
        raise RuntimeError(f"paired runs have unequal episode counts: {sorted(episode_counts)}")
    episodes = min(episode_counts)
    if any(item["episodes"] != episodes for item in [*baseline, *vlm]) and not allow_incomplete:
        raise RuntimeError("incomplete run detected")

    source_rows: list[dict[str, Any]] = []
    for item in [*baseline, *vlm]:
        for index, (success, rolling_value) in enumerate(zip(item["outcomes"], trailing(item["outcomes"], window))):
            source_rows.append(
                {
                    "method": item["method"],
                    "seed": item["seed"],
                    "episode": index + 1,
                    "success": success,
                    "trailing_success_rate": rolling_value,
                }
            )

    overall_differences = [v["overall"] - b["overall"] for b, v in zip(baseline, vlm)]
    last20_differences = [v["last20"] - b["last20"] for b, v in zip(baseline, vlm)]
    baseline_low_contact = sum(item["failures"].get("low_contact", 0) for item in baseline)
    vlm_low_contact = sum(item["failures"].get("low_contact", 0) for item in vlm)
    baseline_crushed = sum(item["failures"].get("crushed", 0) for item in baseline)
    vlm_crushed = sum(item["failures"].get("crushed", 0) for item in vlm)
    mean_gain = mean(overall_differences)
    gates = {
        "mean_gain_at_least_requested": mean_gain >= minimum_gain_pp / 100.0,
        "at_least_two_of_three_seeds_improve": sum(value > 0 for value in overall_differences) >= min(2, len(seeds)),
        "overall_and_last20_both_improve": mean(overall_differences) > 0.0 and mean(last20_differences) > 0.0,
        "low_contact_does_not_increase": vlm_low_contact <= baseline_low_contact,
        "crushed_does_not_increase": vlm_crushed <= baseline_crushed,
        "success_or_low_contact_proposal_changes_zero": sum(item["unsafe_guard_changes"] for item in vlm) == 0,
    }
    report = {
        "baseline_method": "dsrl_only",
        "vlm_method": vlm_method,
        "paired_seeds": seeds,
        "episodes_per_run": episodes,
        "baseline_mean_overall": mean(item["overall"] for item in baseline),
        "vlm_mean_overall": mean(item["overall"] for item in vlm),
        "mean_gain_pp": 100.0 * mean_gain,
        "per_seed_overall_gain_pp": [100.0 * value for value in overall_differences],
        "baseline_mean_last20": mean(item["last20"] for item in baseline),
        "vlm_mean_last20": mean(item["last20"] for item in vlm),
        "per_seed_last20_gain_pp": [100.0 * value for value in last20_differences],
        "paired_seed_sign_test_p": exact_sign_test_two_sided(overall_differences),
        "failure_counts": {
            "dsrl_only": dict(sum((Counter(item["failures"]) for item in baseline), Counter())),
            vlm_method: dict(sum((Counter(item["failures"]) for item in vlm), Counter())),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "interpretation_limit": "online outcomes are autocorrelated; frozen paired holdout remains required",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "online_curve_source_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    (output_dir / "comparison_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    render_figure(baseline, vlm, vlm_method=vlm_method, window=window, output_dir=output_dir)
    return report


def render_figure(
    baseline: list[dict[str, Any]],
    vlm: list[dict[str, Any]],
    *,
    vlm_method: str,
    window: int,
    output_dir: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig = plt.figure(figsize=(7.09, 4.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.65, 1.0), height_ratios=(1.0, 1.0))
    ax_curve = fig.add_subplot(grid[:, 0])
    ax_gain = fig.add_subplot(grid[0, 1])
    ax_failure = fig.add_subplot(grid[1, 1])

    for method, values in (("dsrl_only", baseline), (vlm_method, vlm)):
        color = COLORS[method]
        curves = np.asarray([trailing(item["outcomes"], window) for item in values], dtype=float)
        x = np.arange(1, curves.shape[1] + 1)
        for curve in curves:
            ax_curve.plot(x, 100.0 * curve, color=color, alpha=0.20, linewidth=0.7)
        average = curves.mean(axis=0)
        ax_curve.plot(x, 100.0 * average, color=color, linewidth=2.0, label=METHOD_LABELS[method])
        ax_curve.text(x[-1] + 0.7, 100.0 * average[-1], METHOD_LABELS[method], color=color, va="center", fontsize=6)
    ax_curve.set(xlabel="Online episode", ylabel=f"Trailing {window}-episode success (%)", ylim=(0, 103))
    ax_curve.set_xlim(1, curves.shape[1] + 16)
    ax_curve.grid(axis="y", color="#d8dde3", linewidth=0.5)

    differences = np.asarray([100.0 * (v["overall"] - b["overall"]) for b, v in zip(baseline, vlm)])
    x_seed = np.arange(len(differences))
    ax_gain.axhline(0.0, color="#6b7075", linewidth=0.8)
    ax_gain.scatter(x_seed, differences, color=COLORS[vlm_method], s=26, zorder=3)
    ax_gain.plot(x_seed, differences, color=COLORS[vlm_method], alpha=0.45, linewidth=0.8)
    ax_gain.set_xticks(x_seed, [f"S{i + 1}" for i in x_seed])
    ax_gain.set(ylabel="Paired overall gain (pp)", xlabel="Training seed")
    ax_gain.grid(axis="y", color="#e0e3e7", linewidth=0.5)

    categories = ("low_contact", "crushed", "slipped", "other_failure")
    labels = ("Low contact", "Crushed", "Slipped", "Other")
    bottoms = np.zeros(2)
    total = np.asarray([sum(item["episodes"] for item in baseline), sum(item["episodes"] for item in vlm)])
    palette = ("#b8bdc5", "#d76b65", "#d5a55a", "#8ba6bf")
    for category, label, color in zip(categories, labels, palette):
        counts = np.asarray(
            [
                sum(item["failures"].get(category, 0) for item in baseline),
                sum(item["failures"].get(category, 0) for item in vlm),
            ]
        )
        heights = 100.0 * counts / total
        ax_failure.bar((0, 1), heights, bottom=bottoms, width=0.62, color=color, label=label)
        bottoms += heights
    ax_failure.set_xticks((0, 1), ("DSRL", "VLM+DSRL"))
    ax_failure.set(ylabel="Failure rate (%)", ylim=(0, max(10.0, float(bottoms.max()) * 1.18)))
    ax_failure.legend(ncol=2, fontsize=5.5, loc="upper center")

    for label, axis in zip(("a", "b", "c"), (ax_curve, ax_gain, ax_failure)):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=9)

    stem = output_dir / "online_success_comparison"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "comparison" / args.stage / args.profile
    )
    runs = discover_runs(root, args.stage, args.profile)
    if args.stage == "calibration":
        report = calibration_report(runs, output_dir)
    else:
        report = paired_report(
            runs,
            vlm_method=args.vlm_method,
            window=args.window,
            minimum_gain_pp=args.minimum_gain_pp,
            output_dir=output_dir,
            allow_incomplete=args.allow_incomplete,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
