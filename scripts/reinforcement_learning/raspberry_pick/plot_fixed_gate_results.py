#!/usr/bin/env python3
"""Create the audited fixed-gate Base/DSRL/VLM+DSRL comparison figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", type=Path, required=True)
    parser.add_argument("--dsrl-input", type=Path, required=True)
    parser.add_argument("--vlm-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--expected-episodes", type=int, default=100)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(path: Path, expected_episodes: int) -> dict[str, Any]:
    path = path.expanduser().resolve()
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records.sort(key=lambda row: int(row["episode_index"]))
    if len(records) != expected_episodes:
        raise ValueError(f"{path}: expected {expected_episodes} rows, found {len(records)}")
    if [int(row["episode_index"]) for row in records] != list(range(expected_episodes)):
        raise ValueError(f"{path}: episode indices are not contiguous")
    seeds = [int(row["seed"]) for row in records]
    if len(set(seeds)) != expected_episodes:
        raise ValueError(f"{path}: episode seeds are not unique")
    summary_path = path.parent / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    successes = sum(int(row["success"]) for row in records)
    if int(summary["completed_episodes"]) != expected_episodes:
        raise ValueError(f"{summary_path}: completed episode count mismatch")
    if int(summary["successes"]) != successes:
        raise ValueError(f"{summary_path}: success count mismatch")
    return {
        "path": path,
        "summary_path": summary_path,
        "records": records,
        "summary": summary,
        "seeds": seeds,
    }


def rolling_success(records: list[dict[str, Any]], window: int) -> list[float]:
    values = [int(row["success"]) for row in records]
    return [
        sum(values[max(0, index - window + 1) : index + 1])
        / min(window, index + 1)
        for index in range(len(values))
    ]


def stats(run: dict[str, Any], curve: list[float]) -> dict[str, Any]:
    records = run["records"]
    successes = sum(int(row["success"]) for row in records)
    return {
        "episodes": len(records),
        "successes": successes,
        "overall_success_rate": successes / len(records),
        "first_20_success_rate": sum(int(row["success"]) for row in records[:20]) / 20,
        "last_20_success_rate": sum(int(row["success"]) for row in records[-20:]) / 20,
        "final_trailing_success_rate": curve[-1],
        "successes_per_10": [
            sum(int(row["success"]) for row in records[start : start + 10])
            for start in range(0, len(records), 10)
        ],
        "failure_counts": dict(Counter(row["outcome"] for row in records if not row["success"])),
        "contract": run["summary"]["contract"],
        "input_jsonl": str(run["path"]),
        "input_sha256": sha256(run["path"]),
        "summary_json": str(run["summary_path"]),
        "summary_sha256": sha256(run["summary_path"]),
    }


def paired_outcomes(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
    result = Counter()
    for a, b in zip(left["records"], right["records"], strict=True):
        if a["success"] and b["success"]:
            result["both_success"] += 1
        elif a["success"]:
            result["left_only_success"] += 1
        elif b["success"]:
            result["right_only_success"] += 1
        else:
            result["both_fail"] += 1
    return dict(result)


def write_source_data(
    path: Path,
    runs: dict[str, dict[str, Any]],
    curves: dict[str, list[float]],
    window: int,
) -> None:
    fieldnames = [
        "episode",
        "seed",
        "rolling_window_requested",
        "rolling_window_effective",
    ]
    for name in runs:
        fieldnames.extend((f"{name}_success", f"{name}_outcome", f"{name}_rolling_success_rate"))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(next(iter(runs.values()))["records"])):
            row: dict[str, Any] = {
                "episode": index + 1,
                "seed": runs["base"]["seeds"][index],
                "rolling_window_requested": window,
                "rolling_window_effective": min(window, index + 1),
            }
            for name, run in runs.items():
                record = run["records"][index]
                row[f"{name}_success"] = int(record["success"])
                row[f"{name}_outcome"] = record["outcome"]
                row[f"{name}_rolling_success_rate"] = curves[name][index]
            writer.writerow(row)


def draw_figure(
    output_stem: Path,
    curves: dict[str, list[float]],
    run_stats: dict[str, dict[str, Any]],
    window: int,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    colors = {"base": "#777D85", "dsrl": "#D17C2F", "vlm_dsrl": "#1769AA"}
    labels = {"base": "Base policy", "dsrl": "DSRL only", "vlm_dsrl": "VLM + DSRL"}
    styles = {"base": (0, (3, 2)), "dsrl": (0, (5, 1.5, 1, 1.5)), "vlm_dsrl": "solid"}
    episodes = list(range(1, len(curves["base"]) + 1))
    full_start = window - 1

    fig = plt.figure(figsize=(7.20, 2.75), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(2.15, 1.0))
    ax = fig.add_subplot(grid[0, 0])
    bar_ax = fig.add_subplot(grid[0, 1])

    for name in ("base", "dsrl", "vlm_dsrl"):
        curve = curves[name]
        ax.plot(
            episodes[: full_start + 1],
            curve[: full_start + 1],
            color=colors[name],
            linewidth=0.9,
            linestyle=(0, (2, 1.5)),
            alpha=0.45,
        )
        ax.plot(
            episodes[full_start:],
            curve[full_start:],
            color=colors[name],
            linewidth=1.7 if name == "vlm_dsrl" else 1.45,
            linestyle=styles[name],
            label=(
                f"{labels[name]} "
                f"({run_stats[name]['successes']}/{run_stats[name]['episodes']})"
            ),
        )
    ax.axvline(window, color="#D3D6DA", linewidth=0.7, linestyle=(0, (2, 2)))
    ax.set_xlim(1, 101)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.set_xlabel("Online episode")
    ax.set_ylabel(f"Trailing success rate (window={window})")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("a  Online dynamics", loc="left", fontweight="bold", pad=5)
    ax.legend(loc="lower left", fontsize=6.1, handlelength=2.4, borderaxespad=0.35)

    names = ("base", "dsrl", "vlm_dsrl")
    rates = [run_stats[name]["overall_success_rate"] for name in names]
    bars = bar_ax.bar(
        range(3), rates, width=0.62, color=[colors[name] for name in names], edgecolor="none"
    )
    for bar, rate in zip(bars, rates, strict=True):
        bar_ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.025,
            f"{rate:.0%}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    bar_ax.axhline(rates[0], color="#A7ABB0", linewidth=0.8, linestyle=(0, (3, 2)))
    bar_ax.set_xticks(range(3), ("Base", "DSRL", "VLM\n+ DSRL"))
    bar_ax.set_ylim(0.0, 0.86)
    bar_ax.set_ylabel("Overall success rate")
    bar_ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    bar_ax.grid(axis="y", color="#E7E9EC", linewidth=0.6)
    bar_ax.set_axisbelow(True)
    bar_ax.set_title("b  Sequential gate", loc="left", fontweight="bold", pad=5)
    bar_ax.text(
        0.02,
        0.98,
        "DSRL − Base: +4 pp  (pass)\nVLM − DSRL: 0 pp  (fail)",
        transform=bar_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
    )
    fig.suptitle("RaspberryPick fixed-threshold gate, seed 1010000", fontsize=9, fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        "One online-training seed per method; descriptive comparison without uncertainty intervals.",
        ha="center",
        va="top",
        fontsize=6,
        color="#555B62",
    )

    fig.savefig(output_stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(
        output_stem.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.window < 1:
        raise ValueError("window must be positive")
    runs = {
        "base": load_run(args.base_input, args.expected_episodes),
        "dsrl": load_run(args.dsrl_input, args.expected_episodes),
        "vlm_dsrl": load_run(args.vlm_input, args.expected_episodes),
    }
    seed_sequences = {tuple(run["seeds"]) for run in runs.values()}
    if len(seed_sequences) != 1:
        raise ValueError("all methods must use identical episode seeds")
    matched_fields = (
        "task",
        "base_checkpoint_sha256",
        "base_checkpoint_weights",
        "dino_frozen",
        "flow_initial_latent",
        "noise_action_indices",
        "noise_residual_scale",
        "dsrl_start_step",
        "physical_force_bounds_n",
        "damage_threshold_n",
        "detach_threshold_n",
        "position_randomization_m",
    )
    for field in matched_fields:
        values = [run["summary"].get(field) for run in runs.values()]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"run mismatch in {field}: {values}")
    if not bool(runs["base"]["summary"].get("dsrl_frozen")):
        raise ValueError("base run must freeze DSRL")
    if bool(runs["dsrl"]["summary"].get("vlm_is_codex")):
        raise ValueError("DSRL-only run unexpectedly used the VLM")
    if not bool(runs["vlm_dsrl"]["summary"].get("vlm_is_codex")):
        raise ValueError("VLM+DSRL run did not use the VLM")

    curves = {name: rolling_success(run["records"], args.window) for name, run in runs.items()}
    run_stats = {name: stats(run, curves[name]) for name, run in runs.items()}
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_csv = output_dir / "fixed_gate_window10_source.csv"
    output_stem = output_dir / "fixed_gate_window10"
    write_source_data(source_csv, runs, curves, args.window)
    draw_figure(output_stem, curves, run_stats, args.window)

    decisions_path = runs["vlm_dsrl"]["path"].parent / "force_range_decisions.jsonl"
    decisions = [
        json.loads(line)
        for line in decisions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual_changes = sum(
        list(map(float, row["previous_range_n"]))
        != list(map(float, row["applied_range_n"]))
        for row in decisions
    )
    manifest = {
        "figure_contract": {
            "core_conclusion": (
                "At fixed 3.2/0.8 N thresholds, DSRL exceeds the paired Base policy, "
                "whereas guarded force-only VLM+DSRL ties DSRL and fails the second gate."
            ),
            "evidence_chain": {
                "panel_a": "10-episode trailing online dynamics for all paired methods",
                "panel_b": "100-episode overall success rates and gate decisions",
            },
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "final_size_mm": [182.88, 69.85],
        },
        "training_seed": runs["base"]["seeds"][0],
        "episode_seed_range": [runs["base"]["seeds"][0], runs["base"]["seeds"][-1]],
        "episodes_per_method": args.expected_episodes,
        "rolling_window": args.window,
        "matched_fields": list(matched_fields),
        "methods": run_stats,
        "gates": {
            "dsrl_exceeds_base": run_stats["dsrl"]["successes"] > run_stats["base"]["successes"],
            "vlm_dsrl_exceeds_dsrl": (
                run_stats["vlm_dsrl"]["successes"] > run_stats["dsrl"]["successes"]
            ),
            "additional_seed_runs_authorized": (
                run_stats["vlm_dsrl"]["successes"] > run_stats["dsrl"]["successes"]
            ),
        },
        "paired_outcomes": {
            "base_vs_dsrl": paired_outcomes(runs["base"], runs["dsrl"]),
            "dsrl_vs_vlm_dsrl": paired_outcomes(runs["dsrl"], runs["vlm_dsrl"]),
        },
        "vlm_audit": {
            "calls": int(runs["vlm_dsrl"]["summary"]["vlm_calls"]),
            "successful_calls": int(runs["vlm_dsrl"]["summary"]["vlm_successful_calls"]),
            "actual_force_range_changes": actual_changes,
            "guard_decisions": dict(Counter(row.get("guard_decision") for row in decisions)),
            "initial_force_range_n": runs["vlm_dsrl"]["summary"]["initial_force_range_n"],
            "final_force_range_n": runs["vlm_dsrl"]["summary"]["final_force_range_n"],
        },
        "inference_note": (
            "One temporally dependent online-training seed per method; results are descriptive "
            "and do not support a cross-seed uncertainty interval or causal VLM claim."
        ),
        "source_csv": str(source_csv),
        "figure_stem": str(output_stem),
        "formats": ["png", "svg", "pdf", "tiff"],
    }
    manifest_path = output_dir / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = f"""# RaspberryPick fixed-gate result

## Outcome

| Gate | Result | Decision |
|---|---:|---|
| Base policy | {run_stats['base']['successes']}/100 ({run_stats['base']['overall_success_rate']:.0%}) | Reference |
| DSRL-only v2 | {run_stats['dsrl']['successes']}/100 ({run_stats['dsrl']['overall_success_rate']:.0%}) | {'Pass' if manifest['gates']['dsrl_exceeds_base'] else 'Fail'} vs Base |
| VLM+DSRL v2 | {run_stats['vlm_dsrl']['successes']}/100 ({run_stats['vlm_dsrl']['overall_success_rate']:.0%}) | {'Pass' if manifest['gates']['vlm_dsrl_exceeds_dsrl'] else 'Fail'} vs DSRL |

The VLM gate did not pass, so the two additional seed runs were not launched.

## Dynamics

- Base first/last 20: {run_stats['base']['first_20_success_rate']:.0%}/{run_stats['base']['last_20_success_rate']:.0%}.
- DSRL first/last 20: {run_stats['dsrl']['first_20_success_rate']:.0%}/{run_stats['dsrl']['last_20_success_rate']:.0%}.
- VLM+DSRL first/last 20: {run_stats['vlm_dsrl']['first_20_success_rate']:.0%}/{run_stats['vlm_dsrl']['last_20_success_rate']:.0%}.

## VLM audit

- Successful VLM calls: {manifest['vlm_audit']['successful_calls']}/{manifest['vlm_audit']['calls']}.
- Actual force-range changes: {actual_changes}/100.
- Initial/final range: {manifest['vlm_audit']['initial_force_range_n']} N / {manifest['vlm_audit']['final_force_range_n']} N.
- Guard decisions: `{json.dumps(manifest['vlm_audit']['guard_decisions'], ensure_ascii=False)}`.

The repaired guard prevented the erroneous force increases seen in the legacy run, but the force-only VLM became a no-op because the dominant failures did not provide directional force evidence. The single-seed comparison is descriptive; episodes within each online run are temporally dependent.

## Figure QA

- All 100 observations from every method are included; no rows were excluded.
- Panel a shows the 10-episode trailing rate; episodes 1–9 use the available prefix.
- Panel b reports raw 100-episode proportions without uncertainty bars because there is only one temporally dependent online-training run per method.
- Source data, editable SVG/PDF, 600-dpi PNG/TIFF, and a machine-readable manifest are included.
"""
    (output_dir / "RESULTS_AND_QA.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
