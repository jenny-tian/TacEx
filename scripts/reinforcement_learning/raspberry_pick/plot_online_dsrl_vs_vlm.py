#!/usr/bin/env python3
"""Plot a paired RaspberryPick DSRL-only versus VLM+DSRL online comparison."""

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
    parser.add_argument("--dsrl-input", type=Path, required=True)
    parser.add_argument("--vlm-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument(
        "--base-evaluation-json",
        type=Path,
        default=Path("exp_report/raspberry_pick/bc/evaluation/holdout/aggregate_n90.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(path: Path, expected_episodes: int, *, expect_vlm: bool) -> dict[str, Any]:
    path = path.expanduser().resolve()
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records.sort(key=lambda item: int(item["episode_index"]))
    if len(records) != expected_episodes:
        raise ValueError(f"{path}: expected {expected_episodes} episodes, found {len(records)}")
    if [int(item["episode_index"]) for item in records] != list(range(expected_episodes)):
        raise ValueError(f"{path}: episode indices must be contiguous from zero")
    seeds = [int(item["seed"]) for item in records]
    if len(set(seeds)) != expected_episodes:
        raise ValueError(f"{path}: episode seeds are not unique")
    if any(type(item.get("success")) is not bool for item in records):
        raise ValueError(f"{path}: every episode must have a Boolean success field")

    summary_path = path.parent / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["completed_episodes"]) != expected_episodes:
        raise ValueError(f"{summary_path}: completed episode count mismatch")
    if int(summary["successes"]) != sum(int(item["success"]) for item in records):
        raise ValueError(f"{summary_path}: success count mismatch")
    if bool(summary["vlm_is_codex"]) != expect_vlm:
        raise ValueError(f"{summary_path}: VLM-enabled status does not match the group")
    expected_calls = expected_episodes if expect_vlm else 0
    if int(summary["vlm_successful_calls"]) != expected_calls:
        raise ValueError(f"{summary_path}: VLM successful-call count mismatch")
    return {
        "path": path,
        "summary_path": summary_path,
        "records": records,
        "summary": summary,
        "seeds": seeds,
    }


def rolling_success(records: list[dict[str, Any]], window: int) -> list[float]:
    if window < 1:
        raise ValueError("window must be positive")
    outcomes = [int(item["success"]) for item in records]
    result: list[float] = []
    for index in range(len(outcomes)):
        start = max(0, index - window + 1)
        values = outcomes[start : index + 1]
        result.append(sum(values) / len(values))
    return result


def run_statistics(run: dict[str, Any], curve: list[float], window: int) -> dict[str, Any]:
    records = run["records"]
    successes = sum(int(item["success"]) for item in records)
    full_curve = curve[window - 1 :]
    maximum = max(full_curve)
    first_max_end = next(
        index + 1
        for index, value in enumerate(curve)
        if index >= window - 1 and value == maximum
    )
    return {
        "training_seed": int(run["summary"]["training_seed"]),
        "episodes": len(records),
        "successes": successes,
        "overall_success_rate": successes / len(records),
        "first_20_success_rate": sum(int(item["success"]) for item in records[:20]) / 20,
        "last_20_success_rate": sum(int(item["success"]) for item in records[-20:]) / 20,
        "initial_full_window_success_rate": curve[window - 1],
        "maximum_full_window_success_rate": maximum,
        "first_maximum_window_episode_range": [first_max_end - window + 1, first_max_end],
        "final_window_success_rate": curve[-1],
        "failure_counts": dict(
            Counter(item["outcome"] for item in records if not item["success"])
        ),
        "initial_force_range_n": run["summary"]["initial_force_range_n"],
        "final_force_range_n": run["summary"]["final_force_range_n"],
        "sac_optimizer_updates": int(run["summary"]["sac_optimizer_updates"]),
        "replay_transitions": int(run["summary"]["replay_transitions"]),
        "vlm_calls": int(run["summary"]["vlm_calls"]),
        "vlm_successful_calls": int(run["summary"]["vlm_successful_calls"]),
        "input_jsonl": str(run["path"]),
        "input_sha256": sha256(run["path"]),
        "summary_json": str(run["summary_path"]),
        "summary_sha256": sha256(run["summary_path"]),
    }


def build_rows(
    dsrl: dict[str, Any],
    vlm: dict[str, Any],
    dsrl_curve: list[float],
    vlm_curve: list[float],
    window: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (dsrl_record, vlm_record) in enumerate(
        zip(dsrl["records"], vlm["records"], strict=True)
    ):
        dsrl_range = list(map(float, dsrl_record["attempted_force_range_n"]))
        vlm_range = list(map(float, vlm_record["attempted_force_range_n"]))
        rows.append(
            {
                "episode": index + 1,
                "seed": int(dsrl_record["seed"]),
                "rolling_window_requested": window,
                "rolling_window_effective": min(window, index + 1),
                "dsrl_success": int(dsrl_record["success"]),
                "dsrl_outcome": str(dsrl_record["outcome"]),
                "dsrl_rolling_success_rate": dsrl_curve[index],
                "dsrl_force_low_n": dsrl_range[0],
                "dsrl_force_high_n": dsrl_range[1],
                "vlm_dsrl_success": int(vlm_record["success"]),
                "vlm_dsrl_outcome": str(vlm_record["outcome"]),
                "vlm_dsrl_rolling_success_rate": vlm_curve[index],
                "vlm_dsrl_force_low_n": vlm_range[0],
                "vlm_dsrl_force_high_n": vlm_range[1],
                "vlm_minus_dsrl_rolling_success_rate": vlm_curve[index]
                - dsrl_curve[index],
            }
        )
    return rows


def write_source_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(
    rows: list[dict[str, Any]],
    output_stem: Path,
    *,
    window: int,
    training_seed: int,
    dsrl_overall: float,
    vlm_overall: float,
    base_rate: float,
    base_episodes: int,
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
    episodes = [int(row["episode"]) for row in rows]
    dsrl_curve = [float(row["dsrl_rolling_success_rate"]) for row in rows]
    vlm_curve = [float(row["vlm_dsrl_rolling_success_rate"]) for row in rows]
    full_start = min(window - 1, len(rows) - 1)

    fig, ax = plt.subplots(figsize=(3.50, 2.75), constrained_layout=True)
    if full_start > 0:
        ax.plot(
            episodes[: full_start + 1],
            dsrl_curve[: full_start + 1],
            color="#E3A66E",
            linewidth=1.1,
            linestyle=(0, (2, 1.5)),
            zorder=2,
        )
        ax.plot(
            episodes[: full_start + 1],
            vlm_curve[: full_start + 1],
            color="#74A9CF",
            linewidth=1.2,
            linestyle=(0, (2, 1.5)),
            zorder=2,
        )
    ax.plot(
        episodes[full_start:],
        dsrl_curve[full_start:],
        color="#D17C2F",
        linewidth=1.65,
        linestyle=(0, (5, 1.5, 1, 1.5)),
        label=f"DSRL only (overall {dsrl_overall:.0%})",
        zorder=3,
    )
    ax.plot(
        episodes[full_start:],
        vlm_curve[full_start:],
        color="#1769AA",
        linewidth=1.9,
        label=f"VLM + DSRL (overall {vlm_overall:.0%})",
        zorder=4,
    )
    ax.axhline(
        base_rate,
        color="#747B83",
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        label=f"Frozen BC ({base_rate:.1%}, n={base_episodes})",
        zorder=1,
    )
    ax.scatter(
        episodes[-1], dsrl_curve[-1], s=18, color="#D17C2F", edgecolor="white", linewidth=0.5, zorder=5
    )
    ax.scatter(
        episodes[-1], vlm_curve[-1], s=18, color="#1769AA", edgecolor="white", linewidth=0.5, zorder=5
    )
    close_endpoints = abs(dsrl_curve[-1] - vlm_curve[-1]) < 0.11
    ax.annotate(
        f"{dsrl_curve[-1]:.0%}",
        xy=(episodes[-1], dsrl_curve[-1]),
        xytext=(-5, -9 if close_endpoints else 6),
        textcoords="offset points",
        ha="right",
        va="top" if close_endpoints else "bottom",
        color="#9B581C",
        fontsize=6.3,
        fontweight="bold",
    )
    ax.annotate(
        f"{vlm_curve[-1]:.0%}",
        xy=(episodes[-1], vlm_curve[-1]),
        xytext=(-5, 7),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="#1769AA",
        fontsize=6.3,
        fontweight="bold",
    )
    ax.axvline(window, color="#CDD1D5", linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
    ax.set_xlim(1, len(rows) + 1.5)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Online post-training episode")
    ax.set_ylabel("Trailing success rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    if len(rows) == 100:
        ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("RaspberryPick online success", loc="left", fontweight="bold", pad=5)
    ax.legend(loc="lower left", fontsize=5.6, handlelength=2.2, borderaxespad=0.3)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
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
    dsrl = load_run(args.dsrl_input, args.expected_episodes, expect_vlm=False)
    vlm = load_run(args.vlm_input, args.expected_episodes, expect_vlm=True)
    if dsrl["seeds"] != vlm["seeds"]:
        raise ValueError("DSRL-only and VLM+DSRL runs must use identical episode seeds")

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
        if dsrl["summary"].get(field) != vlm["summary"].get(field):
            raise ValueError(f"run mismatch in {field}")

    dsrl_curve = rolling_success(dsrl["records"], args.window)
    vlm_curve = rolling_success(vlm["records"], args.window)
    dsrl_stats = run_statistics(dsrl, dsrl_curve, args.window)
    vlm_stats = run_statistics(vlm, vlm_curve, args.window)
    rows = build_rows(dsrl, vlm, dsrl_curve, vlm_curve, args.window)
    output_dir = args.output_dir.expanduser().resolve()
    source_csv = output_dir / f"dsrl_vs_vlm_online_success_window{args.window}_source.csv"
    output_stem = output_dir / f"dsrl_vs_vlm_online_success_window{args.window}"
    write_source_csv(source_csv, rows)

    base_path = args.base_evaluation_json.expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base_rate = float(base["success_rate"])
    base_episodes = int(base["episodes"])
    draw_figure(
        rows,
        output_stem,
        window=args.window,
        training_seed=dsrl_stats["training_seed"],
        dsrl_overall=dsrl_stats["overall_success_rate"],
        vlm_overall=vlm_stats["overall_success_rate"],
        base_rate=base_rate,
        base_episodes=base_episodes,
    )

    both_success = sum(
        int(dsrl_item["success"] and vlm_item["success"])
        for dsrl_item, vlm_item in zip(dsrl["records"], vlm["records"], strict=True)
    )
    vlm_only = sum(
        int((not dsrl_item["success"]) and vlm_item["success"])
        for dsrl_item, vlm_item in zip(dsrl["records"], vlm["records"], strict=True)
    )
    dsrl_only = sum(
        int(dsrl_item["success"] and (not vlm_item["success"]))
        for dsrl_item, vlm_item in zip(dsrl["records"], vlm["records"], strict=True)
    )
    full_rows = rows[args.window - 1 :]
    manifest = {
        "experiment": "RaspberryPick paired DSRL-only versus online Codex VLM+DSRL",
        "figure_contract": {
            "core_conclusion": (
                "The paired curves compare online success dynamics with and without "
                "episodic VLM force adaptation while holding the DSRL and task protocol fixed."
            ),
            "archetype": "single-panel quantitative comparison",
            "backend": "Python/matplotlib",
            "final_size_mm": [88.9, 69.85],
        },
        "paired_training_seed": dsrl_stats["training_seed"],
        "paired_episode_seed_range": [dsrl["seeds"][0], dsrl["seeds"][-1]],
        "episodes_per_run": args.expected_episodes,
        "rolling_window": args.window,
        "early_window_rule": "episodes before the requested window use the available prefix",
        "uncertainty": "none; one paired training seed per method",
        "matched_fields": list(matched_fields),
        "intended_difference": (
            "VLM+DSRL adapts the force interval after each episode; DSRL-only keeps "
            "the initial force interval fixed."
        ),
        "dsrl_only": dsrl_stats,
        "vlm_dsrl": vlm_stats,
        "comparison": {
            "overall_success_rate_difference_vlm_minus_dsrl": (
                vlm_stats["overall_success_rate"] - dsrl_stats["overall_success_rate"]
            ),
            "final_window_difference_vlm_minus_dsrl": (
                vlm_stats["final_window_success_rate"]
                - dsrl_stats["final_window_success_rate"]
            ),
            "complete_windows_vlm_above_dsrl": sum(
                float(row["vlm_minus_dsrl_rolling_success_rate"]) > 0.0
                for row in full_rows
            ),
            "complete_windows_equal": sum(
                float(row["vlm_minus_dsrl_rolling_success_rate"]) == 0.0
                for row in full_rows
            ),
            "complete_windows_dsrl_above_vlm": sum(
                float(row["vlm_minus_dsrl_rolling_success_rate"]) < 0.0
                for row in full_rows
            ),
            "paired_episode_outcomes": {
                "both_success": both_success,
                "vlm_only_success": vlm_only,
                "dsrl_only_success": dsrl_only,
                "both_fail": args.expected_episodes - both_success - vlm_only - dsrl_only,
            },
            "inference_note": (
                "Descriptive paired-seed comparison only; one adaptive training run per "
                "method does not support a cross-seed confidence interval or significance test."
            ),
        },
        "frozen_bc_reference": {
            "successes": int(base["successes"]),
            "episodes": base_episodes,
            "success_rate": base_rate,
            "input_json": str(base_path),
            "input_sha256": sha256(base_path),
        },
        "source_csv": str(source_csv),
        "figure_stem": str(output_stem),
        "formats": ["png", "svg", "pdf", "tiff"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
