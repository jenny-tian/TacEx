#!/usr/bin/env python3
"""Compare repeated SealTwist DSRL and DSRL+VLM online success curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from plot_online_success_repeats import linear_slope, load_run, rolling_success, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsrl-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--vlm-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument(
        "--base-evaluation-json",
        type=Path,
        help="Optional frozen-base evaluation report with success_rate and episodes fields.",
    )
    return parser.parse_args()


def group_curves(runs: list[dict[str, Any]], window: int) -> list[list[float]]:
    return [rolling_success(run["outcomes"], window) for run in runs]


def group_series(curves: list[list[float]]) -> tuple[list[float], list[float]]:
    mean: list[float] = []
    sd: list[float] = []
    for values in zip(*curves, strict=True):
        mean.append(statistics.fmean(values))
        sd.append(statistics.stdev(values) if len(values) > 1 else 0.0)
    return mean, sd


def run_statistics(run: dict[str, Any], window: int) -> dict[str, Any]:
    outcomes = run["outcomes"]
    curve = rolling_success(outcomes, window)[window - 1 :]
    return {
        "training_seed": run["training_seed"],
        "successes": sum(outcomes),
        "overall_success_rate": statistics.fmean(outcomes),
        "first_20_success_rate": statistics.fmean(outcomes[:20]),
        "last_20_success_rate": statistics.fmean(outcomes[-20:]),
        "rolling_window_slope_per_10_episodes": linear_slope(curve, window) * 10.0,
        "final_rolling_success_rate": curve[-1],
        "vlm_calls": int(run["summary"]["vlm_calls"]),
        "vlm_successful_calls": int(run["summary"]["vlm_successful_calls"]),
        "final_force_range_n": run["summary"]["final_force_range_n"],
        "input_jsonl": str(run["path"]),
        "input_sha256": sha256(run["path"]),
        "summary_json": str(run["summary_path"]),
        "summary_sha256": sha256(run["summary_path"]),
    }


def aggregate_statistics(
    runs: list[dict[str, Any]],
    curves: list[list[float]],
    window: int,
) -> dict[str, Any]:
    per_run = [run_statistics(run, window) for run in runs]
    mean_curve, _ = group_series(curves)
    complete_mean = mean_curve[window - 1 :]
    slope = linear_slope(complete_mean, window)
    episodes = list(range(window, window + len(complete_mean)))
    intercept = statistics.fmean(complete_mean) - slope * statistics.fmean(episodes)
    return {
        "mean_overall_success_rate": statistics.fmean(item["overall_success_rate"] for item in per_run),
        "sample_sd_overall_success_rate": (
            statistics.stdev(item["overall_success_rate"] for item in per_run)
            if len(per_run) > 1
            else 0.0
        ),
        "mean_first_20_success_rate": statistics.fmean(item["first_20_success_rate"] for item in per_run),
        "mean_last_20_success_rate": statistics.fmean(item["last_20_success_rate"] for item in per_run),
        "mean_rolling_window_slope_per_10_episodes": slope * 10.0,
        "mean_complete_window_success_rate": statistics.fmean(complete_mean),
        "final_mean_rolling_success_rate": complete_mean[-1],
        "linear_trend_at_first_complete_window": intercept + slope * episodes[0],
        "linear_trend_at_final_episode": intercept + slope * episodes[-1],
        "per_run": per_run,
    }


def build_rows(
    dsrl_runs: list[dict[str, Any]],
    vlm_runs: list[dict[str, Any]],
    window: int,
) -> list[dict[str, Any]]:
    dsrl_curves = group_curves(dsrl_runs, window)
    vlm_curves = group_curves(vlm_runs, window)
    dsrl_mean, dsrl_sd = group_series(dsrl_curves)
    vlm_mean, vlm_sd = group_series(vlm_curves)
    rows: list[dict[str, Any]] = []
    for index in range(len(dsrl_mean)):
        row: dict[str, Any] = {
            "episode": index + 1,
            "rolling_window_requested": window,
            "rolling_window_effective": min(window, index + 1),
            "dsrl_mean_rolling_success_rate": dsrl_mean[index],
            "dsrl_sample_sd_across_runs": dsrl_sd[index],
            "dsrl_mean_minus_1sd_clipped": max(0.0, dsrl_mean[index] - dsrl_sd[index]),
            "dsrl_mean_plus_1sd_clipped": min(1.0, dsrl_mean[index] + dsrl_sd[index]),
            "vlm_mean_rolling_success_rate": vlm_mean[index],
            "vlm_sample_sd_across_runs": vlm_sd[index],
            "vlm_mean_minus_1sd_clipped": max(0.0, vlm_mean[index] - vlm_sd[index]),
            "vlm_mean_plus_1sd_clipped": min(1.0, vlm_mean[index] + vlm_sd[index]),
            "vlm_minus_dsrl_mean_rolling_success_rate": vlm_mean[index] - dsrl_mean[index],
        }
        for run, curve in zip(dsrl_runs, dsrl_curves, strict=True):
            seed = run["training_seed"]
            row[f"dsrl_seed_{seed}_success"] = run["outcomes"][index]
            row[f"dsrl_seed_{seed}_rolling_success_rate"] = curve[index]
        for run, curve in zip(vlm_runs, vlm_curves, strict=True):
            seed = run["training_seed"]
            row[f"vlm_seed_{seed}_success"] = run["outcomes"][index]
            row[f"vlm_seed_{seed}_rolling_success_rate"] = curve[index]
        rows.append(row)
    return rows


def write_source_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trend(values: list[float], episodes: list[int]) -> tuple[list[float], float]:
    slope = linear_slope(values, episodes[0])
    intercept = statistics.fmean(values) - slope * statistics.fmean(episodes)
    return [intercept + slope * episode for episode in episodes], slope


def draw_figure(
    rows: list[dict[str, Any]],
    output_stem: Path,
    window: int,
    runs: int,
    base_success_rate: float | None,
    base_episodes: int | None,
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
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    episodes = [int(row["episode"]) for row in rows]
    full_start = min(window - 1, len(rows) - 1)
    full_episodes = episodes[full_start:]
    dsrl_mean = [float(row["dsrl_mean_rolling_success_rate"]) for row in rows]
    vlm_mean = [float(row["vlm_mean_rolling_success_rate"]) for row in rows]
    dsrl_lower = [float(row["dsrl_mean_minus_1sd_clipped"]) for row in rows]
    dsrl_upper = [float(row["dsrl_mean_plus_1sd_clipped"]) for row in rows]
    vlm_lower = [float(row["vlm_mean_minus_1sd_clipped"]) for row in rows]
    vlm_upper = [float(row["vlm_mean_plus_1sd_clipped"]) for row in rows]
    dsrl_trend, dsrl_slope = trend(dsrl_mean[full_start:], full_episodes)
    vlm_trend, vlm_slope = trend(vlm_mean[full_start:], full_episodes)

    fig, ax = plt.subplots(figsize=(3.50, 2.55), constrained_layout=True)
    ax.fill_between(
        full_episodes,
        dsrl_lower[full_start:],
        dsrl_upper[full_start:],
        color="#E39A55",
        alpha=0.15,
        linewidth=0.0,
        zorder=1,
    )
    ax.fill_between(
        full_episodes,
        vlm_lower[full_start:],
        vlm_upper[full_start:],
        color="#4C92C3",
        alpha=0.17,
        linewidth=0.0,
        zorder=1,
    )
    ax.plot(
        full_episodes,
        dsrl_mean[full_start:],
        color="#D17C2F",
        linewidth=1.65,
        label="DSRL only (mean ± 1 SD)",
        zorder=3,
    )
    ax.plot(
        full_episodes,
        vlm_mean[full_start:],
        color="#1769AA",
        linewidth=1.9,
        label="DSRL + guarded VLM (mean ± 1 SD)",
        zorder=4,
    )
    ax.plot(
        full_episodes,
        dsrl_trend,
        color="#9B581C",
        linewidth=1.0,
        linestyle=(0, (1.5, 1.5)),
        label=f"DSRL trend ({dsrl_slope * 1000:+.1f} pp / 10 ep)",
        zorder=2,
    )
    if base_success_rate is not None:
        sample_label = "" if base_episodes is None else f", n={base_episodes}"
        ax.axhline(
            base_success_rate,
            color="#6D747C",
            linewidth=1.0,
            linestyle=(0, (4, 2)),
            label=f"Frozen base ({base_success_rate:.0%}{sample_label})",
            zorder=1,
        )
    ax.plot(
        full_episodes,
        vlm_trend,
        color="#0B3C5D",
        linewidth=1.0,
        linestyle=(0, (1.5, 1.5)),
        label=f"DSRL + VLM trend ({vlm_slope * 1000:+.1f} pp / 10 ep)",
        zorder=2,
    )
    if full_start > 0:
        ax.plot(
            episodes[: full_start + 1],
            dsrl_mean[: full_start + 1],
            color="#E3A66E",
            linewidth=1.0,
            linestyle=(0, (2, 1.5)),
            zorder=2,
        )
        ax.plot(
            episodes[: full_start + 1],
            vlm_mean[: full_start + 1],
            color="#74A9CF",
            linewidth=1.1,
            linestyle=(0, (2, 1.5)),
            zorder=2,
        )
    ax.scatter(episodes[-1], dsrl_mean[-1], s=16, color="#D17C2F", edgecolor="white", linewidth=0.5, zorder=5)
    ax.scatter(episodes[-1], vlm_mean[-1], s=18, color="#1769AA", edgecolor="white", linewidth=0.5, zorder=5)
    ax.annotate(
        f"{dsrl_mean[-1]:.0%}",
        xy=(episodes[-1], dsrl_mean[-1]),
        xytext=(-5, -8),
        textcoords="offset points",
        ha="right",
        va="top",
        color="#9B581C",
        fontsize=6.3,
        fontweight="bold",
    )
    ax.annotate(
        f"{vlm_mean[-1]:.0%}",
        xy=(episodes[-1], vlm_mean[-1]),
        xytext=(-5, 7),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="#1769AA",
        fontsize=6.3,
        fontweight="bold",
    )
    ax.axvline(window, color="#CDD1D5", linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
    ax.text(
        window + 1.0,
        0.055,
        "full window",
        color="#737A82",
        fontsize=5.5,
        rotation=90,
        rotation_mode="anchor",
        va="bottom",
    )
    ax.set_xlim(1, len(rows))
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Online training episode")
    ax.set_ylabel("Success rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    if len(rows) == 100:
        ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        f"SealTwist online success (n={runs} paired seeds)",
        loc="left",
        fontweight="bold",
        pad=5,
    )
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.99, 0.025),
        frameon=False,
        fontsize=5.1,
        handlelength=2.2,
        borderaxespad=0.0,
    )

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
    if len(args.dsrl_inputs) != args.expected_runs or len(args.vlm_inputs) != args.expected_runs:
        raise ValueError(f"each group must contain exactly {args.expected_runs} runs")

    dsrl_runs = [
        load_run(path.expanduser().resolve(), args.expected_episodes, expected_vlm_enabled=False)
        for path in args.dsrl_inputs
    ]
    vlm_runs = [
        load_run(path.expanduser().resolve(), args.expected_episodes, expected_vlm_enabled=True)
        for path in args.vlm_inputs
    ]
    dsrl_runs.sort(key=lambda run: run["training_seed"])
    vlm_runs.sort(key=lambda run: run["training_seed"])
    dsrl_seeds = [run["training_seed"] for run in dsrl_runs]
    vlm_seeds = [run["training_seed"] for run in vlm_runs]
    if dsrl_seeds != vlm_seeds or len(set(dsrl_seeds)) != args.expected_runs:
        raise ValueError("DSRL and VLM groups must have the same unique training seeds")
    flow_hashes = {
        run["summary"]["flow_checkpoint_sha256"] for run in [*dsrl_runs, *vlm_runs]
    }
    if len(flow_hashes) != 1:
        raise ValueError("all runs must use the same frozen Flow checkpoint")

    rows = build_rows(dsrl_runs, vlm_runs, args.window)
    dsrl_curves = group_curves(dsrl_runs, args.window)
    vlm_curves = group_curves(vlm_runs, args.window)
    dsrl_stats = aggregate_statistics(dsrl_runs, dsrl_curves, args.window)
    vlm_stats = aggregate_statistics(vlm_runs, vlm_curves, args.window)
    full_rows = [row for row in rows if row["rolling_window_effective"] == args.window]
    paired_seed_differences = [
        {
            "training_seed": seed,
            "overall_success_rate_difference_vlm_minus_dsrl": (
                vlm["overall_success_rate"] - dsrl["overall_success_rate"]
            ),
            "last_20_success_rate_difference_vlm_minus_dsrl": (
                vlm["last_20_success_rate"] - dsrl["last_20_success_rate"]
            ),
            "rolling_slope_difference_per_10_episodes_vlm_minus_dsrl": (
                vlm["rolling_window_slope_per_10_episodes"]
                - dsrl["rolling_window_slope_per_10_episodes"]
            ),
        }
        for seed, dsrl, vlm in zip(
            dsrl_seeds, dsrl_stats["per_run"], vlm_stats["per_run"], strict=True
        )
    ]
    comparison = {
        "mean_overall_success_rate_difference_vlm_minus_dsrl": (
            vlm_stats["mean_overall_success_rate"] - dsrl_stats["mean_overall_success_rate"]
        ),
        "mean_first_20_success_rate_difference_vlm_minus_dsrl": (
            vlm_stats["mean_first_20_success_rate"] - dsrl_stats["mean_first_20_success_rate"]
        ),
        "mean_last_20_success_rate_difference_vlm_minus_dsrl": (
            vlm_stats["mean_last_20_success_rate"] - dsrl_stats["mean_last_20_success_rate"]
        ),
        "rolling_slope_difference_per_10_episodes_vlm_minus_dsrl": (
            vlm_stats["mean_rolling_window_slope_per_10_episodes"]
            - dsrl_stats["mean_rolling_window_slope_per_10_episodes"]
        ),
        "final_mean_rolling_success_rate_difference_vlm_minus_dsrl": (
            vlm_stats["final_mean_rolling_success_rate"]
            - dsrl_stats["final_mean_rolling_success_rate"]
        ),
        "mean_complete_window_difference_vlm_minus_dsrl": statistics.fmean(
            float(row["vlm_minus_dsrl_mean_rolling_success_rate"]) for row in full_rows
        ),
        "complete_windows_vlm_mean_above_dsrl_mean": sum(
            float(row["vlm_minus_dsrl_mean_rolling_success_rate"]) > 0.0 for row in full_rows
        ),
        "complete_windows_equal": sum(
            float(row["vlm_minus_dsrl_mean_rolling_success_rate"]) == 0.0 for row in full_rows
        ),
        "complete_windows_total": len(full_rows),
        "paired_seed_differences": paired_seed_differences,
        "inference_note": (
            f"Descriptive comparison only: n={args.expected_runs} paired training "
            "seed(s) is too small for a reliable cross-seed significance test."
        ),
    }

    output_dir = args.output_dir.expanduser().resolve()
    base_path = (
        None
        if args.base_evaluation_json is None
        else args.base_evaluation_json.expanduser().resolve()
    )
    base_success_rate: float | None = None
    base_episodes: int | None = None
    if base_path is not None:
        base_report = json.loads(base_path.read_text(encoding="utf-8"))
        base_success_rate = float(base_report["success_rate"])
        base_episodes = int(base_report["episodes"])
        if not 0.0 <= base_success_rate <= 1.0 or base_episodes < 1:
            raise ValueError("base evaluation success rate/episode count is invalid")
    source_csv = output_dir / f"dsrl_vs_vlm_window{args.window}_source.csv"
    output_stem = output_dir / f"dsrl_vs_vlm_window{args.window}"
    write_source_csv(source_csv, rows)
    draw_figure(
        rows,
        output_stem,
        args.window,
        args.expected_runs,
        base_success_rate,
        base_episodes,
    )
    manifest = {
        "experiment": "SealTwist pure DSRL versus DSRL plus guarded Codex VLM",
        "paired_training_seeds": dsrl_seeds,
        "runs_per_group": args.expected_runs,
        "episodes_per_run": args.expected_episodes,
        "rolling_window": args.window,
        "aggregation": "mean per-run trailing success at the same online episode",
        "uncertainty_band": "sample standard deviation across paired training seeds",
        "flow_checkpoint_sha256": next(iter(flow_hashes)),
        "dsrl_only": dsrl_stats,
        "dsrl_plus_guarded_vlm": vlm_stats,
        "comparison": comparison,
        "frozen_base_evaluation": (
            None
            if base_path is None
            else {
                "path": str(base_path),
                "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
                "episodes": base_episodes,
                "success_rate": base_success_rate,
            }
        ),
        "source_csv": str(source_csv),
        "figure_stem": str(output_stem),
        "formats": ["png", "svg", "pdf", "tiff"],
    }
    manifest_path = output_dir / "dsrl_vs_vlm_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
