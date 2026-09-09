#!/usr/bin/env python3
"""Aggregate repeated SealTwist DSRL+VLM online success trajectories."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--baseline-success-rate", type=float, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def linear_slope(values: list[float], first_episode: int) -> float:
    episodes = list(range(first_episode, first_episode + len(values)))
    mean_x = statistics.fmean(episodes)
    mean_y = statistics.fmean(values)
    denominator = sum((x - mean_x) ** 2 for x in episodes)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(episodes, values, strict=True)) / denominator


def load_run(
    path: Path,
    expected_episodes: int,
    *,
    expected_vlm_enabled: bool = True,
) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    records.sort(key=lambda item: int(item["episode_index"]))
    if len(records) != expected_episodes:
        raise ValueError(f"{path}: expected {expected_episodes} episodes, found {len(records)}")
    if [int(item["episode_index"]) for item in records] != list(range(expected_episodes)):
        raise ValueError(f"{path}: episode indices are not contiguous from zero")
    seeds = [int(item["seed"]) for item in records]
    if len(set(seeds)) != expected_episodes:
        raise ValueError(f"{path}: episode seeds are not unique")
    if any(type(item.get("success")) is not bool for item in records):
        raise ValueError(f"{path}: every episode needs a Boolean success value")

    summary_path = path.parent / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["completed_episodes"]) != expected_episodes:
        raise ValueError(f"{summary_path}: run is incomplete")
    vlm_enabled = bool(summary.get("vlm_is_codex"))
    if vlm_enabled != expected_vlm_enabled:
        expectation = "enabled" if expected_vlm_enabled else "disabled"
        raise ValueError(f"{summary_path}: expected VLM to be {expectation}")
    expected_vlm_calls = expected_episodes if expected_vlm_enabled else 0
    if int(summary.get("vlm_calls", -1)) != expected_vlm_calls:
        raise ValueError(f"{summary_path}: unexpected number of VLM attempts")
    if int(summary.get("vlm_successful_calls", -1)) != expected_vlm_calls:
        raise ValueError(f"{summary_path}: unexpected number of successful VLM responses")
    if int(summary["training_seed"]) != seeds[0]:
        raise ValueError(f"{summary_path}: training seed does not match episode records")

    outcomes = [int(item["success"]) for item in records]
    return {
        "path": path,
        "summary_path": summary_path,
        "summary": summary,
        "training_seed": seeds[0],
        "outcomes": outcomes,
    }


def rolling_success(outcomes: list[int], window: int) -> list[float]:
    return [
        sum(outcomes[max(0, index - window + 1) : index + 1])
        / min(window, index + 1)
        for index in range(len(outcomes))
    ]


def aggregate_rows(runs: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    curves = [rolling_success(run["outcomes"], window) for run in runs]
    rows: list[dict[str, Any]] = []
    for index, values in enumerate(zip(*curves, strict=True)):
        mean = statistics.fmean(values)
        sample_sd = statistics.stdev(values) if len(values) > 1 else 0.0
        row: dict[str, Any] = {
            "episode": index + 1,
            "rolling_window_requested": window,
            "rolling_window_effective": min(window, index + 1),
            "mean_rolling_success_rate": mean,
            "sample_sd_across_runs": sample_sd,
            "mean_minus_1sd_clipped": max(0.0, mean - sample_sd),
            "mean_plus_1sd_clipped": min(1.0, mean + sample_sd),
        }
        for run, curve in zip(runs, curves, strict=True):
            seed = run["training_seed"]
            row[f"seed_{seed}_success"] = run["outcomes"][index]
            row[f"seed_{seed}_rolling_success_rate"] = curve[index]
        rows.append(row)
    return rows


def write_source_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(
    rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    output_stem: Path,
    window: int,
    baseline_success_rate: float | None,
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
    mean = [float(row["mean_rolling_success_rate"]) for row in rows]
    lower = [float(row["mean_minus_1sd_clipped"]) for row in rows]
    upper = [float(row["mean_plus_1sd_clipped"]) for row in rows]
    full_start = min(window - 1, len(rows) - 1)

    fig, ax = plt.subplots(figsize=(3.50, 2.55), constrained_layout=True)
    for run in runs:
        seed = run["training_seed"]
        curve = [float(row[f"seed_{seed}_rolling_success_rate"]) for row in rows]
        ax.plot(
            episodes[full_start:],
            curve[full_start:],
            color="#9AA1A8",
            linewidth=0.7,
            alpha=0.42,
            zorder=1,
        )
    ax.fill_between(
        episodes[full_start:],
        lower[full_start:],
        upper[full_start:],
        color="#4C92C3",
        alpha=0.20,
        linewidth=0.0,
        label="Mean ± 1 SD across runs",
        zorder=2,
    )
    ax.plot(
        episodes[full_start:],
        mean[full_start:],
        color="#1769AA",
        linewidth=2.0,
        label=f"Mean trailing success (window={window})",
        zorder=3,
    )
    full_episodes = episodes[full_start:]
    full_mean = mean[full_start:]
    trend_slope = linear_slope(full_mean, full_episodes[0])
    trend_intercept = statistics.fmean(full_mean) - trend_slope * statistics.fmean(
        full_episodes
    )
    trend = [trend_intercept + trend_slope * episode for episode in full_episodes]
    ax.plot(
        full_episodes,
        trend,
        color="#0B3C5D",
        linewidth=1.15,
        linestyle=(0, (1.5, 1.5)),
        label=f"Linear trend ({trend_slope * 1000:+.1f} pp / 10 ep)",
        zorder=3,
    )
    if full_start > 0:
        ax.plot(
            episodes[: full_start + 1],
            mean[: full_start + 1],
            color="#74A9CF",
            linewidth=1.2,
            linestyle=(0, (2.0, 1.5)),
            zorder=3,
        )
    if baseline_success_rate is not None:
        ax.axhline(
            baseline_success_rate,
            color="#C44E52",
            linewidth=1.1,
            linestyle=(0, (4, 2)),
            label=f"Frozen base benchmark ({baseline_success_rate:.1%})",
            zorder=2,
        )
    ax.scatter(
        episodes[-1], mean[-1], s=18, facecolor="#1769AA", edgecolor="white", linewidth=0.5, zorder=4
    )
    ax.annotate(
        f"{mean[-1]:.0%}",
        xy=(episodes[-1], mean[-1]),
        xytext=(-5, 6 if mean[-1] <= 0.82 else -10),
        textcoords="offset points",
        ha="right",
        va="bottom" if mean[-1] <= 0.82 else "top",
        color="#1769AA",
        fontsize=6.5,
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
    ax.set_xticks([1, 20, 40, 60, 80, 100] if len(rows) == 100 else None)
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"SealTwist DSRL + guarded VLM (n={len(runs)} runs)", loc="left", fontweight="bold", pad=5)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.99, 0.025),
        frameon=False,
        fontsize=5.4,
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
    if len(args.inputs) != args.expected_runs:
        raise ValueError(f"expected {args.expected_runs} runs, received {len(args.inputs)}")
    if args.baseline_success_rate is not None and not 0.0 <= args.baseline_success_rate <= 1.0:
        raise ValueError("baseline success rate must be in [0, 1]")

    input_paths = [path.expanduser().resolve() for path in args.inputs]
    runs = [load_run(path, args.expected_episodes) for path in input_paths]
    training_seeds = [run["training_seed"] for run in runs]
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError("training seeds must be unique")
    flow_hashes = {run["summary"]["flow_checkpoint_sha256"] for run in runs}
    if len(flow_hashes) != 1:
        raise ValueError("runs do not use the same frozen Flow checkpoint")

    rows = aggregate_rows(runs, args.window)
    output_dir = args.output_dir.expanduser().resolve()
    source_csv = output_dir / f"mean_online_success_window{args.window}_source.csv"
    output_stem = output_dir / f"mean_online_success_window{args.window}"
    write_source_csv(source_csv, rows)
    draw_figure(rows, runs, output_stem, args.window, args.baseline_success_rate)

    full_mean = [
        float(row["mean_rolling_success_rate"])
        for row in rows
        if int(row["rolling_window_effective"]) == args.window
    ]
    mean_slope = linear_slope(full_mean, args.window)
    full_episodes = list(range(args.window, args.expected_episodes + 1))
    trend_intercept = statistics.fmean(full_mean) - mean_slope * statistics.fmean(
        full_episodes
    )
    per_run = []
    for run in runs:
        outcomes = run["outcomes"]
        curve = rolling_success(outcomes, args.window)[args.window - 1 :]
        per_run.append(
            {
                "training_seed": run["training_seed"],
                "successes": sum(outcomes),
                "overall_success_rate": statistics.fmean(outcomes),
                "first_20_success_rate": statistics.fmean(outcomes[:20]),
                "last_20_success_rate": statistics.fmean(outcomes[-20:]),
                "rolling_window_slope_per_10_episodes": linear_slope(curve, args.window) * 10.0,
                "vlm_calls": int(run["summary"]["vlm_calls"]),
                "vlm_successful_calls": int(run["summary"]["vlm_successful_calls"]),
                "final_force_range_n": run["summary"]["final_force_range_n"],
                "input_jsonl": str(run["path"]),
                "input_sha256": sha256(run["path"]),
                "summary_json": str(run["summary_path"]),
                "summary_sha256": sha256(run["summary_path"]),
            }
        )
    manifest = {
        "experiment": "SealTwist semantic DSRL plus guarded Codex VLM repeated online training",
        "runs": len(runs),
        "episodes_per_run": args.expected_episodes,
        "training_seeds": training_seeds,
        "rolling_window": args.window,
        "aggregation": "mean of each run's trailing success rate at the same episode index",
        "uncertainty_band": "sample standard deviation across the three training runs",
        "early_window_rule": "episodes before the requested window use the available prefix and are dashed",
        "baseline_success_rate": args.baseline_success_rate,
        "mean_overall_success_rate": statistics.fmean(item["overall_success_rate"] for item in per_run),
        "mean_first_20_success_rate": statistics.fmean(item["first_20_success_rate"] for item in per_run),
        "mean_last_20_success_rate": statistics.fmean(item["last_20_success_rate"] for item in per_run),
        "mean_rolling_window_slope_per_10_episodes": mean_slope * 10.0,
        "linear_trend_at_first_complete_window": trend_intercept + mean_slope * args.window,
        "linear_trend_at_final_episode": trend_intercept + mean_slope * args.expected_episodes,
        "complete_windows_above_baseline": (
            None
            if args.baseline_success_rate is None
            else sum(value > args.baseline_success_rate for value in full_mean)
        ),
        "complete_windows_total": len(full_mean),
        "final_mean_trailing_success_rate": full_mean[-1],
        "flow_checkpoint_sha256": next(iter(flow_hashes)),
        "per_run": per_run,
        "source_csv": str(source_csv),
        "figure_stem": str(output_stem),
        "formats": ["png", "svg", "pdf", "tiff"],
    }
    manifest_path = output_dir / "mean_online_success_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
