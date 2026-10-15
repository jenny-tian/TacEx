#!/usr/bin/env python3
"""Audit and plot matched RaspberryPick Residual RL, AWR, and DPPO runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


METHODS = ("residual_rl", "awr", "dppo")
DISPLAY_NAMES = {
    "residual_rl": "Residual RL",
    "awr": "AWR",
    "dppo": "DPPO",
}
COLORS = {
    "residual_rl": "#5B4B8A",
    "awr": "#2E8B57",
    "dppo": "#C44E52",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("exp_report/raspberry_0909/scenario_revision_zero_pull"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(970000, 980000, 990000))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--window", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rolling_success(outcomes: list[int], window: int) -> list[float]:
    return [
        statistics.fmean(outcomes[max(0, index - window + 1) : index + 1])
        for index in range(len(outcomes))
    ]


def load_run(
    report_root: Path, method: str, seed: int, episodes: int
) -> dict[str, Any]:
    run_root = report_root / "baseline_comparison_n100" / method / f"seed{seed}_n{episodes}"
    episode_path = run_root / "training_episodes.jsonl"
    summary_path = run_root / "training_summary.json"
    checkpoint_path = run_root / "checkpoints/final.pt"
    for path in (episode_path, summary_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in episode_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: int(row["episode_index"]))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(rows) != episodes or int(summary.get("completed_episodes", -1)) != episodes:
        raise ValueError(f"incomplete run: {run_root}")
    if [int(row["episode_index"]) for row in rows] != list(range(episodes)):
        raise ValueError(f"non-contiguous episode indices: {run_root}")
    if [int(row["seed"]) for row in rows] != list(range(seed, seed + episodes)):
        raise ValueError(f"non-contiguous episode seeds: {run_root}")
    if int(summary.get("training_seed", -1)) != seed:
        raise ValueError(f"training seed mismatch: {run_root}")
    if summary.get("online_algorithm") != method:
        raise ValueError(f"online algorithm mismatch: {run_root}")
    if summary.get("force_range_source") != "fixed":
        raise ValueError(f"force source mismatch: {run_root}")
    if summary.get("controller_mode") != "single_stage":
        raise ValueError(f"controller mismatch: {run_root}")
    if int(summary.get("vlm_calls", -1)) != 0:
        raise ValueError(f"baseline called the VLM: {run_root}")
    outcomes = [int(bool(row["success"])) for row in rows]
    return {
        "method": method,
        "seed": seed,
        "root": run_root.resolve(),
        "episode_path": episode_path.resolve(),
        "summary_path": summary_path.resolve(),
        "summary": summary,
        "rows": rows,
        "outcomes": outcomes,
    }


def run_metrics(run: dict[str, Any], window: int) -> dict[str, Any]:
    outcomes = run["outcomes"]
    curve = rolling_success(outcomes, window)
    failures = Counter(
        str(row["outcome"]) for row in run["rows"] if not bool(row["success"])
    )
    return {
        "training_seed": run["seed"],
        "episodes": len(outcomes),
        "successes": sum(outcomes),
        "success_rate": statistics.fmean(outcomes),
        "first_20_success_rate": statistics.fmean(outcomes[:20]),
        "last_20_success_rate": statistics.fmean(outcomes[-20:]),
        "final_rolling_success_rate": curve[-1],
        "optimizer_updates": int(run["summary"].get("optimizer_updates", 0)),
        "replay_transitions": int(run["summary"].get("replay_transitions", 0)),
        "failure_counts": dict(failures),
        "episode_jsonl": str(run["episode_path"]),
        "episode_jsonl_sha256": sha256(run["episode_path"]),
        "summary_json": str(run["summary_path"]),
        "summary_json_sha256": sha256(run["summary_path"]),
    }


def group_metrics(runs: list[dict[str, Any]], window: int) -> dict[str, Any]:
    per_run = [run_metrics(run, window) for run in runs]
    curves = np.asarray(
        [rolling_success(run["outcomes"], window) for run in runs], dtype=np.float64
    )

    def mean_sd(field: str) -> tuple[float, float]:
        values = [float(run[field]) for run in per_run]
        return statistics.fmean(values), statistics.stdev(values)

    success_mean, success_sd = mean_sd("success_rate")
    first_mean, first_sd = mean_sd("first_20_success_rate")
    last_mean, last_sd = mean_sd("last_20_success_rate")
    return {
        "success_rate_mean": success_mean,
        "success_rate_sample_sd": success_sd,
        "first_20_success_rate_mean": first_mean,
        "first_20_success_rate_sample_sd": first_sd,
        "last_20_success_rate_mean": last_mean,
        "last_20_success_rate_sample_sd": last_sd,
        "final_mean_rolling_success_rate": float(curves.mean(axis=0)[-1]),
        "curve_mean": curves.mean(axis=0).tolist(),
        "curve_sample_sd": curves.std(axis=0, ddof=1).tolist(),
        "per_run": per_run,
    }


def write_source_csv(
    path: Path, runs_by_method: dict[str, list[dict[str, Any]]], window: int
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "method",
                "display_name",
                "training_seed",
                "episode",
                "success",
                f"rolling_success_window{window}",
            ),
        )
        writer.writeheader()
        for method in METHODS:
            for run in runs_by_method[method]:
                curve = rolling_success(run["outcomes"], window)
                for index, (success, rolling) in enumerate(
                    zip(run["outcomes"], curve, strict=True), start=1
                ):
                    writer.writerow(
                        {
                            "method": method,
                            "display_name": DISPLAY_NAMES[method],
                            "training_seed": run["seed"],
                            "episode": index,
                            "success": success,
                            f"rolling_success_window{window}": f"{rolling:.8f}",
                        }
                    )


def plot_curves(path_stem: Path, metrics: dict[str, Any], episodes: int) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    x = np.arange(1, episodes + 1)
    for method in METHODS:
        mean = np.asarray(metrics[method]["curve_mean"])
        sd = np.asarray(metrics[method]["curve_sample_sd"])
        color = COLORS[method]
        ax.fill_between(
            x,
            np.clip(mean - sd, 0.0, 1.0),
            np.clip(mean + sd, 0.0, 1.0),
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(x, mean, color=color, linewidth=2.0, label=DISPLAY_NAMES[method])
    ax.axvspan(81, 100, color="#E8E8E8", alpha=0.45, zorder=0)
    ax.text(90.5, 0.965, "last 20", ha="center", va="top", color="#666666", fontsize=7)
    ax.set_xlim(1, episodes)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Online training episode")
    ax.set_ylabel("Success rate (trailing 10 episodes)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper left", ncol=3, handlelength=2.2, columnspacing=1.2)
    ax.set_title("Matched RaspberryPick online-learning comparison", loc="left", pad=8)
    fig.savefig(path_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        path_stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def write_report(path: Path, metrics: dict[str, Any], seeds: list[int]) -> None:
    lines = [
        "# RaspberryPick 三方法在线对比实验",
        "",
        (
            "Residual RL、AWR 和 DPPO 使用同一冻结 BC、固定 [2.2, 2.8] N、"
            "single-stage controller、zero open-loop pull，并在相同的 3 个训练 seed 上"
            "各运行 100 个 online episode。曲线是 trailing-10 成功率的 seed 均值，"
            "阴影为 ±1 sample SD；独立重复单位是训练 seed（n=3）。"
        ),
        "",
        "| 方法 | 总成功率（均值 ± SD） | 前 20 | 后 20 | 最终滑窗-10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = metrics[method]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | "
            f"{item['success_rate_mean']:.1%} ± {item['success_rate_sample_sd']:.1%} | "
            f"{item['first_20_success_rate_mean']:.1%} | "
            f"{item['last_20_success_rate_mean']:.1%} | "
            f"{item['final_mean_rolling_success_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 各 seed",
            "",
            "| 方法 | Seed | 成功率 | 后 20 | Optimizer updates |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        for run in metrics[method]["per_run"]:
            lines.append(
                f"| {DISPLAY_NAMES[method]} | {run['training_seed']} | "
                f"{run['success_rate']:.1%} | {run['last_20_success_rate']:.1%} | "
                f"{run['optimizer_updates']} |"
            )
    lines.extend(
        [
            "",
            "仅作描述性比较，不把 episode 当作独立重复，也不作三方法显著性声明。",
            "",
            f"配对训练 seeds：{', '.join(map(str, seeds))}。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.window < 1:
        raise ValueError("episodes and window must be positive")
    report_root = args.report_root.resolve()
    output_dir = report_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_by_method = {
        method: [load_run(report_root, method, seed, args.episodes) for seed in args.seeds]
        for method in METHODS
    }
    metrics = {
        method: group_metrics(runs_by_method[method], args.window) for method in METHODS
    }
    manifest = {
        "experiment": "raspberry_0909_zero_pull_matched_online_baselines",
        "methods": list(METHODS),
        "training_seeds": list(args.seeds),
        "episodes_per_seed": args.episodes,
        "rolling_window": args.window,
        "independent_unit": "training_seed",
        "uncertainty": "sample standard deviation across three paired training seeds",
        "matched_contract": {
            "frozen_bc": True,
            "initial_force_range_n": [2.2, 2.8],
            "controller_mode": "single_stage",
            "pull_z_offset_m": 0.0,
            "residual": "contact-gated 1-D closing-axis residual, maximum 0.006 m",
        },
        "metrics": metrics,
    }
    manifest_path = output_dir / "baseline_online_metrics.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    source_path = output_dir / "baseline_online_success_window10_source.csv"
    write_source_csv(source_path, runs_by_method, args.window)
    figure_stem = output_dir / "baseline_online_success_window10"
    plot_curves(figure_stem, metrics, args.episodes)
    write_report(output_dir / "BASELINE_RESULTS.md", metrics, list(args.seeds))
    qa = {
        "all_nine_runs_complete": True,
        "episode_rows": len(METHODS) * len(args.seeds) * args.episodes,
        "source_csv_sha256": sha256(source_path),
        "metrics_json_sha256": sha256(manifest_path),
        "figure_contract": {
            "core_conclusion": (
                "Compare the learning dynamics of three online residual algorithms under "
                "one matched RaspberryPick protocol without pooling seeds."
            ),
            "evidence": "mean trailing-10 success curves with sample SD across n=3 seeds",
            "archetype": "quantitative grid (single comparison panel)",
            "exports": ["svg", "pdf", "png_300dpi", "tiff_600dpi_lzw"],
        },
    }
    (output_dir / "BASELINE_QA.md").write_text(
        "# Baseline comparison QA\n\n```json\n"
        + json.dumps(qa, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
