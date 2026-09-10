#!/usr/bin/env python3
"""Plot paired online success-rate curves for one revised formal seed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_ROOT = ROOT / "exp_report/raspberry_0909/scenario_revision_zero_pull"
METHODS = (
    ("dsrl_only", "DSRL-only", "#777777"),
    ("vlm_two_stage_dsrl", "VLM + two-stage + DSRL", "#2F6EA5"),
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--seed", type=int, default=970_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--window", type=int, default=10)
    return parser.parse_args()


def load_records(report_root: Path, method: str, seed: int, episodes: int) -> list[dict]:
    path = report_root / method / f"seed{seed}_n{episodes}" / "training_episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != episodes:
        raise ValueError(f"expected {episodes} episodes for {method}, found {len(rows)}")
    if [int(row["episode_index"]) for row in rows] != list(range(episodes)):
        raise ValueError(f"non-contiguous episode indices for {method}")
    if any(row["method"] != method for row in rows):
        raise ValueError(f"method mismatch in {path}")
    return rows


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    if not 1 <= window <= len(values):
        raise ValueError(f"window must be within [1, {len(values)}], got {window}")
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main() -> None:
    args = parse_args()
    output_root = args.report_root / "comparison"
    output_root.mkdir(parents=True, exist_ok=True)
    loaded = {
        method: load_records(args.report_root, method, args.seed, args.episodes)
        for method, _, _ in METHODS
    }

    source_rows = []
    summary: dict[str, dict[str, float]] = {}
    for method, label, _ in METHODS:
        success = np.asarray([int(row["success"]) for row in loaded[method]], dtype=int)
        curve = rolling(success, args.window)
        summary[method] = {
            "overall": 100.0 * float(success.mean()),
            "last20": 100.0 * float(success[-20:].mean()),
            "endpoint": 100.0 * float(curve[-1]),
        }
        for episode, outcome in enumerate(success, start=1):
            source_rows.append(
                {
                    "method": method,
                    "display_label": label,
                    "training_seed": args.seed,
                    "episode": episode,
                    "success": int(outcome),
                    f"window{args.window}_success_rate": (
                        "" if episode < args.window else float(curve[episode - args.window])
                    ),
                }
            )

    source_path = output_root / f"seed{args.seed}_online_sr_source.csv"
    with source_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    fig, ax = plt.subplots(figsize=(7.09, 3.45), constrained_layout=True)
    x = np.arange(args.window, args.episodes + 1)
    ax.axvspan(
        args.episodes - 19.5,
        args.episodes + 0.5,
        color="#EEF3F7",
        linewidth=0,
        zorder=0,
    )
    ax.text(
        args.episodes - 9.5,
        103.0,
        "Last-20 evaluation",
        ha="center",
        va="bottom",
        fontsize=6,
        color="#59636E",
    )

    for method, label, color in METHODS:
        values = np.asarray([int(row["success"]) for row in loaded[method]], dtype=float)
        curve = 100.0 * rolling(values, args.window)
        ax.plot(x, curve, color=color, linewidth=1.8, zorder=2)
        ax.scatter(
            [args.episodes],
            [curve[-1]],
            s=15,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            args.episodes + 1.2,
            curve[-1],
            f"{label}  {curve[-1]:.0f}%",
            color=color,
            va="center",
            ha="left",
            fontsize=7,
            clip_on=False,
        )

    baseline = summary["dsrl_only"]
    composite = summary["vlm_two_stage_dsrl"]
    ax.text(
        0.015,
        0.975,
        "100-episode SR  |  Last-20 SR\n"
        f"DSRL-only  {baseline['overall']:.0f}%  |  {baseline['last20']:.0f}%\n"
        f"VLM + two-stage + DSRL  {composite['overall']:.0f}%  |  {composite['last20']:.0f}%\n"
        f"Paired difference  +{composite['overall'] - baseline['overall']:.0f} pp"
        f"  |  +{composite['last20'] - baseline['last20']:.0f} pp",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#D3D7DC"},
        zorder=4,
    )
    ax.set_title(f"Zero-open-loop-pull formal run — seed {args.seed}", loc="left", pad=8)
    ax.set_xlabel("Online episode")
    ax.set_ylabel(f"Online success rate (window = {args.window}, %)")
    ax.set_xlim(args.window - 0.5, args.episodes + 17)
    ax.set_ylim(-2, 108)
    ax.set_xticks((10, 20, 40, 60, 80, 100))
    ax.set_yticks((0, 20, 40, 60, 80, 100))
    ax.grid(axis="y", color="#E5E8EB", linewidth=0.6, zorder=1)
    ax.text(
        1.0,
        -0.19,
        f"Descriptive paired run; n = 1 training seed, {args.episodes} episodes per method",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        color="#555B61",
    )

    stem = output_root / f"seed{args.seed}_online_sr_window{args.window}"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
