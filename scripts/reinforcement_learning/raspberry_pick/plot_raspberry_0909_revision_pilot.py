#!/usr/bin/env python3
"""Plot the paired n=1 raspberry_0909 scenario-revision pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PILOT_ROOT = ROOT / "exp_report/raspberry_0909/scenario_revision_zero_pull/pilot"
OUTPUT_ROOT = PILOT_ROOT / "comparison"
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


def load_records(method: str) -> list[dict]:
    path = PILOT_ROOT / method / "seed967000_n30/training_episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 30:
        raise ValueError(f"expected 30 pilot episodes for {method}, found {len(rows)}")
    return rows


def rolling(values: np.ndarray, window: int = 10) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    loaded = {method: load_records(method) for method, _, _ in METHODS}
    source_rows = []
    for method, label, _ in METHODS:
        success = np.asarray([int(row["success"]) for row in loaded[method]], dtype=int)
        curve = rolling(success)
        for episode, outcome in enumerate(success, start=1):
            source_rows.append(
                {
                    "method": method,
                    "display_label": label,
                    "training_seed": 967000,
                    "episode": episode,
                    "success": int(outcome),
                    "window10_success_rate": (
                        "" if episode < 10 else float(curve[episode - 10])
                    ),
                }
            )
    with (OUTPUT_ROOT / "pilot_online_sr_source.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    fig, ax = plt.subplots(figsize=(7.09, 3.35), constrained_layout=True)
    x = np.arange(10, 31)
    overall = {}
    for method, label, color in METHODS:
        values = np.asarray([int(row["success"]) for row in loaded[method]], dtype=float)
        curve = 100.0 * rolling(values)
        overall[method] = 100.0 * values.mean()
        ax.plot(x, curve, color=color, linewidth=2.0, marker="o", markersize=2.5)
        ax.text(
            30.35,
            curve[-1],
            label,
            color=color,
            va="center",
            ha="left",
            fontsize=7,
            clip_on=False,
        )

    ax.text(
        0.015,
        0.975,
        "Overall 30-episode SR\n"
        f"DSRL-only  {overall['dsrl_only']:.1f}%\n"
        f"VLM + two-stage + DSRL  {overall['vlm_two_stage_dsrl']:.1f}%\n"
        f"Difference  +{overall['vlm_two_stage_dsrl'] - overall['dsrl_only']:.1f} pp",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#D3D7DC"},
    )
    ax.set_title("Zero-open-loop-pull pilot (one paired training seed)", loc="left", pad=8)
    ax.set_xlabel("Online episode")
    ax.set_ylabel("Online success rate (window = 10, %)")
    ax.set_xlim(9.5, 36.5)
    ax.set_ylim(-2, 102)
    ax.set_xticks((10, 15, 20, 25, 30))
    ax.set_yticks((0, 20, 40, 60, 80, 100))
    ax.grid(axis="y", color="#E5E8EB", linewidth=0.6)
    ax.text(
        1.0,
        -0.19,
        "Descriptive pilot; n = 1 paired training seed, 30 episodes per arm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        color="#555B61",
    )

    stem = OUTPUT_ROOT / "pilot_online_sr_window10"
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
