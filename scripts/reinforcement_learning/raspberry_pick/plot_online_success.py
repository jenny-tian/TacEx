#!/usr/bin/env python3
"""Plot RaspberryPick online post-training success from complete episode logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument(
        "--base-evaluation-json",
        type=Path,
        default=Path(
            "exp_report/raspberry_pick/bc/evaluation/holdout/aggregate_n90.json"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path: Path, expected_episodes: int) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records.sort(key=lambda item: int(item["episode_index"]))
    if len(records) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} episodes, found {len(records)}")
    if [int(item["episode_index"]) for item in records] != list(range(expected_episodes)):
        raise ValueError("episode indices must be unique and contiguous from zero")
    if len({int(item["seed"]) for item in records}) != expected_episodes:
        raise ValueError("episode seeds must be unique")
    if any(type(item.get("success")) is not bool for item in records):
        raise ValueError("every episode must contain a Boolean success field")
    return records


def make_rows(records: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    if window < 1:
        raise ValueError("window must be positive")
    outcomes = [int(record["success"]) for record in records]
    rows: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        start = max(0, position - window + 1)
        trailing = outcomes[start : position + 1]
        attempted = list(map(float, record["attempted_force_range_n"]))
        following = list(map(float, record["next_force_range_n"]))
        rows.append(
            {
                "episode": position + 1,
                "episode_index": int(record["episode_index"]),
                "seed": int(record["seed"]),
                "outcome": str(record["outcome"]),
                "success": outcomes[position],
                "rolling_window_requested": window,
                "rolling_window_effective": len(trailing),
                "trailing_successes": sum(trailing),
                "rolling_success_rate": sum(trailing) / len(trailing),
                "attempted_force_low_n": attempted[0],
                "attempted_force_high_n": attempted[1],
                "next_force_low_n": following[0],
                "next_force_high_n": following[1],
                "peak_grip_force_n": float(record["peak_grip_force_n"]),
                "peak_pedicle_tension_n": float(record["peak_pedicle_tension_n"]),
                "dsrl_active_decisions": int(record["dsrl_active_decisions"]),
                "replay_size": int(record["replay_size"]),
                "sac_updates": int(record["sac_updates"]),
                "vlm_call_succeeded": int(bool(record["vlm_call_succeeded"])),
            }
        )
    return rows


def write_source_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_base_success_rate(path: Path) -> float:
    report = json.loads(path.read_text(encoding="utf-8"))
    rate = float(report["success_rate"])
    if not 0.0 <= rate <= 1.0:
        raise ValueError("base success rate must be in [0, 1]")
    return rate


def draw_figure(
    rows: list[dict[str, Any]], output_stem: Path, window: int, base_rate: float
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
    successes = [int(row["success"]) for row in rows]
    rolling = [float(row["rolling_success_rate"]) for row in rows]
    full_start = min(window - 1, len(rows) - 1)

    fig, ax = plt.subplots(figsize=(3.50, 2.55), constrained_layout=True)
    ax.scatter(
        episodes,
        successes,
        s=7,
        marker="|",
        linewidths=0.75,
        color="#A7ADB4",
        alpha=0.45,
        label="Episode outcome",
        zorder=1,
    )
    if full_start > 0:
        ax.plot(
            episodes[: full_start + 1],
            rolling[: full_start + 1],
            color="#74A9CF",
            linewidth=1.25,
            linestyle=(0, (2.0, 1.5)),
            zorder=2,
        )
    ax.plot(
        episodes[full_start:],
        rolling[full_start:],
        color="#1769AA",
        linewidth=1.8,
        label=f"Trailing success rate (window={window})",
        zorder=3,
    )
    ax.axhline(
        base_rate,
        color="#C44E52",
        linewidth=1.1,
        linestyle=(0, (4, 2)),
        label=f"Frozen BC ({base_rate:.1%}, 90-episode holdout)",
        zorder=2,
    )
    ax.scatter(
        episodes[-1],
        rolling[-1],
        s=18,
        color="#1769AA",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    ax.annotate(
        f"{rolling[-1]:.0%}",
        xy=(episodes[-1], rolling[-1]),
        xytext=(-5, 6 if rolling[-1] <= 0.82 else -10),
        textcoords="offset points",
        ha="right",
        va="bottom" if rolling[-1] <= 0.82 else "top",
        color="#1769AA",
        fontsize=6.5,
        fontweight="bold",
    )
    if window > 1:
        ax.axvline(window, color="#CDD1D5", linewidth=0.7, linestyle=(0, (2, 2)))
        ax.text(
            window + max(0.5, len(rows) / 100),
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
    ax.set_xlabel("Online post-training episode")
    ax.set_ylabel("Success rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    if len(rows) == 100:
        ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("RaspberryPick VLM–DSRL online success", loc="left", fontweight="bold", pad=5)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.99, 0.025),
        fontsize=5.6,
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
    input_path = args.input_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    base_path = args.base_evaluation_json.expanduser().resolve()
    records = load_records(input_path, args.expected_episodes)
    rows = make_rows(records, args.window)
    source_csv = output_dir / f"online_success_rate_window{args.window}_source.csv"
    output_stem = output_dir / f"online_success_rate_window{args.window}"
    write_source_csv(source_csv, rows)
    base_rate = load_base_success_rate(base_path)
    draw_figure(rows, output_stem, args.window, base_rate)
    manifest = {
        "figure_contract": {
            "core_conclusion": (
                "The curve reports how RaspberryPick online success changes during one "
                "VLM-DSRL post-training run relative to the independently evaluated frozen BC."
            ),
            "archetype": "single-panel quantitative time series",
            "backend": "Python/matplotlib",
            "final_size_mm": [88.9, 64.8],
        },
        "input_jsonl": str(input_path),
        "input_sha256": sha256(input_path),
        "episodes": len(rows),
        "unique_episode_seeds": len({row["seed"] for row in rows}),
        "rolling_window": args.window,
        "early_window_rule": "episodes before the requested window use the available prefix",
        "single_run_uncertainty": "No confidence interval; episodes belong to one online training run.",
        "overall_success_rate": sum(row["success"] for row in rows) / len(rows),
        "final_trailing_success_rate": rows[-1]["rolling_success_rate"],
        "base_evaluation_json": str(base_path),
        "base_evaluation_sha256": sha256(base_path),
        "base_policy_holdout_success_rate": base_rate,
        "source_csv": str(source_csv),
        "figure_stem": str(output_stem),
        "formats": ["png", "svg", "pdf", "tiff"],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
