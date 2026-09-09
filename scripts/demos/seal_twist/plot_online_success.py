#!/usr/bin/env python3
"""Plot the single-seed SealTwist online success trajectory."""

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
        "--paired-evaluation-json",
        type=Path,
        help="Optional initial-vs-final holdout report used to draw the frozen base-policy rate.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path: Path, expected_episodes: int) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    records.sort(key=lambda item: int(item["episode_index"]))
    if len(records) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} records, found {len(records)}")
    indices = [int(item["episode_index"]) for item in records]
    if indices != list(range(expected_episodes)):
        raise ValueError("episode indices must be unique and contiguous from zero")
    seeds = [int(item["seed"]) for item in records]
    if len(set(seeds)) != expected_episodes:
        raise ValueError("episode seeds must be unique")
    if any(type(item.get("success")) is not bool for item in records):
        raise ValueError("every record must contain a Boolean success field")
    return records


def make_rows(records: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    if window < 1:
        raise ValueError("window must be positive")
    outcomes = [int(item["success"]) for item in records]
    rows: list[dict[str, Any]] = []
    for position, (record, success) in enumerate(zip(records, outcomes, strict=True)):
        start = max(0, position - window + 1)
        trailing = outcomes[start : position + 1]
        rows.append(
            {
                "episode": position + 1,
                "episode_index": int(record["episode_index"]),
                "seed": int(record["seed"]),
                "outcome": str(record["outcome"]),
                "success": success,
                "rolling_window_requested": window,
                "rolling_window_effective": len(trailing),
                "trailing_successes": sum(trailing),
                "rolling_success_rate": sum(trailing) / len(trailing),
                "rolling_success_rate_percent": 100.0 * sum(trailing) / len(trailing),
            }
        )
    return rows


def write_source_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_base_success_rate(path: Path | None) -> float | None:
    if path is None:
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    rate = float(report["initial_controller"]["success_rate"])
    if not 0.0 <= rate <= 1.0:
        raise ValueError("base-policy success rate must be in [0, 1]")
    return rate


def draw_figure(
    rows: list[dict[str, Any]],
    output_stem: Path,
    window: int,
    base_success_rate: float | None,
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
    episodes = [row["episode"] for row in rows]
    successes = [row["success"] for row in rows]
    rolling = [row["rolling_success_rate"] for row in rows]
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
    if base_success_rate is not None:
        ax.axhline(
            base_success_rate,
            color="#C44E52",
            linewidth=1.1,
            linestyle=(0, (4, 2)),
            label=f"Frozen base policy ({base_success_rate:.0%}, holdout)",
            zorder=2,
        )
    ax.scatter(
        episodes[-1],
        rolling[-1],
        s=18,
        facecolor="#1769AA",
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
    ax.set_title("SealTwist online success", loc="left", fontweight="bold", pad=5)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.99, 0.025),
        frameon=False,
        ncol=1,
        fontsize=5.8,
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
    records = load_records(input_path, args.expected_episodes)
    rows = make_rows(records, args.window)
    paired_evaluation_path = (
        None
        if args.paired_evaluation_json is None
        else args.paired_evaluation_json.expanduser().resolve()
    )
    base_success_rate = load_base_success_rate(paired_evaluation_path)
    source_csv = output_dir / f"online_success_rate_window{args.window}_source.csv"
    output_stem = output_dir / f"online_success_rate_window{args.window}"
    write_source_csv(source_csv, rows)
    draw_figure(rows, output_stem, args.window, base_success_rate)
    manifest = {
        "input_jsonl": str(input_path),
        "input_sha256": sha256(input_path),
        "episodes": len(rows),
        "unique_episode_seeds": len({row["seed"] for row in rows}),
        "training_seed": rows[0]["seed"],
        "rolling_window": args.window,
        "early_window_rule": "episodes before the requested window use the available prefix",
        "single_seed_uncertainty": "No confidence interval; this is one online training run.",
        "overall_success_rate": sum(row["success"] for row in rows) / len(rows),
        "final_trailing_success_rate": rows[-1]["rolling_success_rate"],
        "paired_evaluation_json": (
            None if paired_evaluation_path is None else str(paired_evaluation_path)
        ),
        "paired_evaluation_sha256": (
            None if paired_evaluation_path is None else sha256(paired_evaluation_path)
        ),
        "base_policy_holdout_success_rate": base_success_rate,
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
