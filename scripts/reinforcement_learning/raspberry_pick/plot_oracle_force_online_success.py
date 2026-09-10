#!/usr/bin/env python3
"""Plot the paired Base/DSRL/ground-truth-force online success trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


COLORS = {
    "base": "#7A7F87",
    "dsrl": "#D17C2F",
    "oracle": "#1769AA",
}
LABELS = {
    "base": "Base policy",
    "dsrl": "DSRL",
    "oracle": "Ground-truth force + DSRL",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.5,
        "axes.linewidth": 0.75,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 6.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--dsrl-run", type=Path, required=True)
    parser.add_argument("--oracle-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--expected-episodes", type=int, default=30)
    parser.add_argument(
        "--allow-incomplete-oracle",
        action="store_true",
        help=(
            "Plot a contiguous, incomplete oracle log when its training summary is "
            "not yet available. Base and DSRL runs must still be complete."
        ),
    )
    return parser.parse_args()


def load_episodes(
    run_dir: Path, expected_episodes: int, *, allow_incomplete: bool = False
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    summary_path = run_dir / "training_summary.json"
    records_path = run_dir / "training_episodes.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(f"incomplete run: {run_dir}")
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else None
    )
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"empty episode log: {run_dir}")
    if len(records) > expected_episodes:
        raise ValueError(f"too many episodes: {run_dir}")
    if not allow_incomplete and (
        summary is None
        or int(summary["completed_episodes"]) != expected_episodes
        or len(records) != expected_episodes
    ):
        raise ValueError(f"incomplete run: {run_dir}")
    if [int(row["episode_index"]) for row in records] != list(range(len(records))):
        raise ValueError(f"non-contiguous episode indices: {run_dir}")
    successes = sum(int(row["success"]) for row in records)
    if summary is not None and int(summary["successes"]) != successes:
        raise ValueError(f"summary/episode success mismatch: {run_dir}")
    return summary, records


def validate_runs(
    summaries: dict[str, dict[str, Any] | None],
    records: dict[str, list[dict[str, Any]]],
    oracle_run: Path,
) -> None:
    base_summary = summaries["base"]
    dsrl_summary = summaries["dsrl"]
    oracle_summary = summaries["oracle"]
    if base_summary is None or dsrl_summary is None:
        raise ValueError("base and DSRL runs require training summaries")
    if base_summary.get("post_training_method") != "dsrl_only" or not bool(
        base_summary.get("dsrl_frozen")
    ):
        raise ValueError("base run must be dsrl_only with DSRL frozen")
    if dsrl_summary.get("post_training_method") != "dsrl_only" or bool(
        dsrl_summary.get("dsrl_frozen")
    ):
        raise ValueError("DSRL run must be dsrl_only with DSRL enabled")

    matched_fields = (
        "training_seed",
        "completed_episodes",
        "berry_profile",
        "base_checkpoint_sha256",
        "residual_mode",
        "noise_action_indices",
        "noise_residual_scale",
        "dsrl_start_step",
        "position_randomization_m",
        "physical_force_bounds_n",
        "damage_threshold_n",
        "detach_threshold_n",
    )
    for field in matched_fields:
        names = ("base", "dsrl", "oracle") if oracle_summary is not None else ("base", "dsrl")
        values = [summaries[name].get(field) for name in names]  # type: ignore[union-attr]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"run mismatch for {field}: {values}")

    base_seeds = [int(row["seed"]) for row in records["base"]]
    if [int(row["seed"]) for row in records["dsrl"]] != base_seeds:
        raise ValueError("base and DSRL episode seed sequences do not match")
    oracle_seeds = [int(row["seed"]) for row in records["oracle"]]
    if oracle_seeds != base_seeds[: len(oracle_seeds)]:
        raise ValueError("oracle episode seed sequence does not match the paired prefix")

    if oracle_summary is not None:
        if oracle_summary.get("post_training_method") != "oracle_force_dsrl":
            raise ValueError("oracle run must use oracle_force_dsrl")
        if bool(oracle_summary.get("vlm_is_codex")) or int(
            oracle_summary.get("vlm_calls", -1)
        ) != 0:
            raise ValueError("oracle run must not call a VLM")
    else:
        decisions_path = oracle_run / "force_range_decisions.jsonl"
        decisions = [
            json.loads(line)
            for line in decisions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(decisions) != len(records["oracle"]):
            raise ValueError("incomplete oracle run has mismatched force decisions")
        if any(row.get("method") != "oracle_force_dsrl" for row in decisions):
            raise ValueError("incomplete oracle run must use oracle_force_dsrl")
        base_damage = float(base_summary["damage_threshold_n"])
        if any(
            not np.isclose(float(row["damage_threshold_n"]), base_damage)
            for row in records["oracle"]
        ):
            raise ValueError("incomplete oracle run has a mismatched damage threshold")


def trailing_rate(values: np.ndarray, window: int) -> np.ndarray:
    """Return a strict trailing-window rate; the first window - 1 values are NaN."""
    if values.ndim != 1 or values.size == 0:
        raise ValueError("success values must be a non-empty one-dimensional array")
    if window < 1 or window > values.size:
        raise ValueError("window must be between 1 and the number of episodes")
    result = np.full(values.size, np.nan, dtype=float)
    result[window - 1 :] = np.convolve(values, np.ones(window), mode="valid") / window
    return result


def main() -> None:
    args = parse_args()
    run_paths = {
        "base": args.base_run.expanduser().resolve(),
        "dsrl": args.dsrl_run.expanduser().resolve(),
        "oracle": args.oracle_run.expanduser().resolve(),
    }
    summaries: dict[str, dict[str, Any] | None] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for name, path in run_paths.items():
        summaries[name], records[name] = load_episodes(
            path,
            args.expected_episodes,
            allow_incomplete=name == "oracle" and args.allow_incomplete_oracle,
        )
    validate_runs(summaries, records, run_paths["oracle"])

    base_summary = summaries["base"]
    if base_summary is None:
        raise AssertionError("base summary was validated above")

    episode = np.arange(1, args.expected_episodes + 1)
    successes = {
        name: np.asarray([int(row["success"]) for row in records[name]], dtype=float)
        for name in ("base", "dsrl", "oracle")
    }
    rolling = {name: trailing_rate(values, args.window) for name, values in successes.items()}

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / f"online_success_window{args.window}_source.csv"
    episode_seeds = [int(row["seed"]) for row in records["base"]]
    with source_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "episode",
                "episode_seed",
                "rolling_window",
                "base_success",
                "base_trailing_success_rate",
                "dsrl_success",
                "dsrl_trailing_success_rate",
                "ground_truth_force_dsrl_success",
                "ground_truth_force_dsrl_trailing_success_rate",
            )
        )
        for index in range(args.expected_episodes):
            oracle_available = index < len(successes["oracle"])
            writer.writerow(
                (
                    int(episode[index]),
                    episode_seeds[index],
                    args.window,
                    int(successes["base"][index]),
                    "" if np.isnan(rolling["base"][index]) else rolling["base"][index],
                    int(successes["dsrl"][index]),
                    "" if np.isnan(rolling["dsrl"][index]) else rolling["dsrl"][index],
                    int(successes["oracle"][index]) if oracle_available else "",
                    (
                        ""
                        if not oracle_available or np.isnan(rolling["oracle"][index])
                        else rolling["oracle"][index]
                    ),
                )
            )

    # Single-column journal figure: 89 x 64 mm.
    fig, axis = plt.subplots(figsize=(3.504, 2.520), constrained_layout=True)
    styles = {"base": (0, (3, 2)), "dsrl": (0, (5, 1.5, 1, 1.5)), "oracle": "solid"}
    widths = {"base": 1.35, "dsrl": 1.6, "oracle": 1.6}
    zorders = {"base": 4, "dsrl": 3, "oracle": 2}
    for name in ("base", "dsrl", "oracle"):
        series_episode = np.arange(1, len(successes[name]) + 1)
        valid = series_episode >= args.window
        overall = 100.0 * float(successes[name].mean())
        count_text = f"{int(successes[name].sum())}/{len(successes[name])}"
        if len(successes[name]) < args.expected_episodes:
            count_text += "; partial"
        axis.plot(
            series_episode[valid],
            rolling[name][valid] * 100.0,
            color=COLORS[name],
            linewidth=widths[name],
            linestyle=styles[name],
            marker="o",
            markersize=2.4 if name == "oracle" else 2.1,
            markeredgewidth=0.0,
            label=f"{LABELS[name]} ({count_text}; {overall:.1f}%)",
            zorder=zorders[name],
        )
    axis.set(
        xlim=(args.window, args.expected_episodes),
        ylim=(-2, 102),
        xticks=(10, 15, 20, 25, 30),
        yticks=(0, 20, 40, 60, 80, 100),
        xlabel="Online training episode",
        ylabel="Success rate",
        title=f"Online success rate ({args.window}-episode sliding window)",
    )
    axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.6, zorder=0)
    axis.tick_params(length=2.5, pad=2.0)
    axis.legend(
        loc="lower left",
        handlelength=2.4,
        title=(
            f"Overall success; paired seed {int(base_summary['training_seed']):,}"
        ),
        title_fontsize=5.5,
    )

    stem = output_dir / f"online_success_window{args.window}"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    metadata = {
        "window": args.window,
        "rolling_definition": (
            f"strict trailing window; the first plotted point is episode {args.window} "
            f"and covers episodes 1-{args.window}"
        ),
        "episodes_per_method": {name: len(values) for name, values in successes.items()},
        "training_seed": int(base_summary["training_seed"]),
        "incomplete_methods": [
            name for name, values in successes.items() if len(values) < args.expected_episodes
        ],
        "successes": {name: int(values.sum()) for name, values in successes.items()},
        "overall_success_rates": {name: float(values.mean()) for name, values in successes.items()},
        "final_trailing_success_rates": {
            name: float(values[-1]) for name, values in rolling.items()
        },
        "uncertainty": "none; one temporally dependent online-training run per method",
        "source_csv": source_path.name,
    }
    (output_dir / f"online_success_window{args.window}_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
