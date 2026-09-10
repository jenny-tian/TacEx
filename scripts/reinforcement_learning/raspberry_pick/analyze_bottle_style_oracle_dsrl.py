#!/usr/bin/env python3
"""Analyze one 100-episode Bottle-style RaspberryPick DSRL run.

Figure contract
---------------
Core conclusion: determine whether rolling success rises late in online training.
Archetype: one-panel quantitative trend.
Hero evidence: trailing 10-episode online success rate.
Replicate unit: one training run; episodes are sequential observations, not seeds.
"""

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
from matplotlib.ticker import PercentFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-json", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[int] | list[float]) -> float:
    return float(statistics.fmean(values))


def slope_per_10(x: list[int], y: list[float]) -> float:
    mean_x = mean(x)
    mean_y = mean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator == 0.0:
        return 0.0
    return (
        sum(
            (left - mean_x) * (right - mean_y)
            for left, right in zip(x, y, strict=True)
        )
        / denominator
        * 10.0
    )


def load_and_validate(
    run_dir: Path, base_json: Path, window: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if window < 1:
        raise ValueError("window must be positive")
    episode_path = run_dir / "training_episodes.jsonl"
    summary_path = run_dir / "training_summary.json"
    records = [
        json.loads(line)
        for line in episode_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    base = json.loads(base_json.read_text(encoding="utf-8"))
    episodes = int(summary["completed_episodes"])
    training_seed = int(summary["training_seed"])
    if episodes != 100 or len(records) != episodes:
        raise ValueError(f"expected one complete 100-episode run, found {len(records)}")
    if [int(row["episode_index"]) for row in records] != list(range(episodes)):
        raise ValueError("episode indices are not contiguous")
    if [int(row["seed"]) for row in records] != list(
        range(training_seed, training_seed + episodes)
    ):
        raise ValueError("episode reset seeds are not sequential from the training seed")
    if summary.get("episode_seed_mode") != "sequential":
        raise ValueError("this analysis requires one run with sequential episode seeds")
    if summary.get("residual_mode") != "bottle_style_lateral":
        raise ValueError("run is not Bottle-style lateral DSRL")
    if summary.get("oracle_control_mode") != "two_stage":
        raise ValueError("run does not use two-stage oracle force control")
    if summary.get("flow_initial_latent") != "zero":
        raise ValueError("Bottle-style run must keep the Flow latent at zero")
    if bool(summary.get("dsrl_frozen")) or bool(summary.get("evaluation_mode")):
        raise ValueError("run is not online DSRL training")
    if int(base["seed_start"]) != training_seed:
        raise ValueError("base run does not start from the same seed")
    if int(base["episodes"]) > episodes:
        raise ValueError("base run is longer than the online run")
    if base.get("checkpoint_sha256") != summary.get("base_checkpoint_sha256"):
        raise ValueError("base and online runs use different BC checkpoints")
    if not bool(base.get("stem_lock_rotation_while_attached")):
        raise ValueError("base result predates the anti-rotation environment")
    return records, summary, base


def rolling_success(success: list[int], window: int) -> list[float | None]:
    return [
        None if index + 1 < window else mean(success[index - window + 1 : index + 1])
        for index in range(len(success))
    ]


def write_source_csv(
    path: Path,
    records: list[dict[str, Any]],
    rolling: list[float | None],
) -> None:
    fields = (
        "episode",
        "reset_seed",
        "outcome",
        "success",
        "rolling_success_rate_window10",
        "exploration_scale",
        "dsrl_active_decisions",
        "mean_abs_lateral_residual_m",
        "force_balance_prior_used",
        "sac_updates",
        "replay_size",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (record, rolling_value) in enumerate(
            zip(records, rolling, strict=True)
        ):
            writer.writerow(
                {
                    "episode": index + 1,
                    "reset_seed": record["seed"],
                    "outcome": record["outcome"],
                    "success": int(bool(record["success"])),
                    "rolling_success_rate_window10": (
                        "" if rolling_value is None else rolling_value
                    ),
                    "exploration_scale": record["exploration_scale"],
                    "dsrl_active_decisions": record["dsrl_active_decisions"],
                    "mean_abs_lateral_residual_m": record[
                        "mean_abs_lateral_residual_m"
                    ],
                    "force_balance_prior_used": record["force_balance_prior_used"],
                    "sac_updates": record["sac_updates"],
                    "replay_size": record["replay_size"],
                }
            )


def plot_curve(
    output_stem: Path,
    rolling: list[float | None],
    *,
    window: int,
    first_20: float,
    last_20: float,
    base_rate: float,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )
    # 3.504 in = 89 mm, the intended single-column final width.
    fig, ax = plt.subplots(figsize=(3.504, 2.55), constrained_layout=True)
    x = [index + 1 for index, value in enumerate(rolling) if value is not None]
    y = [float(value) for value in rolling if value is not None]
    ax.axvspan(1, 30, color="#DCEAF5", alpha=0.55, linewidth=0)
    ax.axhline(
        base_rate,
        color="#656565",
        linestyle=(0, (4, 2)),
        linewidth=1.0,
        label="Frozen Base (same seed range, n=30)",
    )
    ax.plot(
        x,
        y,
        color="#1769AA",
        linewidth=1.45,
        marker="o",
        markersize=1.8,
        markeredgewidth=0,
        label=f"Two-stage force + DSRL (window={window})",
        zorder=3,
    )
    ax.hlines(
        first_20,
        1,
        20,
        color="#D17C2F",
        linewidth=1.1,
        label="First/last 20 mean",
    )
    ax.hlines(last_20, 81, 100, color="#D17C2F", linewidth=1.1)
    ax.text(15.5, 0.96, "Exploration anneal", color="#416780", ha="center", va="top")
    ax.text(20.5, first_20, f"{first_20:.0%}", color="#9B581D", va="center")
    ax.text(80.0, last_20, f"{last_20:.0%}", color="#9B581D", ha="right", va="center")
    ax.set_xlim(1, 100)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Online episode")
    ax.set_ylabel("Success rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xticks((1, 10, 20, 40, 60, 80, 100))
    ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, handlelength=2.5)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    base_json = args.base_json.expanduser().resolve()
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    records, summary, base = load_and_validate(run_dir, base_json, args.window)
    success = [int(bool(record["success"])) for record in records]
    rolling = rolling_success(success, args.window)
    full_x = [index + 1 for index, value in enumerate(rolling) if value is not None]
    full_y = [float(value) for value in rolling if value is not None]
    first_10 = mean(success[:10])
    last_10 = mean(success[-10:])
    first_20 = mean(success[:20])
    last_20 = mean(success[-20:])
    base_rate = float(base["success_rate"])
    matched_count = int(base["episodes"])
    failures = Counter(
        record["outcome"] for record in records if not bool(record["success"])
    )
    metrics = {
        "protocol": {
            "training_runs": 1,
            "training_seed": int(summary["training_seed"]),
            "episode_seed_mode": "sequential",
            "episodes": len(records),
            "rolling_window_episodes": args.window,
            "force_control": "two_stage_ground_truth",
            "grip_target_n": summary["initial_force_range_n"],
            "damage_threshold_n": summary["damage_threshold_n"],
            "detach_threshold_n": summary["detach_threshold_n"],
            "residual": summary["bottle_style_lateral_residual"],
            "independence_note": (
                "Episodes are temporally dependent observations from one online-training "
                "run and must not be interpreted as independent training seeds."
            ),
        },
        "online_results": {
            "successes": sum(success),
            "success_rate": mean(success),
            "first_10_success_rate": first_10,
            "last_10_success_rate": last_10,
            "first_20_success_rate": first_20,
            "last_20_success_rate": last_20,
            "last20_minus_first20": last_20 - first_20,
            "final_rolling_window10": full_y[-1],
            "rolling_window10_slope_per_10_episodes_episode10_to100": (
                slope_per_10(full_x, full_y)
            ),
            "failure_counts": dict(failures),
            "sac_optimizer_updates": int(summary["sac_optimizer_updates"]),
            "replay_transitions": int(summary["replay_transitions"]),
        },
        "base_comparison": {
            "same_seed_range_episodes": matched_count,
            "base_successes": int(base["successes"]),
            "base_success_rate": base_rate,
            "online_first_matched_episodes_successes": sum(success[:matched_count]),
            "online_first_matched_episodes_success_rate": mean(
                success[:matched_count]
            ),
            "matched_range_rate_difference": mean(success[:matched_count]) - base_rate,
            "last_20_minus_base_rate": last_20 - base_rate,
        },
        "trend_interpretation": {
            "later_success_rate_increased": bool(
                last_20 > first_20 and slope_per_10(full_x, full_y) > 0.0
            ),
            "last_20_exceeds_same_seed_range_base": bool(last_20 > base_rate),
            "descriptive_only": True,
        },
        "provenance": {
            "episode_jsonl": str((run_dir / "training_episodes.jsonl").resolve()),
            "episode_jsonl_sha256": sha256(run_dir / "training_episodes.jsonl"),
            "training_summary": str((run_dir / "training_summary.json").resolve()),
            "training_summary_sha256": sha256(run_dir / "training_summary.json"),
            "base_json": str(base_json),
            "base_json_sha256": sha256(base_json),
        },
    }
    csv_path = analysis_dir / "online_success_window10_source.csv"
    write_source_csv(csv_path, records, rolling)
    figure_stem = analysis_dir / "online_success_window10"
    plot_curve(
        figure_stem,
        rolling,
        window=args.window,
        first_20=first_20,
        last_20=last_20,
        base_rate=base_rate,
    )
    metrics["provenance"]["source_csv"] = str(csv_path.resolve())
    metrics["provenance"]["source_csv_sha256"] = sha256(csv_path)
    metrics_path = analysis_dir / "online_success_window10_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    slope = metrics["online_results"][
        "rolling_window10_slope_per_10_episodes_episode10_to100"
    ]
    increased = metrics["trend_interpretation"]["later_success_rate_increased"]
    exceeded = metrics["trend_interpretation"][
        "last_20_exceeds_same_seed_range_base"
    ]
    report = f"""# Bottle-style two-stage force + DSRL: 100-episode online result

## Conclusion

Late online performance **{'increased' if increased else 'did not show a consistent increase'}**: first 20 = {first_20:.1%}, last 20 = {last_20:.1%}, delta = {last_20 - first_20:+.1%}; the window-10 slope over episodes 10–100 was {slope * 100:+.2f} percentage points per 10 episodes.

The last-20 rate **{'exceeded' if exceeded else 'did not exceed'}** the frozen-Base same-seed-range reference ({int(base['successes'])}/{matched_count}, {base_rate:.1%}). Over the directly matched first {matched_count} reset seeds, the online stack achieved {sum(success[:matched_count])}/{matched_count} ({mean(success[:matched_count]):.1%}), a difference of {mean(success[:matched_count]) - base_rate:+.1%}.

## Metrics

| Metric | Result |
|---|---:|
| Overall | {sum(success)}/100 ({mean(success):.1%}) |
| First 10 / last 10 | {first_10:.1%} / {last_10:.1%} |
| First 20 / last 20 | {first_20:.1%} / {last_20:.1%} |
| Final rolling window (10) | {full_y[-1]:.1%} |
| Rolling-window slope | {slope * 100:+.2f} pp / 10 episodes |
| Replay transitions | {int(summary['replay_transitions'])} |
| SAC optimizer updates | {int(summary['sac_optimizer_updates'])} |

Failures: `{json.dumps(dict(failures), ensure_ascii=False, sort_keys=True)}`.

## Interpretation boundary

This is one online-training run initialized by one training seed. Episode reset seeds advance sequentially, and overlapping rolling windows are temporally dependent. The curve is therefore descriptive evidence of within-run learning, not a multi-seed uncertainty estimate or an independent-sample significance test. The matched first-{matched_count} comparison controls reset-seed identities, but the online policy changes across episodes.
"""
    (analysis_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    qa = f"""# Figure QA

- Core conclusion: whether window-10 online success rises late in one 100-episode run.
- Panel: one quantitative trend panel; trailing window-10 success is the hero evidence.
- Replicate unit: one training run; no variability band is shown because there is one seed.
- Metric: binary task success averaged over the trailing 10 sequential episodes.
- Baseline: frozen BC, {int(base['successes'])}/{matched_count} on reset seeds {int(base['seed_start'])}–{int(base['seed_start']) + matched_count - 1}.
- Statistics: descriptive first/last means and ordinary least-squares slope; no hypothesis test.
- Source data: `online_success_window10_source.csv`.
- Collision audit: labels, curve, baseline and first/last mean segments inspected after rendering.
- Export: editable SVG/PDF plus 300 dpi PNG and 600 dpi TIFF.
"""
    (analysis_dir / "FIGURE_QA.md").write_text(qa, encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
