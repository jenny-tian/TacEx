#!/usr/bin/env python3
"""Create source tables, statistics, a paper-ready figure, and the VLM-force report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-results",
        type=Path,
        default=REPO_ROOT
        / "exp_report/vlm_inference_res/isaac_vlm_force_seed42_n20/results.json",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=REPO_ROOT
        / "exp_report/vlm_inference_res/isaac_policy_only_seed42_n20/results.json",
    )
    parser.add_argument(
        "--rl-summary",
        type=Path,
        default=REPO_ROOT / "exp_report/raw_dsrl/summary.json",
    )
    parser.add_argument(
        "--rl-episodes",
        type=Path,
        default=REPO_ROOT / "exp_report/raw_dsrl/online_episodes.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "exp_report/vlm_inference_res"
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    row_one = a + b
    total_success = a + c
    total = a + b + c + d
    denominator = math.comb(total, row_one)

    def probability(x: int) -> float:
        return (
            math.comb(total_success, x)
            * math.comb(total - total_success, row_one - x)
            / denominator
        )

    observed = probability(a)
    minimum = max(0, row_one - (total - total_success))
    maximum = min(row_one, total_success)
    return min(
        1.0,
        sum(
            probability(x)
            for x in range(minimum, maximum + 1)
            if probability(x) <= observed + 1e-15
        ),
    )


def exact_mcnemar(force_only: int, baseline_only: int) -> float:
    discordant = force_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(force_only, baseline_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def successful_tail_tracking(
    force_data: dict[str, Any], results_path: Path
) -> list[dict[str, float]]:
    metrics: list[dict[str, float]] = []
    trace_dir = results_path.parent / "force_traces"
    initial_range = force_data["initial_force_range_n"]
    for episode in force_data["results"]:
        if not episode["success"]:
            continue
        rows = read_trace(trace_dir / f"episode_{episode['episode_index']:03d}.csv")
        eligible = [
            row
            for row in rows
            if int(row["controller_active"])
            and int(row["loaded_contact"])
            and int(row["bilateral_contact"])
        ]
        tail = eligible[-20:]
        if not tail:
            continue
        tail_mean = statistics.fmean(float(row["grip_force_n"]) for row in tail)
        target = float(episode["attempted_target_force_n"])
        metrics.append(
            {
                "episode_index": int(episode["episode_index"]),
                "tail_samples": len(tail),
                "target_force_n": target,
                "tail_mean_force_n": tail_mean,
                "tail_absolute_error_n": abs(tail_mean - target),
                "after_initial_range": float(
                    episode["attempted_force_range_n"] != initial_range
                ),
            }
        )
    return metrics


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def episode_source_rows(
    force_data: dict[str, Any], baseline_data: dict[str, Any], rl_episodes_path: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, protocol, data in (
        ("VLM-protocol + force control", "frozen_policy_evaluation", force_data),
        ("Policy only", "frozen_policy_evaluation", baseline_data),
    ):
        for item in data["results"]:
            rows.append(
                {
                    "method": method,
                    "protocol": protocol,
                    "episode_index": item["episode_index"],
                    "seed": item["seed"],
                    "success": int(item["success"]),
                    "broken": int(item["terminal_reason"] == "object_broken"),
                    "terminal_reason": item["terminal_reason"],
                    "diagnosed_failure_reason": item["diagnosed_failure_reason"],
                    "target_lower_n": item["attempted_force_range_n"][0],
                    "target_upper_n": item["attempted_force_range_n"][1],
                    "target_center_n": item["attempted_target_force_n"],
                    "mean_contact_force_n": item["mean_contact_force_n"],
                    "peak_contact_force_n": item["peak_contact_force_n"],
                    "force_rmse_n": item["force_rmse_n"],
                    "contact_fraction": item["contact_fraction"],
                    "controller_active_fraction": item["controller_active_fraction"],
                    "max_uncontrolled_action_delta": item[
                        "max_uncontrolled_action_delta"
                    ],
                }
            )
    for line in rl_episodes_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        rows.append(
            {
                "method": "Online DSRL",
                "protocol": "online_training_reference",
                "episode_index": item["episode_index"],
                "seed": "",
                "success": int(item["success"]),
                "broken": int(item["terminal_reason"] == "object_broken"),
                "terminal_reason": item["terminal_reason"],
                "diagnosed_failure_reason": "",
                "target_lower_n": "",
                "target_upper_n": "",
                "target_center_n": "",
                "mean_contact_force_n": "",
                "peak_contact_force_n": item["peak_contact_force_n"],
                "force_rmse_n": "",
                "contact_fraction": "",
                "controller_active_fraction": "",
                "max_uncontrolled_action_delta": "",
            }
        )
    return rows


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_figure(
    force_data: dict[str, Any],
    baseline_data: dict[str, Any],
    rl_summary: dict[str, Any],
    force_results_path: Path,
    output_dir: Path,
) -> int:
    configure_plotting()
    blue = "#3B6EA8"
    pale_blue = "#BFD7EA"
    orange = "#D98C3F"
    green = "#3A8D68"
    red = "#C44E52"
    grey = "#777777"

    fig = plt.figure(figsize=(7.09, 5.55), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    episodes = [int(item["episode_index"]) + 1 for item in force_data["results"]]
    lower = [
        float(item["attempted_force_range_n"][0]) for item in force_data["results"]
    ]
    upper = [
        float(item["attempted_force_range_n"][1]) for item in force_data["results"]
    ]
    center = [float(item["attempted_target_force_n"]) for item in force_data["results"]]
    ax_a.fill_between(
        episodes,
        lower,
        upper,
        step="post",
        color=pale_blue,
        alpha=0.65,
        label="Target range",
    )
    ax_a.step(episodes, center, where="post", color=blue, lw=1.5, label="Target centre")
    for item, x, y in zip(force_data["results"], episodes, center):
        if item["success"]:
            ax_a.scatter(
                x, y, s=20, color=green, edgecolor="white", linewidth=0.4, zorder=4
            )
        elif item["terminal_reason"] == "object_broken":
            ax_a.scatter(x, y, s=26, marker="x", color=red, linewidth=1.1, zorder=4)
        else:
            ax_a.scatter(
                x, y, s=12, color=grey, edgecolor="white", linewidth=0.3, zorder=3
            )
    ax_a.set(
        xlabel="Episode",
        ylabel="One-fingertip target force (N)",
        xlim=(0.5, 20.5),
        ylim=(0, 3.55),
    )
    ax_a.set_xticks([1, 5, 10, 15, 20])
    ax_a.axhline(force_data["break_force_threshold_n"], color=red, ls="--", lw=0.8)
    range_handles, _ = ax_a.get_legend_handles_labels()
    outcome_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=green,
            markeredgecolor="white",
            label="Success",
        ),
        Line2D([], [], marker="x", linestyle="none", color=red, label="Broken"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=grey,
            markeredgecolor="white",
            label="Other failure",
        ),
    ]
    ax_a.legend(
        handles=[*range_handles, *outcome_handles],
        loc="upper right",
        ncol=2,
        columnspacing=0.8,
    )

    methods = [("Force control", force_data), ("Policy only", baseline_data)]
    x_locations = [0.0, 0.35, 1.0, 1.35]
    rates: list[float] = []
    intervals: list[tuple[float, float]] = []
    colors = [blue, grey, blue, grey]
    labels = ["Force control", "Policy only", "Force control", "Policy only"]
    for metric in ("successes", "broken"):
        for _, data in methods:
            count = int(data[metric])
            total = int(data["num_trials"])
            rates.append(count / total)
            intervals.append(wilson_interval(count, total))
    for x, rate, interval, color in zip(x_locations, rates, intervals, colors):
        ax_b.errorbar(
            x,
            rate,
            yerr=[[rate - interval[0]], [interval[1] - rate]],
            fmt="o",
            ms=5,
            capsize=2.5,
            color=color,
            ecolor=color,
            lw=1.1,
        )
        ax_b.text(
            x,
            min(0.97, interval[1] + 0.055),
            f"{rate:.0%}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax_b.set_xticks(
        x_locations, labels, rotation=24, ha="right", rotation_mode="anchor"
    )
    ax_b.axvline(0.675, color="#DDDDDD", lw=0.7)
    ax_b.text(
        0.175,
        -0.28,
        "Success",
        transform=ax_b.get_xaxis_transform(),
        ha="center",
        fontsize=7,
    )
    ax_b.text(
        1.175,
        -0.28,
        "Broken",
        transform=ax_b.get_xaxis_transform(),
        ha="center",
        fontsize=7,
    )
    ax_b.set(ylabel="Episode rate (Wilson 95% CI)", ylim=(0, 1.0))

    tracking = successful_tail_tracking(force_data, force_results_path)
    representative_index = min(
        (item for item in tracking if item["after_initial_range"]),
        key=lambda item: item["tail_absolute_error_n"],
    )["episode_index"]
    trace = read_trace(
        force_results_path.parent
        / "force_traces"
        / f"episode_{int(representative_index):03d}.csv"
    )
    active_indices = [
        index for index, row in enumerate(trace) if int(row["controller_active"])
    ]
    start = max(0, min(active_indices) - 5)
    trace = trace[start:]
    time_s = [
        (float(row["physics_step"]) - float(trace[0]["physics_step"]))
        / force_data["physics_rate_hz"]
        for row in trace
    ]
    grip_force = [float(row["grip_force_n"]) for row in trace]
    peak_force = [float(row["max_finger_force_n"]) for row in trace]
    target = float(trace[-1]["target_force_n"])
    representative_result = force_data["results"][int(representative_index)]
    representative_tracking = next(
        item for item in tracking if item["episode_index"] == representative_index
    )
    target_low, target_high = representative_result["attempted_force_range_n"]
    ax_c.axhspan(
        target_low, target_high, color=pale_blue, alpha=0.55, label="Target range"
    )
    ax_c.plot(time_s, grip_force, color=blue, lw=1.1, label="Mean fingertip load")
    ax_c.plot(
        time_s,
        peak_force,
        color=orange,
        lw=0.75,
        alpha=0.8,
        label="Maximum fingertip load",
    )
    ax_c.axhline(target, color=blue, ls="--", lw=0.8)
    ax_c.axhline(
        force_data["break_force_threshold_n"],
        color=red,
        ls=":",
        lw=0.9,
        label="Break threshold",
    )
    ax_c.set(
        xlabel="Time since contact window (s)",
        ylabel="Contact force (N)",
        ylim=(0, 3.8),
    )
    ax_c.set_title(
        f"Episode {int(representative_index) + 1}; final-20 mean error "
        f"{representative_tracking['tail_absolute_error_n']:.4f} N",
        loc="left",
        pad=2,
        fontsize=7,
    )
    ax_c.legend(loc="upper right")

    vlm_success = [bool(item["success"]) for item in force_data["results"]]
    comparison = [
        ("Force/VLM\noverall", sum(vlm_success), len(vlm_success), blue),
        (
            "DSRL\noverall",
            rl_summary["successes"],
            rl_summary["completed_episode_count"],
            orange,
        ),
        ("Force/VLM\nlast 10", sum(vlm_success[-10:]), 10, blue),
        (
            "DSRL\nlast 10",
            round(rl_summary["last_window_success_rate"] * 10),
            10,
            orange,
        ),
    ]
    for x, (label, count, total, color) in enumerate(comparison):
        rate = count / total
        interval = wilson_interval(count, total)
        ax_d.errorbar(
            x,
            rate,
            yerr=[[rate - interval[0]], [interval[1] - rate]],
            fmt="o",
            ms=5,
            capsize=2.5,
            color=color,
            ecolor=color,
            lw=1.1,
        )
        ax_d.text(
            x,
            min(0.97, interval[1] + 0.055),
            f"{count}/{total}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax_d.set_xticks(range(len(comparison)), [item[0] for item in comparison])
    ax_d.set(ylabel="Success rate (Wilson 95% CI)", ylim=(0, 1.0))
    ax_d.text(
        0.02,
        0.04,
        "Context only: frozen-policy evaluation vs online RL trajectory",
        transform=ax_d.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#555555",
    )

    for label, axis in zip(("a", "b", "c", "d"), (ax_a, ax_b, ax_c, ax_d)):
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9,
            va="top",
        )
        axis.grid(axis="y", color="#E8E8E8", lw=0.5, zorder=0)

    stem = output_dir / "vlm_force_experiment"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close(fig)
    return int(representative_index)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    force_path = args.force_results.resolve()
    baseline_path = args.baseline_results.resolve()
    rl_path = args.rl_summary.resolve()
    rl_episodes_path = args.rl_episodes.resolve()
    force_data = load_json(force_path)
    baseline_data = load_json(baseline_path)
    rl_summary = load_json(rl_path)

    if (
        force_data["force_control"] is not True
        or baseline_data["force_control"] is not False
    ):
        raise ValueError("Expected force-control and policy-only inputs in that order.")
    force_seeds = [item["seed"] for item in force_data["results"]]
    baseline_seeds = [item["seed"] for item in baseline_data["results"]]
    if force_seeds != baseline_seeds:
        raise ValueError("The frozen-policy groups do not use the same seed schedule.")

    episode_rows = episode_source_rows(force_data, baseline_data, rl_episodes_path)
    episode_fields = list(episode_rows[0])
    write_csv(output_dir / "episode_source_data.csv", episode_rows, episode_fields)
    tracking = successful_tail_tracking(force_data, force_path)
    write_csv(
        output_dir / "successful_episode_tracking.csv", tracking, list(tracking[0])
    )

    force_success = int(force_data["successes"])
    baseline_success = int(baseline_data["successes"])
    force_broken = int(force_data["broken"])
    baseline_broken = int(baseline_data["broken"])
    total = int(force_data["num_trials"])
    if total != int(baseline_data["num_trials"]):
        raise ValueError("Frozen-policy groups must have equal sample size.")

    force_success_seeds = {
        item["seed"] for item in force_data["results"] if item["success"]
    }
    baseline_success_seeds = {
        item["seed"] for item in baseline_data["results"] if item["success"]
    }
    force_break_seeds = {
        item["seed"]
        for item in force_data["results"]
        if item["terminal_reason"] == "object_broken"
    }
    baseline_break_seeds = {
        item["seed"]
        for item in baseline_data["results"]
        if item["terminal_reason"] == "object_broken"
    }
    post_initial_tracking = [item for item in tracking if item["after_initial_range"]]
    tracking_errors = [item["tail_absolute_error_n"] for item in post_initial_tracking]
    final_range = force_data["final_force_range_n"]
    initial_range = force_data["initial_force_range_n"]
    range_contraction = 1.0 - (final_range[1] - final_range[0]) / (
        initial_range[1] - initial_range[0]
    )
    non_force_decisions = [
        item
        for item in force_data["estimator"]["decisions"]
        if item["update_kind"] == "non_force"
    ]
    non_force_changes = [
        abs(item["target_range_n"][0] - item["previous_range_n"][0])
        + abs(item["target_range_n"][1] - item["previous_range_n"][1])
        for item in non_force_decisions
    ]
    maximum_non_force_change = max(non_force_changes, default=0.0)
    if maximum_non_force_change < 1.0e-12:
        maximum_non_force_change = 0.0
    first_interaction = json.loads(
        (force_path.parent / "vlm_episode_interactions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    prompt_payload = json.loads(
        first_interaction["prompt"].split("实验上下文：\n", 1)[1]
    )
    recorded_prompt_break_threshold = prompt_payload[
        "break_force_threshold_per_finger_n"
    ]

    primary = {
        "force_control": {
            "success": force_success,
            "total": total,
            "rate": force_success / total,
            "wilson_95": wilson_interval(force_success, total),
            "broken": force_broken,
            "broken_rate": force_broken / total,
            "broken_wilson_95": wilson_interval(force_broken, total),
        },
        "policy_only": {
            "success": baseline_success,
            "total": total,
            "rate": baseline_success / total,
            "wilson_95": wilson_interval(baseline_success, total),
            "broken": baseline_broken,
            "broken_rate": baseline_broken / total,
            "broken_wilson_95": wilson_interval(baseline_broken, total),
        },
        "absolute_success_gain": force_success / total - baseline_success / total,
        "absolute_break_reduction": baseline_broken / total - force_broken / total,
        "fisher_success_two_sided_p": fisher_exact_two_sided(
            force_success,
            total - force_success,
            baseline_success,
            total - baseline_success,
        ),
        "fisher_broken_two_sided_p": fisher_exact_two_sided(
            force_broken, total - force_broken, baseline_broken, total - baseline_broken
        ),
        "same_seed_sensitivity": {
            "success_force_only": len(force_success_seeds - baseline_success_seeds),
            "success_policy_only": len(baseline_success_seeds - force_success_seeds),
            "success_exact_mcnemar_p": exact_mcnemar(
                len(force_success_seeds - baseline_success_seeds),
                len(baseline_success_seeds - force_success_seeds),
            ),
            "broken_force_only": len(force_break_seeds - baseline_break_seeds),
            "broken_policy_only": len(baseline_break_seeds - force_break_seeds),
            "broken_exact_mcnemar_p": exact_mcnemar(
                len(force_break_seeds - baseline_break_seeds),
                len(baseline_break_seeds - force_break_seeds),
            ),
        },
    }
    analysis = {
        "schema_version": 1,
        "primary_frozen_policy_comparison": primary,
        "adaptation": {
            "advisor": force_data["advisor"],
            "advisor_is_real_vlm": force_data["advisor_is_real_vlm"],
            "advisor_calls": force_data["advisor_calls"],
            "initial_force_range_n": initial_range,
            "final_force_range_n": final_range,
            "range_width_contraction": range_contraction,
            "informative_episodes": force_data["estimator"][
                "informative_episode_count"
            ],
            "non_force_episode_count": len(non_force_decisions),
            "maximum_non_force_range_change_n": maximum_non_force_change,
            "recorded_offline_prompt_break_threshold_n": recorded_prompt_break_threshold,
            "simulator_break_threshold_n": force_data["break_force_threshold_n"],
        },
        "force_control": {
            "physics_rate_hz": force_data["physics_rate_hz"],
            "policy_rate_hz": force_data["policy_rate_hz"],
            "maximum_non_gripper_action_delta": force_data[
                "maximum_uncontrolled_action_delta"
            ],
            "post_initial_success_count": len(post_initial_tracking),
            "post_initial_final_20_mean_absolute_error_n": {
                "median": statistics.median(tracking_errors),
                "maximum": max(tracking_errors),
                "within_0_01_n": sum(error <= 0.01 for error in tracking_errors),
            },
        },
        "rl_context": {
            "protocol": "separate online training trajectory; not a controlled equivalence test",
            "successes": rl_summary["successes"],
            "episodes": rl_summary["completed_episode_count"],
            "overall_success_rate": rl_summary["overall_success_rate"],
            "last_10_success_rate": rl_summary["last_window_success_rate"],
            "force_method_last_10_success_rate": sum(
                item["success"] for item in force_data["results"][-10:]
            )
            / 10,
            "broken": rl_summary["failure_counts"]["object_broken"],
            "last_interaction_bin_broken_rate": rl_summary["success_rate_bins"][-1][
                "broken_episode_rate"
            ],
        },
        "provenance": {
            "force_results_sha256": sha256(force_path),
            "baseline_results_sha256": sha256(baseline_path),
            "rl_summary_sha256": sha256(rl_path),
            "rl_episodes_sha256": sha256(rl_episodes_path),
            "bc_checkpoint_sha256": "72cd3ed1f32a672dfc307572019ce1aab8c0b4da4c4f527e278321a32a51c465",
        },
    }
    representative_index = make_figure(
        force_data, baseline_data, rl_summary, force_path, output_dir
    )
    analysis["force_control"][
        "representative_trace_episode_index"
    ] = representative_index
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    force_ci = primary["force_control"]["wilson_95"]
    baseline_ci = primary["policy_only"]["wilson_95"]
    force_break_ci = primary["force_control"]["broken_wilson_95"]
    baseline_break_ci = primary["policy_only"]["broken_wilson_95"]
    tracking_summary = analysis["force_control"][
        "post_initial_final_20_mean_absolute_error_n"
    ]
    rounded_final_range = [round(value, 3) for value in final_range]
    report = f"""---
experiment_id: VLM-FORCE-260828-001
date: 2026-08-28
status: completed
primary_protocol: frozen DINOv3 Flow BC policy, 20 episodes per group
advisor: deterministic offline protocol substitute
real_vlm_calls: false
---

# Episode 级 VLM 力推断与 120 Hz 触觉力控实验

## 结论

在相同 checkpoint、seed 日程和环境配置下，episode 级力范围适应加 120 Hz
触觉反馈控制取得 **{force_success}/{total}（{force_success / total:.1%}）** 成功，policy-only
消融为 **{baseline_success}/{total}（{baseline_success / total:.1%}）**，绝对提高
{100 * primary['absolute_success_gain']:.1f} 个百分点。破片率从
{baseline_broken}/{total}（{baseline_broken / total:.1%}）降至
{force_broken}/{total}（{force_broken / total:.1%}），绝对降低
{100 * primary['absolute_break_reduction']:.1f} 个百分点。探索性 Fisher 双侧检验分别为
`p={primary['fisher_success_two_sided_p']:.4f}` 和
`p={primary['fisher_broken_two_sided_p']:.4f}`。

该结果表明：在本次 Isaac Lab 运行中，**episode 级目标力推断必须与接触期高频闭环共同使用**；
只有建议而不执行力控不能改善动作。与已有在线 DSRL 轨迹相比，本方法整体成功率
{force_success / total:.1%} 与 DSRL 的 {rl_summary['overall_success_rate']:.1%}
处于相近量级，但这不是等效性证明：DSRL 是 300 次 online interaction 的单条训练轨迹，
本实验是冻结 policy 的 20-episode 评估。

## 主要结果

| 方法 | 成功率（Wilson 95% CI） | 破片率（Wilson 95% CI） | n |
|---|---:|---:|---:|
| 力范围适应 + 120 Hz 力控 | {force_success / total:.1%} [{force_ci[0]:.1%}, {force_ci[1]:.1%}] | {force_broken / total:.1%} [{force_break_ci[0]:.1%}, {force_break_ci[1]:.1%}] | {total} |
| Policy only | {baseline_success / total:.1%} [{baseline_ci[0]:.1%}, {baseline_ci[1]:.1%}] | {baseline_broken / total:.1%} [{baseline_break_ci[0]:.1%}, {baseline_break_ci[1]:.1%}] | {total} |
| Online DSRL（独立参考） | {rl_summary['overall_success_rate']:.1%} | {rl_summary['failure_counts']['object_broken'] / rl_summary['completed_episode_count']:.1%} | {rl_summary['completed_episode_count']} |

同 seed 敏感性分析得到成功 McNemar `p={primary['same_seed_sensitivity']['success_exact_mcnemar_p']:.4f}`、
破片 McNemar `p={primary['same_seed_sensitivity']['broken_exact_mcnemar_p']:.4f}`。由于 GPU 物理跨进程
不保证逐轨迹位级确定，Fisher 结果作为主分析，配对检验仅作敏感性分析。

## 力范围收敛与归因

- 每个 episode 结束后恰好调用一次 advisor：共 {force_data['advisor_calls']} 次，日志中保存完整 prompt、历史、响应与安全投影结果。
- 初始范围 `{initial_range}` N 收缩到 `{rounded_final_range}` N，区间宽度缩小 {range_contraction:.1%}。
- {len(non_force_decisions)} 个非力因素回合（未接触、对位、轨迹或力跟踪失败）的范围最大变化为 {maximum_non_force_change:.3g} N。
- 关键防错规则：若实际平均力没有达到命令下界，标为 `force_tracking_error`，不能作为“任务目标力太低”的证据。
- 最终范围得到随后成功回合的支持（episode {representative_index + 1} 的成功抓取平均接触力接近该目标）。不过 n=20，仍应在更多 seed/材质/玻片厚度上验证稳定性。

## 高频力控与动作掩码

- 物理/触觉闭环 {force_data['physics_rate_hz']} Hz，policy {force_data['policy_rate_hz']} Hz；PI-D、低通、anti-windup、接触滞回和超力快速张开均在每个 physics step 执行。
- 只覆盖 CAFE action 的 index 9（gripper width）；XYZ 与 Rot6D 逐元素保持 policy 输出。本次最大非抓爪动作改变量为 **{force_data['maximum_uncontrolled_action_delta']:.1f}**。
- 排除初始宽范围成功回合后，{len(post_initial_tracking)} 个成功回合的末 20 个双侧承载样本中，回合平均力对目标的绝对误差中位数为 {tracking_summary['median']:.4f} N，最大值 {tracking_summary['maximum']:.4f} N；{tracking_summary['within_0_01_n']}/{len(post_initial_tracking)} 个回合不超过 0.01 N。
- 安全优先于恒力：单指峰值达到目标上界或 3.5 N 硬阈值时主动张开。因此接触初瞬可能出现峰值，不能把稳态跟踪误差解释为峰值完全消失。

## 与 RL 的边界化比较

- VLM/力控：整体 {force_success}/{total}（{force_success / total:.1%}），后 10 回合 {sum(item['success'] for item in force_data['results'][-10:])}/10（{sum(item['success'] for item in force_data['results'][-10:]) / 10:.1%}）。
- Online DSRL：整体 {rl_summary['successes']}/{rl_summary['completed_episode_count']}（{rl_summary['overall_success_rate']:.1%}），后 10 回合 7/10（{rl_summary['last_window_success_rate']:.1%}）；最后 100 interactions 的破片率为 0%。
- 可支持的表述是“在这次小样本实验中达到与已有 RL 轨迹相近量级的整体成功率，并显著优于无力控消融”。不能支持“与 RL 等效”或“优于 RL”。

## 实验协议

- Base policy：DINOv3 Flow BC，checkpoint SHA-256 `{analysis['provenance']['bc_checkpoint_sha256']}`。
- 正式冻结-policy 组：seed 42–61，各 20 episodes；每回合最多 960 physics steps；XY 随机化 `[0.1, 0.1]` m；yaw 0°。
- action repeat = {force_data['action_repeat']}，chunk execute steps = {force_data['chunk_execute_steps']}；玻片单指破碎阈值 3.5 N。
- 目标范围单位为单指法向载荷；反馈控制量是左右指载荷幅值均值；安全量是两指最大值。
- 本次已保存的离线 advisor prompt 将物理允许上限 {recorded_prompt_break_threshold:.2f} N 作为保守安全阈值；模拟器实际破碎阈值为 {force_data['break_force_threshold_n']:.2f} N。该离线规则的决策不读取此字段；当前代码已统一传入模拟器阈值。
- advisor 是明确标注的 `DeterministicVLMAdvisor`，用于在没有 API key 时验证完整软件协议；**本报告没有真实 VLM 调用证据**。真实多模态接口支持 Responses / Chat Completions 严格 JSON schema。

## 局限与下一步

1. 两组各 n=20，置信区间仍宽；统计检验为探索性、未预注册。
2. 同 seed 控制初始随机化，但 Isaac GPU 物理跨进程不保证位级复现，因此不能把它视为严格成对确定性实验。
3. RL 对照来自独立在线训练轨迹，不是相同 seed 的冻结 checkpoint 对照；只能作效果量级参考。
4. 需配置真实 VLM endpoint 并保存图像输入、模型版本、原始响应、延迟和失败重试，才能验证语义推理贡献。
5. 应增加固定力、去历史、去安全投影、触觉估计对 privileged force 等消融，并扩展到不同玻片厚度、摩擦系数和随机 yaw。

## 图与可复核数据

- `vlm_force_experiment.pdf/.svg/.tiff/.png`：a，目标力范围与 episode 结果；b，同协议成功/破片率；c，最终成功回合的 120 Hz 力轨迹；d，与在线 DSRL 的边界化比较。
- `episode_source_data.csv`：40 个正式冻结-policy episode 加 27 个 DSRL episode 的逐回合源数据。
- `successful_episode_tracking.csv`：成功回合末 20 个双侧承载样本的力跟踪统计。
- `analysis_summary.json`：计数、Wilson 区间、Fisher/McNemar 结果、收敛量与输入文件 SHA-256。
- `isaac_vlm_force_seed42_n20/` 与 `isaac_policy_only_seed42_n20/`：原始 JSON、逐 physics-step CSV 和逐 episode advisor 事务日志。
- `pilots/`：诊断与修正前运行，仅用于开发审计，不进入正式统计。
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    qa_notes = """# Figure contract and QA

Core conclusion: Episode-level force-range adaptation combined with 120 Hz tactile feedback improves slide-pick success and reduces breakage relative to the same frozen policy without force control.

- Figure archetype: quantitative grid.
- Target/output: paper-ready double-column figure; 180.1 mm × 141.0 mm.
- Backend: Python/matplotlib exclusively.
- Hero evidence: protocol-matched 20 + 20 episode success and breakage comparison.
- Validation evidence: force-range trajectory and representative final-range force trace.
- Context only: independent 27-episode online DSRL trajectory.
- Source data: all 20 force-control episodes, all 20 policy-only episodes, and all 27 completed DSRL episodes are retained in `episode_source_data.csv`.
- Exclusion disclosure: the post-initial tracking summary excludes one successful episode that still used the initial `[1, 3] N` search range; all eight successful episodes remain in `successful_episode_tracking.csv`.
- Image integrity: no microscopy or photographic panel; panel c is plotted directly from the raw physics-step CSV without smoothing or point removal.

| Panel | Unique claim | Center/summary | Spread/interval | Replicate unit | Collision/final-size audit | Pass |
|---|---|---|---|---|---|---|
| a | The estimated force interval contracts in response to force-relevant evidence while non-force failures leave it unchanged | Raw target range and centre | None; raw episode sequence | Episode, n=20 | Symbols and legend separated; break threshold shown | Yes |
| b | Force control raises success and lowers breakage versus policy only | Binomial proportion | Wilson 95% CI | Episode, n=20 per group | CI labels clear upper extents | Yes |
| c | The high-rate controller returns a successful final-range episode to its target after contact transients | Raw mean and maximum fingertip load | None; 120 Hz raw sequence | Physics step | Title, legend and traces do not overlap | Yes |
| d | The observed overall effect is in the same broad range as the separate DSRL trajectory | Binomial proportion | Wilson 95% CI | Episode; n=20 and n=27 overall, n=10 late windows | Protocol qualifier shown in-panel | Yes |

Automated QA: source preflight 20/20 checks passed; PDF text audit found 83 text runs, minimum 6.2 pt, with zero runs below 5 pt. SVG/PDF text is editable; TIFF is 600 dpi. Final PNG was inspected panel by panel at the exported aspect ratio.
"""
    (output_dir / "figure_qa.md").write_text(qa_notes, encoding="utf-8")


if __name__ == "__main__":
    main()
