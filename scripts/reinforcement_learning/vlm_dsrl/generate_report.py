#!/usr/bin/env python3
"""Aggregate the VLM/DSRL matrix and generate publication-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


METHOD_ORDER = ("joint", "joint_bilateral", "dsrl", "vlm", "base")
METHOD_LABELS = {
    "joint": "VLM + DSRL",
    "joint_bilateral": "VLM + DSRL (bilat.)",
    "dsrl": "DSRL only",
    "vlm": "VLM only",
    "base": "Frozen base",
}
METHOD_COLORS = {
    "joint": "#0F4D92",
    "joint_bilateral": "#7851A9",
    "dsrl": "#767676",
    "vlm": "#42949E",
    "base": "#C78A43",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("exp_report/vlm_with_dsrl"),
    )
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument(
        "--compile-pdf", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def wilson(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total < 1:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def rolling(values: list[float], window: int = 10) -> np.ndarray:
    output = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        output.append(sum(values[start : index + 1]) / (index - start + 1))
    return np.asarray(output, dtype=float)


def validate_trajectory(
    path: Path,
    *,
    episode_index: int,
    expected_steps: int,
    bilateral_gate: bool,
) -> None:
    line_count = 0
    first_activation: dict[str, Any] | None = None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_count, line in enumerate(stream, start=1):
            row = json.loads(line)
            if int(row.get("episode_index", -1)) != episode_index:
                raise ValueError(f"{path} contains a mismatched episode index.")
            if int(row.get("physics_step", -1)) != line_count:
                raise ValueError(
                    f"{path} contains a missing or reordered physics step."
                )
            if float(row.get("max_non_gripper_action_delta", math.inf)) > 1.0e-7:
                raise ValueError(f"{path} changed a non-gripper action dimension.")
            if first_activation is None and bool(row.get("controller_active")):
                first_activation = row
    if line_count != expected_steps:
        raise ValueError(
            f"{path} has {line_count} physics rows; expected {expected_steps}."
        )
    if bilateral_gate and first_activation is not None:
        if (
            first_activation.get("pre_bilateral_contact") is not True
            or float(first_activation.get("pre_left_force_n", -1.0)) < 0.01
            or float(first_activation.get("pre_right_force_n", -1.0)) < 0.01
        ):
            raise ValueError(
                f"{path} first activated before bilateral tactile/force contact."
            )


def load_runs(root: Path, expected_episodes: int) -> list[dict[str, Any]]:
    paths = sorted((root / "runs").glob("*/results.json"))
    if not paths:
        raise FileNotFoundError(f"No completed results found under {root / 'runs'}")
    runs = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("completed_episodes", -1)) != expected_episodes:
            raise ValueError(
                f"{path} has {payload.get('completed_episodes')} episodes; "
                f"expected {expected_episodes}."
            )
        if int(payload.get("trajectory_count", -1)) != expected_episodes:
            raise ValueError(f"{path} does not contain every trajectory.")
        expected_calls = 0 if payload["mode"] in {"dsrl", "base"} else expected_episodes
        if int(payload.get("advisor_calls", -1)) != expected_calls:
            raise ValueError(
                f"{path} violates the one-advisor-call-per-episode contract."
            )
        if float(payload.get("maximum_non_gripper_action_delta", math.inf)) > 1.0e-7:
            raise ValueError(f"{path} changed a non-gripper action dimension.")
        episodes = payload.get("results", [])
        if [item.get("episode_index") for item in episodes] != list(
            range(expected_episodes)
        ):
            raise ValueError(f"{path} has missing or reordered episode indices.")
        for episode in episodes:
            trajectory = Path(str(episode.get("trajectory", "")))
            if not trajectory.is_file():
                raise ValueError(
                    f"{path} references a missing trajectory: {trajectory}."
                )
            reason = str(episode.get("terminal_reason", "")).strip()
            diagnosed = str(episode.get("diagnosed_failure_reason", "")).strip()
            if not reason or not diagnosed:
                raise ValueError(f"{path} contains an episode without a reason.")
            validate_trajectory(
                trajectory,
                episode_index=int(episode["episode_index"]),
                expected_steps=int(episode["physics_steps"]),
                bilateral_gate=payload["mode"] == "joint_bilateral",
            )
            if payload["mode"] in {"joint", "joint_bilateral", "vlm"} and any(
                episode.get(key) is None
                for key in (
                    "attempted_force_range_n",
                    "vlm_recommended_force_range_n",
                    "next_force_range_n",
                )
            ):
                raise ValueError(f"{path} contains an episode without a force range.")
        if payload["mode"] == "joint_bilateral":
            if int(payload.get("bilateral_gate_violations", -1)) != 0:
                raise ValueError(f"{path} violates the bilateral activation contract.")
            for episode in episodes:
                if episode.get("bilateral_activation_contract_satisfied") is not True:
                    raise ValueError(
                        f"{path} has an invalid bilateral activation record."
                    )
                if episode.get("first_controller_activation_physics_step") is None:
                    continue
                if (
                    episode.get("first_controller_activation_bilateral") is not True
                    or float(
                        episode.get("first_controller_activation_left_force_n", -1.0)
                    )
                    < 0.01
                    or float(
                        episode.get("first_controller_activation_right_force_n", -1.0)
                    )
                    < 0.01
                ):
                    raise ValueError(
                        f"{path} first activated before bilateral tactile/force contact."
                    )
        payload["_path"] = str(path)
        runs.append(payload)

    conditions = {
        (float(run["break_force_threshold_n"]), str(run["mode"])) for run in runs
    }
    thresholds = sorted({condition[0] for condition in conditions})
    expected = {
        (threshold, method) for threshold in thresholds for method in METHOD_ORDER
    }
    if conditions != expected:
        raise ValueError(
            f"Incomplete experiment matrix: found={conditions}, expected={expected}"
        )
    hashes = {run["bc_checkpoint_sha256"] for run in runs}
    seeds = {int(run["seed"]) for run in runs}
    xy_ranges = {tuple(run["labware_random_xy_m"]) for run in runs}
    yaw_ranges = {float(run["labware_random_yaw_deg"]) for run in runs}
    if (
        len(hashes) != 1
        or len(seeds) != 1
        or len(xy_ranges) != 1
        or len(yaw_ranges) != 1
    ):
        raise ValueError(
            "Runs do not share the same checkpoint and environment setting."
        )
    return runs


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in sorted(
        runs,
        key=lambda item: (
            float(item["break_force_threshold_n"]),
            METHOD_ORDER.index(item["mode"]),
        ),
    ):
        episodes = run["results"]
        successes = sum(int(item["success"]) for item in episodes)
        broken = sum(
            int(item["terminal_reason"] == "object_broken") for item in episodes
        )
        success_low, success_high = wilson(successes, len(episodes))
        broken_low, broken_high = wilson(broken, len(episodes))
        peak_mean, peak_sd = mean_sd(
            [float(item["peak_contact_force_n"]) for item in episodes]
        )
        lift_mean, lift_sd = mean_sd([float(item["max_lift_m"]) for item in episodes])
        rows.append(
            {
                "threshold_n": float(run["break_force_threshold_n"]),
                "method": run["mode"],
                "episodes": len(episodes),
                "successes": successes,
                "success_rate": successes / len(episodes),
                "success_ci_low": success_low,
                "success_ci_high": success_high,
                "broken": broken,
                "broken_rate": broken / len(episodes),
                "broken_ci_low": broken_low,
                "broken_ci_high": broken_high,
                "mean_peak_force_n": peak_mean,
                "sd_peak_force_n": peak_sd,
                "mean_max_lift_m": lift_mean,
                "sd_max_lift_m": lift_sd,
                "outer_interactions": int(run["outer_interactions"]),
                "dsrl_gradient_updates": int(run["dsrl_gradient_updates"]),
                "advisor_calls": int(run["advisor_calls"]),
                "first_controller_activation_count": sum(
                    item.get("first_controller_activation_physics_step") is not None
                    for item in episodes
                ),
                "bilateral_gate_violations": (
                    int(run.get("bilateral_gate_violations", 0))
                    if run["mode"] == "joint_bilateral"
                    else None
                ),
                "final_force_range_n": run["final_force_range_n"],
            }
        )
    return rows


def write_source_data(
    root: Path, runs: list[dict[str, Any]], summary: list[dict[str, Any]]
) -> None:
    with (root / "episode_source_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "threshold_n",
            "method",
            "episode_index",
            "success",
            "terminal_reason",
            "diagnosed_failure_reason",
            "episode_return",
            "physics_steps",
            "outer_interactions",
            "peak_contact_force_n",
            "mean_contact_force_n",
            "force_rmse_n",
            "max_lift_m",
            "tactile_contact_fraction",
            "bilateral_contact_fraction",
            "controller_active_fraction",
            "attempted_force_min_n",
            "attempted_force_max_n",
            "vlm_force_min_n",
            "vlm_force_max_n",
            "next_force_min_n",
            "next_force_max_n",
            "first_controller_activation_physics_step",
            "first_controller_activation_bilateral",
            "first_controller_activation_left_force_n",
            "first_controller_activation_right_force_n",
            "bilateral_activation_contract_satisfied",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for episode in run["results"]:
                attempted = episode["attempted_force_range_n"] or [None, None]
                inferred = episode["vlm_recommended_force_range_n"] or [None, None]
                next_range = episode["next_force_range_n"] or [None, None]
                writer.writerow(
                    {
                        "threshold_n": run["break_force_threshold_n"],
                        "method": run["mode"],
                        "episode_index": episode["episode_index"],
                        "success": int(episode["success"]),
                        "terminal_reason": episode["terminal_reason"],
                        "diagnosed_failure_reason": episode["diagnosed_failure_reason"],
                        "episode_return": episode["episode_return"],
                        "physics_steps": episode["physics_steps"],
                        "outer_interactions": episode["outer_interactions"],
                        "peak_contact_force_n": episode["peak_contact_force_n"],
                        "mean_contact_force_n": episode["mean_contact_force_n"],
                        "force_rmse_n": episode["force_rmse_n"],
                        "max_lift_m": episode["max_lift_m"],
                        "tactile_contact_fraction": episode.get(
                            "tactile_contact_fraction"
                        ),
                        "bilateral_contact_fraction": episode.get(
                            "bilateral_contact_fraction"
                        ),
                        "controller_active_fraction": episode[
                            "controller_active_fraction"
                        ],
                        "attempted_force_min_n": attempted[0],
                        "attempted_force_max_n": attempted[1],
                        "vlm_force_min_n": inferred[0],
                        "vlm_force_max_n": inferred[1],
                        "next_force_min_n": next_range[0],
                        "next_force_max_n": next_range[1],
                        "first_controller_activation_physics_step": episode.get(
                            "first_controller_activation_physics_step"
                        ),
                        "first_controller_activation_bilateral": episode.get(
                            "first_controller_activation_bilateral"
                        ),
                        "first_controller_activation_left_force_n": episode.get(
                            "first_controller_activation_left_force_n"
                        ),
                        "first_controller_activation_right_force_n": episode.get(
                            "first_controller_activation_right_force_n"
                        ),
                        "bilateral_activation_contract_satisfied": episode.get(
                            "bilateral_activation_contract_satisfied"
                        ),
                    }
                )
    with (root / "condition_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    reasons = sorted(
        {
            item["terminal_reason"]
            for run in runs
            for item in run["results"]
            if not item["success"]
        }
    )
    with (root / "failure_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["threshold_n", "method", *reasons])
        for run in sorted(
            runs,
            key=lambda item: (
                float(item["break_force_threshold_n"]),
                METHOD_ORDER.index(item["mode"]),
            ),
        ):
            counts = Counter(
                item["terminal_reason"]
                for item in run["results"]
                if not item["success"]
            )
            writer.writerow(
                [run["break_force_threshold_n"], run["mode"]]
                + [counts.get(reason, 0) for reason in reasons]
            )


def make_figure(
    root: Path, runs: list[dict[str, Any]], summary: list[dict[str, Any]]
) -> None:
    thresholds = sorted({float(row["threshold_n"]) for row in summary})
    lookup = {(float(row["threshold_n"]), row["method"]): row for row in summary}
    run_lookup = {
        (float(run["break_force_threshold_n"]), run["mode"]): run for run in runs
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    width = 0.145
    x = np.arange(len(thresholds), dtype=float)

    for panel, metric, low_key, high_key, ylabel in (
        (
            axes[0, 0],
            "success_rate",
            "success_ci_low",
            "success_ci_high",
            "Success rate (%)",
        ),
        (
            axes[0, 1],
            "broken_rate",
            "broken_ci_low",
            "broken_ci_high",
            "Breakage rate (%)",
        ),
    ):
        for index, method in enumerate(METHOD_ORDER):
            values = np.asarray([lookup[(t, method)][metric] for t in thresholds])
            lows = np.asarray([lookup[(t, method)][low_key] for t in thresholds])
            highs = np.asarray([lookup[(t, method)][high_key] for t in thresholds])
            bars = panel.bar(
                x + (index - (len(METHOD_ORDER) - 1) / 2.0) * width,
                100.0 * values,
                width,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.5,
                label=METHOD_LABELS[method],
                yerr=np.vstack(
                    (
                        100.0 * np.maximum(values - lows, 0.0),
                        100.0 * np.maximum(highs - values, 0.0),
                    )
                ),
                capsize=2,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
            )
            for bar, value, high in zip(bars, values, highs):
                panel.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(104.0, 100.0 * high + 1.5),
                    f"{100 * value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                )
        panel.set_xticks(x, [f"{value:g}" for value in thresholds])
        panel.set_xlabel("Break-force threshold (N)")
        panel.set_ylabel(ylabel)
        panel.set_ylim(0, 108)
        panel.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncols=5,
        handlelength=1.0,
        columnspacing=0.8,
        fontsize=5.5,
    )

    representative = min(thresholds, key=lambda value: abs(value - 4.0))
    for method in METHOD_ORDER:
        outcomes = [
            float(item["success"])
            for item in run_lookup[(representative, method)]["results"]
        ]
        axes[1, 0].plot(
            np.arange(1, len(outcomes) + 1),
            100.0 * rolling(outcomes, window=10),
            color=METHOD_COLORS[method],
            linewidth=1.6,
            label=METHOD_LABELS[method],
        )
    axes[1, 0].set_xlabel("Online episode")
    axes[1, 0].set_ylabel("Rolling success rate (%)")
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[1, 0].text(
        0.98,
        0.04,
        f"10-episode window; threshold = {representative:g} N",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#4D4D4D",
    )

    for method in ("joint", "joint_bilateral", "vlm"):
        episodes = run_lookup[(representative, method)]["results"]
        ranges = np.asarray([item["attempted_force_range_n"] for item in episodes])
        centers = ranges.mean(axis=1)
        episode_axis = np.arange(1, len(episodes) + 1)
        color = METHOD_COLORS[method]
        axes[1, 1].fill_between(
            episode_axis,
            ranges[:, 0],
            ranges[:, 1],
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        axes[1, 1].plot(
            episode_axis,
            centers,
            color=color,
            linewidth=1.5,
            label=METHOD_LABELS[method],
        )
    axes[1, 1].axhline(
        representative,
        color="#B64342",
        linestyle="--",
        linewidth=0.9,
        label="Break threshold",
    )
    axes[1, 1].set_xlabel("Online episode")
    axes[1, 1].set_ylabel("Target contact force (N)")
    axes[1, 1].grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[1, 1].legend(loc="best")

    for label, axis in zip("abcd", axes.flat):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    figure_base = root / "vlm_dsrl_comparison"
    fig.savefig(figure_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _best_statement(summary: list[dict[str, Any]]) -> tuple[str, str]:
    best_by_threshold = []
    for threshold in sorted({row["threshold_n"] for row in summary}):
        candidates = [row for row in summary if row["threshold_n"] == threshold]
        best = max(
            candidates, key=lambda row: (row["success_rate"], -row["broken_rate"])
        )
        best_by_threshold.append(
            f"{threshold:g} N 时 {METHOD_LABELS[best['method']]} 为 "
            f"{100 * best['success_rate']:.1f}%"
        )
    success_means = {
        method: statistics.mean(
            row["success_rate"] for row in summary if row["method"] == method
        )
        for method in METHOD_ORDER
    }
    break_means = {
        method: statistics.mean(
            row["broken_rate"] for row in summary if row["method"] == method
        )
        for method in METHOD_ORDER
    }
    success_text = "、".join(
        f"{METHOD_LABELS[method]} {100 * success_means[method]:.1f}%"
        for method in METHOD_ORDER
    )
    break_text = "、".join(
        f"{METHOD_LABELS[method]} {100 * break_means[method]:.1f}%"
        for method in METHOD_ORDER
    )
    comparison = f"跨阈值平均成功率为：{success_text}；对应平均破碎率为：{break_text}。"
    return "；".join(best_by_threshold) + "。", comparison


def _signed_percentage_points(value: float) -> str:
    return f"{100 * value:+.1f}"


def _english_list(items: list[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def write_reports(
    root: Path, runs: list[dict[str, Any]], summary: list[dict[str, Any]]
) -> None:
    thresholds = sorted({row["threshold_n"] for row in summary})
    _, comparison_text = _best_statement(summary)
    row_lookup = {
        (float(row["threshold_n"]), str(row["method"])): row for row in summary
    }
    if {3.5, 4.0, 4.5}.issubset(thresholds):
        threshold_tradeoff_text = (
            "相对冻结 base，联合方法在 3.5 N 将成功率从 "
            f"{100 * row_lookup[(3.5, 'base')]['success_rate']:.1f}% 提高到 "
            f"{100 * row_lookup[(3.5, 'joint')]['success_rate']:.1f}%，并将破碎率从 "
            f"{100 * row_lookup[(3.5, 'base')]['broken_rate']:.1f}% 降至 "
            f"{100 * row_lookup[(3.5, 'joint')]['broken_rate']:.1f}%；但在 4 N 和 "
            f"4.5 N，其成功率分别为 {100 * row_lookup[(4.0, 'joint')]['success_rate']:.1f}% "
            f"和 {100 * row_lookup[(4.5, 'joint')]['success_rate']:.1f}%，低于 base 的 "
            f"{100 * row_lookup[(4.0, 'base')]['success_rate']:.1f}% 和 "
            f"{100 * row_lookup[(4.5, 'base')]['success_rate']:.1f}%。"
            "因此方法排序随破坏阈值反转，没有一种增强策略在所有阈值上占优。"
        )
        threshold_tradeoff_english = (
            "Relative to the frozen base, the joint controller increased success at "
            f"3.5 N from {100 * row_lookup[(3.5, 'base')]['success_rate']:.1f}% to "
            f"{100 * row_lookup[(3.5, 'joint')]['success_rate']:.1f}% and reduced "
            f"breakage from {100 * row_lookup[(3.5, 'base')]['broken_rate']:.1f}% to "
            f"{100 * row_lookup[(3.5, 'joint')]['broken_rate']:.1f}%. At 4 and 4.5 N, "
            f"however, joint success was {100 * row_lookup[(4.0, 'joint')]['success_rate']:.1f}% "
            f"and {100 * row_lookup[(4.5, 'joint')]['success_rate']:.1f}%, below the base "
            f"rates of {100 * row_lookup[(4.0, 'base')]['success_rate']:.1f}% and "
            f"{100 * row_lookup[(4.5, 'base')]['success_rate']:.1f}%. Thus, method "
            "rankings reversed with the break-force threshold and no augmented method "
            "dominated across all settings."
        )
    else:
        threshold_tradeoff_text = ""
        threshold_tradeoff_english = ""
    force_contracts = []
    for threshold in thresholds:
        threshold_runs = [
            run
            for run in runs
            if float(run["break_force_threshold_n"]) == threshold
            and run["mode"] in {"joint", "joint_bilateral", "vlm"}
        ]
        physical_ranges = {
            tuple(float(value) for value in run["physical_force_range_n"])
            for run in threshold_runs
        }
        initial_ranges = {
            tuple(float(value) for value in run["initial_force_range_n"])
            for run in threshold_runs
        }
        if len(physical_ranges) != 1 or len(initial_ranges) != 1:
            raise ValueError(
                f"Threshold {threshold:g} N has inconsistent force ranges."
            )
        physical = next(iter(physical_ranges))
        initial = next(iter(initial_ranges))
        force_contracts.append(
            f"{threshold:g} N 阈值：允许 {physical[0]:g}–{physical[1]:g} N，"
            f"初始 {initial[0]:g}–{initial[1]:g} N"
        )
    force_contract_text = "；".join(force_contracts) + "。"
    low_threshold_rows = [row for row in summary if row["threshold_n"] == 2.0]
    if len(low_threshold_rows) == len(METHOD_ORDER):
        low_lookup = {row["method"]: row for row in low_threshold_rows}
        low_threshold_text = (
            "在 2 N 条件下，联合、双侧门控联合、DSRL-only、VLM-only 与冻结 base "
            "的成功率分别为 "
            f"{100 * low_lookup['joint']['success_rate']:.1f}%、"
            f"{100 * low_lookup['joint_bilateral']['success_rate']:.1f}%、"
            f"{100 * low_lookup['dsrl']['success_rate']:.1f}%、"
            f"{100 * low_lookup['vlm']['success_rate']:.1f}% 和 "
            f"{100 * low_lookup['base']['success_rate']:.1f}%；破碎率分别为 "
            f"{100 * low_lookup['joint']['broken_rate']:.1f}%、"
            f"{100 * low_lookup['joint_bilateral']['broken_rate']:.1f}%、"
            f"{100 * low_lookup['dsrl']['broken_rate']:.1f}%、"
            f"{100 * low_lookup['vlm']['broken_rate']:.1f}% 和 "
            f"{100 * low_lookup['base']['broken_rate']:.1f}%。"
        )
        low_threshold_english = (
            "At 2 N, success rates for the joint, bilateral-gated joint, DSRL-only, "
            "VLM-only, and frozen-base "
            f"conditions were {100 * low_lookup['joint']['success_rate']:.1f}%, "
            f"{100 * low_lookup['joint_bilateral']['success_rate']:.1f}%, "
            f"{100 * low_lookup['dsrl']['success_rate']:.1f}%, "
            f"{100 * low_lookup['vlm']['success_rate']:.1f}% and "
            f"{100 * low_lookup['base']['success_rate']:.1f}%, respectively; "
            f"breakage rates were {100 * low_lookup['joint']['broken_rate']:.1f}%, "
            f"{100 * low_lookup['joint_bilateral']['broken_rate']:.1f}%, "
            f"{100 * low_lookup['dsrl']['broken_rate']:.1f}%, "
            f"{100 * low_lookup['vlm']['broken_rate']:.1f}% and "
            f"{100 * low_lookup['base']['broken_rate']:.1f}%."
        )
    else:
        low_threshold_text = ""
        low_threshold_english = ""
    advisor_real = all(
        bool(run["advisor_is_real_vlm"])
        for run in runs
        if run["mode"] in {"joint", "joint_bilateral", "vlm"}
    )
    advisor_label = "真实多模态 VLM" if advisor_real else "确定性离线 VLM 协议替代器"
    table_lines = [
        "| 阈值 (N) | 方法 | 成功率 (95% Wilson CI) | 破坏率 | 峰值力 (N, mean ± SD) |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in summary:
        table_lines.append(
            f"| {row['threshold_n']:g} | {METHOD_LABELS[row['method']]} | "
            f"{100 * row['success_rate']:.1f}% "
            f"({100 * row['success_ci_low']:.1f}–{100 * row['success_ci_high']:.1f}%) | "
            f"{100 * row['broken_rate']:.1f}% | "
            f"{row['mean_peak_force_n']:.2f} ± {row['sd_peak_force_n']:.2f} |"
        )
    success_means = {
        method: statistics.mean(
            row["success_rate"] for row in summary if row["method"] == method
        )
        for method in METHOD_ORDER
    }
    break_means = {
        method: statistics.mean(
            row["broken_rate"] for row in summary if row["method"] == method
        )
        for method in METHOD_ORDER
    }
    bilateral_delta_text = (
        "双侧门控相对原联合方法的逐阈值变化（百分点）为："
        + "；".join(
            f"{threshold:g} N 成功率 "
            f"{_signed_percentage_points(row_lookup[(threshold, 'joint_bilateral')]['success_rate'] - row_lookup[(threshold, 'joint')]['success_rate'])}、"
            f"破碎率 {_signed_percentage_points(row_lookup[(threshold, 'joint_bilateral')]['broken_rate'] - row_lookup[(threshold, 'joint')]['broken_rate'])}"
            for threshold in thresholds
        )
        + "。"
    )
    bilateral_delta_english = (
        "Relative to the original joint controller, bilateral gating changed "
        "success/breakage rates by "
        + "; ".join(
            f"{_signed_percentage_points(row_lookup[(threshold, 'joint_bilateral')]['success_rate'] - row_lookup[(threshold, 'joint')]['success_rate'])}/"
            f"{_signed_percentage_points(row_lookup[(threshold, 'joint_bilateral')]['broken_rate'] - row_lookup[(threshold, 'joint')]['broken_rate'])} percentage points at {threshold:g} N"
            for threshold in thresholds
        )
        + "."
    )
    bilateral_runs = [run for run in runs if run["mode"] == "joint_bilateral"]
    bilateral_activation_count = sum(
        episode.get("first_controller_activation_physics_step") is not None
        for run in bilateral_runs
        for episode in run["results"]
    )
    bilateral_episode_count = sum(len(run["results"]) for run in bilateral_runs)
    bilateral_gate_violations = sum(
        int(run.get("bilateral_gate_violations", 0)) for run in bilateral_runs
    )
    success_summary_english = _english_list(
        [
            f"{METHOD_LABELS[method]} ({100 * success_means[method]:.1f}%)"
            for method in METHOD_ORDER
        ]
    )
    break_summary_english = _english_list(
        [
            f"{METHOD_LABELS[method]} ({100 * break_means[method]:.1f}%)"
            for method in METHOD_ORDER
        ]
    )
    markdown = f"""# VLM 与 DSRL 联合触觉力控实验

## 结论

{threshold_tradeoff_text} {comparison_text} {bilateral_delta_text} {low_threshold_text} 这些结果来自单一在线训练 seed 的顺序 episode，因而支持本设置下的描述性比较，但不构成跨 seed 的统计显著性结论。

## 实验设置

- 方法：原 VLM + DSRL、增加首次双侧接触门控的 VLM + DSRL、DSRL only、VLM only，以及冻结原始 Flow BC（无 RL 后训练、无顾问、无力控）的 base policy。
- 每个方法—阈值条件：{summary[0]['episodes']} 个在线 episode；共 {sum(row['episodes'] for row in summary)} 个 episode。
- 破坏力阈值：{', '.join(f'{value:g} N' for value in thresholds)}。
- 力范围安全约束：{force_contract_text}
- 场景随机化：平面位置 ±0.10 m，偏航角固定为 0°；所有条件使用相同 BC checkpoint、初始 seed 和控制频率。
- DSRL：联合与 DSRL-only 条件在每个 outer interaction 记录 transition，并在 replay warm-up 后执行一次 SAC gradient update；联合方法的顾问推理在该终止 transition 的 DSRL 更新完成后执行。VLM-only 与 frozen-base 均采用原生 BC 推理且梯度更新为零。
- 力控：自由空间完全透传；接触后仅覆盖 CAFE 动作第 10 维（夹爪宽度），其余 9 维继续由 BC/DSRL 策略控制。双侧门控条件要求左右触觉均接触且左右单指力均达到 0.01 N 后才能首次激活；激活后的释放滞回及紧急过力张开保持不变。
- 顾问：{advisor_label}。`advisor_is_real_vlm={str(advisor_real).lower()}`。

## 定量结果

{chr(10).join(table_lines)}

![Comparison](vlm_dsrl_comparison.png)

图 1｜原联合策略、双侧门控联合策略、DSRL-only、VLM-only 与 frozen-base 在 {len(thresholds)} 个破坏力阈值下的在线结果。a，成功率与 95% Wilson 区间（n={summary[0]['episodes']} episodes/condition）。b，物体破坏率与 95% Wilson 区间。c，4 N 条件下 10-episode 滑动成功率；该曲线描述单次在线学习轨迹，不表示独立重复间不确定性。d，两个联合方法与 VLM-only 在 4 N 条件下实际用于各 episode 的目标力范围；实线为区间中心，阴影为上下界。

## 轨迹与审计

每个条件保存完整 `episodes.jsonl`、每个 episode 一份逐物理步 `trajectories/*.jsonl.gz`、运行配置、失败原因、动作、触觉接触、力、奖励与顾问建议力范围。所有联合/VLM 条件均满足“一 episode 一次顾问调用”，且非夹爪动作维度最大改变量不超过 `1e-7`。双侧门控的 {bilateral_activation_count}/{bilateral_episode_count} 个 episode 记录到力控首次激活，所有这些首次激活均满足双侧触觉与双侧单指力门槛，门控违例数为 {bilateral_gate_violations}。

## 解释边界

本实验检验的是一个 slide 抓取任务、一个 BC checkpoint、一个在线训练 seed 和 {len(thresholds)} 档仿真破坏阈值。确定性离线协议替代器验证了闭环算法与因果消融，但不能替代真实 VLM 的语义推理证据；正式投稿前应使用冻结的真实 VLM 模型和多个独立训练 seed 复现实验。

## 可直接用于论文的英文结果段落

Across {len(thresholds)} simulated break-force thresholds, mean online success rates were {success_summary_english}; corresponding mean breakage rates were {break_summary_english}. {threshold_tradeoff_english} {bilateral_delta_english} {low_threshold_english} Force feedback changed only gripper width after activation; the other nine action dimensions remained policy-controlled. In the bilateral condition, first activation required bilateral tactile contact and at least 0.01 N on each finger; all {bilateral_activation_count} recorded first activations satisfied this contract ({bilateral_gate_violations} violations). These single-seed runs used a deterministic protocol substitute, so they establish integration and threshold sensitivity, not cross-run generalization or VLM semantic reasoning.

## 可直接用于论文的英文结论段落

We combined episode-level force-range adaptation with online DSRL correction in a contact-aware control loop and evaluated both the original activation rule and a stricter rule that waits for bilateral tactile contact and per-finger force evidence. Comparisons with DSRL-only, VLM-only and a frozen pretrained-policy baseline record complete success and failure trajectories and reveal how the admissible contact-force interval evolves under matched task settings and multiple break-force thresholds. The present evidence is limited to simulated slide grasping, one pretrained policy and one training seed; evaluation with a frozen multimodal model, repeated training seeds and physical hardware is required before claiming general task-level force reasoning.
"""
    (root / "report.md").write_text(markdown, encoding="utf-8")

    latex_rows = []
    for row in summary:
        latex_rows.append(
            f"{row['threshold_n']:g} & {_latex_escape(METHOD_LABELS[row['method']])} & "
            f"{100 * row['success_rate']:.1f}\\% "
            f"[{100 * row['success_ci_low']:.1f}, {100 * row['success_ci_high']:.1f}] & "
            f"{100 * row['broken_rate']:.1f}\\% & "
            f"{row['mean_peak_force_n']:.2f} $\\pm$ {row['sd_peak_force_n']:.2f} \\\\"
        )
    tex = rf"""\documentclass[10pt]{{ctexart}}
\usepackage[a4paper,margin=20mm]{{geometry}}
\usepackage{{booktabs,graphicx,array,xcolor,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=blue}}
\title{{VLM 与 DSRL 联合触觉力控实验报告}}
\author{{TacEx reproducible experiment pipeline}}
\date{{}}
\begin{{document}}
\maketitle
\section*{{核心结论}}
{_latex_escape(threshold_tradeoff_text)} {_latex_escape(comparison_text)} {_latex_escape(bilateral_delta_text)} 结果来自每个条件一个顺序在线训练 seed，因而用于描述性比较，不作跨 seed 显著性声明。

\section*{{实验协议}}
每个方法--阈值条件运行 {summary[0]['episodes']} 个 episode，共 {sum(row['episodes'] for row in summary)} 个 episode。比较原 VLM + DSRL、双侧门控 VLM + DSRL、DSRL only、VLM only 和冻结原始 Flow BC；破坏力阈值为 {', '.join(f'{value:g} N' for value in thresholds)}。所有条件共享同一 BC checkpoint、初始 seed、平面位置随机化范围（$\pm 0.10$ m）和零偏航随机化。两个联合方法与 DSRL-only 在每个 outer interaction 后更新（replay warm-up 后）；联合方法仅在 episode 结束且该 transition 已完成 DSRL 更新后调用顾问。冻结 base 无 RL 后训练、无顾问且无力控。

力范围安全约束：{_latex_escape(force_contract_text)}

自由空间中力控不生效；检测到接触后只替换 10 维 CAFE 动作中的夹爪宽度，XYZ 和 Rot6D 始终由策略控制。双侧门控条件要求左右触觉均接触且左右单指力均达到 0.01 N 后才能首次激活；激活后的释放滞回及紧急过力张开保持不变。所有记录的非夹爪动作最大差值均不超过 10\textsuperscript{{−7}}。

\begin{{table}}[ht]
\centering
\caption{{五种方法在不同破坏力阈值下的结果。区间为二项比例的 95\% Wilson CI；峰值力报告 mean $\pm$ SD。}}
\scriptsize
\setlength{{\tabcolsep}}{{3.5pt}}
\renewcommand{{\arraystretch}}{{0.92}}
\begin{{tabular}}{{llccc}}
\toprule
阈值 (N) & 方法 & 成功率 [95\% CI] & 破坏率 & 峰值力 (N) \\
\midrule
{chr(10).join(latex_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[ht]
\centering
\includegraphics[width=0.98\linewidth]{{vlm_dsrl_comparison.pdf}}
\caption{{原联合策略、双侧门控联合策略、DSRL-only、VLM-only 与 frozen-base 在 {len(thresholds)} 个破坏阈值下的在线结果。a，成功率与 95\% Wilson 区间（$n={summary[0]['episodes']}$ episodes/condition）。b，物体破坏率与 95\% Wilson 区间。c，4 N 条件下 10-episode 滑动成功率；该曲线描述单次在线学习轨迹。d，两个联合方法与 VLM-only 在 4 N 条件下实际采用的目标接触力范围；实线为中心，阴影为范围。}}
\end{{figure}}

\section*{{可复现性与证据边界}}
每个条件均保存 episode 汇总、逐物理步压缩轨迹、失败原因、运行配置和力范围更新。联合/VLM 条件均满足每 episode 一次顾问调用。双侧门控的 {bilateral_activation_count}/{bilateral_episode_count} 个 episode 记录到力控首次激活，全部满足双侧触觉和双侧单指力门槛，门控违例数为 {bilateral_gate_violations}。本次使用的是\textbf{{确定性离线 VLM 协议替代器}}（advisor\_is\_real\_vlm=false），用于验证闭环软件与消融逻辑，不能作为真实 VLM 视觉推理性能的证据。结论还受限于单一 slide 任务、单一 checkpoint 和单个训练 seed；正式投稿应增加多个独立训练 seed、冻结的真实多模态模型以及实体机器人验证。

\section*{{Paper-ready Results}}
\scriptsize
Across {len(thresholds)} simulated break-force thresholds, mean online success rates were {_latex_escape(success_summary_english)}; corresponding mean breakage rates were {_latex_escape(break_summary_english)}. {_latex_escape(threshold_tradeoff_english)} {_latex_escape(bilateral_delta_english)} Force feedback changed only gripper width after activation; the other nine action dimensions remained policy-controlled. All {bilateral_activation_count} recorded first activations satisfied bilateral tactile contact and the 0.01 N per-finger threshold ({bilateral_gate_violations} violations).

\end{{document}}
"""
    (root / "report.tex").write_text(tex, encoding="utf-8")

    analysis = {
        "schema_version": 1,
        "core_conclusion": (
            f"{threshold_tradeoff_text} {comparison_text} {bilateral_delta_text} "
            f"{low_threshold_text}".strip()
        ),
        "figure_archetype": "quantitative grid",
        "backend": "python/matplotlib",
        "replicate_unit": "online episode within one sequential training run",
        "uncertainty": "95% Wilson interval for binomial episode outcomes",
        "advisor_is_real_vlm": advisor_real,
        "bilateral_activation_contract": {
            "required_tactile_contact": "left AND right",
            "minimum_per_finger_force_n": 0.01,
            "recorded_first_activations": bilateral_activation_count,
            "episodes": bilateral_episode_count,
            "violations": bilateral_gate_violations,
        },
        "trajectory_audit": {
            "files_parsed": sum(len(run["results"]) for run in runs),
            "checks": [
                "gzip JSONL parse",
                "physics-step count and ordering",
                "episode-index consistency",
                "non-gripper action pass-through",
                "bilateral first-activation tactile and per-finger force gate",
            ],
        },
        "result_allocation": {
            "main": [
                "success rate",
                "breakage rate",
                "bilateral-vs-original activation ablation",
                "threshold sensitivity",
            ],
            "supporting": [
                "rolling online success",
                "force-range convergence",
                "first-activation bilateral gate audit",
            ],
            "source_data": [
                "episode metrics",
                "failure counts",
                "first-activation per-finger forces",
                "full trajectories",
            ],
        },
        "limitations": [
            "one online training seed per condition",
            "single simulated slide task and pretrained checkpoint",
            "deterministic advisor substitute unless advisor_is_real_vlm is true",
        ],
    }
    (root / "analysis_manifest.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def compile_pdf(root: Path) -> None:
    command = [
        "xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(root),
        str(root / "report.tex"),
    ]
    for _ in range(2):
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            (root / "latex_error.log").write_text(completed.stdout, encoding="utf-8")
            raise RuntimeError(f"xelatex failed; see {root / 'latex_error.log'}")


def main() -> None:
    args = parse_args()
    root = args.input_root.expanduser().resolve()
    runs = load_runs(root, args.expected_episodes)
    summary = summarize(runs)
    write_source_data(root, runs, summary)
    make_figure(root, runs, summary)
    write_reports(root, runs, summary)
    if args.compile_pdf:
        compile_pdf(root)
    print(f"[REPORT] {root / 'report.pdf'}")


if __name__ == "__main__":
    main()
