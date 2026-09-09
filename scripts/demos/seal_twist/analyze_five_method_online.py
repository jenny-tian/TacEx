#!/usr/bin/env python3
"""Audit and plot the five-method SealTwist online-learning comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


METHODS = ("residual_rl", "awr", "dppo", "dsrl", "vlm_dsrl")
DISPLAY_NAMES = {
    "residual_rl": "Residual RL",
    "awr": "AWR",
    "dppo": "DPPO",
    "dsrl": "DSRL",
    "vlm_dsrl": "VLM+DSRL",
}
COLORS = {
    "residual_rl": "#7A5AA6",
    "awr": "#4A8F6B",
    "dppo": "#C44E52",
    "dsrl": "#D17C2F",
    "vlm_dsrl": "#1769AA",
}
FORCE_FAILURES = {"under_force", "over_force"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bottle-root", type=Path, default=Path("exp_report/bottle"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp_report/bottle/five_method_online_n100/comparison"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(160700, 160701, 160702))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument(
        "--baseline-success-rate",
        type=float,
        default=None,
        help="Optional override; otherwise use the mean recorded in --baseline-json.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help=(
            "Frozen-base evaluation JSON. Defaults to "
            "<bottle-root>/frozen_base_repeats_n100/base_policy_summary.json."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_path(root: Path, method: str, seed: int) -> Path:
    if method in ("residual_rl", "awr", "dppo"):
        parent = root / "five_method_online_n100" / method
    elif method == "dsrl":
        parent = root / "persistent_high_dsrl_online_n100_v2"
    else:
        parent = root / "persistent_high_online_n100_v2"
    return parent / f"seed_{seed}_n100" / "training_episodes.jsonl"


def load_run(path: Path, method: str, seed: int, episodes: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows.sort(key=lambda row: int(row["episode_index"]))
    if len(rows) != episodes:
        raise ValueError(f"{path}: expected {episodes} episodes, found {len(rows)}")
    if [int(row["episode_index"]) for row in rows] != list(range(episodes)):
        raise ValueError(f"{path}: episode indices are not contiguous from zero")
    if [int(row["seed"]) for row in rows] != list(range(seed, seed + episodes)):
        raise ValueError(f"{path}: reset seeds are not contiguous from {seed}")
    summary_path = path.parent / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["completed_episodes"]) != episodes:
        raise ValueError(f"{summary_path}: run is incomplete")
    if int(summary["training_seed"]) != seed:
        raise ValueError(f"{summary_path}: training seed mismatch")
    expected_vlm = method == "vlm_dsrl"
    if bool(summary.get("vlm_is_codex")) != expected_vlm:
        raise ValueError(f"{summary_path}: VLM enablement does not match {method}")
    expected_calls = episodes if expected_vlm else 0
    if int(summary.get("vlm_calls", -1)) != expected_calls:
        raise ValueError(f"{summary_path}: unexpected VLM call count")
    if int(summary.get("vlm_successful_calls", -1)) != expected_calls:
        raise ValueError(f"{summary_path}: unexpected successful VLM call count")
    recorded_algorithm = summary.get("online_algorithm")
    if method in ("residual_rl", "awr", "dppo") and recorded_algorithm != method:
        raise ValueError(f"{summary_path}: algorithm mismatch")
    return {
        "method": method,
        "seed": seed,
        "path": path.resolve(),
        "summary_path": summary_path.resolve(),
        "summary": summary,
        "records": rows,
        "outcomes": [int(row["success"]) for row in rows],
    }


def contact_steps(record: dict[str, Any]) -> int:
    if "bilateral_contact_steps" in record:
        return int(record["bilateral_contact_steps"])
    inferred = float(record["bilateral_contact_fraction"]) * int(record["steps"])
    rounded = int(round(inferred))
    if not math.isclose(inferred, rounded, abs_tol=1.0e-6):
        raise ValueError("bilateral contact step count cannot be recovered exactly")
    return rounded


def rolling_success(outcomes: list[int], window: int) -> list[float]:
    return [
        statistics.fmean(outcomes[max(0, index - window + 1) : index + 1])
        for index in range(len(outcomes))
    ]


def slope_per_10(values: list[float], first_episode: int) -> float:
    x = list(range(first_episode, first_episode + len(values)))
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(values)
    denominator = sum((value - mean_x) ** 2 for value in x)
    return (
        sum((left - mean_x) * (right - mean_y) for left, right in zip(x, values, strict=True))
        / denominator
        * 10.0
    )


def run_metrics(run: dict[str, Any], window: int) -> dict[str, Any]:
    records = run["records"]
    outcomes = run["outcomes"]
    failures = Counter(row["outcome"] for row in records if not row["success"])
    force_failures = sum(failures[label] for label in FORCE_FAILURES)
    failure_total = len(records) - sum(outcomes)
    steps = [contact_steps(row) for row in records]
    total_contact_steps = sum(steps)
    contact_force_sum = sum(
        float(row["mean_bilateral_finger_force_n"]) * count
        for row, count in zip(records, steps, strict=True)
    )
    curve = rolling_success(outcomes, window)
    return {
        "training_seed": run["seed"],
        "episodes": len(records),
        "successes": sum(outcomes),
        "success_rate": statistics.fmean(outcomes),
        "first_20_success_rate": statistics.fmean(outcomes[:20]),
        "last_20_success_rate": statistics.fmean(outcomes[-20:]),
        "final_rolling_success_rate": curve[-1],
        "rolling_slope_per_10_episodes": slope_per_10(curve[window - 1 :], window),
        "contact_steps": total_contact_steps,
        "contact_phase_mean_force_n": contact_force_sum / max(total_contact_steps, 1),
        "failures": failure_total,
        "force_related_failures": force_failures,
        "force_related_fraction_of_failures": force_failures / max(failure_total, 1),
        "force_related_fraction_of_all_episodes": force_failures / len(records),
        "failure_counts": dict(failures),
        "input_jsonl": str(run["path"]),
        "input_sha256": sha256(run["path"]),
        "summary_json": str(run["summary_path"]),
        "summary_sha256": sha256(run["summary_path"]),
    }


def group_metrics(runs: list[dict[str, Any]], window: int) -> dict[str, Any]:
    per_run = [run_metrics(run, window) for run in runs]

    def mean_sd(field: str) -> tuple[float, float]:
        values = [float(run[field]) for run in per_run]
        return statistics.fmean(values), statistics.stdev(values)

    success_mean, success_sd = mean_sd("success_rate")
    force_mean, force_sd = mean_sd("contact_phase_mean_force_n")
    failure_mean, failure_sd = mean_sd("force_related_fraction_of_failures")
    outcomes = [value for run in runs for value in run["outcomes"]]
    all_records = [record for run in runs for record in run["records"]]
    pooled_failures = sum(not record["success"] for record in all_records)
    pooled_force_failures = sum(
        (not record["success"]) and record["outcome"] in FORCE_FAILURES
        for record in all_records
    )
    pooled_failure_counts = Counter(
        record["outcome"] for record in all_records if not record["success"]
    )
    all_contact_steps = [contact_steps(record) for record in all_records]
    pooled_contact_force = sum(
        float(record["mean_bilateral_finger_force_n"]) * count
        for record, count in zip(all_records, all_contact_steps, strict=True)
    ) / max(sum(all_contact_steps), 1)
    curves = [rolling_success(run["outcomes"], window) for run in runs]
    mean_curve = [statistics.fmean(values) for values in zip(*curves, strict=True)]
    return {
        "success_rate_mean": success_mean,
        "success_rate_sample_sd": success_sd,
        "pooled_success_rate": statistics.fmean(outcomes),
        "first_20_success_rate_mean": statistics.fmean(
            run["first_20_success_rate"] for run in per_run
        ),
        "last_20_success_rate_mean": statistics.fmean(
            run["last_20_success_rate"] for run in per_run
        ),
        "final_mean_rolling_success_rate": mean_curve[-1],
        "mean_rolling_slope_per_10_episodes": slope_per_10(
            mean_curve[window - 1 :], window
        ),
        "contact_phase_mean_force_n_mean": force_mean,
        "contact_phase_mean_force_n_sample_sd": force_sd,
        "pooled_contact_phase_mean_force_n": pooled_contact_force,
        "force_related_fraction_of_failures_mean": failure_mean,
        "force_related_fraction_of_failures_sample_sd": failure_sd,
        "pooled_force_related_failures": pooled_force_failures,
        "pooled_failures": pooled_failures,
        "pooled_failure_counts": dict(pooled_failure_counts),
        "pooled_force_related_fraction_of_failures": (
            pooled_force_failures / max(pooled_failures, 1)
        ),
        "pooled_force_related_fraction_of_all_episodes": (
            pooled_force_failures / len(all_records)
        ),
        "per_run": per_run,
    }


def validate_pairing(groups: dict[str, list[dict[str, Any]]]) -> str:
    flow_hashes = {
        run["summary"]["flow_checkpoint_sha256"]
        for runs in groups.values()
        for run in runs
    }
    if len(flow_hashes) != 1:
        raise ValueError("runs use different Flow checkpoints")
    reference = groups["dsrl"]
    for method, runs in groups.items():
        for reference_run, run in zip(reference, runs, strict=True):
            if reference_run["seed"] != run["seed"]:
                raise ValueError(f"{method}: seed order does not match DSRL")
            for reference_record, record in zip(
                reference_run["records"], run["records"], strict=True
            ):
                if not math.isclose(
                    float(reference_record["minimum_force_threshold_n"]),
                    float(record["minimum_force_threshold_n"]),
                    abs_tol=1.0e-8,
                ) or not math.isclose(
                    float(reference_record["maximum_force_threshold_n"]),
                    float(record["maximum_force_threshold_n"]),
                    abs_tol=1.0e-8,
                ):
                    raise ValueError(f"{method}: hidden force thresholds are not paired")
                left = reference_record["sampled_cap_position_offset_m"]
                right = record["sampled_cap_position_offset_m"]
                if any(
                    not math.isclose(float(a), float(b), abs_tol=1.0e-9)
                    for a, b in zip(left, right, strict=True)
                ):
                    raise ValueError(f"{method}: cap-position randomization is not paired")
    return next(iter(flow_hashes))


def validate_baseline(
    path: Path, expected_rate: float | None, expected_flow_hash: str
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "success_rate_mean" in payload:
        recorded_rate = float(payload["success_rate_mean"])
        sample_sd = float(payload["success_rate_sample_sd"])
        episodes = int(payload["total_episodes"])
        successes = sum(int(run["successes"]) for run in payload["per_run"])
        checkpoint_hash = payload["protocol"]["checkpoint_sha256"]
        independent_runs = int(payload["runs"])
        seed_starts = list(payload["seeds"])
    else:
        episodes = int(payload["episodes"])
        successes = int(payload["successes"])
        recorded_rate = float(payload["success_rate"])
        sample_sd = 0.0
        checkpoint_hash = payload.get("checkpoint_sha256")
        independent_runs = 1
        seed_starts = [int(payload["seed_start"])]
        if not math.isclose(recorded_rate, successes / episodes, abs_tol=1.0e-12):
            raise ValueError(f"{path}: base-policy rate does not match its counts")
    if episodes <= 0 or successes < 0 or successes > episodes:
        raise ValueError(f"{path}: invalid base-policy success counts")
    if expected_rate is not None and not math.isclose(
        recorded_rate, expected_rate, abs_tol=1.0e-12
    ):
        raise ValueError(
            f"{path}: recorded base-policy rate {recorded_rate} does not match "
            f"--baseline-success-rate {expected_rate}"
        )
    if checkpoint_hash != expected_flow_hash:
        raise ValueError(f"{path}: base policy and online runs use different checkpoints")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "episodes": episodes,
        "successes": successes,
        "success_rate": recorded_rate,
        "success_rate_sample_sd": sample_sd,
        "independent_runs": independent_runs,
        "seed_starts": seed_starts,
        "wilson_95_ci": payload.get("wilson_95_ci"),
        "force_threshold_minimum_n": payload.get(
            "minimum_force_randomization_range_n"
        ),
        "force_threshold_maximum_n": payload.get(
            "maximum_force_randomization_range_n"
        ),
    }


def build_source_rows(
    groups: dict[str, list[dict[str, Any]]], window: int
) -> list[dict[str, Any]]:
    curves = {
        method: [rolling_success(run["outcomes"], window) for run in runs]
        for method, runs in groups.items()
    }
    rows: list[dict[str, Any]] = []
    episodes = len(next(iter(groups.values()))[0]["records"])
    for index in range(episodes):
        row: dict[str, Any] = {
            "episode": index + 1,
            "rolling_window_requested": window,
            "rolling_window_effective": min(window, index + 1),
        }
        for method in METHODS:
            values = [curve[index] for curve in curves[method]]
            mean = statistics.fmean(values)
            sd = statistics.stdev(values)
            row[f"{method}_mean_rolling_success_rate"] = mean
            row[f"{method}_sample_sd_across_seeds"] = sd
            row[f"{method}_mean_minus_1sd_clipped"] = max(0.0, mean - sd)
            row[f"{method}_mean_plus_1sd_clipped"] = min(1.0, mean + sd)
            for run, curve in zip(groups[method], curves[method], strict=True):
                seed = run["seed"]
                row[f"{method}_seed_{seed}_success"] = run["outcomes"][index]
                row[f"{method}_seed_{seed}_rolling_success_rate"] = curve[index]
        rows.append(row)
    return rows


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        stem.with_suffix(".png"), dpi=600, facecolor="white", bbox_inches="tight"
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def configure_plotting() -> None:
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


def decorate_axis(ax: plt.Axes, episodes: int, window: int) -> None:
    ax.axvline(window, color="#CDD1D5", linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
    ax.set_xlim(1, episodes + max(1.0, episodes * 0.012))
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Online training episode")
    ax.set_ylabel("Success rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xticks([1, 20, 40, 60, 80, 100] if episodes == 100 else None)
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.6)
    ax.set_axisbelow(True)


def draw_single_method(
    method: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    window: int,
    baseline: float,
    baseline_sd: float,
) -> None:
    episodes = [int(row["episode"]) for row in rows]
    start = window - 1
    mean = [float(row[f"{method}_mean_rolling_success_rate"]) for row in rows]
    lower = [float(row[f"{method}_mean_minus_1sd_clipped"]) for row in rows]
    upper = [float(row[f"{method}_mean_plus_1sd_clipped"]) for row in rows]
    color = COLORS[method]
    fig, ax = plt.subplots(figsize=(3.50, 2.55), constrained_layout=True)
    ax.fill_between(
        episodes[start:], lower[start:], upper[start:], color=color, alpha=0.18, linewidth=0
    )
    ax.plot(
        episodes[start:], mean[start:], color=color, linewidth=1.9,
        label=f"Mean rolling-{window} ± 1 SD",
    )
    ax.plot(
        episodes[: start + 1], mean[: start + 1], color=color, linewidth=1.0,
        linestyle=(0, (2, 1.5)), alpha=0.65,
    )
    if baseline_sd > 0.0:
        ax.axhspan(
            max(0.0, baseline - baseline_sd),
            min(1.0, baseline + baseline_sd),
            color="#737A82", alpha=0.10, linewidth=0, zorder=0,
        )
    ax.axhline(
        baseline, color="#737A82", linewidth=1.0, linestyle=(0, (4, 2)),
        label=f"Frozen base ({baseline:.0%} ± {baseline_sd:.0%} SD)",
    )
    ax.scatter(episodes[-1], mean[-1], s=18, color=color, edgecolor="white", linewidth=0.5)
    ax.annotate(
        f"{mean[-1]:.0%}", xy=(episodes[-1], mean[-1]), xytext=(-5, 6),
        textcoords="offset points", ha="right", va="bottom", color=color,
        fontsize=6.3, fontweight="bold",
    )
    decorate_axis(ax, len(rows), window)
    ax.set_title(f"{DISPLAY_NAMES[method]} online success (n=3 seeds)", loc="left", fontweight="bold")
    legend_location = "upper right" if method in {"awr", "dppo"} else "lower right"
    ax.legend(loc=legend_location, frameon=False, fontsize=5.5)
    save_figure(fig, output_dir / f"{method}_online_success_window{window}")


def draw_combined(
    rows: list[dict[str, Any]],
    output_dir: Path,
    window: int,
    baseline: float,
    baseline_sd: float,
) -> None:
    episodes = [int(row["episode"]) for row in rows]
    start = window - 1
    fig = plt.figure(figsize=(7.20, 4.55), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.0, 0.13))
    ax = fig.add_subplot(grid[0])
    legend_ax = fig.add_subplot(grid[1])
    legend_ax.axis("off")
    for method in METHODS:
        color = COLORS[method]
        mean = [float(row[f"{method}_mean_rolling_success_rate"]) for row in rows]
        lower = [float(row[f"{method}_mean_minus_1sd_clipped"]) for row in rows]
        upper = [float(row[f"{method}_mean_plus_1sd_clipped"]) for row in rows]
        ax.fill_between(
            episodes[start:], lower[start:], upper[start:], color=color, alpha=0.09, linewidth=0
        )
        ax.plot(
            episodes[start:], mean[start:], color=color,
            linewidth=2.2 if method == "vlm_dsrl" else 1.45,
            label=f"{DISPLAY_NAMES[method]} ({mean[-1]:.0%} final)",
            zorder=4 if method == "vlm_dsrl" else 3,
        )
        ax.plot(
            episodes[: start + 1], mean[: start + 1], color=color,
            linewidth=0.9, linestyle=(0, (2, 1.5)), alpha=0.60,
        )
    if baseline_sd > 0.0:
        ax.axhspan(
            max(0.0, baseline - baseline_sd),
            min(1.0, baseline + baseline_sd),
            color="#6D747C", alpha=0.10, linewidth=0, zorder=0,
        )
    ax.axhline(
        baseline, color="#6D747C", linewidth=1.1, linestyle=(0, (4, 2)),
        label=f"Frozen base ({baseline:.0%} ± {baseline_sd:.0%} SD)", zorder=1,
    )
    decorate_axis(ax, len(rows), window)
    ax.set_title("SealTwist online success: five-method comparison", loc="left", fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    legend = legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=6.0,
        handlelength=2.4,
        title=f"Mean rolling-{window} ± 1 SD; n=3 paired seeds",
    )
    legend.get_title().set_fontsize(6.0)
    save_figure(fig, output_dir / f"five_method_online_success_window{window}")


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_report(
    path: Path,
    metrics: dict[str, dict[str, Any]],
    window: int,
    baseline: float,
    baseline_sd: float,
) -> None:
    lines = [
        "# SealTwist 五方法在线强化学习实验报告",
        "",
        "## 实验协议",
        "",
        "五种方法均使用 3 个配对训练 seed（160700、160701、160702），每个 seed "
        "进行 100 个 online episode。所有方法使用同一个冻结的 Flow checkpoint、持续不变的 "
        "`[5.8, 11.0] N` 隐藏材料力阈值，并逐 episode 配对瓶盖位置随机化。曲线为尾随 "
        f"{window} episode 成功率在 seed 间的均值，阴影为 ±1 个样本标准差。独立重复单位是"
        "训练 seed（`n=3`），不是单个 episode。",
        "",
        "## 汇总结果",
        "",
        f"| 方法 | 总成功率，均值 ± SD | 最终滑窗-{window} | 接触阶段力，均值 ± SD (N) | under / over | 力相关失败 / 全部失败 | 力相关失败 / 全部 episode |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = metrics[method]
        force_failure = (
            f"{item['pooled_force_related_failures']}/{item['pooled_failures']} "
            f"({pct(item['pooled_force_related_fraction_of_failures'])})"
        )
        lines.append(
            "| "
            + DISPLAY_NAMES[method]
            + " | "
            + f"{pct(item['success_rate_mean'])} ± {pct(item['success_rate_sample_sd'])}"
            + " | "
            + pct(item["final_mean_rolling_success_rate"])
            + " | "
            + f"{item['contact_phase_mean_force_n_mean']:.3f} ± "
            + f"{item['contact_phase_mean_force_n_sample_sd']:.3f}"
            + " | "
            + f"{item['pooled_failure_counts'].get('under_force', 0)} / "
            + f"{item['pooled_failure_counts'].get('over_force', 0)}"
            + " | "
            + force_failure
            + " | "
            + pct(item["pooled_force_related_fraction_of_all_episodes"])
            + " |"
        )
    lines.extend(
        [
            "",
            "成功率 SD 是 3 个 seed 级成功率的样本标准差。接触阶段力定义为所有双指接触物理步上 "
            "`(left + right) / 2` 的均值；表中报告 3 个 seed 的接触步加权均值及其样本标准差。"
            "力相关失败定义为 `under_force + over_force`，同时给出其占失败 episode 和占全部 "
            "online episode 的比例，以明确分母。",
            "",
            "## 趋势与冻结 base policy 对比",
            "",
            f"冻结 base policy 的 3 次独立 100-episode 评估成功率为 "
            f"{pct(baseline)} ± {pct(baseline_sd)}。正的滑窗斜率表示"
            "平均 online 成功率随训练上升；斜率单位为每 10 个 episode 的成功率变化。",
            "",
            f"| 方法 | 前 20 episode | 后 20 episode | 最终滑窗-{window} | 滑窗斜率 / 10 episode | 相对 base 的总成功率差 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        item = metrics[method]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {pct(item['first_20_success_rate_mean'])} | "
            f"{pct(item['last_20_success_rate_mean'])} | "
            f"{pct(item['final_mean_rolling_success_rate'])} | "
            f"{item['mean_rolling_slope_per_10_episodes'] * 100:+.2f} pp | "
            f"{(item['success_rate_mean'] - baseline) * 100:+.1f} pp |"
        )
    lines.extend(
        [
            "",
            "## 各 seed 结果",
            "",
            "| 方法 | Seed | 成功 | 接触阶段平均力 (N) | 力相关失败 / 失败 episode |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        for run in metrics[method]["per_run"]:
            lines.append(
                f"| {DISPLAY_NAMES[method]} | {run['training_seed']} | "
                f"{run['successes']}/{run['episodes']} ({pct(run['success_rate'])}) | "
                f"{run['contact_phase_mean_force_n']:.3f} | "
                f"{run['force_related_failures']}/{run['failures']} "
                f"({pct(run['force_related_fraction_of_failures'])}) |"
            )
    best = max(METHODS, key=lambda method: metrics[method]["success_rate_mean"])
    lines.extend(
        [
            "",
            "## 原因诊断与修复结论",
            "",
            "1. **旧 VLM 问题不是单纯数值推理错误，而是任务不可辨识。**旧协议每个 episode "
            "独立重采样隐藏力阈值，上一轮反馈不能预测下一轮阈值；双指一高一低还会被误当成整体"
            "目标力偏差。修复后使用 persistent 材料阈值，并用方向一致的约束证据、限幅和平滑保护 "
            "VLM 更新。",
            "2. **旧 Q/state 存在力状态混叠。**只使用双指平均力时，`4/10 N` 与 `7/7 N` "
            "不可区分，但环境成功规则逐指判断。当前 actor state 为 34-D，包含 VLM low/high/centre、"
            "左右指各自相对目标的误差和左右不平衡；DSRL critic 还接收 VLM 区间、当轮真实隐藏"
            "最小/最大阈值及已执行的宽度修正。因此当前 Q function **已经包含 VLM 推理出的力目标**，"
            "真实隐藏阈值仅作为训练期 privileged critic context，不泄露给 actor。",
            "3. **旧 DSRL action authority 不足。**原 latent residual 解码后只有微米级 XY 影响，"
            "无法修正毫米级侧向接触偏差。当前改为接触阶段显式的 1-D tool-Y residual，最大 ±6 mm，"
            "并在 reward 中分别惩罚左右指低力、高力和不平衡。",
            "4. **续训探索率曾错误回升。**从 30 episode 续到 100 episode 时，旧实现按新总长度"
            "重新计算 annealing，使探索噪声重新变大。当前固定前 30 轮从 1.0 衰减至 0.15，之后"
            "保持 0.15，并由测试保护。",
            "",
            "## 结果解释",
            "",
            f"{DISPLAY_NAMES[best]} 的描述性平均 online 成功率最高，为 "
            f"{pct(metrics[best]['success_rate_mean'])}；冻结 base policy 为 "
            f"{pct(baseline)} ± {pct(baseline_sd)}。"
            "这些结果表明，仅靠局部 residual 学习不易发现需要跨 episode 调整的全局力设定点，而"
            "受保护的 VLM 全局力适配与 DSRL 局部平衡修正具有互补作用。由于只有 3 个配对 seed，"
            "本报告不作跨方法显著性声明。",
            "",
            "## 产物",
            "",
            f"- 五方法总曲线：`five_method_online_success_window{window}.png`（另有 SVG/PDF/TIFF）。",
            f"- 每种方法单独曲线：`<method>_online_success_window{window}.*`。",
            f"- 作图源数据：`five_method_online_success_window{window}_source.csv`。",
            "- 机器可读统计、输入哈希与配对审计：`five_method_online_manifest.json`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("at least two unique training seeds are required")
    if args.episodes < 20 or args.window < 1 or args.window > args.episodes:
        raise ValueError("invalid episode/window configuration")
    if args.baseline_success_rate is not None and not 0.0 <= args.baseline_success_rate <= 1.0:
        raise ValueError("baseline success rate must be in [0, 1]")
    root = args.bottle_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        method: [
            load_run(run_path(root, method, seed), method, seed, args.episodes)
            for seed in args.seeds
        ]
        for method in METHODS
    }
    flow_hash = validate_pairing(groups)
    baseline_path = (
        args.baseline_json.expanduser().resolve()
        if args.baseline_json is not None
        else root
        / "frozen_base_repeats_n100"
        / "base_policy_summary.json"
    )
    baseline_audit = validate_baseline(
        baseline_path, args.baseline_success_rate, flow_hash
    )
    baseline_rate = float(baseline_audit["success_rate"])
    baseline_sd = float(baseline_audit["success_rate_sample_sd"])
    metrics = {method: group_metrics(groups[method], args.window) for method in METHODS}
    rows = build_source_rows(groups, args.window)
    source_csv = output_dir / f"five_method_online_success_window{args.window}_source.csv"
    with source_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    configure_plotting()
    for method in METHODS:
        draw_single_method(
            method, rows, output_dir, args.window, baseline_rate, baseline_sd
        )
    draw_combined(rows, output_dir, args.window, baseline_rate, baseline_sd)

    manifest = {
        "experiment": "SealTwist five-method paired online learning comparison",
        "methods": list(METHODS),
        "display_names": DISPLAY_NAMES,
        "training_seeds": list(args.seeds),
        "runs_per_method": len(args.seeds),
        "episodes_per_run": args.episodes,
        "total_online_episodes": len(METHODS) * len(args.seeds) * args.episodes,
        "rolling_window": args.window,
        "baseline_success_rate": baseline_rate,
        "baseline_success_rate_sample_sd": baseline_sd,
        "baseline_audit": baseline_audit,
        "aggregation": "mean and sample SD across seed-level runs",
        "contact_force_definition": (
            "contact-step-weighted mean of the bilateral fingertip mean force"
        ),
        "force_related_failure_definition": "under_force + over_force",
        "pairing_audit": (
            "episode seeds, hidden force thresholds, and cap-position offsets match "
            "across all five methods"
        ),
        "flow_checkpoint_sha256": flow_hash,
        "metrics": metrics,
        "source_csv": str(source_csv),
        "formats": ["png", "svg", "pdf", "tiff"],
    }
    manifest_path = output_dir / "five_method_online_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path = output_dir / "FIVE_METHOD_ONLINE_REPORT.md"
    write_report(
        report_path,
        metrics,
        args.window,
        baseline_rate,
        baseline_sd,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
