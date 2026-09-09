#!/usr/bin/env python3
"""Validate, aggregate, and plot the formal six-method LabPick comparison."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ADAPTIVE_METHODS = ("residual_rl", "awr", "dppo", "dsrl", "rapid_tactile")
ALL_METHODS = ("base", *ADAPTIVE_METHODS)
DISPLAY_NAMES = {
    "base": "Base Policy",
    "residual_rl": "Residual RL",
    "awr": "AWR",
    "dppo": "DPPO",
    "dsrl": "DSRL",
    "rapid_tactile": "RAPID-Tactile",
}
COLORS = {
    "residual_rl": "#7A5AA6",
    "awr": "#4A8F6B",
    "dppo": "#C44E52",
    "dsrl": "#D17C2F",
    "rapid_tactile": "#1769AA",
}
EXPECTED_IDENTIFIERS = {
    "base": "base",
    "residual_rl": "residual_rl_tactile",
    "awr": "flow_rwr",
    "dppo": "dppo_tactile",
    "dsrl": "dsrl_tactile",
    "rapid_tactile": "ours_tactile",
}
SEEDS = (5200, 6200, 7200)
EPISODES = 100
WINDOW = 10
T_CRITICAL_95_DF2 = 4.302652729911275
BASE_COLOR = "#6D747C"
FORCE_CLASSIFICATION = {
    "object_broken": "over_force",
    "object_dropped": "under_force",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=repo_root / "exp_report" / "lab_pick",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=WINDOW)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def clipped_t_interval(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 3:
        raise ValueError("pointwise Student's t intervals require exactly three runs")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half_width = T_CRITICAL_95_DF2 * sd / math.sqrt(len(values))
    return mean, sd, max(0.0, mean - half_width), min(1.0, mean + half_width)


def trailing_success(outcomes: list[int], window: int) -> list[float]:
    if len(outcomes) != EPISODES or not 1 <= window <= len(outcomes):
        raise ValueError("invalid outcome count or trailing window")
    return [
        statistics.fmean(outcomes[end - window : end])
        for end in range(window, len(outcomes) + 1)
    ]


def force_category(success: bool, terminal_reason: str) -> str:
    if success:
        return "success"
    return FORCE_CLASSIFICATION.get(terminal_reason, "other_failure")


def method_identifier(metadata: dict[str, Any], result: dict[str, Any]) -> str:
    return str(
        metadata.get("method")
        or metadata.get("mode")
        or result.get("method")
        or result.get("mode")
        or ""
    )


def validate_common_metadata(method: str, metadata: dict[str, Any]) -> None:
    if int(metadata.get("num_episodes", -1)) != EPISODES:
        raise ValueError(f"{method}: metadata does not specify 100 episodes")
    if int(metadata.get("seed", -1)) not in SEEDS:
        raise ValueError(f"{method}: unexpected training seed")
    if metadata.get("phase") != "online_training":
        raise ValueError(f"{method}: run is not an online-training run")
    if "LabPick-Slide" not in str(metadata.get("task", "")):
        raise ValueError(f"{method}: wrong task {metadata.get('task')}")
    if not math.isclose(
        float(metadata.get("break_force_threshold_n", -1.0)), 4.5, abs_tol=1e-12
    ):
        raise ValueError(f"{method}: break-force threshold is not 4.5 N")
    if [float(value) for value in metadata.get("labware_random_xy_m", [])] != [0.1, 0.1]:
        raise ValueError(f"{method}: unexpected XY randomization")
    if not math.isclose(
        float(metadata.get("labware_random_yaw_deg", math.nan)), 0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{method}: unexpected yaw randomization")
    if method == "residual_rl":
        expected = {
            "residual_contract_version": 3,
            "residual_scale": 0.01,
            "residual_action_l2_weight": 1.0,
            "initial_log_std": -3.0,
            "learning_starts": 32,
            "batch_size": 32,
        }
        for name, wanted in expected.items():
            actual = metadata.get(name)
            if isinstance(wanted, float):
                valid = actual is not None and math.isclose(
                    float(actual), wanted, rel_tol=0.0, abs_tol=1.0e-12
                )
            else:
                valid = actual == wanted
            if not valid:
                raise ValueError(
                    f"{method}: {name}={actual!r}; expected {wanted!r}"
                )
        if metadata.get("residual_contact_gate") is not True:
            raise ValueError(f"{method}: residual contact gate is not enabled")
        rates = [float(value) for value in metadata.get("learning_rates", [])]
        if rates != [3.0e-5, 3.0e-5]:
            raise ValueError(
                f"{method}: learning rates are {rates}; expected [3e-5, 3e-5]"
            )


def run_directory(raw_root: Path, method: str, seed: int) -> Path:
    matches = sorted((raw_root / "runs" / method).glob(f"repeat_*_seed_{seed}"))
    if len(matches) != 1:
        raise ValueError(f"{method} seed {seed}: expected one run directory")
    return matches[0]


def validate_trajectory(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty trajectory: {path}")
    with gzip.open(path, "rb") as stream:
        if not stream.read(1):
            raise ValueError(f"empty compressed trajectory: {path}")


def load_run(raw_root: Path, method: str, repeat: int, seed: int) -> dict[str, Any]:
    directory = run_directory(raw_root, method, seed)
    result_path = directory / "results.json"
    episodes_path = directory / "episodes.jsonl"
    metadata_path = directory / "run_metadata.json"
    result = read_json(result_path)
    metadata = read_json(metadata_path)
    episode_rows = read_jsonl(episodes_path)
    validate_common_metadata(method, metadata)

    identifier = method_identifier(metadata, result)
    if identifier != EXPECTED_IDENTIFIERS[method]:
        raise ValueError(
            f"{method}: method identifier {identifier!r} does not match "
            f"{EXPECTED_IDENTIFIERS[method]!r}"
        )
    if int(metadata["seed"]) != seed:
        raise ValueError(f"{method}: metadata seed mismatch")
    records = result.get("results", [])
    if int(result.get("completed_episodes", -1)) != EPISODES:
        raise ValueError(f"{method} seed {seed}: incomplete results")
    if len(records) != EPISODES or len(episode_rows) != EPISODES:
        raise ValueError(f"{method} seed {seed}: expected 100 episode records")
    expected_indices = list(range(EPISODES))
    if [int(row["episode_index"]) for row in records] != expected_indices:
        raise ValueError(f"{method} seed {seed}: non-contiguous result indices")
    if [int(row["episode_index"]) for row in episode_rows] != expected_indices:
        raise ValueError(f"{method} seed {seed}: non-contiguous JSONL indices")

    clean_records: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(records, episode_rows, strict=True)):
        left_reason = str(left.get("terminal_reason", left.get("failure_reason", "")))
        right_reason = str(right.get("terminal_reason", right.get("failure_reason", "")))
        if bool(left["success"]) != bool(right["success"]) or left_reason != right_reason:
            raise ValueError(f"{method} seed {seed} episode {index}: JSON/JSONL mismatch")
        left_force = float(left.get("peak_contact_force_n", 0.0) or 0.0)
        right_force = float(right.get("peak_contact_force_n", 0.0) or 0.0)
        if not math.isclose(left_force, right_force, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{method} seed {seed} episode {index}: force mismatch")
        if not math.isfinite(left_force) or left_force < 0.0:
            raise ValueError(f"{method} seed {seed} episode {index}: invalid force")
        trajectory = directory / "trajectories" / f"episode_{index:03d}.jsonl.gz"
        validate_trajectory(trajectory)
        success = bool(left["success"])
        clean_records.append(
            {
                "episode": index + 1,
                "episode_index": index,
                "success": int(success),
                "terminal_reason": left_reason,
                "force_failure_category": force_category(success, left_reason),
                "peak_contact_force_n": left_force,
                "max_lift_m": float(left.get("max_lift_m", 0.0) or 0.0),
                "physics_steps": int(left.get("physics_steps", 0) or 0),
                "outer_interactions": int(left.get("outer_interactions", 0) or 0),
                "local_trajectory": str(trajectory.relative_to(raw_root)),
            }
        )

    successes = sum(row["success"] for row in clean_records)
    if successes != int(result["successes"]):
        raise ValueError(f"{method} seed {seed}: success count mismatch")
    return {
        "method": method,
        "repeat": repeat,
        "seed": seed,
        "directory": directory,
        "metadata": metadata,
        "records": clean_records,
        "outcomes": [row["success"] for row in clean_records],
        "hashes": {
            "results.json": sha256(result_path),
            "episodes.jsonl": sha256(episodes_path),
            "run_metadata.json": sha256(metadata_path),
            "resolved_env.yaml": sha256(directory / "resolved_env.yaml"),
        },
        "trajectory_count": EPISODES,
    }


def load_all_runs(raw_root: Path) -> dict[str, list[dict[str, Any]]]:
    groups = {
        method: [
            load_run(raw_root, method, repeat, seed)
            for repeat, seed in enumerate(SEEDS, start=1)
        ]
        for method in ALL_METHODS
    }
    for method, runs in groups.items():
        if [run["seed"] for run in runs] != list(SEEDS):
            raise ValueError(f"{method}: seed order mismatch")
    return groups


def run_metrics(run: dict[str, Any], window: int) -> dict[str, Any]:
    records = run["records"]
    outcomes = run["outcomes"]
    rolling = trailing_success(outcomes, window)
    failure_reasons = Counter(
        row["terminal_reason"] for row in records if not row["success"]
    )
    force_categories = Counter(row["force_failure_category"] for row in records)
    force_failures = force_categories["under_force"] + force_categories["over_force"]
    forces = [float(row["peak_contact_force_n"]) for row in records]
    return {
        "repeat": run["repeat"],
        "seed": run["seed"],
        "episodes": len(records),
        "successes": sum(outcomes),
        "success_rate": statistics.fmean(outcomes),
        "first_20_success_rate": statistics.fmean(outcomes[:20]),
        "last_20_success_rate": statistics.fmean(outcomes[-20:]),
        "nauc_complete_rolling_windows": statistics.fmean(rolling),
        "final_rolling_success_rate": rolling[-1],
        "mean_episode_peak_contact_force_n": statistics.fmean(forces),
        "max_episode_peak_contact_force_n": max(forces),
        "under_force_failures": force_categories["under_force"],
        "over_force_failures": force_categories["over_force"],
        "force_related_failures": force_failures,
        "force_related_failure_rate": force_failures / len(records),
        "failure_counts": dict(failure_reasons),
        "raw_directory": str(run["directory"]),
        "input_hashes": run["hashes"],
    }


def group_metrics(method: str, runs: list[dict[str, Any]], window: int) -> dict[str, Any]:
    per_run = [run_metrics(run, window) for run in runs]
    run_rates = [float(run["success_rate"]) for run in per_run]
    curves = [trailing_success(run["outcomes"], window) for run in runs]
    mean_curve = [statistics.fmean(values) for values in zip(*curves, strict=True)]
    all_records = [row for run in runs for row in run["records"]]
    failure_counts = Counter(
        row["terminal_reason"] for row in all_records if not row["success"]
    )
    force_categories = Counter(row["force_failure_category"] for row in all_records)
    force_failures = force_categories["under_force"] + force_categories["over_force"]
    t80 = None
    nauc = statistics.fmean(run_rates)
    if method != "base":
        nauc = statistics.fmean(mean_curve)
        t80 = next(
            (
                episode
                for episode, value in enumerate(mean_curve, start=window)
                if value >= 0.8 - 1e-12
            ),
            None,
        )
    return {
        "display_name": DISPLAY_NAMES[method],
        "runs": len(runs),
        "episodes_per_run": EPISODES,
        "total_episodes": len(all_records),
        "successes": sum(int(row["success"]) for row in all_records),
        "success_rate_mean": statistics.fmean(run_rates),
        "success_rate_sample_sd": statistics.stdev(run_rates),
        "nauc": nauc,
        "t80_episode": t80,
        "under_force_failures": force_categories["under_force"],
        "over_force_failures": force_categories["over_force"],
        "force_related_failures": force_failures,
        "force_related_failure_rate": force_failures / len(all_records),
        "other_failures": force_categories["other_failure"],
        "failure_counts": dict(failure_counts),
        "first_20_success_rate_mean": statistics.fmean(
            float(run["first_20_success_rate"]) for run in per_run
        ),
        "last_20_success_rate_mean": statistics.fmean(
            float(run["last_20_success_rate"]) for run in per_run
        ),
        "final_mean_rolling_success_rate": mean_curve[-1],
        "per_run": per_run,
    }


def episode_rows(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        for run in groups[method]:
            for record in run["records"]:
                rows.append(
                    {
                        "method": method,
                        "display_name": DISPLAY_NAMES[method],
                        "repeat": run["repeat"],
                        "training_seed": run["seed"],
                        **record,
                    }
                )
    return rows


def build_curve_rows(
    groups: dict[str, list[dict[str, Any]]], window: int
) -> list[dict[str, Any]]:
    curves = {
        method: [trailing_success(run["outcomes"], window) for run in groups[method]]
        for method in ALL_METHODS
    }
    rows: list[dict[str, Any]] = []
    for position, episode in enumerate(range(window, EPISODES + 1)):
        row: dict[str, Any] = {
            "online_episode": episode,
            "rolling_window": window,
            "independent_runs": 3,
        }
        for method in ALL_METHODS:
            values = [curve[position] for curve in curves[method]]
            mean, sd, lower, upper = clipped_t_interval(values)
            for seed, value in zip(SEEDS, values, strict=True):
                row[f"{method}_seed_{seed}_rolling_success_rate"] = value
            row[f"{method}_mean_rolling_success_rate"] = mean
            row[f"{method}_sample_sd"] = sd
            row[f"{method}_student_t_95_ci_lower"] = lower
            row[f"{method}_student_t_95_ci_upper"] = upper
        rows.append(row)
    return rows


def table_rows(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        item = metrics[method]
        rows.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "success_rate_mean_pct": 100.0 * item["success_rate_mean"],
                "success_rate_sample_sd_pct": 100.0 * item["success_rate_sample_sd"],
                "nauc_pct": 100.0 * item["nauc"],
                "t80_episode": "" if item["t80_episode"] is None else item["t80_episode"],
                "force_related_failure_rate_pct": 100.0 * item["force_related_failure_rate"],
                "under_force_failures": item["under_force_failures"],
                "over_force_failures": item["over_force_failures"],
                "other_failures": item["other_failures"],
                "successes": item["successes"],
                "total_episodes": item["total_episodes"],
            }
        )
    return rows


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.axisbelow": True,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def decorate_axis(axis: plt.Axes, window: int) -> None:
    axis.set_xlim(window - 0.5, EPISODES + 0.5)
    axis.set_ylim(-0.04, 1.04)
    axis.set_xticks([10, 20, 40, 60, 80, 100])
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_xlabel("Online training episode")
    axis.set_ylabel("Success rate")
    axis.grid(axis="y", color="#E7E9EC", linewidth=0.6)
    axis.tick_params(direction="out", pad=2)


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": "tight", "pad_inches": 0.02, "facecolor": "white"}
    figure.savefig(stem.with_suffix(".pdf"), **common)
    figure.savefig(stem.with_suffix(".svg"), **common)
    figure.savefig(stem.with_suffix(".png"), dpi=600, **common)
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        **common,
    )
    plt.close(figure)


def draw_single(
    method: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    baseline_metrics: dict[str, Any],
) -> None:
    x = [int(row["online_episode"]) for row in rows]
    mean = [float(row[f"{method}_mean_rolling_success_rate"]) for row in rows]
    lower = [float(row[f"{method}_student_t_95_ci_lower"]) for row in rows]
    upper = [float(row[f"{method}_student_t_95_ci_upper"]) for row in rows]
    baseline = float(baseline_metrics["success_rate_mean"])
    _, _, baseline_lower, baseline_upper = clipped_t_interval(
        [float(run["success_rate"]) for run in baseline_metrics["per_run"]]
    )
    color = COLORS[method]
    figure = plt.figure(figsize=(3.50, 2.62), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(1.0, 0.22))
    axis = figure.add_subplot(grid[0])
    legend_axis = figure.add_subplot(grid[1])
    legend_axis.axis("off")
    axis.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0)
    axis.plot(x, mean, color=color, linewidth=2.0, label=DISPLAY_NAMES[method])
    axis.axhspan(baseline_lower, baseline_upper, color=BASE_COLOR, alpha=0.10, linewidth=0)
    axis.axhline(
        baseline,
        color=BASE_COLOR,
        linewidth=1.1,
        linestyle=(0, (4, 2)),
        label="Frozen base",
    )
    decorate_axis(axis, int(rows[0]["rolling_window"]))
    handles, labels = axis.get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc="center", ncol=2, handlelength=2.2)
    save_figure(
        figure,
        output_dir / "figures" / f"{method}_online_success_window10",
    )


def draw_combined(
    rows: list[dict[str, Any]],
    output_dir: Path,
    baseline_metrics: dict[str, Any],
) -> None:
    x = [int(row["online_episode"]) for row in rows]
    baseline = float(baseline_metrics["success_rate_mean"])
    _, _, baseline_lower, baseline_upper = clipped_t_interval(
        [float(run["success_rate"]) for run in baseline_metrics["per_run"]]
    )
    figure = plt.figure(figsize=(3.50, 2.90), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(1.0, 0.34))
    axis = figure.add_subplot(grid[0])
    legend_axis = figure.add_subplot(grid[1])
    legend_axis.axis("off")
    for method in ADAPTIVE_METHODS:
        mean = [float(row[f"{method}_mean_rolling_success_rate"]) for row in rows]
        lower = [float(row[f"{method}_student_t_95_ci_lower"]) for row in rows]
        upper = [float(row[f"{method}_student_t_95_ci_upper"]) for row in rows]
        color = COLORS[method]
        axis.fill_between(x, lower, upper, color=color, alpha=0.09, linewidth=0)
        axis.plot(
            x,
            mean,
            color=color,
            linewidth=1.8 if method == "rapid_tactile" else 1.15,
            label=DISPLAY_NAMES[method],
            zorder=4 if method == "rapid_tactile" else 3,
        )
    axis.axhspan(baseline_lower, baseline_upper, color=BASE_COLOR, alpha=0.10, linewidth=0)
    axis.axhline(
        baseline,
        color=BASE_COLOR,
        linewidth=1.1,
        linestyle=(0, (4, 2)),
        label="Frozen base",
        zorder=1,
    )
    decorate_axis(axis, int(rows[0]["rolling_window"]))
    handles, labels = axis.get_legend_handles_labels()
    legend_axis.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        handlelength=2.0,
        columnspacing=1.3,
    )
    save_figure(
        figure,
        output_dir / "figures" / "five_method_online_success_window10",
    )


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_report(
    path: Path,
    metrics: dict[str, dict[str, Any]],
    source_selection: dict[str, Any],
) -> None:
    lines = [
        "# LabPick / Thin Sheet Transfer 在线对比实验报告",
        "",
        "## 实验协议",
        "",
        "六种方法均使用 3 个训练 seed（5200、6200、7200），每个 run 包含 100 个完整 "
        "online episode。成功率以 seed 为独立重复计算均值和样本标准差。nAUC 是三个 seed "
        "平均 trailing-10 成功曲线在 episode 10--100 上的均值。T80 是该平均曲线首次达到 "
        "80% 的实际 episode 编号。Frozen base 的 nAUC 按论文定义取其恒定成功率参考，T80 不适用。",
        "",
        "## 论文表格数值",
        "",
        "| 方法 | SR（mean ± SD） | nAUC | T80 | Force failure | under / over |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ALL_METHODS:
        item = metrics[method]
        t80 = "/" if item["t80_episode"] is None else str(item["t80_episode"])
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {pct(item['success_rate_mean'])} ± "
            f"{pct(item['success_rate_sample_sd'])} | {pct(item['nauc'])} | {t80} | "
            f"{pct(item['force_related_failure_rate'])} | "
            f"{item['under_force_failures']} / {item['over_force_failures']} |"
        )
    lines.extend(
        [
            "",
            "## Force-failure 操作定义",
            "",
            "LabPick 的原始终止标签不同于 Bottle：`object_broken` 映射为 `over_force`，"
            "`object_dropped` 映射为 `under_force`；`timeout` 和 `object_too_far` 保留为 "
            "`other_failure`。该规则用于可复现统计，但 `object_dropped` 是失去抓持的操作性代理，"
            "不能单独证明根因一定是夹持力不足。",
            "",
            "## 数据来源与边界",
            "",
        ]
    )
    source_by_method = {item["method"]: item for item in source_selection["methods"]}
    for method in ALL_METHODS:
        item = source_by_method[method]
        lines.append(
            f"- **{DISPLAY_NAMES[method]}**：`{item['source_dataset']}/{item['source_method']}`；"
            f"{item['selection_reason']}"
        )
    lines.extend(
        [
            "",
            "所有方法共享 4.5 N 破损阈值、±0.10 m XY 随机范围、0° yaw 和相同的三个 "
            "run seed。旧日志没有为每个 episode 保存显式场景位姿，部分 runner 只在第一轮写入 "
            "`reset_seed`，因此可以验证 run-level seed/config pairing，不能从现有产物独立证明每个 "
            "episode 的场景随机量逐项完全相同。",
            "",
            "RAPID-Tactile 的 `advisor=deterministic` 且 `advisor_is_real_vlm=false`。这些结果支持"
            "确定性顾问协议下的完整方法，不应表述为真实视觉语言模型在线推理的直接证据。",
            "",
            "## 产物",
            "",
            "- `paper_table_values.csv`：可直接填入论文表格的数值。",
            "- `online_episode_outcomes.csv`：1800 个 episode 的清洗记录与 force-failure 分类。",
            "- `online_success_window10_source.csv`：所有曲线坐标、三个 seed 和 95% t 区间。",
            "- `per_run_summary.csv`：18 个正式 run 的统计和关键输入哈希。",
            "- `comparison_manifest.json`：协议、方法来源、统计定义、完整性审计与全部汇总。",
            "- `raw_file_inventory.csv`：原始目录中每个文件的大小和 SHA-256。",
            "- `figures/`：五方法总图及各方法单图的 PDF、SVG、PNG、TIFF。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def raw_inventory(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in raw_root.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": str(path.relative_to(raw_root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.window != WINDOW:
        raise ValueError("the manuscript protocol requires a trailing window of 10")
    source_selection = read_json(raw_root / "source_selection.json")
    groups = load_all_runs(raw_root)
    metrics = {
        method: group_metrics(method, groups[method], args.window)
        for method in ALL_METHODS
    }

    outcomes = episode_rows(groups)
    curves = build_curve_rows(groups, args.window)
    table = table_rows(metrics)
    per_runs = [
        {
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            **run,
            "failure_counts": json.dumps(run["failure_counts"], sort_keys=True),
            "input_hashes": json.dumps(run["input_hashes"], sort_keys=True),
        }
        for method in ALL_METHODS
        for run in metrics[method]["per_run"]
    ]
    write_csv(output_dir / "online_episode_outcomes.csv", outcomes)
    write_csv(output_dir / "online_success_window10_source.csv", curves)
    write_csv(output_dir / "paper_table_values.csv", table)
    write_csv(output_dir / "per_run_summary.csv", per_runs)

    configure_plotting()
    for method in ADAPTIVE_METHODS:
        draw_single(method, curves, output_dir, metrics["base"])
    draw_combined(curves, output_dir, metrics["base"])

    inventory = raw_inventory(raw_root)
    write_csv(output_dir / "raw_file_inventory.csv", inventory)
    manifest = {
        "schema_version": 1,
        "experiment": "LabPick / Thin Sheet Transfer six-method online comparison",
        "methods": list(ALL_METHODS),
        "adaptive_methods": list(ADAPTIVE_METHODS),
        "training_seeds": list(SEEDS),
        "runs_per_method": 3,
        "episodes_per_run": EPISODES,
        "total_episodes": len(ALL_METHODS) * len(SEEDS) * EPISODES,
        "rolling_window": args.window,
        "nauc_definition": "mean of the across-run mean trailing-10 curve over episodes 10--100",
        "t80_definition": "first actual episode where the across-run mean trailing-10 curve is >= 0.8",
        "base_nauc_definition": "constant frozen-base mean success-rate reference",
        "force_related_failure_definition": {
            "under_force": "object_dropped",
            "over_force": "object_broken",
            "other_failure": ["timeout", "object_too_far"],
            "interpretation_limit": (
                "object_dropped is an operational lost-grasp proxy and does not by itself "
                "prove a causal under-force mechanism"
            ),
        },
        "uncertainty": "sample SD for table SR; pointwise 95% Student's t CI for curves (df=2)",
        "pairing_audit": {
            "verified": "same three run seeds and shared task randomization configuration",
            "not_independently_verifiable": (
                "exact per-episode scene draws because explicit reset seeds/poses were not "
                "persisted for every episode by all runners"
            ),
        },
        "rapid_tactile_advisor": {
            "advisor": "deterministic",
            "advisor_is_real_vlm": False,
        },
        "source_selection": source_selection,
        "metrics": metrics,
        "raw_file_count": len(inventory),
        "raw_total_bytes": sum(int(row["size_bytes"]) for row in inventory),
        "outputs": {
            "paper_table_values": "paper_table_values.csv",
            "episode_outcomes": "online_episode_outcomes.csv",
            "curve_source": "online_success_window10_source.csv",
            "per_run_summary": "per_run_summary.csv",
            "raw_file_inventory": "raw_file_inventory.csv",
            "report": "LABPICK_ONLINE_REPORT.md",
            "figure_formats": ["pdf", "svg", "png", "tiff"],
        },
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "LABPICK_ONLINE_REPORT.md", metrics, source_selection)
    print(json.dumps({"table": table, "raw_files": len(inventory)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
