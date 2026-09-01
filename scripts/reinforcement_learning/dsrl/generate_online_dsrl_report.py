#!/usr/bin/env python3
"""Validate online DSRL JSONL and generate success/failure trend artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAILURE_REASONS = (
    "object_broken",
    "object_dropped",
    "object_too_far",
    "ee_outside_workspace",
    "timeout",
    "unknown_terminal",
)
FAILURE_LABELS = {
    "object_broken": "Broken",
    "object_dropped": "Dropped",
    "object_too_far": "Too far",
    "ee_outside_workspace": "EE outside",
    "timeout": "Timeout",
    "unknown_terminal": "Unknown",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_dir", type=Path, required=True)
    parser.add_argument("--expected_interactions", type=int, default=0)
    parser.add_argument("--rolling_episodes", type=int, default=10)
    parser.add_argument("--experiment_id", default="DSRL-B-260827-001")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - margin, center + margin


def validate_records(
    interactions: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    *,
    expected_interactions: int = 0,
) -> None:
    if not interactions:
        raise ValueError("online_interactions.jsonl is empty.")
    indices = [int(row["interaction_index"]) for row in interactions]
    if indices != list(range(1, len(interactions) + 1)):
        raise ValueError("Interaction indices must be contiguous and one-based.")
    if expected_interactions and len(interactions) != expected_interactions:
        raise ValueError(
            f"Expected {expected_interactions} interactions, found {len(interactions)}."
        )
    terminal_rows = [row for row in interactions if bool(row["terminal"])]
    if len(terminal_rows) != len(episodes):
        raise ValueError(
            "Terminal interaction count does not match completed episode count: "
            f"{len(terminal_rows)} != {len(episodes)}."
        )
    for expected_episode, (interaction, episode) in enumerate(
        zip(terminal_rows, episodes), start=1
    ):
        if int(episode["episode_index"]) != expected_episode:
            raise ValueError("Episode indices must be contiguous and one-based.")
        if int(interaction["interaction_index"]) != int(
            episode["ending_interaction_index"]
        ):
            raise ValueError("Episode ending interaction does not match interaction JSONL.")
        if interaction["success"] is None or episode["success"] is None:
            raise ValueError("Completed episodes must have a boolean success outcome.")
        if bool(episode["success"]) == bool(episode.get("failure_reason")):
            raise ValueError("Each episode must have exactly one success/failure outcome.")


def _make_bins(total_interactions: int, count: int = 3) -> list[tuple[int, int]]:
    edges = np.linspace(0, total_interactions, count + 1, dtype=int)
    return [(int(edges[index]) + 1, int(edges[index + 1])) for index in range(count)]


def _summarize_bin(
    episodes: list[dict[str, Any]], start: int, end: int
) -> dict[str, Any]:
    selected = [
        row
        for row in episodes
        if start <= int(row["ending_interaction_index"]) <= end
    ]
    successes = sum(bool(row["success"]) for row in selected)
    total = len(selected)
    failure_counts = Counter(
        row["failure_reason"] for row in selected if not bool(row["success"])
    )
    lower, upper = _wilson_interval(successes, total)
    peak_forces = [float(row["peak_contact_force_n"]) for row in selected]
    return {
        "interaction_start": start,
        "interaction_end": end,
        "label": f"{start}-{end}",
        "completed_episodes": total,
        "successes": successes,
        "failures": total - successes,
        "success_rate": None if total == 0 else successes / total,
        "success_rate_wilson_95": (
            None if total == 0 else [lower, upper]
        ),
        "broken_episode_rate": (
            None if total == 0 else failure_counts.get("object_broken", 0) / total
        ),
        "mean_peak_contact_force_n": (
            None if total == 0 else float(np.mean(peak_forces))
        ),
        "max_peak_contact_force_n": (
            None if total == 0 else float(np.max(peak_forces))
        ),
        "failure_counts": {
            reason: int(failure_counts.get(reason, 0)) for reason in FAILURE_REASONS
        },
    }


def build_summary(
    interactions: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    *,
    rolling_episodes: int = 10,
) -> dict[str, Any]:
    if rolling_episodes < 1:
        raise ValueError("rolling_episodes must be positive.")
    total_interactions = len(interactions)
    bins = [
        _summarize_bin(episodes, start, end)
        for start, end in _make_bins(total_interactions)
    ]
    nonempty_bins = [item for item in bins if item["completed_episodes"]]
    first = nonempty_bins[0]
    last = nonempty_bins[-1]
    change = 100.0 * (last["success_rate"] - first["success_rate"])
    if change > 1.0e-12:
        direction = "increased"
    elif change < -1.0e-12:
        direction = "decreased"
    else:
        direction = "unchanged"

    successes = sum(bool(row["success"]) for row in episodes)
    failure_counts = Counter(
        row["failure_reason"] for row in episodes if not bool(row["success"])
    )
    ending_indices = np.asarray(
        [int(row["ending_interaction_index"]) for row in episodes], dtype=float
    )
    outcomes = np.asarray([bool(row["success"]) for row in episodes], dtype=float)
    slope = (
        float(np.polyfit(ending_indices, outcomes, 1)[0] * 100.0 * 100.0)
        if len(episodes) >= 2 and np.ptp(ending_indices) > 0
        else 0.0
    )
    incomplete_interactions = (
        0
        if bool(interactions[-1]["terminal"])
        else sum(
            int(row["episode_index"]) == int(interactions[-1]["episode_index"])
            for row in interactions
        )
    )
    comparison_window = min(rolling_episodes, len(episodes))
    first_window_success_rate = (
        sum(bool(row["success"]) for row in episodes[:comparison_window])
        / comparison_window
    )
    last_window_success_rate = (
        sum(bool(row["success"]) for row in episodes[-comparison_window:])
        / comparison_window
    )
    return {
        "schema_version": 1,
        "interaction_count": total_interactions,
        "completed_episode_count": len(episodes),
        "final_incomplete_episode_interactions": incomplete_interactions,
        "successes": successes,
        "failures": len(episodes) - successes,
        "overall_success_rate": successes / len(episodes),
        "failure_counts": {
            reason: int(failure_counts.get(reason, 0)) for reason in FAILURE_REASONS
        },
        "rolling_window_episodes": rolling_episodes,
        "comparison_window_episodes": comparison_window,
        "first_window_success_rate": first_window_success_rate,
        "last_window_success_rate": last_window_success_rate,
        "first_to_last_window_success_rate_change_percentage_points": 100.0
        * (last_window_success_rate - first_window_success_rate),
        "success_rate_bins": bins,
        "first_nonempty_bin": first["label"],
        "last_nonempty_bin": last["label"],
        "first_to_last_success_rate_change_percentage_points": change,
        "observed_success_rate_direction": direction,
        "linear_success_trend_percentage_points_per_100_interactions": slope,
        "episode_peak_contact_force_n": {
            "mean": float(np.mean([row["peak_contact_force_n"] for row in episodes])),
            "max": float(np.max([row["peak_contact_force_n"] for row in episodes])),
        },
    }


def _write_success_csv(
    path: Path, episodes: list[dict[str, Any]], rolling_episodes: int
) -> None:
    successes = []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "episode_index",
                "ending_interaction_index",
                "success",
                "failure_reason",
                "cumulative_success_rate",
                "rolling_success_rate",
            ),
        )
        writer.writeheader()
        for index, row in enumerate(episodes):
            successes.append(int(bool(row["success"])))
            window = successes[max(0, len(successes) - rolling_episodes) :]
            writer.writerow(
                {
                    "episode_index": row["episode_index"],
                    "ending_interaction_index": row["ending_interaction_index"],
                    "success": successes[-1],
                    "failure_reason": row.get("failure_reason") or "",
                    "cumulative_success_rate": sum(successes) / len(successes),
                    "rolling_success_rate": sum(window) / len(window),
                }
            )


def _write_failure_csv(path: Path, bins: list[dict[str, Any]]) -> None:
    fields = [
        "interaction_range",
        "completed_episodes",
        "successes",
        "failures",
        "success_rate",
        "broken_episode_rate",
        "mean_peak_contact_force_n",
        "max_peak_contact_force_n",
        *FAILURE_REASONS,
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in bins:
            writer.writerow(
                {
                    "interaction_range": item["label"],
                    "completed_episodes": item["completed_episodes"],
                    "successes": item["successes"],
                    "failures": item["failures"],
                    "success_rate": item["success_rate"],
                    "broken_episode_rate": item["broken_episode_rate"],
                    "mean_peak_contact_force_n": item["mean_peak_contact_force_n"],
                    "max_peak_contact_force_n": item["max_peak_contact_force_n"],
                    **item["failure_counts"],
                }
            )


def _plot_success_curve(
    path: Path, episodes: list[dict[str, Any]], rolling_episodes: int
) -> None:
    x = np.asarray([row["ending_interaction_index"] for row in episodes])
    outcomes = np.asarray([bool(row["success"]) for row in episodes], dtype=float)
    cumulative = np.cumsum(outcomes) / np.arange(1, len(outcomes) + 1)
    rolling = np.asarray(
        [
            outcomes[max(0, index - rolling_episodes + 1) : index + 1].mean()
            for index in range(len(outcomes))
        ]
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(x, 100.0 * cumulative, color="#1769aa", linewidth=2.2, label="Cumulative")
    ax.plot(
        x,
        100.0 * rolling,
        color="#d95f02",
        linewidth=2.0,
        label=f"Rolling ({rolling_episodes} episodes)",
    )
    success_mask = outcomes > 0.5
    ax.scatter(
        x[success_mask],
        np.full(success_mask.sum(), 95.0),
        color="#2ca25f",
        s=18,
        alpha=0.55,
        label="Successful episode",
    )
    ax.scatter(
        x[~success_mask],
        np.full((~success_mask).sum(), 5.0),
        color="#de2d26",
        s=18,
        alpha=0.55,
        label="Failed episode",
    )
    ax.set(xlabel="Online interaction", ylabel="Success rate (%)", ylim=(-2, 102))
    ax.set_title("Clean DSRL online success rate")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_failure_trends(path: Path, bins: list[dict[str, Any]]) -> None:
    labels = [item["label"] for item in bins]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = ("#d73027", "#fc8d59", "#fee08b", "#91bfdb", "#4575b4", "#777777")
    for reason, color in zip(FAILURE_REASONS, colors):
        values = np.asarray([item["failure_counts"][reason] for item in bins])
        ax.bar(x, values, bottom=bottom, label=FAILURE_LABELS[reason], color=color)
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_xlabel("Online interaction range (episode assigned by terminal interaction)")
    ax.set_ylabel("Completed failed episodes")
    ax.set_title("Failure reasons during online DSRL")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _format_rate(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def _write_report(
    path: Path,
    summary: dict[str, Any],
    metadata: dict[str, Any],
    *,
    experiment_id: str,
) -> None:
    first = next(
        item
        for item in summary["success_rate_bins"]
        if item["label"] == summary["first_nonempty_bin"]
    )
    last = next(
        item
        for item in summary["success_rate_bins"]
        if item["label"] == summary["last_nonempty_bin"]
    )
    direction_cn = {
        "increased": "上升",
        "decreased": "下降",
        "unchanged": "持平",
    }[summary["observed_success_rate_direction"]]
    failure_lines = []
    for reason in FAILURE_REASONS:
        first_count = first["failure_counts"][reason]
        last_count = last["failure_counts"][reason]
        failure_lines.append(f"- `{reason}`：{first_count} → {last_count}")
    table_rows = "\n".join(
        "| {label} | {successes}/{completed_episodes} | {rate} | {failures} | {broken} | {broken_rate} | {dropped} | {far} | {outside} | {timeout} | {mean_force:.2f} |".format(
            label=item["label"],
            successes=item["successes"],
            completed_episodes=item["completed_episodes"],
            rate=_format_rate(item["success_rate"]),
            failures=item["failures"],
            broken=item["failure_counts"]["object_broken"],
            broken_rate=_format_rate(item["broken_episode_rate"]),
            dropped=item["failure_counts"]["object_dropped"],
            far=item["failure_counts"]["object_too_far"],
            outside=item["failure_counts"]["ee_outside_workspace"],
            timeout=item["failure_counts"]["timeout"],
            mean_force=item["mean_peak_contact_force_n"],
        )
        for item in summary["success_rate_bins"]
    )
    report = f"""---
experiment_id: {experiment_id}
system: DINOv3-Flow-BC-Clean-DSRL
experiment_type: online_reinforcement_learning
date: {datetime.now().astimezone().date().isoformat()}
status: completed
seed: {metadata.get('seed', 'unknown')}
break_force_threshold_n: {metadata.get('break_force_threshold_n', 'unknown')}
base_policy: {metadata.get('bc_policy', 'unknown')}
---

# DINOv3 Flow Matching BC + Clean DSRL 在线实验

## 结论

在 {summary['interaction_count']} 次 online outer interactions 中，共完成
{summary['completed_episode_count']} 个 episode，成功 {summary['successes']} 次，整体成功率
为 {summary['overall_success_rate']:.1%}。按交互区间比较，成功率从
`{first['label']}` 的 {_format_rate(first['success_rate'])} 变为
`{last['label']}` 的 {_format_rate(last['success_rate'])}，变化
{summary['first_to_last_success_rate_change_percentage_points']:+.1f} 个百分点，观察方向为
**{direction_cn}**。线性趋势为每 100 次交互
{summary['linear_success_trend_percentage_points_per_100_interactions']:+.1f} 个百分点。
以固定的前/后 {summary['comparison_window_episodes']} 个完整 episode 比较，rolling 成功率为
{summary['first_window_success_rate']:.1%} → {summary['last_window_success_rate']:.1%}。

这是同一次在线训练轨迹的描述性结果；各区间完成的 episode 数有限，不能单凭该曲线作统计显著性结论。

## 成功率与失败类型

| 交互区间 | 成功 | 成功率 | 失败 | 破碎 | 破碎率 | 掉落 | 过远 | 越界 | 超时 | 平均峰值力/N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table_rows}

首段到末段的失败数变化：

{chr(10).join(failure_lines)}

## 实验协议

- Base policy：`{metadata.get('bc_policy', 'unknown')}`
- Base policy SHA-256：`{metadata.get('bc_checkpoint_sha256', 'unknown')}`
- DSRL：absolute Flow-noise SAC，learned noise steps = {metadata.get('learned_noise_steps', 'unknown')}
- 训练：{metadata.get('timesteps', summary['interaction_count'])} 次 outer interactions，seed = {metadata.get('seed', 'unknown')}，learning starts = {metadata.get('learning_starts', 'unknown')}，batch size = {metadata.get('batch_size', 'unknown')}
- 每次 chunk 执行 {metadata.get('chunk_execute_steps', 'unknown')} 个动作，每个动作重复 {metadata.get('action_repeat', 'unknown')} 个 physics steps
- 场景 XY 随机化：`{metadata.get('labware_random_xy_m', 'unknown')}` m；yaw：{metadata.get('labware_random_yaw_deg', 'unknown')}°
- 破碎阈值：{metadata.get('break_force_threshold_n', 'unknown')} N
- visual-XY override：{metadata.get('use_visual_xy_override', 'unknown')}
- 训练日志：`{metadata.get('training_log_dir', 'unknown')}`

## 文件

- `online_interactions.jsonl`：每次 outer interaction 的状态、回报、力峰值与终止标志
- `online_episodes.jsonl`：每个已完成 episode 的成功/失败及失败原因
- `online_success_rate.csv` / `online_success_rate_curve.png`：累计与滚动成功率
- `failure_reason_bins.csv` / `failure_reason_trends.png`：分阶段失败构成
- `summary.json`：经校验的汇总统计
- `checkpoint_manifest.json`：step-100/200/300 与 best checkpoint 的路径、大小和 SHA-256

训练结束时最后一个 episode 若尚未终止，不强行归类为成功或失败；本次该未完成 episode 含
{summary['final_incomplete_episode_interactions']} 次 interaction。
"""
    path.write_text(report, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_checkpoint_manifest(
    path: Path, metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    training_log_dir = metadata.get("training_log_dir")
    checkpoint_dir = (
        None if not training_log_dir else Path(training_log_dir) / "checkpoints"
    )
    checkpoints = [] if checkpoint_dir is None else sorted(checkpoint_dir.glob("*.pt"))
    records = [
        {
            "name": checkpoint.name,
            "path": str(checkpoint.resolve()),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
        }
        for checkpoint in checkpoints
    ]
    path.write_text(
        json.dumps({"checkpoints": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return records


def generate_artifacts(
    metrics_dir: Path,
    *,
    expected_interactions: int = 0,
    rolling_episodes: int = 10,
    experiment_id: str = "DSRL-B-260827-001",
) -> dict[str, Any]:
    metrics_dir = metrics_dir.expanduser().resolve()
    interactions = load_jsonl(metrics_dir / "online_interactions.jsonl")
    episodes = load_jsonl(metrics_dir / "online_episodes.jsonl")
    validate_records(
        interactions, episodes, expected_interactions=expected_interactions
    )
    if not episodes:
        raise ValueError("No completed episodes are available for a success-rate curve.")
    metadata_path = metrics_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    summary = build_summary(
        interactions, episodes, rolling_episodes=rolling_episodes
    )
    summary["metadata"] = metadata
    summary["checkpoint_count"] = len(
        _write_checkpoint_manifest(
            metrics_dir / "checkpoint_manifest.json", metadata
        )
    )
    (metrics_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_success_csv(
        metrics_dir / "online_success_rate.csv", episodes, rolling_episodes
    )
    _write_failure_csv(
        metrics_dir / "failure_reason_bins.csv", summary["success_rate_bins"]
    )
    _plot_success_curve(
        metrics_dir / "online_success_rate_curve.png", episodes, rolling_episodes
    )
    _plot_failure_trends(
        metrics_dir / "failure_reason_trends.png", summary["success_rate_bins"]
    )
    _write_report(
        metrics_dir / "report.md",
        summary,
        metadata,
        experiment_id=experiment_id,
    )
    return summary


def main() -> None:
    args = _parse_args()
    summary = generate_artifacts(
        args.metrics_dir,
        expected_interactions=args.expected_interactions,
        rolling_episodes=args.rolling_episodes,
        experiment_id=args.experiment_id,
    )
    print(
        f"[DONE] {summary['interaction_count']} interactions, "
        f"{summary['completed_episode_count']} completed episodes; "
        f"artifacts written to {args.metrics_dir.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
