#!/usr/bin/env python3
"""Audit recorded episodes and explain joint-controller failures at 3.5 N."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METRICS = (
    "peak_contact_force_n",
    "mean_contact_force_n",
    "max_lift_m",
    "contact_fraction",
    "bilateral_contact_fraction",
    "force_rmse_n",
    "controller_active_fraction",
    "min_grasp_distance_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path(
            "exp_report/vlm_with_dsrl/runs/"
            "threshold_3p5_joint_seed4200_n50/results.json"
        ),
    )
    parser.add_argument(
        "--video-results",
        type=Path,
        default=Path(
            "exp_report/vlm_with_dsrl/video_diagnostics/"
            "threshold_3p5_joint_seed4200_n10/results.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp_report/vlm_with_dsrl/video_diagnostics"),
    )
    return parser.parse_args()


def load_joint_run(path: Path, *, expected_episodes: int) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("mode") != "joint":
        raise ValueError(f"Expected a joint run: {path}")
    if not math.isclose(float(payload["break_force_threshold_n"]), 3.5):
        raise ValueError(f"Expected the 3.5 N condition: {path}")
    if int(payload.get("completed_episodes", -1)) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes: {path}")
    if len(payload.get("results", [])) != expected_episodes:
        raise ValueError(f"Incomplete episode array: {path}")
    return payload


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return center - radius, center + radius


def probe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    return streams[0]


def group_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        key = "success" if episode["success"] else episode["diagnosed_failure_reason"]
        grouped[str(key)].append(episode)
    output: dict[str, Any] = {}
    for key, items in sorted(grouped.items()):
        metric_summary = {}
        for metric in METRICS:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            metric_summary[metric] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        output[key] = {"episodes": len(items), "metrics": metric_summary}
    return output


def force_range_strata(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: dict[str, list[dict[str, Any]]] = {
        "center >= 1.0 N": [],
        "0.8 <= center < 1.0 N": [],
        "center < 0.8 N": [],
    }
    for episode in episodes:
        bounds = episode["attempted_force_range_n"]
        center = 0.5 * (float(bounds[0]) + float(bounds[1]))
        if center >= 1.0:
            key = "center >= 1.0 N"
        elif center >= 0.8:
            key = "0.8 <= center < 1.0 N"
        else:
            key = "center < 0.8 N"
        bins[key].append(episode)
    return [
        {
            "stratum": key,
            "episodes": len(items),
            "successes": sum(int(item["success"]) for item in items),
            "success_rate": sum(int(item["success"]) for item in items) / len(items),
            "broken": sum(
                int(item["terminal_reason"] == "object_broken") for item in items
            ),
        }
        for key, items in bins.items()
    ]


def main() -> None:
    args = parse_args()
    reference_path = args.reference_results.expanduser().resolve()
    video_path = args.video_results.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_joint_run(reference_path, expected_episodes=50)
    diagnostic = load_joint_run(video_path, expected_episodes=10)
    reference_episodes = reference["results"]
    diagnostic_episodes = diagnostic["results"]
    advisor_log_path = reference_path.parent / "vlm_episode_interactions.jsonl"
    advisor_records = [
        json.loads(line)
        for line in advisor_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(advisor_records) != 50 or any(
        item.get("image_paths") for item in advisor_records
    ):
        raise ValueError(
            "Expected 50 deterministic advisor calls without image attachments."
        )

    if reference["bc_checkpoint_sha256"] != diagnostic["bc_checkpoint_sha256"]:
        raise ValueError("Reference and video runs use different BC checkpoints.")
    for field in ("seed", "labware_random_xy_m", "labware_random_yaw_deg"):
        if reference[field] != diagnostic[field]:
            raise ValueError(f"Reference and video runs differ in {field}.")

    manifest_fields = [
        "episode_index",
        "success",
        "terminal_reason",
        "diagnosed_failure_reason",
        "physics_steps",
        "video_frames",
        "duration_s",
        "resolution",
        "peak_contact_force_n",
        "mean_contact_force_n",
        "max_lift_m",
        "contact_fraction",
        "bilateral_contact_fraction",
        "force_rmse_n",
        "attempted_force_min_n",
        "attempted_force_max_n",
        "video",
        "trajectory",
    ]
    manifest_rows = []
    for episode in diagnostic_episodes:
        episode_video = Path(str(episode.get("video", ""))).expanduser().resolve()
        trajectory = Path(str(episode.get("trajectory", ""))).expanduser().resolve()
        if not episode_video.is_file() or not trajectory.is_file():
            raise FileNotFoundError(episode_video if not episode_video.is_file() else trajectory)
        probe = probe_video(episode_video)
        if probe["codec_name"] != "h264":
            raise ValueError(f"Unexpected video codec: {episode_video}")
        if int(probe["nb_frames"]) != int(episode["video_frame_count"]):
            raise ValueError(f"Video frame-count mismatch: {episode_video}")
        attempted = episode["attempted_force_range_n"]
        manifest_rows.append(
            {
                "episode_index": episode["episode_index"],
                "success": int(episode["success"]),
                "terminal_reason": episode["terminal_reason"],
                "diagnosed_failure_reason": episode["diagnosed_failure_reason"],
                "physics_steps": episode["physics_steps"],
                "video_frames": episode["video_frame_count"],
                "duration_s": round(float(probe["duration"]), 3),
                "resolution": f"{probe['width']}x{probe['height']}",
                "peak_contact_force_n": episode["peak_contact_force_n"],
                "mean_contact_force_n": episode["mean_contact_force_n"],
                "max_lift_m": episode["max_lift_m"],
                "contact_fraction": episode["contact_fraction"],
                "bilateral_contact_fraction": episode["bilateral_contact_fraction"],
                "force_rmse_n": episode["force_rmse_n"],
                "attempted_force_min_n": attempted[0],
                "attempted_force_max_n": attempted[1],
                "video": os.path.relpath(episode_video, output_dir),
                "trajectory": os.path.relpath(trajectory, output_dir),
            }
        )

    manifest_path = output_dir / "video_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    successes = sum(int(item["success"]) for item in reference_episodes)
    success_ci = wilson(successes, len(reference_episodes))
    raw_failures = Counter(
        item["terminal_reason"] for item in reference_episodes if not item["success"]
    )
    diagnosed_failures = Counter(
        item["diagnosed_failure_reason"]
        for item in reference_episodes
        if not item["success"]
    )
    metrics_by_group = group_metrics(reference_episodes)
    strata = force_range_strata(reference_episodes)
    binary_agreement = sum(
        int(bool(old["success"]) == bool(new["success"]))
        for old, new in zip(reference_episodes[:10], diagnostic_episodes)
    )
    terminal_agreement = sum(
        int(old["terminal_reason"] == new["terminal_reason"])
        for old, new in zip(reference_episodes[:10], diagnostic_episodes)
    )

    analysis = {
        "schema_version": 1,
        "reference_results": str(reference_path),
        "diagnostic_video_results": str(video_path),
        "threshold_n": 3.5,
        "reference": {
            "episodes": len(reference_episodes),
            "successes": successes,
            "success_rate": successes / len(reference_episodes),
            "success_wilson_95": list(success_ci),
            "raw_failure_counts": dict(raw_failures),
            "diagnosed_failure_counts": dict(diagnosed_failures),
            "dsrl_gradient_updates": reference["dsrl_gradient_updates"],
            "advisor_calls": reference["advisor_calls"],
            "advisor_is_real_vlm": reference["advisor_is_real_vlm"],
            "advisor_calls_with_images": sum(
                int(bool(item.get("image_paths"))) for item in advisor_records
            ),
            "initial_force_range_n": reference["initial_force_range_n"],
            "final_force_range_n": reference["final_force_range_n"],
            "metrics_by_group": metrics_by_group,
            "force_range_strata": strata,
        },
        "diagnostic_video_run": {
            "episodes": len(diagnostic_episodes),
            "successes": sum(int(item["success"]) for item in diagnostic_episodes),
            "broken": sum(
                int(item["terminal_reason"] == "object_broken")
                for item in diagnostic_episodes
            ),
            "video_count": len(manifest_rows),
            "binary_outcome_agreement_with_reference_first_10": binary_agreement,
            "terminal_reason_agreement_with_reference_first_10": terminal_agreement,
        },
    }
    (output_dir / "failure_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    group_order = (
        "force_tracking_error",
        "trajectory_error",
        "bad_alignment",
        "object_broken",
        "no_contact",
    )
    failure_total = len(reference_episodes) - successes
    failure_rows = []
    for reason in group_order:
        group = metrics_by_group[reason]
        metrics = group["metrics"]
        failure_rows.append(
            f"| `{reason}` | {group['episodes']} | "
            f"{100 * group['episodes'] / failure_total:.1f}% | "
            f"{100 * metrics['contact_fraction']['mean']:.1f}% | "
            f"{100 * metrics['bilateral_contact_fraction']['mean']:.1f}% | "
            f"{metrics['force_rmse_n']['mean']:.2f} | "
            f"{100 * metrics['max_lift_m']['mean']:.1f} | "
            f"{metrics['peak_contact_force_n']['mean']:.2f} |"
        )

    video_rows = []
    for row in manifest_rows:
        label = "成功" if row["success"] else "失败"
        video_rows.append(
            f"| {row['episode_index']} | {label} | `{row['diagnosed_failure_reason']}` | "
            f"{float(row['peak_contact_force_n']):.2f} | "
            f"{100 * float(row['max_lift_m']):.1f} | "
            f"{int(row['video_frames'])} | [MP4]({row['video']}) |"
        )

    success_metrics = metrics_by_group["success"]["metrics"]
    stratum_text = "；".join(
        f"{item['stratum']} 为 {item['successes']}/{item['episodes']} "
        f"({100 * item['success_rate']:.1f}%)"
        for item in strata
    )
    markdown = f"""# 3.5 N 下 VLM + DSRL 轨迹视频与失败分析

## 结论

原始 50-episode 联合实验成功 **{successes}/50（{100 * successes / 50:.1f}%）**，95% Wilson CI 为 {100 * success_ci[0]:.1f}–{100 * success_ci[1]:.1f}%。联合方法已经相对 frozen base（2% 成功、80% 破坏）明显改善，但 36 次失败中，30 次是 timeout；成功率上限主要受接触建立和轨迹稳定性约束，而不是单一的目标力范围选择。

最直接的证据是：成功 episode 的平均接触占比为 {100 * success_metrics['contact_fraction']['mean']:.1f}%，双指接触占比为 {100 * success_metrics['bilateral_contact_fraction']['mean']:.1f}%，力控制器激活占比为 {100 * success_metrics['controller_active_fraction']['mean']:.1f}%；对应中位最大抬升为 {100 * success_metrics['max_lift_m']['median']:.1f} cm。失败组通常没有形成持续、双指、可承载的接触，因此“选对力”并不足以让任务成功。

## 10 条诊断视频

这些视频是同 checkpoint、seed、随机化、3.5 N 阈值和控制超参数下的**诊断复现实验**，不是原始 50 条无图像轨迹的逐像素回放。录制会启用渲染路径，Isaac 接触动力学与在线 SAC 并非严格确定：成功/失败二分类与原前 10 条一致 {binary_agreement}/10，精确终止原因一致 {terminal_agreement}/10。因此正式成功率与失败计数使用原始 50 条数据，视频只用于展示机制。

| Episode | 结果 | 诊断 | 峰值力 (N) | 最大抬升 (cm) | 帧数 | 视频 |
|---:|---|---|---:|---:|---:|---|
{chr(10).join(video_rows)}

完整机器可读索引见 [video_manifest.csv](video_manifest.csv)，录制运行见 [results.json](threshold_3p5_joint_seed4200_n10/results.json)。10 个 MP4 均为 H.264、640×480、30 fps，每 4 个物理步采样一帧，且帧数已与 JSON 逐条核对。

## 原始 50 条轨迹的失败构成

| 诊断原因 | 次数 | 占失败 | 平均接触占比 | 平均双指接触占比 | 平均力 RMSE (N) | 平均最大抬升 (cm) | 平均峰值力 (N) |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(failure_rows)}

原始终止事件为：`timeout` {raw_failures['timeout']} 次、`object_broken` {raw_failures['object_broken']} 次、`object_dropped` {raw_failures['object_dropped']} 次。诊断层把 timeout/drop 进一步拆成力跟踪、轨迹、对齐和未接触问题，避免把“实测力没达到目标”错误解释为“目标力太低”。

## 为什么最终只有 28%

1. **大多数失败发生在力控能稳定工作的前提之外。** 14 次 `force_tracking_error` 的平均接触占比仅 3.5%，12 次 `trajectory_error` 也只有 9.5%；成功组为 40.7%。力控只有在接触后才覆盖夹爪维度，无法修复自由空间阶段的视觉定位偏差，也无法靠夹爪动作挽救不稳定的单边接触。
2. **双指接触持续性是最清楚的分界。** 成功组平均双指接触占比 45.5%，轨迹失败为 19.6%，力跟踪失败为 15.6%，bad-alignment 为 0%。DSRL 虽控制其余九个动作维度，但 50 个在线 episode、653 次梯度更新仍未把接近与夹持轨迹稳定到足够高的覆盖率。
3. **目标力范围持续收缩，但没有带来单调成功率提升。** 范围从 1–3 N 收缩到 {reference['final_force_range_n'][0]:.3f}–{reference['final_force_range_n'][1]:.3f} N。按实际区间中心分层，{stratum_text}。这是顺序数据而非随机对照，不能作因果估计，但至少说明继续降低目标力不是当前主要解法。
4. **接触瞬态仍会越过 3.5 N。** 4 次破坏的平均峰值为 {metrics_by_group['object_broken']['metrics']['peak_contact_force_n']['mean']:.2f} N（范围 {metrics_by_group['object_broken']['metrics']['peak_contact_force_n']['minimum']:.2f}–{metrics_by_group['object_broken']['metrics']['peak_contact_force_n']['maximum']:.2f} N）。控制器是接触门控的，初次碰撞发生后才有反馈，因而不能完全消除第一帧冲击峰值。
5. **两个在线适应环同时变化，信用分配是非平稳的。** DSRL 每个 outer interaction 更新位置修正；力顾问又在每个 episode 后改变目标范围。顾问提示词中的“保持位姿策略不变”在联合实验中并不成立，因此一次失败究竟源于位置策略还是力范围会发生混淆。
6. **这里的“VLM”是确定性离线协议替代器。** 原运行 50 次 advisor call 的 `image_paths` 为空，决策来自结构化力/失败指标和规则，不是实际多模态图像理解。因此 28% 只能证明联合闭环软件与力范围适应的效果，不能代表真实 VLM 的视觉推理上限。

## 对下一轮实验最有价值的改动

- 先冻结力范围，在同一范围下把 DSRL 训练到接触/双指接触覆盖率稳定，再冻结 DSRL 做力范围适应，减少双重非平稳性。
- 把首次接触前的相对 XY/姿态误差、首次双指接触时刻和冲击峰值纳入 DSRL reward/observation；当前 episode 级失败标签太晚且稀疏。
- 对 3.5 N 增加接触前减速或预测性软着陆；仅靠接触后的安全 override 无法拦截第一帧峰值。
- 正式论文实验应使用冻结的真实 VLM，并至少增加多个独立 seed；当前 14/50 的区间较宽，不能把 28% 当成精确的总体性能。
"""
    (output_dir / "analysis_3p5_joint.md").write_text(markdown, encoding="utf-8")
    print(f"[ANALYSIS] {output_dir / 'analysis_3p5_joint.md'}")
    print(f"[VIDEOS] {len(manifest_rows)} verified in {manifest_path}")


if __name__ == "__main__":
    main()
