#!/usr/bin/env python3
"""Audit and summarize repeated frozen-base SealTwist evaluations."""

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


FORCE_FAILURES = {"under_force", "over_force"}
PROTOCOL_KEYS = (
    "checkpoint_sha256",
    "episodes",
    "minimum_force_randomization_range_n",
    "maximum_force_randomization_range_n",
    "position_randomization_half_range_m",
    "alignment_tolerance_m",
    "torque_window_nm",
    "evaluation_horizon_steps",
    "success_hold_steps",
    "force_violation_dwell_steps",
    "chunk_execute_steps",
    "action_repeat",
    "integration_steps",
    "ode_solver",
    "action_smoothing",
    "force_feedback_target_n",
    "force_feedback_gain_m_per_n",
    "force_feedback_max_step_m",
    "force_feedback_start_step",
    "gripper_width_limits_m",
    "normalized_observation_clip",
    "noise_mode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("exp_report/bottle/frozen_base_repeats_n100"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=(170700, 171700, 172700))
    parser.add_argument("--episodes", type=int, default=100)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(path: Path, seed: int, episodes: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["seed_start"]) != seed:
        raise ValueError(f"{path}: seed_start mismatch")
    if int(payload["episodes"]) != episodes or len(payload["outcomes"]) != episodes:
        raise ValueError(f"{path}: expected {episodes} complete outcomes")
    expected_seeds = list(range(seed, seed + episodes))
    recorded_seeds = [int(row["seed"]) for row in payload["outcomes"]]
    if recorded_seeds != expected_seeds:
        raise ValueError(f"{path}: episode seeds are not contiguous")
    successes = sum(row["outcome"] == "success" for row in payload["outcomes"])
    if successes != int(payload["successes"]):
        raise ValueError(f"{path}: success count mismatch")
    if not math.isclose(float(payload["success_rate"]), successes / episodes):
        raise ValueError(f"{path}: success rate mismatch")
    for row in payload["outcomes"]:
        if "bilateral_contact_steps" not in row or "mean_bilateral_finger_force_n" not in row:
            raise ValueError(f"{path}: missing contact-phase force fields")
    return {"path": path.resolve(), "sha256": sha256(path), "payload": payload}


def ensure_matched_protocol(runs: list[dict[str, Any]], episodes: int) -> None:
    reference = runs[0]["payload"]
    for run in runs[1:]:
        payload = run["payload"]
        for key in PROTOCOL_KEYS:
            if payload.get(key) != reference.get(key):
                raise ValueError(f"protocol mismatch for {key}")
    seed_starts = sorted(int(run["payload"]["seed_start"]) for run in runs)
    if any(right - left < episodes for left, right in zip(seed_starts, seed_starts[1:])):
        raise ValueError("base runs use overlapping episode-seed ranges")


def run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    payload = run["payload"]
    outcomes = payload["outcomes"]
    counts = Counter(row["outcome"] for row in outcomes)
    failures = len(outcomes) - counts["success"]
    force_failures = sum(counts[name] for name in FORCE_FAILURES)
    contact_steps = sum(int(row["bilateral_contact_steps"]) for row in outcomes)
    force_sum = sum(
        float(row["mean_bilateral_finger_force_n"])
        * int(row["bilateral_contact_steps"])
        for row in outcomes
    )
    return {
        "seed_start": int(payload["seed_start"]),
        "episodes": len(outcomes),
        "successes": counts["success"],
        "success_rate": counts["success"] / len(outcomes),
        "contact_steps": contact_steps,
        "contact_phase_mean_force_n": force_sum / max(contact_steps, 1),
        "failures": failures,
        "under_force": counts["under_force"],
        "over_force": counts["over_force"],
        "alignment_failure": counts["alignment_failure"],
        "force_related_failures": force_failures,
        "force_related_fraction_of_failures": force_failures / max(failures, 1),
        "force_related_fraction_of_all_episodes": force_failures / len(outcomes),
        "input_json": str(run["path"]),
        "input_sha256": run["sha256"],
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("at least two unique seeds are required")
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        load_run(input_dir / f"base_seed_{seed}_n{args.episodes}.json", seed, args.episodes)
        for seed in args.seeds
    ]
    ensure_matched_protocol(runs, args.episodes)
    per_run = [run_metrics(run) for run in runs]
    success_mean, success_sd = mean_sd([row["success_rate"] for row in per_run])
    force_mean, force_sd = mean_sd(
        [row["contact_phase_mean_force_n"] for row in per_run]
    )
    failure_fraction_mean, failure_fraction_sd = mean_sd(
        [row["force_related_fraction_of_all_episodes"] for row in per_run]
    )
    pooled_failures = sum(row["failures"] for row in per_run)
    pooled_force_failures = sum(row["force_related_failures"] for row in per_run)
    pooled_contact_steps = sum(row["contact_steps"] for row in per_run)
    pooled_force = sum(
        row["contact_phase_mean_force_n"] * row["contact_steps"] for row in per_run
    ) / pooled_contact_steps
    summary = {
        "experiment": "SealTwist frozen base policy, three independent 100-episode evaluations",
        "replicate_unit": "non-overlapping episode-seed range",
        "seeds": list(args.seeds),
        "runs": len(per_run),
        "episodes_per_run": args.episodes,
        "total_episodes": len(per_run) * args.episodes,
        "success_rate_mean": success_mean,
        "success_rate_sample_sd": success_sd,
        "contact_phase_mean_force_n_mean": force_mean,
        "contact_phase_mean_force_n_sample_sd": force_sd,
        "pooled_contact_phase_mean_force_n": pooled_force,
        "force_related_fraction_of_all_episodes_mean": failure_fraction_mean,
        "force_related_fraction_of_all_episodes_sample_sd": failure_fraction_sd,
        "pooled_under_force": sum(row["under_force"] for row in per_run),
        "pooled_over_force": sum(row["over_force"] for row in per_run),
        "pooled_alignment_failure": sum(row["alignment_failure"] for row in per_run),
        "pooled_force_related_failures": pooled_force_failures,
        "pooled_failures": pooled_failures,
        "pooled_force_related_fraction_of_failures": (
            pooled_force_failures / max(pooled_failures, 1)
        ),
        "pooled_force_related_fraction_of_all_episodes": (
            pooled_force_failures / (len(per_run) * args.episodes)
        ),
        "protocol": {key: runs[0]["payload"].get(key) for key in PROTOCOL_KEYS},
        "per_run": per_run,
    }
    manifest = output_dir / "base_policy_summary.json"
    manifest.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    csv_path = output_dir / "base_policy_per_run.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)
    report = [
        "# SealTwist frozen base policy 三次重复评估",
        "",
        "## 协议",
        "",
        f"使用冻结 Flow base policy 完成 3 个独立 run，每个 run {args.episodes} 个 episode。"
        "三个 run 使用互不重叠的 reset seed 区间；力阈值固定为 `[5.8, 11.0] N`，"
        "瓶盖位置在 X/Y 方向各随机化 ±1 cm。力反馈目标为 7.7 N，policy 不进行任何更新。",
        "",
        "## 结果",
        "",
        "| Seed | 成功率 | 接触阶段平均力 (N) | Under | Over | Alignment | 力相关失败 / 全部 episode |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_run:
        report.append(
            f"| {row['seed_start']} | {row['successes']}/{row['episodes']} "
            f"({pct(row['success_rate'])}) | {row['contact_phase_mean_force_n']:.4f} | "
            f"{row['under_force']} | {row['over_force']} | {row['alignment_failure']} | "
            f"{pct(row['force_related_fraction_of_all_episodes'])} |"
        )
    report.extend(
        [
            "",
            "## 汇总",
            "",
            f"- 成功率：**{pct(success_mean)} ± {pct(success_sd)}**（3 个 run 的样本标准差）。",
            f"- 接触阶段平均力：**{force_mean:.4f} ± {force_sd:.4f} N**。",
            f"- 力相关失败占全部 episode：**{pct(failure_fraction_mean)} ± "
            f"{pct(failure_fraction_sd)}**。",
            f"- 合并计数：under={summary['pooled_under_force']}，"
            f"over={summary['pooled_over_force']}，alignment={summary['pooled_alignment_failure']}。",
            f"- 力相关失败占全部失败：{pooled_force_failures}/{pooled_failures} "
            f"({pct(summary['pooled_force_related_fraction_of_failures'])})；占全部 300 episode："
            f"{pct(summary['pooled_force_related_fraction_of_all_episodes'])}。",
            "",
            "接触阶段平均力按所有双指接触物理步加权；不是终止时刻力。",
        ]
    )
    (output_dir / "BASE_POLICY_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
