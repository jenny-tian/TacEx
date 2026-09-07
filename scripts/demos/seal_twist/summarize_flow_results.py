#!/usr/bin/env python3
"""Audit and aggregate repeated SealTwist 6D-wrench Flow BC evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


FAILURE_NAMES = ("under_force", "over_force", "alignment_failure")


def wilson_interval(successes: int, total: int) -> list[float]:
    if total < 1:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def audit_episode_evidence(report: dict, seen_seeds: set[int]) -> bool:
    records = report.get("outcomes", [])
    if len(records) != int(report["episodes"]):
        raise ValueError(f"seed {report['seed_start']} has an incomplete outcome list")
    observed_counts = Counter(record["outcome"] for record in records)
    expected_counts = Counter(report["failure_counts"])
    expected_counts["success"] = int(report["successes"])
    if observed_counts != expected_counts:
        raise ValueError(f"seed {report['seed_start']} outcome counts disagree")

    torque_lower, torque_upper = report["torque_window_nm"]
    valid = True
    for record in records:
        episode_seed = int(record["seed"])
        if episode_seed in seen_seeds:
            raise ValueError(f"duplicate evaluated episode seed: {episode_seed}")
        seen_seeds.add(episode_seed)
        under_evidence = (
            record["final_minimum_finger_force_n"] < record["minimum_force_threshold_n"]
            or record["final_torque_nm"] < torque_lower
        )
        over_evidence = (
            max(record["peak_left_finger_force_n"], record["peak_right_finger_force_n"])
            > record["maximum_force_threshold_n"]
            or record["peak_torque_nm"] > torque_upper
        )
        alignment_evidence = (
            record["alignment_error_m"] > record["alignment_tolerance_m"]
            or not record["bilateral_contact"]
        )
        success_evidence = (
            torque_lower <= record["final_torque_nm"] <= torque_upper
            and record["final_minimum_finger_force_n"]
            >= record["minimum_force_threshold_n"]
            and record["final_maximum_finger_force_n"]
            <= record["maximum_force_threshold_n"]
            and record["alignment_error_m"] <= record["alignment_tolerance_m"]
            and record["bilateral_contact"]
        )
        valid &= {
            "success": success_evidence,
            "under_force": under_evidence,
            "over_force": over_evidence,
            "alignment_failure": alignment_evidence,
        }[record["outcome"]]
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("exp_report/bottle/evaluation/flow_6d_force"),
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("exp_report/bottle/comparison_summary.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("exp_report/bottle/flow_6d_force_summary.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("exp_report/bottle/flow_6d_force_summary.csv"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("exp_report/bottle/FLOW_6D_FORCE_RESULTS.md"),
    )
    args = parser.parse_args()

    paths = sorted(
        args.evaluation_dir.glob("seed_*.json"),
        key=lambda path: int(path.stem.removeprefix("seed_")),
    )
    if not paths:
        raise ValueError(f"no seed_*.json reports under {args.evaluation_dir}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(
        report.get("policy") != "seal_twist_6d_wrench_conditional_flow_matching"
        for report in reports
    ):
        raise ValueError("evaluation directory contains a non-Flow policy report")
    if any(report.get("wrench_observation_indices") != [10, 16] for report in reports):
        raise ValueError("an evaluation report does not prove inclusion of all six wrench values")

    checkpoint_hashes = {report["checkpoint_sha256"] for report in reports}
    if len(checkpoint_hashes) != 1:
        raise ValueError("evaluation reports do not share one frozen checkpoint")
    protocol_fields = (
        "minimum_force_randomization_range_n",
        "maximum_force_randomization_range_n",
        "position_randomization_half_range_m",
        "alignment_tolerance_m",
        "torque_window_nm",
        "evaluation_horizon_steps",
        "policy_time_horizon_steps",
        "success_hold_steps",
        "force_violation_dwell_steps",
        "action_horizon",
        "chunk_execute_steps",
        "action_repeat",
        "integration_steps",
        "ode_solver",
        "action_smoothing",
        "gripper_width_offset_m",
        "force_feedback_target_n",
        "force_feedback_gain_m_per_n",
        "force_feedback_max_step_m",
        "force_feedback_start_step",
        "gripper_width_limits_m",
        "normalized_observation_clip",
        "noise_mode",
        "action_representation",
    )
    for field in protocol_fields:
        values = {json.dumps(report[field], sort_keys=True) for report in reports}
        if len(values) != 1:
            raise ValueError(f"evaluation reports disagree on protocol field {field}")

    seen_seeds: set[int] = set()
    evidence_valid = all(audit_episode_evidence(report, seen_seeds) for report in reports)
    total_episodes = sum(int(report["episodes"]) for report in reports)
    total_successes = sum(int(report["successes"]) for report in reports)
    total_failures = total_episodes - total_successes
    failure_counts: Counter[str] = Counter()
    for report in reports:
        failure_counts.update(report["failure_counts"])
    unknown_failures = set(failure_counts) - set(FAILURE_NAMES)
    if unknown_failures:
        raise ValueError(f"unexpected failure labels: {sorted(unknown_failures)}")
    if total_successes + sum(failure_counts.values()) != total_episodes:
        raise ValueError("success and failure counts do not partition all episodes")

    success_rates = [float(report["success_rate"]) for report in reports]
    aggregate_rate = total_successes / total_episodes
    baseline = None
    if args.baseline_summary.is_file():
        baseline_report = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
        baseline = {
            "policy": baseline_report["policy"],
            "checkpoint_sha256": baseline_report["checkpoint_sha256"],
            "episodes": int(baseline_report["aggregate"]["episodes"]),
            "successes": int(baseline_report["aggregate"]["successes"]),
            "success_rate": float(baseline_report["aggregate"]["success_rate"]),
            "flow_minus_baseline_success_rate": aggregate_rate
            - float(baseline_report["aggregate"]["success_rate"]),
        }

    summary = {
        "task": reports[0]["task"],
        "policy": reports[0]["policy"],
        "checkpoint": reports[0]["checkpoint"],
        "checkpoint_sha256": reports[0]["checkpoint_sha256"],
        "observation_fields": reports[0]["observation_fields"],
        "wrench_observation_indices": reports[0]["wrench_observation_indices"],
        "six_dimensional_force_source": reports[0]["six_dimensional_force_source"],
        "evaluation_repeats": len(reports),
        "episodes_per_repeat": [int(report["episodes"]) for report in reports],
        "seed_starts": [int(report["seed_start"]) for report in reports],
        "success_rate_per_repeat": success_rates,
        "success_rate_mean": statistics.mean(success_rates),
        "success_rate_sample_std": statistics.stdev(success_rates) if len(reports) > 1 else 0.0,
        "aggregate": {
            "episodes": total_episodes,
            "successes": total_successes,
            "success_rate": aggregate_rate,
            "wilson_95_ci": wilson_interval(total_successes, total_episodes),
            "failures": total_failures,
            "failure_counts": {name: failure_counts[name] for name in FAILURE_NAMES},
            "failure_fraction_among_failures": {
                name: failure_counts[name] / max(total_failures, 1)
                for name in FAILURE_NAMES
            },
        },
        "baseline_comparison": baseline,
        "protocol": {field: reports[0][field] for field in protocol_fields},
        "audit_checks": {
            "single_frozen_checkpoint": len(checkpoint_hashes) == 1,
            "full_6d_wrench_declared": reports[0]["wrench_observation_indices"] == [10, 16],
            "exclusive_outcomes_cover_all_episodes": (
                total_successes + sum(failure_counts.values()) == total_episodes
            ),
            "all_episode_labels_have_matching_physical_evidence": evidence_valid,
            "all_episode_seeds_are_unique": len(seen_seeds) == total_episodes,
        },
        "source_reports": [str(path.resolve()) for path in paths],
    }
    summary["all_audit_checks_pass"] = all(summary["audit_checks"].values())
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    rows: list[dict[str, object]] = []
    for report in reports:
        failures = int(report["failures"])
        rows.append(
            {
                "scope": f"seed_{report['seed_start']}",
                "episodes": report["episodes"],
                "successes": report["successes"],
                "success_rate": report["success_rate"],
                **{
                    f"{name}_count": report["failure_counts"].get(name, 0)
                    for name in FAILURE_NAMES
                },
                **{
                    f"{name}_fraction_of_failures": report["failure_counts"].get(name, 0)
                    / max(failures, 1)
                    for name in FAILURE_NAMES
                },
            }
        )
    rows.append(
        {
            "scope": "aggregate",
            "episodes": total_episodes,
            "successes": total_successes,
            "success_rate": aggregate_rate,
            **{f"{name}_count": failure_counts[name] for name in FAILURE_NAMES},
            **{
                f"{name}_fraction_of_failures": failure_counts[name]
                / max(total_failures, 1)
                for name in FAILURE_NAMES
            },
        }
    )
    with args.csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    baseline_text = ""
    if baseline is not None:
        baseline_text = (
            f"\n与原 MLP BC（{baseline['successes']}/{baseline['episodes']}，"
            f"{baseline['success_rate']:.1%}）相比，Flow Matching 的成功率变化为 "
            f"{baseline['flow_minus_baseline_success_rate']:+.1%}。\n"
        )
    markdown = f"""# SealTwist 6D Force Flow Matching BC

策略使用 28 维观测，其中索引 `[10:16]` 是完整的
`[Fx, Fy, Fz, Tx, Ty, Tz]` wrench；动作头通过条件 Flow Matching 从标准高斯噪声
积分得到 {reports[0]['action_horizon']} 步 CAFE 动作序列。

| 测试种子 | Episode | 成功数 | 成功率 | 欠力 | 过力 | 对齐失败 |
|---:|---:|---:|---:|---:|---:|---:|
"""
    for report in reports:
        markdown += (
            f"| {report['seed_start']:,} | {report['episodes']} | {report['successes']} | "
            f"{report['success_rate']:.1%} | "
            f"{report['failure_counts'].get('under_force', 0)} | "
            f"{report['failure_counts'].get('over_force', 0)} | "
            f"{report['failure_counts'].get('alignment_failure', 0)} |\n"
        )
    markdown += (
        f"| **合计** | **{total_episodes}** | **{total_successes}** | "
        f"**{aggregate_rate:.1%}** | **{failure_counts['under_force']}** | "
        f"**{failure_counts['over_force']}** | "
        f"**{failure_counts['alignment_failure']}** |\n"
        f"\nWilson 95% 置信区间为 {summary['aggregate']['wilson_95_ci'][0]:.1%}–"
        f"{summary['aggregate']['wilson_95_ci'][1]:.1%}；三次重复的成功率为 "
        f"{summary['success_rate_mean']:.1%} ± {summary['success_rate_sample_std']:.1%}"
        "（均值 ± 样本标准差）。\n"
        f"{baseline_text}\n"
        f"Checkpoint SHA-256: `{summary['checkpoint_sha256']}`。全部审计检查："
        f"`{summary['all_audit_checks_pass']}`。\n"
    )
    args.markdown_output.write_text(markdown, encoding="utf-8")
    print("[SUMMARY] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
