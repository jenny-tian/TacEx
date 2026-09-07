"""Aggregate repeated SealTwist BC evaluations into audited JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation_dir", type=Path, default=Path("exp_report/bottle/evaluation/base")
    )
    parser.add_argument(
        "--json_output", type=Path, default=Path("exp_report/bottle/comparison_summary.json")
    )
    parser.add_argument(
        "--csv_output", type=Path, default=Path("exp_report/bottle/comparison_summary.csv")
    )
    args = parser.parse_args()

    paths = sorted(
        args.evaluation_dir.glob("seed_*.json"),
        key=lambda path: int(path.stem.removeprefix("seed_")),
    )
    if not paths:
        raise ValueError(f"No seed_*.json reports under {args.evaluation_dir}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    checkpoint_hashes = {report["checkpoint_sha256"] for report in reports}
    if len(checkpoint_hashes) != 1:
        raise ValueError("Evaluation reports do not use one frozen checkpoint")
    protocol_fields = (
        "minimum_force_randomization_range_n",
        "maximum_force_randomization_range_n",
        "position_randomization_half_range_m",
        "alignment_tolerance_m",
        "torque_window_nm",
        "evaluation_horizon_steps",
        "policy_time_horizon_steps",
        "success_hold_steps",
    )
    for field in protocol_fields:
        values = {json.dumps(report[field], sort_keys=True) for report in reports}
        if len(values) != 1:
            raise ValueError(f"Evaluation reports disagree on protocol field {field}")

    observed_episode_seeds: set[int] = set()
    raw_episode_evidence_valid = True
    for report in reports:
        records = report.get("outcomes", [])
        if len(records) != int(report["episodes"]):
            raise ValueError(f"seed {report['seed_start']} has an incomplete outcome list")
        outcome_counter = Counter(record["outcome"] for record in records)
        expected_counter = Counter(report["failure_counts"])
        expected_counter["success"] = int(report["successes"])
        if outcome_counter != expected_counter:
            raise ValueError(f"seed {report['seed_start']} outcome counts disagree with its summary")
        torque_lower, torque_upper = report["torque_window_nm"]
        for record in records:
            episode_seed = int(record["seed"])
            if episode_seed in observed_episode_seeds:
                raise ValueError(f"Duplicate evaluated episode seed: {episode_seed}")
            observed_episode_seeds.add(episode_seed)
            outcome = record["outcome"]
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
            raw_episode_evidence_valid &= {
                "success": success_evidence,
                "under_force": under_evidence,
                "over_force": over_evidence,
                "alignment_failure": alignment_evidence,
            }[outcome]

    total_episodes = sum(int(report["episodes"]) for report in reports)
    total_successes = sum(int(report["successes"]) for report in reports)
    failure_counts: Counter[str] = Counter()
    for report in reports:
        failure_counts.update(report["failure_counts"])
    total_failures = total_episodes - total_successes
    failure_fraction = {
        name: failure_counts[name] / total_failures
        for name in ("under_force", "over_force", "alignment_failure")
    }
    success_rates = [float(report["success_rate"]) for report in reports]
    aggregate_success_rate = total_successes / total_episodes
    checks = {
        "aggregate_success_within_8pp_of_50_percent": abs(aggregate_success_rate - 0.50) <= 0.08,
        "under_force_within_8pp_of_30_percent_of_failures": abs(
            failure_fraction["under_force"] - 0.30
        )
        <= 0.08,
        "over_force_within_8pp_of_30_percent_of_failures": abs(
            failure_fraction["over_force"] - 0.30
        )
        <= 0.08,
        "alignment_failures_present": failure_counts["alignment_failure"] > 0,
        "exactly_three_exclusive_outcomes": (
            total_successes + sum(failure_counts.values()) == total_episodes
            and set(failure_counts) == {"under_force", "over_force", "alignment_failure"}
        ),
        "single_frozen_checkpoint": len(checkpoint_hashes) == 1,
        "all_episode_labels_have_matching_physical_evidence": raw_episode_evidence_valid,
        "all_evaluated_episode_seeds_are_unique": len(observed_episode_seeds) == total_episodes,
    }
    summary = {
        "task": reports[0]["task"],
        "policy": reports[0]["policy"],
        "checkpoint": reports[0]["checkpoint"],
        "checkpoint_sha256": reports[0]["checkpoint_sha256"],
        "evaluation_repeats": len(reports),
        "episodes_per_repeat": [int(report["episodes"]) for report in reports],
        "seed_starts": [int(report["seed_start"]) for report in reports],
        "success_rate_per_repeat": success_rates,
        "success_rate_mean": statistics.mean(success_rates),
        "success_rate_sample_std": statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0,
        "aggregate": {
            "episodes": total_episodes,
            "successes": total_successes,
            "success_rate": aggregate_success_rate,
            "wilson_95_ci": wilson_interval(total_successes, total_episodes),
            "failures": total_failures,
            "failure_counts": dict(failure_counts),
            "failure_fraction_among_failures": failure_fraction,
        },
        "protocol": {field: reports[0][field] for field in protocol_fields},
        "acceptance_checks": checks,
        "all_acceptance_checks_pass": all(checks.values()),
        "source_reports": [str(path.resolve()) for path in paths],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    rows = []
    for report in reports:
        failures = int(report["failures"])
        rows.append(
            {
                "scope": f"seed_{report['seed_start']}",
                "episodes": report["episodes"],
                "successes": report["successes"],
                "success_rate": report["success_rate"],
                "under_force_count": report["failure_counts"].get("under_force", 0),
                "over_force_count": report["failure_counts"].get("over_force", 0),
                "alignment_failure_count": report["failure_counts"].get("alignment_failure", 0),
                "under_force_fraction_of_failures": report["failure_counts"].get("under_force", 0)
                / failures,
                "over_force_fraction_of_failures": report["failure_counts"].get("over_force", 0)
                / failures,
                "alignment_failure_fraction_of_failures": report["failure_counts"].get(
                    "alignment_failure", 0
                )
                / failures,
            }
        )
    rows.append(
        {
            "scope": "aggregate",
            "episodes": total_episodes,
            "successes": total_successes,
            "success_rate": aggregate_success_rate,
            "under_force_count": failure_counts["under_force"],
            "over_force_count": failure_counts["over_force"],
            "alignment_failure_count": failure_counts["alignment_failure"],
            "under_force_fraction_of_failures": failure_fraction["under_force"],
            "over_force_fraction_of_failures": failure_fraction["over_force"],
            "alignment_failure_fraction_of_failures": failure_fraction["alignment_failure"],
        }
    )
    with args.csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("[SUMMARY] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
