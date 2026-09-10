#!/usr/bin/env python3
"""Summarize a paired RaspberryPick base-versus-two-stage evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


MATCH_FIELDS = (
    "task",
    "checkpoint_sha256",
    "seed_start",
    "num_envs",
    "episodes",
    "max_episode_steps",
    "damage_force_randomization_range_n",
    "detach_force_randomization_range_n",
    "randomize_force_thresholds",
    "fixed_damage_threshold_n",
    "fixed_detach_threshold_n",
    "randomize_raspberry_pose",
    "raspberry_position_randomization_m",
    "action_repeat",
    "chunk_execute_steps",
    "action_smoothing",
    "position_command_offset_m",
    "gripper_command_offset_m",
    "pull_z_offset_m",
    "visual_xy_override",
    "visual_xy_lock_phase",
    "checkpoint_weights",
    "noise_mode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--two-stage", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report.get("outcomes"), list):
        raise ValueError(f"{path} has no outcomes list")
    return report


def exact_mcnemar_p(base_loss_to_win: int, base_win_to_loss: int) -> float:
    discordant = base_loss_to_win + base_win_to_loss
    if discordant == 0:
        return 1.0
    lower = min(base_loss_to_win, base_win_to_loss)
    probability = sum(math.comb(discordant, k) for k in range(lower + 1)) / 2**discordant
    return min(1.0, 2.0 * probability)


def mean_metric(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return fmean(values) if values else 0.0


def main() -> None:
    args = parse_args()
    base = load_report(args.base.resolve())
    two_stage = load_report(args.two_stage.resolve())

    mismatches = {
        field: {"base": base.get(field), "two_stage": two_stage.get(field)}
        for field in MATCH_FIELDS
        if base.get(field) != two_stage.get(field)
    }
    if mismatches:
        raise ValueError(f"paired reports have mismatched deployment settings: {mismatches}")
    if base.get("control") != "base_policy_only":
        raise ValueError("base report is not labeled base_policy_only")
    if two_stage.get("control") != "oracle_two_stage_grip_and_pedicle_tension":
        raise ValueError("two-stage report has the wrong control label")

    base_by_seed = {int(record["seed"]): record for record in base["outcomes"]}
    force_by_seed = {int(record["seed"]): record for record in two_stage["outcomes"]}
    if set(base_by_seed) != set(force_by_seed):
        raise ValueError("paired reports do not contain the same episode seeds")

    seeds = sorted(base_by_seed)
    pairs = []
    base_loss_to_win = 0
    base_win_to_loss = 0
    for seed in seeds:
        before = base_by_seed[seed]
        after = force_by_seed[seed]
        before_success = before["outcome"] == "success"
        after_success = after["outcome"] == "success"
        base_loss_to_win += int(not before_success and after_success)
        base_win_to_loss += int(before_success and not after_success)
        pairs.append(
            {
                "seed": seed,
                "base_outcome": before["outcome"],
                "two_stage_outcome": after["outcome"],
                "changed": before["outcome"] != after["outcome"],
            }
        )

    total = len(seeds)
    base_successes = sum(record["outcome"] == "success" for record in base_by_seed.values())
    force_successes = sum(
        record["outcome"] == "success" for record in force_by_seed.values()
    )
    intervention_seeds = [
        seed
        for seed in seeds
        if int(force_by_seed[seed].get("force_grip_active_steps", 0)) > 0
        or int(force_by_seed[seed].get("force_safety_override_steps", 0)) > 0
        or int(force_by_seed[seed].get("force_pull_command_steps", 0)) > 0
    ]
    intervention_base_successes = sum(
        base_by_seed[seed]["outcome"] == "success" for seed in intervention_seeds
    )
    intervention_force_successes = sum(
        force_by_seed[seed]["outcome"] == "success" for seed in intervention_seeds
    )
    changed_without_intervention = sum(
        pair["changed"] and pair["seed"] not in intervention_seeds for pair in pairs
    )
    base_rate = base_successes / total if total else 0.0
    force_rate = force_successes / total if total else 0.0
    delta = force_rate - base_rate
    p_value = exact_mcnemar_p(base_loss_to_win, base_win_to_loss)
    statistically_supported = delta > 0.0 and p_value < 0.05
    policy_name = str(base.get("policy", ""))
    is_groot = "groot" in policy_name.lower() or "gr00t" in policy_name.lower()

    summary = {
        "task": base.get("task"),
        "evidence_scope": (
            "requested_groot_validation" if is_groot else "compatible_controller_calibration_not_groot"
        ),
        "policy": policy_name,
        "checkpoint": base.get("checkpoint"),
        "checkpoint_sha256": base.get("checkpoint_sha256"),
        "paired_seed_start": base.get("seed_start"),
        "paired_episodes": total,
        "base": {
            "successes": base_successes,
            "success_rate": base_rate,
            "failure_counts": dict(
                Counter(
                    record["outcome"]
                    for record in base_by_seed.values()
                    if record["outcome"] != "success"
                )
            ),
            "mean_peak_grip_force_n": mean_metric(list(base_by_seed.values()), "peak_grip_force_n"),
            "mean_peak_pedicle_tension_n": mean_metric(
                list(base_by_seed.values()), "peak_stem_tension_n"
            ),
        },
        "two_stage": {
            "successes": force_successes,
            "success_rate": force_rate,
            "failure_counts": dict(
                Counter(
                    record["outcome"]
                    for record in force_by_seed.values()
                    if record["outcome"] != "success"
                )
            ),
            "mean_peak_grip_force_n": mean_metric(
                list(force_by_seed.values()), "peak_grip_force_n"
            ),
            "mean_peak_pedicle_tension_n": mean_metric(
                list(force_by_seed.values()), "peak_stem_tension_n"
            ),
            "controller_config": two_stage.get("two_stage_force_controller_config"),
            "oracle_grip_damage_margin_n": two_stage.get("oracle_grip_damage_margin_n"),
            "oracle_grip_range_width_n": two_stage.get("oracle_grip_range_width_n"),
            "oracle_tension_margin_n": two_stage.get("oracle_tension_margin_n"),
        },
        "paired_effect": {
            "absolute_success_rate_change": delta,
            "relative_success_rate_change": (
                delta / base_rate if base_rate > 0.0 else None
            ),
            "base_failure_to_two_stage_success": base_loss_to_win,
            "base_success_to_two_stage_failure": base_win_to_loss,
            "exact_mcnemar_p_value": p_value,
            "effective_improvement_supported_at_alpha_0_05": statistically_supported,
        },
        "controller_intervention_subset": {
            "episodes": len(intervention_seeds),
            "seeds": intervention_seeds,
            "base_successes": intervention_base_successes,
            "two_stage_successes": intervention_force_successes,
            "outcome_changes_without_controller_intervention": changed_without_intervention,
        },
        "deployment_settings": {field: base.get(field) for field in MATCH_FIELDS},
        "paired_outcomes": pairs,
    }

    if statistically_supported:
        conclusion = "该配对样本支持双阶段力控提高成功率。"
    elif delta > 0.0:
        conclusion = "成功率有描述性提升，但该快速样本尚不足以在 0.05 水平证明有效提升。"
    elif delta < 0.0 and changed_without_intervention > 0:
        conclusion = (
            "观测成功率下降，但成败反转发生在控制器未实际介入的回合，"
            "因此不能把下降归因于力控；该样本仍不支持有效提升。"
        )
    elif delta < 0.0:
        conclusion = "该配对样本中双阶段力控未提高成功率，且成功率下降。"
    else:
        conclusion = "该配对样本中双阶段力控没有改变成功率。"
    if not is_groot:
        conclusion += " 本结果仅验证控制器与兼容 DINOv3 base policy 的组合，不是 GR00T 证据。"
    summary["conclusion"] = conclusion

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    markdown = f"""# RaspberryPick 双阶段力控配对验证

{conclusion}

| 指标 | Base | 双阶段力控 | 变化 |
|---|---:|---:|---:|
| 成功数 | {base_successes}/{total} | {force_successes}/{total} | {force_successes - base_successes:+d} |
| 成功率 | {base_rate:.1%} | {force_rate:.1%} | {delta:+.1%} |
| 平均峰值夹持力 | {summary['base']['mean_peak_grip_force_n']:.3f} N | {summary['two_stage']['mean_peak_grip_force_n']:.3f} N | {summary['two_stage']['mean_peak_grip_force_n'] - summary['base']['mean_peak_grip_force_n']:+.3f} N |
| 平均峰值果梗张力 | {summary['base']['mean_peak_pedicle_tension_n']:.3f} N | {summary['two_stage']['mean_peak_pedicle_tension_n']:.3f} N | {summary['two_stage']['mean_peak_pedicle_tension_n'] - summary['base']['mean_peak_pedicle_tension_n']:+.3f} N |

- Base 失败变成功：{base_loss_to_win}
- Base 成功变失败：{base_win_to_loss}
- 控制器实际介入：{len(intervention_seeds)}/{total} 回合；该子集成功数 {intervention_base_successes} → {intervention_force_successes}
- 控制器未介入但结果改变：{changed_without_intervention} 回合（GPU 物理仿真的重复运行差异）
- 配对精确 McNemar p 值：{p_value:.6f}
- Policy：`{policy_name}`
- Checkpoint SHA-256：`{base.get('checkpoint_sha256')}`
- 固定真值阈值：压碎 {base.get('fixed_damage_threshold_n')} N，脱落 {base.get('fixed_detach_threshold_n')} N
"""
    with args.output_markdown.open("w", encoding="utf-8") as stream:
        stream.write(markdown)


if __name__ == "__main__":
    main()
