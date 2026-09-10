#!/usr/bin/env python3
"""Audit and summarize one paired DSRL vs privileged-force DSRL run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--oracle-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=30)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(path: Path, expected_episodes: int) -> dict[str, Any]:
    summary_path = path / "training_summary.json"
    episodes_path = path / "training_episodes.jsonl"
    if not summary_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"incomplete run: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if int(summary["completed_episodes"]) != expected_episodes:
        raise ValueError(f"{summary_path}: expected {expected_episodes} completed episodes")
    if len(episodes) != expected_episodes:
        raise ValueError(f"{episodes_path}: expected {expected_episodes} episode records")
    if int(summary["successes"]) != sum(int(row["success"]) for row in episodes):
        raise ValueError(f"{path}: summary success count does not match episode records")
    if [int(row["episode_index"]) for row in episodes] != list(range(expected_episodes)):
        raise ValueError(f"{path}: episode indices are incomplete or out of order")
    return {
        "path": path.resolve(),
        "summary_path": summary_path.resolve(),
        "episodes_path": episodes_path.resolve(),
        "summary": summary,
        "episodes": episodes,
    }


def rate(rows: list[dict[str, Any]]) -> float:
    return sum(int(row["success"]) for row in rows) / max(len(rows), 1)


def format_force_range(values: list[float]) -> str:
    return f"[{float(values[0]):.1f}, {float(values[1]):.1f}]"


def validate_pair(baseline: dict[str, Any], oracle: dict[str, Any]) -> None:
    base_summary = baseline["summary"]
    oracle_summary = oracle["summary"]
    if base_summary["post_training_method"] != "dsrl_only":
        raise ValueError("baseline run is not dsrl_only")
    if oracle_summary["post_training_method"] != "oracle_force_dsrl":
        raise ValueError("oracle run is not oracle_force_dsrl")
    if bool(oracle_summary["vlm_is_codex"]) or int(oracle_summary["vlm_calls"]) != 0:
        raise ValueError("oracle run made a VLM call")
    for field in (
        "training_seed",
        "completed_episodes",
        "berry_profile",
        "base_checkpoint_sha256",
        "residual_mode",
        "noise_action_indices",
        "noise_residual_scale",
        "dsrl_start_step",
        "position_randomization_m",
    ):
        if base_summary.get(field) != oracle_summary.get(field):
            raise ValueError(f"paired contract mismatch for {field}")
    for base_row, oracle_row in zip(baseline["episodes"], oracle["episodes"], strict=True):
        if int(base_row["seed"]) != int(oracle_row["seed"]):
            raise ValueError("paired episode seeds do not match")
    oracle_spec = oracle_summary.get("oracle_force_spec")
    if not isinstance(oracle_spec, dict):
        raise ValueError("oracle run does not contain its ground-truth force derivation")
    expected_range = [oracle_spec["minimum_n"], oracle_spec["maximum_n"]]
    if oracle_summary["initial_force_range_n"] != expected_range:
        raise ValueError("oracle initial force range does not match its derivation")
    if oracle_summary["final_force_range_n"] != expected_range:
        raise ValueError("oracle force range changed during training")
    if any(row["attempted_force_range_n"] != expected_range for row in oracle["episodes"]):
        raise ValueError("oracle force range was not held for every episode")


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    rows = run["episodes"]
    successes = sum(int(row["success"]) for row in rows)
    return {
        "method": run["summary"]["post_training_method"],
        "successes": successes,
        "episodes": len(rows),
        "success_rate": successes / len(rows),
        "first_10_success_rate": rate(rows[:10]),
        "last_10_success_rate": rate(rows[-10:]),
        "failure_counts": dict(Counter(row["outcome"] for row in rows if not row["success"])),
        "force_range_n": run["summary"]["initial_force_range_n"],
        "vlm_calls": int(run["summary"]["vlm_calls"]),
        "sac_optimizer_updates": int(run["summary"]["sac_optimizer_updates"]),
        "summary_json": str(run["summary_path"]),
        "summary_sha256": sha256(run["summary_path"]),
        "episodes_jsonl": str(run["episodes_path"]),
        "episodes_sha256": sha256(run["episodes_path"]),
    }


def main() -> None:
    args = parse_args()
    baseline = load_run(args.baseline_run.expanduser().resolve(), args.expected_episodes)
    oracle = load_run(args.oracle_run.expanduser().resolve(), args.expected_episodes)
    validate_pair(baseline, oracle)

    baseline_stats = summarize(baseline)
    oracle_stats = summarize(oracle)
    paired = Counter()
    source_rows: list[dict[str, Any]] = []
    for base_row, oracle_row in zip(baseline["episodes"], oracle["episodes"], strict=True):
        base_success = bool(base_row["success"])
        oracle_success = bool(oracle_row["success"])
        key = (
            "both_success"
            if base_success and oracle_success
            else "oracle_only"
            if oracle_success
            else "baseline_only"
            if base_success
            else "both_fail"
        )
        paired[key] += 1
        source_rows.append(
            {
                "episode_index": int(base_row["episode_index"]),
                "episode_seed": int(base_row["seed"]),
                "baseline_outcome": base_row["outcome"],
                "baseline_success": int(base_success),
                "oracle_outcome": oracle_row["outcome"],
                "oracle_success": int(oracle_success),
            }
        )

    gain = oracle_stats["success_rate"] - baseline_stats["success_rate"]
    result = {
        "experiment": "single-seed privileged ground-truth force + DSRL ablation",
        "interpretation": "descriptive single-seed online-training comparison",
        "paired_training_seed": int(baseline["summary"]["training_seed"]),
        "berry_profile": baseline["summary"]["berry_profile"],
        "episodes_per_method": args.expected_episodes,
        "vlm_was_called": False,
        "oracle_derivation": oracle["summary"]["oracle_force_spec"],
        "baseline": baseline_stats,
        "oracle_force_dsrl": oracle_stats,
        "absolute_gain": gain,
        "absolute_gain_percentage_points": 100.0 * gain,
        "oracle_exceeds_dsrl": gain > 0.0,
        "paired_episode_counts": dict(paired),
        "contract_validation": "passed",
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "episode_source.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    outcome = "超过" if result["oracle_exceeds_dsrl"] else "未超过"
    baseline_force = format_force_range(baseline_stats["force_range_n"])
    oracle_force = format_force_range(oracle_stats["force_range_n"])
    report = f"""# RaspberryPick 真值力 + DSRL 单 seed 实验

## 结论

真值力 + DSRL 在该单 seed 在线训练实验中{outcome}纯 DSRL：
`{oracle_stats['successes']}/{args.expected_episodes}`
（{100 * oracle_stats['success_rate']:.1f}%）对
`{baseline_stats['successes']}/{args.expected_episodes}`
（{100 * baseline_stats['success_rate']:.1f}%），差值
{100 * gain:+.1f} 个百分点。

这是单个、时间相关的 online-training run，只能作为机制验证，不能给出跨 seed
统计显著性结论。

## 严格配对协议

- training seed：`{result['paired_training_seed']}`，每臂 {args.expected_episodes} episodes；
- profile：`{result['berry_profile']}`，两臂 episode seed 完全一致；
- BC checkpoint、DSRL residual、SAC、位姿随机化和训练预算一致；
- DSRL 基线固定力区间：`{baseline_force} N`；
- 真值臂使用真实 damage threshold `{result['oracle_derivation']['source_damage_threshold_n']} N`，
  减去 `{result['oracle_derivation']['damage_margin_n']} N` 安全裕量，并保持
  `{result['oracle_derivation']['interval_width_n']} N` 带宽，得到
  `{oracle_force} N`；
- 真值臂 VLM 调用：`{oracle_stats['vlm_calls']}`；力区间在全部 episodes 中保持不变；
- 自动配对合同审计：`{result['contract_validation']}`。

## 结果明细

| 方法 | 成功 | 成功率 | 首 10 | 末 10 | VLM 调用 | 失败分布 |
|---|---:|---:|---:|---:|---:|---|
| DSRL | {baseline_stats['successes']}/{args.expected_episodes} | {100 * baseline_stats['success_rate']:.1f}% | {100 * baseline_stats['first_10_success_rate']:.1f}% | {100 * baseline_stats['last_10_success_rate']:.1f}% | {baseline_stats['vlm_calls']} | `{baseline_stats['failure_counts']}` |
| 真值力 + DSRL | {oracle_stats['successes']}/{args.expected_episodes} | {100 * oracle_stats['success_rate']:.1f}% | {100 * oracle_stats['first_10_success_rate']:.1f}% | {100 * oracle_stats['last_10_success_rate']:.1f}% | {oracle_stats['vlm_calls']} | `{oracle_stats['failure_counts']}` |

逐 episode 同 seed 计数：`{dict(paired)}`。由于 GPU PhysX 的接触求解不保证跨进程
逐步确定，这一计数用于诊断，不当作严格 matched-pair 因果检验。

原始结果、哈希和输入路径见 `comparison_summary.json`，逐 episode 对照见
`episode_source.csv`。
"""
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
