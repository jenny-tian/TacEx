#!/usr/bin/env python3
"""Audit a completed RaspberryPick online Codex-VLM/DSRL training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--window", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    return True


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    episodes_path = run_dir / "training_episodes.jsonl"
    decisions_path = run_dir / "force_range_decisions.jsonl"
    calls_path = run_dir / "codex_vlm_calls.jsonl"
    summary_path = run_dir / "training_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "final.pt"
    records = load_jsonl(episodes_path)
    decisions = load_jsonl(decisions_path)
    calls = load_jsonl(calls_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    expected = int(args.expected_episodes)
    if len(records) != expected or len(decisions) != expected:
        raise ValueError("episode/force-decision count does not match the expected run size")
    if [int(row["episode_index"]) for row in records] != list(range(expected)):
        raise ValueError("episode indices are not contiguous")
    seeds = [int(row["seed"]) for row in records]
    if len(set(seeds)) != expected:
        raise ValueError("episode seeds are not unique")
    successful_calls = [row for row in calls if bool(row.get("success"))]
    if len(successful_calls) != expected:
        raise ValueError("not every episode has one successful Codex VLM call")
    if {int(row["episode_index"]) for row in successful_calls} != set(range(expected)):
        raise ValueError("successful Codex calls do not cover every episode")
    if {row.get("model") for row in successful_calls} != {"gpt-5.6-sol"}:
        raise ValueError("unexpected VLM model in audit log")
    if {row.get("reasoning_effort") for row in successful_calls} != {"high"}:
        raise ValueError("unexpected VLM reasoning effort in audit log")
    if not all(row.get("model_audit_confirmed") for row in successful_calls):
        raise ValueError("model audit was not confirmed for every successful call")
    if not all(row.get("reasoning_effort_audit_confirmed") for row in successful_calls):
        raise ValueError("reasoning-effort audit was not confirmed for every call")
    if not all_finite(records) or not all_finite(decisions) or not all_finite(checkpoint):
        raise ValueError("non-finite value detected in run artifacts")
    base_checkpoint = Path(checkpoint["base_checkpoint"])
    if sha256(base_checkpoint) != checkpoint["base_checkpoint_sha256"]:
        raise ValueError("base checkpoint hash mismatch")
    if checkpoint["base_checkpoint_sha256"] != summary["base_checkpoint_sha256"]:
        raise ValueError("summary and DSRL checkpoint reference different base checkpoints")

    image_count = len(list((run_dir / "advisor_frames").glob("episode_*.png")))
    trajectory_count = len(list((run_dir / "trajectories").glob("episode_*.jsonl.gz")))
    response_count = len(list((run_dir / "codex_vlm").glob("episode_*_response.json")))
    if image_count != expected or trajectory_count != expected or response_count < expected:
        raise ValueError("episode evidence files are incomplete")

    outcomes = [int(row["success"]) for row in records]
    rolling = []
    for index in range(expected):
        start = max(0, index - args.window + 1)
        rolling.append(sum(outcomes[start : index + 1]) / (index - start + 1))
    full_windows = rolling[min(args.window - 1, expected - 1) :]
    maximum = max(full_windows)
    maximum_end = min(
        index + 1
        for index, value in enumerate(rolling)
        if index >= args.window - 1 and value == maximum
    )
    successes = sum(outcomes)
    failures = Counter(row["outcome"] for row in records if not row["success"])
    result = {
        "contract": checkpoint["contract"],
        "run_dir": str(run_dir),
        "training_seed": seeds[0],
        "episode_seed_range": [min(seeds), max(seeds)],
        "episodes": expected,
        "unique_episode_seeds": len(set(seeds)),
        "successes": successes,
        "online_training_success_rate": successes / expected,
        "failure_counts": dict(failures),
        "rolling_window": args.window,
        "initial_full_window_success_rate": rolling[args.window - 1],
        "maximum_full_window_success_rate": maximum,
        "maximum_window_episode_range": [maximum_end - args.window + 1, maximum_end],
        "final_window_success_rate": rolling[-1],
        "first_half_successes": sum(outcomes[: expected // 2]),
        "second_half_successes": sum(outcomes[expected // 2 :]),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": checkpoint["base_checkpoint_sha256"],
        "final_checkpoint": str(checkpoint_path),
        "final_checkpoint_sha256": sha256(checkpoint_path),
        "replay_transitions": int(checkpoint["replay_transitions"]),
        "sac_optimizer_updates": int(checkpoint["updates"]),
        "noise_action_indices": list(checkpoint["noise_action_indices"]),
        "noise_residual_scale": float(checkpoint["noise_residual_scale"]),
        "initial_force_range_n": summary["initial_force_range_n"],
        "final_force_range_n": summary["final_force_range_n"],
        "force_range_changes": sum(
            row["attempted_force_range_n"] != row["next_force_range_n"]
            for row in records
        ),
        "vlm_calls_total_attempts": len(calls),
        "vlm_successful_calls": len(successful_calls),
        "vlm_model": "gpt-5.6-sol",
        "vlm_reasoning_effort": "high",
        "advisor_images": image_count,
        "trajectory_files": trajectory_count,
        "response_json_files_total_attempts": response_count,
        "training_log_sha256": sha256(episodes_path),
        "all_numeric_values_finite": True,
        "all_base_checkpoint_hashes_match": True,
        "all_vlm_model_audits_confirmed": True,
    }
    output_path = run_dir / "run_audit.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
