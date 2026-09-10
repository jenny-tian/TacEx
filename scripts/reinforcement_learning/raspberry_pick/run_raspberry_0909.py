#!/usr/bin/env python3
"""Run and gate the preregistered raspberry_0909 smoke/formal experiments."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
TRAINER = ROOT / "scripts/reinforcement_learning/raspberry_pick/train_online_codex_vlm_dsrl.py"
ANALYZER = ROOT / "scripts/reinforcement_learning/raspberry_pick/analyze_raspberry_0909.py"
DEFAULT_PYTHON = Path("/home/limx/anaconda3/envs/env_isaaclab/bin/python")
REPORT_ROOT = ROOT / "exp_report/raspberry_0909"
FORMAL_SEEDS = (970_000, 980_000, 990_000)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assert_finite(value: Any, location: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{location}[{index}]")


def command_for(
    python: Path,
    *,
    method: str,
    output_root: Path,
    seed: int,
    episodes: int,
) -> list[str]:
    if method not in {"dsrl_only", "vlm_two_stage_dsrl"}:
        raise ValueError(method)
    command = [
        str(python),
        str(TRAINER),
        "--output-root",
        str(output_root),
        "--method",
        method,
        "--force-range-source",
        "fixed" if method == "dsrl_only" else "vlm",
        "--controller-mode",
        "single_stage" if method == "dsrl_only" else "two_stage",
        "--bottle-style-lateral-dsrl",
        "--initial-force-range",
        "2.2",
        "2.8",
        "--train-seed",
        str(seed),
        "--train-episodes",
        str(episodes),
        "--exploration-anneal-episodes",
        "30",
        "--checkpoint-every",
        "10" if episodes == 100 else "1",
        "--enhanced-determinism",
        "--headless",
    ]
    if method == "vlm_two_stage_dsrl":
        command.extend(
            (
                "--public-grip-safety-limit-n",
                "3.1",
                "--public-tension-target-n",
                "0.9",
                "--codex-model",
                "gpt-5.6-sol",
                "--codex-reasoning-effort",
                "high",
            )
        )
    if episodes < 100:
        # The three-episode smoke must exercise optimizer updates; the formal
        # learning-start threshold (256) is often not reached that quickly.
        command.extend(("--learning-starts", "64", "--batch-size", "64"))
    return command


def is_complete(output_root: Path, episodes: int) -> bool:
    summary_path = output_root / "training_summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return (
        int(summary.get("completed_episodes", -1)) == episodes
        and (output_root / "checkpoints/final.pt").is_file()
        and (output_root / "file_hashes.json").is_file()
    )


def execute(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("\n[COMMAND] " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"run failed with exit code {result.returncode}; see {log_path}")


def run_one(
    python: Path,
    *,
    method: str,
    output_root: Path,
    seed: int,
    episodes: int,
    staged_resume: bool = False,
) -> None:
    if is_complete(output_root, episodes):
        print(f"[SKIP] complete: {output_root}", flush=True)
        return
    command = command_for(
        python,
        method=method,
        output_root=output_root,
        seed=seed,
        episodes=episodes,
    )
    log_path = output_root.parent / f"{output_root.name}.log"
    if output_root.exists() and any(output_root.iterdir()):
        execute([*command, "--resume"], log_path=log_path)
        return
    if staged_resume:
        execute([*command, "--episodes-this-invocation", "1"], log_path=log_path)
        if not (output_root / "interrupted_summary.json").is_file():
            raise RuntimeError("controlled resume prelude did not leave an incomplete run")
        execute([*command, "--resume"], log_path=log_path)
    else:
        execute(command, log_path=log_path)


def validate_run(
    output_root: Path,
    *,
    method: str,
    episodes: int,
    require_training: bool = True,
) -> dict[str, Any]:
    if not is_complete(output_root, episodes):
        raise ValueError(f"incomplete run: {output_root}")
    records = read_jsonl(output_root / "training_episodes.jsonl")
    decisions = read_jsonl(output_root / "force_range_decisions.jsonl")
    summary = json.loads((output_root / "training_summary.json").read_text(encoding="utf-8"))
    if len(records) != episodes or len(decisions) != episodes:
        raise ValueError(f"episode/decision count mismatch: {output_root}")
    if [int(row["episode_index"]) for row in records] != list(range(episodes)):
        raise ValueError(f"non-contiguous episode indices: {output_root}")
    expected_seeds = list(range(int(summary["training_seed"]), int(summary["training_seed"]) + episodes))
    if [int(row["seed"]) for row in records] != expected_seeds:
        raise ValueError(f"non-contiguous episode seeds: {output_root}")
    assert_finite(records)
    if any(float(row["max_abs_lateral_residual_m"]) > 0.0060001 for row in records):
        raise ValueError(f"lateral residual exceeded 6 mm: {output_root}")
    if require_training and (
        int(summary["replay_transitions"]) <= 0
        or int(summary["sac_optimizer_updates"]) <= 0
    ):
        raise ValueError(f"DSRL did not train: {output_root}")
    if method == "dsrl_only":
        if summary["vlm_calls"] != 0 or summary["vlm_successful_calls"] != 0:
            raise ValueError("fixed arm unexpectedly called the VLM")
        if any(row["attempted_force_range_n"] != [2.2, 2.8] for row in records):
            raise ValueError("fixed arm changed its force interval")
        if any(row["next_force_range_n"] != [2.2, 2.8] for row in records):
            raise ValueError("fixed arm scheduled a force interval change")
    else:
        audit_path = output_root / "codex_vlm_calls.jsonl"
        audits = read_jsonl(audit_path) if audit_path.is_file() else []
        call_policy = str(
            json.loads((output_root / "protocol_manifest.json").read_text(encoding="utf-8")).get(
                "vlm_call_policy", "every_episode"
            )
        )
        called_episodes = {
            int(row["episode_index"])
            for row in records
            if bool(row.get("vlm_called", row.get("vlm_call_index") is not None))
        }
        audited_episodes = {int(row["episode_index"]) for row in audits}
        expected_called_episodes = (
            {int(row["episode_index"]) for row in records if not bool(row["success"])}
            if call_policy == "failure_only"
            else set(range(episodes))
        )
        if called_episodes != expected_called_episodes:
            raise ValueError("VLM call policy does not match episode outcomes")
        if audited_episodes != called_episodes:
            raise ValueError("VLM audit and called-episode sets differ")
        successful_episodes = {
            int(row["episode_index"]) for row in audits if bool(row.get("success"))
        }
        recorded_successes = {
            int(row["episode_index"]) for row in records if bool(row["vlm_call_succeeded"])
        }
        if successful_episodes != recorded_successes:
            raise ValueError("VLM audit and episode success sets differ")
        if int(summary["vlm_calls"]) != len(audits):
            raise ValueError("VLM summary call count does not match the audit log")
        forbidden_keys = ("damage_threshold_n", "detach_threshold_n", "oracle_audit_reference_n")
        for audit in audits:
            request_text = json.dumps(audit.get("request", {}), sort_keys=True)
            if any(key in request_text for key in forbidden_keys):
                raise ValueError("hidden threshold leaked into a VLM request")
        if not any(
            row["bilateral_contact_fraction"] > 0 and row["two_stage_pull_steps"] > 0
            for row in records
        ):
            raise ValueError("no contacted episode entered two-stage tension pulling")
    return {
        "output_root": str(output_root),
        "method": method,
        "episodes": episodes,
        "successes": sum(int(row["success"]) for row in records),
        "last20_success_rate": sum(int(row["success"]) for row in records[-20:])
        / min(20, episodes),
        "vlm_calls": int(summary["vlm_calls"]),
        "vlm_successful_calls": int(summary["vlm_successful_calls"]),
        "replay_transitions": int(summary["replay_transitions"]),
        "sac_updates": int(summary["sac_optimizer_updates"]),
        "resumed": bool(summary.get("resumed")),
    }


def smoke(python: Path) -> None:
    roots = {
        "dsrl_only": REPORT_ROOT / "smoke/dsrl_only/seed969900_n3",
        "vlm_two_stage_dsrl": REPORT_ROOT / "smoke/vlm_two_stage_dsrl/seed969900_n3",
    }
    run_one(
        python,
        method="dsrl_only",
        output_root=roots["dsrl_only"],
        seed=969_900,
        episodes=3,
    )
    run_one(
        python,
        method="vlm_two_stage_dsrl",
        output_root=roots["vlm_two_stage_dsrl"],
        seed=969_900,
        episodes=3,
        staged_resume=True,
    )
    qa = [validate_run(root, method=method, episodes=3) for method, root in roots.items()]
    if not qa[1]["resumed"]:
        raise ValueError("VLM smoke did not exercise checkpoint resume")
    qa_path = REPORT_ROOT / "smoke/SMOKE_QA.json"
    qa_path.write_text(json.dumps({"passed": True, "runs": qa}, indent=2) + "\n")
    print(f"[PASS] smoke gate: {qa_path}", flush=True)


def formal(python: Path) -> None:
    smoke_qa = REPORT_ROOT / "smoke/SMOKE_QA.json"
    if not smoke_qa.is_file() or not json.loads(smoke_qa.read_text())["passed"]:
        raise RuntimeError("formal runs require a passing smoke/SMOKE_QA.json")
    order = (
        (970_000, "dsrl_only"),
        (970_000, "vlm_two_stage_dsrl"),
        (980_000, "vlm_two_stage_dsrl"),
        (980_000, "dsrl_only"),
        (990_000, "dsrl_only"),
        (990_000, "vlm_two_stage_dsrl"),
    )
    completed = []
    for seed, method in order:
        root = REPORT_ROOT / method / f"seed{seed}_n100"
        run_one(python, method=method, output_root=root, seed=seed, episodes=100)
        completed.append(validate_run(root, method=method, episodes=100))
    (REPORT_ROOT / "formal_run_qa.json").write_text(
        json.dumps({"passed": True, "runs": completed}, indent=2) + "\n"
    )
    subprocess.run(
        (str(python), str(ANALYZER), "--report-root", str(REPORT_ROOT)),
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal", "all"), default="all")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parsed = parser.parse_args()
    if parsed.mode in {"smoke", "all"}:
        smoke(parsed.python)
    if parsed.mode in {"formal", "all"}:
        formal(parsed.python)


if __name__ == "__main__":
    main()
