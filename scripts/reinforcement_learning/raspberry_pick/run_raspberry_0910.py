#!/usr/bin/env python3
"""Run and gate the chunked, failure-triggered raspberry_0910 experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE_RUNNER = ROOT / "scripts/reinforcement_learning/raspberry_pick/run_raspberry_0909.py"
DEFAULT_PYTHON = Path("/home/limx/anaconda3/envs/env_isaaclab/bin/python")
REPORT_ROOT = ROOT / "exp_report/raspberry_0910"
PROTOCOL = "raspberry_0910_v1"
FORMAL_SEEDS = (970_000, 980_000, 990_000)


def load_base_runner():
    spec = importlib.util.spec_from_file_location("raspberry_0910_base_runner", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load base RaspberryPick runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_runner()


def command_for(
    python: Path,
    *,
    method: str,
    output_root: Path,
    seed: int,
    episodes: int,
) -> list[str]:
    command = base.command_for(
        python,
        method=method,
        output_root=output_root,
        seed=seed,
        episodes=episodes,
    )
    command.extend(
        (
            "--protocol-name",
            PROTOCOL,
            "--num-inference-steps",
            "10",
            "--chunk-execute-steps",
            "8",
            "--vlm-call-policy",
            "failure_only",
        )
    )
    if episodes < 100:
        # Full-chunk execution yields roughly one eighth as many replay
        # transitions, so the smoke test needs a proportionally smaller gate.
        command.extend(("--learning-starts", "8", "--batch-size", "8"))
    return command


def validate_run(output_root: Path, *, method: str, episodes: int) -> dict[str, object]:
    result = base.validate_run(
        output_root,
        method=method,
        episodes=episodes,
        require_training=episodes >= 100,
    )
    manifest = json.loads((output_root / "protocol_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "protocol": PROTOCOL,
        "flow_num_inference_steps": 10,
        "flow_chunk_size": 8,
        "chunk_execute_steps": 8,
        "action_repeat": 4,
        "vlm_call_policy": "failure_only",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"raspberry_0910 manifest mismatch for {key}: {output_root}")
    records = base.read_jsonl(output_root / "training_episodes.jsonl")
    if any(int(row["chunk_execute_steps"]) != 8 for row in records):
        raise ValueError(f"episode log did not execute 8-action chunks: {output_root}")
    if any(int(row["flow_num_inference_steps"]) != 10 for row in records):
        raise ValueError(f"episode log did not use 10 Flow steps: {output_root}")
    if method == "vlm_two_stage_dsrl":
        called = {int(row["episode_index"]) for row in records if bool(row["vlm_called"])}
        failed = {int(row["episode_index"]) for row in records if not bool(row["success"])}
        if called != failed:
            raise ValueError(f"VLM was not called exactly on failed episodes: {output_root}")
    return result


def run_one(
    python: Path,
    *,
    method: str,
    output_root: Path,
    seed: int,
    episodes: int,
    staged_resume: bool = False,
) -> None:
    if base.is_complete(output_root, episodes):
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
        base.execute([*command, "--resume"], log_path=log_path)
    elif staged_resume:
        base.execute([*command, "--episodes-this-invocation", "1"], log_path=log_path)
        if not (output_root / "interrupted_summary.json").is_file():
            raise RuntimeError("controlled resume prelude did not leave an incomplete run")
        base.execute([*command, "--resume"], log_path=log_path)
    else:
        base.execute(command, log_path=log_path)


def smoke(python: Path) -> None:
    roots = {
        "dsrl_only": REPORT_ROOT / "smoke/dsrl_only/seed969910_n3",
        "vlm_two_stage_dsrl": REPORT_ROOT / "smoke/vlm_two_stage_dsrl/seed969910_n3",
    }
    run_one(
        python,
        method="dsrl_only",
        output_root=roots["dsrl_only"],
        seed=969_910,
        episodes=3,
    )
    run_one(
        python,
        method="vlm_two_stage_dsrl",
        output_root=roots["vlm_two_stage_dsrl"],
        seed=969_910,
        episodes=3,
        staged_resume=True,
    )
    qa_path = REPORT_ROOT / "smoke/SMOKE_QA.json"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        qa = [validate_run(root, method=method, episodes=3) for method, root in roots.items()]
        if not qa[1]["resumed"]:
            raise ValueError("VLM smoke did not exercise checkpoint resume")
    except Exception as exc:
        qa_path.write_text(
            json.dumps(
                {"passed": False, "error": f"{type(exc).__name__}: {exc}"},
                indent=2,
            )
            + "\n"
        )
        raise
    qa_path.write_text(json.dumps({"passed": True, "runs": qa}, indent=2) + "\n")
    print(f"[PASS] smoke gate: {qa_path}", flush=True)


def formal(python: Path) -> None:
    smoke_qa = REPORT_ROOT / "smoke/SMOKE_QA.json"
    if not smoke_qa.is_file() or not json.loads(smoke_qa.read_text())["passed"]:
        raise RuntimeError("formal runs require a passing raspberry_0910 smoke gate")
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
        run_one(
            python,
            method=method,
            output_root=root,
            seed=seed,
            episodes=100,
        )
        completed.append(validate_run(root, method=method, episodes=100))
    (REPORT_ROOT / "formal_run_qa.json").write_text(
        json.dumps({"passed": True, "runs": completed}, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal", "all"), default="all")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    args = parser.parse_args()
    if args.mode in {"smoke", "all"}:
        smoke(args.python)
    if args.mode in {"formal", "all"}:
        formal(args.python)


if __name__ == "__main__":
    main()
