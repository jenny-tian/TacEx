#!/usr/bin/env python3
"""Run an audited real-GPT/Qwen RAPID-Tactile online ablation sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_rapid_ablation import ARMS, read_json, summarize, validate_run


ROOT = Path(__file__).resolve().parents[3]
BC_SHA256 = "72cd3ed1f32a672dfc307572019ce1aab8c0b4da4c4f527e278321a32a51c465"
DINO_SHA256 = "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
CODE_FILES = (
    "scripts/reinforcement_learning/vlm_dsrl/run_experiment.py",
    "scripts/reinforcement_learning/vlm_dsrl/run_experiment_runtime.py",
    "scripts/reinforcement_learning/vlm_dsrl/hybrid_wrapper.py",
    "scripts/reinforcement_learning/vlm_dsrl/episode_trainer.py",
    "scripts/reinforcement_learning/vlm_dsrl/run_rapid_ablation.py",
    "scripts/reinforcement_learning/vlm_dsrl/analyze_rapid_ablation.py",
    "scripts/reinforcement_learning/dsrl/clean_dsrl_sac.py",
    "scripts/reinforcement_learning/dsrl/clean_dsrl_agent.py",
    "scripts/reinforcement_learning/dsrl/clean_dsrl_wrapper.py",
    "scripts/bc_training/train_bc.py", "scripts/bc_training/dinov3_flow.py",
    "source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py",
    "source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env_cfg.py",
    "source/tacex_tasks/tacex_tasks/lab_pick/vlm_force/advisor.py",
    "source/tacex_tasks/tacex_tasks/lab_pick/vlm_force/loop.py",
    "source/tacex_tasks/tacex_tasks/lab_pick/vlm_force/estimator.py",
    "source/tacex_tasks/tacex_tasks/lab_pick/vlm_force/contracts.py",
    "exp_report/method0829/guarded_controller.py", "exp_report/method0829/guarded_wrapper.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def source_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in CODE_FILES}


def command_for(spec: dict[str, Any], protocol: dict[str, Any]) -> list[str]:
    arm = ARMS[spec["arm"]]
    command = [sys.executable, str(Path(__file__).with_name("run_experiment.py")),
        "--mode", arm["mode"], "--phase", "online_training",
        "--bc-policy", protocol["bc_policy"], "--output-dir", spec["output_dir"],
        "--num-episodes", str(spec["episodes"]), "--seed", str(spec["seed"]),
        "--paired-episode-seeds", "--break-force-threshold-n", "4.5",
        "--physical-force-range-n", "0.5", "3.8", "--initial-force-range-n", "0.8", "1.4",
        "--labware-random-xy-m", "0.10", "0.10", "--labware-random-yaw-deg", "0",
        "--learned-noise-steps", "1", "--dsrl-noise-residual-scale", "0.25",
        "--dsrl-action-l2-weight", "10", "--dsrl-max-gradient-updates", str(protocol["dsrl_max_gradient_updates"]),
        "--chunk-execute-steps", "32", "--chunk-discount", "0.99", "--flow-num-inference-steps", "20",
        "--learning-starts", "32", "--batch-size", "32", "--initial-log-std", "-2.0",
        "--actor-lr", "3e-5", "--critic-lr", "3e-5", "--alpha-lr", "3e-5",
        "--gpu-max-rigid-contact-count", "1048576", "--gpu-max-rigid-patch-count", "1048576",
        "--max-outer-interactions", str(40 * spec["episodes"]),
        "--checkpoint-interval", "250", "--write-interval", "25"]
    if arm["advisor"]:
        command += ["--advisor", arm["advisor"], "--save-advisor-images",
                    "--force-adaptation-schedule", arm["schedule"]]
        if arm["advisor"] == "openai":
            command += ["--vlm-model", "gpt-5.6-sol", "--vlm-reasoning-effort", "high",
                        "--vlm-timeout-s", "180", "--api-base", protocol["api_base"], "--api-mode", "responses"]
        else:
            command += ["--qwen-model-path", protocol["qwen_model_path"], "--qwen-dtype", "bfloat16",
                        "--qwen-max-new-tokens", "512", "--qwen-max-validation-attempts", "3"]
    if not arm["dsrl"]:
        command += ["--disable-dsrl"]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bc-policy", type=Path, default=Path.home() / "github_repo/TacEx/outputs/lab_pick_dinov3_flow_bc200_yaw0/best.pt")
    parser.add_argument("--dino-path", type=Path, default=Path.home() / "models/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--qwen-model-path", type=Path, default=Path.home() / "models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--api-base", default="https://litellm.limx.cn/v1")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--smoke-root", type=Path, help="Required verified smoke run before a formal launch.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds or ([85200] if args.stage == "smoke" else [5200, 6200, 7200])
    episodes = args.num_episodes or (4 if args.stage == "smoke" else 100)
    if episodes < 4:
        raise ValueError("Use at least four episodes to verify DSRL updates after warm-up.")
    if len(set(seeds)) != len(seeds) or len(set(args.arms)) != len(args.arms):
        raise ValueError("Seeds and arms must be unique.")
    output_root = args.output_root.expanduser().resolve()
    bc, dino, qwen = (p.expanduser().resolve() for p in (args.bc_policy, args.dino_path, args.qwen_model_path))
    if sha256(bc) != BC_SHA256 or sha256(dino / "model.safetensors") != DINO_SHA256:
        raise ValueError("Frozen BC or DINO checkpoint fingerprint mismatch.")
    if not (dino / "config.json").is_file():
        raise FileNotFoundError(dino / "config.json")
    if "qwen_rapid" in args.arms and read_json(qwen / "config.json").get("model_type") != "qwen3_vl":
        raise ValueError("Expected the downloaded Qwen3-VL snapshot.")
    code = source_hashes()
    if args.stage == "formal":
        if args.smoke_root is None:
            raise ValueError("A formal launch requires --smoke-root with validated results.")
        smoke_root = args.smoke_root.expanduser().resolve()
        smoke = read_json(smoke_root / "protocol.json")
        summarize(smoke_root, require_complete=True)
        if smoke["code_sha256"] != code or not set(args.arms).issubset(smoke["arms"]):
            raise ValueError("The formal code/arms differ from the verified smoke experiment.")
    runs = []
    for seed in seeds:
        for arm in args.arms:
            name = f"{arm}_seed{seed}_n{episodes}"
            runs.append({"name": name, "arm": arm, "seed": seed, "episodes": episodes,
                         "output_dir": str(output_root / "runs" / name)})
    protocol = {"schema_version": 1, "stage": args.stage, "created_at_utc": utcnow(),
        "arms": args.arms, "seeds": seeds, "episodes": episodes, "runs": runs,
        "bc_policy": str(bc), "bc_checkpoint_sha256": BC_SHA256,
        "dino_path": str(dino), "dino_sha256": DINO_SHA256,
        "qwen_model_path": str(qwen), "api_base": args.api_base,
        "gpt_model": "gpt-5.6-sol", "gpt_reasoning_effort": "high",
        "dsrl_contract_version": 4, "dsrl_max_gradient_updates": 500,
        "first_fixed_definition": "Episode zero uses [0.8,1.4] N and is counted; its first GPT recommendation is used directly for every later episode without estimator fusion.",
        "metric": "Mean and sample SD of per-seed online success rates, including all 100 episodes for formal runs.",
        "no_vlm_definition": "DSRL only; removes both VLM and the external force controller, matching the historical column's component definition.",
        "historical_results_policy": "Historical deterministic-advisor v3 results are provenance only, not merged into these v4 real-VLM results.",
        "code_sha256": code, "smoke_root": str(args.smoke_root) if args.smoke_root else None}
    if args.dry_run:
        print(json.dumps({"protocol": protocol, "commands": [command_for(spec, protocol) for spec in runs]}, indent=2))
        return
    output_root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output_root / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists():
        previous = read_json(protocol_path)
        if {k: v for k, v in previous.items() if k != "created_at_utc"} != {k: v for k, v in protocol.items() if k != "created_at_utc"}:
            raise ValueError("Refusing to resume with changed experiment inputs.")
        protocol = previous
    else:
        write_json(protocol_path, protocol)
    env = os.environ.copy()
    env["TACEX_ISAAC_PYTHON"] = sys.executable
    env["TACEX_DINOV3_PATH"] = str(dino)
    env["PYTHONUNBUFFERED"] = "1"
    if any(ARMS[arm]["advisor"] == "openai" for arm in args.arms) and not env.get("OPENAI_API_KEY"):
        # Credential stays in the child environment; never serialize it.
        auth = read_json(Path.home() / ".codex/auth.json")
        key = auth.get("OPENAI_API_KEY")
        if not isinstance(key, str) or not key.strip():
            raise RuntimeError("The configured GPT API credential is unavailable.")
        env["OPENAI_API_KEY"] = key
    status: dict[str, Any] = {"status": "running", "runner_pid": os.getpid(), "started_at_utc": utcnow(), "completed_runs": [], "total_runs": len(runs)}
    child: subprocess.Popen | None = None

    def interrupted(signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        status.update(status="interrupted", signal=signum, updated_at_utc=utcnow())
        write_json(output_root / "status.json", status)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for spec in runs:
            if source_hashes() != code:
                raise RuntimeError("Experiment source changed after protocol freeze.")
            run_dir = Path(spec["output_dir"])
            result_path = run_dir / "results.json"
            if result_path.is_file():
                validate_run(result_path, spec, protocol)
                status["completed_runs"].append(spec["name"])
                continue
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(f"Incomplete run exists; preserve it and use a fresh output root: {run_dir}")
            command = command_for(spec, protocol)
            log_path = output_root / "console_logs" / f"{spec['name']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(output_root / "launches" / f"{spec['name']}.json", {"command": command, "spec": spec, "started_at_utc": utcnow()})
            print(f"[RUN] {spec['name']}", flush=True)
            with log_path.open("w") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while True:
                    returncode = child.poll()
                    episode_path = run_dir / "episodes.jsonl"
                    completed = sum(1 for line in episode_path.open() if line.strip()) if episode_path.exists() else 0
                    status.update(current_run=spec["name"], child_pid=child.pid,
                                  current_completed_episodes=completed, current_episode_target=spec["episodes"],
                                  current_log=str(log_path), updated_at_utc=utcnow())
                    write_json(output_root / "status.json", status)
                    if returncode is not None:
                        break
                    time.sleep(10)
            if returncode != 0:
                raise RuntimeError(f"{spec['name']} exited {returncode}; see {log_path}")
            audited = validate_run(result_path, spec, protocol)
            write_json(output_root / "audits" / f"{spec['name']}.json", audited)
            status["completed_runs"].append(spec["name"])
            summarize(output_root)
            print(f"[VERIFIED] {spec['name']}: {audited['successes']}/{spec['episodes']}, advisor_calls={audited['advisor_calls']}, updates={audited['dsrl_gradient_updates']}", flush=True)
        summarize(output_root, require_complete=True)
        status.update(status="complete", finished_at_utc=utcnow(), child_pid=None)
        write_json(output_root / "status.json", status)
        print("[COMPLETE] All requested runs passed execution audits.", flush=True)
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", updated_at_utc=utcnow())
        write_json(output_root / "status.json", status)
        raise


if __name__ == "__main__":
    main()
