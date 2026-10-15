#!/usr/bin/env python3
"""Evaluate six frozen RaspberryPick policies on paired held-out scenes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "exp_report/raspberry_0909/scenario_revision_zero_pull"
TRAINER = Path(__file__).with_name("train_online_codex_vlm_dsrl.py")
PYTHON = Path("/home/limx/anaconda3/envs/env_isaaclab/bin/python")
SEEDS = (970000, 980000, 990000)
METHODS = {
    "base": ("Base policy", "dsrl_only"),
    "residual_rl": ("Residual RL", "baseline_comparison_n100/residual_rl"),
    "awr": ("AWR", "baseline_comparison_n100/awr"),
    "dppo": ("DPPO", "baseline_comparison_n100/dppo"),
    "dsrl": ("DSRL", "dsrl_only"),
    "rapid_tactile": ("RAPID-Tactile", "vlm_two_stage_dsrl"),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_option(argv, flag, value):
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv.extend((flag, str(value)))


def write_progress(output, runs):
    """Publish completed episodes while the simulation writes its next episode."""
    per_run = []
    for run in runs:
        root = Path(run["run_root"])
        path = root / "training_episodes.jsonl"
        lines = path.read_text().splitlines(keepends=True) if path.exists() else []
        rows = [json.loads(line) for line in lines if line.endswith("\n") and line.strip()]
        per_run.append({
            "method": run["method"], "training_seed": run["training_seed"],
            "completed_episodes": len(rows), "target_episodes": run["episodes"],
            "successes": sum(row["success"] for row in rows),
            "run_complete": (root / "training_summary.json").is_file(),
        })
    progress = {
        "updated_at": time.time(),
        "completed_episodes": sum(row["completed_episodes"] for row in per_run),
        "target_episodes": sum(row["target_episodes"] for row in per_run),
        "completed_runs": sum(row["run_complete"] for row in per_run),
        "target_runs": len(runs), "runs": per_run,
    }
    write_json(output / "progress.json", progress)
    lines = ["# RaspberryPick 评估进度", "",
             f"已完成 {progress['completed_episodes']} / {progress['target_episodes']} 回合；"
             f"{progress['completed_runs']} / {progress['target_runs']} 组。", "",
             "运行中的成功率是临时结果，完整校验后的结果见 RESULTS.md。", "",
             "| 方法 | 训练 seed | 已完成 / 计划 | 成功次数 | 状态 |",
             "|---|---:|---:|---:|---|"]
    lines.extend(
        f"| {METHODS[row['method']][0]} | {row['training_seed']} | "
        f"{row['completed_episodes']} / {row['target_episodes']} | {row['successes']} | "
        f"{'完成' if row['run_complete'] else '未完成'} |" for row in per_run
    )
    (output / "PROGRESS.md").write_text("\n".join(lines) + "\n")


def make_run(source, output, method, seed, episodes, python):
    source_run = source / METHODS[method][1] / f"seed{seed}_n100"
    manifest = read_json(source_run / "protocol_manifest.json")
    summary = read_json(source_run / "training_summary.json")
    if summary["completed_episodes"] != 100:
        raise ValueError(f"incomplete training: {source_run}")
    eval_seed = seed + 1000000
    run_root = output / method / f"trainseed{seed}_evalseed{eval_seed}_n{episodes}"
    # Reuse each training invocation so all unmodified scene/controller options persist.
    argv = list(manifest["argv"])
    argv = argv[next(i for i, value in enumerate(argv) if value.startswith("--")):]
    if "--resume" in argv:
        argv.remove("--resume")
    for flag, value in (
        ("--output-root", run_root), ("--train-seed", eval_seed),
        ("--train-episodes", episodes), ("--chunk-execute-steps", 4),
        ("--action-repeat", 4), ("--pull-z-offset-m", 0.0),
        ("--checkpoint", summary["base_checkpoint"]),
        ("--protocol-name", "raspberry_zero_pull_final_evaluation_chunk4_v1"),
        ("--checkpoint-every", episodes),
    ):
        set_option(argv, flag, value)
    argv.extend(("--evaluation-mode", "--freeze-force-adaptation"))
    checkpoint = None if method == "base" else source_run / "checkpoints/final.pt"
    if checkpoint is None:
        argv.append("--freeze-dsrl")
    else:
        argv.extend(("--agent-checkpoint", str(checkpoint)))
    training_episode_seeds = {row["seed"] for row in read_jsonl(source_run / "training_episodes.jsonl")}
    if training_episode_seeds.intersection(range(eval_seed, eval_seed + episodes)):
        raise ValueError("evaluation scenes overlap training")
    return {
        "method": method, "display_name": METHODS[method][0],
        "training_seed": seed, "evaluation_seed_start": eval_seed,
        "episodes": episodes, "source_run": str(source_run),
        "source_checkpoint": None if checkpoint is None else str(checkpoint),
        "source_checkpoint_sha256": None if checkpoint is None else sha256(checkpoint),
        "base_checkpoint_sha256": summary["base_checkpoint_sha256"],
        "run_root": str(run_root), "command": [str(python), str(TRAINER), *argv],
    }


def audit_run(run):
    import torch

    root = Path(run["run_root"])
    summary = read_json(root / "training_summary.json")
    rows = read_jsonl(root / "training_episodes.jsonl")
    expected = {
        "evaluation_mode": True, "freeze_force_adaptation": True,
        "completed_episodes": run["episodes"], "chunk_execute_steps": 4,
        "action_repeat": 4, "flow_num_inference_steps": 20,
        "optimizer_updates_this_run": 0, "replay_transitions": 0, "vlm_calls": 0,
        "base_checkpoint_sha256": run["base_checkpoint_sha256"],
        "dsrl_frozen": run["method"] == "base",
        "loaded_agent_checkpoint": run["source_checkpoint"],
        "controller_mode": "two_stage" if run["method"] == "rapid_tactile" else "single_stage",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"{root}: {key}: {summary.get(key)!r} != {value!r}")
    if len(rows) != run["episodes"]:
        raise ValueError(f"incomplete episode log: {root}")
    final = torch.load(root / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    if run["source_checkpoint"]:
        if sha256(run["source_checkpoint"]) != run["source_checkpoint_sha256"]:
            raise ValueError("source checkpoint changed during evaluation")
        initial = torch.load(run["source_checkpoint"], map_location="cpu", weights_only=False)
        for key in ("actor", "policy", "value", "q1", "q2", "target_q1", "target_q2"):
            if key in initial:
                if initial[key].keys() != final[key].keys() or any(
                    not torch.equal(value, final[key][name]) for name, value in initial[key].items()
                ):
                    raise ValueError(f"{root}: network weights changed: {key}")
        force_range = initial["current_force_range_n"]
    else:
        force_range = [2.2, 2.8]
    if summary["initial_force_range_n"] != force_range or summary["final_force_range_n"] != force_range:
        raise ValueError(f"force interval changed: {root}")
    for index, row in enumerate(rows):
        if row["episode_index"] != index or row["seed"] != run["evaluation_seed_start"] + index:
            raise ValueError(f"evaluation seed/index mismatch: {root}")
        if row["chunk_execute_steps"] != 4 or row["action_repeat"] != 4:
            raise ValueError(f"incorrect execution frequency: {root}")
        if row["policy_decisions"] != math.ceil(row["steps"] / 16):
            raise ValueError(f"inference was not once per four actions: {root}")
        if row["executed_chunk_actions"] != math.ceil(row["steps"] / 4):
            raise ValueError(f"action count mismatch: {root}")
        if row["replay_size"] or row["vlm_called"]:
            raise ValueError(f"unexpected adaptation: {root}")
        if row["attempted_force_range_n"] != force_range or row["next_force_range_n"] != force_range:
            raise ValueError(f"episode force interval changed: {root}")
        if run["method"] == "base" and row["mean_abs_residual_correction"] != [0.0]:
            raise ValueError(f"base policy has a residual: {root}")
    successes = sum(row["success"] for row in rows)
    if summary["successes"] != successes:
        raise ValueError("summary success count differs from raw episodes")
    return {
        "method": run["method"], "training_seed": run["training_seed"],
        "episodes": len(rows), "successes": successes, "success_rate": successes / len(rows),
        "failure_counts": dict(Counter(row["outcome"] for row in rows if not row["success"])),
        "elapsed_seconds": summary["elapsed_seconds"],
        "policy_decisions": sum(row["policy_decisions"] for row in rows),
        "executed_chunk_actions": sum(row["executed_chunk_actions"] for row in rows),
        "evaluation_weights_unchanged": True,
        "raw_episode_sha256": sha256(root / "training_episodes.jsonl"),
        "run_root": str(root),
    }, rows


def summarize(output, runs):
    audits, episodes = [], []
    for run in runs:
        audit, rows = audit_run(run)
        audits.append(audit)
        for row in rows:
            episodes.append({
                "method": run["method"], "training_seed": run["training_seed"],
                "episode_seed": row["seed"], "success": int(row["success"]),
                "terminal_reason": row["outcome"], "steps": row["steps"],
                "policy_decisions": row["policy_decisions"],
                "executed_chunk_actions": row["executed_chunk_actions"],
            })
    results = []
    for method in dict.fromkeys(run["method"] for run in runs):
        subset = [r for r in audits if r["method"] == method]
        successes, count = sum(r["successes"] for r in subset), sum(r["episodes"] for r in subset)
        rates = [row["success_rate"] for row in subset]
        results.append({"method": method, "display_name": METHODS[method][0],
                        "successes": successes, "episodes": count, "success_rate": successes / count,
                        "n_training_seeds": len(subset), "seed_success_rate_mean": statistics.mean(rates),
                        "seed_success_rate_sd": statistics.stdev(rates) if len(rates) > 1 else 0.0})
    per_seed = [{key: row[key] for key in ("method", "training_seed", "successes", "episodes", "success_rate")}
                for row in audits]
    for name, rows in (("success_rates.csv", results), ("per_seed_success_rates.csv", per_seed),
                       ("episode_outcomes.csv", episodes)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / "evaluation_audit.json", {"passed": True, "runs": audits})
    lines = ["# RaspberryPick zero-pull 最终策略评估", "",
             "每次 Flow 推理后执行预测序列的前 4 个动作，每动作 4 个物理步；Flow 仍使用 20 步推理。",
             f"训练 seed：{', '.join(str(seed) for seed in dict.fromkeys(r['training_seed'] for r in runs))}。",
             "配对测试场景：" + "; ".join(
                 f"{r['evaluation_seed_start']}–{r['evaluation_seed_start'] + r['episodes'] - 1}"
                 for r in runs if r['method'] == runs[0]['method']
             ) + "。",
             "Base 为相同冻结 BC + 单阶段力控制器，残差为零。各方法使用配对测试 seed。",
             "模型参数及最终力区间被冻结，无优化器更新、replay 写入或新 VLM 调用；RAPID 保留两阶段控制器。", "",
             "| 方法 | 成功 / 回合数 | 成功率 |", "|---|---:|---:|"]
    lines.extend(f"| {r['display_name']} | {r['successes']} / {r['episodes']} | {r['success_rate']:.1%} |" for r in results)
    lines.extend(["", "| 方法 | 训练 seed | 成功 / 回合数 | 成功率 |",
                  "|---|---:|---:|---:|"])
    lines.extend(f"| {METHODS[r['method']][0]} | {r['training_seed']} | {r['successes']} / {r['episodes']} | {r['success_rate']:.1%} |"
                 for r in per_seed)
    lines.extend(["", "这是改变动作执行频率后的独立评估，不能替代原始逐动作推理的训练曲线。",
                  "RAPID 使用两阶段夹持和张力拉取控制器，其他方法使用单阶段夹持控制器；结果比较的是各自完整控制系统。",
                  "", "逐回合结果：`episode_outcomes.csv`；逐 seed 结果：`per_seed_success_rates.csv`；原始日志与 checkpoint 来源：`evaluation_manifest.json`；校验：`evaluation_audit.json`。"])
    command = [runs[0]["command"][0], str(Path(__file__).relative_to(ROOT)),
               "--source-root", str(Path(runs[0]["source_run"]).parents[1] if runs[0]["method"] not in {"residual_rl", "awr", "dppo"}
                                    else Path(runs[0]["source_run"]).parents[2]),
               "--output-root", str(output), "--episodes", str(runs[0]["episodes"]),
               "--seeds", *[str(seed) for seed in dict.fromkeys(r["training_seed"] for r in runs)],
               "--methods", *dict.fromkeys(r["method"] for r in runs)]
    lines.extend(["", "运行命令（在 TacEx 根目录）：", "", "```bash",
                  shlex.join(command), "```"])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--output-root", type=Path, default=SOURCE / "final_evaluation_chunk4_n10")
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=(SEEDS[0],))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 1000:
        parser.error("episodes must be between 1 and 1000")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("seeds and methods must not contain duplicates")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = [make_run(args.source_root.resolve(), output, method, seed, args.episodes, args.python)
            for seed in args.seeds for method in args.methods]
    manifest = {"scenario": str(args.source_root.resolve()), "chunk_execute_steps": 4,
                "action_repeat": 4, "evaluation_mode": "deterministic_frozen_final_policy",
                "runs": runs}
    manifest_path = output / "evaluation_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("output manifest differs from the requested evaluation")
    write_json(manifest_path, manifest)
    write_progress(output, runs)
    if args.plan_only:
        print(f"Prepared {len(runs)} runs / {len(runs) * args.episodes} episodes: {manifest_path}")
        return
    if not args.analyze_only:
        for index, run in enumerate(runs, 1):
            root = Path(run["run_root"])
            if (root / "training_summary.json").exists():
                audit_run(run)
                print(f"[SKIP] {run['method']} seed={run['training_seed']}", flush=True)
                continue
            if root.exists() and any(root.iterdir()):
                raise RuntimeError(f"Inspect incomplete output before retrying: {root}")
            root.parent.mkdir(parents=True, exist_ok=True)
            log = root.with_suffix(".log")
            print(f"[RUN {index}/{len(runs)}] {run['method']} seed={run['training_seed']} log={log}", flush=True)
            with log.open("a") as stream:
                process = subprocess.Popen(run["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                write_json(output / "active_run.json", {"driver_pid": os.getpid(), "child_pid": process.pid,
                           "started_at": time.time(), "method": run["method"], "training_seed": run["training_seed"],
                           "run_root": str(root), "log": str(log)})
                while True:
                    try:
                        code = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        write_progress(output, runs)
            if code:
                raise RuntimeError(f"evaluation exited {code}: {log}")
            audit, _ = audit_run(run)
            write_progress(output, runs)
            print(f"[DONE] {run['method']} seed={run['training_seed']} successes={audit['successes']}/{audit['episodes']}", flush=True)
    summarize(output, runs)
    write_progress(output, runs)
    write_json(output / "active_run.json", {"status": "complete", "completed_at": time.time()})


if __name__ == "__main__":
    main()
