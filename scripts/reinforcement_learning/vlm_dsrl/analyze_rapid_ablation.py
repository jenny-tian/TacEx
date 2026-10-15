#!/usr/bin/env python3
"""Validate and summarize the real-VLM RAPID-Tactile online ablation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ARMS = {
    "gpt_rapid": {"mode": "ours_tactile", "advisor": "openai", "dsrl": True, "schedule": "every_episode", "label": "RAPID-Tactile (GPT-5.6-sol)"},
    "gpt_first_fixed": {"mode": "ours_tactile", "advisor": "openai", "dsrl": True, "schedule": "first_episode_only", "label": "Fixed first GPT range"},
    "qwen_rapid": {"mode": "ours_tactile", "advisor": "qwen_local", "dsrl": True, "schedule": "every_episode", "label": "RAPID-Tactile (Qwen3-VL-2B)"},
    "gpt_no_dsrl": {"mode": "ours_tactile", "advisor": "openai", "dsrl": False, "schedule": "every_episode", "label": "w/o DSRL (same guarded controller)"},
    "no_vlm": {"mode": "dsrl_tactile", "advisor": None, "dsrl": True, "schedule": "every_episode", "label": "w/o VLM and force controller (DSRL only)"},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def same_range(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (list, tuple))
        and isinstance(right, (list, tuple))
        and len(left) == len(right) == 2
        and all(abs(float(a) - float(b)) <= 1e-9 for a, b in zip(left, right))
    )


def validate_run(path: Path, spec: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """Check measured execution, not only the requested launch arguments."""
    result = read_json(path)
    arm = str(spec["arm"])
    contract = ARMS[arm]
    count, seed = int(spec["episodes"]), int(spec["seed"])
    rows = result.get("results", [])
    require(result.get("phase") == "online_training", f"{path}: wrong phase")
    require(result.get("mode") == contract["mode"], f"{path}: wrong mode")
    require(result.get("completed_episodes") == count and len(rows) == count, f"{path}: incomplete run")
    require([r.get("episode_index") for r in rows] == list(range(count)), f"{path}: episode indices")
    require([r.get("reset_seed") for r in rows] == list(range(seed, seed + count)), f"{path}: reset seeds are not paired")
    require(result.get("bc_checkpoint_sha256") == protocol["bc_checkpoint_sha256"], f"{path}: BC mismatch")
    require(result.get("labware_random_yaw_deg") == 0, f"{path}: yaw must be zero")
    require(same_range(result.get("labware_random_xy_m"), [0.1, 0.1]), f"{path}: XY randomization mismatch")
    require(result.get("break_force_threshold_n") == 4.5, f"{path}: break threshold mismatch")
    require(same_range(result.get("physical_force_range_n"), [0.5, 3.8]), f"{path}: physical bounds mismatch")
    require(same_range(result.get("initial_force_range_n"), [0.8, 1.4]), f"{path}: initial range mismatch")
    require(result.get("chunk_execute_steps") == 32 and result.get("flow_num_inference_steps") == 20, f"{path}: Flow configuration mismatch")
    require(result.get("action_repeat") == 2, f"{path}: action repeat mismatch")
    updates = int(result.get("dsrl_gradient_updates", -1))
    require(0 < updates <= protocol["dsrl_max_gradient_updates"] if contract["dsrl"] else updates == 0, f"{path}: unexpected DSRL update count {updates}")
    require(result.get("dsrl_contract_version") == 4, f"{path}: DSRL contract is not v4")
    require(result.get("input_dsrl_checkpoint") is None, f"{path}: run started from trained DSRL")
    require(result.get("dsrl_enabled") is contract["dsrl"], f"{path}: DSRL switch mismatch")
    require(result.get("paired_episode_seeds") is True, f"{path}: paired resets disabled")
    if contract["dsrl"]:
        require((path.parent / "dsrl_final.pt").is_file(), f"{path}: missing learned checkpoint")
    else:
        require(not (path.parent / "dsrl_final.pt").exists(), f"{path}: unexpected learned checkpoint")
    expected_calls = 0 if contract["advisor"] is None else 1 if contract["schedule"] == "first_episode_only" else count
    require(result.get("advisor_calls") == expected_calls, f"{path}: wrong advisor count")
    interaction_path = path.parent / "vlm_episode_interactions.jsonl"
    transactions = [json.loads(line) for line in interaction_path.read_text().splitlines() if line.strip()] if interaction_path.is_file() else []
    interactions = [row for row in transactions if row.get("advisor_called") is True]
    require(len(transactions) == (count if expected_calls else 0), f"{path}: missing episode adaptation transactions")
    require(len(interactions) == expected_calls, f"{path}: interaction log count mismatch")
    require(sum(row.get("advisor_called") is True for row in rows) == expected_calls, f"{path}: episode-level advisor count mismatch")
    require(result.get("advisor") == contract["advisor"], f"{path}: advisor mismatch")
    require(bool(result.get("advisor_is_real_vlm")) == bool(expected_calls), f"{path}: real-VLM flag mismatch")
    controller = result.get("force_controller")
    if expected_calls:
        require(isinstance(controller, dict) and bool(controller.get("guarded_supervisor")), f"{path}: PEARS guarded controller missing")
        require(same_range(rows[0].get("attempted_force_range_n"), [0.8, 1.4]), f"{path}: incorrect first episode range")
        for call in interactions:
            images = call.get("image_paths", [])
            require(len(images) == 4 and all(Path(p).is_file() for p in images), f"{path}: missing four-view image evidence")
            require(bool(call.get("prompt")), f"{path}: missing prompt")
            if contract["advisor"] == "openai":
                require(call.get("advisor_returned_model") == "gpt-5.6-sol", f"{path}: returned GPT identity mismatch")
        if contract["advisor"] == "openai":
            require(result.get("vlm_model") == "gpt-5.6-sol", f"{path}: requested GPT identity mismatch")
            require(result.get("vlm_reasoning_effort") == "high", f"{path}: GPT reasoning effort mismatch")
            require(all(call.get("advisor_reasoning_effort") == "high" for call in interactions), f"{path}: missing reasoning audit")
        else:
            require("Qwen3-VL-2B-Instruct" in str(result.get("vlm_model")), f"{path}: wrong Qwen model")
    else:
        require(controller is None, f"{path}: no-VLM arm unexpectedly used a controller")
    first_range = None
    if contract["schedule"] == "first_episode_only":
        require([row.get("advisor_called") for row in transactions] == [True] + [False] * (count - 1), f"{path}: fixed schedule called after episode zero")
        require(all(row.get("advisor_generation_attempts") == 0 and row.get("advisor_transport_attempts") == 0 for row in transactions[1:]), f"{path}: fixed schedule regenerated a response")
        first_range = interactions[0]["vlm_recommendation"]["target_contact_force_range_n"]
        require(same_range(rows[0].get("next_force_range_n"), first_range), f"{path}: first recommendation was transformed")
        for row in rows[1:]:
            require(same_range(row.get("attempted_force_range_n"), first_range), f"{path}: fixed range changed")
            require(same_range(row.get("next_force_range_n"), first_range), f"{path}: fixed next range changed")
        require(same_range(result.get("final_force_range_n"), first_range), f"{path}: final fixed range changed")
    successes = sum(bool(row["success"]) for row in rows)
    require(result.get("successes") == successes, f"{path}: success total mismatch")
    failures = Counter(row["terminal_reason"] for row in rows if not row["success"])
    return {
        "arm": arm, "label": contract["label"], "seed": seed, "episodes": count,
        "successes": successes, "sr_pct": 100 * successes / count,
        "dsrl_gradient_updates": updates, "advisor_calls": expected_calls,
        "first_fixed_range_n": first_range,
        "final_force_range_n": result.get("final_force_range_n"),
        "failures": dict(failures), "results_path": str(path), "audit_passed": True,
    }


def summarize(root: Path, *, require_complete: bool = False) -> dict[str, Any]:
    protocol = read_json(root / "protocol.json")
    runs, missing, invalid = [], [], []
    for spec in protocol["runs"]:
        path = Path(spec["output_dir"]) / "results.json"
        if not path.is_file():
            missing.append(spec["name"])
            continue
        try:
            runs.append(validate_run(path, spec, protocol))
        except (KeyError, ValueError, OSError) as exc:
            invalid.append({"name": spec["name"], "error": str(exc)})
    summary = []
    for arm in protocol["arms"]:
        selected = [run for run in runs if run["arm"] == arm]
        if not selected:
            continue
        rates = [run["sr_pct"] for run in selected]
        summary.append({
            "arm": arm, "label": ARMS[arm]["label"], "completed_seeds": len(selected),
            "expected_seeds": len(protocol["seeds"]), "episodes_per_seed": protocol["episodes"],
            "sr_mean_pct": statistics.mean(rates),
            "sr_sample_sd_pct": statistics.stdev(rates) if len(rates) > 1 else None,
            "successes": sum(run["successes"] for run in selected),
            "episodes": sum(run["episodes"] for run in selected),
        })
    report = {"stage": protocol["stage"], "complete": not missing and not invalid,
              "missing_runs": missing, "invalid_runs": invalid, "per_run": runs, "summary": summary}
    (root / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (root / "summary.csv").open("w", newline="") as stream:
        fields = list(summary[0]) if summary else ["arm", "completed_seeds", "sr_mean_pct", "sr_sample_sd_pct"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    lines = ["RAPID-Tactile real-VLM ablation", "", f"Stage: {protocol['stage']}; complete: {report['complete']}.", "",
             "SR includes all online episodes, including the first episode before the first GPT recommendation. ± is sample SD across independent run seeds. Smoke results are engineering checks and must not be used as formal paper results.", "",
             "| Method | Completed seeds | SR (%) | Successes / episodes |", "|---|---:|---:|---:|"]
    for row in summary:
        sd = row["sr_sample_sd_pct"]
        rate = f"{row['sr_mean_pct']:.1f}" + (f" ± {sd:.1f}" if sd is not None else " (one seed)")
        lines.append(f"| {row['label']} | {row['completed_seeds']}/{row['expected_seeds']} | {rate} | {row['successes']}/{row['episodes']} |")
    lines += ["", f"Missing runs: {len(missing)}. Invalid runs: {len(invalid)}."]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    if require_complete:
        require(report["complete"], f"Incomplete experiment: missing={missing}, invalid={invalid}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = summarize(args.root.resolve(), require_complete=args.require_complete)
    print(json.dumps({key: report[key] for key in ("complete", "missing_runs", "invalid_runs", "summary")}, indent=2))


if __name__ == "__main__":
    main()
