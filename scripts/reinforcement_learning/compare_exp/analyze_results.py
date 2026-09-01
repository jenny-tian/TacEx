#!/usr/bin/env python3
"""Build machine-readable tables for the compare_exp online/evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DISPLAY = {
    "guarded_joint": "Guarded VLM+DSRL (ours)",
    "joint": "VLM+DSRL",
    "dsrl": "DSRL",
    "vlm": "VLM",
    "residual_rl": "Residual RL (SAC)",
    "flow_rwr": "Direct Flow-RWR",
    "base": "Frozen base",
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total < 1:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def outcome_rows(method: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(payload.get("results", []), start=1):
        reason = item.get("terminal_reason", item.get("failure_reason"))
        rows.append(
            {
                "method": method,
                "display_name": DISPLAY.get(method, method),
                "episode": index,
                "reset_seed": item.get("reset_seed", item.get("seed")),
                "success": int(bool(item["success"])),
                "failure_reason": "" if item["success"] else reason,
                "episode_return": item.get("episode_return", ""),
                "outer_interactions": item.get("outer_interactions", item.get("outer_steps", "")),
                "ending_outer_interaction": item.get("ending_outer_interaction", ""),
                "physics_steps": item.get("physics_steps", ""),
                "peak_contact_force_n": item.get("peak_contact_force_n", item.get("peak_force_n", "")),
            }
        )
    return rows


def analyze(root: Path) -> dict[str, Any]:
    manifest_path = root / "comparison/manifest.json"
    manifest = read_json(manifest_path)
    online_rows: list[dict[str, Any]] = []
    evaluation_outcome_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    online_payloads: dict[str, dict[str, Any]] = {}
    for entry in manifest["methods"]:
        method = str(entry["method"])
        online = read_json(entry["online_result"])
        online_payloads[method] = online
        method_online = outcome_rows(method, online)
        if not method_online:
            raise ValueError(f"{method} completed no online episode within its interaction budget.")
        online_rows.extend(method_online)
        successes = 0
        window: list[int] = []
        cumulative_physics_steps = 0
        for row in method_online:
            value = int(row["success"])
            successes += value
            window.append(value)
            if row["physics_steps"] != "":
                cumulative_physics_steps += int(row["physics_steps"])
            if len(window) > 10:
                window.pop(0)
            curve_rows.append(
                {
                    "method": method,
                    "episode": row["episode"],
                    "ending_outer_interaction": row["ending_outer_interaction"],
                    "nominal_physics_steps_consumed": (
                        ""
                        if row["ending_outer_interaction"] == ""
                        else int(row["ending_outer_interaction"])
                        * (2 if method == "residual_rl" else 64)
                    ),
                    "actual_completed_episode_physics_steps": cumulative_physics_steps,
                    "success": value,
                    "cumulative_success_rate": successes / int(row["episode"]),
                    "rolling_10_success_rate": sum(window) / len(window),
                }
            )
        evaluated = read_json(entry["evaluation_result"])
        method_evaluation = outcome_rows(method, evaluated)
        if len(method_evaluation) != int(manifest["evaluation_episodes"]):
            raise ValueError(
                f"{method} has {len(method_evaluation)} evaluation episodes; "
                f"expected {manifest['evaluation_episodes']}."
            )
        evaluation_outcome_rows.extend(method_evaluation)
        n = int(evaluated["completed_episodes"] if "completed_episodes" in evaluated else evaluated["num_trials"])
        successes = int(evaluated["successes"])
        low, high = wilson(successes, n)
        evaluation_rows.append(
            {
                "method": method,
                "display_name": DISPLAY.get(method, method),
                "dsrl_equivalent_outer_interactions": manifest[
                    "dsrl_outer_interaction_budget"
                ],
                "training_outer_interactions": entry.get(
                    "training_outer_interactions",
                    manifest["dsrl_outer_interaction_budget"],
                ),
                "evaluation_episodes": n,
                "successes": successes,
                "success_rate": successes / n,
                "wilson_95_low": low,
                "wilson_95_high": high,
                "failure_counts": json.dumps(evaluated.get("failure_counts", {}), sort_keys=True),
                "online_result": entry["online_result"],
                "evaluation_result": entry["evaluation_result"],
            }
        )

    seeded_groups: dict[str, list[int]] = {}
    for row in evaluation_outcome_rows:
        if row["reset_seed"] not in (None, ""):
            seeded_groups.setdefault(str(row["method"]), []).append(int(row["reset_seed"]))
    if seeded_groups:
        expected_seeds = next(iter(seeded_groups.values()))
        mismatched = {
            method: seeds for method, seeds in seeded_groups.items() if seeds != expected_seeds
        }
        if mismatched:
            raise ValueError(f"Evaluation seed pairing mismatch: {mismatched}")

    fields = [
        "method",
        "display_name",
        "episode",
        "reset_seed",
        "success",
        "failure_reason",
        "episode_return",
        "outer_interactions",
        "ending_outer_interaction",
        "physics_steps",
        "peak_contact_force_n",
    ]
    write_csv(root / "online_episode_outcomes.csv", online_rows, fields)
    write_csv(
        root / "online_success_curve.csv",
        curve_rows,
        [
            "method",
            "episode",
            "ending_outer_interaction",
            "nominal_physics_steps_consumed",
            "actual_completed_episode_physics_steps",
            "success",
            "cumulative_success_rate",
            "rolling_10_success_rate",
        ],
    )
    write_csv(
        root / "evaluation_summary.csv",
        evaluation_rows,
        list(evaluation_rows[0]) if evaluation_rows else [],
    )
    write_csv(
        root / "evaluation_episode_outcomes.csv",
        evaluation_outcome_rows,
        fields,
    )
    ranking = sorted(evaluation_rows, key=lambda row: (-float(row["success_rate"]), str(row["method"])))
    ablation_methods = {"dsrl", "vlm", "joint"}
    budget_search_path = root / "budget_search/budget_search.json"
    budget_rows: list[dict[str, Any]] = []
    budget_search = None
    if budget_search_path.is_file():
        budget_search = read_json(budget_search_path)
        for item in budget_search.get("runs", []):
            budget_rows.append(
                {
                    "method": "dsrl",
                    "train_interactions": item["train_interactions"],
                    "evaluation_episodes": manifest["evaluation_episodes"],
                    "successes": item["successes"],
                    "success_rate": item["success_rate"],
                    "actor_lr": item.get("actor_lr", ""),
                    "initial_log_std": item.get("initial_log_std", ""),
                    "base_successes": budget_search["base_successes"],
                    "base_success_rate": budget_search["base_success_rate"],
                    "strictly_exceeds_base": item["strictly_exceeds_base"],
                }
            )
        write_csv(
            root / "budget_search.csv",
            budget_rows,
            [
                "method",
                "train_interactions",
                "evaluation_episodes",
                "successes",
                "success_rate",
                "actor_lr",
                "initial_log_std",
                "base_successes",
                "base_success_rate",
                "strictly_exceeds_base",
            ],
        )
    summary = {
        "schema_version": 1,
        "dsrl_outer_interaction_budget": manifest["dsrl_outer_interaction_budget"],
        "nominal_training_physics_steps": manifest[
            "nominal_training_physics_steps"
        ],
        "evaluation_episodes": manifest["evaluation_episodes"],
        "ranking": ranking,
        "ablation": [row for row in evaluation_rows if row["method"] in ablation_methods],
        "budget_search": budget_search,
        "online_episode_outcomes": str(root / "online_episode_outcomes.csv"),
        "online_success_curve": str(root / "online_success_curve.csv"),
        "evaluation_summary": str(root / "evaluation_summary.csv"),
        "evaluation_episode_outcomes": str(root / "evaluation_episode_outcomes.csv"),
        "vlm_advisor": online_payloads.get("vlm", {}).get("advisor"),
        "vlm_advisor_is_real_vlm": online_payloads.get("vlm", {}).get(
            "advisor_is_real_vlm"
        ),
    }
    (root / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# compare_exp 结果摘要",
        "",
        f"在线预算：DSRL/Flow 每种 {manifest['dsrl_outer_interaction_budget']} 个 32-action 决策，"
        f"Residual RL 为 {manifest['residual_outer_interaction_budget']} 个单步决策；"
        f"名义物理步均为 {manifest['nominal_training_physics_steps']}。"
        f"独立评测：{manifest['evaluation_episodes']} 个显式配对 seed。",
        "",
        "| 方法 | 成功/评测 | 成功率 | 95% Wilson CI |",
        "|---|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(
            f"| {row['display_name']} | {row['successes']}/{row['evaluation_episodes']} | "
            f"{100 * float(row['success_rate']):.1f}% | "
            f"{100 * float(row['wilson_95_low']):.1f}–{100 * float(row['wilson_95_high']):.1f}% |"
        )

    if budget_search is not None:
        selected_budget = budget_search.get("selected_budget")
        selected_trial = next(
            (
                item
                for item in budget_search.get("runs", [])
                if int(item["train_interactions"]) == int(selected_budget)
                and bool(item.get("selected", item.get("strictly_exceeds_base", False)))
            ),
            None,
        )
        if selected_trial is not None:
            search_n = manifest["evaluation_episodes"]
            search_seed_start = int(budget_search["evaluation_seed_start"])
            lines.extend(
                [
                    "",
                    "## DSRL 预算选择",
                    "",
                    f"在搜索 seed {search_seed_start}–{search_seed_start + search_n - 1} 上，"
                    f"100 次交互后的 DSRL 为 {selected_trial['successes']}/{search_n} "
                    f"({100 * float(selected_trial['success_rate']):.1f}%)，"
                    f"同 seed frozen base 为 {budget_search['base_successes']}/{search_n} "
                    f"({100 * float(budget_search['base_success_rate']):.1f}%)。"
                    f"因此选择 100 次交互、actor LR={float(selected_trial['actor_lr']):g}。",
                ]
            )

    rows_by_method = {str(row["method"]): row for row in evaluation_rows}
    if all(method in rows_by_method for method in ("dsrl", "vlm", "joint")):
        lines.extend(
            [
                "",
                "## 消融",
                "",
                "| 分支 | 成功/评测 | 成功率 |",
                "|---|---:|---:|",
            ]
        )
        for method in ("dsrl", "vlm", "joint"):
            row = rows_by_method[method]
            lines.append(
                f"| {row['display_name']} | {row['successes']}/{row['evaluation_episodes']} | "
                f"{100 * float(row['success_rate']):.1f}% |"
            )

    lines.extend(["", "## 在线记录", ""])
    lines.extend(
        [
            "`online_episode_outcomes.csv` 保存每个在线 episode 的成功/失败；",
            "`online_success_curve.csv` 提供逐 episode 累计与 10-episode 滑动成功率。",
        ]
    )
    if summary["vlm_advisor_is_real_vlm"] is False:
        lines.extend(
            [
                "",
                "本次 VLM 分支使用 `advisor=deterministic` 的可复现协议替代器，"
                "`advisor_is_real_vlm=false`，并非外部多模态模型调用；"
                "配置 API key 后可用 `--advisor openai` 原协议重跑。",
            ]
        )
    lines.append("")
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("exp_report/compare_exp"))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    summary = analyze(root)
    print(json.dumps(summary["ranking"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
