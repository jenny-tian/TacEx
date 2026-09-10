#!/usr/bin/env python3
"""Audit, summarize and plot the complete raspberry_0909 paired experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_ROOT = ROOT / "exp_report/raspberry_0909"
METHODS = ("dsrl_only", "vlm_two_stage_dsrl")
SEEDS = (970_000, 980_000, 990_000)
REFERENCE = (2.4, 3.0)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty source data: {path}")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def rolling_success(records: list[dict[str, Any]], window: int = 10) -> np.ndarray:
    values = np.asarray([float(row["success"]) for row in records], dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def slope_percentage_points_per_10(records: list[dict[str, Any]]) -> float:
    values = rolling_success(records)
    episodes = np.arange(10, len(records) + 1, dtype=float)
    return float(np.polyfit(episodes, 100.0 * values, 1)[0] * 10.0)


def longest_true_streak(values: Iterable[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def export_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def validate_run(root: Path, method: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = read_jsonl(root / "training_episodes.jsonl")
    summary = read_json(root / "training_summary.json")
    if len(records) != 100 or int(summary.get("completed_episodes", -1)) != 100:
        raise ValueError(f"incomplete formal run: {root}")
    if [row["episode_index"] for row in records] != list(range(100)):
        raise ValueError(f"non-contiguous episode indices: {root}")
    if [row["seed"] for row in records] != list(range(seed, seed + 100)):
        raise ValueError(f"non-contiguous episode seeds: {root}")
    manifest = read_json(root / "protocol_manifest.json")
    expected_source = "fixed" if method == "dsrl_only" else "vlm"
    expected_controller = "single_stage" if method == "dsrl_only" else "two_stage"
    if manifest["force_range_source"] != expected_source:
        raise ValueError(f"force source mismatch: {root}")
    if manifest["controller_mode"] != expected_controller:
        raise ValueError(f"controller mismatch: {root}")
    if not manifest["enhanced_determinism"]:
        raise ValueError(f"enhanced determinism disabled: {root}")
    if method == "dsrl_only":
        if int(summary["vlm_calls"]) != 0:
            raise ValueError(f"baseline called VLM: {root}")
        if any(row["attempted_force_range_n"] != [2.2, 2.8] for row in records):
            raise ValueError(f"baseline force range changed: {root}")
    else:
        audits = read_jsonl(root / "codex_vlm_calls.jsonl")
        for audit in audits:
            request_text = json.dumps(audit.get("request", {}), sort_keys=True)
            for forbidden in (
                "damage_threshold_n",
                "detach_threshold_n",
                "oracle_audit_reference_n",
            ):
                if forbidden in request_text:
                    raise ValueError(f"hidden truth leaked into VLM request: {root}")
    if any(float(row["max_abs_lateral_residual_m"]) > 0.0060001 for row in records):
        raise ValueError(f"lateral residual bound violated: {root}")
    return records, summary


def plot_success(all_runs: dict[tuple[str, int], list[dict[str, Any]]], output: Path) -> None:
    # Figure contract: paired online learning curves show whether the composite
    # method improves the preregistered last-20 outcome across training seeds.
    fig, ax = plt.subplots(figsize=(7.09, 3.15), constrained_layout=True)
    colors = {"dsrl_only": "#777777", "vlm_two_stage_dsrl": "#2F6EA5"}
    labels = {"dsrl_only": "DSRL-only", "vlm_two_stage_dsrl": "VLM + two-stage + DSRL"}
    x = np.arange(10, 101)
    for method in METHODS:
        curves = np.stack([rolling_success(all_runs[(method, seed)]) for seed in SEEDS])
        for curve in curves:
            ax.plot(x, 100.0 * curve, color=colors[method], alpha=0.18, linewidth=0.8)
        average = 100.0 * curves.mean(axis=0)
        spread = 100.0 * curves.std(axis=0, ddof=1)
        ax.fill_between(x, average - spread, average + spread, color=colors[method], alpha=0.14)
        ax.plot(x, average, color=colors[method], linewidth=1.8, label=labels[method])
    ax.axvspan(81, 100, color="#D9A441", alpha=0.09, linewidth=0)
    ax.text(90.5, 3, "Primary\nlast-20", ha="center", va="bottom", color="#8A6418")
    ax.set(xlabel="Online episode", ylabel="Success rate (window = 10, %)", xlim=(10, 100), ylim=(0, 103))
    ax.legend(loc="lower left")
    export_figure(fig, output / "online_sr_window10")
    plt.close(fig)


def plot_force_intervals(
    audit_rows: list[dict[str, Any]], output: Path
) -> None:
    # Figure contract: raw proposals are visually distinct from guarded applied
    # intervals so correctness cannot be attributed to the guard by accident.
    fig, axes = plt.subplots(3, 1, figsize=(7.09, 5.7), sharex=True, constrained_layout=True)
    for ax, seed in zip(axes, SEEDS):
        rows = [row for row in audit_rows if int(row["training_seed"]) == seed]
        x = np.asarray([int(row["episode"]) for row in rows])
        raw_low = np.asarray([float(row["raw_low_n"]) for row in rows])
        raw_high = np.asarray([float(row["raw_high_n"]) for row in rows])
        applied_low = np.asarray([float(row["applied_low_n"]) for row in rows])
        applied_high = np.asarray([float(row["applied_high_n"]) for row in rows])
        ax.axhspan(REFERENCE[0] - 0.1, REFERENCE[0] + 0.1, color="#78A878", alpha=0.12)
        ax.axhspan(REFERENCE[1] - 0.1, REFERENCE[1] + 0.1, color="#78A878", alpha=0.12)
        ax.axhline(REFERENCE[0], color="#487A48", linewidth=0.8, linestyle=":")
        ax.axhline(REFERENCE[1], color="#487A48", linewidth=0.8, linestyle=":")
        ax.plot(x, raw_low, color="#A7A7A7", linewidth=0.8, label="Raw endpoints" if seed == SEEDS[0] else None)
        ax.plot(x, raw_high, color="#A7A7A7", linewidth=0.8)
        ax.plot(x, applied_low, color="#2F6EA5", linewidth=1.4, label="Applied endpoints" if seed == SEEDS[0] else None)
        ax.plot(x, applied_high, color="#2F6EA5", linewidth=1.4)
        ax.set_ylabel(f"Seed {seed}\nForce (N)")
        ax.set_ylim(1.15, 3.2)
    axes[0].legend(loc="lower right", ncol=2)
    axes[-1].set_xlabel("Completed episode")
    export_figure(fig, output / "force_interval_vs_reference")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()
    report_root = args.report_root.resolve()
    output = report_root / "comparison"
    output.mkdir(parents=True, exist_ok=True)

    all_runs: dict[tuple[str, int], list[dict[str, Any]]] = {}
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            run_root = report_root / method / f"seed{seed}_n100"
            records, summary = validate_run(run_root, method, seed)
            all_runs[(method, seed)] = records
            summaries[(method, seed)] = summary
            for row in records:
                source_rows.append(
                    {
                        "method": method,
                        "training_seed": seed,
                        "episode": int(row["episode_index"]) + 1,
                        "episode_seed": row["seed"],
                        "success": int(row["success"]),
                        "outcome": row["outcome"],
                        "attempted_low_n": row["attempted_force_range_n"][0],
                        "attempted_high_n": row["attempted_force_range_n"][1],
                        "next_low_n": row["next_force_range_n"][0],
                        "next_high_n": row["next_force_range_n"][1],
                        "bilateral_contact_fraction": row["bilateral_contact_fraction"],
                        "peak_grip_force_n": row["peak_grip_force_n"],
                        "peak_pedicle_tension_n": row["peak_pedicle_tension_n"],
                        "replay_size": row["replay_size"],
                        "sac_updates": row["sac_updates"],
                        "mean_abs_lateral_residual_m": row["mean_abs_lateral_residual_m"],
                        "max_abs_lateral_residual_m": row["max_abs_lateral_residual_m"],
                    }
                )
            success = [int(row["success"]) for row in records]
            failures = {name: sum(row["outcome"] == name for row in records) for name in (
                "no_contact", "no_detach_with_contact", "slipped", "crushed"
            )}
            seed_rows.append(
                {
                    "method": method,
                    "training_seed": seed,
                    "overall_success_rate": mean(success),
                    "first20_success_rate": mean(success[:20]),
                    "last20_success_rate": mean(success[-20:]),
                    "last20_minus_first20": mean(success[-20:]) - mean(success[:20]),
                    "final_window10_success_rate": mean(success[-10:]),
                    "window10_slope_pp_per_10_episodes": slope_percentage_points_per_10(records),
                    **failures,
                    "replay_transitions": summary["replay_transitions"],
                    "sac_updates": summary["sac_optimizer_updates"],
                }
            )

    for seed in SEEDS:
        records = all_runs[("vlm_two_stage_dsrl", seed)]
        for row in records:
            raw = row["vlm_response"]["force_range_n"]
            applied = row["next_force_range_n"]
            raw_correct = bool(row["vlm_call_succeeded"]) and all(
                abs(float(value) - reference) <= 0.1 + 1e-9
                for value, reference in zip(raw, REFERENCE)
            )
            applied_correct = all(
                abs(float(value) - reference) <= 0.1 + 1e-9
                for value, reference in zip(applied, REFERENCE)
            )
            audit_rows.append(
                {
                    "training_seed": seed,
                    "episode": int(row["episode_index"]) + 1,
                    "vlm_call_succeeded": int(row["vlm_call_succeeded"]),
                    "raw_low_n": raw[0],
                    "raw_high_n": raw[1],
                    "applied_low_n": applied[0],
                    "applied_high_n": applied[1],
                    "reference_low_n": REFERENCE[0],
                    "reference_high_n": REFERENCE[1],
                    "raw_endpoint_mae_n": (abs(raw[0] - REFERENCE[0]) + abs(raw[1] - REFERENCE[1])) / 2,
                    "applied_endpoint_mae_n": (abs(applied[0] - REFERENCE[0]) + abs(applied[1] - REFERENCE[1])) / 2,
                    "raw_correct": int(raw_correct),
                    "applied_correct": int(applied_correct),
                    "guard_decision": row["guard_decision"],
                }
            )

    per_seed = {(row["method"], row["training_seed"]): row for row in seed_rows}
    paired = [
        per_seed[("vlm_two_stage_dsrl", seed)]["last20_success_rate"]
        - per_seed[("dsrl_only", seed)]["last20_success_rate"]
        for seed in SEEDS
    ]
    crushed_a = sum(per_seed[("dsrl_only", seed)]["crushed"] for seed in SEEDS) / 300
    crushed_b = sum(per_seed[("vlm_two_stage_dsrl", seed)]["crushed"] for seed in SEEDS) / 300
    vlm_by_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        rows = [row for row in audit_rows if row["training_seed"] == seed]
        post20 = rows[20:]
        correct_episodes = [row["episode"] for row in rows if row["raw_correct"]]
        vlm_by_seed[seed] = {
            "successful_calls": sum(row["vlm_call_succeeded"] for row in rows),
            "raw_correct_fraction_episodes_21_100": mean(row["raw_correct"] for row in post20),
            "applied_correct_fraction_episodes_21_100": mean(row["applied_correct"] for row in post20),
            "first_raw_correct_episode": min(correct_episodes) if correct_episodes else None,
            "longest_raw_correct_streak": longest_true_streak(bool(row["raw_correct"]) for row in rows),
            "final_raw_correct": bool(rows[-1]["raw_correct"]),
        }
    performance_gate = (
        sum(value > 0 for value in paired) >= 2
        and mean(paired) >= 0.10 - 1e-12
        and crushed_b <= crushed_a + 1e-12
    )
    correctness_gate = (
        all(value["successful_calls"] >= 95 for value in vlm_by_seed.values())
        and mean(value["raw_correct_fraction_episodes_21_100"] for value in vlm_by_seed.values()) >= 0.80
        and all(value["final_raw_correct"] for value in vlm_by_seed.values())
    )
    metrics = {
        "independent_unit": "training_seed",
        "n_paired_seeds": 3,
        "paired_last20_difference": dict(zip(map(str, SEEDS), paired)),
        "paired_last20_difference_mean": mean(paired),
        "paired_last20_difference_sample_sd": stdev(paired),
        "positive_paired_differences": sum(value > 0 for value in paired),
        "dsrl_only_crushed_rate": crushed_a,
        "vlm_two_stage_dsrl_crushed_rate": crushed_b,
        "performance_gate_passed": performance_gate,
        "direction_consistent_3_of_3": all(value > 0 for value in paired),
        "vlm_correctness_by_seed": vlm_by_seed,
        "vlm_correctness_gate_passed": correctness_gate,
    }
    write_csv(output / "per_episode_source.csv", source_rows)
    write_csv(output / "per_seed_metrics.csv", seed_rows)
    write_csv(output / "vlm_force_interval_audit.csv", audit_rows)
    (output / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_success(all_runs, output)
    plot_force_intervals(audit_rows, output)

    verdict = (
        "组合方法通过预注册性能门槛。"
        if performance_gate
        else "组合方法未通过预注册性能门槛。"
    )
    vlm_verdict = (
        "VLM 区间推理通过预注册正确性门槛。"
        if correctness_gate
        else "VLM 区间推理未通过预注册正确性门槛；不能把组合收益单独归因于 VLM。"
    )
    results = [
        "# RaspberryPick 0909 实验结果",
        "",
        verdict,
        vlm_verdict,
        "",
        "| Seed | DSRL-only last-20 | VLM+two-stage+DSRL last-20 | 配对差 |",
        "|---:|---:|---:|---:|",
    ]
    for seed, difference in zip(SEEDS, paired):
        a = per_seed[("dsrl_only", seed)]["last20_success_rate"]
        b = per_seed[("vlm_two_stage_dsrl", seed)]["last20_success_rate"]
        results.append(f"| {seed} | {a:.1%} | {b:.1%} | {difference:+.1%} |")
    results.extend(
        (
            "",
            f"配对差均值 ± sample SD：{mean(paired):+.1%} ± {stdev(paired):.1%}。",
            f"Crushed rate：DSRL-only {crushed_a:.1%}；组合方法 {crushed_b:.1%}。",
            "",
            "解释边界：这是 VLM、two-stage controller 与 DSRL 的组合栈比较；两臂结果不能拆分各组件的独立因果贡献。",
        )
    )
    (output / "RESULTS.md").write_text("\n".join(results) + "\n", encoding="utf-8")
    qa = {
        "all_six_runs_complete": True,
        "episode_rows": len(source_rows),
        "seed_rows": len(seed_rows),
        "vlm_audit_rows": len(audit_rows),
        "hidden_truth_request_audit_passed": True,
        "residual_bound_audit_passed": True,
        "statistics_unit": "three paired training seeds; no pooled-episode significance test",
        "figure_contract": {
            "archetype": "quantitative grid",
            "success_figure_claim": "paired online curves and last-20 region expose the composite-method effect without pooling seeds",
            "force_figure_claim": "raw and applied interval endpoints remain distinct from the posthoc reference",
            "exports": ["svg", "pdf", "png_300dpi", "tiff_600dpi_lzw"],
        },
    }
    (output / "QA.md").write_text(
        "# QA\n\n```json\n" + json.dumps(qa, indent=2, ensure_ascii=False) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
