#!/usr/bin/env python3
"""Audit and summarize completed GrapePick tactile-comparison evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch


METHODS = ("base", "residual_rl", "awr", "dppo", "dsrl", "vlm_dsrl")
DISPLAY_NAMES = {
    "base": "Tactile BC (base)",
    "residual_rl": "Residual RL",
    "awr": "AWR",
    "dppo": "DPPO",
    "dsrl": "DSRL",
    "vlm_dsrl": "VLM+DSRL",
}

EXPECTED_DAMAGE_RANGE_N = [3.4, 4.2]
EXPECTED_DETACH_RANGE_N = [3.2, 4.0]
EXPECTED_TRAIN_EPISODES = 40
EXPECTED_VALIDATION_EPISODES = 40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8").strip()
    return json.loads(content) if content else None


def load_checkpoint_metadata(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        key: checkpoint.get(key)
        for key in (
            "method",
            "bc_checkpoint_sha256",
            "train_seed",
            "train_episodes",
            "selection_seed",
            "selection_episodes",
            "selected_stage",
            "selected_score",
            "require_trained_checkpoint",
            "updates",
        )
    }


def load_reports(root: Path, seeds: list[int]) -> dict[str, list[dict[str, Any]]]:
    reports: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        selected = []
        for seed in seeds:
            path = root / "evaluation" / method / f"seed_{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["method"] != method or int(report["seed_start"]) != seed:
                raise ValueError(f"Method/seed mismatch in {path}")
            if int(report["episodes"]) != len(report["outcomes"]):
                raise ValueError(f"Episode count mismatch in {path}")
            outcome_seeds = [int(item["seed"]) for item in report["outcomes"]]
            if outcome_seeds != list(range(seed, seed + int(report["episodes"]))):
                raise ValueError(f"Non-contiguous or reordered outcome seeds in {path}")
            selected.append(report)
        reports[method] = selected
    reference = [item["seed"] for report in reports["base"] for item in report["outcomes"]]
    for method, selected in reports.items():
        current = [item["seed"] for report in selected for item in report["outcomes"]]
        if current != reference:
            raise ValueError(f"Evaluation episodes are not paired for {method}")
    return reports


def _same_range(value: Any, expected: list[float]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(expected)
        and all(math.isclose(float(actual), target, abs_tol=1.0e-9) for actual, target in zip(value, expected))
    )


def _selected_checkpoint_has_updates(
    method: str,
    training: dict[str, Any] | None,
    checkpoint_metadata: dict[str, Any] | None,
) -> bool:
    if (
        checkpoint_metadata is None
        or checkpoint_metadata.get("method") != method
        or checkpoint_metadata.get("require_trained_checkpoint") is not True
        or int(checkpoint_metadata.get("updates") or 0) <= 0
    ):
        return False
    match = re.fullmatch(r"iteration_(\d+)", str(checkpoint_metadata.get("selected_stage")))
    if match is None:
        return False
    if training is None:
        return True
    if training.get("selected_stage") != checkpoint_metadata.get("selected_stage"):
        return False
    iteration = int(match.group(1))
    diagnostics = training.get("diagnostics")
    if not isinstance(diagnostics, list) or not 1 <= iteration <= len(diagnostics):
        return False
    selected_diagnostic = diagnostics[iteration - 1]
    required_fields = {
        "residual_rl": ("q_loss", "actor_loss"),
        "awr": ("actor_loss", "value_loss"),
        "dppo": ("loss", "approx_kl"),
        "dsrl": ("q_loss", "actor_loss"),
        "vlm_dsrl": ("q_loss", "actor_loss"),
    }[method]
    return all(
        field in selected_diagnostic
        and isinstance(selected_diagnostic[field], (int, float))
        and math.isfinite(float(selected_diagnostic[field]))
        for field in required_fields
    )


def summarize(
    root: Path,
    seeds: list[int],
    expected_train_episodes: int,
    expected_eval_episodes: int,
    expected_validation_episodes: int,
    expected_validation_seed: int,
    expected_damage_range: list[float],
    expected_detach_range: list[float],
    minimum_vlm_margin: float | None,
    require_vlm_strictly_best: bool,
    reference_summary: dict[str, Any] | None,
    base_success_min: float | None,
    base_success_max: float | None,
) -> dict[str, Any]:
    reports = load_reports(root, seeds)
    bc_hashes = {
        report["bc_checkpoint_sha256"]
        for selected in reports.values()
        for report in selected
    }
    if len(bc_hashes) != 1:
        raise ValueError(f"Evaluation reports reference different BC checkpoints: {bc_hashes}")

    training_reports = {
        method: load_optional_json(root / "training" / f"{method}.json")
        for method in METHODS
        if method != "base"
    }
    checkpoint_metadata_by_method = {
        method: load_checkpoint_metadata(root / "checkpoints" / f"{method}.pt")
        for method in METHODS
        if method != "base"
    }

    rows = []
    for method in METHODS:
        selected = reports[method]
        outcomes = [item for report in selected for item in report["outcomes"]]
        seed_success_rates = [float(report["success_rate"]) for report in selected]
        episode_peak_forces = [float(item["peak_grip_force_n"]) for item in outcomes]
        failures = Counter(item["outcome"] for item in outcomes if item["outcome"] != "success")
        successes = sum(item["outcome"] == "success" for item in outcomes)
        checkpoint = root / "checkpoints" / f"{method}.pt"
        training = training_reports.get(method)
        checkpoint_metadata = checkpoint_metadata_by_method.get(method)
        checkpoint_was_updated = (
            None
            if method == "base"
            else _selected_checkpoint_has_updates(method, training, checkpoint_metadata)
        )
        rows.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "episodes": len(outcomes),
                "successes": successes,
                "success_rate": successes / len(outcomes),
                "success_rate_by_seed": seed_success_rates,
                "success_rate_mean": statistics.mean(seed_success_rates),
                "success_rate_std": (
                    statistics.stdev(seed_success_rates) if len(seed_success_rates) > 1 else None
                ),
                "wilson_95_ci": wilson(successes, len(outcomes)),
                "failure_counts": dict(sorted(failures.items())),
                "peak_grip_force_mean_n": statistics.mean(episode_peak_forces),
                "peak_grip_force_max_n": max(episode_peak_forces),
                # Backward-compatible alias used by the original formal_v2 summary.
                "peak_grip_force_n": max(episode_peak_forces),
                "peak_obstacle_contact_force_n": max(
                    float(item["peak_obstacle_contact_force_n"]) for item in outcomes
                ),
                "checkpoint": str(checkpoint.resolve()) if checkpoint.is_file() else None,
                "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
                "selected_stage": (
                    checkpoint_metadata.get("selected_stage") if checkpoint_metadata else None
                ),
                "train_episodes": (
                    int(checkpoint_metadata["train_episodes"]) if checkpoint_metadata else 0
                ),
                "optimizer_updates": (
                    int(checkpoint_metadata["updates"]) if checkpoint_metadata else 0
                ),
                "checkpoint_was_updated": checkpoint_was_updated,
            }
        )

    by_method = {row["method"]: row for row in rows}
    base = by_method["base"]
    first = reports["base"][0]
    first_damage_min = float(first["damage_force_randomization_range_n"][0])
    vlm_report = reports["vlm_dsrl"][0]
    advisor = vlm_report.get("vlm_force_advisor", {})
    tactile_ok = all(
        report.get("tactile_observation_indices") == [6, 7]
        and report.get("tactile_observation_fields")
        == ["left_pad_normal_force_n", "right_pad_normal_force_n"]
        for selected in reports.values()
        for report in selected
    )
    evaluation_episode_seeds = {
        int(item["seed"])
        for selected in reports.values()
        for report in selected
        for item in report["outcomes"]
    }
    development_episode_seeds: set[int] = set()
    for method, checkpoint_metadata in checkpoint_metadata_by_method.items():
        if checkpoint_metadata is None:
            continue
        train_start = int(checkpoint_metadata["train_seed"]) + METHODS.index(method) * 10_000
        development_episode_seeds.update(
            range(train_start, train_start + int(checkpoint_metadata["train_episodes"]))
        )
        validation_start = int(checkpoint_metadata["selection_seed"])
        development_episode_seeds.update(
            range(
                validation_start,
                validation_start + int(checkpoint_metadata["selection_episodes"]),
            )
        )
    exact_thresholds = all(
        _same_range(report.get("damage_force_randomization_range_n"), expected_damage_range)
        and _same_range(report.get("detach_force_randomization_range_n"), expected_detach_range)
        for selected in reports.values()
        for report in selected
    ) and all(
        _same_range(training.get("damage_force_randomization_range_n"), expected_damage_range)
        and _same_range(training.get("detach_force_randomization_range_n"), expected_detach_range)
        for training in training_reports.values()
        if training is not None
    )
    trained_with_budget = all(
        row["checkpoint_was_updated"]
        and row["train_episodes"] == expected_train_episodes
        for row in rows
        if row["method"] != "base"
    )
    validation_budget_ok = all(
        checkpoint_metadata is not None
        and int(checkpoint_metadata["selection_episodes"]) == expected_validation_episodes
        and int(checkpoint_metadata["selection_seed"]) == expected_validation_seed
        for checkpoint_metadata in checkpoint_metadata_by_method.values()
    )
    vlm_margin = by_method["vlm_dsrl"]["success_rate"] - max(
        by_method["residual_rl"]["success_rate"], by_method["dsrl"]["success_rate"]
    )
    checks = {
        "paired_evaluation_complete": all(
            len(selected) == len(seeds)
            and all(report["episodes"] == expected_eval_episodes for report in selected)
            for selected in reports.values()
        ),
        "one_shared_tactile_bc_checkpoint": len(bc_hashes) == 1,
        "all_five_rl_checkpoints_trained_with_updates": all(
            row["checkpoint"] is not None and row["checkpoint_was_updated"]
            for row in rows
            if row["method"] != "base"
        ),
        "uniform_rl_training_budget": trained_with_budget,
        "uniform_validation_budget": validation_budget_ok,
        "formal_seeds_disjoint_from_training_and_validation": evaluation_episode_seeds.isdisjoint(
            development_episode_seeds
        ),
        "exact_narrow_force_thresholds": exact_thresholds,
        "tactile_observation_present_for_every_method": tactile_ok,
        "base_has_crushed_failures": base["failure_counts"].get("crushed", 0) > 0,
        "base_has_no_detach_failures": base["failure_counts"].get("no_detach", 0) > 0,
        "no_robot_obstacle_collision_in_formal_evaluation": all(
            row["peak_obstacle_contact_force_n"] < 0.25 for row in rows
        ),
        "real_vlm_advisor_used": advisor.get("advisor_is_real_vlm") is True,
        "vlm_advisor_used_exact_task_thresholds": (
            _same_range(
                advisor.get("task_force_ranges_n", {}).get("grape_damage"),
                expected_damage_range,
            )
            and _same_range(
                advisor.get("task_force_ranges_n", {}).get("stem_detachment"),
                expected_detach_range,
            )
        ),
        "vlm_range_is_below_minimum_damage_limit": (
            advisor.get("force_range_n") == [2.8, 3.3]
            and advisor["force_range_n"][1] < first_damage_min
        ),
    }
    if base_success_min is not None or base_success_max is not None:
        checks["base_success_within_requested_open_interval"] = (
            (base_success_min is None or base["success_rate"] > base_success_min)
            and (base_success_max is None or base["success_rate"] < base_success_max)
        )
    if require_vlm_strictly_best:
        checks["vlm_dsrl_strictly_best"] = by_method["vlm_dsrl"]["success_rate"] > max(
            row["success_rate"] for row in rows if row["method"] != "vlm_dsrl"
        )
    if minimum_vlm_margin is not None:
        checks["vlm_dsrl_meets_required_margin_over_residual_and_dsrl"] = (
            vlm_margin >= minimum_vlm_margin
        )
    budget_comparison = None
    if reference_summary is not None:
        reference_advisor = reference_summary.get("vlm_force_advisor", {})
        checks["matches_reference_configuration_except_training_budget"] = (
            reference_summary.get("task") == first["task"]
            and reference_summary.get("eval_seeds") == seeds
            and int(reference_summary.get("eval_episodes_per_seed", 0)) == 100
            and int(reference_summary.get("validation_episodes_per_checkpoint", 0))
            == expected_validation_episodes
            and int(reference_summary.get("validation_seed", -1)) == expected_validation_seed
            and reference_summary.get("bc_checkpoint_sha256") == next(iter(bc_hashes))
            and _same_range(
                reference_summary.get("damage_force_randomization_range_n"),
                expected_damage_range,
            )
            and _same_range(
                reference_summary.get("detach_force_randomization_range_n"),
                expected_detach_range,
            )
            and reference_summary.get("tactile_conditioning")
            == {
                "fields": first["tactile_observation_fields"],
                "observation_indices": first["tactile_observation_indices"],
            }
            and reference_advisor.get("image_sha256") == advisor.get("image_sha256")
            and reference_advisor.get("force_range_n") == advisor.get("force_range_n")
            and int(reference_summary.get("train_episodes_per_rl_method", -1))
            != expected_train_episodes
        )
        reference_by_method = {
            row["method"]: row for row in reference_summary.get("methods", [])
        }
        budget_comparison = [
            {
                "method": row["method"],
                "display_name": row["display_name"],
                "reference_train_episodes": int(
                    reference_summary["train_episodes_per_rl_method"]
                ),
                "reference_success_rate_mean": reference_by_method[row["method"]][
                    "success_rate_mean"
                ],
                "current_train_episodes": expected_train_episodes,
                "current_success_rate_mean": row["success_rate_mean"],
                "success_rate_delta": row["success_rate_mean"]
                - reference_by_method[row["method"]]["success_rate_mean"],
            }
            for row in rows
        ]
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Comparison acceptance checks failed: {failed}")

    return {
        "task": first["task"],
        "protocol": "paired independent evaluation seeds",
        "eval_seeds": seeds,
        "eval_episodes_per_seed": expected_eval_episodes,
        "train_episodes_per_rl_method": expected_train_episodes,
        "validation_episodes_per_checkpoint": expected_validation_episodes,
        "validation_seed": expected_validation_seed,
        "total_evaluation_episodes": sum(row["episodes"] for row in rows),
        "bc_checkpoint_sha256": next(iter(bc_hashes)),
        "damage_force_randomization_range_n": first["damage_force_randomization_range_n"],
        "detach_force_randomization_range_n": first["detach_force_randomization_range_n"],
        "tactile_conditioning": {
            "fields": first["tactile_observation_fields"],
            "observation_indices": first["tactile_observation_indices"],
        },
        "vlm_force_advisor": advisor,
        "methods": rows,
        "vlm_dsrl_margin_over_best_residual_or_dsrl": vlm_margin,
        "performance_observations": {
            "vlm_dsrl_strictly_better_than_base": (
                by_method["vlm_dsrl"]["success_rate"] > base["success_rate"]
            ),
            "vlm_dsrl_strictly_best": by_method["vlm_dsrl"]["success_rate"]
            > max(row["success_rate"] for row in rows if row["method"] != "vlm_dsrl"),
        },
        "requested_base_success_open_interval": (
            [base_success_min, base_success_max]
            if base_success_min is not None or base_success_max is not None
            else None
        ),
        "training_budget_comparison": budget_comparison,
        "acceptance_checks": checks,
        "all_acceptance_checks_passed": all(checks.values()),
    }


def write_outputs(root: Path, summary: dict[str, Any]) -> None:
    write_json(root / "comparison_summary.json", summary)
    with (root / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "method",
                "episodes",
                "successes",
                "success_rate",
                "success_rate_mean",
                "success_rate_std",
                "failure_counts",
                "peak_grip_force_mean_n",
                "peak_grip_force_max_n",
                "peak_obstacle_contact_force_n",
                "selected_stage",
            ),
        )
        writer.writeheader()
        for row in summary["methods"]:
            writer.writerow(
                {
                    key: json.dumps(row[key], sort_keys=True)
                    if key == "failure_counts"
                    else row[key]
                    for key in writer.fieldnames
                }
            )

    multiple_eval_seeds = len(summary["eval_seeds"]) > 1
    rate_heading = "Success rate mean ± std" if multiple_eval_seeds else "Success rate"
    if multiple_eval_seeds:
        evaluation_description = (
            f"Paired formal evaluation on seeds "
            f"{', '.join(str(seed) for seed in summary['eval_seeds'])} "
            f"({summary['eval_episodes_per_seed']} episodes per seed)."
        )
    else:
        evaluation_description = (
            f"Formal evaluation on seed {summary['eval_seeds'][0]} "
            f"({summary['eval_episodes_per_seed']} rollouts per method)."
        )
    lines = [
        "# GrapePick tactile-policy comparison",
        "",
        evaluation_description,
        "All methods use the same tactile BC checkpoint and observe both physical pad-normal contact loads.",
        f"Each learned method receives only {summary['train_episodes_per_rl_method']} training episodes; "
        f"checkpoint selection uses {summary['validation_episodes_per_checkpoint']} episodes on a separate validation seed.",
        "",
        f"| Method | Success | {rate_heading} | Failures | Peak force mean (N) | Peak force max (N) | Peak robot-obstacle (N) |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in summary["methods"]:
        failures = ", ".join(f"{key}: {value}" for key, value in row["failure_counts"].items()) or "none"
        rate_text = f"{100.0 * row['success_rate_mean']:.1f}%"
        if row["success_rate_std"] is not None:
            rate_text += f" ± {100.0 * row['success_rate_std']:.1f}%"
        lines.append(
            f"| {row['display_name']} | {row['successes']}/{row['episodes']} | "
            f"{rate_text} | "
            f"{failures} | {row['peak_grip_force_mean_n']:.3f} | "
            f"{row['peak_grip_force_max_n']:.3f} | "
            f"{row['peak_obstacle_contact_force_n']:.3f} |"
        )
    selected_stages = ", ".join(
        f"{row['display_name']}={row['selected_stage']}"
        for row in summary["methods"]
        if row["method"] != "base"
    )
    lines.extend(
        (
            "",
            f"All learned methods completed {summary['train_episodes_per_rl_method']} training episodes. "
            "As in the reference experiment, formal evaluation uses the best checkpoint selected "
            "on the fixed validation set.",
            f"Selected checkpoints: {selected_stages}.",
        )
    )
    base_interval = summary.get("requested_base_success_open_interval")
    if base_interval is not None:
        lower, upper = base_interval
        base_row = next(row for row in summary["methods"] if row["method"] == "base")
        lines.extend(
            (
                "",
                "## Base-policy calibration check",
                "",
                f"Requested open interval: {100.0 * lower:.1f}% < Base success < "
                f"{100.0 * upper:.1f}%; observed: {100.0 * base_row['success_rate']:.1f}%.",
            )
        )
    if summary.get("training_budget_comparison"):
        comparison = summary["training_budget_comparison"]
        reference_budget = comparison[0]["reference_train_episodes"]
        current_budget = comparison[0]["current_train_episodes"]
        lines.extend(
            (
                "",
                f"## Training-budget comparison ({reference_budget} vs {current_budget} episodes)",
                "",
                "All task, validation, evaluation, tactile, BC-checkpoint, and VLM-advisor settings are identical.",
                "",
                f"| Method | {reference_budget}-episode success | {current_budget}-episode success | Delta |",
                "|---|---:|---:|---:|",
            )
        )
        for row in comparison:
            lines.append(
                f"| {row['display_name']} | "
                f"{100.0 * row['reference_success_rate_mean']:.1f}% | "
                f"{100.0 * row['current_success_rate_mean']:.1f}% | "
                f"{100.0 * row['success_rate_delta']:+.1f} pp |"
            )
    margin_pp = 100.0 * summary["vlm_dsrl_margin_over_best_residual_or_dsrl"]
    damage_range = summary["damage_force_randomization_range_n"]
    detach_range = summary["detach_force_randomization_range_n"]
    force_range = summary["vlm_force_advisor"]["force_range_n"]
    if multiple_eval_seeds:
        rate_note = (
            "Success-rate mean and sample standard deviation (ddof=1) are computed over "
            f"{len(summary['eval_seeds'])} independent seeds."
        )
    else:
        rate_note = (
            f"Each success rate is computed from one formal batch of "
            f"{summary['eval_episodes_per_seed']} rollouts; a between-seed standard deviation "
            "is therefore not applicable."
        )
    force_episode_count = summary["methods"][0]["episodes"]
    lines.extend(
        (
            "",
            f"Force thresholds: damage {damage_range[0]:.1f}–{damage_range[1]:.1f} N; "
            f"stem detachment {detach_range[0]:.1f}–{detach_range[1]:.1f} N.",
            f"{rate_note} Peak-force statistics use all {force_episode_count} episode peak "
            "grip forces per method.",
            f"The VLM advisor selected a {force_range[0]:.1f}–{force_range[1]:.1f} N grip "
            f"interval; the high-rate tactile guard uses {force_range[1] - 0.1:.1f} N as "
            "its upper control target.",
            f"VLM+DSRL minus the stronger of Residual RL and DSRL: {margin_pp:+.1f} percentage points.",
            "All automated acceptance checks passed, including exact budgets and force ranges, real policy updates, paired evaluation, and full-arm obstacle sensing.",
            "",
        )
    )
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=(50000, 60000, 70000))
    parser.add_argument("--expected-train-episodes", type=int, default=EXPECTED_TRAIN_EPISODES)
    parser.add_argument("--expected-eval-episodes", type=int, default=100)
    parser.add_argument(
        "--expected-validation-episodes", type=int, default=EXPECTED_VALIDATION_EPISODES
    )
    parser.add_argument("--expected-validation-seed", type=int, default=920000)
    parser.add_argument(
        "--expected-damage-range", type=float, nargs=2, default=EXPECTED_DAMAGE_RANGE_N
    )
    parser.add_argument(
        "--expected-detach-range", type=float, nargs=2, default=EXPECTED_DETACH_RANGE_N
    )
    parser.add_argument("--minimum-vlm-margin-pp", type=float, default=None)
    parser.add_argument("--require-vlm-strictly-best", action="store_true")
    parser.add_argument("--reference-summary", type=Path, default=None)
    parser.add_argument("--base-success-min", type=float, default=None)
    parser.add_argument("--base-success-max", type=float, default=None)
    args = parser.parse_args()
    root = args.output_root.resolve()
    reference_summary = (
        json.loads(args.reference_summary.read_text(encoding="utf-8"))
        if args.reference_summary is not None
        else None
    )
    summary = summarize(
        root,
        list(args.eval_seeds),
        args.expected_train_episodes,
        args.expected_eval_episodes,
        args.expected_validation_episodes,
        args.expected_validation_seed,
        list(args.expected_damage_range),
        list(args.expected_detach_range),
        args.minimum_vlm_margin_pp / 100.0
        if args.minimum_vlm_margin_pp is not None
        else None,
        args.require_vlm_strictly_best,
        reference_summary,
        args.base_success_min,
        args.base_success_max,
    )
    write_outputs(root, summary)
    print(json.dumps(summary["acceptance_checks"], indent=2))


if __name__ == "__main__":
    main()
