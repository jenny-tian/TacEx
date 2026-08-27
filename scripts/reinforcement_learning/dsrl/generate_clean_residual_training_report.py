#!/usr/bin/env python3
"""Generate plots and consolidated metrics for fixed-noise residual-SAC runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rolling_episodes", type=int, default=10)
    return parser.parse_args()


def _load_evaluations(eval_dir: Path) -> list[dict]:
    evaluations = [
        json.loads(path.read_text()) for path in sorted(eval_dir.glob("eval_step_*.json"))
    ]
    evaluations.sort(key=lambda item: int(item["training_step"]))
    if not evaluations or int(evaluations[0]["training_step"]) != 0:
        raise ValueError("Evaluation data must include the zero-residual step-0 baseline.")
    trial_counts = {int(item["num_trials"]) for item in evaluations}
    seeds = {int(item["seed"]) for item in evaluations}
    if len(trial_counts) != 1 or len(seeds) != 1:
        raise ValueError("All evaluation points must use the same trial count and seed base.")
    return evaluations


def _save_success_curve(evaluations: list[dict], output_dir: Path) -> None:
    steps = np.asarray([int(item["training_step"]) for item in evaluations])
    rates = np.asarray([float(item["success_rate"]) * 100.0 for item in evaluations])
    successes = [int(item["successes"]) for item in evaluations]
    trials = [int(item["num_trials"]) for item in evaluations]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(steps / 1000.0, rates, color="#1769aa", marker="o", linewidth=2.2)
    for x, y, success, trial_count in zip(steps / 1000.0, rates, successes, trials):
        ax.annotate(
            f"{success}/{trial_count}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("Training steps (k)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Fixed-noise chunk-10 checkpoint evaluation")
    ax.set_ylim(-3, 68)
    ax.set_xticks(steps / 1000.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "success_rate_curve.png", dpi=180)
    plt.close(fig)


def _save_failure_comparison(evaluations: list[dict], output_dir: Path) -> None:
    before = evaluations[0]
    after = evaluations[-1]
    reasons = ("object_broken", "time_limit", "object_dropped", "object_too_far")
    labels = ("Broken", "Timeout", "Dropped", "Too far")
    before_counts = [int(before["failure_counts"].get(reason, 0)) for reason in reasons]
    after_counts = [int(after["failure_counts"].get(reason, 0)) for reason in reasons]
    x = np.arange(len(reasons))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, before_counts, width, label="Before training", color="#6baed6")
    ax.bar(x + width / 2, after_counts, width, label="After 50k", color="#ef8a62")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Trials")
    ax.set_title("Failure reasons on the same 20 seeds")
    ax.set_ylim(0, max(before_counts + after_counts) + 2)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "failure_reason_comparison.png", dpi=180)
    plt.close(fig)


def _save_reward_curve(
    run_dir: Path,
    output_dir: Path,
    *,
    rolling_episodes: int,
) -> dict[str, float | int]:
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tag = "Reward / Total reward (mean)"
    if tag not in accumulator.Tags()["scalars"]:
        raise KeyError(f"TensorBoard run has no scalar tag {tag!r}.")
    events = accumulator.Scalars(tag)
    steps = np.asarray([event.step for event in events], dtype=np.float64)
    rewards = np.asarray([event.value for event in events], dtype=np.float64)
    if rewards.size < rolling_episodes:
        raise ValueError("Not enough completed episodes for the requested rolling window.")
    kernel = np.ones(rolling_episodes, dtype=np.float64) / rolling_episodes
    rolling = np.convolve(rewards, kernel, mode="valid")
    rolling_steps = steps[rolling_episodes - 1 :]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(steps / 1000.0, rewards, color="#9ecae1", linewidth=1.0, alpha=0.7, label="Episode return")
    ax.plot(
        rolling_steps / 1000.0,
        rolling,
        color="#d62728",
        linewidth=2.2,
        label=f"Rolling mean ({rolling_episodes} episodes)",
    )
    ax.axvline(5.0, color="#555555", linestyle="--", linewidth=1.2, label="Learning starts")
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_xlabel("Training steps (k)")
    ax.set_ylabel("Total episode reward")
    ax.set_title("Training total reward")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "total_reward_curve.png", dpi=180)
    plt.close(fig)
    return {
        "episode_count": int(rewards.size),
        "reward_mean": float(rewards.mean()),
        "reward_median": float(np.median(rewards)),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
    }


def main() -> None:
    args = _parse_args()
    if args.rolling_episodes < 1:
        raise ValueError("--rolling_episodes must be positive.")
    run_dir = args.run_dir.expanduser().resolve()
    eval_dir = args.eval_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = _load_evaluations(eval_dir)

    _save_success_curve(evaluations, output_dir)
    _save_failure_comparison(evaluations, output_dir)
    reward_summary = _save_reward_curve(
        run_dir,
        output_dir,
        rolling_episodes=args.rolling_episodes,
    )
    metrics = {
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "evaluations": [
            {
                "training_step": item["training_step"],
                "successes": item["successes"],
                "num_trials": item["num_trials"],
                "success_rate": item["success_rate"],
                "failure_counts": item["failure_counts"],
                "mean_episode_reward": item["mean_episode_reward"],
                "mean_residual_rms": item["mean_residual_rms"],
            }
            for item in evaluations
        ],
        "training_reward": reward_summary,
    }
    (output_dir / "report_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[DONE] report assets written to {output_dir}")


if __name__ == "__main__":
    main()
