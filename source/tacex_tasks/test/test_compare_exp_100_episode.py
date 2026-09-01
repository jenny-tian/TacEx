from __future__ import annotations

import csv
import json
from pathlib import Path

import gymnasium as gym
import torch


ROOT = Path(__file__).resolve().parents[3]

from exp_report.compare_exp_100_episode.analyze_results import analyze  # noqa: E402
from exp_report.compare_exp_100_episode.run_comparison import _complete  # noqa: E402


def _write_result(path: Path, outcomes: list[bool], forces: list[float]) -> None:
    path.parent.mkdir(parents=True)
    results = []
    trajectory_dir = path.parent / "trajectories"
    trajectory_dir.mkdir()
    episode_lines = []
    for index, (success, force) in enumerate(zip(outcomes, forces)):
        trajectory = trajectory_dir / f"episode_{index:03d}.jsonl.gz"
        trajectory.touch()
        result = {
            "episode_index": index,
            "success": success,
            "terminal_reason": "success" if success else "timeout",
            "episode_return": float(success),
            "outer_interactions": index + 1,
            "physics_steps": 2 * (index + 1),
            "peak_contact_force_n": force,
            "trajectory": str(trajectory),
        }
        results.append(result)
        episode_lines.append(json.dumps(result))
    path.write_text(
        json.dumps(
            {
                "completed_episodes": len(results),
                "successes": sum(outcomes),
                "results": results,
            }
        )
    )
    (path.parent / "episodes.jsonl").write_text("\n".join(episode_lines) + "\n")


def test_exact_episode_completion_requires_outcomes_logs_and_trajectories(tmp_path):
    result = tmp_path / "run" / "results.json"
    _write_result(result, [True, False], [1.0, 2.0])
    assert _complete(result, 2)
    (result.parent / "trajectories" / "episode_001.jsonl.gz").unlink()
    assert not _complete(result, 2)


def test_three_repeat_analyzer_includes_peak_force(tmp_path):
    methods = []
    for method in ("base", "flow_ppo"):
        runs = []
        for repeat, seed in enumerate((11, 22, 33), start=1):
            result = tmp_path / "runs" / method / str(repeat) / "results.json"
            outcomes = [method == "flow_ppo", repeat == 3]
            forces = [float(repeat), float(repeat + 1)]
            _write_result(result, outcomes, forces)
            runs.append({"repeat": repeat, "seed": seed, "result": str(result)})
        methods.append({"method": method, "runs": runs})
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "protocol": "exact_complete_episode_budget",
                "episode_budget_per_run": 2,
                "repeats": 3,
                "repeat_seeds": [11, 22, 33],
                "methods": methods,
            }
        )
    )

    summary = analyze(tmp_path)
    assert summary["total_episodes_per_method"] == 6
    assert summary["ranking"][0]["method"] == "flow_ppo"
    assert summary["ranking"][0]["mean_episode_peak_contact_force_n"] == 2.5
    with (tmp_path / "online_episode_outcomes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert {row["repeat"] for row in rows} == {"1", "2", "3"}
    assert "peak_contact_force_n" in rows[0]


def test_flow_ppo_models_respect_latent_noise_contract():
    import sys

    for relative in (
        "scripts/reinforcement_learning/compare_exp",
        "scripts/reinforcement_learning/dsrl",
    ):
        path = str(ROOT / relative)
        if path not in sys.path:
            sys.path.insert(0, path)

    from clean_dsrl_sac import CleanDSRLLayout
    from flow_ppo import build_flow_ppo_models

    layout = CleanDSRLLayout(policy=7)
    models = build_flow_ppo_models(
        gym.spaces.Box(-1.0, 1.0, (7,)),
        gym.spaces.Box(-1.0, 1.0, (19,)),
        gym.spaces.Box(-1.0, 1.0, (10,)),
        "cpu",
        layout=layout,
        actor_hidden_dims=(8, 8),
        value_hidden_dims=(8, 8),
    )
    actions, outputs = models["policy"].act(
        {"observations": torch.zeros(2, 7)}, role="policy"
    )
    values, _ = models["value"].act(
        {"observations": torch.zeros(2, 7), "states": torch.zeros(2, 19)},
        role="value",
    )
    assert actions.shape == (2, 10)
    assert outputs["log_prob"].shape == (2, 1)
    assert values.shape == (2, 1)
    assert bool((actions.abs() <= 1.0).all())
