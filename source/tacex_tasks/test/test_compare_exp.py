from __future__ import annotations

import gzip
import json
import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
for relative in (
    "scripts/reinforcement_learning/dsrl",
    "scripts/reinforcement_learning/vlm_dsrl",
    "scripts/reinforcement_learning/compare_exp",
    "scripts/bc_training",
    "bc_policy",
    "source/tacex_tasks/tacex_tasks/lab_pick",
):
    path = ROOT / relative
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_results import analyze  # noqa: E402
from flow_rwr import FlowRWRFineTuner  # noqa: E402
from recording import OnlineEpisodeRecorder  # noqa: E402


class _TinyFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity_net = torch.nn.Linear(1, 1)
        self.frozen = torch.nn.Linear(1, 1)

    def extract_dino_features(self, images):
        return images.mean(dim=(-1, -2), keepdim=True)

    def compute_loss(self, batch):
        prediction = self.velocity_net(batch["action"].mean().reshape(1, 1))
        loss = prediction.square().mean()
        return {"loss": loss, "flow_loss": loss, "visual_xy_loss": loss.detach() * 0.0}


class _TinyNormalizer:
    @staticmethod
    def normalize_tensor(key, value):
        assert key == "action"
        return value


class _StateKeyTinyRunner:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cpu")
        self.config = types.SimpleNamespace(chunk_size=2, action_dim=1)
        self.normalizer = _TinyNormalizer()

    @staticmethod
    def build_model_obs():
        return {
            "state": torch.zeros(1, 2, 1),
            "images": torch.ones(1, 1, 1, 1, 1, 1),
            "phase": torch.zeros(1, 1),
        }


def test_flow_rwr_updates_only_velocity_network_after_success():
    model = _TinyFlow()
    runner = types.SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        config=types.SimpleNamespace(chunk_size=2, action_dim=1),
        normalizer=_TinyNormalizer(),
        build_model_obs=lambda: {
            "robot0_pos": torch.zeros(1, 2, 1),
            "images": torch.ones(1, 1, 1, 1, 1, 1),
            "phase": torch.zeros(1, 1),
        },
    )
    tuner = FlowRWRFineTuner(
        runner,
        batch_size=1,
        gradient_steps_per_success=2,
        replay_capacity=4,
    )
    observation = tuner.capture_observation()
    tuner.add_pending_chunk(observation, torch.ones(2, 1), executed_actions=1)
    assert tuner.complete_episode(success=False) == 0
    assert not tuner.replay
    tuner.add_pending_chunk(observation, torch.ones(2, 1), executed_actions=2)
    velocity_before = [value.detach().clone() for value in model.velocity_net.parameters()]
    frozen_before = [value.detach().clone() for value in model.frozen.parameters()]
    assert tuner.complete_episode(success=True) == 2
    assert len(tuner.replay) == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(velocity_before, model.velocity_net.parameters())
    )
    assert all(not parameter.requires_grad for parameter in model.frozen.parameters())
    for before, after in zip(frozen_before, model.frozen.parameters()):
        torch.testing.assert_close(before, after)


def test_flow_rwr_accepts_dinov3_state_key():
    model = _TinyFlow()
    tuner = FlowRWRFineTuner(_StateKeyTinyRunner(model), batch_size=1)
    observation = tuner.capture_observation()
    tuner.add_pending_chunk(observation, torch.ones(2, 1), executed_actions=2)
    assert tuner.complete_episode(success=True) == tuner.gradient_steps_per_success
    assert tuner.state_key == "state"


class _TerminalEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        del action
        return (
            np.zeros(1, dtype=np.float32),
            1.0,
            True,
            False,
            {
                "log": {"LabPick/success_terminal_step": 1.0},
                "tactile_actor": torch.zeros(1, 5),
            },
        )


def test_online_recorder_persists_every_interaction_and_episode(tmp_path):
    env = OnlineEpisodeRecorder(
        _TerminalEnv(), output_dir=tmp_path, mode="residual_rl", experiment_seed=7
    )
    env.reset(seed=31)
    env.step(np.zeros(1, dtype=np.float32))
    result = env.complete_pending_episode(dsrl_updates_completed=4)
    env.close()
    assert result["success"] is True
    assert result["reset_seed"] == 31
    assert len((tmp_path / "interactions.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 1
    trajectories = list((tmp_path / "trajectories").glob("episode_*.jsonl.gz"))
    assert len(trajectories) == 1
    assert result["trajectory"] == str(trajectories[0])
    with gzip.open(trajectories[0], "rt", encoding="utf-8") as stream:
        trajectory = [json.loads(line) for line in stream]
    assert len(trajectory) == 1
    assert trajectory[0]["action"] == [0.0]
    assert trajectory[0]["terminal"] is True


def test_analyzer_writes_online_curve_and_ablation(tmp_path):
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    methods = []
    for method, outcomes in (("dsrl", [True, False]), ("vlm", [False, True]), ("joint", [True, True])):
        online_path = tmp_path / f"{method}_online.json"
        eval_path = tmp_path / f"{method}_eval.json"
        payload = {
            "completed_episodes": 2,
            "successes": sum(outcomes),
            "success_rate": sum(outcomes) / 2,
            "failure_counts": {},
            "results": [
                {"success": value, "terminal_reason": "success" if value else "timeout"}
                for value in outcomes
            ],
        }
        online_path.write_text(json.dumps(payload))
        eval_path.write_text(json.dumps(payload))
        methods.append(
            {"method": method, "online_result": str(online_path), "evaluation_result": str(eval_path)}
        )
    (comparison / "manifest.json").write_text(
        json.dumps(
            {
                "dsrl_outer_interaction_budget": 2,
                "residual_outer_interaction_budget": 64,
                "nominal_training_physics_steps": 128,
                "evaluation_episodes": 2,
                "methods": methods,
            }
        )
    )
    summary = analyze(tmp_path)
    assert len(summary["ablation"]) == 3
    assert (tmp_path / "online_success_curve.csv").is_file()
    assert summary["ranking"][0]["method"] == "joint"
