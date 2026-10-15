from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def load_learning():
    path = Path(__file__).resolve().parents[3] / "scripts/reinforcement_learning/grape_pick/learning.py"
    spec = importlib.util.spec_from_file_location("grape_evaluation_learning", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("class_name", ["AWRAgent", "DiffusionPPOAgent"])
def test_saved_policy_restores_deterministic_actions_and_update_count(class_name):
    agent_class = getattr(load_learning(), class_name)
    original = agent_class(9, 2, torch.device("cpu"))
    network = original.actor if class_name == "AWRAgent" else original.policy
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.uniform_(-0.1, 0.1)
    original.updates = 17
    observations = torch.randn(5, 9)
    restored = agent_class(9, 2, torch.device("cpu"))
    restored.load_checkpoint(original.checkpoint())
    before = original.act(observations, deterministic=True)
    after = restored.act(observations, deterministic=True)
    if isinstance(before, tuple):
        for expected, actual in zip(before, after):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert restored.updates == 17
    invalid = {**original.checkpoint(), "algorithm": "wrong"}
    with pytest.raises(ValueError, match="algorithm mismatch"):
        restored.load_checkpoint(invalid)


def test_dppo_rejects_different_denoising_chain():
    agent = load_learning().DiffusionPPOAgent(9, 2, torch.device("cpu"))
    invalid = {**agent.checkpoint(), "denoising_steps": 99}
    with pytest.raises(ValueError, match="denoising-step mismatch"):
        agent.load_checkpoint(invalid)
