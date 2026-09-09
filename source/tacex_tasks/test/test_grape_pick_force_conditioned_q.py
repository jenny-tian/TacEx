from pathlib import Path
import importlib.util
import math

import torch


ROOT = Path(__file__).resolve().parents[3]
LEARNING = ROOT / "scripts" / "reinforcement_learning" / "grape_pick" / "learning.py"


def load_learning_module():
    spec = importlib.util.spec_from_file_location("grape_pick_learning", LEARNING)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_force_conditioned_sac_uses_context_and_effective_action_mask():
    learning = load_learning_module()
    agent = learning.SACResidualAgent(
        observation_dim=4,
        action_dim=3,
        device=torch.device("cpu"),
        critic_action_context_dim=4,
        executed_action_mask=(1.0, 0.0, 1.0),
    )
    actor_action = torch.tensor([[0.2, 0.9, -0.1]])
    context = torch.tensor([[0.56, 0.64, 0.1, 0.2]])
    encoded = agent._critic_action(actor_action, context)
    torch.testing.assert_close(
        encoded,
        torch.tensor([[0.2, 0.0, -0.1, 0.56, 0.64, 0.1, 0.2]]),
    )

    batch = learning.SACBatch(
        observation=torch.randn(16, 4),
        action=torch.tanh(torch.randn(16, 3)),
        reward=torch.randn(16),
        next_observation=torch.randn(16, 4),
        done=torch.zeros(16),
        action_context=torch.randn(16, 4),
        next_action_context=torch.randn(16, 4),
    )
    metrics = agent.update(batch)
    assert agent.updates == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_force_conditioned_replay_round_trips_context():
    learning = load_learning_module()
    replay = learning.ReplayBuffer(
        capacity=16,
        observation_dim=2,
        action_dim=1,
        device=torch.device("cpu"),
        action_context_dim=4,
    )
    context = torch.tensor([[0.56, 0.64, 0.0, 0.0]]).repeat(4, 1)
    replay.add(
        learning.SACBatch(
            observation=torch.randn(4, 2),
            action=torch.randn(4, 1),
            reward=torch.randn(4),
            next_observation=torch.randn(4, 2),
            done=torch.zeros(4),
            action_context=context,
            next_action_context=context + 0.01,
        )
    )
    sampled = replay.sample(8)
    assert sampled.action_context is not None
    assert sampled.next_action_context is not None
    assert sampled.action_context.shape == (8, 4)
    assert sampled.next_action_context.shape == (8, 4)

    restored = learning.ReplayBuffer(
        capacity=16,
        observation_dim=2,
        action_dim=1,
        device=torch.device("cpu"),
        action_context_dim=4,
    )
    restored.load_state_dict(replay.state_dict())
    assert restored.size == replay.size
    assert restored.position == replay.position
    torch.testing.assert_close(restored.observation[:4], replay.observation[:4])
    torch.testing.assert_close(restored.action[:4], replay.action[:4])
    assert restored.action_context is not None and replay.action_context is not None
    torch.testing.assert_close(restored.action_context[:4], replay.action_context[:4])


def test_replay_supports_delayed_terminal_credit():
    learning = load_learning_module()
    replay = learning.ReplayBuffer(
        capacity=4,
        observation_dim=2,
        action_dim=1,
        device=torch.device("cpu"),
    )
    index = replay.add(
        learning.SACBatch(
            observation=torch.zeros(1, 2),
            action=torch.zeros(1, 1),
            reward=torch.tensor([0.25]),
            next_observation=torch.zeros(1, 2),
            done=torch.ones(1),
        )
    )

    replay.add_reward(index, 1.0)

    assert index == 0
    torch.testing.assert_close(replay.reward[0], torch.tensor(1.25))


def test_sac_checkpoint_restores_training_and_optimizer_state():
    learning = load_learning_module()
    torch.manual_seed(55)
    agent = learning.SACResidualAgent(
        observation_dim=4,
        action_dim=2,
        device=torch.device("cpu"),
        critic_action_context_dim=2,
    )
    batch = learning.SACBatch(
        observation=torch.randn(8, 4),
        action=torch.tanh(torch.randn(8, 2)),
        reward=torch.randn(8),
        next_observation=torch.randn(8, 4),
        done=torch.zeros(8),
        action_context=torch.randn(8, 2),
        next_action_context=torch.randn(8, 2),
    )
    agent.update(batch)
    checkpoint = agent.checkpoint()

    restored = learning.SACResidualAgent(
        observation_dim=4,
        action_dim=2,
        device=torch.device("cpu"),
        critic_action_context_dim=2,
    )
    restored.load_checkpoint(checkpoint)
    assert restored.updates == agent.updates
    torch.testing.assert_close(restored.log_alpha, agent.log_alpha)
    observation = torch.randn(3, 4)
    action = torch.randn(3, 2)
    context = torch.randn(3, 2)
    critic_action = torch.cat((action, context), dim=-1)
    torch.testing.assert_close(
        restored.q1(observation, critic_action),
        agent.q1(observation, critic_action),
    )
    assert restored.actor_optimizer.state_dict()["state"]
    assert restored.q_optimizer.state_dict()["state"]
    assert restored.alpha_optimizer.state_dict()["state"]


def test_force_context_adapter_preserves_initial_dsrl_actor_and_q():
    learning = load_learning_module()
    torch.manual_seed(123)
    dsrl = learning.SACResidualAgent(
        observation_dim=4,
        action_dim=3,
        device=torch.device("cpu"),
    )
    torch.manual_seed(123)
    vlm_dsrl = learning.SACResidualAgent(
        observation_dim=4,
        action_dim=3,
        device=torch.device("cpu"),
        critic_action_context_dim=2,
        critic_context_mode="bias_adapter",
    )

    for left, right in zip(dsrl.actor.parameters(), vlm_dsrl.actor.parameters()):
        torch.testing.assert_close(left, right)
    for left, right in zip(dsrl.q1.network.parameters(), vlm_dsrl.q1.network.parameters()):
        torch.testing.assert_close(left, right)

    observation = torch.randn(8, 4)
    action = torch.randn(8, 3)
    context = torch.randn(8, 2)
    torch.testing.assert_close(
        dsrl.q1(observation, action),
        vlm_dsrl.q1(observation, torch.cat((action, context), dim=-1)),
    )


def test_sac_exploration_schedule_has_exact_endpoints_and_geometric_decay():
    learning = load_learning_module()
    values = [
        learning.exploration_scale_schedule(1.0, 0.05, step, 5, "exponential")
        for step in range(5)
    ]
    assert values[0] == 1.0
    assert math.isclose(values[-1], 0.05, rel_tol=1.0e-12)
    ratios = [right / left for left, right in zip(values, values[1:])]
    assert max(ratios) - min(ratios) < 1.0e-12
    assert all(left > right for left, right in zip(values, values[1:]))


def test_sac_exploration_schedule_preserves_legacy_constant_behavior():
    learning = load_learning_module()
    values = [
        learning.exploration_scale_schedule(0.7, 0.1, step, 5, "constant")
        for step in range(5)
    ]
    assert values == [0.7] * 5
