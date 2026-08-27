from __future__ import annotations

import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from skrl.memories.torch import RandomMemory


ROOT = Path(__file__).resolve().parents[3]
DSRL_ROOT = ROOT / "scripts" / "reinforcement_learning" / "dsrl"
SKRL_ROOT = ROOT / "scripts" / "reinforcement_learning" / "skrl"
BC_POLICY_ROOT = ROOT / "bc_policy"
for path in (DSRL_ROOT, BC_POLICY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import clean_dsrl_wrapper as wrapper_module  # noqa: E402
from clean_dsrl_agent import CleanDSRLSAC  # noqa: E402
from clean_dsrl_sac import (  # noqa: E402
    CLEAN_DSRL_CONTRACT_BUFFER,
    CLEAN_DSRL_CONTRACT_VERSION,
    CleanDSRLLayout,
    build_clean_dsrl_sac_models,
    validate_absolute_dsrl_policy_state,
)
from clean_dsrl_wrapper import (  # noqa: E402
    CleanDSRLLabPickWrapper,
    pack_dsrl_critic_state,
)


def _spaces(layout: CleanDSRLLayout):
    observation_space = gym.spaces.Box(
        -np.inf, np.inf, (layout.policy_dim,), dtype=np.float32
    )
    state_space = gym.spaces.Box(
        -np.inf, np.inf, (layout.state_dim,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        -1.0, 1.0, (layout.action_dim,), dtype=np.float32
    )
    return observation_space, state_space, action_space


def test_layout_repeats_the_last_absolute_noise_step_without_scaling():
    layout = CleanDSRLLayout(
        policy=24,
        flow_horizon=5,
        learned_noise_steps=2,
    )
    actions = torch.tensor(
        [[-1.0, -0.5, 0.0, 0.5, 1.0, -0.8, -0.4, 0.0, 0.4, 0.8,
          0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    )
    expanded = layout.expand_noise(actions)

    assert expanded.shape == (1, 5, 10)
    torch.testing.assert_close(expanded[:, :2], actions.reshape(1, 2, 10))
    for index in range(2, 5):
        torch.testing.assert_close(expanded[:, index], expanded[:, 1])


def test_dsrl_actor_starts_at_zero_mean_and_models_do_not_share_parameters():
    layout = CleanDSRLLayout(policy=24)
    observation_space, state_space, action_space = _spaces(layout)
    models = build_clean_dsrl_sac_models(
        observation_space,
        state_space,
        action_space,
        "cpu",
        layout=layout,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
    )
    torch.manual_seed(11)
    observations = torch.randn(4096, layout.policy_dim)
    states = torch.randn(3, layout.state_dim)
    mean, distribution_outputs = models["policy"].compute(
        {"observations": observations}, role="policy"
    )
    deterministic = models["policy"].deterministic_action(
        {"observations": observations}
    )
    actions, outputs = models["policy"].act(
        {"observations": observations}, role="policy"
    )

    torch.testing.assert_close(mean, torch.zeros_like(mean), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        distribution_outputs["log_std"],
        torch.zeros_like(distribution_outputs["log_std"]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(deterministic, torch.zeros_like(deterministic), rtol=0.0, atol=0.0)
    assert actions.shape == (4096, layout.action_dim)
    assert outputs["log_prob"].shape == (4096, 1)
    assert torch.isfinite(actions).all()
    assert bool((actions.abs() <= 1.0).all())
    assert abs(float(actions.mean())) < 0.02

    critic_observations = observations[:3]
    critic_actions = actions[:3]
    q_values = models["critic_1"].act(
        {
            "observations": critic_observations,
            "states": states,
            "taken_actions": critic_actions,
        },
        role="critic_1",
    )[0]
    assert q_values.shape == (3, 1)
    critic_input = models["critic_1"].build_network_input(
        {
            "observations": critic_observations,
            "states": states,
            "taken_actions": critic_actions,
        }
    )
    torch.testing.assert_close(critic_input[:, -layout.action_dim :], critic_actions)
    parameter_sets = [
        {id(parameter) for parameter in model.parameters()}
        for model in models.values()
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(parameter_sets)
        for right in parameter_sets[index + 1 :]
    )


def test_absolute_v2_checkpoint_marker_rejects_legacy_policy_state():
    layout = CleanDSRLLayout(policy=24)
    observation_space, state_space, action_space = _spaces(layout)
    policy = build_clean_dsrl_sac_models(
        observation_space,
        state_space,
        action_space,
        "cpu",
        layout=layout,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
    )["policy"]
    policy_state = policy.state_dict()

    validate_absolute_dsrl_policy_state(policy_state)
    assert int(policy_state[CLEAN_DSRL_CONTRACT_BUFFER].item()) == CLEAN_DSRL_CONTRACT_VERSION

    legacy_state = dict(policy_state)
    legacy_state.pop(CLEAN_DSRL_CONTRACT_BUFFER)
    with pytest.raises(ValueError, match="Legacy Clean DSRL checkpoint"):
        validate_absolute_dsrl_policy_state(legacy_state)

    wrong_version_state = dict(policy_state)
    wrong_version_state[CLEAN_DSRL_CONTRACT_BUFFER] = torch.tensor(1)
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_absolute_dsrl_policy_state(wrong_version_state)


def test_absolute_dsrl_entrypoints_expose_no_residual_or_scale_switches():
    trainer_source = (SKRL_ROOT / "train_clean_dsrl_sac.py").read_text()
    evaluator_source = (DSRL_ROOT / "eval_lab_pick_clean_dsrl_sac_runtime.py").read_text()

    for source in (trainer_source, evaluator_source):
        assert '"--noise_strategy"' not in source
        assert '"--noise_scale"' not in source
    assert "flow_noise_dsrl_absolute_repeat_last_v2" in trainer_source
    assert "clean_dsrl_sac_absolute_l" in trainer_source


def test_dsrl_actor_accepts_wrapper_inference_tensor_during_evaluation():
    layout = CleanDSRLLayout(policy=24)
    observation_space, state_space, action_space = _spaces(layout)
    actor = build_clean_dsrl_sac_models(
        observation_space,
        state_space,
        action_space,
        "cpu",
        layout=layout,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
    )["policy"]
    with torch.inference_mode():
        wrapper_observation = torch.randn(1, layout.policy_dim)
        action = actor.deterministic_action(
            {"observations": wrapper_observation}
        )
    assert action.shape == (1, layout.action_dim)


def test_pack_dsrl_critic_state_has_the_privileged_state_contract():
    state = pack_dsrl_critic_state(
        torch.zeros(2, 10),
        torch.ones(2, 3),
        torch.full((2, 6), 2.0),
    )
    assert state.shape == (2, 19)
    torch.testing.assert_close(state[:, :10], torch.zeros(2, 10))
    torch.testing.assert_close(state[:, 10:13], torch.ones(2, 3))
    torch.testing.assert_close(state[:, 13:], torch.full((2, 6), 2.0))


class _FakeFlowAdapter:
    n_action_steps = 32
    action_dim = 10
    observation_dim = 24

    def __init__(self):
        self.decode_inputs: list[torch.Tensor | None] = []
        self.reset_calls = 0
        self.update_calls = 0
        self.runner = types.SimpleNamespace(
            generator=torch.Generator().manual_seed(0),
            device=torch.device("cpu"),
        )

    @property
    def is_ready(self) -> bool:
        return True

    def reset(self):
        self.reset_calls += 1

    def update(self, observation):
        del observation
        self.update_calls += 1

    def encode_observation(self):
        return torch.arange(24, dtype=torch.float32).reshape(1, 24)

    def normalize_proprioception(self, value, *, phase):
        del phase
        return value

    def decode(self, flat_noise=None):
        self.decode_inputs.append(
            None if flat_noise is None else flat_noise.detach().clone()
        )
        return torch.arange(320, dtype=torch.float32).reshape(32, 10)


class _FakeLabPickEnv(gym.Env):
    metadata = {}

    def __init__(self, *, truncate_after: int | None = None):
        super().__init__()
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.cfg = types.SimpleNamespace(rl_align_cafe_action_yaw=False)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (10,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32)
        self.single_action_space = self.action_space
        self.single_observation_space = self.observation_space
        self.actions: list[torch.Tensor] = []
        self.truncate_after = truncate_after

    def get_cafe_observation(self):
        return {"robot0_pos": torch.arange(10, dtype=torch.float32).reshape(1, 10)}

    def get_privileged_object_pose(self):
        return torch.zeros(1, 3), torch.zeros(1, 6)

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.actions.clear()
        return {"policy": torch.zeros(1, 16)}, {}

    def step(self, action):
        self.actions.append(action.detach().clone())
        truncated = self.truncate_after is not None and len(self.actions) >= self.truncate_after
        return (
            {"policy": torch.zeros(1, 16)},
            torch.ones(1),
            torch.tensor([False]),
            torch.tensor([truncated]),
            {"log": {}},
        )


def _wrapper(monkeypatch, *, truncate_after=None, **kwargs):
    adapter = _FakeFlowAdapter()
    monkeypatch.setattr(
        wrapper_module.FlowMatchingNoiseAdapter,
        "from_pretrained",
        staticmethod(lambda checkpoint, **adapter_kwargs: adapter),
    )
    wrapper = CleanDSRLLabPickWrapper(
        _FakeLabPickEnv(truncate_after=truncate_after),
        "fake_flow.pt",
        device="cpu",
        camera_warmup_steps=0,
        **kwargs,
    )
    monkeypatch.setattr(wrapper, "_raw_flow_observation", lambda: {})
    return wrapper, adapter


def test_wrapper_expands_noise_and_returns_discounted_chunk_reward(monkeypatch):
    wrapper, adapter = _wrapper(
        monkeypatch,
        chunk_execute_steps=3,
        chunk_discount=0.5,
    )
    observation, _ = wrapper.reset()
    action = torch.linspace(-1.0, 1.0, 10).reshape(1, 10)
    next_observation, reward, terminated, truncated, info = wrapper.step(action)

    assert observation["policy"].shape == next_observation["policy"].shape == (1, 24)
    assert observation["critic"].shape == next_observation["critic"].shape == (1, 19)
    assert len(wrapper.env.actions) == 6
    assert len(adapter.decode_inputs) == 1
    full_noise = adapter.decode_inputs[0].reshape(1, 32, 10)
    for index in range(32):
        torch.testing.assert_close(full_noise[:, index], action)
    torch.testing.assert_close(reward, torch.tensor([3.5]))
    assert not bool(terminated.item())
    assert not bool(truncated.item())
    assert info["clean_dsrl/action_steps_executed"] == 3
    assert info["clean_dsrl/physics_steps_executed"] == 6
    assert wrapper.outer_discount_factor == pytest.approx(0.125)


def test_wrapper_masks_timeout_as_terminal_and_native_bc_uses_native_noise(monkeypatch):
    wrapper, adapter = _wrapper(monkeypatch, truncate_after=1)
    wrapper.reset(flow_noise_seed=17)
    _, reward, terminated, truncated, info = wrapper.step_bc()

    assert adapter.decode_inputs == [None]
    torch.testing.assert_close(reward, torch.ones(1))
    assert bool(terminated.item())
    assert bool(truncated.item())
    assert info["clean_dsrl/action_steps_executed"] == 1
    assert "clean_dsrl/base_noise_seed" not in info
    assert "CleanDSRL/base_noise_seed" not in info["log"]


def test_zero_absolute_noise_decodes_an_all_zero_flow_tensor(monkeypatch):
    wrapper, adapter = _wrapper(
        monkeypatch,
        chunk_execute_steps=1,
    )
    wrapper.reset(flow_noise_seed=23)
    wrapper.step(torch.zeros(1, 10))
    first_noise = adapter.decode_inputs[-1].reshape(1, 32, 10).clone()
    wrapper.step(torch.zeros(1, 10))
    second_noise = adapter.decode_inputs[-1].reshape(1, 32, 10)

    torch.testing.assert_close(first_noise, torch.zeros_like(first_noise), rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_noise, second_noise, rtol=0.0, atol=0.0)
    assert not hasattr(wrapper, "_base_noise_template")


def test_dsrl_agent_updates_actor_critics_and_entropy():
    torch.manual_seed(7)
    layout = CleanDSRLLayout(policy=24)
    observation_space, state_space, action_space = _spaces(layout)
    models = build_clean_dsrl_sac_models(
        observation_space,
        state_space,
        action_space,
        "cpu",
        layout=layout,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
    )
    memory = RandomMemory(memory_size=16, num_envs=1, device="cpu")
    agent = CleanDSRLSAC(
        models=models,
        memory=memory,
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg={
            "gradient_steps": 1,
            "batch_size": 4,
            "discount_factor": 0.9,
            "polyak": 0.05,
            "learning_rate": [1.0e-3, 1.0e-3, 1.0e-3],
            "random_timesteps": 0,
            "learning_starts": 0,
            "learn_entropy": True,
            "initial_entropy_value": 0.01,
            "target_entropy": -5.0,
            "mixed_precision": False,
            "experiment": {"write_interval": 1, "checkpoint_interval": 0},
        },
    )
    agent.init()
    agent.enable_training_mode(True)
    for _ in range(4):
        memory.add_samples(
            observations=torch.randn(1, layout.policy_dim),
            states=torch.randn(1, layout.state_dim),
            actions=torch.zeros(1, layout.action_dim),
            rewards=torch.ones(1, 1),
            next_observations=torch.randn(1, layout.policy_dim),
            next_states=torch.randn(1, layout.state_dim),
            terminated=torch.zeros(1, 1, dtype=torch.bool),
        )
    actor_before = [parameter.detach().clone() for parameter in models["policy"].parameters()]
    alpha_before = agent.log_entropy_coefficient.detach().clone()
    agent.update(timestep=0, timesteps=1)

    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, models["policy"].parameters())
    )
    assert not torch.equal(alpha_before, agent.log_entropy_coefficient)
    assert torch.isfinite(agent.log_entropy_coefficient).all()
    assert "Loss / Entropy loss" in agent.tracking_data
