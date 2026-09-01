from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model


ROOT = Path(__file__).resolve().parents[3]
DSRL_ROOT = ROOT / "scripts" / "reinforcement_learning" / "dsrl"
BC_POLICY_ROOT = ROOT / "bc_policy"
for path in (DSRL_ROOT, BC_POLICY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import clean_residual_wrapper as clean_wrapper  # noqa: E402
from clean_alpha_zero_sac import CleanAlphaZeroSAC  # noqa: E402
from clean_residual_sac import (  # noqa: E402
    CLEAN_RESIDUAL_CONTRACT_BUFFER,
    CLEAN_RESIDUAL_CONTRACT_VERSION,
    CleanResidualActor,
    CleanResidualCritic,
    CleanResidualLayout,
    build_clean_residual_sac_models,
    validate_tactile_residual_policy_state,
)
from clean_residual_wrapper import (  # noqa: E402
    CleanResidualLabPickWrapper,
    compose_normalized_action,
    pack_clean_residual_observation,
)


def _spaces(layout: CleanResidualLayout | None = None):
    layout = CleanResidualLayout() if layout is None else layout
    observation_space = gym.spaces.Box(
        -np.inf,
        np.inf,
        (layout.policy_dim,),
        dtype=np.float32,
    )
    state_space = gym.spaces.Box(
        -np.inf,
        np.inf,
        (layout.state_dim,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(
        -1.0,
        1.0,
        (layout.action_dim,),
        dtype=np.float32,
    )
    return observation_space, state_space, action_space


def _alpha_zero_cfg(
    *,
    batch_size: int = 2,
    policy_lr: float = 1.0e-2,
    critic_lr: float = 1.0e-2,
    polyak: float = 0.05,
) -> dict:
    return {
        "gradient_steps": 1,
        "batch_size": batch_size,
        "discount_factor": 0.5,
        "polyak": polyak,
        "learning_rate": [policy_lr, critic_lr, 0.0],
        "random_timesteps": 0,
        "learning_starts": 0,
        "learn_entropy": False,
        "initial_entropy_value": 0.0,
        "mixed_precision": False,
        "experiment": {"write_interval": 1, "checkpoint_interval": 0},
    }


def _new_agent(
    *,
    models=None,
    batch_size: int = 2,
    policy_lr: float = 1.0e-2,
    critic_lr: float = 1.0e-2,
    polyak: float = 0.05,
):
    layout = CleanResidualLayout()
    observation_space, state_space, action_space = _spaces(layout)
    if models is None:
        models = build_clean_residual_sac_models(
            observation_space,
            state_space,
            action_space,
            "cpu",
            layout=layout,
        )
    memory = RandomMemory(memory_size=32, num_envs=1, device="cpu")
    agent = CleanAlphaZeroSAC(
        models=models,
        memory=memory,
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg=_alpha_zero_cfg(
            batch_size=batch_size,
            policy_lr=policy_lr,
            critic_lr=critic_lr,
            polyak=polyak,
        ),
    )
    agent.init()
    agent.enable_training_mode(True)
    return agent, models, memory, layout


def _add_batch(
    memory: RandomMemory,
    layout: CleanResidualLayout,
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
) -> None:
    batch_size = rewards.shape[0]
    observations = torch.randn(batch_size, layout.policy_dim)
    states = torch.randn(batch_size, layout.state_dim)
    next_observations = torch.randn(batch_size, layout.policy_dim)
    next_states = torch.randn(batch_size, layout.state_dim)
    # Add one single-environment transition at a time. RandomMemory's bulk
    # insertion path advances its cursor differently from vector-env writes.
    for index in range(batch_size):
        row = slice(index, index + 1)
        memory.add_samples(
            observations=observations[row],
            states=states[row],
            actions=torch.zeros(1, layout.action_dim),
            rewards=rewards[row],
            next_observations=next_observations[row],
            next_states=next_states[row],
            terminated=terminated[row],
        )


def _parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def test_clean_layout_dimensions_and_point_one_five_composition_contract():
    layout = CleanResidualLayout()

    assert layout.policy_dim == 34
    assert layout.state_dim == 19
    assert layout.action_dim == 4
    assert layout.bc_action_dim == 10
    assert layout.tactile_dim == 5
    assert layout.critic_input_dim == 38
    assert layout.controlled_action_indices == (0, 1, 2, 9)
    assert layout.residual_scale == 0.15

    full_bc_action = torch.tensor(
        [[1.0, -0.4, 0.2, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, -0.25]]
    )
    residual = torch.tensor([[0.25, -0.5, 0.75, -1.0]])
    controlled = layout.compose_controlled_action(full_bc_action, residual)
    expected_full_action = full_bc_action.clone()
    expected_full_action[:, layout.controlled_action_indices] = controlled

    torch.testing.assert_close(
        controlled,
        torch.tensor([[1.0375, -0.475, 0.3125, -0.4]]),
    )
    torch.testing.assert_close(
        expected_full_action[:, 3:9],
        full_bc_action[:, 3:9],
    )
    assert CleanResidualLayout(scale=0.05).residual_scale == 0.05
    for invalid_scale in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            CleanResidualLayout(scale=invalid_scale)


def test_tactile_residual_checkpoint_marker_rejects_legacy_policy_state():
    layout = CleanResidualLayout()
    observation_space, _, action_space = _spaces(layout)
    actor = CleanResidualActor(observation_space, action_space, "cpu", layout=layout)
    state = actor.state_dict()

    validate_tactile_residual_policy_state(state)
    assert int(state[CLEAN_RESIDUAL_CONTRACT_BUFFER].item()) == CLEAN_RESIDUAL_CONTRACT_VERSION
    legacy = dict(state)
    legacy.pop(CLEAN_RESIDUAL_CONTRACT_BUFFER)
    with pytest.raises(ValueError, match="Legacy Clean Residual checkpoint"):
        validate_tactile_residual_policy_state(legacy)


def test_clean_wrapper_pure_helpers_pack_asymmetric_observation_and_scatter_once():
    layout = CleanResidualLayout()
    proprio = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    relative_position = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    object_rot6d = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    full_bc_action = torch.arange(20, dtype=torch.float32).reshape(2, 10) / 10
    tactile_actor = torch.tensor(
        [[0.0, 0.5, 0.0, 1.0, 0.0], [2.0, 1.0, 1.0, 1.0, 1.0]]
    )
    residual = torch.tensor(
        [[0.25, -0.5, 0.75, -1.0], [-0.2, 0.4, -0.6, 0.8]]
    )

    observation = pack_clean_residual_observation(
        proprio,
        relative_position,
        object_rot6d,
        full_bc_action,
        tactile_actor,
        layout=layout,
    )
    full_bc_before = full_bc_action.clone()
    composed = compose_normalized_action(
        full_bc_action,
        residual,
        layout=layout,
    )

    assert set(observation) == {"policy", "critic"}
    assert observation["policy"].shape == (2, 34)
    assert observation["critic"].shape == (2, 19)
    torch.testing.assert_close(observation["policy"][:, :19], observation["critic"])
    torch.testing.assert_close(observation["policy"][:, 19:29], full_bc_action)
    torch.testing.assert_close(observation["policy"][:, 29:], tactile_actor)
    torch.testing.assert_close(full_bc_action, full_bc_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(composed[:, 3:9], full_bc_action[:, 3:9])
    torch.testing.assert_close(
        composed[:, layout.indices],
        full_bc_action[:, layout.indices] + 0.15 * residual,
    )


def test_clean_actor_has_one_tanh_transform_correct_log_prob_and_deterministic_mean():
    torch.manual_seed(11)
    layout = CleanResidualLayout()
    observation_space, _, action_space = _spaces(layout)
    actor = CleanResidualActor(
        observation_space,
        action_space,
        "cpu",
        layout=layout,
        initial_log_std=-0.7,
    )
    with torch.no_grad():
        actor.mean_head.bias.fill_(0.25)

    inputs = {"observations": torch.randn(3, layout.policy_dim)}
    deterministic = actor.deterministic_action(inputs, role="policy")
    actions, outputs = actor.act(inputs, role="policy")
    pre_tanh = outputs["pre_tanh_actions"]
    distribution = torch.distributions.Normal(
        torch.full_like(pre_tanh, 0.25),
        torch.full_like(pre_tanh, math.exp(-0.7)),
    )
    expected_log_prob = distribution.log_prob(pre_tanh)
    expected_log_prob -= actor._tanh_log_abs_det_jacobian(pre_tanh)
    expected_log_prob = expected_log_prob.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(actions, torch.tanh(pre_tanh))
    torch.testing.assert_close(
        deterministic,
        torch.full_like(deterministic, torch.tanh(torch.tensor(0.25))),
    )
    torch.testing.assert_close(outputs["mean_actions"], deterministic)
    torch.testing.assert_close(outputs["log_prob"], expected_log_prob)
    assert actions.shape == (3, 4)
    assert outputs["log_prob"].shape == (3, 1)
    assert bool((actions > -1.0).all() and (actions < 1.0).all())


def test_clean_actor_reparameterization_backpropagates_to_mean_and_log_std_heads():
    torch.manual_seed(7)
    layout = CleanResidualLayout()
    observation_space, _, action_space = _spaces(layout)
    actor = CleanResidualActor(
        observation_space,
        action_space,
        "cpu",
        layout=layout,
        initial_log_std=-1.0,
    )
    actions, outputs = actor.act(
        {"observations": torch.randn(8, layout.policy_dim)},
        role="policy",
    )
    actions.sum().backward()

    assert outputs["pre_tanh_actions"].grad_fn is not None
    assert actor.mean_head.bias.grad is not None
    assert actor.log_std_head.bias.grad is not None
    assert actor.mean_head.bias.grad.abs().sum().item() > 0.0
    assert actor.log_std_head.bias.grad.abs().sum().item() > 0.0


def test_clean_critic_input_is_state_full_bc_context_and_raw_residual():
    layout = CleanResidualLayout()
    observation_space, state_space, action_space = _spaces(layout)
    critic = CleanResidualCritic(
        state_space,
        action_space,
        "cpu",
        layout=layout,
    )
    states = torch.arange(2 * layout.state_dim, dtype=torch.float32).reshape(2, -1)
    actor_only_state = torch.full_like(states, -999.0)
    full_bc_action = torch.arange(2 * layout.bc_action_dim, dtype=torch.float32).reshape(2, -1)
    tactile_actor = torch.tensor(
        [[0.0, 0.5, 0.0, 1.0, 0.0], [2.0, 1.0, 1.0, 1.0, 1.0]]
    )
    observations = torch.cat((actor_only_state, full_bc_action, tactile_actor), dim=-1)
    residual = torch.tensor(
        [[0.1, -0.2, 0.3, -0.4], [-0.5, 0.6, -0.7, 0.8]],
        requires_grad=True,
    )

    network_input = critic.build_network_input(
        {
            "states": states,
            "observations": observations,
            "taken_actions": residual,
        }
    )

    assert critic.network_input_dim == 38
    assert network_input.shape == (2, 38)
    torch.testing.assert_close(network_input[:, :19], states)
    torch.testing.assert_close(network_input[:, 19:29], full_bc_action)
    torch.testing.assert_close(network_input[:, 29:34], tactile_actor)
    torch.testing.assert_close(network_input[:, 34:38], residual)
    assert not torch.equal(network_input[:, :19], actor_only_state)


def test_clean_factory_returns_exactly_five_independent_models_and_clean_agent_syncs_targets():
    layout = CleanResidualLayout()
    observation_space, state_space, action_space = _spaces(layout)
    models = build_clean_residual_sac_models(
        observation_space,
        state_space,
        action_space,
        "cpu",
        layout=layout,
    )

    assert set(models) == {
        "policy",
        "critic_1",
        "critic_2",
        "target_critic_1",
        "target_critic_2",
    }
    assert "target_policy" not in models
    assert len({id(model) for model in models.values()}) == 5
    parameter_ids = [
        {id(parameter) for parameter in model.parameters()}
        for model in models.values()
    ]
    for index, left in enumerate(parameter_ids):
        for right in parameter_ids[index + 1 :]:
            assert left.isdisjoint(right)

    agent, models, _, _ = _new_agent(models=models)
    assert "target_policy" not in agent.checkpoint_modules
    for online_name, target_name in (
        ("critic_1", "target_critic_1"),
        ("critic_2", "target_critic_2"),
    ):
        for online, target in zip(
            models[online_name].parameters(),
            models[target_name].parameters(),
        ):
            assert online.data_ptr() != target.data_ptr()
            torch.testing.assert_close(target, online, rtol=0.0, atol=0.0)
            assert target.requires_grad is False


class _LinearActionCritic(DeterministicMixin, Model):
    """Small Q=bias+sum(w*a) critic for exact native-SAC update tests."""

    def __init__(self, state_space, action_space, *, bias: float = 0.0):
        Model.__init__(
            self,
            observation_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        self.action_weight = torch.nn.Parameter(torch.ones(self.num_actions))
        self.bias = torch.nn.Parameter(torch.tensor(float(bias)))
        self.actions_seen: list[torch.Tensor] = []

    def compute(self, inputs, role):
        del role
        actions = inputs["taken_actions"]
        self.actions_seen.append(actions.detach().clone())
        values = (actions * self.action_weight).sum(dim=-1, keepdim=True)
        return values + self.bias, {}


def _linear_models(*, critic_1_bias: float = 0.0, critic_2_bias: float = 0.0):
    layout = CleanResidualLayout()
    observation_space, state_space, action_space = _spaces(layout)
    policy = CleanResidualActor(
        observation_space,
        action_space,
        "cpu",
        layout=layout,
        initial_log_std=-1.0,
    )

    def critic(bias: float):
        return _LinearActionCritic(state_space, action_space, bias=bias)

    return {
        "policy": policy,
        "critic_1": critic(critic_1_bias),
        "critic_2": critic(critic_2_bias),
        "target_critic_1": critic(critic_1_bias),
        "target_critic_2": critic(critic_2_bias),
    }


def test_clean_agent_alpha_zero_target_is_one_step_min_twin_q_with_terminal_mask():
    agent, models, memory, layout = _new_agent(
        policy_lr=0.0,
        critic_lr=0.0,
        polyak=0.0,
    )
    with torch.no_grad():
        for target_name, bias in (
            ("target_critic_1", 2.0),
            ("target_critic_2", 4.0),
        ):
            target = models[target_name]
            for parameter in target.parameters():
                parameter.zero_()
            target.net[-1].bias.fill_(bias)
    _add_batch(
        memory,
        layout,
        rewards=torch.ones(2, 1),
        terminated=torch.tensor([[False], [True]]),
    )

    agent.update(timestep=0, timesteps=1)

    # Non-terminal: 1 + 0.5 * min(2, 4) = 2. Terminal: 1.
    assert agent.tracking_data["Target / Target (mean)"][-1] == pytest.approx(1.5)
    assert agent.cfg.learn_entropy is False
    assert agent._entropy_coefficient == 0.0
    assert not hasattr(agent, "log_entropy_coefficient")
    assert not hasattr(agent, "entropy_optimizer")
    assert "entropy_optimizer" not in agent.checkpoint_modules
    assert "Loss / Entropy loss" not in agent.tracking_data


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"learn_entropy": True}, "learn_entropy=False"),
        ({"initial_entropy_value": 0.1}, "initial_entropy_value=0.0"),
        ({"random_timesteps": 1}, "random_timesteps=0"),
    ),
)
def test_clean_agent_rejects_entropy_and_a_second_behavior_policy(override, message):
    layout = CleanResidualLayout()
    observation_space, state_space, action_space = _spaces(layout)
    models = build_clean_residual_sac_models(
        observation_space,
        state_space,
        action_space,
        "cpu",
        layout=layout,
    )
    cfg = _alpha_zero_cfg()
    cfg.update(override)

    with pytest.raises(ValueError, match=message):
        CleanAlphaZeroSAC(
            models=models,
            memory=RandomMemory(memory_size=8, num_envs=1, device="cpu"),
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
            cfg=cfg,
        )


def test_clean_agent_actor_uses_sampled_min_twin_q_and_updates_mean_and_log_std():
    torch.manual_seed(23)
    models = _linear_models(critic_1_bias=10.0, critic_2_bias=0.0)
    agent, models, memory, layout = _new_agent(
        models=models,
        policy_lr=1.0e-2,
        critic_lr=0.0,
        polyak=0.0,
    )
    _add_batch(
        memory,
        layout,
        rewards=torch.zeros(2, 1),
        terminated=torch.zeros(2, 1, dtype=torch.bool),
    )
    mean_before = models["policy"].mean_head.bias.detach().clone()
    log_std_before = models["policy"].log_std_head.bias.detach().clone()

    agent.update(timestep=0, timesteps=1)

    actor_actions = models["critic_2"].actions_seen[-1]
    expected_policy_loss = -actor_actions.sum(dim=-1).mean()
    assert agent.tracking_data["Loss / Policy loss"][-1] == pytest.approx(
        expected_policy_loss.item()
    )
    assert not torch.equal(models["policy"].mean_head.bias, mean_before)
    assert not torch.equal(models["policy"].log_std_head.bias, log_std_before)
    assert "Loss / Entropy loss" not in agent.tracking_data


def test_clean_agent_polyak_updates_both_target_critics_without_a_target_actor():
    tau = 0.25
    agent, models, memory, layout = _new_agent(polyak=tau)
    _add_batch(
        memory,
        layout,
        rewards=torch.ones(2, 1),
        terminated=torch.zeros(2, 1, dtype=torch.bool),
    )
    target_before = {
        name: _parameters(models[name])
        for name in ("target_critic_1", "target_critic_2")
    }

    agent.update(timestep=0, timesteps=1)

    for online_name, target_name in (
        ("critic_1", "target_critic_1"),
        ("critic_2", "target_critic_2"),
    ):
        for before, online_after, target_after in zip(
            target_before[target_name],
            models[online_name].parameters(),
            models[target_name].parameters(),
        ):
            torch.testing.assert_close(
                target_after,
                (1.0 - tau) * before + tau * online_after,
            )
    assert "target_policy" not in models


def test_policy_sample_is_used_from_timestep_zero_and_replay_stores_it_exactly():
    torch.manual_seed(31)
    agent, _, memory, layout = _new_agent()
    observations = torch.randn(1, layout.policy_dim)
    states = torch.randn(1, layout.state_dim)
    next_observations = torch.randn(1, layout.policy_dim)
    next_states = torch.randn(1, layout.state_dim)
    with torch.no_grad():
        actions, outputs = agent.act(
            observations,
            states,
            timestep=0,
            timesteps=100,
        )

    assert agent.cfg.random_timesteps == 0
    assert "log_prob" in outputs
    assert "pre_tanh_actions" in outputs
    agent.record_transition(
        observations=observations,
        states=states,
        actions=actions,
        rewards=torch.tensor([[1.25]]),
        next_observations=next_observations,
        next_states=next_states,
        terminated=torch.zeros(1, 1, dtype=torch.bool),
        truncated=torch.zeros(1, 1, dtype=torch.bool),
        infos={},
        timestep=0,
        timesteps=100,
    )

    stored_actions = memory.get_tensor_by_name("actions")[0, 0]
    torch.testing.assert_close(stored_actions, actions[0], rtol=0.0, atol=0.0)


class _FakeFlowAdapter:
    n_action_steps = 32
    action_dim = 10

    def __init__(self):
        self.normalized_chunk = torch.linspace(-0.8, 0.8, 320).reshape(32, 10)
        self.decode_calls = 0
        self.reset_calls = 0
        self.unnormalized_inputs: list[torch.Tensor] = []
        self.decode_noise_seeds: list[int | None] = []

    @property
    def is_ready(self) -> bool:
        return True

    def reset(self) -> None:
        self.reset_calls += 1

    def decode_with_normalized(self, *, noise_seed=None):
        self.decode_calls += 1
        self.decode_noise_seeds.append(noise_seed)
        normalized = self.normalized_chunk.clone()
        return normalized + 50.0, normalized

    def unnormalize_action(self, normalized_action):
        self.unnormalized_inputs.append(normalized_action.detach().clone())
        return normalized_action + 10.0

    def normalize_proprioception(self, physical_proprioception, *, phase):
        del phase
        return physical_proprioception

    def update(self, observation) -> None:
        del observation


class _FakeLabPickEnv(gym.Env):
    metadata = {}

    def __init__(self, *, truncated: bool):
        super().__init__()
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.cfg = types.SimpleNamespace(rl_align_cafe_action_yaw=True)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (10,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32)
        self.single_action_space = self.action_space
        self.single_observation_space = self.observation_space
        self.actions: list[torch.Tensor] = []
        self.has_touched = torch.zeros(1, dtype=torch.bool)
        self.left_touch = torch.zeros(1)
        self.right_touch = torch.zeros(1)
        self._truncated = bool(truncated)

    def get_cafe_observation(self):
        return {"robot0_pos": torch.arange(10, dtype=torch.float32).reshape(1, 10)}

    def get_privileged_object_pose(self):
        return (
            torch.tensor([[0.1, -0.2, 0.3]]),
            torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
        )

    def tactile_contact_depths(self):
        return self.left_touch, self.right_touch

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.has_touched.zero_()
        self.left_touch.zero_()
        self.right_touch.zero_()
        return {"policy": torch.zeros(1, 16)}, {}

    def step(self, action):
        self.actions.append(action.detach().clone())
        return (
            {"policy": torch.zeros(1, 16)},
            torch.tensor([1.25]),
            torch.tensor([False]),
            torch.tensor([self._truncated]),
            {"source": "fake"},
        )


def test_clean_wrapper_executes_exact_composed_action_and_masks_timeout(monkeypatch):
    adapter = _FakeFlowAdapter()
    adapter_call = {}

    def from_pretrained(checkpoint, **kwargs):
        adapter_call.update(checkpoint=checkpoint, **kwargs)
        return adapter

    monkeypatch.setattr(
        clean_wrapper.FlowMatchingNoiseAdapter,
        "from_pretrained",
        staticmethod(from_pretrained),
    )
    base_env = _FakeLabPickEnv(truncated=True)
    wrapper = CleanResidualLabPickWrapper(
        base_env,
        "fake_flow.pt",
        device="cpu",
        residual_scale=0.15,
        flow_num_inference_steps=7,
        camera_warmup_steps=0,
        seed=17,
    )
    observation, _ = wrapper.reset()
    original_bc_action = wrapper.current_bc_action
    residual = torch.tensor([[0.25, -0.5, 0.75, -1.0]])

    next_observation, reward, terminated, truncated, info = wrapper.step(residual)

    expected_normalized = compose_normalized_action(original_bc_action, residual)
    assert adapter_call == {
        "checkpoint": "fake_flow.pt",
        "device": "cpu",
        "num_inference_steps": 7,
        "visual_xy_lock_phase": 0.30,
        "use_visual_xy_override": True,
        "seed": 17,
    }
    assert observation["policy"].shape == next_observation["policy"].shape == (1, 34)
    assert observation["critic"].shape == next_observation["critic"].shape == (1, 19)
    assert len(base_env.actions) == 1
    assert len(adapter.unnormalized_inputs) == 1
    torch.testing.assert_close(
        adapter.unnormalized_inputs[0],
        expected_normalized,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        base_env.actions[0],
        expected_normalized + 10.0,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        adapter.unnormalized_inputs[0][:, 3:9],
        original_bc_action[:, 3:9],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(reward, torch.tensor([1.25]))
    assert bool(terminated.item()) is True
    assert bool(truncated.item()) is True
    assert info["clean_residual/action_index"] == 0
    assert info["clean_residual/action_repeat"] == 1
    assert info["clean_residual/replanned"] is True
    assert adapter.decode_calls == 2
    assert adapter.decode_noise_seeds == [17, 18]


def test_clean_wrapper_reuses_fixed_flow_noise_for_every_ten_action_replan(monkeypatch):
    adapter = _FakeFlowAdapter()
    monkeypatch.setattr(
        clean_wrapper.FlowMatchingNoiseAdapter,
        "from_pretrained",
        staticmethod(lambda checkpoint, **kwargs: adapter),
    )
    wrapper = CleanResidualLabPickWrapper(
        _FakeLabPickEnv(truncated=False),
        "fake_flow.pt",
        device="cpu",
        camera_warmup_steps=0,
        seed=23,
    )
    monkeypatch.setattr(wrapper, "_raw_flow_observation", lambda: {})

    wrapper.reset()
    for _ in range(clean_wrapper.REPLAN_STEPS):
        wrapper.step(torch.zeros(1, 4))

    assert clean_wrapper.REPLAN_STEPS == 10
    assert wrapper.flow_noise_seed == 23
    assert adapter.decode_calls == 2
    assert adapter.decode_noise_seeds == [23, 23]
