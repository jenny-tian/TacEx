from __future__ import annotations

import json
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
from generate_online_dsrl_report import generate_artifacts  # noqa: E402


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


def test_layout_adds_a_scaled_repeated_correction_to_native_noise():
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
    native = torch.randn(1, 5, 10)
    composed = layout.compose_noise(native, actions)

    assert expanded.shape == (1, 5, 10)
    torch.testing.assert_close(expanded[:, :2], actions.reshape(1, 2, 10))
    for index in range(2, 5):
        torch.testing.assert_close(expanded[:, index], expanded[:, 1])
    torch.testing.assert_close(composed, native + 0.25 * expanded)
    torch.testing.assert_close(
        CleanDSRLLayout(policy=24, flow_horizon=5, residual_scale=0.0).compose_noise(
            torch.clone(native), torch.zeros(1, 10)
        ),
        native,
        rtol=0.0,
        atol=0.0,
    )


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
        torch.full_like(distribution_outputs["log_std"], -2.0),
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


def test_base_anchored_v4_checkpoint_marker_rejects_legacy_policy_state():
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


def test_base_anchored_dsrl_entrypoints_expose_residual_scale():
    trainer_source = (SKRL_ROOT / "train_clean_dsrl_sac.py").read_text()
    evaluator_source = (DSRL_ROOT / "eval_lab_pick_clean_dsrl_sac_runtime.py").read_text()

    for source in (trainer_source, evaluator_source):
        assert '"--noise_strategy"' not in source
    assert '"--noise_residual_scale"' in trainer_source
    assert "flow_noise_dsrl_base_anchored_repeat_last_v4_tactile" in trainer_source
    assert "clean_dsrl_sac_base_anchored_l" in trainer_source


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

    def sample_native_noise(self, *, batch_size=1):
        return torch.randn(
            batch_size,
            self.n_action_steps,
            self.action_dim,
            generator=self.runner.generator,
        )


class _FakeLabPickEnv(gym.Env):
    metadata = {}

    def __init__(
        self,
        *,
        truncate_after: int | None = None,
        terminate_after: int | None = None,
        log_by_step: dict[int, dict] | None = None,
    ):
        super().__init__()
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.cfg = types.SimpleNamespace(
            rl_align_cafe_action_yaw=False,
            terminate_break_force_threshold_n=3.5,
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (10,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32)
        self.single_action_space = self.action_space
        self.single_observation_space = self.observation_space
        self.actions: list[torch.Tensor] = []
        self.has_touched = torch.zeros(1, dtype=torch.bool)
        self.left_touch = torch.zeros(1)
        self.right_touch = torch.zeros(1)
        self.truncate_after = truncate_after
        self.terminate_after = terminate_after
        self.log_by_step = log_by_step or {}

    def get_cafe_observation(self):
        return {"robot0_pos": torch.arange(10, dtype=torch.float32).reshape(1, 10)}

    def get_privileged_object_pose(self):
        return torch.zeros(1, 3), torch.zeros(1, 6)

    def tactile_contact_depths(self):
        return self.left_touch, self.right_touch

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.actions.clear()
        self.has_touched.zero_()
        self.left_touch.zero_()
        self.right_touch.zero_()
        return {"policy": torch.zeros(1, 16)}, {}

    def step(self, action):
        self.actions.append(action.detach().clone())
        truncated = self.truncate_after is not None and len(self.actions) >= self.truncate_after
        terminated = self.terminate_after is not None and len(self.actions) >= self.terminate_after
        return (
            {"policy": torch.zeros(1, 16)},
            torch.ones(1),
            torch.tensor([terminated]),
            torch.tensor([truncated]),
            {"log": self.log_by_step.get(len(self.actions), {})},
        )


def _wrapper(
    monkeypatch,
    *,
    truncate_after=None,
    terminate_after=None,
    log_by_step=None,
    **kwargs,
):
    adapter = _FakeFlowAdapter()
    monkeypatch.setattr(
        wrapper_module.FlowMatchingNoiseAdapter,
        "from_pretrained",
        staticmethod(lambda checkpoint, **adapter_kwargs: adapter),
    )
    wrapper = CleanDSRLLabPickWrapper(
        _FakeLabPickEnv(
            truncate_after=truncate_after,
            terminate_after=terminate_after,
            log_by_step=log_by_step,
        ),
        "fake_flow.pt",
        device="cpu",
        camera_warmup_steps=0,
        **kwargs,
    )
    monkeypatch.setattr(wrapper, "_raw_flow_observation", lambda: {})
    return wrapper, adapter


def test_wrapper_adds_scaled_noise_correction_and_returns_discounted_chunk_reward(monkeypatch):
    wrapper, adapter = _wrapper(
        monkeypatch,
        chunk_execute_steps=3,
        chunk_discount=0.5,
    )
    observation, _ = wrapper.reset()
    action = torch.linspace(-1.0, 1.0, 10).reshape(1, 10)
    next_observation, reward, terminated, truncated, info = wrapper.step(action)

    assert observation["policy"].shape == next_observation["policy"].shape == (1, 29)
    torch.testing.assert_close(observation["policy"][:, -5:], torch.zeros(1, 5))
    assert observation["critic"].shape == next_observation["critic"].shape == (1, 19)
    assert len(wrapper.env.actions) == 6
    assert len(adapter.decode_inputs) == 1
    full_noise = adapter.decode_inputs[0].reshape(1, 32, 10)
    native = torch.randn(1, 32, 10, generator=torch.Generator().manual_seed(0))
    for index in range(32):
        torch.testing.assert_close(full_noise[:, index], native[:, index] + 0.25 * action)
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

    assert adapter.decode_inputs[0] is not None
    torch.testing.assert_close(reward, torch.ones(1))
    assert bool(terminated.item())
    assert bool(truncated.item())
    assert info["clean_dsrl/action_steps_executed"] == 1
    assert "clean_dsrl/native_noise_rms" in info
    assert "CleanDSRL/native_noise_rms" in info["log"]


def test_zero_correction_decodes_the_exact_native_flow_noise(monkeypatch):
    wrapper, adapter = _wrapper(
        monkeypatch,
        chunk_execute_steps=1,
    )
    wrapper.reset(flow_noise_seed=23)
    wrapper.step_bc()
    native_noise = adapter.decode_inputs[-1].clone()
    wrapper.reset(flow_noise_seed=23)
    wrapper.step(torch.zeros(1, 10))
    zero_correction_noise = adapter.decode_inputs[-1]

    torch.testing.assert_close(zero_correction_noise, native_noise, rtol=0.0, atol=0.0)


def test_online_logger_writes_every_interaction_and_prioritizes_breakage(
    monkeypatch, tmp_path
):
    log_by_step = {
        1: {"LabPick/contact_force_n": torch.tensor(1.0)},
        2: {"LabPick/contact_force_n": torch.tensor(2.0)},
        3: {"LabPick/contact_force_n": torch.tensor(3.0)},
        4: {
            "LabPick/contact_force_n": torch.tensor(4.0),
            "LabPick/net_contact_force_n": torch.tensor(6.0),
            "LabPick/success_terminal_step": torch.tensor(1.0),
            "LabPick/broken_terminal_step": torch.tensor(1.0),
        },
    }
    wrapper, _ = _wrapper(
        monkeypatch,
        terminate_after=4,
        log_by_step=log_by_step,
        chunk_execute_steps=1,
        online_metrics_dir=tmp_path,
    )
    wrapper.reset()
    wrapper.step(torch.zeros(1, 10))
    wrapper.step(torch.zeros(1, 10))
    wrapper.close()

    interactions = [
        json.loads(line)
        for line in (tmp_path / "online_interactions.jsonl").read_text().splitlines()
    ]
    episodes = [
        json.loads(line)
        for line in (tmp_path / "online_episodes.jsonl").read_text().splitlines()
    ]
    assert len(interactions) == 2
    assert interactions[0]["status"] == "ongoing"
    assert interactions[0]["success"] is None
    assert interactions[1]["status"] == "failure"
    assert interactions[1]["success"] is False
    assert interactions[1]["failure_reason"] == "object_broken"
    assert interactions[1]["chunk_peak_contact_force_n"] == pytest.approx(4.0)
    assert episodes[0]["failure_reason"] == "object_broken"
    assert episodes[0]["num_interactions"] == 2
    assert wrapper._online_episode_index == 2
    assert wrapper._online_episode_step == 0


def test_online_report_handles_completed_and_ongoing_interactions(tmp_path):
    interactions = []
    terminal_by_index = {
        2: (1, True, None),
        4: (2, False, "object_broken"),
        5: (3, True, None),
    }
    episode_index = 1
    for index in range(1, 7):
        terminal = index in terminal_by_index
        outcome = terminal_by_index.get(index)
        interactions.append(
            {
                "interaction_index": index,
                "episode_index": episode_index,
                "terminal": terminal,
                "success": None if outcome is None else outcome[1],
                "failure_reason": None if outcome is None else outcome[2],
            }
        )
        if terminal:
            episode_index += 1
    episodes = [
        {
            "episode_index": ep,
            "ending_interaction_index": ending,
            "success": success,
            "failure_reason": failure,
            "peak_contact_force_n": float(ep),
        }
        for ending, (ep, success, failure) in terminal_by_index.items()
    ]
    (tmp_path / "online_interactions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in interactions)
    )
    (tmp_path / "online_episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes)
    )

    summary = generate_artifacts(
        tmp_path,
        expected_interactions=6,
        rolling_episodes=2,
        experiment_id="TEST-B-260827-001",
    )

    assert summary["completed_episode_count"] == 3
    assert summary["final_incomplete_episode_interactions"] == 1
    assert summary["failure_counts"]["object_broken"] == 1
    assert (tmp_path / "online_success_rate_curve.png").stat().st_size > 0
    assert (tmp_path / "failure_reason_trends.png").stat().st_size > 0
    assert (tmp_path / "report.md").read_text().startswith("---\n")


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
        max_gradient_updates=1,
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
    assert agent.optimizer_updates_completed == 1

    actor_after_first_update = [
        parameter.detach().clone() for parameter in models["policy"].parameters()
    ]
    agent.update(timestep=1, timesteps=2)
    assert agent.optimizer_updates_completed == 1
    assert all(
        torch.equal(before, after)
        for before, after in zip(actor_after_first_update, models["policy"].parameters())
    )
