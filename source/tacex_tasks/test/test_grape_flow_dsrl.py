from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


flow = _load(
    "grape_flow_policy_test",
    "scripts/reinforcement_learning/grape_pick/grape_flow_policy.py",
)
control = _load(
    "grape_force_control_test",
    "scripts/reinforcement_learning/grape_pick/grape_force_control.py",
)


def test_flow_decode_is_deterministic_for_fixed_noise():
    torch.manual_seed(11)
    config = flow.GrapeFlowConfig(horizon=3, hidden_dim=32, depth=1, time_embedding_dim=8)
    model = flow.GrapeFlowPolicy(config).eval()
    observation = torch.randn(2, 11)
    noise = torch.randn(2, 3, 10)
    first = model.decode(observation, noise, integration_steps=3)
    second = model.decode(observation, noise, integration_steps=3)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_zero_dsrl_correction_exactly_recovers_native_flow_noise():
    layout = flow.FlowNoiseDSRLLayout(
        horizon=8,
        action_dim=10,
        residual_scale=0.25,
        active_action_indices=(2,),
    )
    native_noise = torch.randn(4, 8, 10)
    correction = torch.zeros(4, 1)
    composed = layout.compose_noise(native_noise, correction)
    torch.testing.assert_close(composed, native_noise, rtol=0.0, atol=0.0)


def test_dsrl_correction_is_applied_before_flow_decode():
    torch.manual_seed(4)
    config = flow.GrapeFlowConfig(horizon=2, hidden_dim=32, depth=1, time_embedding_dim=8)
    model = flow.GrapeFlowPolicy(config).eval()
    layout = flow.FlowNoiseDSRLLayout(horizon=2, action_dim=10, residual_scale=0.3)
    observation = torch.randn(1, 11)
    native_noise = torch.randn(1, 2, 10)
    correction = torch.full((1, 10), 0.5)
    corrected_noise = layout.compose_noise(native_noise, correction)
    assert not torch.equal(corrected_noise, native_noise)
    base_action = model.decode(observation, native_noise, integration_steps=2)
    dsrl_action = model.decode(observation, corrected_noise, integration_steps=2)
    assert not torch.equal(dsrl_action, base_action)


def test_one_dimensional_dsrl_changes_only_z_noise_for_the_whole_horizon():
    layout = flow.FlowNoiseDSRLLayout(
        horizon=8,
        action_dim=10,
        residual_scale=5.0,
        active_action_indices=(2,),
    )
    native_noise = torch.randn(2, 8, 10)
    correction = torch.tensor([[0.25], [-0.5]])
    delta = layout.compose_noise(native_noise, correction) - native_noise
    torch.testing.assert_close(delta[:, :, 2], 5.0 * correction.expand(-1, 8))
    non_z = torch.cat((delta[:, :, :2], delta[:, :, 3:]), dim=-1)
    torch.testing.assert_close(non_z, torch.zeros_like(non_z), rtol=0.0, atol=0.0)


def test_pull_gate_preserves_exact_zero_correction_until_contacted_pull_phase():
    contact = torch.tensor([True, True, False, True])
    attached = torch.tensor([True, False, True, True])
    before = flow.pull_phase_noise_gate(
        623, contact, attached, minimum_step=624
    )
    after = flow.pull_phase_noise_gate(
        624, contact, attached, minimum_step=624
    )
    assert before.tolist() == [False, False, False, False]
    assert after.tolist() == [True, False, False, True]


def test_force_controller_changes_only_gripper_width():
    controller = control.GrapeForceController(control.ForceRange(2.9, 3.2))
    action = torch.randn(2, 10)
    action[:, 9] = 0.02
    executed, diagnostics = controller.control(
        action,
        contact_force_n=torch.tensor([3.3, 3.4]),
        bilateral_contact=torch.tensor([True, True]),
        dt_s=1.0 / 120.0,
    )
    torch.testing.assert_close(executed[:, :9], action[:, :9], rtol=0.0, atol=0.0)
    assert bool((executed[:, 9] > action[:, 9]).all())
    maximum_open_delta = controller.config.safety_open_rate_m_s / 120.0
    assert bool(((executed[:, 9] - action[:, 9]) <= maximum_open_delta + 1.0e-8).all())
    assert bool(diagnostics.safety_override.all())


def test_force_controller_accepts_a_lower_privileged_safety_limit_without_changing_target():
    controller = control.GrapeForceController(
        control.ForceRange(2.2, 2.8),
        control.ForceControllerConfig(hard_force_limit_n=3.1),
    )
    action = torch.zeros(1, 10)
    action[:, 9] = 0.020

    executed, diagnostics = controller.control(
        action,
        contact_force_n=torch.tensor([2.71]),
        bilateral_contact=torch.tensor([True]),
        dt_s=1.0 / 120.0,
        safety_force_limit_n=2.70,
    )

    assert controller.target_range == control.ForceRange(2.2, 2.8)
    assert diagnostics.safety_override.item()
    torch.testing.assert_close(diagnostics.safety_threshold_n, torch.tensor([2.70]))
    assert executed[0, 9] > action[0, 9]
    torch.testing.assert_close(executed[:, :9], action[:, :9], rtol=0.0, atol=0.0)


def test_force_controller_preserves_flow_width_before_physical_contact():
    controller = control.GrapeForceController(control.ForceRange(2.9, 3.2))
    first = torch.zeros(2, 10)
    first[:, 9] = 0.038
    controller.control(
        first,
        contact_force_n=torch.zeros(2),
        bilateral_contact=torch.zeros(2, dtype=torch.bool),
        dt_s=1.0 / 120.0,
    )
    closing = first.clone()
    closing[:, 9] = 0.018
    executed, diagnostics = controller.control(
        closing,
        contact_force_n=torch.zeros(2),
        bilateral_contact=torch.zeros(2, dtype=torch.bool),
        dt_s=1.0 / 120.0,
    )
    torch.testing.assert_close(executed, closing, rtol=0.0, atol=0.0)
    assert not bool(diagnostics.active.any())


def test_episode_n_vlm_range_is_applied_to_episode_n_plus_one():
    adapter = control.OnlineForceRangeAdapter(
        control.ForceRange(2.9, 3.2),
        control.ForceRangeAdapterConfig(smoothing=0.5),
    )
    decision = adapter.update_after_episode(
        episode_index=0,
        proposed_range=control.ForceRange(2.7, 3.0),
        raw_response="{}",
    )
    assert decision["completed_episode_index"] == 0
    assert decision["applies_to_episode_index"] == 1
    assert decision["applied_range_n"] == [2.8, 3.1]
    assert adapter.current_range.as_list() == [2.8, 3.1]


def test_vlm_range_is_projected_below_minimum_damage_threshold():
    adapter = control.OnlineForceRangeAdapter(
        control.ForceRange(2.9, 3.2),
        control.ForceRangeAdapterConfig(
            physical_minimum_n=2.6,
            physical_maximum_n=3.25,
            minimum_width_n=0.15,
            smoothing=1.0,
        ),
    )
    decision = adapter.update_after_episode(
        episode_index=0,
        proposed_range=control.ForceRange(3.4, 5.0),
        raw_response="unsafe proposal",
    )
    low, high = decision["applied_range_n"]
    assert 2.6 <= low < high <= 3.25
    assert high - low >= 0.15 - 1.0e-12
