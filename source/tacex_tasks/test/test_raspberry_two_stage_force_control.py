from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
HELPERS = ROOT / "scripts" / "reinforcement_learning" / "grape_pick"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))


def _load_controller():
    path = HELPERS / "two_stage_force_control.py"
    spec = importlib.util.spec_from_file_location("two_stage_force_control_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


control = _load_controller()


def _step(
    controller,
    action: torch.Tensor,
    *,
    grip: float,
    tension: float,
    bilateral: bool = True,
    attached: bool = True,
):
    return controller.control(
        action,
        grip_force_n=torch.tensor([grip]),
        bilateral_contact=torch.tensor([bilateral]),
        pedicle_tension_n=torch.tensor([tension]),
        stem_attached=torch.tensor([attached]),
        pull_direction=torch.tensor([[0.0, 0.0, -1.0]]),
        target_tension_n=torch.tensor([0.9]),
        dt_s=0.1,
        grip_safety_limit_n=torch.tensor([2.8]),
    )


def test_two_stage_controller_passes_base_policy_through_before_contact():
    controller = control.TwoStageBerryForceController(control.ForceRange(2.2, 2.6))
    action = torch.tensor([[0.50, 0.10, 0.40, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.03]])

    executed, diagnostics = _step(
        controller, action, grip=0.0, tension=0.0, bilateral=False
    )

    torch.testing.assert_close(executed, action, rtol=0.0, atol=0.0)
    assert diagnostics.stage.item() == control.APPROACH


def test_single_contact_spike_does_not_seize_policy_action():
    controller = control.TwoStageBerryForceController(
        control.ForceRange(2.2, 2.6),
        control.TwoStageForceControllerConfig(contact_confirm_steps=3),
    )
    action = torch.tensor(
        [[0.50, 0.10, 0.40, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.03]]
    )

    executed, diagnostics = _step(controller, action, grip=2.4, tension=0.1)

    assert diagnostics.stage.item() == control.APPROACH
    torch.testing.assert_close(executed, action, rtol=0.0, atol=0.0)


def test_stage_one_holds_pull_axis_until_grip_is_stable():
    controller = control.TwoStageBerryForceController(
        control.ForceRange(2.2, 2.6),
        control.TwoStageForceControllerConfig(
            contact_confirm_steps=1, grip_stable_steps=2
        ),
    )
    contact_action = torch.tensor(
        [[0.50, 0.10, 0.40, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.03]]
    )
    _step(controller, contact_action, grip=2.4, tension=0.1)
    pulling_policy_action = contact_action.clone()
    pulling_policy_action[:, 2] = 0.35

    executed, diagnostics = _step(
        controller, pulling_policy_action, grip=2.4, tension=0.1
    )

    assert diagnostics.stage.item() == control.TENSION_PULL
    assert executed[0, 2] < contact_action[0, 2]
    assert executed[0, 2] > pulling_policy_action[0, 2]
    torch.testing.assert_close(executed[:, :2], pulling_policy_action[:, :2])


def test_stage_two_tracks_truth_tension_and_keeps_grip_loop_active():
    controller = control.TwoStageBerryForceController(
        control.ForceRange(2.2, 2.6),
        control.TwoStageForceControllerConfig(
            contact_confirm_steps=1,
            grip_stable_steps=1,
            tension_filter_alpha=1.0,
            tension_kp_velocity_m_s_per_n=0.1,
            maximum_pull_velocity_m_s=0.02,
        ),
    )
    action = torch.tensor([[0.50, 0.10, 0.40, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.02]])

    executed, diagnostics = _step(controller, action, grip=2.4, tension=0.2)

    assert diagnostics.stage.item() == control.TENSION_PULL
    torch.testing.assert_close(diagnostics.pull_velocity_m_s, torch.tensor([0.02]))
    torch.testing.assert_close(diagnostics.pull_offset_m, torch.tensor([0.002]))
    torch.testing.assert_close(executed[:, 2], torch.tensor([0.398]))
    assert diagnostics.grip.active.item()

    # The two force channels remain active together: excess pad-normal force
    # opens the gripper while low pedicle tension continues motion along the
    # independent pull axis.
    executed, diagnostics = _step(controller, action, grip=2.7, tension=0.2)

    assert diagnostics.stage.item() == control.TENSION_PULL
    assert executed[0, 9] > action[0, 9]
    assert executed[0, 2] < 0.398
    assert diagnostics.pull_velocity_m_s.item() > 0.0


def test_damage_safety_override_pauses_pull_and_opens_gripper():
    controller = control.TwoStageBerryForceController(
        control.ForceRange(2.2, 2.6),
        control.TwoStageForceControllerConfig(grip_stable_steps=1),
    )
    action = torch.tensor([[0.50, 0.10, 0.40, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.02]])

    executed, diagnostics = _step(controller, action, grip=2.9, tension=0.2)

    assert diagnostics.grip.safety_override.item()
    assert diagnostics.pull_velocity_m_s.item() == 0.0
    assert executed[0, 9] > action[0, 9]


def test_lost_contact_releases_pull_axis_and_allows_reacquisition():
    controller = control.TwoStageBerryForceController(
        control.ForceRange(2.2, 2.6),
        control.TwoStageForceControllerConfig(
            contact_confirm_steps=1, grip_stable_steps=3
        ),
    )
    contact_action = torch.tensor(
        [[0.50, 0.10, 0.40, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.03]]
    )
    _step(controller, contact_action, grip=2.4, tension=0.1)
    policy_recovery_action = contact_action.clone()
    policy_recovery_action[:, 2] = 0.35

    for _ in range(7):
        executed, diagnostics = _step(
            controller,
            policy_recovery_action,
            grip=0.0,
            tension=0.0,
            bilateral=False,
        )

    assert diagnostics.stage.item() == control.APPROACH
    torch.testing.assert_close(executed[:, :3], policy_recovery_action[:, :3])
    torch.testing.assert_close(diagnostics.pull_offset_m, torch.tensor([0.0]))

    _, diagnostics = _step(
        controller, policy_recovery_action, grip=2.4, tension=0.1
    )
    assert diagnostics.stage.item() == control.GRIP_STABILIZE


def test_detachment_transitions_to_retention_without_more_pull():
    controller = control.TwoStageBerryForceController(
        control.ForceRange(2.2, 2.6),
        control.TwoStageForceControllerConfig(
            contact_confirm_steps=1, grip_stable_steps=1
        ),
    )
    action = torch.tensor([[0.50, 0.10, 0.40, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.02]])
    _step(controller, action, grip=2.4, tension=0.2)

    _, diagnostics = _step(
        controller, action, grip=2.4, tension=0.0, attached=False
    )

    assert diagnostics.stage.item() == control.RETAIN
    assert diagnostics.pull_velocity_m_s.item() == 0.0


def test_force_balance_aligner_moves_toward_contacted_pad_and_holds_offset():
    aligner = control.ForceBalanceAligner(
        control.ForceBalanceAlignmentConfig(
            contact_force_threshold_n=0.05,
            alignment_velocity_m_s=0.03,
            maximum_alignment_offset_m=0.012,
        )
    )
    action = torch.zeros(1, 10)
    axis = torch.tensor([[0.0, 1.0, 0.0]])

    executed, diagnostics = aligner.control(
        action,
        left_force_n=torch.tensor([1.0]),
        right_force_n=torch.tensor([0.0]),
        bilateral_contact=torch.tensor([False]),
        closing_axis_b=axis,
        dt_s=0.1,
    )

    assert diagnostics.active.item()
    torch.testing.assert_close(
        diagnostics.normalized_force_imbalance, torch.tensor([-1.0])
    )
    torch.testing.assert_close(diagnostics.alignment_offset_m, torch.tensor([-0.003]))
    torch.testing.assert_close(executed[0, :3], torch.tensor([0.0, -0.003, 0.0]))

    # The force pulse supplies a direction; continue searching after transient
    # contact disappears until bilateral contact is acquired.
    executed, diagnostics = aligner.control(
        action,
        left_force_n=torch.tensor([0.0]),
        right_force_n=torch.tensor([0.0]),
        bilateral_contact=torch.tensor([False]),
        closing_axis_b=axis,
        dt_s=0.1,
    )

    assert diagnostics.active.item()
    torch.testing.assert_close(diagnostics.alignment_offset_m, torch.tensor([-0.006]))
    torch.testing.assert_close(executed[0, :3], torch.tensor([0.0, -0.006, 0.0]))

    executed, diagnostics = aligner.control(
        action,
        left_force_n=torch.tensor([1.0]),
        right_force_n=torch.tensor([1.0]),
        bilateral_contact=torch.tensor([True]),
        closing_axis_b=axis,
        dt_s=0.1,
    )
    assert not diagnostics.active.item()
    torch.testing.assert_close(diagnostics.alignment_offset_m, torch.tensor([-0.006]))
    torch.testing.assert_close(executed[0, :3], torch.tensor([0.0, -0.006, 0.0]))


def test_force_balance_aligner_clamps_offset_and_resets():
    aligner = control.ForceBalanceAligner(
        control.ForceBalanceAlignmentConfig(
            alignment_velocity_m_s=0.03,
            maximum_alignment_offset_m=0.004,
        )
    )
    action = torch.zeros(1, 10)
    kwargs = {
        "left_force_n": torch.tensor([0.0]),
        "right_force_n": torch.tensor([1.0]),
        "bilateral_contact": torch.tensor([False]),
        "closing_axis_b": torch.tensor([[0.0, 1.0, 0.0]]),
        "dt_s": 0.1,
    }
    for _ in range(4):
        _, diagnostics = aligner.control(action, **kwargs)
    torch.testing.assert_close(diagnostics.alignment_offset_m, torch.tensor([0.004]))

    aligner.reset()
    executed, diagnostics = aligner.control(
        action,
        left_force_n=torch.tensor([0.0]),
        right_force_n=torch.tensor([0.0]),
        bilateral_contact=torch.tensor([False]),
        closing_axis_b=kwargs["closing_axis_b"],
        dt_s=0.1,
    )
    torch.testing.assert_close(diagnostics.alignment_offset_m, torch.tensor([0.0]))
    torch.testing.assert_close(executed, action)
