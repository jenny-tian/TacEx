from pathlib import Path
import importlib.util
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
SUPPORT = ROOT / "scripts" / "demos" / "seal_twist" / "online_rl_support.py"


def load_support_module():
    spec = importlib.util.spec_from_file_location("seal_twist_online_rl_support", SUPPORT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resumed_exploration_does_not_restart_annealing_for_larger_endpoint():
    support = load_support_module()

    resumed = support.exploration_scale_for_episode(
        episode_index=30,
        total_episodes=100,
        exploration_start=1.0,
        exploration_end=0.15,
        resumed=True,
    )
    restarted = support.exploration_scale_for_episode(
        episode_index=30,
        total_episodes=100,
        exploration_start=1.0,
        exploration_end=0.15,
        resumed=False,
    )

    assert resumed == 0.15
    assert restarted > 0.70


def test_fresh_paired_baseline_can_match_original_30_episode_annealing():
    support = load_support_module()

    at_end = support.exploration_scale_for_episode(
        episode_index=29,
        total_episodes=100,
        exploration_start=1.0,
        exploration_end=0.15,
        resumed=False,
        anneal_episodes=30,
    )
    after_end = support.exploration_scale_for_episode(
        episode_index=80,
        total_episodes=100,
        exploration_start=1.0,
        exploration_end=0.15,
        resumed=False,
        anneal_episodes=30,
    )

    assert at_end == 0.15
    assert after_end == 0.15


def test_actor_state_keeps_vlm_target_and_individual_finger_errors():
    support = load_support_module()
    flow = torch.zeros(2, 28)
    observation = support.force_conditioned_observation(
        flow,
        force_low_n=7.2,
        force_high_n=8.2,
        left_force_n=torch.tensor([4.0, 7.0]),
        right_force_n=torch.tensor([10.0, 7.0]),
    )

    assert observation.shape == (2, support.ACTOR_OBSERVATION_DIM)
    torch.testing.assert_close(observation[0, 28:31], torch.tensor([0.72, 0.82, 0.77]))
    torch.testing.assert_close(observation[0, 31:], torch.tensor([0.37, -0.23, -0.60]))
    assert observation[0, 31] != observation[0, 32]


def test_critic_context_contains_vlm_interval_and_true_episode_thresholds():
    support = load_support_module()
    context = support.critic_force_context(
        reference=torch.zeros(1, support.ACTOR_OBSERVATION_DIM),
        force_low_n=7.2,
        force_high_n=8.2,
        hidden_minimum_n=torch.tensor([5.4]),
        hidden_maximum_n=torch.tensor([9.6]),
        width_correction_m=torch.tensor([2.5e-4]),
        maximum_width_correction_m=5.0e-4,
    )

    assert context.shape == (1, support.CRITIC_FORCE_CONTEXT_DIM)
    torch.testing.assert_close(context[0], torch.tensor([0.72, 0.82, 0.54, 0.96, 0.5]))


def test_force_balance_prior_routes_force_difference_to_semantic_y():
    support = load_support_module()
    reference = torch.zeros(2, 3)
    prior, mask = support.force_balance_action_prior(
        reference,
        action_dim=3,
        actor_coordinate=1,
        target_value=-0.1,
        limit=0.2,
    )

    torch.testing.assert_close(prior[:, 1], torch.tensor([-0.1, -0.1]))
    torch.testing.assert_close(mask[:, 1], torch.ones(2))
    assert torch.count_nonzero(prior[:, (0, 2)]) == 0
    assert torch.count_nonzero(mask[:, (0, 2)]) == 0


def test_force_reward_distinguishes_balanced_from_same_mean_imbalanced_contact():
    support = load_support_module()
    balanced = support.force_shaping_term(
        left_force_n=torch.tensor([7.0]),
        right_force_n=torch.tensor([7.0]),
        force_low_n=6.5,
        force_high_n=7.5,
        bilateral_contact=torch.tensor([True]),
    )
    imbalanced = support.force_shaping_term(
        left_force_n=torch.tensor([4.0]),
        right_force_n=torch.tensor([10.0]),
        force_low_n=6.5,
        force_high_n=7.5,
        bilateral_contact=torch.tensor([True]),
    )

    assert balanced.item() > 0.0
    assert imbalanced.item() < 0.0
    assert balanced.item() > imbalanced.item()


def test_vlm_guard_rejects_global_shift_for_bilateral_imbalance():
    support = load_support_module()
    guarded = support.guard_vlm_force_proposal(
        current_range_n=(7.2, 8.2),
        proposed_range_n=(6.7, 7.7),
        physical_bounds_n=(6.2, 8.8),
        terminal_left_force_n=4.5,
        terminal_right_force_n=9.5,
        outcome="over_force",
        maximum_center_step_n=0.05,
    )

    assert guarded.force_range_n == (7.2, 8.2)
    assert guarded.evidence == "bilateral_imbalance"
    assert guarded.decision == "hold_bilateral_imbalance"


def test_vlm_guard_limits_supported_symmetric_change_and_preserves_width():
    support = load_support_module()
    guarded = support.guard_vlm_force_proposal(
        current_range_n=(7.2, 8.2),
        proposed_range_n=(7.7, 8.7),
        physical_bounds_n=(6.2, 8.8),
        terminal_left_force_n=6.0,
        terminal_right_force_n=6.1,
        outcome="under_force",
        maximum_center_step_n=0.05,
    )

    torch.testing.assert_close(
        torch.tensor(guarded.force_range_n), torch.tensor([7.25, 8.25])
    )
    assert guarded.decision == "accept_symmetric_force_evidence"


def test_persistent_guard_uses_directional_constraint_feedback_inside_target():
    support = load_support_module()
    guarded = support.guard_vlm_force_proposal(
        current_range_n=(7.2, 8.2),
        proposed_range_n=(7.5, 8.5),
        physical_bounds_n=(6.2, 8.8),
        terminal_left_force_n=7.4,
        terminal_right_force_n=8.0,
        outcome="under_force",
        maximum_center_step_n=0.30,
        threshold_schedule="persistent",
        force_constraint_violation="under",
    )

    torch.testing.assert_close(
        torch.tensor(guarded.force_range_n), torch.tensor([7.5, 8.5])
    )
    assert guarded.evidence == "balanced_in_target"
    assert guarded.decision == "accept_persistent_under_constraint_evidence"


def test_iid_guard_rejects_directional_constraint_feedback_without_tracking_evidence():
    support = load_support_module()
    guarded = support.guard_vlm_force_proposal(
        current_range_n=(7.2, 8.2),
        proposed_range_n=(7.5, 8.5),
        physical_bounds_n=(6.2, 8.8),
        terminal_left_force_n=7.4,
        terminal_right_force_n=8.0,
        outcome="under_force",
        maximum_center_step_n=0.30,
        threshold_schedule="iid",
        force_constraint_violation="under",
    )

    assert guarded.force_range_n == (7.2, 8.2)
    assert guarded.decision == "hold_without_symmetric_force_evidence"


def test_block_guard_holds_when_material_changes_before_next_episode():
    support = load_support_module()
    guarded = support.guard_vlm_force_proposal(
        current_range_n=(7.2, 8.2),
        proposed_range_n=(6.9, 7.9),
        physical_bounds_n=(6.2, 8.8),
        terminal_left_force_n=7.4,
        terminal_right_force_n=8.0,
        outcome="over_force",
        maximum_center_step_n=0.30,
        threshold_schedule="block",
        force_constraint_violation="over",
        threshold_persists_next_episode=False,
    )

    assert guarded.force_range_n == (7.2, 8.2)
    assert guarded.decision == "hold_without_symmetric_force_evidence"
