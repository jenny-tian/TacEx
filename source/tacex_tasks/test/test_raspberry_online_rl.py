from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
SUPPORT = (
    ROOT
    / "scripts"
    / "reinforcement_learning"
    / "raspberry_pick"
    / "online_rl_support.py"
)
ANALYZER = (
    ROOT
    / "scripts"
    / "reinforcement_learning"
    / "raspberry_pick"
    / "analyze_retrain_protocol.py"
)


def load_support_module():
    spec = importlib.util.spec_from_file_location("raspberry_online_rl_support", SUPPORT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_analyzer_module():
    spec = importlib.util.spec_from_file_location("raspberry_online_rl_analyzer", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def proposal(support, **overrides):
    arguments = {
        "current_range_n": (2.2, 2.8),
        "proposed_range_n": (2.3, 2.9),
        "physical_bounds_n": (1.2, 3.1),
        "outcome": "no_detach",
        "bilateral_contact_fraction": 0.30,
        "mean_left_force_n": 1.9,
        "mean_right_force_n": 1.95,
        "peak_grip_force_n": 2.0,
        "damage_threshold_n": 3.2,
        "damage_risk_fraction": 0.0,
        "recent_candidate_directions": ("up",),
    }
    arguments.update(overrides)
    return support.guard_force_proposal(**arguments)


def test_success_and_no_contact_always_hold_force_proposal():
    support = load_support_module()

    after_success = proposal(support, outcome="success")
    without_contact = proposal(support, bilateral_contact_fraction=0.02)

    assert after_success.force_range_n == (2.2, 2.8)
    assert after_success.decision == "hold_after_success"
    assert without_contact.force_range_n == (2.2, 2.8)
    assert without_contact.decision == "hold_no_contact"


def test_low_tension_or_no_detach_with_force_in_band_does_not_authorize_increase():
    support = load_support_module()

    guarded = proposal(
        support,
        mean_left_force_n=2.42,
        mean_right_force_n=2.48,
        peak_grip_force_n=2.81,
        recent_candidate_directions=("up", "up"),
    )

    assert guarded.force_range_n == (2.2, 2.8)
    assert guarded.candidate_direction == "hold"
    assert guarded.decision == "hold_without_directional_force_evidence"


def test_consistent_bilateral_under_force_requires_confirmation_before_increase():
    support = load_support_module()

    first = proposal(support, recent_candidate_directions=())
    confirmed = proposal(support, recent_candidate_directions=("up",))

    assert first.force_range_n == (2.2, 2.8)
    assert first.decision == "hold_pending_confirmation"
    torch.testing.assert_close(
        torch.tensor(confirmed.force_range_n), torch.tensor((2.25, 2.85))
    )
    assert confirmed.decision == "accept_confirmed_directional_evidence"


def test_consistent_damage_risk_allows_only_small_width_preserving_decrease():
    support = load_support_module()

    guarded = proposal(
        support,
        proposed_range_n=(1.8, 2.4),
        outcome="crushed",
        mean_left_force_n=2.55,
        mean_right_force_n=2.60,
        peak_grip_force_n=2.65,
        damage_threshold_n=2.6,
        damage_risk_fraction=0.08,
        recent_candidate_directions=("down",),
    )

    torch.testing.assert_close(
        torch.tensor(guarded.force_range_n), torch.tensor((2.15, 2.75))
    )
    assert guarded.evidence == "consistent_over_force_or_damage_risk"


def test_early_crush_is_damage_evidence_even_when_full_episode_contact_fraction_is_short():
    support = load_support_module()

    guarded = proposal(
        support,
        proposed_range_n=(2.0, 2.6),
        outcome="crushed",
        bilateral_contact_fraction=0.045,
        mean_left_force_n=1.90,
        mean_right_force_n=1.89,
        peak_grip_force_n=2.64,
        damage_threshold_n=2.6,
        damage_risk_fraction=0.40,
        recent_candidate_directions=("down",),
    )

    assert guarded.candidate_direction == "down"
    assert guarded.decision == "accept_confirmed_directional_evidence"
    torch.testing.assert_close(
        torch.tensor(guarded.force_range_n), torch.tensor((2.15, 2.75))
    )


def test_analyzer_does_not_misclassify_early_crush_as_unsafe_low_contact_change():
    analyzer = load_analyzer_module()
    common = {
        "success": False,
        "bilateral_contact_fraction": 0.04,
        "attempted_force_range_n": [2.2, 2.8],
        "next_force_range_n": [2.15, 2.75],
    }
    run = {
        "summary": {
            "post_training_method": "vlm_semantic_v2",
            "berry_profile": "delicate",
            "training_seed": 1,
        },
        "records": [
            {**common, "outcome": "crushed"},
            {**common, "outcome": "no_detach"},
        ],
    }

    summary = analyzer.summarize_run(run)

    assert summary["failures"] == {"crushed": 1, "low_contact": 1}
    assert summary["unsafe_guard_changes"] == 1


def test_bilateral_imbalance_holds_even_when_vlm_direction_matches_outcome():
    support = load_support_module()

    guarded = proposal(
        support,
        proposed_range_n=(2.0, 2.6),
        outcome="crushed",
        mean_left_force_n=1.4,
        mean_right_force_n=2.8,
        peak_grip_force_n=2.8,
        damage_risk_fraction=0.2,
        recent_candidate_directions=("down", "down"),
    )

    assert guarded.force_range_n == (2.2, 2.8)
    assert guarded.decision == "hold_alignment_or_imbalance"


def test_environment_reward_is_independent_of_vlm_target_and_rewards_balance():
    support = load_support_module()

    kwargs = {
        "left_force_n": 2.35,
        "right_force_n": 2.40,
        "bilateral_contact": True,
        "pedicle_tension_n": 0.65,
        "detach_threshold_n": 0.8,
        "damage_threshold_n": 3.2,
    }
    reward = support.environment_shaped_reward(**kwargs)
    same_physics = support.environment_shaped_reward(**kwargs)
    imbalanced = support.environment_shaped_reward(
        **{**kwargs, "left_force_n": 1.2, "right_force_n": 3.0}
    )

    assert reward == same_physics
    assert reward > imbalanced
    assert "force_range" not in support.environment_shaped_reward.__annotations__


def test_terminal_reward_can_be_assigned_as_delayed_semimarkov_credit():
    support = load_support_module()

    assert support.terminal_outcome_reward("success") == 10.0
    assert support.terminal_outcome_reward("no_detach") == -3.0


def test_semantic_residual_has_explicit_bounded_physical_units():
    support = load_support_module()
    layout = support.SemanticResidualLayout(
        translation_scale_m=(0.004, 0.005, 0.006), gripper_scale_m=0.001
    )
    chunk = torch.zeros(2, 3, 10)
    correction = torch.tensor(((1.0, -1.0, 2.0, -2.0), (0.5, 0.0, -0.5, 0.25)))

    result = layout.apply(chunk, correction)

    torch.testing.assert_close(result[0, 0, :3], torch.tensor((0.004, -0.005, 0.006)))
    torch.testing.assert_close(result[0, 0, 9], torch.tensor(-0.001))
    torch.testing.assert_close(result[1, 0, :3], torch.tensor((0.002, 0.0, -0.003)))
    torch.testing.assert_close(result[1, 0, 9], torch.tensor(0.00025))
    torch.testing.assert_close(result[:, 0], result[:, 2])


def test_semantic_xy_residual_cannot_change_pull_or_gripper_channels():
    support = load_support_module()
    layout = support.SemanticXYResidualLayout(translation_scale_m=(0.004, 0.005))
    chunk = torch.zeros(1, 3, 10)

    result = layout.apply(chunk, torch.tensor(((2.0, -2.0),)))

    torch.testing.assert_close(result[0, 0, :2], torch.tensor((0.004, -0.005)))
    torch.testing.assert_close(result[0, :, 2:], torch.zeros(3, 8))


def test_persistent_profiles_are_coherent_and_fixed_profile_is_original_protocol():
    support = load_support_module()
    fixed = support.RASPBERRY_PROFILES["fixed"]

    assert fixed.detach_threshold_n == 0.8
    assert fixed.damage_threshold_n == 3.2
    for profile in support.RASPBERRY_PROFILES.values():
        assert 0.0 < profile.detach_threshold_n < profile.damage_threshold_n
        assert profile.stem_stiffness_n_per_m > 0.0
        assert profile.stem_damping_n_s_per_m > 0.0


def test_oracle_force_range_uses_true_damage_threshold_with_fixed_safety_margin():
    support = load_support_module()

    oracle = support.theoretical_oracle_force_range(
        support.RASPBERRY_PROFILES["delicate"],
        interval_width_n=0.60,
        damage_margin_n=0.20,
    )

    torch.testing.assert_close(torch.tensor(oracle.as_tuple()), torch.tensor((1.8, 2.4)))
    assert oracle.source_damage_threshold_n == 2.6
    assert oracle.maximum_n < support.RASPBERRY_PROFILES["delicate"].damage_threshold_n


def test_oracle_force_range_rejects_invalid_contracts():
    support = load_support_module()
    profile = support.RASPBERRY_PROFILES["delicate"]

    for kwargs in (
        {"interval_width_n": 0.0},
        {"damage_margin_n": 0.0},
        {"interval_width_n": 3.0},
    ):
        try:
            support.theoretical_oracle_force_range(profile, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid oracle contract to fail: {kwargs}")


def test_semantic_residual_stops_before_calibrated_pull_but_legacy_latent_remains_active():
    support = load_support_module()
    common = {
        "physics_step": 700,
        "start_step": 360,
        "phase": 0.78,
        "pull_start_phase": 0.72,
        "stem_attached": True,
    }

    assert not support.residual_is_active(**common, semantic_residual=True)
    assert support.residual_is_active(**common, semantic_residual=False)
    assert not support.residual_is_active(
        **{**common, "physics_step": 300, "phase": 0.33},
        semantic_residual=True,
    )
