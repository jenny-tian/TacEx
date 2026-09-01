from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
LAB_PICK_ROOT = ROOT / "source" / "tacex_tasks" / "tacex_tasks" / "lab_pick"
if str(LAB_PICK_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_PICK_ROOT))

from vlm_force import (  # noqa: E402
    ConvergentForceEstimator,
    DeterministicVLMAdvisor,
    EpisodeFeedback,
    EpisodeForceAdaptationLoop,
    ForceControllerConfig,
    ForceEstimatorConfig,
    ForceRange,
    OpenAICompatibleVLMAdvisor,
    TactileForceController,
    VLMRecommendation,
    build_force_advisor_prompt,
    diagnose_episode_failure,
)


def feedback(
    episode_index: int,
    *,
    success: bool = False,
    reason: str = "object_dropped",
    attempted: ForceRange = ForceRange(1.0, 2.0),
    mean_force: float = 1.1,
    peak_force: float = 1.3,
    contact_fraction: float = 0.6,
) -> EpisodeFeedback:
    return EpisodeFeedback(
        episode_index=episode_index,
        success=success,
        failure_reason=reason,
        attempted_range_n=attempted,
        target_force_n=attempted.center_n,
        mean_contact_force_n=mean_force,
        peak_contact_force_n=peak_force,
        force_rmse_n=0.2,
        contact_fraction=contact_fraction,
        max_lift_m=0.03,
    )


def test_prompt_contains_current_episode_and_aggregated_failure_history():
    previous = feedback(0, reason="object_broken", mean_force=3.4, peak_force=3.7)
    current = feedback(1, reason="object_dropped")
    prompt = build_force_advisor_prompt(
        current_range_n=ForceRange(1.0, 3.0),
        episode=current,
        history=[previous],
        physical_range_n=ForceRange(0.25, 3.25),
        break_force_threshold_n=3.5,
    )

    assert '"current_estimate_n": [' in prompt
    assert '"failure_reason_counts": {' in prompt
    assert '"object_broken": 1' in prompt
    assert '"failure_reason": "object_dropped"' in prompt
    assert "姿态偏差、未接触、轨迹或工作空间错误不能被误判为抓力不足" in prompt
    assert "实测力未达到目标下界" in prompt


def test_episode_loop_calls_advisor_exactly_once_and_persists_transaction(
    tmp_path: Path,
):
    advisor = DeterministicVLMAdvisor(
        physical_range_n=ForceRange(0.25, 3.25), break_force_threshold_n=3.5
    )
    estimator = ConvergentForceEstimator()
    log_path = tmp_path / "episodes.jsonl"
    loop = EpisodeForceAdaptationLoop(
        advisor=advisor, estimator=estimator, log_path=log_path
    )
    episode = feedback(0)

    decision = loop.complete_episode(episode)

    assert advisor.call_count == 1
    assert decision.episode_index == 0
    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert row["advisor_call_index"] == 1
    assert row["history_size_before_call"] == 0
    assert row["episode"]["failure_reason"] == "object_dropped"
    assert '"break_force_threshold_per_finger_n": 3.5' in row["prompt"]
    assert len(row["prompt_sha256"]) == 64
    with pytest.raises(RuntimeError, match="already updated"):
        loop.complete_episode(episode)
    assert advisor.call_count == 1


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
def test_openai_compatible_advisor_uses_strict_structured_output_contract(
    api_mode: str,
):
    advisor = OpenAICompatibleVLMAdvisor(
        model="test-vision-model",
        api_key="test-key",
        physical_range_n=ForceRange(0.25, 3.25),
        break_force_threshold_n=3.5,
        api_mode=api_mode,
    )
    captured: dict[str, object] = {}
    response_text = json.dumps(
        {
            "target_contact_force_range_n": [0.8, 1.2],
            "failure_cause": "success",
            "force_assessment": "safe",
            "rationale": "stable grasp",
            "next_experiment": "repeat",
            "confidence": 0.8,
        }
    )

    def fake_post(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        captured.update(endpoint=endpoint, payload=payload)
        if api_mode == "responses":
            return {"output_text": response_text}
        return {"choices": [{"message": {"content": response_text}}]}

    advisor._post = fake_post  # type: ignore[method-assign]
    recommendation = advisor.recommend(
        current_range_n=ForceRange(1.0, 3.0),
        episode=feedback(0, success=True, reason="success", mean_force=1.0),
        history=(),
    )

    assert recommendation.target_range_n == ForceRange(0.8, 1.2)
    assert advisor.call_count == 1
    payload = captured["payload"]
    if api_mode == "responses":
        assert captured["endpoint"] == "responses"
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["text"]["format"]["strict"] is True
    else:
        assert captured["endpoint"] == "chat/completions"
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True


def test_estimator_tightens_force_evidence_and_converges_around_successes():
    config = ForceEstimatorConfig(
        physical_range_n=ForceRange(0.25, 3.25),
        initial_range_n=ForceRange(0.5, 3.0),
        minimum_range_width_n=0.30,
    )
    estimator = ConvergentForceEstimator(config)
    advisor = DeterministicVLMAdvisor(
        physical_range_n=config.physical_range_n, step_n=0.4
    )
    loop = EpisodeForceAdaptationLoop(advisor=advisor, estimator=estimator)
    widths = [loop.current_range_n.width_n]

    sequence = [
        feedback(
            0, reason="object_dropped", attempted=loop.current_range_n, mean_force=1.0
        ),
        feedback(
            1, reason="object_dropped", attempted=ForceRange(1.2, 2.8), mean_force=1.4
        ),
        feedback(
            2,
            reason="object_broken",
            attempted=ForceRange(1.8, 3.1),
            mean_force=2.9,
            peak_force=3.6,
        ),
        feedback(
            3,
            success=True,
            reason="success",
            attempted=ForceRange(1.7, 2.6),
            mean_force=2.20,
            peak_force=2.55,
        ),
        feedback(
            4,
            success=True,
            reason="success",
            attempted=ForceRange(1.8, 2.5),
            mean_force=2.25,
            peak_force=2.50,
        ),
        feedback(
            5,
            success=True,
            reason="success",
            attempted=ForceRange(1.9, 2.5),
            mean_force=2.22,
            peak_force=2.48,
        ),
    ]
    for item in sequence:
        widths.append(loop.complete_episode(item).target_range_n.width_n)

    assert estimator.evidence_lower_bound_n > config.physical_range_n.minimum_n
    assert estimator.evidence_upper_bound_n < config.physical_range_n.maximum_n
    assert widths[-1] < widths[0]
    assert 1.9 < estimator.current_range_n.center_n < 2.5
    assert estimator.current_range_n.width_n >= config.minimum_range_width_n - 1.0e-9


def test_force_evidence_bounds_are_monotone_even_when_bracket_is_narrow_or_evidence_conflicts():
    estimator = ConvergentForceEstimator()

    cases = [
        (
            feedback(
                0,
                reason="insufficient_force",
                attempted=ForceRange(1.8, 3.0),
                mean_force=2.4,
            ),
            VLMRecommendation(
                ForceRange(2.4, 2.8),
                "insufficient_force",
                "too_low",
                "low",
                "raise",
                1.0,
            ),
        ),
        (
            feedback(
                1,
                reason="object_broken",
                attempted=ForceRange(2.2, 3.0),
                mean_force=2.1,
            ),
            VLMRecommendation(
                ForceRange(2.2, 2.6), "object_broken", "too_high", "high", "lower", 1.0
            ),
        ),
        # These final two candidates contradict the narrow evidence bracket and
        # must be ignored rather than moving either established bound outward.
        (
            feedback(
                2,
                reason="insufficient_force",
                attempted=ForceRange(3.0, 3.2),
                mean_force=3.1,
            ),
            VLMRecommendation(
                ForceRange(3.0, 3.2),
                "insufficient_force",
                "too_low",
                "low",
                "raise",
                1.0,
            ),
        ),
        (
            feedback(
                3,
                reason="object_broken",
                attempted=ForceRange(0.8, 1.2),
                mean_force=1.0,
            ),
            VLMRecommendation(
                ForceRange(0.8, 1.2), "object_broken", "too_high", "high", "lower", 1.0
            ),
        ),
    ]

    lower_bounds = [estimator.evidence_lower_bound_n]
    upper_bounds = [estimator.evidence_upper_bound_n]
    for episode, recommendation in cases:
        estimator.update(recommendation, episode)
        lower_bounds.append(estimator.evidence_lower_bound_n)
        upper_bounds.append(estimator.evidence_upper_bound_n)

    assert all(after >= before for before, after in zip(lower_bounds, lower_bounds[1:]))
    assert all(after <= before for before, after in zip(upper_bounds, upper_bounds[1:]))
    assert upper_bounds[2] - lower_bounds[2] < estimator.config.minimum_range_width_n
    assert lower_bounds[2:] == [lower_bounds[2], lower_bounds[2], lower_bounds[2]]
    assert upper_bounds[2:] == [upper_bounds[2], upper_bounds[2], upper_bounds[2]]


def test_failure_diagnosis_does_not_treat_unrealized_command_as_low_target_force():
    target = ForceRange(1.2, 1.6)
    assert (
        diagnose_episode_failure(
            "object_dropped",
            touched=True,
            contact_fraction=0.3,
            bilateral_contact_fraction=0.2,
            mean_force_n=0.8,
            attempted_range_n=target,
        )
        == "force_tracking_error"
    )
    assert (
        diagnose_episode_failure(
            "object_dropped",
            touched=True,
            contact_fraction=0.3,
            bilateral_contact_fraction=0.2,
            mean_force_n=1.4,
            attempted_range_n=target,
        )
        == "object_dropped"
    )


def test_non_force_failure_does_not_move_or_shrink_estimate():
    estimator = ConvergentForceEstimator()
    current = estimator.current_range_n
    recommendation = VLMRecommendation(
        target_range_n=ForceRange(2.0, 2.5),
        failure_cause="bad_alignment",
        force_assessment="not_force_related",
        rationale="pose error",
        next_experiment="fix pose",
        confidence=0.9,
    )
    episode = feedback(0, reason="bad_alignment", attempted=current, mean_force=2.2)

    decision = estimator.update(recommendation, episode)

    assert decision.update_kind == "non_force"
    assert decision.target_range_n == current
    assert estimator.informative_episode_count == 0


def test_force_controller_only_overrides_gripper_width_and_changes_correct_direction():
    controller = TactileForceController(
        ForceRange(1.8, 2.2),
        ForceControllerConfig(force_filter_alpha=1.0, kd_width_per_n=0.0),
    )
    policy = torch.tensor(
        [[0.41, -0.02, 0.12, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.010]],
        dtype=torch.float32,
    )

    low_action = policy
    for _ in range(4):
        low_action, low_diag = controller.control(
            policy,
            contact_force_n=torch.tensor([1.0]),
            contact_mask=torch.tensor([True]),
            dt_s=1.0 / 120.0,
        )
    assert low_diag.active.item()
    assert low_action[0, 9] < policy[0, 9]
    torch.testing.assert_close(low_action[:, :9], policy[:, :9], rtol=0.0, atol=0.0)

    high_action, _ = controller.control(
        policy,
        contact_force_n=torch.tensor([3.0]),
        contact_mask=torch.tensor([True]),
        dt_s=1.0 / 120.0,
    )
    assert high_action[0, 9] > low_action[0, 9]
    torch.testing.assert_close(high_action[:, :9], policy[:, :9], rtol=0.0, atol=0.0)


def test_force_controller_can_require_explicit_bilateral_contact_gate():
    controller = TactileForceController(
        ForceRange(1.0, 2.0),
        ForceControllerConfig(require_contact_mask_for_activation=True),
    )
    policy = torch.zeros((1, 10), dtype=torch.float32)
    policy[:, 9] = 0.03

    passed, diagnostics = controller.control(
        policy,
        contact_force_n=torch.tensor([1.0]),
        contact_mask=torch.tensor([False]),
        safety_force_n=torch.tensor([1.0]),
        dt_s=1.0 / 120.0,
    )
    assert not diagnostics.active.item()
    torch.testing.assert_close(passed, policy, rtol=0.0, atol=0.0)

    controlled, diagnostics = controller.control(
        policy,
        contact_force_n=torch.tensor([1.0]),
        contact_mask=torch.tensor([True]),
        safety_force_n=torch.tensor([1.0]),
        dt_s=1.0 / 120.0,
    )
    assert diagnostics.active.item()
    torch.testing.assert_close(controlled[:, :9], policy[:, :9], rtol=0.0, atol=0.0)


def test_force_controller_passes_every_dimension_through_before_contact_and_opens_on_hard_limit():
    controller = TactileForceController(ForceRange(1.5, 2.5))
    policy = torch.linspace(-0.4, 0.5, 10).reshape(1, 10)
    policy[:, 9] = 0.008

    inactive, diagnostics = controller.control(
        policy,
        contact_force_n=torch.zeros(1),
        contact_mask=torch.tensor([False]),
        dt_s=1.0 / 120.0,
    )
    torch.testing.assert_close(inactive, policy, rtol=0.0, atol=0.0)
    assert not diagnostics.active.item()

    emergency, diagnostics = controller.control(
        policy,
        contact_force_n=torch.tensor([2.0]),
        safety_force_n=torch.tensor([3.6]),
        contact_mask=torch.tensor([True]),
        dt_s=1.0 / 120.0,
    )
    assert diagnostics.safety_override.item()
    assert emergency[0, 9] > policy[0, 9]
    torch.testing.assert_close(emergency[:, :9], policy[:, :9], rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="safety_force_n"):
        controller.control(
            policy,
            contact_force_n=torch.tensor([2.0]),
            safety_force_n=torch.tensor([float("nan")]),
            contact_mask=torch.tensor([True]),
            dt_s=1.0 / 120.0,
        )


def test_force_controller_converges_on_a_monotone_contact_plant():
    controller = TactileForceController(
        ForceRange(1.8, 2.2),
        ForceControllerConfig(
            force_filter_alpha=1.0, kd_width_per_n=0.0, contact_settle_steps=0
        ),
    )
    width = torch.tensor([[0.015]], dtype=torch.float32)
    force = torch.zeros(1)
    for _ in range(500):
        force = torch.clamp((0.020 - width[:, 0]) * 500.0, min=0.02)
        policy_action = torch.cat((torch.zeros((1, 9)), width), dim=1)
        controlled_action, _ = controller.control(
            policy_action,
            contact_force_n=force,
            safety_force_n=force,
            contact_mask=torch.tensor([True]),
            dt_s=1.0 / 120.0,
        )
        width = controlled_action[:, 9:10]

    assert abs(float(force.item()) - controller.target_force_n) < 0.02
