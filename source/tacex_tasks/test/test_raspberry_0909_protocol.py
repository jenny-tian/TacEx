from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
TRAINER = ROOT / "scripts/reinforcement_learning/raspberry_pick/train_online_codex_vlm_dsrl.py"
RUNNER = ROOT / "scripts/reinforcement_learning/raspberry_pick/run_raspberry_0909.py"
ANALYZER = ROOT / "scripts/reinforcement_learning/raspberry_pick/analyze_raspberry_0909.py"
BASELINE_RUNNER = (
    ROOT
    / "scripts/reinforcement_learning/raspberry_pick/run_raspberry_0909_baseline_comparison.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("run_raspberry_0909_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_raspberry_0909_test", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_baseline_runner():
    spec = importlib.util.spec_from_file_location(
        "run_raspberry_0909_baseline_test", BASELINE_RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def function_source(name: str) -> str:
    source = TRAINER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def test_formal_runner_builds_the_two_preregistered_contracts():
    runner = load_runner()
    python = Path("/python")
    fixed = runner.command_for(
        python,
        method="dsrl_only",
        output_root=Path("/tmp/fixed"),
        seed=970000,
        episodes=100,
    )
    vlm = runner.command_for(
        python,
        method="vlm_two_stage_dsrl",
        output_root=Path("/tmp/vlm"),
        seed=970000,
        episodes=100,
    )

    assert fixed[fixed.index("--force-range-source") + 1] == "fixed"
    assert fixed[fixed.index("--controller-mode") + 1] == "single_stage"
    assert vlm[vlm.index("--force-range-source") + 1] == "vlm"
    assert vlm[vlm.index("--controller-mode") + 1] == "two_stage"
    for command in (fixed, vlm):
        assert "--bottle-style-lateral-dsrl" in command
        assert "--enhanced-determinism" in command
        assert command[command.index("--initial-force-range") + 1 :][:2] == ["2.2", "2.8"]


def test_actor_and_bottle_critic_do_not_read_hidden_thresholds():
    for name in ("actor_observation", "bottle_style_critic_context"):
        source = function_source(name)
        assert "damage_threshold" not in source
        assert "detach_threshold" not in source
        assert "public_grip_safety_limit_n" in source


def test_checkpoint_contains_resumable_vlm_guard_and_rng_state():
    source = function_source("save_training_checkpoint")
    for key in (
        '"force_range_source"',
        '"controller_mode"',
        '"recent_history"',
        '"recent_candidate_directions"',
        '"vlm_calls"',
        '"vlm_successful_calls"',
        '"replay_state"',
        '"python_random_state"',
        '"torch_random_state"',
        '"episode_records"',
    ):
        assert key in source


def test_formal_vlm_controller_uses_only_public_force_targets():
    source = TRAINER.read_text(encoding="utf-8")
    control_call = source[source.index("executed_action, two_stage_diagnostics") :]
    control_call = control_call[: control_call.index("diagnostics = two_stage_diagnostics.grip")]
    assert "public_tension_target_n" in control_call
    assert "public_grip_safety_limit_n" in control_call
    assert "detach_threshold_n" not in control_call
    assert "damage_threshold_n" not in control_call


def test_analysis_uses_seed_level_rolling_metrics():
    analyzer = load_analyzer()
    records = [{"success": index % 2 == 0} for index in range(100)]

    curve = analyzer.rolling_success(records)

    assert curve.shape == (91,)
    assert all(abs(value - 0.5) < 1e-12 for value in curve)
    assert analyzer.longest_true_streak([False, True, True, False, True]) == 2


def test_baseline_runner_changes_only_the_online_learner():
    runner = load_baseline_runner()
    commands = {
        method: runner.command_for(
            Path("/python"),
            method=method,
            output_root=Path(f"/tmp/{method}"),
            seed=970000,
            episodes=100,
        )
        for method in runner.METHODS
    }

    for method, command in commands.items():
        assert command[command.index("--online-algorithm") + 1] == method
        assert command[command.index("--method") + 1] == "dsrl_only"
        assert command[command.index("--force-range-source") + 1] == "fixed"
        assert command[command.index("--controller-mode") + 1] == "single_stage"
        assert command[command.index("--pull-z-offset-m") + 1] == "0.0"
        assert command[command.index("--initial-force-range") + 1 :][:2] == ["2.2", "2.8"]
        assert "--bottle-style-lateral-dsrl" in command
        assert "--enhanced-determinism" in command
