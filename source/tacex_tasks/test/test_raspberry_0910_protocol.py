from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
TRAINER = ROOT / "scripts/reinforcement_learning/raspberry_pick/train_online_codex_vlm_dsrl.py"
RUNNER = ROOT / "scripts/reinforcement_learning/raspberry_pick/run_raspberry_0910.py"
SUPPORT = ROOT / "scripts/reinforcement_learning/raspberry_pick/online_rl_support.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_raspberry_0910_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_support():
    spec = importlib.util.spec_from_file_location("raspberry_0910_support_test", SUPPORT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_0910_runner_freezes_chunked_failure_only_protocol():
    runner = load_runner()
    for method in ("dsrl_only", "vlm_two_stage_dsrl"):
        command = runner.command_for(
            Path("/python"),
            method=method,
            output_root=Path(f"/tmp/{method}"),
            seed=970_000,
            episodes=100,
        )
        assert option(command, "--protocol-name") == "raspberry_0910_v1"
        assert option(command, "--num-inference-steps") == "10"
        assert option(command, "--chunk-execute-steps") == "8"
        assert option(command, "--vlm-call-policy") == "failure_only"


def test_trainer_executes_each_selected_chunk_action_before_replanning():
    source = TRAINER.read_text(encoding="utf-8")
    assert "args.chunk_execute_steps * args.action_repeat" in source
    assert "chunk_action_index = chunk_physics_index // args.action_repeat" in source
    assert "target = chunk[:, chunk_action_index]" in source
    assert "(chunk_physics_index + 1) % args.action_repeat == 0" in source
    assert "deferring DINO/Flow inference" in source


def test_trainer_skips_vlm_on_success_under_failure_only_policy():
    source = TRAINER.read_text(encoding="utf-8")
    assert "should_call_vlm(" in source
    assert '"update_kind"] = "vlm_skipped_success_hold"' in source
    assert '"vlm_called": vlm_called' in source


def test_failure_only_vlm_gate_calls_failures_and_skips_successes():
    support = load_support()
    assert not support.should_call_vlm(
        force_range_source="vlm", call_policy="failure_only", outcome="success"
    )
    assert support.should_call_vlm(
        force_range_source="vlm", call_policy="failure_only", outcome="no_contact"
    )
    assert support.should_call_vlm(
        force_range_source="vlm", call_policy="every_episode", outcome="success"
    )
    assert not support.should_call_vlm(
        force_range_source="fixed", call_policy="failure_only", outcome="no_contact"
    )
