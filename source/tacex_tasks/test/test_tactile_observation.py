from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
DSRL_ROOT = ROOT / "scripts" / "reinforcement_learning" / "dsrl"
if str(DSRL_ROOT) not in sys.path:
    sys.path.insert(0, str(DSRL_ROOT))

from tactile_observation import (  # noqa: E402
    TACTILE_CONTACT_THRESHOLD_MM,
    TACTILE_INDENTATION_SCALE_MM,
    build_tactile_actor,
    build_tactile_actor_from_env,
)


def test_tactile_vector_shape_dtype_clipping_threshold_and_batch_contract():
    tactile = build_tactile_actor(
        torch.tensor([-1.0, TACTILE_INDENTATION_SCALE_MM, 9.0]),
        torch.tensor([0.05, 0.050001, TACTILE_INDENTATION_SCALE_MM / 2]),
        torch.tensor([False, True, False]),
    )

    assert tactile.shape == (3, 5)
    assert tactile.dtype == torch.float32
    torch.testing.assert_close(tactile[:, 0], torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(tactile[:, 2], torch.tensor([0.0, 1.0, 1.0]))
    torch.testing.assert_close(tactile[:, 3], torch.tensor([0.0, 1.0, 1.0]))
    torch.testing.assert_close(tactile[:, 4], torch.tensor([0.0, 1.0, 0.0]))
    assert TACTILE_CONTACT_THRESHOLD_MM == 0.05


def test_tactile_vector_rejects_nonfinite_and_mismatched_batches():
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_tactile_actor(torch.tensor([float("nan")]), torch.zeros(1), torch.zeros(1))
    with pytest.raises(ValueError, match="same batch size"):
        build_tactile_actor(torch.zeros(2), torch.zeros(1), torch.zeros(2))


def test_tactile_vector_reads_resettable_episode_history_without_force_access():
    env = types.SimpleNamespace(
        device="cpu",
        has_touched=torch.tensor([True]),
        tactile_contact_depths=lambda: (torch.tensor([0.1]), torch.tensor([0.0])),
    )
    before_reset = build_tactile_actor_from_env(env)
    env.has_touched.zero_()
    after_reset = build_tactile_actor_from_env(env)

    assert before_reset[0, 4].item() == 1.0
    assert after_reset[0, 4].item() == 0.0
    torch.testing.assert_close(before_reset[:, :4], after_reset[:, :4])
