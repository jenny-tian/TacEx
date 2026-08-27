from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[3]
BC_SCRIPT_DIR = REPO_ROOT / "scripts" / "bc_training"
if str(BC_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BC_SCRIPT_DIR))

import dinov3_act as act  # noqa: E402


class _DummyDINO(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            images.shape[0], 196, self.config.hidden_size, device=images.device, dtype=images.dtype
        ) + self.anchor


def _small_policy() -> act.DINOv3ACTPolicy:
    config = act.DINOv3ACTConfig(
        state_dim=10,
        action_dim=10,
        num_cameras=2,
        state_obs_steps=2,
        image_obs_steps=2,
        chunk_size=8,
        feature_grid_size=2,
        condition_grid_size=2,
        dino_hidden_size=32,
        model_dim=64,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
    )
    return act.DINOv3ACTPolicy(config, _DummyDINO())


def test_dinov3_act_cached_feature_forward_and_loss() -> None:
    model = _small_policy()
    batch_size = 3
    observation = {
        "state": torch.randn(batch_size, 2, 10),
        "dino_features": torch.randn(batch_size, 2, 2, 4, 32),
        "phase": torch.rand(batch_size, 1),
    }
    prediction = model(observation)
    assert prediction.shape == (batch_size, 8, 10)
    losses = model.compute_loss(
        {
            "obs": observation,
            "action": torch.randn_like(prediction),
            "action_is_pad": torch.zeros(batch_size, 8, dtype=torch.bool),
        }
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert all(parameter.grad is None for parameter in model.dino.parameters())


def test_dinov3_act_rgb_and_cached_feature_shapes_match() -> None:
    model = _small_policy().eval()
    images = torch.rand(2, 2, 2, 3, 224, 224)
    features = model.extract_dino_features(images)
    assert features.shape == (2, 2, 2, 4, 32)


def test_policy_checkpoint_excludes_frozen_dino() -> None:
    state = act.policy_state_dict(_small_policy())
    assert state
    assert not any(key.startswith("dino.") for key in state)
