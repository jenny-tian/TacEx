from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[3]
BC_SCRIPT_DIR = REPO_ROOT / "scripts" / "bc_training"
DSRL_SCRIPT_DIR = REPO_ROOT / "scripts" / "reinforcement_learning" / "dsrl"
BC_POLICY_ROOT = REPO_ROOT / "bc_policy"
for path in (BC_SCRIPT_DIR, DSRL_SCRIPT_DIR, BC_POLICY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dinov3_flow as flow  # noqa: E402
from flow_matching_noise_adapter import FlowMatchingNoiseAdapter  # noqa: E402


class _DummyDINO(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            images.shape[0], 196, self.config.hidden_size, device=images.device, dtype=images.dtype
        ) + self.anchor


def _policy() -> flow.DINOv3FlowPolicy:
    config = flow.DINOv3FlowConfig(
        state_dim=10,
        action_dim=10,
        num_cameras=2,
        state_obs_steps=2,
        image_obs_steps=2,
        chunk_size=8,
        feature_grid_size=2,
        condition_grid_size=2,
        dino_hidden_size=32,
        cond_dim=64,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dim=64,
        transformer_cond_layers=1,
        num_inference_steps=2,
    )
    return flow.DINOv3FlowPolicy(config, _DummyDINO())


def _observation(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "state": torch.randn(batch_size, 2, 10),
        "dino_features": torch.randn(batch_size, 2, 2, 4, 32),
        "phase": torch.rand(batch_size, 1),
    }


def test_dinov3_flow_loss_and_dsrl_noise_decode() -> None:
    model = _policy()
    target = torch.randn(2, 8, 10)
    losses = model.compute_loss(
        {
            "obs": _observation(),
            "action": target,
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        }
    )
    losses["loss"].backward()
    assert set(losses) == {"loss", "flow_loss", "visual_xy_loss"}
    assert torch.isfinite(losses["loss"])
    assert all(parameter.grad is None for parameter in model.dino.parameters())

    model.eval()
    noise = torch.zeros_like(target)
    first = model.predict_action(_observation(), initial_noise=noise, num_inference_steps=2)
    assert first["action"].shape == (2, 8, 10)
    assert first["visual_xy"].shape == (2, 2)
    assert first["dsrl_observation"].shape == (2, 64)


def test_dinov3_flow_config_and_checkpoint_are_dsrl_compatible() -> None:
    model = _policy()
    assert model.config.n_action_steps == 8
    assert model.config.n_state_obs_steps == 2
    assert model.config.n_image_obs_steps == 2
    assert model.config.robot0_pos_dim == 10
    state = flow.policy_state_dict(model)
    assert state
    assert not any(key.startswith("dino.") for key in state)


def test_dsrl_adapter_keeps_separate_phase_out_of_dinov3_proprioception() -> None:
    class _IdentityNormalizer:
        @staticmethod
        def normalize_tensor(key: str, value: torch.Tensor) -> torch.Tensor:
            assert key == "robot0_pos"
            return value

    runner = SimpleNamespace(
        config=SimpleNamespace(n_action_steps=8, action_dim=10, robot0_pos_dim=10),
        include_phase=True,
        include_demo_mode=False,
        normalizer=_IdentityNormalizer(),
    )
    adapter = FlowMatchingNoiseAdapter(runner)
    proprioception = torch.randn(2, 10)
    normalized = adapter.normalize_proprioception(proprioception, phase=0.5)
    assert normalized.shape == (2, 10)
    assert torch.equal(normalized, proprioception)
