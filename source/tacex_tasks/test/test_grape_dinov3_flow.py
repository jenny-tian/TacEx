from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]


def _load():
    path = ROOT / "scripts/reinforcement_learning/grape_pick/train_grape_dinov3_flow_bc.py"
    spec = importlib.util.spec_from_file_location("train_grape_dinov3_flow_bc_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


training = _load()


def test_trajectory_state_uses_robot_and_physical_pad_loads_only(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.npz"
    robot = np.arange(30, dtype=np.float32).reshape(3, 10)
    pad_force = np.arange(6, dtype=np.float32).reshape(3, 2)
    np.savez_compressed(
        path,
        robot0_pos=robot,
        finger_normal_force_n=pad_force,
        grape_pos_b=np.full((3, 3), 999.0, dtype=np.float32),
    )
    with np.load(path) as trajectory:
        state = training.trajectory_state(trajectory)
    assert state.shape == (3, 12)
    np.testing.assert_array_equal(state[:, :10], robot)
    np.testing.assert_array_equal(state[:, 10:], pad_force)
    assert not np.any(state == 999.0)


def test_grape_dino_dataset_aligns_history_features_and_future_actions(
    tmp_path: Path,
) -> None:
    length = 3
    episode = training.Episode(
        path=tmp_path / "record_000000" / "trajectory.npz",
        name="record_000000",
        state=np.arange(length * 12, dtype=np.float32).reshape(length, 12),
        action=np.arange(length * 10, dtype=np.float32).reshape(length, 10),
        step=np.asarray([0, 4, 8], dtype=np.int64),
        length=length,
    )
    cache = tmp_path / "cache" / episode.name
    cache.mkdir(parents=True)
    for camera_index, key in enumerate(training.RAW_IMAGE_KEYS):
        features = np.full((length, 4, 6), camera_index + 1, dtype=np.float16)
        np.save(cache / f"{key}.npy", features)
    normalizer = training.compute_normalizer([episode], "limits")
    dataset = training.GrapeDINOSequenceDataset(
        [episode],
        normalizer,
        tmp_path / "cache",
        state_obs_steps=2,
        image_obs_steps=2,
        chunk_size=4,
        max_episode_steps=8,
    )
    sample = dataset[2]
    assert sample["obs"]["state"].shape == (2, 12)
    assert sample["obs"]["dino_features"].shape == (2, 2, 4, 6)
    assert sample["action"].shape == (4, 10)
    assert sample["action_is_pad"].tolist() == [False, True, True, True]
    assert sample["obs"]["phase"].item() == 1.0


class _DummyDINO(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 8})()
        self.encoder = torch.nn.Linear(3, 8)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pooled = image.mean(dim=(-2, -1))
        return self.encoder(pooled).unsqueeze(1).expand(-1, 4, -1)


def test_checkpoint_declares_grape_dinov3_flow_contract(tmp_path: Path) -> None:
    config = training.flow.DINOv3FlowConfig(
        state_dim=12,
        action_dim=10,
        num_cameras=2,
        state_obs_steps=2,
        image_obs_steps=2,
        chunk_size=4,
        feature_grid_size=2,
        condition_grid_size=2,
        dino_hidden_size=8,
        cond_dim=32,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dim=32,
        transformer_cond_layers=1,
        dino_path="dummy",
        image_keys=training.DEPLOYMENT_IMAGE_KEYS,
    )
    model = training.flow.DINOv3FlowPolicy(config, _DummyDINO())
    state = np.zeros((2, 12), dtype=np.float32)
    action = np.zeros((2, 10), dtype=np.float32)
    normalizer = training.common.LinearNormalizer(
        {
            "state": training.common.ArrayStats.from_array(state),
            "action": training.common.ArrayStats.from_array(action),
        }
    )
    output = tmp_path / "policy.pt"
    training.save_checkpoint(
        output,
        model,
        normalizer,
        {"dino_frozen": True, "state_fields": list(training.STATE_FIELDS)},
        epoch=1,
        global_step=2,
        validation_loss=0.5,
        ema=None,
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["contract"] == training.CONTRACT
    assert checkpoint["task"] == training.TASK
    assert checkpoint["model_type"] == "tacex_dinov3_flow_matching_bc_v2"
    assert checkpoint["policy_config"]["state_dim"] == 12
    assert not any(key.startswith("dino.") for key in checkpoint["model"])
