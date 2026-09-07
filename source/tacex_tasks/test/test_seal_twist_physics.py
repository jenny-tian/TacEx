import math
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_PHYSICS_PATH = Path(__file__).parents[1] / "tacex_tasks" / "seal_twist" / "physics.py"
_SPEC = importlib.util.spec_from_file_location("seal_twist_physics", _PHYSICS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PHYSICS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PHYSICS)
in_torque_window = _PHYSICS.in_torque_window
seal_compression_from_angle = _PHYSICS.seal_compression_from_angle
seal_torque_from_angle = _PHYSICS.seal_torque_from_angle

_DATASET_PATH = Path(__file__).parents[1] / "tacex_tasks" / "seal_twist" / "dataset.py"
_DATASET_SPEC = importlib.util.spec_from_file_location("seal_twist_dataset", _DATASET_PATH)
assert _DATASET_SPEC is not None and _DATASET_SPEC.loader is not None
_DATASET = importlib.util.module_from_spec(_DATASET_SPEC)
_DATASET_SPEC.loader.exec_module(_DATASET)

_POLICY_PATH = Path(__file__).parents[1] / "tacex_tasks" / "seal_twist" / "policy.py"
_POLICY_SPEC = importlib.util.spec_from_file_location("seal_twist_policy", _POLICY_PATH)
assert _POLICY_SPEC is not None and _POLICY_SPEC.loader is not None
_POLICY = importlib.util.module_from_spec(_POLICY_SPEC)
_POLICY_SPEC.loader.exec_module(_POLICY)

_FLOW_POLICY_PATH = (
    Path(__file__).parents[1] / "tacex_tasks" / "seal_twist" / "flow_policy.py"
)
_FLOW_POLICY_SPEC = importlib.util.spec_from_file_location(
    "seal_twist_flow_policy", _FLOW_POLICY_PATH
)
assert _FLOW_POLICY_SPEC is not None and _FLOW_POLICY_SPEC.loader is not None
_FLOW_POLICY = importlib.util.module_from_spec(_FLOW_POLICY_SPEC)
sys.modules[_FLOW_POLICY_SPEC.name] = _FLOW_POLICY
_FLOW_POLICY_SPEC.loader.exec_module(_FLOW_POLICY)


def test_seal_torque_has_running_and_compression_regions():
    angle = torch.tensor([0.0, 0.25, 0.50, 1.50, 2.50])
    torque = seal_torque_from_angle(
        angle,
        engagement_angle_rad=0.50,
        running_torque_nm=0.08,
        compression_stiffness_nm_per_rad=0.25,
    )
    torch.testing.assert_close(torque, torch.tensor([0.0, 0.08, 0.08, 0.33, 0.58]))


def test_seal_compression_is_zero_then_bounded():
    angle = torch.tensor([0.2, 0.5, 1.5, 20.0])
    compression = seal_compression_from_angle(
        angle,
        engagement_angle_rad=0.5,
        thread_pitch_m_per_turn=0.004,
        maximum_compression_m=0.0012,
    )
    assert compression[0] == 0.0
    assert compression[1] == 0.0
    assert math.isclose(float(compression[-1]), 0.0012, rel_tol=0.0, abs_tol=1.0e-7)


def test_torque_window_is_inclusive():
    torque = torch.tensor([0.349, 0.35, 0.42, 0.50, 0.501])
    assert in_torque_window(torque, 0.35, 0.50).tolist() == [False, True, True, True, False]


def test_force_map_preserves_vector_magnitude():
    force = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    maps = _DATASET.force_vectors_to_maps(force)
    assert maps.shape == (2, 32, 32)
    np.testing.assert_allclose(maps.sum(axis=(1, 2)), np.array([5.0, 0.0]), rtol=1.0e-6)


def test_bc_trajectory_observation_has_declared_state_and_tactile_width():
    length = 3
    trajectory = {
        "robot0_pos": np.zeros((length, 10), dtype=np.float32),
        "cap_pose_b": np.zeros((length, 7), dtype=np.float32),
        "turn_angle_rad": np.zeros(length, dtype=np.float32),
        "seal_torque_nm": np.zeros(length, dtype=np.float32),
        "seal_compression_m": np.zeros(length, dtype=np.float32),
        "finger_force_n": np.ones((length, 2), dtype=np.float32),
        "step": np.arange(length, dtype=np.int32),
    }
    observation = _POLICY.trajectory_policy_observation(
        trajectory, max_episode_steps=900
    )
    assert observation.shape == (length, _POLICY.POLICY_OBSERVATION_DIM)
    np.testing.assert_allclose(observation[:, -3:-1], 1.0)


def test_bc_policy_maps_declared_observation_to_cafe_action():
    model = _POLICY.SealTwistBCPolicy(hidden_dim=32, depth=2)
    output = model(torch.zeros((4, _POLICY.POLICY_OBSERVATION_DIM)))
    assert output.shape == (4, _POLICY.POLICY_ACTION_DIM)


def test_flow_observation_contains_exact_stored_6d_wrench():
    length = 3
    wrench = np.arange(length * 6, dtype=np.float32).reshape(length, 6)
    action = np.zeros((length, 10), dtype=np.float32)
    action[:, 9] = np.array([0.040, 0.035, 0.030], dtype=np.float32)
    robot0_pos = np.zeros((length, 10), dtype=np.float32)
    robot0_pos[:, 9] = action[:, 9]
    trajectory = {
        "robot0_pos": robot0_pos,
        "action": action,
        "wrench": wrench,
        "cap_pose_b": np.zeros((length, 7), dtype=np.float32),
        "turn_angle_rad": np.zeros(length, dtype=np.float32),
        "seal_torque_nm": np.zeros(length, dtype=np.float32),
        "seal_compression_m": np.zeros(length, dtype=np.float32),
        "finger_force_n": np.ones((length, 2), dtype=np.float32),
        "step": np.arange(length, dtype=np.int32),
    }
    observation = _FLOW_POLICY.trajectory_flow_observation(
        trajectory, max_episode_steps=900
    )
    assert observation.shape == (length, _FLOW_POLICY.FLOW_OBSERVATION_DIM)
    np.testing.assert_array_equal(
        observation[:, _FLOW_POLICY.WRENCH_OBSERVATION_SLICE], wrench
    )
    np.testing.assert_allclose(
        observation[:, 9], np.array([0.040, 0.040, 0.035], dtype=np.float32)
    )
    assert observation[:, _FLOW_POLICY.TOOL_POSITION_OBSERVATION_SLICE].shape == (
        length,
        3,
    )
    assert observation[:, _FLOW_POLICY.RELATIVE_CAP_POSITION_OBSERVATION_SLICE].shape == (
        length,
        3,
    )


def test_flow_policy_loss_and_decode_shapes():
    config = _FLOW_POLICY.SealTwistFlowConfig(
        horizon=4, hidden_dim=32, depth=2, time_embedding_dim=16
    )
    model = _FLOW_POLICY.SealTwistFlowPolicy(config)
    observation = torch.randn(5, _FLOW_POLICY.FLOW_OBSERVATION_DIM)
    target = torch.randn(5, config.horizon, _FLOW_POLICY.FLOW_ACTION_DIM)
    loss = model.flow_matching_loss(
        observation,
        target,
        noise=torch.zeros_like(target),
        time=torch.full((5,), 0.5),
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    decoded = model.decode(
        observation, torch.zeros_like(target), integration_steps=2, solver="heun"
    )
    assert decoded.shape == target.shape
