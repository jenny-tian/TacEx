import importlib.util
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CONVERTER_PATH = ROOT / "bc_policy" / "sim_robot" / "scripts" / "convert_records_to_hdf5.py"


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_records_to_hdf5", CONVERTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_record(
    root: Path,
    index: int,
    *,
    success: bool,
    value: int,
    observation_timing: str | None = None,
) -> Path:
    record = root / f"record_{index:06d}"
    aligned = record / "aligned"
    aligned.mkdir(parents=True)
    length = 3
    np.save(aligned / "xyz.npy", np.full((length, 3), value, dtype=np.float32))
    np.save(aligned / "quat.npy", np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (length, 1)))
    np.save(aligned / "width.npy", np.full((length, 1), 0.02, dtype=np.float32))
    np.save(aligned / "ft.npy", np.full((length, 6), value, dtype=np.float32))
    np.save(aligned / "rgb.npy", np.full((length, 224, 224, 3), value, dtype=np.uint8))
    np.save(aligned / "rgb_third.npy", np.full((length, 224, 224, 3), value + 1, dtype=np.uint8))
    action = np.arange(length * 10, dtype=np.float32).reshape(length, 10) + value * 100
    np.save(aligned / "action.npy", action)
    metadata = {
        "success": np.asarray(success, dtype=np.bool_),
        "labware_reset_pos_w": np.array([0.5, 0.0, 0.01], dtype=np.float32),
        "labware_reset_quat_w": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    if observation_timing is not None:
        metadata["observation_timing"] = np.asarray(observation_timing)
    np.savez(record / "metadata.npz", **metadata)
    return record


def test_convert_records_to_flow_matching_hdf5(tmp_path):
    converter = load_converter()
    records = tmp_path / "records"
    write_record(records, 0, success=True, value=3)
    write_record(records, 1, success=False, value=7)
    output = tmp_path / "flow_matching.hdf5"

    converted = converter.convert_records(
        records,
        output,
        success_only=True,
        include_third_camera=True,
    )

    assert converted == 1
    with h5py.File(output, "r") as h5:
        assert h5.attrs["num_demos"] == 1
        assert h5.attrs["freq_ratio"] == 1
        assert h5.attrs["fps"] == 60
        demo = h5["data"]["demo_0"]
        assert demo.attrs["success"] == np.bool_(True)
        assert demo.attrs["source_record"] == "record_000000"
        assert demo.attrs["action_alignment"] == "next-action"
        assert demo["actions"]["high"].shape == (2, 10)
        assert demo["actions"]["low"].shape == (2, 10)
        assert demo["obs"]["robot0_pos"].shape == (2, 10)
        assert demo["obs"]["robot0_force"].shape == (2, 6)
        assert demo["obs"]["robot0_image"].shape == (2, 224, 224, 3)
        assert demo["obs"]["robot0_image_third"].shape == (2, 224, 224, 3)
        np.testing.assert_allclose(demo["actions"]["high"][0], np.arange(10, 20) + 300)
        np.testing.assert_allclose(
            demo["obs"]["robot0_pos"][0],
            [3.0, 3.0, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.02],
        )
        np.testing.assert_allclose(demo.attrs["labware_reset_pos_w"], [0.5, 0.0, 0.01])



def test_convert_pre_action_records_keep_same_index(tmp_path):
    converter = load_converter()
    records = tmp_path / "records"
    write_record(records, 0, success=True, value=3, observation_timing="pre_action")
    output = tmp_path / "flow_matching_pre_action.hdf5"

    converter.convert_records(records, output, success_only=True)

    with h5py.File(output, "r") as h5:
        demo = h5["data"]["demo_0"]
        assert demo.attrs["action_alignment"] == "same-index"
        assert demo["actions"]["high"].shape == (3, 10)
        np.testing.assert_allclose(demo["actions"]["high"][0], np.arange(10) + 300)


def test_phase_conditioning_appends_normalized_progress(tmp_path):
    import sys

    policy_root = str(ROOT / "bc_policy")
    if policy_root not in sys.path:
        sys.path.insert(0, policy_root)
    from sim_robot.data.sequence_dataset import SimRobotHDF5SequenceDataset, compute_normalizer

    converter = load_converter()
    records = tmp_path / "records"
    write_record(records, 0, success=True, value=3, observation_timing="pre_action")
    output = tmp_path / "phase.hdf5"
    converter.convert_records(records, output, success_only=True)

    normalizer = compute_normalizer(output, np.asarray([0]), include_phase=True)
    dataset = SimRobotHDF5SequenceDataset(
        output,
        np.asarray([0]),
        normalizer,
        n_state_obs_steps=2,
        n_image_obs_steps=2,
        n_action_steps=2,
        include_phase=True,
    )
    sample = dataset[1]

    assert dataset.robot0_pos_dim == 11
    np.testing.assert_allclose(sample["obs"]["robot0_pos"][:, -1].numpy(), [-1.0, 0.0])


def test_phase_conditioned_runner_appends_phase():
    import sys
    from types import SimpleNamespace

    policy_root = str(ROOT / "bc_policy")
    if policy_root not in sys.path:
        sys.path.insert(0, policy_root)
    from sim_robot.deployment.policy_runner import SimActionChunkPolicyRunner

    runner = SimActionChunkPolicyRunner.__new__(SimActionChunkPolicyRunner)
    runner.include_phase = True
    runner.config = SimpleNamespace(robot0_pos_dim=11)

    state = runner._prepare_state(np.zeros(10, dtype=np.float32), phase=0.25)
    assert state.shape == (11,)
    assert state[-1] == np.float32(0.25)


def test_multi_camera_dataset_returns_wrist_and_third_images(tmp_path):
    import sys

    policy_root = str(ROOT / "bc_policy")
    if policy_root not in sys.path:
        sys.path.insert(0, policy_root)
    from sim_robot.data.sequence_dataset import SimRobotHDF5SequenceDataset, compute_normalizer

    converter = load_converter()
    records = tmp_path / "records"
    write_record(records, 0, success=True, value=3, observation_timing="pre_action")
    output = tmp_path / "multi_camera.hdf5"
    converter.convert_records(records, output, success_only=True, include_third_camera=True)

    image_keys = ("robot0_image", "robot0_image_third")
    normalizer = compute_normalizer(output, np.asarray([0]), image_key=image_keys, include_phase=True)
    dataset = SimRobotHDF5SequenceDataset(
        output,
        np.asarray([0]),
        normalizer,
        n_state_obs_steps=2,
        n_image_obs_steps=2,
        n_action_steps=2,
        image_key=image_keys,
        include_phase=True,
    )
    sample = dataset[1]

    assert set(sample["obs"]) == {"robot0_pos", "robot0_image", "robot0_image_third"}
    assert sample["obs"]["robot0_image"].shape == (2, 3, 224, 224)
    assert sample["obs"]["robot0_image_third"].shape == (2, 3, 224, 224)
    assert float(sample["obs"]["robot0_image"].mean()) != float(sample["obs"]["robot0_image_third"].mean())


def test_visual_xy_auxiliary_loss_uses_image_tokens():
    import torch

    from sim_robot.policy.flow_matching_policy import SimFlowMatchingConfig, SimFlowMatchingPolicy

    config = SimFlowMatchingConfig(
        robot0_pos_dim=11,
        action_dim=10,
        n_state_obs_steps=2,
        n_image_obs_steps=2,
        image_keys=("robot0_image", "robot0_image_third"),
        n_action_steps=4,
        image_feature_dim=16,
        obs_feature_dim=16,
        transformer_layers=1,
        transformer_heads=2,
        transformer_embedding_dim=16,
        transformer_cond_layers=1,
        visual_xy_loss_weight=2.0,
    )
    model = SimFlowMatchingPolicy(config)
    batch = {
        "obs": {
            "robot0_pos": torch.zeros(2, 2, 11),
            "robot0_image": torch.zeros(2, 2, 3, 32, 32),
            "robot0_image_third": torch.ones(2, 2, 3, 32, 32),
        },
        "action": torch.zeros(2, 4, 10),
    }
    losses = model.compute_loss(batch)
    assert set(losses) == {"loss", "flow_loss", "visual_xy_loss"}
    torch.testing.assert_close(
        losses["loss"],
        losses["flow_loss"] + 2.0 * losses["visual_xy_loss"],
    )
    result = model.predict_action(batch["obs"], num_inference_steps=1)
    assert result["visual_xy"] is not None
    assert result["visual_xy"].shape == (2, 2)

    model.eval()
    initial_noise = torch.randn(2, 4, 10)
    first = model.predict_action(batch["obs"], num_inference_steps=1, initial_noise=initial_noise)
    second = model.predict_action(batch["obs"], num_inference_steps=1, initial_noise=initial_noise)
    torch.testing.assert_close(first["action"], second["action"])


def test_runner_locks_visual_xy_at_phase_threshold_and_reset_clears_lock():
    import sys
    from types import SimpleNamespace

    import torch

    policy_root = str(ROOT / "bc_policy")
    if policy_root not in sys.path:
        sys.path.insert(0, policy_root)
    from sim_robot.deployment.policy_runner import SimActionChunkPolicyRunner

    class SequencePolicy:
        def __init__(self):
            self.visual_xy = iter(([0.1, 0.2], [0.3, 0.4], [0.8, 0.9], [0.6, 0.7]))

        def predict_action(self, *_args, **_kwargs):
            return {
                "action": torch.zeros(1, 2, 10),
                "visual_xy": torch.tensor([next(self.visual_xy)], dtype=torch.float32),
            }

    class IdentityNormalizer:
        @staticmethod
        def unnormalize_numpy(_key, value):
            return value

    runner = SimActionChunkPolicyRunner.__new__(SimActionChunkPolicyRunner)
    runner.model = SequencePolicy()
    runner.normalizer = IdentityNormalizer()
    runner.generator = None
    runner.num_inference_steps = 1
    runner.visual_xy_lock_phase = 0.30
    runner.current_phase = 0.20
    runner.locked_visual_xy = None
    runner.build_model_obs = lambda: {}
    runner.state_history = SimpleNamespace(clear=lambda: None)
    runner.image_history = {}

    before_lock = runner.predict_action_chunk()
    np.testing.assert_allclose(before_lock[:, :2], [[0.1, 0.2], [0.1, 0.2]])
    assert runner.locked_visual_xy is None

    runner.current_phase = 0.30
    at_lock = runner.predict_action_chunk()
    np.testing.assert_allclose(at_lock[:, :2], [[0.3, 0.4], [0.3, 0.4]])
    np.testing.assert_allclose(runner.locked_visual_xy, [0.3, 0.4])

    runner.current_phase = 0.80
    after_lock = runner.predict_action_chunk()
    np.testing.assert_allclose(after_lock[:, :2], [[0.3, 0.4], [0.3, 0.4]])

    runner.reset()
    assert runner.locked_visual_xy is None
    runner.current_phase = 0.10
    after_reset = runner.predict_action_chunk()
    np.testing.assert_allclose(after_reset[:, :2], [[0.6, 0.7], [0.6, 0.7]])




def _load_dsrl_wrapper(monkeypatch):
    import sys
    import types

    module_names = (
        "lerobot",
        "lerobot.policies",
        "lerobot.policies.diffusion",
        "lerobot.policies.diffusion.modeling_diffusion",
        "lerobot.policies.factory",
    )
    modules = {name: types.ModuleType(name) for name in module_names}
    modules["lerobot.policies.diffusion.modeling_diffusion"].DiffusionPolicy = type("DiffusionPolicy", (), {})
    modules["lerobot.policies.factory"].make_pre_post_processors = lambda *_args, **_kwargs: (None, None)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    dsrl_root = str(ROOT / "scripts" / "reinforcement_learning" / "dsrl")
    if dsrl_root not in sys.path:
        sys.path.insert(0, dsrl_root)
    from lab_pick_dsrl_wrapper import LabPickDSRLWrapper

    return LabPickDSRLWrapper


def test_dsrl_episode_metrics_are_not_terminal_step_frequency(monkeypatch):
    import types

    import torch

    LabPickDSRLWrapper = _load_dsrl_wrapper(monkeypatch)
    wrapper = LabPickDSRLWrapper.__new__(LabPickDSRLWrapper)
    wrapper._completed_episodes = 0
    wrapper._successful_episodes = 0
    wrapper._broken_episodes = 0
    wrapper.action_mode = "noise"
    wrapper._curriculum_progress = 0.0
    wrapper.env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            cfg=types.SimpleNamespace(
                labware_pos_randomization_xy=(0.05, 0.05),
                labware_yaw_randomization=0.5235987756,
            )
        )
    )

    ongoing = wrapper._add_episode_metrics(
        {"log": {"LabPick/success_terminal_step": torch.tensor([0.0])}},
        episode_done=False,
    )
    assert ongoing["log"]["LabPick/episode_success_rate"] == 0.0
    assert ongoing["log"]["LabPick/completed_episodes"] == 0.0

    success = wrapper._add_episode_metrics(
        {"log": {"LabPick/success_terminal_step": torch.tensor([1.0])}},
        episode_done=True,
    )
    assert success["log"]["LabPick/episode_success_rate"] == 1.0
    assert success["log"]["LabPick/episode_broken_rate"] == 0.0
    assert success["log"]["LabPick/completed_episodes"] == 1.0

    failure = wrapper._add_episode_metrics(
        {"log": {"LabPick/success_terminal_step": 0.0, "LabPick/broken_terminal_step": 1.0}},
        episode_done=True,
    )
    assert failure["log"]["LabPick/episode_success_rate"] == 0.5
    assert failure["log"]["LabPick/episode_broken_rate"] == 0.5
    assert failure["log"]["LabPick/completed_episodes"] == 2.0


def test_dsrl_flow_residual_offsets_xyz_and_only_closed_width(monkeypatch):
    import torch

    LabPickDSRLWrapper = _load_dsrl_wrapper(monkeypatch)
    wrapper = LabPickDSRLWrapper.__new__(LabPickDSRLWrapper)
    wrapper.residual_position_scale_m = (0.03, 0.02, 0.01)
    wrapper.residual_width_scale_m = 0.002

    action_chunk = torch.zeros(3, 10)
    action_chunk[:, 9] = torch.tensor([0.040, 0.037, 0.0065])
    original = action_chunk.clone()
    adjusted = wrapper._apply_flow_residual(
        action_chunk,
        torch.tensor([[1.0, -0.5, 0.5, -1.0]]),
    )

    torch.testing.assert_close(
        adjusted[:, :3],
        torch.tensor([[0.03, -0.01, 0.005]]).expand(3, -1),
    )
    torch.testing.assert_close(adjusted[:, 9], torch.tensor([0.040, 0.035, 0.006]))
    torch.testing.assert_close(action_chunk, original)


def test_dsrl_residual_penalty_anchors_frozen_bc_prior(monkeypatch):
    import torch

    LabPickDSRLWrapper = _load_dsrl_wrapper(monkeypatch)
    wrapper = LabPickDSRLWrapper.__new__(LabPickDSRLWrapper)
    wrapper.action_mode = "residual"
    wrapper.residual_penalty_scale = 4.0
    info = {"log": {}}
    result = (
        torch.zeros(1, 23),
        torch.tensor([10.0]),
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
        info,
    )

    adjusted = wrapper._apply_residual_penalty(result, torch.ones(1, 4))

    torch.testing.assert_close(adjusted[1], torch.tensor([6.0]))
    torch.testing.assert_close(info["dsrl/residual_penalty"], torch.tensor(4.0))
    torch.testing.assert_close(info["log"]["LabPick/residual_penalty"], torch.tensor(4.0))


def test_dsrl_randomization_curriculum_interpolates_and_saturates(monkeypatch):
    import math
    import types

    LabPickDSRLWrapper = _load_dsrl_wrapper(monkeypatch)
    wrapper = LabPickDSRLWrapper.__new__(LabPickDSRLWrapper)
    wrapper.curriculum_steps = 100
    wrapper.curriculum_start_xy_m = (0.05, 0.05)
    wrapper.curriculum_end_xy_m = (0.10, 0.10)
    wrapper.curriculum_start_yaw_rad = math.radians(30.0)
    wrapper.curriculum_end_yaw_rad = math.radians(45.0)
    cfg = types.SimpleNamespace(
        randomize_labware_position=False,
        labware_pos_randomization_xy=(0.0, 0.0),
        labware_yaw_randomization=0.0,
    )
    wrapper.env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(cfg=cfg))

    wrapper._outer_steps = 50
    wrapper._update_randomization_curriculum()
    assert wrapper._curriculum_progress == 0.5
    assert cfg.randomize_labware_position is True
    assert cfg.labware_pos_randomization_xy == (0.07500000000000001, 0.07500000000000001)
    assert math.isclose(cfg.labware_yaw_randomization, math.radians(37.5))

    wrapper._outer_steps = 200
    wrapper._update_randomization_curriculum()
    assert wrapper._curriculum_progress == 1.0
    assert cfg.labware_pos_randomization_xy == (0.10, 0.10)
    assert math.isclose(cfg.labware_yaw_randomization, math.radians(45.0))


def test_old_policy_config_defaults_to_raw_image_range():
    from sim_robot.policy.flow_matching_policy import SimFlowMatchingConfig

    config = SimFlowMatchingConfig.from_dict({})
    assert config.image_normalization == "none"
