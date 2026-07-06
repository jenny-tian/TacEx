import importlib.util
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = ROOT / "source" / "tacex_tasks" / "tacex_tasks" / "lab_pick"
SCRIPT_ROOT = ROOT / "scripts" / "demos" / "lab_pick"


def read(path: Path) -> str:
    return path.read_text()


def test_lab_pick_package_registers_three_labware_tasks():
    source = read(TASK_ROOT / "__init__.py")
    assert 'id="TacEx-LabPick-Slide-Direct-v0"' in source
    assert 'id="TacEx-LabPick-Coverslip-Direct-v0"' in source
    assert 'id="TacEx-LabPick-Cup-Direct-v0"' in source
    assert 'entry_point=f"{__name__}.lab_pick_env:LabPickEnv"' in source
    assert '"env_cfg_entry_point": LabPickSlideEnvCfg' in source
    assert '"env_cfg_entry_point": LabPickCoverslipEnvCfg' in source
    assert '"env_cfg_entry_point": LabPickCupEnvCfg' in source


def test_lab_pick_cfg_defines_scene_assets_randomization_and_termination_thresholds():
    source = read(TASK_ROOT / "lab_pick_env_cfg.py")
    assert "class LabPickEnvCfg(DirectRLEnvCfg):" in source
    assert "class LabPickSlideEnvCfg(LabPickEnvCfg):" in source
    assert "class LabPickCoverslipEnvCfg(LabPickEnvCfg):" in source
    assert "class LabPickCupEnvCfg(LabPickEnvCfg):" in source
    assert 'labware_name = "slide"' in source
    assert 'labware_name = "coverslip"' in source
    assert 'labware_name = "cup"' in source
    assert "terminate_object_drop_height: float = 0.010" in source
    assert "terminate_object_xy_distance: float = 0.30" in source
    assert "success_lift_height: float = 0.030" in source
    assert "randomize_labware_position: bool = True" in source
    assert "labware_pos_randomization_xy: tuple[float, float] = (0.020, 0.010)" in source
    assert "labware_yaw_randomization: float = 0.20" in source
    assert "action_space = 10" in source
    assert "observation_space = 14" in source
    assert "FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG" in source
    assert "GelSightMiniCfg" in source
    assert "TiledCameraCfg" in source


def test_lab_pick_env_implements_dones_reset_randomization_and_cafe_io():
    source = read(TASK_ROOT / "lab_pick_env.py")
    assert "class LabPickEnv(DirectRLEnv):" in source
    assert "def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:" in source
    assert "terminated = object_dropped | object_too_far | ee_outside_workspace" in source
    assert "time_out = self.episode_length_buf >= self.max_episode_length - 1" in source
    assert "return terminated, time_out" in source
    assert "def _reset_idx(self, env_ids: torch.Tensor | None):" in source
    assert "self.labware.write_root_state_to_sim(root_state, env_ids=env_ids)" in source
    assert "self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)" in source
    assert "self.gsmini_left.reset(env_ids=env_ids)" in source
    assert "self.gsmini_right.reset(env_ids=env_ids)" in source
    assert "self.step_count[env_ids] = 0" in source
    assert "torch.rand((len(env_ids), 2), device=self.device)" in source
    assert "self.labware_reset_pos_w[env_ids]" in source
    assert "def _quat_to_rot6d(" in source
    assert "def get_cafe_observation(self) -> dict[str, torch.Tensor]:" in source
    assert "def get_cafe_action(self) -> torch.Tensor:" in source
    assert "def get_cafe_image(self) -> torch.Tensor:" in source
    assert '"robot0_pos"' in source
    assert '"robot0_force"' in source
    assert "xyz(3) + rot6d(6) + gripper_width(1)" in source
    assert (
        "touch_left, touch_right = self.tactile_contact_depths()\n"
        "        touched = (touch_left > self.cfg.tactile_threshold_mm) | "
        "(touch_right > self.cfg.tactile_threshold_mm)"
    ) in source


def test_lab_pick_cafe_dataset_writer_and_collection_script_exist():
    writer_source = read(TASK_ROOT / "bc_dataset.py")
    script_source = read(SCRIPT_ROOT / "collect_bc_dataset.py")
    assert "class CafeHdf5Writer:" in writer_source
    assert "def append_high_step(" in writer_source
    assert "def append_low_step(" in writer_source
    assert "def flush_episode(" in writer_source
    assert "h5py.File" in writer_source
    assert 'demo_group = data_group.create_group(f"demo_{self.episode_index}")' in writer_source
    assert 'actions_group = demo_group.create_group("actions")' in writer_source
    assert 'actions_group.create_dataset("high"' in writer_source
    assert 'actions_group.create_dataset("low"' in writer_source
    assert 'obs_group.create_dataset("robot0_pos"' in writer_source
    assert 'obs_group.create_dataset("robot0_force"' in writer_source
    assert 'obs_group.create_dataset("robot0_image"' in writer_source
    assert 'h5.attrs["freq_ratio"] = self.freq_ratio' in writer_source
    assert "n_high = n_low * self.freq_ratio" in writer_source
    assert "env.get_cafe_observation()" in script_source
    assert "env.get_cafe_action()" in script_source
    assert "env.get_cafe_image()" in script_source
    assert "--num_demos" in script_source
    assert "--dataset_file" in script_source
    assert "--success_only" in script_source


def test_launch_scripts_are_thin_and_import_shared_env():
    scripted = read(SCRIPT_ROOT / "pick_labware.py")
    keyboard = read(SCRIPT_ROOT / "pick_labware_keyboard.py")
    for source in (scripted, keyboard):
        assert "from tacex_tasks.lab_pick.lab_pick_env import LabPickEnv" in source
        assert "from tacex_tasks.lab_pick.lab_pick_env_cfg import LabPickEnvCfg" in source
        assert "class LabPickEnv(" not in source
        assert "class LabPickEnvCfg(" not in source
        assert "AppLauncher.add_app_launcher_args(parser)" in source


def test_cafe_writer_outputs_forcecapture_compatible_hdf5(tmp_path):
    spec = importlib.util.spec_from_file_location("bc_dataset", TASK_ROOT / "bc_dataset.py")
    bc_dataset = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(bc_dataset)
    CafeHdf5Writer = bc_dataset.CafeHdf5Writer

    dataset_file = tmp_path / "lab_pick_bc.hdf5"
    writer = CafeHdf5Writer(dataset_file, freq_ratio=3)

    for index in range(7):
        obs = {
            "robot0_pos": torch.full((1, 10), float(index)),
            "robot0_force": torch.full((1, 4), float(index)),
        }
        action = torch.full((1, 10), float(index))
        writer.append_high_step(obs, action)
        if index % 3 == 0:
            image = torch.full((1, 224, 224, 3), index, dtype=torch.uint8)
            writer.append_low_step(image, action)

    wrote = writer.flush_episode(
        success=True,
        labware_reset_pos_w=torch.tensor([[0.1, 0.2, 0.3]]),
        labware_reset_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        success_only=True,
    )

    assert wrote is True
    with h5py.File(dataset_file, "r") as h5:
        assert h5.attrs["num_demos"] == 1
        assert h5.attrs["include_images"] == np.bool_(True)
        assert h5.attrs["freq_ratio"] == 3
        assert h5.attrs["high_freq_obs_keys"] == "robot0_pos,robot0_force"
        assert h5.attrs["low_freq_obs_keys"] == "robot0_image"
        assert h5.attrs["high_freq_action_key"] == "high"
        assert h5.attrs["low_freq_action_key"] == "low"

        demo = h5["data"]["demo_0"]
        assert demo.attrs["length_high"] == 6
        assert demo.attrs["length_low"] == 2
        assert demo.attrs["freq_ratio"] == 3
        assert demo.attrs["success"] == np.bool_(True)
        np.testing.assert_allclose(demo.attrs["labware_reset_pos_w"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(demo.attrs["labware_reset_quat_w"], [1.0, 0.0, 0.0, 0.0])

        assert demo["actions"]["high"].shape == (6, 10)
        assert demo["actions"]["high"].dtype == np.float32
        assert demo["actions"]["low"].shape == (2, 10)
        assert demo["actions"]["low"].dtype == np.float32
        assert demo["obs"]["robot0_pos"].shape == (6, 10)
        assert demo["obs"]["robot0_pos"].dtype == np.float32
        assert demo["obs"]["robot0_force"].shape == (6, 4)
        assert demo["obs"]["robot0_force"].dtype == np.float32
        assert demo["obs"]["robot0_image"].shape == (2, 224, 224, 3)
        assert demo["obs"]["robot0_image"].dtype == np.uint8
