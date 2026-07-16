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
    assert "terminate_break_force_threshold_n: float = 6.0" in source
    assert "success_lift_height: float = 0.200" in source
    assert "scripted_lift_assist_on_contact: bool = False" in source
    assert "reset_hold_steps: int = 24" in source
    assert "scripted_lift_steps: int = 180" in source
    assert "randomize_labware_position: bool = True" in source
    assert "labware_pos_randomization_xy: tuple[float, float] = (0.020, 0.010)" in source
    assert "labware_yaw_randomization: float = 0.20" in source
    assert "SLIDE_VISUAL_DIFFUSE_COLOR" in source
    assert "SLIDE_VISUAL_OPACITY" in source
    assert "SLIDE_VISUAL_ROUGHNESS" in source
    assert "action_space = 10" in source
    assert "observation_space = 14" in source
    assert "FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG" in source
    assert "GelSightMiniCfg" in source
    assert "TiledCameraCfg" in source
    assert "ContactSensorCfg" in source
    assert "RenderCfg" in source
    assert "render=RenderCfg(enable_translucency=True)" in source
    assert "left_finger_contact_sensor = ContactSensorCfg(" in source
    assert "right_finger_contact_sensor = ContactSensorCfg(" in source
    assert 'prim_path="/World/envs/env_.*/Robot/gelpad_left"' in source
    assert 'prim_path="/World/envs/env_.*/Robot/gelpad_right"' in source
    assert 'filter_prim_paths_expr=["/World/envs/env_.*/labware"]' in source


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
    assert "torch.rand((len(env_ids), 2), dtype=root_state.dtype, device=self.device)" in source
    assert "self.labware_reset_pos_w[env_ids]" in source
    assert 'self.scene.rigid_objects["labware_support"] = self.labware_support' in source
    assert "support_state[:, 0:2] = root_state[:, 0:2]" in source
    assert "support_state[:, 3:7] = root_state[:, 3:7]" in source
    assert "self.labware_support.write_root_state_to_sim(support_state, env_ids=env_ids)" in source
    assert "self.reset_hold_remaining_steps[env_ids] = max(int(self.cfg.reset_hold_steps), 0)" in source
    assert "def _hold_labware_at_reset_pose(self):" in source
    assert "self.reset_hold_remaining_steps[hold_env_ids] -= 1" in source
    assert "def _force_non_labware_visuals_opaque(self):" in source
    assert "LabPickOpaqueRobotMaterial" in source
    assert "def _create_opaque_override_material(self, stage" in source
    assert "def _bind_opaque_material(self, prim, material):" in source
    assert "UsdShade.Tokens.strongerThanDescendants" in source
    assert "def _is_labware_visual_path(self, prim_path: str) -> bool:" in source
    assert "if self._is_labware_visual_path(prim_path):" in source
    assert 'shader.GetInput("opacity")' in source or 'for input_name in ("opacity", "opacity_constant")' in source
    assert "shader_input.Set(1.0)" in source
    assert "shader_input.Set(False)" in source
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
    assert (
        'elif self.labware_name == "slide":\n'
        "            hover_height = 0.048\n"
        "            grasp_height = 0.0006\n"
        "            lift_height = 0.25\n"
        "            close_width = 0.012"
    ) in source
    assert "def _apply_scripted_lift_assist(self, target_object_pos_b: torch.Tensor):" in source
    assert "self.labware.write_root_state_to_sim(root_state[env_ids], env_ids=env_ids)" in source
    assert "lift_progress = min(max((phase - lift_start) / max(self.cfg.scripted_lift_steps, 1), 0.0), 1.0)" in source
    assert "target_lift[self.has_touched] = grasp_height + (lift_height - grasp_height) * lift_progress" in source


def test_lab_pick_scripted_grasp_targets_physical_pad_center_and_labware_yaw():
    env_source = read(TASK_ROOT / "lab_pick_env.py")
    assert "self.gripper_center_offset_tool" in env_source
    assert "def _calibrate_gripper_center_offset(" in env_source
    assert "left_pos_w = self._robot.data.body_link_pos_w[:, self._left_finger_body_idx]" in env_source
    assert "right_pos_w = self._robot.data.body_link_pos_w[:, self._right_finger_body_idx]" in env_source
    assert "yaw_aligned_gripper_quat(" in env_source
    assert "centered_tool_target(" in env_source
    assert "center_target_b = self.initial_object_pos_b.clone()" in env_source
    assert "target_pos_b = object_pos_b.clone()" not in env_source


def test_lab_pick_env_uses_six_axis_ft_for_cafe_force_and_break_failure():
    cfg_source = read(TASK_ROOT / "lab_pick_env_cfg.py")
    env_source = read(TASK_ROOT / "lab_pick_env.py")
    assert "terminate_break_force_threshold_n: float" in cfg_source
    assert "contact_force_n_per_mm: float" in cfg_source
    assert "contact_torque_arm_m: float" in cfg_source
    assert "from isaaclab.sensors import ContactSensor" in env_source
    assert "self.left_finger_contact_sensor = ContactSensor(self.cfg.left_finger_contact_sensor)" in env_source
    assert "self.right_finger_contact_sensor = ContactSensor(self.cfg.right_finger_contact_sensor)" in env_source
    assert 'self.scene.sensors["left_finger_contact_sensor"] = self.left_finger_contact_sensor' in env_source
    assert 'self.scene.sensors["right_finger_contact_sensor"] = self.right_finger_contact_sensor' in env_source
    assert "def _contact_force_from_sensor(self, sensor: ContactSensor) -> torch.Tensor:" in env_source
    assert "def _contact_sensor_ft(self) -> torch.Tensor | None:" in env_source
    assert "def _estimate_contact_forces_from_tactile(self) -> tuple[torch.Tensor, torch.Tensor]:" in env_source
    assert "def get_cafe_ft(self) -> torch.Tensor:" in env_source
    assert "ft[:, :3]" in env_source
    assert "force_norm = torch.linalg.norm(ft[:, :3], dim=1)" in env_source
    assert "object_broken = self.has_touched & (force_norm > self.cfg.terminate_break_force_threshold_n)" in env_source
    assert "terminated = object_dropped | object_too_far | ee_outside_workspace | object_broken" in env_source
    assert '"robot0_force": self.get_cafe_ft()' in env_source
    assert "CAFE: Fx,Fy,Fz,Tx,Ty,Tz" in env_source
    get_cafe_ft_source = env_source.split("def get_cafe_ft(self) -> torch.Tensor:", maxsplit=1)[1].split(
        "\n    def ", maxsplit=1
    )[0]
    assert "body_incoming_joint_wrench_b" not in get_cafe_ft_source
    assert "contact_ft = self._contact_sensor_ft()" in get_cafe_ft_source
    assert "return self._indentation_ft()" in get_cafe_ft_source
    assert "left_force_n, right_force_n = self._estimate_contact_forces_from_tactile()" in env_source
    assert "torque_y = self.cfg.contact_torque_arm_m * (right_force_n - left_force_n)" in env_source


def test_lab_pick_env_generates_gelsight_marker2d_displacement_field():
    cfg_source = read(TASK_ROOT / "lab_pick_env_cfg.py")
    env_source = read(TASK_ROOT / "lab_pick_env.py")
    assert "marker2d_rows: int = 14" in cfg_source
    assert "marker2d_cols: int = 26" in cfg_source
    assert "marker2d_sigma: float" in cfg_source
    assert "marker2d_depth_scale: float" in cfg_source
    assert "marker2d_shear_scale: float" in cfg_source
    assert "self.last_object_pos_b" in env_source
    assert "def get_cafe_marker2d(self) -> torch.Tensor:" in env_source
    assert "torch.meshgrid" in env_source
    assert "torch.exp(-dist2 / (2.0 * self.cfg.marker2d_sigma**2))" in env_source
    assert "left_touch, right_touch = self.tactile_contact_depths()" in env_source
    assert "object_delta_b = object_pos_b - self.last_object_pos_b" in env_source
    assert "marker2d = torch.stack((dx, dy), dim=-1)" in env_source


def test_lab_pick_env_contact_sensor_ft_uses_filtered_force_vectors_and_base_frame():
    env_source = read(TASK_ROOT / "lab_pick_env.py")
    assert "sensor.data.force_matrix_w" in env_source
    assert "sensor.data.net_forces_w" in env_source
    assert "left_force_w = self._contact_force_from_sensor(self.left_finger_contact_sensor)" in env_source
    assert "right_force_w = self._contact_force_from_sensor(self.right_finger_contact_sensor)" in env_source
    assert "left_pos_w = self._robot.data.body_link_pos_w[:, self._left_finger_body_idx]" in env_source
    assert "right_pos_w = self._robot.data.body_link_pos_w[:, self._right_finger_body_idx]" in env_source
    assert "hand_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]" in env_source
    assert "torque_w = torch.cross(left_pos_w - hand_pos_w, left_force_w, dim=1)" in env_source
    assert "root_rot_b = math_utils.matrix_from_quat(math_utils.quat_inv(self._robot.data.root_link_quat_w))" in env_source
    assert "force_b = torch.bmm(root_rot_b, force_w.unsqueeze(-1)).squeeze(-1)" in env_source
    assert "contact_ft = self._contact_sensor_ft()" in env_source
    assert "if contact_ft is not None:" in env_source
    assert "return self._indentation_ft()" in env_source


def test_lab_pick_cafe_dataset_writer_and_collection_script_exist():
    writer_source = read(TASK_ROOT / "bc_dataset.py")
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


def test_lab_pick_collection_script_uses_forcecapture_cafe_record_layout():
    script_source = read(SCRIPT_ROOT / "collect_bc_dataset.py")
    assert "CafeRecordWriter" in script_source
    assert "--record_dir" in script_source
    assert "--camera_hz" in script_source
    assert "--aligned_hz" in script_source
    assert "--ft_hz" in script_source
    assert "--tracker_hz" in script_source
    assert "--failure_only" in script_source
    assert "--max_attempts" in script_source
    assert "--break_force_threshold_n" in script_source
    assert "parser.error(\"--success_only and --failure_only are mutually exclusive\")" in script_source
    assert "append_aligned_sample" in script_source
    assert "append_camera_sample" in script_source
    assert "append_ft_sample" in script_source
    assert "append_tracker_sample" in script_source
    assert "append_encoder_sample" in script_source
    assert "append_xense_sample" in script_source
    assert "record_dir / f\"record_{recorded:06d}\"" in script_source
    assert "env.wrist_camera.data.output" in script_source
    assert "def _due(next_timestamp: float, current_timestamp: float) -> bool:" in script_source
    assert "while _due(next_ft_t, timestamp)" in script_source
    assert "while _due(next_tracker_t, timestamp)" in script_source
    assert "env.get_cafe_marker2d()" in script_source
    assert "failed_attempts" in script_source
    assert 'prefix="failure_frame"' in script_source
    assert 'prefix="last_frame"' in script_source
    assert 'f"{prefix}_rgb.npy"' in script_source
    assert 'f"{prefix}_ft.npy"' in script_source
    assert 'f"{prefix}_info.txt"' in script_source
    assert "if terminated_now and not episode_failed:" in script_source
    assert "if success and not episode_failed:" in script_source
    assert "if done or success:" not in script_source
    assert "np.zeros((14, 26, 2)" not in script_source


def test_lab_pick_failed_attempt_vlm_analyzer_exists():
    analyzer_source = read(SCRIPT_ROOT / "analyze_failed_attempts.py")
    assert "failure_frame" in analyzer_source
    assert "last_frame" in analyzer_source
    assert 'f"{frame_prefix}_rgb.{suffix}"' in analyzer_source
    assert 'f"{frame_prefix}_ft.npy"' in analyzer_source
    assert 'f"{frame_prefix}_info.txt"' in analyzer_source
    assert "--frame" in analyzer_source
    assert "analysis_frame" in analyzer_source
    assert "OPENAI_API_KEY" in analyzer_source
    assert "OPENAI_API_BASE" in analyzer_source
    assert "--api_mode" in analyzer_source
    assert "chat_completions" in analyzer_source
    assert "/responses" in analyzer_source
    assert "/chat/completions" in analyzer_source
    assert "--dry_run" in analyzer_source
    assert "vlm_failure_analysis.json" in analyzer_source
    assert "failure_summary.csv" in analyzer_source
    assert "suggested_force_range_n" in analyzer_source
    assert "contact_state" in analyzer_source
    assert "force_assessment" in analyzer_source
    assert "risk_level" in analyzer_source
    assert "recommended_policy_change" in analyzer_source
    assert "recommended_next_test" in analyzer_source


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


def test_cafe_writer_reopens_existing_file_and_appends_next_demo(tmp_path):
    spec = importlib.util.spec_from_file_location("bc_dataset", TASK_ROOT / "bc_dataset.py")
    bc_dataset = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(bc_dataset)
    CafeHdf5Writer = bc_dataset.CafeHdf5Writer

    dataset_file = tmp_path / "append_existing.hdf5"

    def write_episode(writer: CafeHdf5Writer, value: float):
        for _ in range(3):
            obs = {
                "robot0_pos": torch.full((1, 10), value),
                "robot0_force": torch.full((1, 4), value),
            }
            action = torch.full((1, 10), value)
            writer.append_high_step(obs, action)
        writer.append_low_step(torch.zeros((1, 224, 224, 3), dtype=torch.uint8), torch.full((1, 10), value))
        return writer.flush_episode(
            success=False,
            labware_reset_pos_w=torch.zeros((1, 3)),
            labware_reset_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            success_only=False,
        )

    assert write_episode(CafeHdf5Writer(dataset_file, freq_ratio=3), 1.0) is True
    assert write_episode(CafeHdf5Writer(dataset_file, freq_ratio=3), 2.0) is True

    with h5py.File(dataset_file, "r") as h5:
        assert h5.attrs["num_demos"] == 2
        assert "demo_0" in h5["data"]
        assert "demo_1" in h5["data"]
        np.testing.assert_allclose(h5["data"]["demo_1"]["actions"]["high"][0], np.full((10,), 2.0))


def test_cafe_record_writer_outputs_forcecapture_cafe_directory(tmp_path):
    spec = importlib.util.spec_from_file_location("bc_dataset", TASK_ROOT / "bc_dataset.py")
    bc_dataset = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(bc_dataset)
    CafeRecordWriter = bc_dataset.CafeRecordWriter

    record_dir = tmp_path / "record_000000"
    writer = CafeRecordWriter(record_dir)

    for index in range(6):
        timestamp = index / 60.0
        sample = {
            "xyz": np.array([index, index + 1, index + 2], dtype=np.float32),
            "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "width": np.array([0.02], dtype=np.float32),
            "ft": np.full((6,), float(index), dtype=np.float32),
            "marker2d": np.full((14, 26, 2), float(index), dtype=np.float32),
            "rgb": np.full((480, 640, 3), index, dtype=np.uint8),
            "third_rgb": np.full((72, 128, 3), index + 10, dtype=np.uint8),
            "action": np.full((10,), float(index), dtype=np.float32),
        }
        writer.append_aligned_sample(timestamp, sample)
        if index % 2 == 0:
            writer.append_camera_sample(timestamp, sample["rgb"], sample["third_rgb"])
        writer.append_ft_sample(timestamp, sample["ft"])
        writer.append_tracker_sample(timestamp, sample["xyz"], sample["quat"])
        writer.append_encoder_sample(timestamp, sample["width"])
        writer.append_xense_sample(timestamp, sample["marker2d"])

    writer.flush_episode(
        success=True,
        labware_reset_pos_w=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        labware_reset_quat_w=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    expected_files = [
        "metadata.npz",
        "encoder/width.npy",
        "encoder/timestamps.npy",
        "tracker/xyz.npy",
        "tracker/quat.npy",
        "tracker/timestamps.npy",
        "ftsensor/ft.npy",
        "ftsensor/ft_compensated.npy",
        "ftsensor/timestamps.npy",
        "xense/marker2d.npy",
        "xense/marker2d_flatten.npy",
        "xense/timestamps.npy",
        "camera/color/rgb.npy",
        "camera/color/timestamps.npy",
        "camera/third/color/rgb.npy",
        "camera/third/color/timestamps.npy",
        "aligned_60Hz/xyz.npy",
        "aligned_60Hz/quat.npy",
        "aligned_60Hz/width.npy",
        "aligned_60Hz/ft.npy",
        "aligned_60Hz/marker2d.npy",
        "aligned_60Hz/rgb.npy",
        "aligned_60Hz/third_rgb.npy",
        "aligned_60Hz/action.npy",
    ]
    for relative_path in expected_files:
        assert (record_dir / relative_path).exists(), relative_path

    assert np.load(record_dir / "aligned_60Hz" / "xyz.npy").shape == (6, 3)
    assert np.load(record_dir / "aligned_60Hz" / "quat.npy").shape == (6, 4)
    assert np.load(record_dir / "aligned_60Hz" / "width.npy").shape == (6, 1)
    assert np.load(record_dir / "aligned_60Hz" / "ft.npy").shape == (6, 6)
    assert np.load(record_dir / "aligned_60Hz" / "marker2d.npy").shape == (6, 14 * 26 * 2)
    assert np.load(record_dir / "aligned_60Hz" / "rgb.npy").shape == (6, 480, 640, 3)
    assert np.load(record_dir / "aligned_60Hz" / "third_rgb.npy").shape == (6, 72, 128, 3)
    assert np.load(record_dir / "ftsensor" / "ft.npy").shape == (6, 6)
    assert np.load(record_dir / "camera" / "color" / "timestamps.npy").shape == (3,)
    assert np.load(record_dir / "camera" / "third" / "color" / "rgb.npy").shape == (3, 72, 128, 3)
    assert np.load(record_dir / "camera" / "third" / "color" / "timestamps.npy").shape == (3,)
    assert writer.camera_rgb == []
    assert writer.third_camera_rgb == []

    metadata = np.load(record_dir / "metadata.npz")
    assert bool(metadata["success"]) is True
    np.testing.assert_allclose(metadata["labware_reset_pos_w"], [0.1, 0.2, 0.3])
