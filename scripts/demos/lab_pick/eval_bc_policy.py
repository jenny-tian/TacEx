from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Pick labware with a Franka/GelSight gripper and two RGB-D cameras.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument(
    "--labware",
    choices=("slide", "coverslip", "cup"),
    default="slide",
    help="Object to spawn and pick: microscope slide, cover slip, or simplified glass cup proxy.",
)
parser.add_argument("--duration", type=float, default=8.0, help="Simulation duration in seconds.")
parser.add_argument("--save_camera_images", action="store_true", help="Save RGB images from both cameras every second.")
parser.add_argument("--save_tactile_images", action="store_true", help="Save left/right GelSight tactile RGB images.")
parser.add_argument("--record_video", action="store_true", help="Record third-person camera video to an mp4 file.")
parser.add_argument(
    "--video_camera",
    choices=("third", "wrist", "viewer", "tactile_left", "tactile_right"),
    default="third",
    help="Camera source for mp4.",
)
parser.add_argument("--video_every_n_steps", type=int, default=4, help="Record one video frame every N sim steps.")
parser.add_argument("--video_fps", type=int, default=30, help="Output video FPS.")
parser.add_argument("--print_state_interval", type=int, default=60, help="Print robot/object state every N sim steps.")
parser.add_argument("--collect_dataset", action="store_true", help="Collect BC demonstrations into an HDF5 file.")
parser.add_argument("--num_demos", type=int, default=50, help="Number of successful demonstrations to collect.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="/home/gtq/TacEx/datasets/lab_pick_slide_bc.hdf5",
    help="Output HDF5 file for collected demonstrations.",
)
parser.add_argument(
    "--dataset_sample_interval_s",
    type=float,
    default=0.05,
    help="Dataset frame interval in seconds. Default 0.05s stores 20 Hz data from a 120 Hz simulation.",
)
parser.add_argument("--success_lift_height", type=float, default=0.2, help="Lift height in meters required to accept a collected demo.")
parser.add_argument("--success_hold_steps", type=int, default=60, help="Consecutive sim steps the lift must remain stable.")
parser.add_argument(
    "--success_gripper_distance",
    type=float,
    default=0.08,
    help="Maximum object-to-tool distance in meters while checking stable lift success.",
)
parser.add_argument("--episode_steps", type=int, default=960, help="Maximum steps per collected demonstration.")
parser.add_argument(
    "--max_attempts",
    type=int,
    default=None,
    help="Optional maximum collection attempts before stopping. Default: unlimited, stop after --num_demos demos.",
)
parser.add_argument("--write_failed", action="store_true", help="Also write failed attempts to the dataset.")
parser.add_argument("--randomize_labware", action="store_true", default=True, help="Randomize labware pose on reset.")
parser.add_argument("--labware_random_x", type=float, default=0.06, help="Uniform randomization range along table x.")
parser.add_argument("--labware_random_y", type=float, default=0.04, help="Uniform randomization range along table y.")
parser.add_argument("--labware_random_yaw", type=float, default=0.25, help="Uniform randomization range for yaw in radians.")
parser.add_argument(
    "--instruction",
    type=str,
    default="pick up the transparent labware",
    help="Task instruction stored with each demonstration for instruction-conditioned BC.",
)
parser.add_argument("--task_id", type=int, default=0, help="Integer task id stored per frame and per demonstration.")
parser.add_argument("--preview_demos", type=int, default=5, help="Save videos for this many recorded demonstrations.")
parser.add_argument(
    "--preview_dir",
    type=str,
    default="/home/gtq/TacEx/logs/lab_pick_dataset_previews",
    help="Directory for saved preview videos.",
)
parser.add_argument(
    "--preview_camera",
    choices=("third", "wrist", "viewer", "tactile_left", "tactile_right"),
    default="third",
    help="Camera source for dataset preview videos.",
)
parser.add_argument("--preview_every_n_steps", type=int, default=4, help="Save one preview video frame every N sim steps.")
parser.add_argument("--seed", type=int, default=7, help="Random seed for labware pose sampling.")
parser.add_argument(
    "--cafe_record_dir",
    type=str,
    default=None,
    help="Optional output directory for ForceCapture-CAFE style raw record directories.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/home/gtq/TacEx/bc_policy/sim_robot/ckps/small_flow_matching_a32_s2_c2/best.pt",
    help="Path to the trained sim_robot BC/flow-matching checkpoint.",
)
parser.add_argument(
    "--policy_root",
    type=str,
    default="/home/gtq/TacEx/bc_policy",
    help="Directory that contains the sim_robot Python package.",
)
parser.add_argument("--num_trials", type=int, default=20, help="Number of closed-loop evaluation episodes.")
parser.add_argument("--num_inference_steps", type=int, default=100, help="Flow matching inference steps per action chunk.")
parser.add_argument("--chunk_execute_steps", type=int, default=16, help="Number of predicted chunk actions to execute.")
parser.add_argument("--eval_episode_steps", type=int, default=720, help="Maximum sim steps per closed-loop evaluation episode.")
parser.add_argument(
    "--eval_sample_interval_s",
    type=float,
    default=0.05,
    help="Control interval for policy actions. Default 0.05s gives 20 Hz in a 120 Hz sim.",
)
parser.add_argument(
    "--eval_video_dir",
    type=str,
    default="/home/gtq/TacEx/logs/lab_pick_bc_eval",
    help="Directory for closed-loop rollout videos.",
)
parser.add_argument("--eval_video_every_n_steps", type=int, default=4, help="Save one evaluation video frame every N sim steps.")
parser.add_argument("--eval_camera", choices=("third", "wrist", "viewer", "tactile_left", "tactile_right"), default="third")
parser.add_argument(
    "--reset_policy_seed_each_trial",
    action="store_true",
    default=True,
    help="Reset the policy sampling RNG at the start of each trial for deterministic flow-matching rollouts.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn.functional as F
import imageio.v2 as imageio
import h5py
import numpy as np
from torchvision.utils import save_image

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg, save_images_to_file
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from tacex import GelSightSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
from dataset_writer import CafeRecordWriter, LabPickHdf5Writer
from sampling import dataset_sample_interval_steps, should_continue_collection
from success import StableLiftSuccessTracker
from visuals import SLIDE_VISUAL_DIFFUSE_COLOR, SLIDE_VISUAL_OPACITY, SLIDE_VISUAL_ROUGHNESS


@configclass
class LabPickEnvCfg(DirectRLEnvCfg):
    """Minimal direct environment for scripted labware grasping."""

    viewer: ViewerCfg = ViewerCfg(
        eye=(1.15, -1.15, 0.65),
        lookat=(0.52, 0.0, 0.05),
        origin_type="env",
        env_index=0,
        resolution=(1280, 720),
    )

    decimation = 1
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=1,
        physx=PhysxCfg(
            enable_ccd=True,
            solver_type=1,
            max_position_iteration_count=128,
            max_velocity_iteration_count=1,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.8,
            dynamic_friction=1.5,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=False,
    )

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.001)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.78, 0.78, 0.78), intensity=2500.0),
    )

    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/lab_table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,
            ),
        ),
    )

    slide = RigidObjectCfg(
        prim_path="/World/envs/env_.*/labware",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.0, 0.0196), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            # Microscope slide proxy with physical dimensions close to 75 x 25 x 1.2 mm.
            # The support below leaves the long edges exposed so the gripper can physically pinch it.
            size=(0.075, 0.025, 0.0012),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=0.3,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=2.5, dynamic_friction=2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=SLIDE_VISUAL_DIFFUSE_COLOR,
                opacity=SLIDE_VISUAL_OPACITY,
                roughness=SLIDE_VISUAL_ROUGHNESS,
            ),
        ),
    )

    labware_support = RigidObjectCfg(
        prim_path="/World/envs/env_.*/labware_support",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.0, 0.009), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            # Narrow support leaves the slide side edges exposed for a physical pinch grasp.
            size=(0.060, 0.010, 0.018),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18), opacity=0.0, roughness=0.5),
        ),
    )

    coverslip = RigidObjectCfg(
        prim_path="/World/envs/env_.*/labware",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.0, 0.0255), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.022, 0.022, 0.0012),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.4, dynamic_friction=1.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.55, 1.0), opacity=0.85, roughness=0.04),
        ),
    )

    cup = RigidObjectCfg(
        prim_path="/World/envs/env_.*/labware",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.0, 0.065), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.032,
            height=0.08,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.9, 1.0), opacity=0.42, roughness=0.02),
        ),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    robot.spawn.activate_contact_sensors = True
    robot.spawn.articulation_props.enabled_self_collisions = False
    robot.spawn.articulation_props.solver_position_iteration_count = 128
    robot.spawn.articulation_props.solver_velocity_iteration_count = 1
    robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)

    wrist_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera",
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(-0.70614, 0.03701, 0.03701, -0.70614),
            convention="ros",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 2.0),
        ),
        width=640,
        height=480,
    )

    third_person_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/third_person_camera",
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0, 0.0, 0.4),
            rot=(0.35355, -0.61237, -0.61237, 0.35355),
            convention="ros",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        width=1280,
        height=720,
    )

    gsmini_left = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(160, 120),
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=False,
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb", "height_map"],
    )
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        with_shadow=False,
        tactile_img_res=(160, 120),
        device="cuda",
    )
    gsmini_right = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )

    left_contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        update_period=0.0,
        history_length=1,
        track_pose=True,
    )
    right_contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
        update_period=0.0,
        history_length=1,
        track_pose=True,
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")

    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0


class LabPickEnv(DirectRLEnv):
    cfg: LabPickEnvCfg

    def __init__(self, cfg: LabPickEnvCfg, labware: str, render_mode: str | None = None, **kwargs):
        self.labware_name = labware
        self.labware_cfg = getattr(cfg, labware)
        super().__init__(cfg, render_mode, **kwargs)

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])

        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.gripper_width = torch.full((self.num_envs, 2), 0.04, device=self.device)
        self.initial_object_height = self.labware.data.root_pos_w[:, 2].clone()
        self.initial_object_pos_b = self.labware.data.root_pos_w - self._robot.data.root_link_pos_w
        self.tactile_threshold_mm = 0.0
        self.has_touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_target_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_target_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.last_target_quat_b[:, 0] = 1.0
        self.nominal_ee_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.nominal_ee_quat_b[:, 0] = 1.0
        self.initial_ee_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.initial_ee_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.initial_ee_quat_b[:, 0] = 1.0
        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.contact_probe_depth_m = 0.018
        self.step_count = 0
        self.randomize_labware = False
        self.labware_random_xy = (0.0, 0.0)
        self.labware_random_yaw = 0.0
        self.labware_reset_pos_w = self.labware.data.root_pos_w.clone()
        self.labware_reset_quat_w = self.labware.data.root_quat_w.clone()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.labware = RigidObject(self.labware_cfg)
        self.scene.rigid_objects["labware"] = self.labware

        self.scene.clone_environments(copy_from_source=False)
        self._spawn_labware_visuals()

        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.third_person_camera = TiledCamera(self.cfg.third_person_camera)
        self.scene.sensors["wrist_camera"] = self.wrist_camera
        self.scene.sensors["third_person_camera"] = self.third_person_camera

        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_left"] = self.gsmini_left
        self.scene.sensors["gsmini_right"] = self.gsmini_right

        self.left_contact_sensor = ContactSensor(self.cfg.left_contact_sensor)
        self.right_contact_sensor = ContactSensor(self.cfg.right_contact_sensor)
        self.scene.sensors["left_contact_sensor"] = self.left_contact_sensor
        self.scene.sensors["right_contact_sensor"] = self.right_contact_sensor

        RigidObject(self.cfg.plate)
        RigidObject(self.cfg.labware_support)
        self.cfg.ground.spawn.func(
            self.cfg.ground.prim_path,
            self.cfg.ground.spawn,
            translation=self.cfg.ground.init_state.pos,
            orientation=self.cfg.ground.init_state.rot,
        )
        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)

    def _spawn_labware_visuals(self):
        if self.labware_name != "slide":
            return

        glass_visual = sim_utils.MeshCuboidCfg(
            # Visual-only microscope slide: 76 x 26 x 1.2 mm.
            size=(0.076, 0.026, 0.0012),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=SLIDE_VISUAL_DIFFUSE_COLOR,
                opacity=SLIDE_VISUAL_OPACITY,
                roughness=SLIDE_VISUAL_ROUGHNESS,
                metallic=0.0,
            ),
        )
        glass_visual.func(
            "/World/envs/env_.*/labware/slide_glass_visual",
            glass_visual,
            translation=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

    def _pre_physics_step(self, actions: torch.Tensor | None):
        _, ee_quat_curr_b = self._compute_frame_pose()
        self._ik_controller.set_command(self.ik_commands, ee_quat=ee_quat_curr_b)

    def _apply_action(self):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if torch.linalg.norm(ee_pos_curr_b) > 0.0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()

        joint_pos_des[:, self._finger_joint_ids] = self.gripper_width
        self._robot.set_joint_position_target(joint_pos_des)
        self.step_count += 1

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return done, done

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _get_observations(self) -> dict:
        return {"policy": torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        root_state = self.labware.data.default_root_state[env_ids].clone()
        if self.randomize_labware:
            xy_range = torch.tensor(self.labware_random_xy, device=self.device, dtype=root_state.dtype)
            xy_noise = (torch.rand((len(env_ids), 2), device=self.device, dtype=root_state.dtype) * 2.0 - 1.0) * xy_range
            root_state[:, 0:2] += xy_noise

            yaw_noise = (torch.rand((len(env_ids),), device=self.device, dtype=root_state.dtype) * 2.0 - 1.0) * self.labware_random_yaw
            yaw_quat = torch.stack(
                (
                    torch.cos(0.5 * yaw_noise),
                    torch.zeros_like(yaw_noise),
                    torch.zeros_like(yaw_noise),
                    torch.sin(0.5 * yaw_noise),
                ),
                dim=-1,
            )
            root_state[:, 3:7] = math_utils.quat_mul(yaw_quat, root_state[:, 3:7])

        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.labware.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.labware_reset_pos_w[env_ids] = root_state[:, :3]
        self.labware_reset_quat_w[env_ids] = root_state[:, 3:7]

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.has_touched[env_ids] = False
        self.initial_object_height[env_ids] = root_state[:, 2]
        self.initial_object_pos_b[env_ids] = root_state[:, :3] - self._robot.data.root_link_pos_w[env_ids]
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        self.initial_ee_pos_b[env_ids] = ee_pos_b[env_ids]
        self.initial_ee_quat_b[env_ids] = ee_quat_b[env_ids]
        self.nominal_ee_quat_b[env_ids] = ee_quat_b[env_ids]
        self.ik_commands[env_ids] = 0.0
        self.ik_commands[env_ids, :3] = ee_pos_b[env_ids]
        self.gripper_width[env_ids] = 0.04
        self.last_target_pos_b[env_ids] = ee_pos_b[env_ids]
        self.last_target_quat_b[env_ids] = ee_quat_b[env_ids]
        self.step_count = 0

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return math_utils.combine_frame_transforms(ee_pos_b, ee_quat_b, self._offset_pos, self._offset_rot)

    def _compute_frame_jacobian(self) -> torch.Tensor:
        jacobian = self.jacobian_b
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian

    def command_pick_state_machine(self, return_home: bool = False):
        object_pos_w = self.labware.data.root_pos_w
        object_pos_b = object_pos_w - self._robot.data.root_link_pos_w
        target_pos_b = object_pos_b.clone()
        touch_left, touch_right = self.tactile_contact_depths()
        touched = (touch_left > self.tactile_threshold_mm) | (touch_right > self.tactile_threshold_mm)
        self.has_touched |= touched

        if self.labware_name == "cup":
            hover_height = 0.040
            grasp_height = 0.030
            lift_height = 0.08
            close_width = 0.020
            close_start = 220
            close_end = 500
            squeeze_steps = 40
        elif self.labware_name == "slide":
            hover_height = 0.048
            grasp_height = 0.0006
            lift_height = 0.25
            close_width = 0.0
            close_start = 120
            close_end = 240
            squeeze_steps = 36
        else:
            hover_height = 0.046
            grasp_height = 0.010
            lift_height = 0.08
            close_width = 0.0
            close_start = 180
            close_end = 600
            squeeze_steps = 60

        phase = self.step_count
        lift_hold_steps = 180
        return_steps = 260
        lift_start = close_end + squeeze_steps
        return_start = lift_start + lift_hold_steps
        lifted_target_pos_b = self.initial_object_pos_b.clone()
        if bool(self.has_touched.any().item()):
            lifted_target_pos_b[:, 2] += lift_height
        else:
            lifted_target_pos_b[:, 2] += grasp_height

        if phase < 90:
            target_pos_b[:, 2] += hover_height
            self.gripper_width[:] = 0.04
        elif phase < close_start:
            approach_progress = min((phase - 90) * 0.0005, hover_height - grasp_height)
            target_pos_b[:, 2] += hover_height - approach_progress
            self.gripper_width[:] = 0.04
        elif phase < close_end:
            close_progress = min(max((phase - close_start) / max(close_end - close_start, 1), 0.0), 1.0)
            if self.labware_name == "cup":
                target_pos_b[:, 2] += max(grasp_height - 0.0015 * close_progress, 0.0)
            else:
                target_pos_b[:, 2] += grasp_height
            self.gripper_width[:] = 0.04 - (0.04 - close_width) * close_progress
        elif phase < close_end + squeeze_steps:
            target_pos_b[:, 2] += grasp_height
            self.gripper_width[:] = close_width
        elif phase < return_start or not return_home:
            target_pos_b[:] = lifted_target_pos_b
            self.gripper_width[:] = close_width
        else:
            return_progress = min(max((phase - return_start) / max(return_steps, 1), 0.0), 1.0)
            target_pos_b[:] = lifted_target_pos_b * (1.0 - return_progress) + self.initial_ee_pos_b * return_progress
            self.gripper_width[:] = close_width

        self.ik_commands[:, :3] = target_pos_b
        self.last_target_pos_b[:] = target_pos_b
        self.last_target_quat_b[:] = self.nominal_ee_quat_b

    def action_vector(self) -> torch.Tensor:
        return torch.cat((self.last_target_pos_b, self._rot6d_from_quat(self.last_target_quat_b), self.gripper_width[:, :1]), dim=-1)

    def cafe_observation(self) -> dict[str, torch.Tensor]:
        tool_pos_b, tool_quat_b = self._compute_frame_pose()
        return {
            "robot0_pos": torch.cat((tool_pos_b, self._rot6d_from_quat(tool_quat_b), self.gripper_width[:, :1]), dim=-1),
            "robot0_force": self.cafe_force_torque(),
        }

    def cafe_force_torque(self) -> torch.Tensor:
        left_force_w = self._contact_force_w(self.left_contact_sensor, "left")
        right_force_w = self._contact_force_w(self.right_contact_sensor, "right")
        force_w = left_force_w + right_force_w

        left_case_w, _ = self.gsmini_left.prim_view.get_world_poses()
        right_case_w, _ = self.gsmini_right.prim_view.get_world_poses()
        tool_pos_b, _ = self._compute_frame_pose()
        tool_pos_w = tool_pos_b + self._robot.data.root_link_pos_w
        torque_w = torch.cross(left_case_w - tool_pos_w, left_force_w, dim=-1) + torch.cross(
            right_case_w - tool_pos_w,
            right_force_w,
            dim=-1,
        )
        return torch.cat((force_w, torque_w), dim=-1).detach().clone()

    def _contact_force_w(self, sensor: ContactSensor, name: str) -> torch.Tensor:
        net_forces_w = getattr(sensor.data, "net_forces_w", None)
        if net_forces_w is None:
            raise RuntimeError(
                f"{name} ContactSensor has no net_forces_w data; real Isaac/PhysX force is required."
            )
        if net_forces_w.ndim != 3 or net_forces_w.shape[-1] != 3:
            raise RuntimeError(
                f"{name} ContactSensor returned invalid net_forces_w shape {tuple(net_forces_w.shape)}."
            )
        return net_forces_w.sum(dim=1)

    def cafe_record_sample(self, timestamp: float) -> dict[str, np.ndarray]:
        tool_pos_b, tool_quat_b = self._compute_frame_pose()
        return {
            "xyz": tool_pos_b[0].detach().cpu().numpy(),
            "quat": tool_quat_b[0].detach().cpu().numpy(),
            "width": self.gripper_width[0, :1].detach().cpu().numpy(),
            "ft": self.cafe_force_torque()[0].detach().cpu().numpy(),
            "marker2d": np.zeros((14, 26, 2), dtype=np.float32),
            "rgb": self.wrist_image_224()[0].detach().cpu().numpy(),
            "action": self.action_vector()[0].detach().cpu().numpy(),
        }

    def wrist_image_224(self) -> torch.Tensor:
        rgb = self.wrist_camera.data.output["rgb"][:, :, :, :3]
        rgb = rgb.permute(0, 3, 1, 2).float()
        image = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        return image.clamp(0, 255).byte().permute(0, 2, 3, 1)

    def demo_success(self) -> bool:
        lifted = self.labware.data.root_pos_w[:, 2] - self.initial_object_height
        tool_pos_b, _ = self._compute_frame_pose()
        home_error = torch.linalg.norm(tool_pos_b - self.initial_ee_pos_b, dim=1)
        return bool(((lifted > 0.03) & (home_error < 0.04) & self.has_touched).all().item())

    def stable_lift_success_state(self) -> tuple[float, bool, float]:
        touch_left, touch_right = self.tactile_contact_depths()
        touched = (touch_left > self.tactile_threshold_mm) | (touch_right > self.tactile_threshold_mm)
        self.has_touched |= touched
        lifted = (self.labware.data.root_pos_w[:, 2] - self.initial_object_height)[0].item()
        tool_pos_b, _ = self._compute_frame_pose()
        object_pos_b = self.labware.data.root_pos_w - self._robot.data.root_link_pos_w
        object_gripper_distance = torch.linalg.norm(object_pos_b - tool_pos_b, dim=1)[0].item()
        return lifted, bool(self.has_touched[0].item()), object_gripper_distance

    @staticmethod
    def _rot6d_from_quat(quat: torch.Tensor) -> torch.Tensor:
        rot = math_utils.matrix_from_quat(quat)
        return rot[:, :, :2].reshape(quat.shape[0], 6)

    def print_state(self):
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        tool_pos_b, _ = self._compute_frame_pose()
        tool_pos_w = tool_pos_b + self._robot.data.root_link_pos_w
        object_pos_w = self.labware.data.root_pos_w
        left_case_w, _ = self.gsmini_left.prim_view.get_world_poses()
        right_case_w, _ = self.gsmini_right.prim_view.get_world_poses()
        left_dist = torch.linalg.norm(left_case_w - object_pos_w, dim=1)
        right_dist = torch.linalg.norm(right_case_w - object_pos_w, dim=1)
        case_gap_y = torch.abs(right_case_w[:, 1] - left_case_w[:, 1])
        case_z_mean = 0.5 * (left_case_w[:, 2] + right_case_w[:, 2])
        left_touch, right_touch = self.tactile_contact_depths()
        print(
            "[STATE] "
            f"step={self.step_count} "
            f"ee_pos={ee_pos_w[0].detach().cpu().numpy().round(4).tolist()} "
            f"tool_pos={tool_pos_w[0].detach().cpu().numpy().round(4).tolist()} "
            f"object_pos={object_pos_w[0].detach().cpu().numpy().round(4).tolist()} "
            f"left_case={left_case_w[0].detach().cpu().numpy().round(4).tolist()} "
            f"right_case={right_case_w[0].detach().cpu().numpy().round(4).tolist()} "
            f"case_dist=({left_dist[0].item():.4f},{right_dist[0].item():.4f}) "
            f"case_gap_y={case_gap_y[0].item():.4f} "
            f"case_z_mean={case_z_mean[0].item():.4f} "
            f"root_pos={self._robot.data.root_link_pos_w[0].detach().cpu().numpy().round(4).tolist()} "
            f"target_z={self.last_target_pos_b[0, 2].item():.4f} "
            f"target_qw={self.last_target_quat_b[0, 0].item():.4f} "
            f"finger={self.gripper_width[0, 0].item():.4f} "
            f"finger_gap={2.0 * self.gripper_width[0, 0].item():.4f} "
            f"indent_left_mm={left_touch[0].item():.4f} "
            f"indent_right_mm={right_touch[0].item():.4f} "
            f"contact={bool(self.has_touched[0].item())}"
        )

    def tactile_contact_depths(self) -> tuple[torch.Tensor, torch.Tensor]:
        left_depth = self.gsmini_left.indentation_depth
        right_depth = self.gsmini_right.indentation_depth
        if left_depth is None or right_depth is None:
            zeros = torch.zeros(self.num_envs, device=self.device)
            return zeros, zeros
        return left_depth.float(), right_depth.float()

    def _normalized_tactile_rgb(self, sensor: GelSightSensor) -> torch.Tensor:
        tactile_rgb = sensor.data.output["tactile_rgb"].float()
        if tactile_rgb.max() > 1.0:
            tactile_rgb = tactile_rgb / 255.0
        return tactile_rgb.clamp(0.0, 1.0)

    def save_camera_images(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        wrist_rgb = self.wrist_camera.data.output["rgb"].float() / 255.0
        third_rgb = self.third_person_camera.data.output["rgb"].float() / 255.0
        save_images_to_file(wrist_rgb, str(output_dir / f"wrist_step_{self.step_count:05d}.png"))
        save_images_to_file(third_rgb, str(output_dir / f"third_step_{self.step_count:05d}.png"))
        viewer_rgb = self.render(recompute=True)
        if viewer_rgb is not None:
            viewer_tensor = torch.from_numpy(viewer_rgb).float().permute(2, 0, 1) / 255.0
            save_image(viewer_tensor, str(output_dir / f"viewer_step_{self.step_count:05d}.png"))

    def save_tactile_images(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        save_images_to_file(self._normalized_tactile_rgb(self.gsmini_left), str(output_dir / f"tactile_left_step_{self.step_count:05d}.png"))
        save_images_to_file(
            self._normalized_tactile_rgb(self.gsmini_right), str(output_dir / f"tactile_right_step_{self.step_count:05d}.png")
        )

    def get_video_frame(self, camera: str):
        if camera == "third":
            return self.third_person_camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
        if camera == "wrist":
            return self.wrist_camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
        if camera == "tactile_left":
            return (self._normalized_tactile_rgb(self.gsmini_left)[0, :, :, :3] * 255).byte().detach().cpu().numpy()
        if camera == "tactile_right":
            return (self._normalized_tactile_rgb(self.gsmini_right)[0, :, :, :3] * 255).byte().detach().cpu().numpy()
        frame = self.render(recompute=True)
        if frame is None:
            raise RuntimeError(
                "Viewer frame is unavailable. The environment must be created with render_mode='rgb_array'."
            )
        return frame


def step_scripted_env(env: LabPickEnv, return_home: bool):
    env.command_pick_state_machine(return_home=return_home)
    env._pre_physics_step(None)
    env._apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    env.sim.render()


def run_simulator(env: LabPickEnv):
    print(f"[INFO] Starting labware pick demo: labware={env.labware_name}, envs={env.num_envs}")
    env.reset()

    max_steps = int(args_cli.duration / env.physics_dt)
    image_dir = Path("/home/tjx/TacEx/logs/lab_pick") / env.labware_name
    save_interval = max(1, int(1.0 / env.physics_dt))
    video_writer = None
    if args_cli.record_video:
        image_dir.mkdir(parents=True, exist_ok=True)
        video_path = image_dir / f"{args_cli.video_camera}_camera.mp4"
        video_writer = imageio.get_writer(str(video_path), fps=args_cli.video_fps, macro_block_size=1)
        print(f"[INFO] Recording {args_cli.video_camera} camera video to: {video_path}")

    try:
        while simulation_app.is_running() and env.step_count < max_steps:
            env.command_pick_state_machine()
            env._pre_physics_step(None)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            env.sim.render()

            if args_cli.print_state_interval > 0 and env.step_count % args_cli.print_state_interval == 0:
                env.print_state()

            if args_cli.save_camera_images and env.step_count % save_interval == 0:
                env.save_camera_images(image_dir)

            if args_cli.save_tactile_images and env.step_count % save_interval == 0:
                env.save_tactile_images(image_dir)

            if video_writer is not None and env.step_count % args_cli.video_every_n_steps == 0:
                video_writer.append_data(env.get_video_frame(args_cli.video_camera))
    finally:
        if video_writer is not None:
            video_writer.close()

    print("[INFO] Demo finished.")
    final_height = env.labware.data.root_pos_w[:, 2]
    lifted = final_height - env.initial_object_height
    print(f"[RESULT] object_lift_delta_z={lifted[0].item():.4f} m, lifted={lifted[0].item() > 0.03}")
    env.close()


def collect_dataset(env: LabPickEnv):
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    np.random.seed(args_cli.seed)

    env.randomize_labware = args_cli.randomize_labware
    env.labware_random_xy = (args_cli.labware_random_x, args_cli.labware_random_y)
    env.labware_random_yaw = args_cli.labware_random_yaw

    sample_interval_steps = dataset_sample_interval_steps(args_cli.dataset_sample_interval_s, env.physics_dt)
    writer = LabPickHdf5Writer(
        args_cli.dataset_file,
        labware=env.labware_name,
        instruction=args_cli.instruction,
        task_id=args_cli.task_id,
    )
    recorded = 0
    attempts = 0
    print(
        "[INFO] Collecting BC dataset: "
        f"target={args_cli.num_demos}, labware={env.labware_name}, file={args_cli.dataset_file}, "
        f"sample_interval_steps={sample_interval_steps}, sample_interval_s={args_cli.dataset_sample_interval_s}, "
        f"cafe_record_dir={args_cli.cafe_record_dir}"
    )

    try:
        while simulation_app.is_running() and should_continue_collection(
            recorded=recorded,
            target_demos=args_cli.num_demos,
            attempts=attempts,
            max_attempts=args_cli.max_attempts,
        ):
            attempts += 1
            env.reset()
            writer.reset_episode()
            record_writer = None
            if args_cli.cafe_record_dir is not None:
                record_writer = CafeRecordWriter(Path(args_cli.cafe_record_dir) / f"record_{recorded:06d}")
            preview_frames = []
            success_tracker = StableLiftSuccessTracker(
                lift_height_m=args_cli.success_lift_height,
                hold_steps=args_cli.success_hold_steps,
                max_object_gripper_distance_m=args_cli.success_gripper_distance,
            )
            success = False

            for _ in range(args_cli.episode_steps):
                step_scripted_env(env, return_home=False)
                if env.step_count % sample_interval_steps == 0:
                    writer.append_high(env)
                    writer.append_low(env)
                    if record_writer is not None:
                        record_writer.append_sample(
                            timestamp=env.step_count * env.physics_dt,
                            sample=env.cafe_record_sample(timestamp=env.step_count * env.physics_dt),
                        )
                if recorded < args_cli.preview_demos and env.step_count % args_cli.preview_every_n_steps == 0:
                    preview_frames.append(env.get_video_frame(args_cli.preview_camera))

                if args_cli.print_state_interval > 0 and env.step_count % args_cli.print_state_interval == 0:
                    env.print_state()

                lifted, has_touched, object_gripper_distance = env.stable_lift_success_state()
                success = success_tracker.update(lifted, has_touched, object_gripper_distance)
                if success:
                    print(
                        "[INFO] stable_lift_success "
                        f"attempt={attempts} step={env.step_count} lift={lifted:.4f}m "
                        f"stable_steps={success_tracker.stable_steps} object_gripper_distance={object_gripper_distance:.4f}m"
                    )
                    break

            if success or args_cli.write_failed:
                writer.write_episode(env, success=success)
                if record_writer is not None:
                    record_writer.flush_episode(
                        success=success,
                        labware_reset_pos_w=env.labware_reset_pos_w[0].detach().cpu().numpy().astype(np.float32),
                        labware_reset_quat_w=env.labware_reset_quat_w[0].detach().cpu().numpy().astype(np.float32),
                        force_source="isaac_physx_contact_sensor_net_forces_w",
                        force_frame="world",
                    )
                if recorded < args_cli.preview_demos and preview_frames:
                    preview_dir = Path(args_cli.preview_dir)
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    preview_path = preview_dir / f"{env.labware_name}_demo_{recorded:03d}_{args_cli.preview_camera}.mp4"
                    with imageio.get_writer(str(preview_path), fps=args_cli.video_fps, macro_block_size=1) as video_writer:
                        for frame in preview_frames:
                            video_writer.append_data(frame)
                    print(f"[INFO] saved_preview={preview_path}")
                recorded += 1
                print(
                    f"[INFO] recorded_demo={recorded}/{args_cli.num_demos} "
                    f"attempt={attempts} success={success} reset_pos={env.labware_reset_pos_w[0].detach().cpu().numpy().round(4).tolist()}"
                )
            else:
                print(f"[WARN] skipped failed attempt={attempts}")
    finally:
        env.close()

    print(f"[RESULT] wrote {recorded} demos to {args_cli.dataset_file}")


def _policy_observation(env: LabPickEnv) -> dict[str, np.ndarray]:
    obs = env.cafe_observation()
    return {
        "robot0_pos": obs["robot0_pos"][0].detach().cpu().numpy().astype(np.float32),
        "robot0_image": env.wrist_image_224()[0].detach().cpu().numpy(),
    }


def _apply_policy_action(env: LabPickEnv, action: np.ndarray):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < 10:
        raise ValueError(f"Expected 10D action, got shape {action.shape}")
    target_pos = torch.tensor(action[:3], device=env.device, dtype=torch.float32).view(1, 3).repeat(env.num_envs, 1)
    target_width = float(np.clip(action[9], 0.0, 0.04))
    env.ik_commands[:, :3] = target_pos
    env.last_target_pos_b[:] = target_pos
    env.gripper_width[:] = target_width


def _step_env_with_current_action(env: LabPickEnv):
    env._pre_physics_step(None)
    env._apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    env.sim.render()


def eval_policy(env: LabPickEnv):
    policy_root = Path(args_cli.policy_root)
    if str(policy_root) not in sys.path:
        sys.path.insert(0, str(policy_root))

    from sim_robot.deployment.policy_runner import SimActionChunkPolicyRunner

    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    np.random.seed(args_cli.seed)

    env.randomize_labware = args_cli.randomize_labware
    env.labware_random_xy = (args_cli.labware_random_x, args_cli.labware_random_y)
    env.labware_random_yaw = args_cli.labware_random_yaw

    control_interval_steps = dataset_sample_interval_steps(args_cli.eval_sample_interval_s, env.physics_dt)
    chunk_execute_steps = min(max(1, args_cli.chunk_execute_steps), 32)
    video_dir = Path(args_cli.eval_video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    runner = SimActionChunkPolicyRunner(
        checkpoint_path=args_cli.checkpoint,
        device=args_cli.device or "cuda",
        use_ema=True,
        num_inference_steps=args_cli.num_inference_steps,
        seed=args_cli.seed,
    )

    successes = 0
    print(
        "[INFO] Closed-loop BC eval: "
        f"trials={args_cli.num_trials}, checkpoint={args_cli.checkpoint}, "
        f"control_interval_steps={control_interval_steps}, chunk_execute_steps={chunk_execute_steps}, "
        f"num_inference_steps={args_cli.num_inference_steps}"
    )

    try:
        for trial in range(args_cli.num_trials):
            env.reset()
            runner.reset()
            if args_cli.reset_policy_seed_each_trial and runner.generator is not None:
                runner.generator.manual_seed(int(args_cli.seed))
            success_tracker = StableLiftSuccessTracker(
                lift_height_m=args_cli.success_lift_height,
                hold_steps=args_cli.success_hold_steps,
                max_object_gripper_distance_m=args_cli.success_gripper_distance,
            )
            success = False
            frames = []

            while simulation_app.is_running() and env.step_count < args_cli.eval_episode_steps:
                obs = _policy_observation(env)
                runner.update(obs)
                action_chunk = runner.predict_action_chunk()

                for action in action_chunk[:chunk_execute_steps]:
                    _apply_policy_action(env, action)
                    for _ in range(control_interval_steps):
                        _step_env_with_current_action(env)
                        if env.step_count % args_cli.eval_video_every_n_steps == 0:
                            frames.append(env.get_video_frame(args_cli.eval_camera))
                        if args_cli.print_state_interval > 0 and env.step_count % args_cli.print_state_interval == 0:
                            env.print_state()

                        lifted, has_touched, object_gripper_distance = env.stable_lift_success_state()
                        success = success_tracker.update(lifted, has_touched, object_gripper_distance)
                        if success or env.step_count >= args_cli.eval_episode_steps:
                            break
                    if not success and env.step_count < args_cli.eval_episode_steps:
                        runner.update(_policy_observation(env))
                    if success or env.step_count >= args_cli.eval_episode_steps:
                        break

            if success:
                successes += 1
            video_path = video_dir / f"{env.labware_name}_trial_{trial:03d}_{'success' if success else 'fail'}_{args_cli.eval_camera}.mp4"
            if frames:
                with imageio.get_writer(str(video_path), fps=args_cli.video_fps, macro_block_size=1) as video_writer:
                    for frame in frames:
                        video_writer.append_data(frame)
            lifted, has_touched, object_gripper_distance = env.stable_lift_success_state()
            reset_pos = env.labware_reset_pos_w[0].detach().cpu().numpy().round(4).tolist()
            reset_quat = env.labware_reset_quat_w[0].detach().cpu().numpy().round(4).tolist()
            print(
                "[RESULT] "
                f"trial={trial} success={success} lift={lifted:.4f}m touched={has_touched} "
                f"object_gripper_distance={object_gripper_distance:.4f}m "
                f"reset_pos={reset_pos} reset_quat={reset_quat} video={video_path}"
            )
    finally:
        env.close()

    success_rate = successes / max(args_cli.num_trials, 1)
    print(f"[SUMMARY] success_rate={successes}/{args_cli.num_trials} ({success_rate:.2%})")


def main():
    env_cfg = LabPickEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = LabPickEnv(env_cfg, labware=args_cli.labware, render_mode="rgb_array")
    print("[INFO] Setup complete.")
    if args_cli.checkpoint:
        eval_policy(env)
    elif args_cli.collect_dataset:
        collect_dataset(env)
    else:
        run_simulator(env)


if __name__ == "__main__":
    main()
    simulation_app.close()
