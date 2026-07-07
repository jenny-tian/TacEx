from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg


@configclass
class LabPickEnvCfg(DirectRLEnvCfg):
    """Direct IsaacLab environment configuration for labware picking."""

    labware_name: str = "slide"
    terminate_object_drop_height: float = 0.010
    terminate_object_xy_distance: float = 0.30
    terminate_ee_workspace_margin: float = 0.05
    terminate_break_force_threshold_n: float = 40.0
    contact_force_n_per_mm: float = 8.0
    contact_torque_arm_m: float = 0.018
    marker2d_rows: int = 14
    marker2d_cols: int = 26
    marker2d_sigma: float = 0.22
    marker2d_depth_scale: float = 0.35
    marker2d_shear_scale: float = 45.0
    success_lift_height: float = 0.030
    tactile_threshold_mm: float = 0.0
    randomize_labware_position: bool = True
    labware_pos_randomization_xy: tuple[float, float] = (0.020, 0.010)
    labware_yaw_randomization: float = 0.20

    viewer: ViewerCfg = ViewerCfg(
        eye=(1.15, -1.15, 0.65),
        lookat=(0.52, 0.0, 0.05),
        origin_type="env",
        env_index=0,
        resolution=(1280, 720),
    )

    decimation = 1
    episode_length_s = 8.0
    action_space = 10
    observation_space = 14
    state_space = 0

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
                diffuse_color=(0.0, 0.0, 0.0), opacity=0.0, roughness=0.18
            ),
        ),
    )

    labware_support = RigidObjectCfg(
        prim_path="/World/envs/env_.*/labware_support",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.0, 0.009), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
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

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")


@configclass
class LabPickSlideEnvCfg(LabPickEnvCfg):
    labware_name = "slide"


@configclass
class LabPickCoverslipEnvCfg(LabPickEnvCfg):
    labware_name = "coverslip"


@configclass
class LabPickCupEnvCfg(LabPickEnvCfg):
    labware_name = "cup"
