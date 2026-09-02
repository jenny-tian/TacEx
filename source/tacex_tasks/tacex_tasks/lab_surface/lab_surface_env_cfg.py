from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.utils import configclass

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg


BOARD_COLOR = (0.16, 0.18, 0.22)
TOP_COLOR = (0.30, 0.33, 0.38)
RAISED_COLOR = (0.95, 0.10, 0.03)
GROOVE_COLOR = (0.02, 0.35, 0.95)


def _cuboid(size, color, *, kinematic=True):
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=kinematic,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0002, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            opacity=1.0,
            roughness=0.35,
            metallic=0.0,
        ),
    )


@configclass
class LabSurfaceForceScanEnvCfg(DirectRLEnvCfg):
    """Single-probe tactile force scan over a colored, height-varying board."""

    viewer = ViewerCfg(eye=(0.72, -0.72, 0.55), lookat=(0.52, 0.0, 0.02), origin_type="env", env_index=0)
    decimation = 1
    episode_length_s = 6.0
    action_space = 4
    # pos(3), rot6d(6), gripper(1), force(1), tactile depth(1),
    # previous action(4), scan target xyz(3), force error(1), contact(1), progress(1).
    observation_space = 22
    state_space = 0

    board_center = (0.52, 0.0, 0.0)
    board_size_xy = (0.24, 0.16)
    board_base_top_z = 0.010
    board_top_z = 0.014
    groove_depth_m = 0.001
    raised_defect_height_m = 0.002
    # Scan nearly the full 240 mm board length while leaving a 10 mm probe
    # margin at each edge so the contact pad stays on the surface.
    scan_start_xy = (0.41, 0.0)
    scan_end_xy = (0.63, 0.0)
    target_force_n = 3.0
    force_tolerance_n = 0.3
    scan_speed_m_s = 0.010
    surface_preview_distance_m = 0.010
    action_position_scale_m = 0.002
    action_yaw_scale_rad = math.radians(3.0)
    nominal_probe_quat_b = (0.0, 1.0, 0.0, 0.0)
    force_estimation_n_per_mm = 8.0
    virtual_contact_stiffness_n_per_m = 1000.0
    probe_contact_offset_m = 0.024
    force_filter_alpha = 0.18
    tactile_threshold_mm = 0.005
    defect_count = 4
    max_position_error_m = 0.015
    terminate_on_overforce = True
    hold_current_orientation_for_collection = False
    use_resolved_rate_position_servo = False
    expert_kinematic_control = False
    randomize_board_pose = False
    board_translation_range_m = (0.01, 0.01)
    board_tilt_range_deg = (2.0, 2.0)
    board_yaw_range_deg = 4.0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=1,
        physx=PhysxCfg(
            enable_ccd=True,
            solver_type=1,
            max_position_iteration_count=64,
            max_velocity_iteration_count=1,
            gpu_max_rigid_contact_count=2**22,
            gpu_max_rigid_patch_count=2**22,
            gpu_max_num_partitions=1,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        render=RenderCfg(enable_translucency=True),
    )
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=True, lazy_sensor_update=False)

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.02)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2200.0),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        # Reuse the working pose validated by the repository's planar tactile
        # tasks.  The default asset pose places the probe about 0.8 m high and
        # forces the collector to sweep through a large, collision-prone IK
        # motion before every episode.
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.43,
                "panda_joint3": 0.0,
                "panda_joint4": -2.37,
                "panda_joint5": 0.0,
                "panda_joint6": 2.79,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            }
        ),
    )
    robot.spawn.activate_contact_sensors = True
    robot.spawn.articulation_props.enabled_self_collisions = False
    robot.spawn.articulation_props.solver_position_iteration_count = 64
    robot.spawn.articulation_props.solver_velocity_iteration_count = 1

    board_base = RigidObjectCfg(
        prim_path="/World/envs/env_.*/surface_board_base",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.0, 0.0)),
        spawn=_cuboid((0.24, 0.16, 0.02), BOARD_COLOR),
    )

    # Five separated top tiles create four real transverse grooves. The blue
    # inserts sit at the lower deck height; 8 mm width keeps the recesses
    # visually distinct in the 320x240 BC camera stream.
    board_tiles = tuple(
        RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/surface_tile_{i}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.52 + (i - 2) * 0.048, 0.0, 0.012),
            ),
            spawn=_cuboid((0.040, 0.16, 0.004), TOP_COLOR),
        )
        for i in range(5)
    )
    groove_inserts = tuple(
        RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/surface_groove_{i}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.52 + (i - 1.5) * 0.048, 0.0, 0.0125),
            ),
            spawn=_cuboid((0.008, 0.16, 0.001), GROOVE_COLOR),
        )
        for i in range(4)
    )

    raised_defects = tuple(
        RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/surface_raised_{i}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.0, 0.014)),
            spawn=_cuboid((0.014, 0.014, 0.004), RAISED_COLOR, kinematic=True),
        )
        for i in range(4)
    )

    # Fixed oblique camera used by the offline visual BC dataset.  It is placed
    # left/front and high enough to keep both the probe tip and the full board
    # visible instead of letting the wrist occlude the contact area.
    scene_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/scene_camera",
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.82, -0.72, 0.72),
            rot=(-0.40277, 0.89404, 0.17881, -0.08055),
            convention="ros",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=0.90,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        width=480,
        height=360,
    )

    left_contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelpad_left",
        update_period=0.0,
        history_length=1,
        track_pose=True,
        # Leave filtering open because the scan surface is intentionally made
        # of multiple rigid bodies (board, tiles, blue grooves and bumps).
        filter_prim_paths_expr=[],
    )
    right_contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelpad_right",
        update_period=0.0,
        history_length=1,
        track_pose=True,
        filter_prim_paths_expr=[],
    )

    gsmini_left = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera", update_period=0, resolution=(160, 120), data_types=["depth"], clipping_range=(0.024, 0.034)
        ),
        device="cuda", debug_vis=False, marker_motion_sim_cfg=None, data_types=["tactile_rgb", "height_map"],
    )
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(with_shadow=False, tactile_img_res=(160, 120), device="cuda")
    gsmini_right = gsmini_left.replace(prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right")
    # Pose IK is used for the approach/initial contact, where the nominal
    # probe orientation keeps the target reachable.  During scanning the
    # environment switches to a translational resolved-rate servo.
    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
