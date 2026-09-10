from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.utils import configclass

from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg


GRAPE_PURPLE = (0.32, 0.035, 0.42)
GRAPE_HIGHLIGHT = (0.58, 0.08, 0.72)
LEAF_GREEN = (0.08, 0.34, 0.12)
LEAF_GREEN_LIGHT = (0.16, 0.48, 0.18)
STEM_GREEN = (0.20, 0.38, 0.08)
WOOD_BROWN = (0.24, 0.105, 0.045)


def _rigid_material(static_friction: float = 1.5, dynamic_friction: float = 1.2):
    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.0,
    )


def _static_cuboid(size, color, *, roughness: float = 0.65):
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            opacity=1.0,
            roughness=roughness,
            metallic=0.0,
        ),
    )


def _static_cylinder(
    radius,
    height,
    color,
    *,
    axis="Z",
    roughness: float = 0.75,
    collidable: bool = True,
):
    return sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        axis=axis,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        collision_props=(
            sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
            if collidable
            else None
        ),
        activate_contact_sensors=collidable,
        physics_material=_rigid_material(1.0, 0.8) if collidable else None,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            opacity=1.0,
            roughness=roughness,
            metallic=0.0,
        ),
    )


def _static_sphere(radius, color):
    return sim_utils.SphereCfg(
        radius=radius,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            opacity=1.0,
            roughness=0.30,
            metallic=0.0,
        ),
    )


@configclass
class GrapePickEnvCfg(DirectRLEnvCfg):
    """Force-constrained grape detachment with two physical GelSight pads."""

    enable_cameras: bool = True
    enable_gelsight: bool = True
    randomize_grape_pose: bool = True
    # Randomize the complete target vine assembly relative to the robot.  The
    # vertical range is intentionally larger to exercise horizontal reaching.
    grape_position_randomization_m: tuple[float, float, float] = (0.020, 0.020, 0.035)

    grape_radius_m: float = 0.018
    grape_mass_kg: float = 0.006
    grape_initial_pos: tuple[float, float, float] = (0.60, 0.0, 0.600)
    # The fruit hangs from a lateral branch on an upright vine.  Keeping the
    # pedicel vertical lets it support gravity without artificial constraints.
    stem_anchor_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.090)
    stem_rest_length_m: float = 0.090
    lateral_branch_length_m: float = 0.150
    stem_stiffness_n_per_m: float = 180.0
    stem_damping_n_s_per_m: float = 0.35
    stem_max_tension_n: float = 8.0
    # Optional pedicel bending/torsion resistance.  Zero preserves the legacy
    # translational-only grape tether; delicate-fruit variants may override it.
    stem_angular_stiffness_nm_per_rad: float = 0.0
    stem_angular_damping_nm_s_per_rad: float = 0.0
    stem_max_restoring_torque_nm: float = 0.0
    stem_lock_rotation_while_attached: bool = False
    detach_force_threshold_n: float = 3.0
    randomize_force_thresholds: bool = True
    # This calibrated range overlaps the learned policy's achievable pull
    # envelope: harder samples can remain attached, while the independent
    # damage range below still produces over-grip failures.
    detach_force_randomization_range_n: tuple[float, float] = (2.80, 3.40)
    detach_dwell_steps: int = 3
    stem_force_filter_alpha: float = 0.30

    # Grip force is the larger of the two pad-normal loads (not their sum).
    damage_grip_force_threshold_n: float = 5.0
    damage_force_randomization_range_n: tuple[float, float] = (3.40, 5.00)
    damage_dwell_steps: int = 1
    bilateral_contact_threshold_n: float = 0.20
    target_grip_force_n: float = 3.7
    success_hold_steps: int = 5
    slip_dwell_steps: int = 12
    obstacle_contact_threshold_n: float = 0.25
    retention_distance_m: float = 0.060
    terminate_drop_height_m: float = 0.075
    terminate_on_success: bool = True

    expert_open_width_m: float = 0.040
    expert_min_width_m: float = 0.010
    expert_close_rate_m_per_step: float = 0.00010
    expert_force_adjust_rate_m_per_step: float = 0.000020
    expert_approach_distance_m: float = 0.080
    expert_pull_distance_m: float = 0.050

    viewer = ViewerCfg(
        eye=(1.05, -0.78, 0.78),
        lookat=(0.60, 0.0, 0.55),
        origin_type="env",
        env_index=0,
        resolution=(1280, 720),
    )
    decimation = 1
    episode_length_s = 790 / 120
    action_space = 10
    observation_space = 16
    state_space = 0

    sim = SimulationCfg(
        dt=1 / 120,
        render_interval=1,
        physx=PhysxCfg(
            enable_ccd=True,
            solver_type=1,
            max_position_iteration_count=128,
            max_velocity_iteration_count=2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,
        ),
        physics_material=_rigid_material(),
        render=RenderCfg(enable_translucency=True),
    )
    scene = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=False,
    )

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.001)),
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.055, 0.060, 0.070),
            physics_material=_rigid_material(0.9, 0.8),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.72, 0.78, 0.90), intensity=1500.0),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.30, -0.35, 0.75)),
        spawn=sim_utils.SphereLightCfg(color=(1.0, 0.78, 0.60), intensity=18000.0, radius=0.18),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.35,
                "panda_joint3": 0.0,
                "panda_joint4": -2.35,
                "panda_joint5": 0.0,
                "panda_joint6": 3.50,
                "panda_joint7": 0.785,
                "panda_finger_joint.*": 0.04,
            }
        ),
    )
    robot.spawn.activate_contact_sensors = True
    robot.spawn.articulation_props.enabled_self_collisions = False
    robot.spawn.articulation_props.solver_position_iteration_count = 128
    robot.spawn.articulation_props.solver_velocity_iteration_count = 2
    robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)

    grape = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target_grape",
        init_state=RigidObjectCfg.InitialStateCfg(pos=grape_initial_pos),
        spawn=sim_utils.SphereCfg(
            radius=grape_radius_m,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.5,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=grape_mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
            physics_material=_rigid_material(1.8, 1.45),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=GRAPE_HIGHLIGHT,
                opacity=1.0,
                roughness=0.25,
                metallic=0.0,
            ),
        ),
    )

    # The target stem is visual; the force law below is the actual attachment.
    stem_visual = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target_stem",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.60, 0.0, 0.645),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        # It visualizes the custom tether force and is deliberately not a
        # second physical constraint.
        spawn=_static_cylinder(
            0.0022,
            stem_rest_length_m,
            STEM_GREEN,
            axis="Z",
            collidable=False,
        ),
    )
    branch = RigidObjectCfg(
        prim_path="/World/envs/env_.*/vine_trunk",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.75, 0.0, 0.430)),
        spawn=_static_cylinder(0.014, 0.86, WOOD_BROWN, axis="Z"),
    )
    lateral_branch = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target_lateral_branch",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.675, 0.0, 0.690)),
        spawn=_static_cylinder(0.007, lateral_branch_length_m, STEM_GREEN, axis="X"),
    )
    trellis_left = RigidObjectCfg(
        prim_path="/World/envs/env_.*/trellis_left",
        # Keep the visual trellis behind the arm's elbow corridor while
        # retaining a real collider and task-level contact monitoring.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.28, 0.220, 0.40)),
        spawn=_static_cylinder(0.010, 0.80, WOOD_BROWN),
    )
    trellis_right = RigidObjectCfg(
        prim_path="/World/envs/env_.*/trellis_right",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.88, 0.220, 0.40)),
        spawn=_static_cylinder(0.010, 0.80, WOOD_BROWN),
    )
    backdrop = RigidObjectCfg(
        prim_path="/World/envs/env_.*/vineyard_backdrop",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.58, 0.24, 0.43)),
        spawn=_static_cuboid((0.74, 0.018, 0.86), (0.035, 0.055, 0.038), roughness=0.90),
    )

    background_grapes = tuple(
        RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/background_grape_{index}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
            spawn=_static_sphere(radius, GRAPE_PURPLE),
        )
        for index, (position, radius) in enumerate(
            (
                ((0.43, 0.145, 0.600), 0.016),
                ((0.46, 0.148, 0.580), 0.017),
                ((0.445, 0.150, 0.555), 0.015),
                ((0.72, 0.152, 0.520), 0.017),
                ((0.75, 0.148, 0.498), 0.016),
                ((0.725, 0.150, 0.475), 0.015),
                ((0.77, 0.155, 0.465), 0.014),
            )
        )
    )
    # (center xyz, in-plane angle rad, length, width, color).  These become
    # two-sided almond meshes instead of box-shaped placeholders.
    leaf_visuals = (
        ((0.38, 0.120, 0.675), 0.58, 0.105, 0.048, LEAF_GREEN),
        ((0.72, 0.122, 0.715), -0.62, 0.110, 0.050, LEAF_GREEN_LIGHT),
        ((0.40, 0.148, 0.465), -0.35, 0.090, 0.042, LEAF_GREEN_LIGHT),
        ((0.77, 0.150, 0.390), 0.42, 0.096, 0.044, LEAF_GREEN),
    )

    wrist_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/grape_wrist_camera",
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(-0.70614, 0.03701, 0.03701, -0.70614),
            convention="ros",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=0.30,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 2.0),
        ),
        width=640,
        height=480,
    )
    third_person_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/grape_hero_camera",
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.05, -0.78, 0.78),
            rot=(0.592529, -0.762891, -0.204285, 0.158666),
            convention="ros",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=34.0,
            focus_distance=0.70,
            horizontal_aperture=30.0,
            clipping_range=(0.05, 5.0),
        ),
        width=960,
        height=540,
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
    gsmini_right = gsmini_left.replace(prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right")

    left_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelpad_left",
        update_period=0.0,
        history_length=1,
        track_pose=True,
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/target_grape"],
    )
    right_contact = left_contact.replace(prim_path="/World/envs/env_.*/Robot/gelpad_right")
    robot_obstacle_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand",
        update_period=0.0,
        history_length=1,
        track_pose=False,
        debug_vis=False,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/vine_trunk",
            "/World/envs/env_.*/target_lateral_branch",
            "/World/envs/env_.*/trellis_left",
            "/World/envs/env_.*/trellis_right",
        ],
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
