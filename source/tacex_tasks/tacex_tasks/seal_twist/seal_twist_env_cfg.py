from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass

from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)
from tacex_tasks.lab_pick.lab_pick_env_cfg import LabPickEnvCfg


CAP_BLUE = (0.02, 0.22, 0.82)
BOTTLE_GLASS = (0.55, 0.78, 0.84)
SEAL_RED = (0.80, 0.055, 0.035)
LAB_NAVY = (0.035, 0.055, 0.085)
LAB_PANEL = (0.075, 0.115, 0.165)
LAB_CYAN = (0.00, 0.58, 0.62)


def _contact_material(static_friction: float, dynamic_friction: float):
    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.0,
    )


def _decorative_cuboid(size, color, *, roughness: float = 0.65, metallic: float = 0.0):
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            opacity=1.0,
            roughness=roughness,
            metallic=metallic,
        ),
    )


def _decorative_cylinder(radius, height, color, *, roughness: float = 0.35, metallic: float = 0.0):
    return sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            opacity=1.0,
            roughness=roughness,
            metallic=metallic,
        ),
    )


@configclass
class SealTwistEnvCfg(LabPickEnvCfg):
    """Franka/GelSight scene for tightening a cap into a valid torque window."""

    labware_name: str = "cap"
    enable_cameras: bool = True
    enable_gelsight: bool = False
    randomize_labware_position: bool = False
    reset_hold_steps: int = 0
    episode_length_s: float = 9.0
    observation_space: int = 21

    cap_radius_m: float = 0.029
    cap_height_m: float = 0.045
    cap_initial_pos: tuple[float, float, float] = (0.56, 0.0, 0.3675)
    thread_pitch_m_per_turn: float = 0.004
    maximum_seal_compression_m: float = 0.0012
    engagement_angle_rad: float = 0.50
    running_torque_nm: float = 0.08
    compression_stiffness_nm_per_rad: float = 0.25
    torque_window_nm: tuple[float, float] = (0.35, 0.50)
    expert_target_torque_nm: float = 0.42
    over_torque_margin_nm: float = 0.015
    success_hold_steps: int = 24
    bilateral_contact_threshold_n: float = 0.05
    slip_dwell_steps: int = 120
    terminate_on_success: bool = True

    # The BC benchmark adds a bounded fingertip-force objective to the torque
    # objective.  Defaults are deliberately permissive so the original filmed
    # demonstration remains unchanged; evaluation scripts opt into narrower,
    # randomized per-episode limits.
    minimum_finger_force_n: float = 0.05
    maximum_finger_force_n: float = 20.0
    randomize_force_thresholds: bool = False
    # ``iid`` preserves the original benchmark.  ``persistent`` models one
    # bottle/seal material batch for an entire run, while ``block`` models a
    # batch change every ``force_threshold_block_episodes`` resets.  Only the
    # latter two schedules make completed episodes informative about a future
    # hidden force window.
    force_threshold_schedule: str = "iid"
    force_threshold_block_episodes: int = 25
    # Optional coupled (minimum, maximum) profiles.  Coupling prevents invalid
    # or unrealistically broad windows when distinct material classes are used.
    # An empty tuple falls back to the independent ranges below.
    force_threshold_profiles_n: tuple[tuple[float, float], ...] = ()
    # Negative samples a profile.  A non-negative value selects a deterministic
    # hidden material profile for paired experiments without revealing that
    # selection to the policy or VLM observation.
    force_threshold_profile_index: int = -1
    minimum_force_randomization_range_n: tuple[float, float] = (0.05, 0.05)
    maximum_force_randomization_range_n: tuple[float, float] = (20.0, 20.0)
    force_violation_dwell_steps: int = 5
    alignment_tolerance_m: float = 0.012
    alignment_failure_dwell_steps: int = 5

    # Successful training demonstrations cover modest pose, terminal torque,
    # and gripping-width variation.  The learned policy is not given the test
    # force thresholds, making the threshold sweep a genuine base-policy test.
    randomize_expert_targets: bool = False
    expert_target_torque_randomization_range_nm: tuple[float, float] = (0.38, 0.46)
    expert_grasp_width_randomization_range_m: tuple[float, float] = (0.0205, 0.0225)
    policy_hold_start_step: int = 805
    policy_max_turn_rate_rad_per_step: float = 0.006
    policy_max_turn_angle_rad: float = 2.40

    expert_open_width_m: float = 0.040
    expert_grasp_width_m: float = 0.0215
    expert_hover_m: float = 0.105
    expert_turn_rate_rad_per_step: float = 0.0045

    viewer = ViewerCfg(
        eye=(1.02, -0.72, 0.72),
        lookat=(0.56, 0.0, 0.34),
        origin_type="env",
        env_index=0,
        resolution=(1280, 720),
    )

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
        # A local primitive avoids the network-hosted default ground USD and
        # keeps headless/offline startup deterministic.
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 4.0, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=_contact_material(0.9, 0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.045, 0.050, 0.060), roughness=0.85
            ),
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.66, 0.76, 0.92), intensity=850.0),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/SealKeyLight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.28, -0.38, 0.82)),
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 0.78, 0.62), intensity=7200.0, radius=0.16
        ),
    )
    rim_light = AssetBaseCfg(
        prim_path="/World/SealRimLight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.84, 0.16, 0.72)),
        spawn=sim_utils.SphereLightCfg(
            color=(0.38, 0.68, 1.0), intensity=5200.0, radius=0.12
        ),
    )

    # LabPick calls this asset "plate".  Here it is a dark laboratory bench.
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/lab_table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.56, 0.0, 0.015)),
        spawn=sim_utils.CuboidCfg(
            size=(0.72, 0.58, 0.030),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=_contact_material(0.9, 0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.12, 0.15), roughness=0.72, metallic=0.05
            ),
        ),
    )

    # Clean, publication-oriented laboratory set.  All objects stay behind or
    # below the manipulation volume and therefore do not change task physics.
    backdrop = RigidObjectCfg(
        prim_path="/World/envs/env_.*/paper_backdrop",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.56, 0.355, 0.48)),
        spawn=_decorative_cuboid((1.18, 0.030, 0.92), LAB_NAVY, roughness=0.82),
    )
    backdrop_left_panel = RigidObjectCfg(
        prim_path="/World/envs/env_.*/paper_backdrop_left_panel",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.16, 0.329, 0.56)),
        spawn=_decorative_cuboid((0.22, 0.018, 0.34), LAB_PANEL, roughness=0.68),
    )
    backdrop_right_panel = RigidObjectCfg(
        prim_path="/World/envs/env_.*/paper_backdrop_right_panel",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.96, 0.329, 0.56)),
        spawn=_decorative_cuboid((0.22, 0.018, 0.34), LAB_PANEL, roughness=0.68),
    )
    accent_strip = RigidObjectCfg(
        prim_path="/World/envs/env_.*/paper_accent_strip",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.56, 0.315, 0.735)),
        spawn=_decorative_cuboid((1.04, 0.014, 0.018), LAB_CYAN, roughness=0.30, metallic=0.15),
    )
    shelf = RigidObjectCfg(
        prim_path="/World/envs/env_.*/paper_shelf",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.56, 0.275, 0.405)),
        spawn=_decorative_cuboid((0.78, 0.145, 0.018), (0.12, 0.16, 0.20), roughness=0.48, metallic=0.20),
    )
    bench_fascia = RigidObjectCfg(
        prim_path="/World/envs/env_.*/paper_bench_fascia",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.56, -0.300, -0.025)),
        spawn=_decorative_cuboid((0.74, 0.028, 0.115), (0.055, 0.075, 0.10), roughness=0.70),
    )
    bench_edge = RigidObjectCfg(
        prim_path="/World/envs/env_.*/paper_bench_edge",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.56, -0.317, 0.035)),
        spawn=_decorative_cuboid((0.74, 0.012, 0.012), LAB_CYAN, roughness=0.25, metallic=0.20),
    )
    decorative_canisters = tuple(
        RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/paper_canister_{index}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
            spawn=_decorative_cylinder(radius, height, color, roughness=0.30, metallic=metallic),
        )
        for index, (position, radius, height, color, metallic) in enumerate(
            (
                ((0.27, 0.255, 0.463), 0.024, 0.098, (0.88, 0.42, 0.12), 0.05),
                ((0.34, 0.255, 0.450), 0.018, 0.072, (0.18, 0.64, 0.67), 0.08),
                ((0.80, 0.255, 0.456), 0.021, 0.084, (0.46, 0.32, 0.72), 0.05),
                ((0.87, 0.255, 0.445), 0.016, 0.062, (0.78, 0.82, 0.86), 0.28),
            )
        )
    )

    # The bottle is fixed, so all measured rotational resistance belongs to
    # the cap/seal interface rather than motion of the container.
    labware_support = RigidObjectCfg(
        prim_path="/World/envs/env_.*/labware_support",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.56, 0.0, 0.190)),
        spawn=sim_utils.CylinderCfg(
            radius=0.052,
            height=0.320,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                solver_position_iteration_count=32,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            physics_material=_contact_material(0.8, 0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=BOTTLE_GLASS, opacity=0.78, roughness=0.14, metallic=0.0
            ),
        ),
    )

    # Kinematic motion is used only for the equivalent screw constraint.  The
    # cap still collides with the physical GelSight pads and generates contact
    # readings as the arm closes and twists.
    cap = RigidObjectCfg(
        prim_path="/World/envs/env_.*/labware",
        init_state=RigidObjectCfg.InitialStateCfg(pos=cap_initial_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=cap_radius_m,
            height=cap_height_m,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0008, rest_offset=0.0),
            activate_contact_sensors=True,
            physics_material=_contact_material(1.8, 1.45),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=CAP_BLUE, opacity=1.0, roughness=0.32, metallic=0.10
            ),
        ),
    )

    third_person_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/seal_hero_camera",
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.02, -0.72, 0.72),
            rot=(0.592529, -0.762891, -0.204285, 0.158666),
            convention="ros",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=34.0,
            focus_distance=0.65,
            horizontal_aperture=28.0,
            clipping_range=(0.04, 5.0),
        ),
        width=960,
        height=540,
    )
    wrist_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/seal_wrist_camera",
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

    left_finger_contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelpad_left",
        update_period=0.0,
        history_length=1,
        track_pose=True,
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/labware"],
    )
    right_finger_contact_sensor = left_finger_contact_sensor.replace(
        prim_path="/World/envs/env_.*/Robot/gelpad_right"
    )

    # Keep enough wrist range for a ~110 degree tightening stroke.
    robot = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.40,
                "panda_joint3": 0.0,
                "panda_joint4": -2.25,
                "panda_joint5": 0.0,
                "panda_joint6": 2.85,
                "panda_joint7": 0.0,
                "panda_finger_joint.*": 0.04,
            }
        ),
    )
    robot.spawn.activate_contact_sensors = True
    robot.spawn.articulation_props.enabled_self_collisions = False
    robot.spawn.articulation_props.solver_position_iteration_count = 128
    robot.spawn.articulation_props.solver_velocity_iteration_count = 2
    robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)


def target_angle_for_torque(cfg: SealTwistEnvCfg, torque_nm: float) -> float:
    """Analytic target used by the expert and documented in metadata."""

    return cfg.engagement_angle_rad + max(torque_nm - cfg.running_torque_nm, 0.0) / max(
        cfg.compression_stiffness_nm_per_rad, 1.0e-8
    )
