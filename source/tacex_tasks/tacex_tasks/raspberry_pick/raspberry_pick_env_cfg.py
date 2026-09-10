from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass

from tacex_tasks.grape_pick.grape_pick_env_cfg import (
    GrapePickEnvCfg,
    _rigid_material,
    _static_cuboid,
    _static_cylinder,
)


RASPBERRY_RED = (0.50, 0.008, 0.026)
RASPBERRY_HIGHLIGHT = (0.70, 0.020, 0.046)
RASPBERRY_DARK = (0.24, 0.003, 0.012)
CANE_GREEN = (0.16, 0.29, 0.055)
CANE_BROWN = (0.23, 0.095, 0.035)
LEAF_COLORS = (
    (0.018, 0.095, 0.028),
    (0.028, 0.145, 0.038),
    (0.043, 0.205, 0.050),
    (0.065, 0.265, 0.066),
)


def _background_assets() -> tuple[RigidObjectCfg, ...]:
    """Layered canes and woody branches for the orchard backdrop.

    Foliage is deliberately created as curved, serrated meshes in the
    environment class.  Keeping spheres out of this list prevents the
    artificial green-ball look of the original scene.
    """

    assets: list[RigidObjectCfg] = []
    cane_specs = (
        ((0.12, 0.38, 0.50), 0.008, 1.00, CANE_BROWN),
        ((0.22, 0.30, 0.54), 0.006, 1.08, CANE_GREEN),
        ((0.34, 0.42, 0.50), 0.009, 1.00, CANE_BROWN),
        ((0.47, 0.34, 0.53), 0.007, 1.06, CANE_GREEN),
        ((0.62, 0.46, 0.50), 0.009, 1.00, CANE_BROWN),
        ((0.78, 0.36, 0.54), 0.006, 1.08, CANE_GREEN),
        ((0.91, 0.43, 0.50), 0.008, 1.00, CANE_BROWN),
        ((1.05, 0.32, 0.53), 0.007, 1.06, CANE_GREEN),
        ((1.16, 0.44, 0.50), 0.009, 1.00, CANE_BROWN),
    )
    for index, (position, radius, height, color) in enumerate(cane_specs):
        assets.append(
            RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/orchard_cane_{index}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                spawn=_static_cylinder(radius, height, color, axis="Z", collidable=False),
            )
        )

    branch_specs = (
        ((0.21, 0.31, 0.76), 0.22, -0.18),
        ((0.41, 0.37, 0.86), 0.27, 0.13),
        ((0.68, 0.39, 0.80), 0.24, -0.12),
        ((0.94, 0.35, 0.69), 0.25, 0.16),
        ((0.30, 0.42, 0.48), 0.20, 0.21),
        ((0.57, 0.43, 0.37), 0.22, -0.16),
        ((0.85, 0.41, 0.40), 0.24, 0.12),
        ((1.05, 0.40, 0.88), 0.18, -0.20),
    )
    for index, (position, length, angle) in enumerate(branch_specs):
        assets.append(
            RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/orchard_branch_{index}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=position,
                    rot=(
                        math.cos(angle * 0.5),
                        0.0,
                        math.sin(angle * 0.5),
                        0.0,
                    ),
                ),
                spawn=_static_cylinder(
                    0.0045, length, CANE_GREEN, axis="X", collidable=False
                ),
            )
        )
    return tuple(assets)


def _leaf_specs():
    """Return overlapping leaflet positions at three different depth planes."""

    centers = (
        (0.10, 0.13, 0.81), (0.16, 0.15, 0.68), (0.20, 0.19, 0.91),
        (0.25, 0.22, 0.50), (0.29, 0.16, 0.76), (0.34, 0.25, 0.61),
        (0.38, 0.18, 0.88), (0.43, 0.30, 0.38), (0.48, 0.23, 0.72),
        (0.53, 0.32, 0.94), (0.58, 0.19, 0.47), (0.64, 0.27, 0.80),
        (0.70, 0.17, 0.93), (0.74, 0.24, 0.57), (0.79, 0.31, 0.36),
        (0.84, 0.18, 0.73), (0.89, 0.27, 0.90), (0.94, 0.16, 0.51),
        (1.00, 0.24, 0.68), (1.07, 0.19, 0.84), (1.13, 0.28, 0.45),
        (0.15, 0.39, 0.40), (0.24, 0.43, 0.60), (0.33, 0.38, 0.94),
        (0.45, 0.44, 0.82), (0.57, 0.40, 0.28), (0.69, 0.45, 0.68),
        (0.81, 0.39, 0.97), (0.93, 0.44, 0.35), (1.05, 0.39, 0.61),
        (1.15, 0.43, 0.88),
    )
    specs = []
    for index, center in enumerate(centers):
        angle = (-1.12, -0.70, -0.32, 0.12, 0.52, 0.88, 1.18)[index % 7]
        length = (0.092, 0.108, 0.124, 0.102)[index % 4]
        width = length * (0.48 if index % 3 else 0.54)
        specs.append((center, angle, length, width, LEAF_COLORS[index % len(LEAF_COLORS)]))
    return tuple(specs)


def _background_raspberry_specs():
    """Visual-only fruit clusters; each entry becomes many small drupelets."""

    return (
        ((0.18, 0.115, 0.59), 0.86),
        ((0.34, 0.205, 0.84), 0.78),
        ((0.84, 0.155, 0.78), 0.92),
        ((0.98, 0.225, 0.48), 0.74),
        ((1.10, 0.285, 0.88), 0.68),
    )


@configclass
class RaspberryPickEnvCfg(GrapePickEnvCfg):
    """A force-limited raspberry harvest scene in a layered berry grove."""

    randomize_grape_pose: bool = True
    grape_position_randomization_m: tuple[float, float, float] = (0.012, 0.016, 0.020)
    grape_radius_m: float = 0.020
    grape_mass_kg: float = 0.004
    grape_initial_pos: tuple[float, float, float] = (0.60, 0.0, 0.600)
    stem_anchor_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.070)
    stem_rest_length_m: float = 0.070
    lateral_branch_length_m: float = 0.160
    stem_stiffness_n_per_m: float = 95.0
    stem_damping_n_s_per_m: float = 0.24
    stem_max_tension_n: float = 4.0
    # The simplified pedicel has no articulated rotational joint.  Lock its
    # rotational degrees of freedom while attached so contact torque is
    # transmitted to the vine instead of spinning the fruit freely.  Linear
    # tether motion and force-based detachment remain unchanged.
    stem_lock_rotation_while_attached: bool = True

    # Raspberries are softer and detach under lower load than the grape proxy.
    detach_force_threshold_n: float = 1.40
    randomize_force_thresholds: bool = False
    detach_force_randomization_range_n: tuple[float, float] = (1.20, 1.60)
    damage_grip_force_threshold_n: float = 2.80
    damage_force_randomization_range_n: tuple[float, float] = (2.40, 3.10)
    target_grip_force_n: float = 2.10
    bilateral_contact_threshold_n: float = 0.12
    damage_dwell_steps: int = 2
    retention_distance_m: float = 0.065

    expert_min_width_m: float = 0.010
    expert_close_rate_m_per_step: float = 0.00008
    expert_force_adjust_rate_m_per_step: float = 0.000014
    expert_approach_distance_m: float = 0.095
    expert_pull_distance_m: float = 0.048
    episode_length_s: float = 900 / 120

    ground = AssetBaseCfg(
        prim_path="/World/orchard_soil_floor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.035)),
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 4.0, 0.07),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0
            ),
            physics_material=_rigid_material(0.9, 0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.115, 0.073, 0.035),
                opacity=1.0,
                roughness=0.94,
                metallic=0.0,
            ),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.73, 0.82, 0.92), intensity=850.0),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.28, -0.40, 0.92)),
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 0.90, 0.76), intensity=7800.0, radius=0.36
        ),
    )

    grape = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target_raspberry",
        init_state=RigidObjectCfg.InitialStateCfg(pos=grape_initial_pos),
        spawn=sim_utils.SphereCfg(
            radius=grape_radius_m,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.4,
                max_angular_velocity=2.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=grape_mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0005, rest_offset=0.0
            ),
            physics_material=_rigid_material(1.65, 1.35),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=RASPBERRY_RED,
                opacity=1.0,
                roughness=0.30,
                metallic=0.0,
            ),
        ),
    )
    stem_visual = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target_raspberry_pedicle",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.60, 0.0, 0.635)),
        spawn=_static_cylinder(
            0.0018, stem_rest_length_m, CANE_GREEN, axis="Z", collidable=False
        ),
    )
    branch = RigidObjectCfg(
        prim_path="/World/envs/env_.*/vine_trunk",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.77, 0.035, 0.48)),
        spawn=_static_cylinder(0.0095, 0.96, CANE_BROWN, axis="Z"),
    )
    lateral_branch = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target_lateral_branch",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.68, 0.020, 0.670)),
        spawn=_static_cylinder(0.0038, lateral_branch_length_m, CANE_GREEN, axis="X"),
    )
    trellis_left = RigidObjectCfg(
        prim_path="/World/envs/env_.*/trellis_left",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.10, 0.28, 0.50)),
        spawn=_static_cylinder(0.013, 1.00, CANE_BROWN),
    )
    trellis_right = RigidObjectCfg(
        prim_path="/World/envs/env_.*/trellis_right",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.10, 0.28, 0.50)),
        spawn=_static_cylinder(0.013, 1.00, CANE_BROWN),
    )
    backdrop = RigidObjectCfg(
        prim_path="/World/envs/env_.*/raspberry_grove_backdrop",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.60, 0.57, 0.54)),
        spawn=_static_cuboid(
            (1.55, 0.025, 1.18), (0.018, 0.052, 0.025), roughness=0.98
        ),
    )
    background_grapes = _background_assets()
    leaf_visuals = _leaf_specs()
    background_raspberries = _background_raspberry_specs()

    wrist_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/raspberry_wrist_camera",
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
        width=960,
        height=720,
    )
    third_person_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/raspberry_hero_camera",
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.05, -0.78, 0.78),
            rot=(0.592529, -0.762891, -0.204285, 0.158666),
            convention="ros",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=42.0,
            focus_distance=0.70,
            horizontal_aperture=30.0,
            clipping_range=(0.05, 5.0),
        ),
        width=1280,
        height=720,
    )
    left_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gelpad_left",
        update_period=0.0,
        history_length=1,
        track_pose=True,
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/target_raspberry"],
    )
    right_contact = left_contact.replace(
        prim_path="/World/envs/env_.*/Robot/gelpad_right"
    )
