from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject
from isaaclab.sensors import ContactSensor, TiledCamera

from tacex import GelSightSensor
from tacex_tasks.lab_pick.grasp_geometry import centered_tool_target
from tacex_tasks.lab_pick.lab_pick_env import LabPickEnv

from .physics import in_torque_window, seal_compression_from_angle, seal_torque_from_angle
from .seal_twist_env_cfg import SEAL_RED, SealTwistEnvCfg, target_angle_for_torque


class _DisabledGelSight:
    """Minimal no-op used when only physical contact tactile streams are requested."""

    indentation_depth = None

    def reset(self, env_ids=None):
        del env_ids


class SealTwistEnv(LabPickEnv):
    """Tactile cap-tightening task with an explicit, bounded torque objective."""

    cfg: SealTwistEnvCfg
    PHASE_NAMES = ("approach", "descend", "grasp", "preload", "twist", "hold")

    def __init__(self, cfg: SealTwistEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.turn_angle_rad = torch.zeros(self.num_envs, device=self.device)
        self.turn_rate_rad_s = torch.zeros(self.num_envs, device=self.device)
        self.seal_torque_nm = torch.zeros(self.num_envs, device=self.device)
        self.seal_compression_m = torch.zeros(self.num_envs, device=self.device)
        self.expert_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.slip_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.under_force_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.over_force_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.alignment_failure_dwell = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.over_torqued = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.slipped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.under_force_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.over_force_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.alignment_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.peak_torque_nm = torch.zeros(self.num_envs, device=self.device)
        self.peak_left_force_n = torch.zeros(self.num_envs, device=self.device)
        self.peak_right_force_n = torch.zeros(self.num_envs, device=self.device)
        self.minimum_force_threshold_n = torch.full(
            (self.num_envs,), self.cfg.minimum_finger_force_n, device=self.device
        )
        self.maximum_force_threshold_n = torch.full(
            (self.num_envs,), self.cfg.maximum_finger_force_n, device=self.device
        )
        self.force_threshold_profile_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.force_threshold_reset_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._force_threshold_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_expert_target_torque_nm = torch.full(
            (self.num_envs,), self.cfg.expert_target_torque_nm, device=self.device
        )
        self.episode_expert_grasp_width_m = torch.full(
            (self.num_envs,), self.cfg.expert_grasp_width_m, device=self.device
        )
        self.sampled_cap_position_offset_m = torch.zeros((self.num_envs, 2), device=self.device)
        self._previous_angle_rad = torch.zeros(self.num_envs, device=self.device)
        self._cached_flags: dict[str, torch.Tensor] | None = None
        self._cached_flags_step = -1

    def _setup_scene(self):
        """Set up the LabPick-compatible scene with optional optical GelSight."""

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.labware = RigidObject(self.labware_cfg)
        self.scene.rigid_objects["labware"] = self.labware
        self.plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = self.plate
        self.labware_support = RigidObject(self.cfg.labware_support)
        self.scene.rigid_objects["labware_support"] = self.labware_support

        decorative_cfgs = {
            "paper_backdrop": self.cfg.backdrop,
            "paper_backdrop_left_panel": self.cfg.backdrop_left_panel,
            "paper_backdrop_right_panel": self.cfg.backdrop_right_panel,
            "paper_accent_strip": self.cfg.accent_strip,
            "paper_shelf": self.cfg.shelf,
            "paper_bench_fascia": self.cfg.bench_fascia,
            "paper_bench_edge": self.cfg.bench_edge,
        }
        for index, canister_cfg in enumerate(self.cfg.decorative_canisters):
            decorative_cfgs[f"paper_canister_{index}"] = canister_cfg
        self._paper_set: dict[str, RigidObject] = {}
        for name, asset_cfg in decorative_cfgs.items():
            asset = RigidObject(asset_cfg)
            self._paper_set[name] = asset
            self.scene.rigid_objects[name] = asset

        self.scene.clone_environments(copy_from_source=False)
        self._spawn_labware_visuals()
        self._force_non_labware_visuals_opaque()

        if self.cfg.enable_cameras:
            self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
            self.third_person_camera = TiledCamera(self.cfg.third_person_camera)
            self.scene.sensors["wrist_camera"] = self.wrist_camera
            self.scene.sensors["third_person_camera"] = self.third_person_camera

        if self.cfg.enable_gelsight:
            self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
            self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
            self.scene.sensors["gsmini_left"] = self.gsmini_left
            self.scene.sensors["gsmini_right"] = self.gsmini_right
        else:
            self.gsmini_left = _DisabledGelSight()
            self.gsmini_right = _DisabledGelSight()

        self.left_finger_contact_sensor = ContactSensor(self.cfg.left_finger_contact_sensor)
        self.right_finger_contact_sensor = ContactSensor(self.cfg.right_finger_contact_sensor)
        self.scene.sensors["left_finger_contact_sensor"] = self.left_finger_contact_sensor
        self.scene.sensors["right_finger_contact_sensor"] = self.right_finger_contact_sensor

        ground = AssetBaseCfg(
            prim_path=self.cfg.ground.prim_path,
            init_state=self.cfg.ground.init_state,
            spawn=self.cfg.ground.spawn,
        )
        ground.spawn.func(
            ground.prim_path,
            ground.spawn,
            translation=ground.init_state.pos,
            orientation=ground.init_state.rot,
        )
        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)
        for light_cfg in (self.cfg.key_light, self.cfg.rim_light):
            light_cfg.spawn.func(
                light_cfg.prim_path,
                light_cfg.spawn,
                translation=light_cfg.init_state.pos,
                orientation=light_cfg.init_state.rot,
            )

    def _spawn_labware_visuals(self):
        """Add a red seal, bottle neck, cap ribs and a rotation index mark."""

        neck_cfg = sim_utils.CylinderCfg(
            radius=0.021,
            height=0.034,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.90, 0.94), opacity=0.94, roughness=0.12
            ),
        )
        neck_cfg.func(
            "/World/envs/env_.*/labware_support/neck_visual",
            neck_cfg,
            translation=(0.0, 0.0, 0.162),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        seal_cfg = sim_utils.CylinderCfg(
            radius=0.0305,
            height=0.005,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=SEAL_RED, opacity=1.0, roughness=0.58, metallic=0.0
            ),
        )
        seal_cfg.func(
            "/World/envs/env_.*/labware_support/seal_ring_visual",
            seal_cfg,
            translation=(0.0, 0.0, 0.155),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        rib_cfg = sim_utils.MeshCuboidCfg(
            size=(0.0040, 0.0100, self.cfg.cap_height_m * 0.78),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.025, 0.18, 0.52), opacity=1.0, roughness=0.38
            ),
        )
        radius = self.cfg.cap_radius_m * 0.94
        for index in range(16):
            angle = 2.0 * math.pi * index / 16.0
            quat = (math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle))
            rib_cfg.func(
                f"/World/envs/env_.*/labware/rib_{index:02d}",
                rib_cfg,
                translation=(radius * math.cos(angle), radius * math.sin(angle), 0.0),
                orientation=quat,
            )

        marker_cfg = sim_utils.MeshCuboidCfg(
            size=(0.005, 0.012, self.cfg.cap_height_m * 0.84),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.82, 0.05), opacity=1.0, roughness=0.30
            ),
        )
        marker_cfg.func(
            "/World/envs/env_.*/labware/rotation_index",
            marker_cfg,
            translation=(self.cfg.cap_radius_m * 0.95, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

    def _pre_physics_step(self, actions: torch.Tensor | None):
        super()._pre_physics_step(actions)
        if hasattr(self, "turn_angle_rad"):
            if actions is not None and actions.numel() > 0:
                self._apply_policy_turn_command()
            self._write_equivalent_screw_pose()

    def _apply_policy_turn_command(self):
        """Convert the absolute CAFE wrist target into a rate-limited screw turn.

        The expert demonstrations encode cap rotation in the target wrist yaw.
        Reconstructing that same scalar from a BC action makes policy rollouts
        affect the equivalent screw model instead of merely moving the arm.
        """

        relative_quat = math_utils.quat_mul(
            self.last_target_quat_b,
            math_utils.quat_inv(self.nominal_ee_quat_b),
        )
        _, _, requested_yaw = math_utils.euler_xyz_from_quat(relative_quat)
        requested_yaw = torch.remainder(requested_yaw, 2.0 * math.pi).clamp(
            0.0, self.cfg.policy_max_turn_angle_rad
        )

        step = self.step_count
        phase = torch.zeros_like(step)
        phase = torch.where(step >= 100, torch.ones_like(phase), phase)
        phase = torch.where(step >= 230, torch.full_like(phase, 2), phase)
        phase = torch.where(step >= 350, torch.full_like(phase, 3), phase)
        phase = torch.where(step >= 390, torch.full_like(phase, 4), phase)
        phase = torch.where(
            step >= max(int(self.cfg.policy_hold_start_step), 390),
            torch.full_like(phase, 5),
            phase,
        )
        contact = self.get_contact_force_metrics()
        bilateral = (
            contact["left_finger_force_n"] >= self.cfg.bilateral_contact_threshold_n
        ) & (contact["right_finger_force_n"] >= self.cfg.bilateral_contact_threshold_n)
        delta = (requested_yaw - self.turn_angle_rad).clamp(
            min=0.0, max=self.cfg.policy_max_turn_rate_rad_per_step
        )
        twisting = (phase == 4) & bilateral
        self.turn_angle_rad[:] = torch.where(
            twisting, self.turn_angle_rad + delta, self.turn_angle_rad
        )
        self.expert_phase[:] = phase
        self._update_seal_state()

    def _apply_action(self):
        super()._apply_action()
        if hasattr(self, "turn_angle_rad"):
            self.turn_rate_rad_s[:] = (self.turn_angle_rad - self._previous_angle_rad) / self.physics_dt
            self._previous_angle_rad[:] = self.turn_angle_rad
            self._cached_flags_step = -1

    def _write_equivalent_screw_pose(self):
        root_state = self.labware.data.root_state_w.clone()
        root_state[:, :3] = self.labware_reset_pos_w
        root_state[:, 2] -= self.seal_compression_m
        zero = torch.zeros_like(self.turn_angle_rad)
        yaw_quat = math_utils.quat_from_euler_xyz(zero, zero, self.turn_angle_rad)
        root_state[:, 3:7] = math_utils.quat_mul(yaw_quat, self.labware_reset_quat_w)
        self.labware.write_root_pose_to_sim(root_state[:, :7])

    def _update_seal_state(self):
        self.seal_torque_nm[:] = seal_torque_from_angle(
            self.turn_angle_rad,
            engagement_angle_rad=self.cfg.engagement_angle_rad,
            running_torque_nm=self.cfg.running_torque_nm,
            compression_stiffness_nm_per_rad=self.cfg.compression_stiffness_nm_per_rad,
        )
        self.seal_compression_m[:] = seal_compression_from_angle(
            self.turn_angle_rad,
            engagement_angle_rad=self.cfg.engagement_angle_rad,
            thread_pitch_m_per_turn=self.cfg.thread_pitch_m_per_turn,
            maximum_compression_m=self.cfg.maximum_seal_compression_m,
        )

    def command_expert(self):
        """Command a slow grasp and stop at the middle of the torque window."""

        step = self.step_count
        phase = torch.zeros_like(step)
        phase = torch.where(step >= 100, torch.ones_like(phase), phase)
        phase = torch.where(step >= 230, torch.full_like(phase, 2), phase)
        phase = torch.where(step >= 350, torch.full_like(phase, 3), phase)
        phase = torch.where(step >= 390, torch.full_like(phase, 4), phase)

        target_angle = self.cfg.engagement_angle_rad + torch.clamp(
            self.episode_expert_target_torque_nm - self.cfg.running_torque_nm,
            min=0.0,
        ) / max(self.cfg.compression_stiffness_nm_per_rad, 1.0e-8)
        contact = self.get_contact_force_metrics()
        bilateral = (
            contact["left_finger_force_n"] >= self.cfg.bilateral_contact_threshold_n
        ) & (contact["right_finger_force_n"] >= self.cfg.bilateral_contact_threshold_n)
        # Tactile servo: pause the screw command whenever either finger loses
        # contact.  The arm can catch up before rotation resumes.
        twisting = (phase == 4) & bilateral & (self.turn_angle_rad < target_angle)
        self.turn_angle_rad[:] = torch.where(
            twisting,
            torch.minimum(
                self.turn_angle_rad + self.cfg.expert_turn_rate_rad_per_step,
                target_angle,
            ),
            self.turn_angle_rad,
        )
        self._update_seal_state()
        phase = torch.where(
            (phase == 4) & (self.turn_angle_rad >= target_angle - 1.0e-5),
            torch.full_like(phase, 5),
            phase,
        )
        self.expert_phase[:] = phase

        cap_center_b = self.initial_object_pos_b.clone()
        hover_center_b = cap_center_b.clone()
        hover_center_b[:, 2] += self.cfg.expert_hover_m
        descend_progress = ((step.float() - 100.0) / 130.0).clamp(0.0, 1.0).unsqueeze(-1)
        center_target_b = torch.where(
            (step < 100).unsqueeze(-1),
            hover_center_b,
            hover_center_b + descend_progress * (cap_center_b - hover_center_b),
        )
        center_target_b = torch.where((step >= 230).unsqueeze(-1), cap_center_b, center_target_b)

        close_progress = ((step.float() - 230.0) / 120.0).clamp(0.0, 1.0).unsqueeze(-1)
        width = self.cfg.expert_open_width_m + close_progress * (
            self.episode_expert_grasp_width_m.unsqueeze(-1) - self.cfg.expert_open_width_m
        )
        self.gripper_width[:] = width

        zero = torch.zeros_like(self.turn_angle_rad)
        yaw_quat = math_utils.quat_from_euler_xyz(zero, zero, self.turn_angle_rad)
        target_quat_b = math_utils.quat_mul(yaw_quat, self.nominal_ee_quat_b)
        target_pos_b = centered_tool_target(
            center_target_b,
            target_quat_b,
            self.gripper_center_offset_tool,
        )
        self.ik_commands[:, :3] = target_pos_b
        self.ik_commands[:, 3:7] = target_quat_b
        self.last_target_pos_b[:] = target_pos_b
        self.last_target_quat_b[:] = target_quat_b

    def get_task_flags(self) -> dict[str, torch.Tensor]:
        step = int(self.step_count.max().item())
        if self._cached_flags is not None and self._cached_flags_step == step:
            return self._cached_flags

        self._update_seal_state()
        forces = self.get_contact_force_metrics()
        left_force = forces["left_finger_force_n"]
        right_force = forces["right_finger_force_n"]
        bilateral = (left_force >= self.cfg.bilateral_contact_threshold_n) & (
            right_force >= self.cfg.bilateral_contact_threshold_n
        )
        minimum_finger_force = torch.minimum(left_force, right_force)
        maximum_finger_force = torch.maximum(left_force, right_force)
        within_force_window = (
            minimum_finger_force >= self.minimum_force_threshold_n
        ) & (maximum_finger_force <= self.maximum_force_threshold_n)
        alignment_error_m = self.get_alignment_error_m()
        aligned = alignment_error_m <= self.cfg.alignment_tolerance_m
        geometric_grasp = (self.expert_phase >= 3) & (
            self.gripper_width[:, 0] <= self.cfg.expert_grasp_width_m + 1.0e-4
        )
        # Success deliberately requires measured two-sided contact.  The
        # geometric flag is retained for diagnostics but cannot satisfy the
        # tactile task on its own.
        contact_valid = bilateral

        twisting_without_contact = (self.expert_phase == 4) & ~contact_valid
        self.slip_dwell[:] = torch.where(
            twisting_without_contact,
            self.slip_dwell + 1,
            torch.zeros_like(self.slip_dwell),
        )
        self.slipped |= self.slip_dwell >= max(int(self.cfg.slip_dwell_steps), 1)
        upper = self.cfg.torque_window_nm[1]
        self.over_torqued |= self.seal_torque_nm > upper + self.cfg.over_torque_margin_nm
        within_window = in_torque_window(
            self.seal_torque_nm,
            self.cfg.torque_window_nm[0],
            self.cfg.torque_window_nm[1],
        )
        hold = self.expert_phase == 5
        below_minimum_force_now = hold & (
            minimum_finger_force < self.minimum_force_threshold_n
        )
        above_maximum_force_now = (self.expert_phase >= 3) & (
            maximum_finger_force > self.maximum_force_threshold_n
        )
        under_force_now = below_minimum_force_now | (
            hold & (self.seal_torque_nm < self.cfg.torque_window_nm[0])
        )
        over_force_now = above_maximum_force_now | (
            (self.expert_phase >= 3)
            & (self.seal_torque_nm > upper + self.cfg.over_torque_margin_nm)
        )
        alignment_failure_now = hold & ~aligned
        self.under_force_dwell[:] = torch.where(
            under_force_now, self.under_force_dwell + 1, torch.zeros_like(self.under_force_dwell)
        )
        self.over_force_dwell[:] = torch.where(
            over_force_now, self.over_force_dwell + 1, torch.zeros_like(self.over_force_dwell)
        )
        self.alignment_failure_dwell[:] = torch.where(
            alignment_failure_now,
            self.alignment_failure_dwell + 1,
            torch.zeros_like(self.alignment_failure_dwell),
        )
        required_force_steps = max(int(self.cfg.force_violation_dwell_steps), 1)
        required_alignment_steps = max(int(self.cfg.alignment_failure_dwell_steps), 1)
        self.under_force_failed |= self.under_force_dwell >= required_force_steps
        self.over_force_failed |= self.over_force_dwell >= required_force_steps
        self.alignment_failed |= self.alignment_failure_dwell >= required_alignment_steps

        valid_hold = (
            hold
            & within_window
            & within_force_window
            & aligned
            & contact_valid
            & ~self.over_torqued
            & ~self.slipped
            & ~self.under_force_failed
            & ~self.over_force_failed
            & ~self.alignment_failed
        )
        self.success_dwell[:] = torch.where(
            valid_hold,
            self.success_dwell + 1,
            torch.zeros_like(self.success_dwell),
        )
        success = self.success_dwell >= max(int(self.cfg.success_hold_steps), 1)

        self.peak_torque_nm[:] = torch.maximum(self.peak_torque_nm, self.seal_torque_nm)
        self.peak_left_force_n[:] = torch.maximum(self.peak_left_force_n, left_force)
        self.peak_right_force_n[:] = torch.maximum(self.peak_right_force_n, right_force)
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        self._cached_flags = {
            **forces,
            "bilateral_contact": bilateral,
            "geometric_grasp": geometric_grasp,
            "contact_valid": contact_valid,
            "minimum_finger_force_n": minimum_finger_force,
            "maximum_finger_force_n": maximum_finger_force,
            "minimum_force_threshold_n": self.minimum_force_threshold_n.clone(),
            "maximum_force_threshold_n": self.maximum_force_threshold_n.clone(),
            "within_force_window": within_force_window,
            # These two diagnostic labels expose the direction of a terminal
            # constraint violation, never the hidden numeric thresholds.  The
            # distinction matters because the legacy under/over flags also
            # include torque failures.
            "below_minimum_force_threshold": below_minimum_force_now,
            "above_maximum_force_threshold": above_maximum_force_now,
            "alignment_error_m": alignment_error_m,
            "aligned": aligned,
            "turn_angle_rad": self.turn_angle_rad.clone(),
            "turn_rate_rad_s": self.turn_rate_rad_s.clone(),
            "seal_torque_nm": self.seal_torque_nm.clone(),
            "seal_compression_m": self.seal_compression_m.clone(),
            "within_torque_window": within_window,
            "under_torqued": hold & (self.seal_torque_nm < self.cfg.torque_window_nm[0]),
            "over_torqued": self.over_torqued.clone(),
            "under_force": self.under_force_failed.clone(),
            "over_force": self.over_force_failed.clone(),
            "alignment_failure": self.alignment_failed.clone(),
            "slipped": self.slipped.clone(),
            "success": success,
            "timeout": timeout,
        }
        self._cached_flags_step = step
        return self._cached_flags

    def get_task_wrench(self) -> torch.Tensor:
        """Return physical fingertip wrench with modeled seal reaction in Tz."""

        wrench = self.get_cafe_ft().clone()
        wrench[:, 5] += self.seal_torque_nm
        return wrench

    def get_alignment_error_m(self) -> torch.Tensor:
        """Physical lateral distance between fingertip midpoint and cap axis."""

        left_pos_w = self._robot.data.body_link_pos_w[:, self._left_finger_body_idx]
        right_pos_w = self._robot.data.body_link_pos_w[:, self._right_finger_body_idx]
        fingertip_midpoint_w = 0.5 * (left_pos_w + right_pos_w)
        return torch.linalg.norm(fingertip_midpoint_w[:, :2] - self.labware.data.root_pos_w[:, :2], dim=-1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        cafe = self.get_cafe_observation()
        flags = self.get_task_flags()
        tactile = torch.stack(
            (flags["left_finger_force_n"], flags["right_finger_force_n"]), dim=-1
        )
        task_state = torch.stack(
            (self.turn_angle_rad, self.seal_torque_nm, self.seal_compression_m), dim=-1
        )
        return {"policy": torch.cat((cafe["robot0_pos"], self.get_task_wrench(), task_state, tactile), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        flags = self.get_task_flags()
        target = self.cfg.expert_target_torque_nm
        torque_error = torch.abs(flags["seal_torque_nm"] - target)
        reward = -torque_error
        reward += 0.3 * flags["contact_valid"].float()
        reward += 1.5 * flags["within_torque_window"].float()
        reward += 10.0 * flags["success"].float()
        reward -= 10.0 * (flags["over_torqued"] | flags["slipped"]).float()
        self.extras["log"] = {
            "SealTwist/torque_nm": flags["seal_torque_nm"].mean(),
            "SealTwist/turn_angle_rad": flags["turn_angle_rad"].mean(),
            "SealTwist/bilateral_contact": flags["bilateral_contact"].float().mean(),
            "SealTwist/success": flags["success"].float().mean(),
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        flags = self.get_task_flags()
        terminated = (
            flags["over_torqued"]
            | flags["over_force"]
            | flags["under_force"]
            | flags["alignment_failure"]
            | flags["slipped"]
        )
        if self.cfg.terminate_on_success:
            terminated |= flags["success"]
        return terminated, flags["timeout"]

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "turn_angle_rad"):
            return
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self.turn_angle_rad[env_ids] = 0.0
        self.turn_rate_rad_s[env_ids] = 0.0
        self.seal_torque_nm[env_ids] = 0.0
        self.seal_compression_m[env_ids] = 0.0
        self.expert_phase[env_ids] = 0
        self.success_dwell[env_ids] = 0
        self.slip_dwell[env_ids] = 0
        self.under_force_dwell[env_ids] = 0
        self.over_force_dwell[env_ids] = 0
        self.alignment_failure_dwell[env_ids] = 0
        self.over_torqued[env_ids] = False
        self.slipped[env_ids] = False
        self.under_force_failed[env_ids] = False
        self.over_force_failed[env_ids] = False
        self.alignment_failed[env_ids] = False
        self.peak_torque_nm[env_ids] = 0.0
        self.peak_left_force_n[env_ids] = 0.0
        self.peak_right_force_n[env_ids] = 0.0
        self._previous_angle_rad[env_ids] = 0.0
        if self.cfg.randomize_force_thresholds:
            self._reset_force_thresholds(env_ids)
        else:
            self.minimum_force_threshold_n[env_ids] = self.cfg.minimum_finger_force_n
            self.maximum_force_threshold_n[env_ids] = self.cfg.maximum_finger_force_n
            self.force_threshold_profile_index[env_ids] = -1
            self._force_threshold_initialized[env_ids] = True
            self.force_threshold_reset_count[env_ids] += 1
        if self.cfg.randomize_expert_targets:
            self.episode_expert_target_torque_nm[env_ids] = self._uniform_for_envs(
                env_ids, self.cfg.expert_target_torque_randomization_range_nm
            )
            self.episode_expert_grasp_width_m[env_ids] = self._uniform_for_envs(
                env_ids, self.cfg.expert_grasp_width_randomization_range_m
            )
        else:
            self.episode_expert_target_torque_nm[env_ids] = self.cfg.expert_target_torque_nm
            self.episode_expert_grasp_width_m[env_ids] = self.cfg.expert_grasp_width_m
        nominal_xy = torch.tensor(
            self.cfg.cap_initial_pos[:2], dtype=torch.float32, device=self.device
        )
        local_cap_xy = self.labware_reset_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2]
        self.sampled_cap_position_offset_m[env_ids] = local_cap_xy - nominal_xy
        self._cached_flags = None
        self._cached_flags_step = -1
        self._update_seal_state()

    def _reset_force_thresholds(self, env_ids: torch.Tensor) -> None:
        """Apply the configured hidden-threshold temporal process.

        The old implementation sampled every reset unconditionally.  That is
        still available as ``iid``.  Persistent and block schedules retain a
        material window long enough for episode outcomes to carry information
        that an online force adapter can use.
        """

        schedule = str(self.cfg.force_threshold_schedule).lower()
        if schedule not in {"iid", "persistent", "block"}:
            raise ValueError(
                "force_threshold_schedule must be one of: iid, persistent, block"
            )
        block_episodes = int(self.cfg.force_threshold_block_episodes)
        if block_episodes < 1:
            raise ValueError("force_threshold_block_episodes must be positive")
        if schedule == "block" and int(self.cfg.force_threshold_profile_index) >= 0:
            raise ValueError(
                "block force-threshold schedule cannot use a fixed profile index"
            )

        initialized = self._force_threshold_initialized[env_ids]
        if schedule == "iid":
            resample = torch.ones_like(initialized)
        elif schedule == "persistent":
            resample = ~initialized
        else:
            counts = self.force_threshold_reset_count[env_ids]
            resample = (~initialized) | (counts.remainder(block_episodes) == 0)

        sample_ids = env_ids[resample]
        if len(sample_ids) > 0:
            self._sample_force_thresholds(sample_ids, avoid_current=schedule == "block")
            self._force_threshold_initialized[sample_ids] = True
        self.force_threshold_reset_count[env_ids] += 1

    def _sample_force_thresholds(
        self, env_ids: torch.Tensor, *, avoid_current: bool
    ) -> None:
        profiles = tuple(self.cfg.force_threshold_profiles_n)
        if profiles:
            profile_values = torch.as_tensor(
                profiles, dtype=torch.float32, device=self.device
            )
            if profile_values.ndim != 2 or profile_values.shape[1] != 2:
                raise ValueError(
                    "force_threshold_profiles_n must contain (minimum, maximum) pairs"
                )
            if bool(torch.any(profile_values[:, 0] >= profile_values[:, 1]).item()):
                raise ValueError("every force-threshold profile must satisfy minimum < maximum")
            profile_count = int(profile_values.shape[0])
            configured_index = int(self.cfg.force_threshold_profile_index)
            if configured_index >= profile_count:
                raise ValueError(
                    "force_threshold_profile_index is outside force_threshold_profiles_n"
                )
            if configured_index >= 0:
                sampled = torch.full(
                    (len(env_ids),), configured_index, dtype=torch.long, device=self.device
                )
            else:
                sampled = torch.randint(profile_count, (len(env_ids),), device=self.device)
            if avoid_current and configured_index < 0 and profile_count > 1:
                current = self.force_threshold_profile_index[env_ids]
                has_current = current >= 0
                # Choose uniformly among all profiles except the current one.
                alternative = torch.randint(
                    profile_count - 1, (len(env_ids),), device=self.device
                )
                alternative += (alternative >= current.clamp_min(0)).long()
                sampled = torch.where(has_current, alternative, sampled)
            self.force_threshold_profile_index[env_ids] = sampled
            selected = profile_values[sampled]
            self.minimum_force_threshold_n[env_ids] = selected[:, 0]
            self.maximum_force_threshold_n[env_ids] = selected[:, 1]
            return

        minimum = self._uniform_for_envs(
            env_ids, self.cfg.minimum_force_randomization_range_n
        )
        maximum = self._uniform_for_envs(
            env_ids, self.cfg.maximum_force_randomization_range_n
        )
        if bool(torch.any(minimum >= maximum).item()):
            raise ValueError(
                "sampled minimum force threshold must be below maximum; use coupled "
                "force_threshold_profiles_n when the independent ranges overlap"
            )
        self.minimum_force_threshold_n[env_ids] = minimum
        self.maximum_force_threshold_n[env_ids] = maximum
        self.force_threshold_profile_index[env_ids] = -1

    def _uniform_for_envs(
        self, env_ids: torch.Tensor, value_range: tuple[float, float]
    ) -> torch.Tensor:
        lower, upper = map(float, value_range)
        if upper < lower:
            raise ValueError(f"Invalid randomization range: {value_range}")
        if upper == lower:
            return torch.full((len(env_ids),), lower, dtype=torch.float32, device=self.device)
        return lower + (upper - lower) * torch.rand(
            (len(env_ids),), dtype=torch.float32, device=self.device
        )

    def get_expert_phase_names(self) -> list[str]:
        return [self.PHASE_NAMES[int(index)] for index in self.expert_phase.detach().cpu().tolist()]

    def get_camera_rgb(self, camera: str = "third") -> torch.Tensor:
        sensor = self.third_person_camera if camera == "third" else self.wrist_camera
        return sensor.data.output["rgb"][..., :3].detach().clone()

    def get_tactile_rgb(self, side: str) -> torch.Tensor:
        sensor = self.gsmini_left if side == "left" else self.gsmini_right
        return (self._normalized_tactile_rgb(sensor) * 255.0).byte().detach().clone()
