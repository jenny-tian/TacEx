from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, TiledCamera

from tacex import GelSightSensor

from .lab_surface_env_cfg import LabSurfaceForceScanEnvCfg


class LabSurfaceForceScanEnv(DirectRLEnv):
    cfg: LabSurfaceForceScanEnvCfg

    def __init__(self, cfg: LabSurfaceForceScanEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        body_ids, _ = self.robot.find_bodies("panda_hand")
        self._hand_idx = body_ids[0]
        self._jacobi_body_idx = self._hand_idx - 1
        left_ids, _ = self.robot.find_bodies("gelpad_left")
        right_ids, _ = self.robot.find_bodies("gelpad_right")
        self._left_idx, self._right_idx = left_ids[0], right_ids[0]
        self._finger_ids, _ = self.robot.find_joints(["panda_finger.*"])
        self.ik = DifferentialIKController(cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device)
        # Isaac Lab exposes the root-frame Jacobian.  Convert it into the
        # robot-base frame expected by the IK controller, matching the
        # convention used by the existing LabPick task.
        self._probe_offset_pos = torch.tensor((0.0, 0.0, 0.11841), device=self.device).repeat(self.num_envs, 1)
        self._probe_offset_rot = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device).repeat(self.num_envs, 1)
        self.command = torch.zeros((self.num_envs, 7), device=self.device)
        self.nominal_probe_quat = torch.tensor(self.cfg.nominal_probe_quat_b, device=self.device).repeat(self.num_envs, 1)
        self.gripper_width = torch.full((self.num_envs, 2), 0.04, device=self.device)
        self.last_action = torch.zeros((self.num_envs, 4), device=self.device)
        self.external_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.external_target_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.external_resolved_rate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.filtered_force_n = torch.zeros(self.num_envs, device=self.device)
        self.board_translation = torch.zeros((self.num_envs, 3), device=self.device)
        self.board_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.board_quat[:, 0] = 1.0
        self.scan_target = torch.zeros((self.num_envs, 2), device=self.device)
        self.scan_progress = torch.zeros(self.num_envs, device=self.device)
        self.defect_centers = torch.zeros((self.num_envs, self.cfg.defect_count, 2), device=self.device)
        self.defect_kind = torch.zeros((self.num_envs, self.cfg.defect_count), dtype=torch.long, device=self.device)
        self._previous_coverage = torch.zeros(self.num_envs, device=self.device)
        self._contact_steps = torch.zeros(self.num_envs, device=self.device)
        self._lost_contact_steps = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self.board_base = RigidObject(self.cfg.board_base)
        self.scene.rigid_objects["surface_board_base"] = self.board_base
        self.board_tiles = []
        for i, cfg in enumerate(self.cfg.board_tiles):
            obj = RigidObject(cfg)
            self.board_tiles.append(obj)
            self.scene.rigid_objects[f"surface_tile_{i}"] = obj
        self.groove_inserts = []
        for i, cfg in enumerate(self.cfg.groove_inserts):
            obj = RigidObject(cfg)
            self.groove_inserts.append(obj)
            self.scene.rigid_objects[f"surface_groove_{i}"] = obj
        self.raised_defects = []
        for i, cfg in enumerate(self.cfg.raised_defects):
            obj = RigidObject(cfg)
            self.raised_defects.append(obj)
            self.scene.rigid_objects[f"surface_raised_{i}"] = obj

        self.scene.clone_environments(copy_from_source=False)
        self.cfg.ground.spawn.func(self.cfg.ground.prim_path, self.cfg.ground.spawn, translation=self.cfg.ground.init_state.pos)
        self.cfg.light.spawn.func("/World/light", self.cfg.light.spawn)
        self.scene_camera = TiledCamera(self.cfg.scene_camera)
        self.scene.sensors["scene_camera"] = self.scene_camera
        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_left"] = self.gsmini_left
        self.scene.sensors["gsmini_right"] = self.gsmini_right
        self.left_contact_sensor = ContactSensor(self.cfg.left_contact_sensor)
        self.right_contact_sensor = ContactSensor(self.cfg.right_contact_sensor)
        self.scene.sensors["left_contact_sensor"] = self.left_contact_sensor
        self.scene.sensors["right_contact_sensor"] = self.right_contact_sensor

    def _compute_frame_pose(self):
        hand_pos_b, hand_quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w,
            self.robot.data.root_link_quat_w,
            self.robot.data.body_link_pos_w[:, self._hand_idx],
            self.robot.data.body_link_quat_w[:, self._hand_idx],
        )
        return math_utils.combine_frame_transforms(hand_pos_b, hand_quat_b, self._probe_offset_pos, self._probe_offset_rot)

    def board_center_world(self) -> torch.Tensor:
        base = torch.tensor(self.cfg.board_center, device=self.device).expand(self.num_envs, 3)
        return base + self.scene.env_origins + self.board_translation

    def board_local_to_world(self, local_pos: torch.Tensor) -> torch.Tensor:
        center_local = torch.tensor(self.cfg.board_center, device=self.device)
        delta = local_pos - center_local
        return self.board_center_world() + math_utils.quat_apply(self.board_quat, delta)

    def board_world_to_local(self, world_pos: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.cfg.board_center, device=self.device) + math_utils.quat_apply(
            math_utils.quat_inv(self.board_quat), world_pos - self.board_center_world()
        )

    def board_local_surface_height(self, local_xy: torch.Tensor) -> torch.Tensor:
        height = torch.full(local_xy.shape[:-1], self.cfg.board_top_z, device=self.device)
        groove_x = torch.tensor(
            [self.cfg.board_center[0] + (i - 1.5) * 0.048 for i in range(4)], device=self.device
        )
        groove = torch.min(torch.abs(local_xy[:, 0:1] - groove_x[None, :]), dim=-1).values < 0.004
        height = torch.where(groove, torch.full_like(height, self.cfg.board_top_z - 0.001), height)
        bump_dist = torch.cdist(local_xy, self.defect_centers[:, :, :2]).amin(dim=-1)
        return torch.where(bump_dist < 0.008, torch.full_like(height, self.cfg.board_top_z + 0.002), height)

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.clamp(-1.0, 1.0)
        raw_force = self._contact_force().reshape(self.num_envs)
        alpha = float(self.cfg.force_filter_alpha)
        self.filtered_force_n = alpha * raw_force + (1.0 - alpha) * self.filtered_force_n
        # The moving target is part of the task state, so demonstrations and
        # learned policies see the same left-to-right scan objective.
        self.scan_progress = (self.episode_length_buf / max(self.max_episode_length - 1, 1)).clamp(0.0, 1.0)
        start_x, start_y = self.cfg.scan_start_xy
        end_x, end_y = self.cfg.scan_end_xy
        local_target = torch.zeros((self.num_envs, 3), device=self.device)
        local_target[:, 0] = start_x + (end_x - start_x) * self.scan_progress
        local_target[:, 1] = start_y + (end_y - start_y) * 0.0
        self.scan_target[:] = self.board_local_to_world(local_target)[:, :2]
        pos, quat = self._compute_frame_pose()
        delta = actions[:, :3] * self.cfg.action_position_scale_m
        target_pos = pos + delta
        if torch.any(self.external_target_valid):
            target_pos = torch.where(self.external_target_valid[:, None], self.external_target_pos, target_pos)
        # Apply workspace limits after the optional scripted absolute target as
        # well; otherwise a collector command can bypass the safety bounds.
        target_local = self.board_world_to_local(target_pos)
        center_x, center_y = self.cfg.board_center[0], self.cfg.board_center[1]
        target_local[:, 0] = target_local[:, 0].clamp(center_x - 0.13, center_x + 0.13)
        target_local[:, 1] = target_local[:, 1].clamp(center_y - 0.09, center_y + 0.09)
        # The probe is the frame exposed to the surface.  Keep its center just
        # above the board so a noisy force estimate cannot drive it through the
        # rigid geometry.  The gel pad can still indent into the 4 mm tile.
        # Allow the probe to follow the recessed grooves.  Clamping against
        # the tile top made the commanded groove height unreachable, so the
        # force loop could not recover to 3 N after a height drop.
        min_probe_z = self.cfg.board_base_top_z + self.cfg.probe_contact_offset_m - 0.006
        target_local[:, 2] = target_local[:, 2].clamp(
            min_probe_z, 0.50
        )
        target_pos = self.board_local_to_world(target_local)
        # Keep the probe normal approximately vertical. The yaw action is
        # intentionally small so force variation is caused by surface height.
        yaw = actions[:, 3] * self.cfg.action_yaw_scale_rad
        yaw_quat = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        if self.cfg.hold_current_orientation_for_collection and torch.any(self.external_target_valid):
            target_quat = quat
        else:
            nominal = self.nominal_probe_quat
            if self.cfg.randomize_board_pose:
                nominal = math_utils.quat_mul(self.board_quat, nominal)
            target_quat = math_utils.quat_mul(yaw_quat, nominal)
        target_quat = target_quat / target_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.command[:, :3] = target_pos
        self.command[:, 3:7] = target_quat
        self.last_action[:] = actions
        self.ik.set_command(self.command)

    def set_external_target(self, target_pos: torch.Tensor, *, use_resolved_rate: bool = False) -> None:
        """Set an absolute Cartesian target for scripted data collection."""
        self.external_target_pos[:] = target_pos
        self.external_target_valid[:] = True
        self.external_resolved_rate[:] = use_resolved_rate

    def _apply_action(self):
        pos, quat = self._compute_frame_pose()
        # Refresh the PhysX Jacobian every step.  Caching the tensor at
        # initialization can leave the lateral columns stale after the arm
        # reaches the board, making scan commands appear frozen.
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :].clone()
        root_rot = math_utils.matrix_from_quat(math_utils.quat_inv(self.robot.data.root_link_quat_w))
        jacobian[:, :3, :] = torch.bmm(root_rot, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(root_rot, jacobian[:, 3:, :])
        jacobian[:, :3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._probe_offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._probe_offset_rot), jacobian[:, 3:, :])
        joint_pos = self.robot.data.joint_pos
        use_resolved_rate = self.cfg.use_resolved_rate_position_servo
        if torch.any(self.external_target_valid):
            use_resolved_rate = self.external_resolved_rate
        if isinstance(use_resolved_rate, bool):
            use_resolved_rate = torch.full((self.num_envs,), use_resolved_rate, dtype=torch.bool, device=self.device)
        if torch.any(use_resolved_rate):
            # Damped resolved-rate servo for scripted scanning.  This is the
            # validated translational controller for the Panda asset; pose IK
            # remains responsible for the approach and initial contact.
            position_error = self.command[:, :3] - pos
            arm_jacobian = jacobian[:, :3, :7]
            damping = 2.0e-3 * torch.eye(3, device=self.device).expand(self.num_envs, 3, 3)
            task_inverse = torch.linalg.solve(
                torch.bmm(arm_jacobian, arm_jacobian.transpose(1, 2)) + damping,
                position_error.unsqueeze(-1),
            )
            delta_q = torch.bmm(arm_jacobian.transpose(1, 2), task_inverse).squeeze(-1)
            # Position targets are applied at the 120 Hz physics rate.  A
            # 0.025 rad cap makes the 0.7 m approach take most of a 20 s
            # episode; allow a bounded, still conservative 0.10 rad step so
            # the probe reaches the board before the scan phase.
            delta_q = delta_q.clamp(-0.10, 0.10)
            joint_des = joint_pos.clone()
            resolved_des = joint_pos[:, :7] + delta_q
            joint_des[:, :7] = torch.where(use_resolved_rate[:, None], resolved_des, joint_pos[:, :7])
            if not torch.all(use_resolved_rate):
                ik_des = self.ik.compute(pos, quat, jacobian, joint_pos)
                joint_des = torch.where(use_resolved_rate[:, None], joint_des, ik_des)
        else:
            joint_des = self.ik.compute(pos, quat, jacobian, joint_pos)
        joint_des[:, self._finger_ids] = self.gripper_width
        if self.cfg.expert_kinematic_control:
            # Expert demonstrations may use the simulator's exact state as a
            # teacher.  Write the solved joint state directly and clear
            # velocity so high-PD dynamics cannot overshoot a Cartesian rail.
            # This path is disabled for BC/DSRL training and evaluation.
            joint_vel = torch.zeros_like(joint_des)
            self.robot.write_joint_state_to_sim(joint_des, joint_vel)
        self.robot.set_joint_position_target(joint_des)

    def _contact_force(self) -> torch.Tensor:
        values = []
        for sensor in (self.left_contact_sensor, self.right_contact_sensor):
            force = None
            # force_matrix_w has one vector per filtered surface body.  Sum
            # those vectors before taking the norm; reading only net_forces_w
            # can be zero when the sensor tracks several rigid surface parts.
            if sensor.data.force_matrix_w is not None and sensor.data.force_matrix_w.numel():
                matrix = sensor.data.force_matrix_w
                force = matrix[:, 0, :, :].sum(dim=1)
            elif sensor.data.net_forces_w is not None and sensor.data.net_forces_w.numel():
                force = sensor.data.net_forces_w
                if force.ndim == 3:
                    force = force[:, 0, :]
            if force is not None:
                values.append(torch.linalg.norm(force, dim=-1))
        # Use a continuous geometric spring estimate for control.  The rigid
        # contact tensor is retained for diagnostics, but can emit a one-frame
        # impulse (or zero) when several surface bodies are involved.
        probe_pos, _ = self._compute_frame_pose()
        probe_local = self.board_world_to_local(probe_pos)
        surface_z = self.board_local_surface_height(probe_local[:, :2])
        penetration = torch.clamp(surface_z + self.cfg.probe_contact_offset_m - probe_local[:, 2], min=0.0)
        virtual_force = penetration * self.cfg.virtual_contact_stiffness_n_per_m
        if values:
            sensor_force = torch.stack(values, dim=-1).max(dim=-1).values
            # Preserve valid tactile readings near the 3 N target, but reject
            # single-frame rigid-contact impulses above the force-control band.
            sensor_limit = self.cfg.target_force_n + 0.8
            usable_sensor = torch.where(
                (sensor_force > 1.0e-5) & (sensor_force <= sensor_limit), sensor_force, virtual_force
            )
            return torch.maximum(usable_sensor, virtual_force).reshape(self.num_envs)
        return virtual_force.reshape(self.num_envs)

    def tactile_depth(self) -> torch.Tensor:
        depths = []
        for sensor in (self.gsmini_left, self.gsmini_right):
            depth = sensor.indentation_depth
            depths.append(depth.float() if depth is not None else torch.zeros(self.num_envs, device=self.device))
        # Keep tactile depth as a sensor-only quantity.  The geometric spring
        # fallback belongs in _contact_force(); calling it here would create a
        # recursive force/depth dependency when the sensor has no reading.
        return torch.stack(depths, dim=-1).amax(dim=-1)

    def _get_observations(self):
        pos, quat = self._compute_frame_pose()
        rot6d = math_utils.matrix_from_quat(quat)[..., :, :2].reshape(self.num_envs, 6)
        force_value = self.filtered_force_n
        force = force_value.unsqueeze(-1) / self.cfg.target_force_n
        depth = self.tactile_depth().unsqueeze(-1) / 0.4
        target = torch.cat((self.scan_target, torch.zeros((self.num_envs, 1), device=self.device)), dim=-1)
        force_error = ((force_value - self.cfg.target_force_n) / self.cfg.target_force_n).unsqueeze(-1)
        contact = (force_value > 0.05).float().unsqueeze(-1)
        progress = self.scan_progress.unsqueeze(-1)
        obs = torch.cat(
            (pos, rot6d, self.gripper_width[:, :1], force, depth, self.last_action, target, force_error, contact, progress),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self):
        pos, _ = self._compute_frame_pose()
        force = self.filtered_force_n
        force_error = (force - self.cfg.target_force_n) / self.cfg.target_force_n
        # Position tracking is measured on the board plane, rather than in
        # world XY.  A tilted board therefore does not incur artificial
        # lateral error from its normal displacement.
        pos_local = self.board_world_to_local(pos)
        start_x, start_y = self.cfg.scan_start_xy
        end_x, end_y = self.cfg.scan_end_xy
        target_local_xy = torch.stack(
            (
                start_x + (end_x - start_x) * self.scan_progress,
                start_y + (end_y - start_y) * self.scan_progress,
            ),
            dim=-1,
        )
        track_error = torch.linalg.norm(pos_local[:, :2] - target_local_xy, dim=-1)
        contact = force > 0.05
        self._contact_steps += contact.float()
        self._lost_contact_steps += (~contact).float()
        reward = -2.0 * force_error.square() - track_error.square() / 0.0009
        reward -= 0.05 * self.last_action.square().mean(dim=-1)
        reward += 0.1 * contact.float() + 0.2 * self.scan_progress
        return reward

    def _get_dones(self):
        force = self._contact_force()
        # Position tracking is rewarded but is not a terminal condition; a
        # transient tracking error should not turn into an artificial failure.
        terminated = (force > 4.0) & self.cfg.terminate_on_overforce
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timeout

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        n = len(env_ids)
        # DirectRLEnv does not automatically rewrite articulation joints on a
        # task reset.  Without this explicit write the Panda keeps the asset's
        # original high stowed pose (about z=0.8 m), and every episode starts
        # with an unnecessarily large IK approach that causes the observed
        # lateral collision and force spike.
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        if self.cfg.randomize_board_pose:
            tx, ty = self.cfg.board_translation_range_m
            tilt_x, tilt_y = self.cfg.board_tilt_range_deg
            self.board_translation[env_ids, 0] = torch.empty(n, device=self.device).uniform_(-tx, tx)
            self.board_translation[env_ids, 1] = torch.empty(n, device=self.device).uniform_(-ty, ty)
            rx = torch.empty(n, device=self.device).uniform_(-tilt_x, tilt_x) * torch.pi / 180.0
            ry = torch.empty(n, device=self.device).uniform_(-tilt_y, tilt_y) * torch.pi / 180.0
            rz = torch.empty(n, device=self.device).uniform_(-self.cfg.board_yaw_range_deg, self.cfg.board_yaw_range_deg) * torch.pi / 180.0
            self.board_quat[env_ids] = math_utils.quat_from_euler_xyz(rx, ry, rz)
        else:
            self.board_translation[env_ids] = 0.0
            self.board_quat[env_ids] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)

        def write_board_object(obj, local_state: torch.Tensor):
            state = obj.data.default_root_state[env_ids].clone()
            state[:, :3] = self.board_local_to_world(local_state[env_ids])
            state[:, 3:7] = self.board_quat[env_ids]
            state[:, 7:] = 0.0
            obj.write_root_state_to_sim(state, env_ids=env_ids)

        board_local = torch.tensor(self.cfg.board_center, device=self.device).expand(self.num_envs, 3).clone()
        write_board_object(self.board_base, board_local)
        for i, obj in enumerate(self.board_tiles):
            local = board_local.clone()
            local[:, 0] = self.cfg.board_center[0] + (i - 2) * 0.048
            local[:, 2] = 0.012
            write_board_object(obj, local)
        for i, obj in enumerate(self.groove_inserts):
            local = board_local.clone()
            local[:, 0] = self.cfg.board_center[0] + (i - 1.5) * 0.048
            local[:, 2] = 0.0125
            write_board_object(obj, local)

        scan_start_local = torch.tensor((*self.cfg.scan_start_xy, 0.0), device=self.device).expand(self.num_envs, 3)
        self.scan_target[env_ids] = self.board_local_to_world(scan_start_local)[env_ids, :2]
        self.scan_progress[env_ids] = 0.0
        self.defect_centers[env_ids, :, 0] = torch.empty((n, self.cfg.defect_count), device=self.device).uniform_(0.45, 0.59)
        self.defect_centers[env_ids, :, 1] = torch.empty((n, self.cfg.defect_count), device=self.device).uniform_(-0.055, 0.055)
        # The scene contains fixed blue recessed channels plus randomized red
        # raised bumps.  Keep this label aligned with the actual randomized
        # geometry (all entries here are raised bumps).
        self.defect_kind[env_ids] = 0
        for i, defect in enumerate(self.raised_defects):
            state = defect.data.default_root_state[env_ids].clone()
            local = torch.cat(
                (self.defect_centers[:, i, :], torch.full((self.num_envs, 1), 0.014, device=self.device)), dim=-1
            )
            state[:, :3] = self.board_local_to_world(local[env_ids])
            state[:, 3:7] = self.board_quat[env_ids]
            state[:, 7:] = 0.0
            defect.write_root_state_to_sim(state, env_ids=env_ids)
        pos, quat = self._compute_frame_pose()
        self.command[env_ids, :3] = pos[env_ids]
        self.command[env_ids, 3:7] = self.nominal_probe_quat[env_ids]
        self.last_action[env_ids] = 0.0
        self.external_target_valid[env_ids] = False
        self.external_resolved_rate[env_ids] = False
        self.filtered_force_n[env_ids] = 0.0
        self._contact_steps[env_ids] = 0.0
        self._lost_contact_steps[env_ids] = 0.0

    def tactile_record(self):
        output = {}
        for name, sensor in (("left", self.gsmini_left), ("right", self.gsmini_right)):
            for key in ("tactile_rgb", "height_map"):
                value = sensor.data.output.get(key)
                if value is not None:
                    output[f"{name}_{key}"] = value[0].detach().cpu().numpy()
        return output

    def visual_record(self):
        rgb = self.scene_camera.data.output.get("rgb")
        depth = self.scene_camera.data.output.get("depth")
        output = {}
        if rgb is not None:
            output["scene_rgb"] = rgb[0, :, :, :3].detach().cpu().numpy()
        if depth is not None:
            output["scene_depth"] = depth[0].detach().cpu().numpy()
        return output

    def record_metrics(self):
        pos, quat = self._compute_frame_pose()
        return {
            "tool_pos": pos[0].detach().cpu().numpy(),
            "tool_quat": quat[0].detach().cpu().numpy(),
            "contact_force_n": float(self._contact_force()[0].detach().cpu()),
            "force_filtered_n": float(self.filtered_force_n[0].detach().cpu()),
            "tactile_depth": float(self.tactile_depth()[0].detach().cpu()),
            "scan_target_xy": self.scan_target[0].detach().cpu().numpy(),
            "defect_centers_xy": self.defect_centers[0].detach().cpu().numpy(),
            "defect_kind": self.defect_kind[0].detach().cpu().numpy(),
        }
