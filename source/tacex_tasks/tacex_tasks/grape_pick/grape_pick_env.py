from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, TiledCamera

from tacex import GelSightSensor

from tacex_tasks.lab_pick.grasp_geometry import centered_tool_target, vector_in_local_frame

from .grape_pick_env_cfg import GrapePickEnvCfg
from .physics import compute_stem_tether_force, update_threshold_dwell


class GrapePickEnv(DirectRLEnv):
    """Pick one grape while obeying physical grip and stem-force limits."""

    cfg: GrapePickEnvCfg

    PHASE_NAMES = ("approach", "align", "force_close", "settle", "pull", "retain")

    def __init__(self, cfg: GrapePickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )
        hand_ids, _ = self._robot.find_bodies("panda_hand")
        left_ids, _ = self._robot.find_bodies("gelpad_left")
        right_ids, _ = self._robot.find_bodies("gelpad_right")
        self._hand_body_idx = hand_ids[0]
        self._jacobi_body_idx = self._hand_body_idx - 1
        self._left_pad_body_idx = left_ids[0]
        self._right_pad_body_idx = right_ids[0]
        self._finger_joint_ids, _ = self._robot.find_joints(["panda_finger.*"])

        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.gripper_width = torch.full(
            (self.num_envs, len(self._finger_joint_ids)),
            self.cfg.expert_open_width_m,
            device=self.device,
        )
        self.last_target_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_target_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.last_target_quat_b[:, 0] = 1.0
        # Point the tool's approach axis along +X so the gripper reaches the
        # fruit horizontally from the robot side of the trellis.  This is the
        # reachable horizontal wrist orientation of the configured home pose.
        self.nominal_ee_quat_b = torch.tensor(
            (0.0, 0.731689, 0.0, 0.681639), device=self.device
        ).repeat(
            self.num_envs, 1
        )
        self._tool_offset_pos = torch.tensor((0.0, 0.0, 0.11841), device=self.device).repeat(self.num_envs, 1)
        self._tool_offset_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device).repeat(
            self.num_envs, 1
        )
        self.gripper_center_offset_tool = torch.zeros((self.num_envs, 3), device=self.device)
        self.workspace_min_b = torch.tensor((0.25, -0.36, 0.055), device=self.device)
        self.workspace_max_b = torch.tensor((0.82, 0.36, 0.76), device=self.device)

        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.grape_reset_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.grape_reset_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.grape_reset_quat_w[:, 0] = 1.0
        self.initial_grape_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.stem_anchor_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.pull_direction_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.stem_attached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.detach_threshold_n = torch.full(
            (self.num_envs,), self.cfg.detach_force_threshold_n, device=self.device
        )
        self.damage_threshold_n = torch.full(
            (self.num_envs,), self.cfg.damage_grip_force_threshold_n, device=self.device
        )
        self.stem_tension_n = torch.zeros(self.num_envs, device=self.device)
        self.stem_extension_m = torch.zeros(self.num_envs, device=self.device)
        self.detach_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.damage_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.slip_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.crushed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.slipped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.collided_obstacle = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_detached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.peak_grip_force_n = torch.zeros(self.num_envs, device=self.device)
        self.peak_stem_tension_n = torch.zeros(self.num_envs, device=self.device)
        self.peak_obstacle_contact_force_n = torch.zeros(self.num_envs, device=self.device)
        self.expert_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.expert_phase_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._expert_contact_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cached_flags: dict[str, torch.Tensor] | None = None
        self._cached_flags_step = -1

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.grape = RigidObject(self.cfg.grape)
        self.stem_visual = RigidObject(self.cfg.stem_visual)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["grape"] = self.grape
        self.scene.rigid_objects["stem_visual"] = self.stem_visual

        decorative_cfgs = {
            "vine_trunk": self.cfg.branch,
            "lateral_branch": self.cfg.lateral_branch,
            "trellis_left": self.cfg.trellis_left,
            "trellis_right": self.cfg.trellis_right,
            "backdrop": self.cfg.backdrop,
        }
        for index, asset_cfg in enumerate(self.cfg.background_grapes):
            decorative_cfgs[f"background_grape_{index}"] = asset_cfg
        self._decorative_objects: dict[str, RigidObject] = {}
        for name, asset_cfg in decorative_cfgs.items():
            rigid_object = RigidObject(asset_cfg)
            self._decorative_objects[name] = rigid_object
            self.scene.rigid_objects[name] = rigid_object

        self._spawn_leaf_visuals()
        self.scene.clone_environments(copy_from_source=False)

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

        self.left_contact_sensor = ContactSensor(self.cfg.left_contact)
        self.right_contact_sensor = ContactSensor(self.cfg.right_contact)
        self.scene.sensors["left_grape_contact"] = self.left_contact_sensor
        self.scene.sensors["right_grape_contact"] = self.right_contact_sensor
        self.obstacle_contact_body_names = (
            "panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
            "gelsight_mini_case_left",
            "gelsight_mini_case_right",
            "gelpad_left",
            "gelpad_right",
        )
        self.obstacle_contact_prim_names = (
            "vine_trunk",
            "target_lateral_branch",
            "trellis_left",
            "trellis_right",
        )
        self.robot_obstacle_contact_sensors: list[ContactSensor] = []
        for body_name in self.obstacle_contact_body_names:
            sensor = ContactSensor(
                self.cfg.robot_obstacle_contact.replace(
                    prim_path=f"/World/envs/env_.*/Robot/{body_name}"
                )
            )
            self.robot_obstacle_contact_sensors.append(sensor)
            self.scene.sensors[f"{body_name}_obstacle_contact"] = sensor

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
        self.cfg.dome_light.spawn.func(self.cfg.dome_light.prim_path, self.cfg.dome_light.spawn)
        self.cfg.key_light.spawn.func(
            self.cfg.key_light.prim_path,
            self.cfg.key_light.spawn,
            translation=self.cfg.key_light.init_state.pos,
            orientation=self.cfg.key_light.init_state.rot,
        )

    def _spawn_leaf_visuals(self):
        """Create lightweight, two-sided almond meshes for the vineyard set."""

        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdGeom, UsdShade
        except Exception:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        for index, (center, angle, length, width, color) in enumerate(self.cfg.leaf_visuals):
            center_v = torch.tensor(center, dtype=torch.float64)
            cosine = float(torch.cos(torch.tensor(angle)).item())
            sine = float(torch.sin(torch.tensor(angle)).item())
            major = torch.tensor((cosine, 0.0, sine), dtype=torch.float64)
            minor = torch.tensor((-sine, 0.0, cosine), dtype=torch.float64)
            outline = (
                (-0.50, 0.00),
                (-0.30, 0.36),
                (0.00, 0.50),
                (0.30, 0.36),
                (0.50, 0.00),
                (0.30, -0.36),
                (0.00, -0.50),
                (-0.30, -0.36),
            )
            points = []
            for u, v in outline:
                point = center_v + major * (u * length) + minor * (v * width)
                # A shallow fold catches the key light and reads as foliage.
                point[1] -= 0.006 * (1.0 - abs(2.0 * u))
                points.append(Gf.Vec3f(*map(float, point.tolist())))

            path = f"/World/envs/env_0/leaf_mesh_{index}"
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr([len(points)])
            mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)

            material = UsdShade.Material.Define(stage, f"{path}/Looks/LeafMaterial")
            shader = UsdShade.Shader.Define(stage, f"{path}/Looks/LeafMaterial/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]
        root_rot_b = math_utils.matrix_from_quat(math_utils.quat_inv(self._robot.data.root_link_quat_w))
        jacobian[:, :3, :] = torch.bmm(root_rot_b, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(root_rot_b, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        hand_pos_w = self._robot.data.body_link_pos_w[:, self._hand_body_idx]
        hand_quat_w = self._robot.data.body_link_quat_w[:, self._hand_body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        hand_pos_b, hand_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, hand_pos_w, hand_quat_w
        )
        return math_utils.combine_frame_transforms(
            hand_pos_b,
            hand_quat_b,
            self._tool_offset_pos,
            self._tool_offset_quat,
        )

    def _compute_frame_jacobian(self) -> torch.Tensor:
        jacobian = self.jacobian_b
        jacobian[:, :3, :] += torch.bmm(
            -math_utils.skew_symmetric_matrix(self._tool_offset_pos), jacobian[:, 3:, :]
        )
        return jacobian

    def _calibrate_gripper_center_offset(self, env_ids: torch.Tensor):
        left_pos_w = self._robot.data.body_link_pos_w[:, self._left_pad_body_idx]
        right_pos_w = self._robot.data.body_link_pos_w[:, self._right_pad_body_idx]
        midpoint_w = 0.5 * (left_pos_w + right_pos_w)
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        midpoint_b = math_utils.quat_apply(math_utils.quat_inv(root_quat_w), midpoint_w - root_pos_w)
        tool_pos_b, tool_quat_b = self._compute_frame_pose()
        offset_b = midpoint_b - tool_pos_b
        local_offset = vector_in_local_frame(offset_b, tool_quat_b)
        self.gripper_center_offset_tool[env_ids] = local_offset[env_ids]

    def _gripper_center_pos_b(self) -> torch.Tensor:
        tool_pos_b, tool_quat_b = self._compute_frame_pose()
        return tool_pos_b + math_utils.quat_apply(tool_quat_b, self.gripper_center_offset_tool)

    def _pre_physics_step(self, actions: torch.Tensor | None):
        if actions is not None and actions.numel() > 0:
            if actions.shape[-1] != 10:
                raise ValueError(f"GrapePick expects 10-D CAFE actions, received {tuple(actions.shape)}")
            target_pos_b = torch.minimum(
                torch.maximum(actions[:, :3], self.workspace_min_b), self.workspace_max_b
            )
            target_quat_b = self._rot6d_to_quat(actions[:, 3:9])
            width = actions[:, 9:10].clamp(self.cfg.expert_min_width_m, self.cfg.expert_open_width_m)
            self.ik_commands[:, :3] = target_pos_b
            self.ik_commands[:, 3:7] = target_quat_b
            self.gripper_width[:] = width
            self.last_target_pos_b[:] = target_pos_b
            self.last_target_quat_b[:] = target_quat_b

        self._update_stem_tether()
        self._ik_controller.set_command(self.ik_commands)

    def _apply_action(self):
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos.clone()
        if torch.linalg.norm(ee_pos_b) > 0.0:
            joint_pos_des = self._ik_controller.compute(
                ee_pos_b,
                ee_quat_b,
                self._compute_frame_jacobian(),
                joint_pos,
            )
        else:
            joint_pos_des = joint_pos
        joint_pos_des[:, self._finger_joint_ids] = self.gripper_width
        self._robot.set_joint_position_target(joint_pos_des)
        self.step_count += 1
        self._cached_flags_step = -1

    def _update_stem_tether(self):
        force_w, raw_tension, extension = compute_stem_tether_force(
            self.grape.data.root_pos_w,
            self.grape.data.root_lin_vel_w,
            self.stem_anchor_pos_w,
            rest_length_m=self.cfg.stem_rest_length_m,
            stiffness_n_per_m=self.cfg.stem_stiffness_n_per_m,
            damping_n_s_per_m=self.cfg.stem_damping_n_s_per_m,
            max_tension_n=self.cfg.stem_max_tension_n,
        )
        alpha = float(self.cfg.stem_force_filter_alpha)
        filtered = alpha * raw_tension + (1.0 - alpha) * self.stem_tension_n
        filtered = torch.where(self.stem_attached, filtered, torch.zeros_like(filtered))
        self.detach_dwell, detach_now = update_threshold_dwell(
            self.detach_dwell,
            filtered,
            self.detach_threshold_n,
            required_steps=self.cfg.detach_dwell_steps,
        )
        detach_now &= self.stem_attached
        self.stem_attached &= ~detach_now
        self.ever_detached |= detach_now
        self.stem_tension_n[:] = torch.where(self.stem_attached, filtered, torch.zeros_like(filtered))
        self.stem_extension_m[:] = torch.where(self.stem_attached, extension, torch.zeros_like(extension))
        self.peak_stem_tension_n[:] = torch.maximum(self.peak_stem_tension_n, filtered)

        force_w = torch.where(self.stem_attached.unsqueeze(-1), force_w, torch.zeros_like(force_w))
        force_b = math_utils.quat_apply(math_utils.quat_inv(self.grape.data.root_quat_w), force_w)
        zero_torque = torch.zeros_like(force_b)
        self.grape.set_external_force_and_torque(force_b.unsqueeze(1), zero_torque.unsqueeze(1))

    @staticmethod
    def _force_from_sensor(sensor: ContactSensor, num_envs: int, device: str) -> torch.Tensor:
        if sensor.data.force_matrix_w is not None and sensor.data.force_matrix_w.numel() > 0:
            return sensor.data.force_matrix_w[:, 0, :, :].sum(dim=1).detach().clone().float()
        if sensor.data.net_forces_w is not None and sensor.data.net_forces_w.numel() > 0:
            return sensor.data.net_forces_w[:, 0, :].detach().clone().float()
        return torch.zeros((num_envs, 3), dtype=torch.float32, device=device)

    def get_contact_force_metrics(self) -> dict[str, torch.Tensor]:
        """Read only physical pad-to-grape contacts; there is no tactile fallback."""

        left_force_w = self._force_from_sensor(self.left_contact_sensor, self.num_envs, self.device)
        right_force_w = self._force_from_sensor(self.right_contact_sensor, self.num_envs, self.device)
        left_pos_w = self._robot.data.body_link_pos_w[:, self._left_pad_body_idx]
        right_pos_w = self._robot.data.body_link_pos_w[:, self._right_pad_body_idx]
        closing_axis_w = F.normalize(right_pos_w - left_pos_w, dim=-1, eps=1.0e-8)
        left_normal_n = torch.abs(torch.sum(left_force_w * closing_axis_w, dim=-1))
        right_normal_n = torch.abs(torch.sum(right_force_w * closing_axis_w, dim=-1))
        grip_force_n = torch.maximum(left_normal_n, right_normal_n)
        return {
            "left_force_w": left_force_w,
            "right_force_w": right_force_w,
            "left_normal_force_n": left_normal_n,
            "right_normal_force_n": right_normal_n,
            "grip_force_n": grip_force_n,
            "net_force_n": torch.linalg.norm(left_force_w + right_force_w, dim=-1),
            "bilateral_contact": (left_normal_n >= self.cfg.bilateral_contact_threshold_n)
            & (right_normal_n >= self.cfg.bilateral_contact_threshold_n),
        }

    def get_obstacle_contact_force(self) -> torch.Tensor:
        """Return the largest robot-to-vine/trellis contact load."""

        peak = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for sensor in self.robot_obstacle_contact_sensors:
            if sensor.data.force_matrix_w is not None and sensor.data.force_matrix_w.numel() > 0:
                contact_norm = torch.linalg.norm(sensor.data.force_matrix_w[:, 0, :, :], dim=-1).amax(dim=-1)
            else:
                contact_force_w = self._force_from_sensor(sensor, self.num_envs, self.device)
                contact_norm = torch.linalg.norm(contact_force_w, dim=-1)
            peak = torch.maximum(peak, contact_norm)
        return peak

    def get_obstacle_contact_force_details(self) -> dict[str, torch.Tensor]:
        """Return per-body and, when available, per-obstacle contact loads."""

        details: dict[str, torch.Tensor] = {}
        for body_name, sensor in zip(
            self.obstacle_contact_body_names,
            self.robot_obstacle_contact_sensors,
            strict=True,
        ):
            if sensor.data.force_matrix_w is not None and sensor.data.force_matrix_w.numel() > 0:
                norms = torch.linalg.norm(sensor.data.force_matrix_w[:, 0, :, :], dim=-1)
                details[body_name] = norms.amax(dim=-1)
                if norms.shape[-1] == len(self.obstacle_contact_prim_names):
                    for obstacle_index, obstacle_name in enumerate(self.obstacle_contact_prim_names):
                        details[f"{body_name}->{obstacle_name}"] = norms[:, obstacle_index]
            else:
                force_w = self._force_from_sensor(sensor, self.num_envs, self.device)
                details[body_name] = torch.linalg.norm(force_w, dim=-1)
        return details

    def get_cafe_ft(self) -> torch.Tensor:
        metrics = self.get_contact_force_metrics()
        left_force_w = metrics["left_force_w"]
        right_force_w = metrics["right_force_w"]
        force_w = left_force_w + right_force_w
        left_pos_w = self._robot.data.body_link_pos_w[:, self._left_pad_body_idx]
        right_pos_w = self._robot.data.body_link_pos_w[:, self._right_pad_body_idx]
        hand_pos_w = self._robot.data.body_link_pos_w[:, self._hand_body_idx]
        torque_w = torch.cross(left_pos_w - hand_pos_w, left_force_w, dim=-1)
        torque_w += torch.cross(right_pos_w - hand_pos_w, right_force_w, dim=-1)
        root_rot_b = math_utils.matrix_from_quat(math_utils.quat_inv(self._robot.data.root_link_quat_w))
        force_b = torch.bmm(root_rot_b, force_w.unsqueeze(-1)).squeeze(-1)
        torque_b = torch.bmm(root_rot_b, torque_w.unsqueeze(-1)).squeeze(-1)
        return torch.cat((force_b, torque_b), dim=-1).detach().clone().float()

    def _compute_task_flags(self) -> dict[str, torch.Tensor]:
        metrics = self.get_contact_force_metrics()
        grip_force_n = metrics["grip_force_n"]
        obstacle_contact_force_n = self.get_obstacle_contact_force()
        self.peak_grip_force_n[:] = torch.maximum(self.peak_grip_force_n, grip_force_n)
        self.peak_obstacle_contact_force_n[:] = torch.maximum(
            self.peak_obstacle_contact_force_n, obstacle_contact_force_n
        )
        self.collided_obstacle |= obstacle_contact_force_n >= self.cfg.obstacle_contact_threshold_n
        self.damage_dwell, damage_now = update_threshold_dwell(
            self.damage_dwell,
            grip_force_n,
            self.damage_threshold_n,
            required_steps=self.cfg.damage_dwell_steps,
        )
        self.crushed |= damage_now

        grape_pos_b = self.grape.data.root_pos_w - self._robot.data.root_link_pos_w
        retention_distance = torch.linalg.norm(grape_pos_b - self._gripper_center_pos_b(), dim=-1)
        dropped = self.grape.data.root_pos_w[:, 2] < self.cfg.terminate_drop_height_m
        retention_lost = (~self.stem_attached) & (
            dropped | (retention_distance > self.cfg.retention_distance_m)
        )
        self.slip_dwell = torch.where(retention_lost, self.slip_dwell + 1, torch.zeros_like(self.slip_dwell))
        self.slipped |= self.slip_dwell >= max(int(self.cfg.slip_dwell_steps), 1)

        retained = (
            (~self.stem_attached)
            & metrics["bilateral_contact"]
            & ~self.crushed
            & ~self.slipped
            & ~self.collided_obstacle
        )
        self.success_dwell = torch.where(retained, self.success_dwell + 1, torch.zeros_like(self.success_dwell))
        success = self.success_dwell >= max(int(self.cfg.success_hold_steps), 1)

        ee_pos_b, _ = self._compute_frame_pose()
        workspace_failure = torch.any((ee_pos_b < self.workspace_min_b) | (ee_pos_b > self.workspace_max_b), dim=-1)
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return {
            **metrics,
            "obstacle_contact_force_n": obstacle_contact_force_n,
            "obstacle_collision": self.collided_obstacle.clone(),
            "stem_attached": self.stem_attached.clone(),
            "detached": (~self.stem_attached).clone(),
            "stem_tension_n": self.stem_tension_n.clone(),
            "stem_extension_m": self.stem_extension_m.clone(),
            "detach_threshold_n": self.detach_threshold_n.clone(),
            "damage_threshold_n": self.damage_threshold_n.clone(),
            "crushed": self.crushed.clone(),
            "slipped": self.slipped.clone(),
            "dropped": dropped,
            "workspace_failure": workspace_failure,
            "success": success,
            "timeout": timeout,
            "no_detach": timeout & self.stem_attached,
            "retention_distance_m": retention_distance,
        }

    def get_task_flags(self) -> dict[str, torch.Tensor]:
        step = int(self.step_count.max().item())
        if self._cached_flags is None or self._cached_flags_step != step:
            self._cached_flags = self._compute_task_flags()
            self._cached_flags_step = step
        return self._cached_flags

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        flags = self.get_task_flags()
        terminated = (
            flags["crushed"]
            | flags["slipped"]
            | flags["workspace_failure"]
            | flags["obstacle_collision"]
        )
        if self.cfg.terminate_on_success:
            terminated |= flags["success"]
        return terminated, flags["timeout"]

    def _get_rewards(self) -> torch.Tensor:
        flags = self.get_task_flags()
        grape_pos_b = self.grape.data.root_pos_w - self._robot.data.root_link_pos_w
        reach_distance = torch.linalg.norm(grape_pos_b - self._gripper_center_pos_b(), dim=-1)
        safe_force = torch.clamp(
            flags["grip_force_n"] / max(self.cfg.damage_grip_force_threshold_n, 1.0e-6), 0.0, 2.0
        )
        reward = -2.0 * reach_distance
        reward += 0.5 * flags["bilateral_contact"].float()
        reward += 2.0 * flags["detached"].float()
        reward += 10.0 * flags["success"].float()
        reward -= 0.1 * safe_force.square()
        reward -= 10.0 * (
            flags["crushed"]
            | flags["slipped"]
            | flags["workspace_failure"]
            | flags["obstacle_collision"]
        ).float()
        self.extras["log"] = {
            "GrapePick/grip_force_n": flags["grip_force_n"].mean(),
            "GrapePick/stem_tension_n": flags["stem_tension_n"].mean(),
            "GrapePick/detached": flags["detached"].float().mean(),
            "GrapePick/success": flags["success"].float().mean(),
            "GrapePick/crushed": flags["crushed"].float().mean(),
            "GrapePick/slipped": flags["slipped"].float().mean(),
            "GrapePick/obstacle_collision": flags["obstacle_collision"].float().mean(),
        }
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = self.get_cafe_observation()
        return {"policy": torch.cat((obs["robot0_pos"], obs["robot0_force"]), dim=-1)}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        grape_state = self.grape.data.default_root_state[env_ids].clone()
        grape_state[:, :3] += self.scene.env_origins[env_ids]
        grape_state[:, 7:] = 0.0
        if self.cfg.randomize_grape_pose:
            random_range = torch.tensor(
                self.cfg.grape_position_randomization_m,
                dtype=grape_state.dtype,
                device=self.device,
            )
            grape_state[:, :3] += (
                2.0 * torch.rand((len(env_ids), 3), device=self.device, dtype=grape_state.dtype) - 1.0
            ) * random_range
        self.grape.write_root_state_to_sim(grape_state, env_ids=env_ids)

        anchor_offset = torch.tensor(self.cfg.stem_anchor_offset_m, device=self.device, dtype=grape_state.dtype)
        anchor_pos_w = grape_state[:, :3] + anchor_offset

        # Move the upright target vine and its lateral branch with the
        # randomized fruit so the pedicel remains visually connected.
        vine_trunk = self._decorative_objects["vine_trunk"]
        vine_state = vine_trunk.data.default_root_state[env_ids].clone()
        vine_state[:, :3] += self.scene.env_origins[env_ids]
        vine_state[:, 0] = anchor_pos_w[:, 0] + self.cfg.lateral_branch_length_m
        vine_state[:, 1] = anchor_pos_w[:, 1]
        vine_state[:, 7:] = 0.0
        vine_trunk.write_root_pose_to_sim(vine_state[:, :7], env_ids=env_ids)

        lateral_branch = self._decorative_objects["lateral_branch"]
        lateral_state = lateral_branch.data.default_root_state[env_ids].clone()
        lateral_state[:, :3] = anchor_pos_w
        lateral_state[:, 0] += 0.5 * self.cfg.lateral_branch_length_m
        lateral_state[:, 7:] = 0.0
        lateral_branch.write_root_pose_to_sim(lateral_state[:, :7], env_ids=env_ids)

        stem_state = self.stem_visual.data.default_root_state[env_ids].clone()
        stem_state[:, :3] = 0.5 * (grape_state[:, :3] + anchor_pos_w)
        stem_state[:, 7:] = 0.0
        self.stem_visual.write_root_pose_to_sim(stem_state[:, :7], env_ids=env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self.grape_reset_pos_w[env_ids] = grape_state[:, :3]
        self.grape_reset_quat_w[env_ids] = grape_state[:, 3:7]
        self.initial_grape_pos_b[env_ids] = grape_state[:, :3] - self._robot.data.root_link_pos_w[env_ids]
        self.stem_anchor_pos_w[env_ids] = anchor_pos_w
        pull_direction_w = F.normalize(grape_state[:, :3] - anchor_pos_w, dim=-1, eps=1.0e-8)
        self.pull_direction_b[env_ids] = math_utils.quat_apply(
            math_utils.quat_inv(self._robot.data.root_link_quat_w[env_ids]), pull_direction_w
        )
        self.stem_attached[env_ids] = True
        if self.cfg.randomize_force_thresholds:
            detach_low, detach_high = self.cfg.detach_force_randomization_range_n
            damage_low, damage_high = self.cfg.damage_force_randomization_range_n
            self.detach_threshold_n[env_ids] = detach_low + (detach_high - detach_low) * torch.rand(
                len(env_ids), device=self.device
            )
            self.damage_threshold_n[env_ids] = damage_low + (damage_high - damage_low) * torch.rand(
                len(env_ids), device=self.device
            )
        else:
            self.detach_threshold_n[env_ids] = self.cfg.detach_force_threshold_n
            self.damage_threshold_n[env_ids] = self.cfg.damage_grip_force_threshold_n
        self.stem_tension_n[env_ids] = 0.0
        self.stem_extension_m[env_ids] = 0.0
        self.detach_dwell[env_ids] = 0
        self.damage_dwell[env_ids] = 0
        self.slip_dwell[env_ids] = 0
        self.success_dwell[env_ids] = 0
        self.crushed[env_ids] = False
        self.slipped[env_ids] = False
        self.collided_obstacle[env_ids] = False
        self.ever_detached[env_ids] = False
        self.peak_grip_force_n[env_ids] = 0.0
        self.peak_stem_tension_n[env_ids] = 0.0
        self.peak_obstacle_contact_force_n[env_ids] = 0.0
        self.step_count[env_ids] = 0
        self.expert_phase[env_ids] = 0
        self.expert_phase_step[env_ids] = 0
        self._expert_contact_dwell[env_ids] = 0
        self.gripper_width[env_ids] = self.cfg.expert_open_width_m
        self._cached_flags = None
        self._cached_flags_step = -1

        zero_force = torch.zeros((len(env_ids), 1, 3), device=self.device)
        self.grape.set_external_force_and_torque(zero_force, zero_force, env_ids=env_ids)
        self._calibrate_gripper_center_offset(env_ids)
        self.reset_control_target(env_ids)
        if self.cfg.enable_gelsight:
            self.gsmini_left.reset(env_ids=env_ids)
            self.gsmini_right.reset(env_ids=env_ids)

    def reset_control_target(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        ee_pos_b, _ = self._compute_frame_pose()
        self.ik_commands[env_ids, :3] = ee_pos_b[env_ids]
        self.ik_commands[env_ids, 3:7] = self.nominal_ee_quat_b[env_ids]
        self.last_target_pos_b[env_ids] = ee_pos_b[env_ids]
        self.last_target_quat_b[env_ids] = self.nominal_ee_quat_b[env_ids]

    def command_expert(self):
        """Run a force-limited approach, close, pull and retention controller."""

        metrics = self.get_contact_force_metrics()
        bilateral = metrics["bilateral_contact"]
        grip_force = metrics["grip_force_n"]
        current_width = self.gripper_width[:, :1].clone()
        phase = self.expert_phase

        # Transition the state machine.  All thresholds are consecutive-step
        # thresholds so isolated force/contact spikes cannot advance it.
        self._expert_contact_dwell = torch.where(
            bilateral & (grip_force >= 2.0),
            self._expert_contact_dwell + 1,
            torch.zeros_like(self._expert_contact_dwell),
        )
        transition = torch.zeros_like(phase, dtype=torch.bool)
        transition |= (phase == 0) & (self.expert_phase_step >= 90)
        transition |= (phase == 1) & (self.expert_phase_step >= 180)
        transition |= (phase == 2) & (
            (self._expert_contact_dwell >= 8) | (self.expert_phase_step >= 360)
        )
        transition |= (phase == 3) & (self.expert_phase_step >= 60)
        transition |= (phase == 4) & (self.expert_phase_step >= 300)
        next_phase = torch.minimum(phase + transition.long(), torch.full_like(phase, 5))
        self.expert_phase_step = torch.where(
            transition, torch.zeros_like(self.expert_phase_step), self.expert_phase_step + 1
        )
        self.expert_phase[:] = next_phase
        phase = self.expert_phase

        center_target_b = self.initial_grape_pos_b.clone()
        approach_mask = phase == 0
        center_target_b[approach_mask, 0] -= self.cfg.expert_approach_distance_m
        align_mask = phase == 1
        align_progress = (self.expert_phase_step.float() / 180.0).clamp(0.0, 1.0)
        center_target_b[align_mask, 0] -= self.cfg.expert_approach_distance_m * (
            1.0 - align_progress[align_mask]
        )
        pull_mask = phase >= 4
        pull_progress = torch.where(
            phase == 4,
            (self.expert_phase_step.float() / 300.0).clamp(0.0, 1.0),
            torch.ones_like(self.expert_phase_step, dtype=torch.float32),
        )
        center_target_b[pull_mask] += (
            self.pull_direction_b[pull_mask]
            * (self.cfg.expert_pull_distance_m * pull_progress[pull_mask]).unsqueeze(-1)
        )

        open_mask = phase <= 1
        current_width[open_mask] = self.cfg.expert_open_width_m
        regulate_mask = phase >= 2
        seek_contact = regulate_mask & ~bilateral
        close_more = regulate_mask & bilateral & (grip_force < self.cfg.target_grip_force_n - 0.20)
        open_more = regulate_mask & (grip_force > self.cfg.target_grip_force_n + 0.20)
        current_width[seek_contact] -= self.cfg.expert_close_rate_m_per_step
        current_width[close_more] -= self.cfg.expert_force_adjust_rate_m_per_step
        current_width[open_more] += self.cfg.expert_force_adjust_rate_m_per_step
        current_width.clamp_(self.cfg.expert_min_width_m, self.cfg.expert_open_width_m)

        target_quat_b = self.nominal_ee_quat_b
        target_pos_b = centered_tool_target(
            center_target_b,
            target_quat_b,
            self.gripper_center_offset_tool,
        )
        target_pos_b = torch.minimum(torch.maximum(target_pos_b, self.workspace_min_b), self.workspace_max_b)
        self.ik_commands[:, :3] = target_pos_b
        self.ik_commands[:, 3:7] = target_quat_b
        self.gripper_width[:] = current_width
        self.last_target_pos_b[:] = target_pos_b
        self.last_target_quat_b[:] = target_quat_b

    @staticmethod
    def _rot6d_to_quat(rot6d: torch.Tensor) -> torch.Tensor:
        matrix_3x2 = rot6d.reshape(-1, 3, 2)
        first = F.normalize(matrix_3x2[:, :, 0], dim=-1, eps=1.0e-6)
        second_raw = matrix_3x2[:, :, 1]
        second = F.normalize(
            second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first,
            dim=-1,
            eps=1.0e-6,
        )
        third = torch.cross(first, second, dim=-1)
        return math_utils.quat_from_matrix(torch.stack((first, second, third), dim=-1))

    @staticmethod
    def _quat_to_rot6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
        return math_utils.matrix_from_quat(quat_wxyz)[:, :, :2].reshape(quat_wxyz.shape[0], 6)

    def get_cafe_observation(self) -> dict[str, torch.Tensor]:
        tool_pos_b, tool_quat_b = self._compute_frame_pose()
        return {
            "robot0_pos": torch.cat(
                (tool_pos_b, self._quat_to_rot6d(tool_quat_b), self.gripper_width[:, :1]), dim=-1
            ).detach().clone(),
            "robot0_force": self.get_cafe_ft(),
        }

    def get_cafe_action(self) -> torch.Tensor:
        return torch.cat(
            (
                self.last_target_pos_b,
                self._quat_to_rot6d(self.last_target_quat_b),
                self.gripper_width[:, :1],
            ),
            dim=-1,
        ).detach().clone()

    def get_expert_phase_names(self) -> list[str]:
        return [self.PHASE_NAMES[int(index)] for index in self.expert_phase.detach().cpu().tolist()]

    def tactile_contact_depths(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.enable_gelsight:
            zeros = torch.zeros(self.num_envs, device=self.device)
            return zeros, zeros
        left = self.gsmini_left.indentation_depth
        right = self.gsmini_right.indentation_depth
        if left is None or right is None:
            zeros = torch.zeros(self.num_envs, device=self.device)
            return zeros, zeros
        return left.float(), right.float()

    def get_camera_rgb(self, camera: str = "third") -> torch.Tensor:
        if not self.cfg.enable_cameras:
            raise RuntimeError("Camera rendering is disabled in GrapePickEnvCfg.")
        sensor = self.third_person_camera if camera == "third" else self.wrist_camera
        return sensor.data.output["rgb"][..., :3].detach().clone()

    def get_tactile_rgb(self, side: str) -> torch.Tensor:
        if not self.cfg.enable_gelsight:
            raise RuntimeError("GelSight rendering is disabled in GrapePickEnvCfg.")
        sensor = self.gsmini_left if side == "left" else self.gsmini_right
        image = sensor.data.output["tactile_rgb"].float()
        if image.max() <= 1.0:
            image = image * 255.0
        return image.clamp(0, 255).byte().detach().clone()

    def get_tactile_height_map(self, side: str) -> torch.Tensor:
        if not self.cfg.enable_gelsight:
            raise RuntimeError("GelSight rendering is disabled in GrapePickEnvCfg.")
        sensor = self.gsmini_left if side == "left" else self.gsmini_right
        return sensor.data.output["height_map"].float().detach().clone()

    def save_scene_preview(self, output_file: str | Path):
        from PIL import Image

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.get_camera_rgb("third")[0].cpu().numpy()
        Image.fromarray(frame).save(output_path)
