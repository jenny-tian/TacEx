from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, TiledCamera, save_images_to_file

from tacex import GelSightSensor

from .grasp_geometry import centered_tool_target, vector_in_local_frame, yaw_aligned_gripper_quat
from .lab_pick_env_cfg import LabPickEnvCfg
from .visuals import SLIDE_VISUAL_DIFFUSE_COLOR, SLIDE_VISUAL_OPACITY, SLIDE_VISUAL_ROUGHNESS


class LabPickEnv(DirectRLEnv):
    cfg: LabPickEnvCfg

    def __init__(self, cfg: LabPickEnvCfg, render_mode: str | None = None, **kwargs):
        self.labware_name = cfg.labware_name
        self.labware_cfg = getattr(cfg, cfg.labware_name)
        super().__init__(cfg, render_mode, **kwargs)

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1
        left_finger_body_ids, _ = self._robot.find_bodies("gelpad_left")
        right_finger_body_ids, _ = self._robot.find_bodies("gelpad_right")
        self._left_finger_body_idx = left_finger_body_ids[0]
        self._right_finger_body_idx = right_finger_body_ids[0]
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])

        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.gripper_width = torch.full((self.num_envs, len(self._finger_joint_ids)), 0.04, device=self.device)
        self.initial_object_height = self.labware.data.root_pos_w[:, 2].clone()
        self.initial_object_pos_b = self.labware.data.root_pos_w - self._robot.data.root_link_pos_w
        self.has_touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_target_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_target_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.last_target_quat_b[:, 0] = 1.0
        self.nominal_ee_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.nominal_ee_quat_b[:, 0] = 1.0
        self.scripted_target_quat_b = self.nominal_ee_quat_b.clone()
        self.gripper_center_offset_tool = torch.zeros((self.num_envs, 3), device=self.device)
        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.workspace_min_b = torch.tensor([0.25, -0.35, 0.015], device=self.device)
        self.workspace_max_b = torch.tensor([0.78, 0.35, 0.50], device=self.device)
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.labware_reset_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.labware_reset_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.labware_reset_quat_w[:, 0] = 1.0
        self.last_object_pos_b = self.initial_object_pos_b.clone()
        self.reset_hold_remaining_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_hold_root_state = torch.zeros_like(self.labware.data.default_root_state)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.labware = RigidObject(self.labware_cfg)
        self.scene.rigid_objects["labware"] = self.labware
        self.plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = self.plate
        self.labware_support = RigidObject(self.cfg.labware_support)
        self.scene.rigid_objects["labware_support"] = self.labware_support

        self.scene.clone_environments(copy_from_source=False)
        self._spawn_labware_visuals()
        self._force_non_labware_visuals_opaque()

        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.third_person_camera = TiledCamera(self.cfg.third_person_camera)
        self.scene.sensors["wrist_camera"] = self.wrist_camera
        self.scene.sensors["third_person_camera"] = self.third_person_camera

        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_left"] = self.gsmini_left
        self.scene.sensors["gsmini_right"] = self.gsmini_right

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
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )
        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)

    def _spawn_labware_visuals(self):
        if self.labware_name != "slide":
            return

        glass_visual = sim_utils.MeshCuboidCfg(
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

    def _force_non_labware_visuals_opaque(self):
        try:
            import omni.usd
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
        except Exception:
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        opaque_robot_material = self._create_opaque_override_material(
            stage,
            material_path="/World/Looks/LabPickOpaqueRobotMaterial",
            color=Gf.Vec3f(0.86, 0.86, 0.84),
            roughness=0.55,
        )
        visited_materials = set()
        for prim in stage.Traverse():
            prim_path = prim.GetPath().pathString
            if self._is_labware_visual_path(prim_path):
                continue

            if "/Robot" in prim_path:
                self._bind_opaque_material(prim, opaque_robot_material)
            self._set_prim_opacity_attr(prim, opaque=True, vt_module=Vt)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not material or not material.GetPrim().IsValid():
                continue

            material_path = material.GetPath().pathString
            if material_path in visited_materials:
                continue
            visited_materials.add(material_path)

            for shader_prim in Usd.PrimRange(material.GetPrim()):
                if not shader_prim.IsA(UsdShade.Shader):
                    continue
                shader = UsdShade.Shader(shader_prim)
                for input_name in ("opacity", "opacity_constant"):
                    shader_input = shader.GetInput(input_name)
                    if shader_input:
                        shader_input.Set(1.0)
                for input_name in ("enable_opacity", "enable_opacity_texture"):
                    shader_input = shader.GetInput(input_name)
                    if shader_input:
                        shader_input.Set(False)

                for attr in shader_prim.GetAttributes():
                    attr_name = attr.GetName()
                    if attr_name in {"inputs:opacity", "inputs:opacity_constant"}:
                        attr.Set(1.0)
                    elif attr_name in {"inputs:enable_opacity", "inputs:enable_opacity_texture"}:
                        attr.Set(False)
                    elif attr.GetTypeName() in {Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Double}:
                        if attr_name.endswith(":opacity") or attr_name.endswith(":opacity_constant"):
                            attr.Set(1.0)

    def _create_opaque_override_material(self, stage, *, material_path: str, color, roughness: float):
        from pxr import Sdf, UsdShade

        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _bind_opaque_material(self, prim, material):
        from pxr import UsdGeom, UsdShade

        if not material or not material.GetPrim().IsValid():
            return
        if not prim.IsA(UsdGeom.Imageable):
            return
        UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)

    def _is_labware_visual_path(self, prim_path: str) -> bool:
        return "labware" in prim_path.split("/")

    def _set_prim_opacity_attr(self, prim, *, opaque: bool, vt_module=None):
        attr = prim.GetAttribute("primvars:displayOpacity")
        if not attr:
            return
        current_value = attr.Get()
        opacity = 1.0 if opaque else SLIDE_VISUAL_OPACITY
        type_name = str(attr.GetTypeName())
        if vt_module is not None and type_name == "float[]":
            length = len(current_value) if current_value is not None else 1
            attr.Set(vt_module.FloatArray([float(opacity)] * max(length, 1)))
            return
        if vt_module is not None and type_name == "double[]":
            length = len(current_value) if current_value is not None else 1
            attr.Set(vt_module.DoubleArray([float(opacity)] * max(length, 1)))
            return
        if isinstance(current_value, (list, tuple)):
            attr.Set([float(opacity) for _ in current_value])
            return
        attr.Set(float(opacity))

    def _pre_physics_step(self, actions: torch.Tensor | None):
        if actions is not None and actions.numel() > 0:
            target_pos_b = torch.minimum(torch.maximum(actions[:, :3], self.workspace_min_b), self.workspace_max_b)
            self.ik_commands[:, :3] = target_pos_b
            self.ik_commands[:, 3:7] = self.nominal_ee_quat_b
            self.gripper_width[:] = actions[:, 9:10].clamp(0.0, 0.04)
            self.last_target_pos_b[:] = target_pos_b
            self.last_target_quat_b[:] = self.nominal_ee_quat_b
        self._hold_labware_at_reset_pose()
        self._ik_controller.set_command(self.ik_commands)

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

    def _hold_labware_at_reset_pose(self):
        hold_env_ids = torch.nonzero(self.reset_hold_remaining_steps > 0, as_tuple=False).squeeze(-1)
        if hold_env_ids.numel() == 0:
            return

        root_state = self._reset_hold_root_state[hold_env_ids].clone()
        root_state[:, :3] = self.labware_reset_pos_w[hold_env_ids]
        root_state[:, 3:7] = self.labware_reset_quat_w[hold_env_ids]
        root_state[:, 7:] = 0.0
        self.labware.write_root_state_to_sim(root_state, env_ids=hold_env_ids)
        self.reset_hold_remaining_steps[hold_env_ids] -= 1

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        object_pos_b = self.labware.data.root_pos_w - self._robot.data.root_link_pos_w
        object_drop_delta = self.labware.data.root_pos_w[:, 2] - self.initial_object_height
        object_dropped = object_drop_delta < -self.cfg.terminate_object_drop_height

        object_xy_delta = object_pos_b[:, :2] - self.initial_object_pos_b[:, :2]
        object_too_far = torch.linalg.norm(object_xy_delta, dim=1) > self.cfg.terminate_object_xy_distance

        ee_pos_b, _ = self._compute_frame_pose()
        workspace_min = self.workspace_min_b - self.cfg.terminate_ee_workspace_margin
        workspace_max = self.workspace_max_b + self.cfg.terminate_ee_workspace_margin
        ee_outside_workspace = torch.any((ee_pos_b < workspace_min) | (ee_pos_b > workspace_max), dim=1)

        left_touch, right_touch = self.tactile_contact_depths()
        touched = (left_touch > self.cfg.tactile_threshold_mm) | (right_touch > self.cfg.tactile_threshold_mm)
        self.has_touched |= touched
        ft = self.get_cafe_ft()
        force_norm = torch.linalg.norm(ft[:, :3], dim=1)
        object_broken = self.has_touched & (force_norm > self.cfg.terminate_break_force_threshold_n)

        terminated = object_dropped | object_too_far | ee_outside_workspace | object_broken
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        lift_delta = self.labware.data.root_pos_w[:, 2] - self.initial_object_height
        return torch.clamp(lift_delta, min=0.0)

    def _get_observations(self) -> dict:
        obs = self.get_cafe_observation()
        policy = torch.cat((obs["robot0_pos"], obs["robot0_force"]), dim=-1)
        return {"policy": policy}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        root_state = self.labware.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 7:] = 0.0
        support_state = self.labware_support.data.default_root_state[env_ids].clone()
        support_state[:, :3] += self.scene.env_origins[env_ids]
        support_state[:, 7:] = 0.0

        if self.cfg.randomize_labware_position:
            xy_range = torch.tensor(self.cfg.labware_pos_randomization_xy, dtype=root_state.dtype, device=self.device)
            xy_noise = (2.0 * torch.rand((len(env_ids), 2), dtype=root_state.dtype, device=self.device) - 1.0) * xy_range
            root_state[:, 0:2] += xy_noise

            yaw_range = self.cfg.labware_yaw_randomization
            yaw = (2.0 * torch.rand((len(env_ids),), dtype=root_state.dtype, device=self.device) - 1.0) * yaw_range
            yaw_quat = math_utils.quat_from_euler_xyz(
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                yaw,
            )
            root_state[:, 3:7] = math_utils.quat_mul(yaw_quat, root_state[:, 3:7])

        support_state[:, 0:2] = root_state[:, 0:2]
        support_state[:, 3:7] = root_state[:, 3:7]
        self.labware_support.write_root_state_to_sim(support_state, env_ids=env_ids)
        self.labware.write_root_state_to_sim(root_state, env_ids=env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self.has_touched[env_ids] = False
        self.initial_object_height[env_ids] = root_state[:, 2]
        self.initial_object_pos_b[env_ids] = root_state[:, :3] - self._robot.data.root_link_pos_w[env_ids]
        self.step_count[env_ids] = 0
        self.gripper_width[env_ids] = 0.04
        self.labware_reset_pos_w[env_ids] = root_state[:, :3]
        self.labware_reset_quat_w[env_ids] = root_state[:, 3:7]
        self.last_object_pos_b[env_ids] = self.initial_object_pos_b[env_ids]
        self._reset_hold_root_state[env_ids] = root_state
        self.reset_hold_remaining_steps[env_ids] = max(int(self.cfg.reset_hold_steps), 0)

        _, ee_quat_b = self._compute_frame_pose()
        self.nominal_ee_quat_b[env_ids] = ee_quat_b[env_ids]
        labware_quat_b = math_utils.quat_mul(
            math_utils.quat_inv(self._robot.data.root_link_quat_w[env_ids]),
            self.labware_reset_quat_w[env_ids],
        )
        self.scripted_target_quat_b[env_ids] = yaw_aligned_gripper_quat(
            self.nominal_ee_quat_b[env_ids],
            labware_quat_b,
            self.labware_name,
        )
        self._calibrate_gripper_center_offset(env_ids)
        self.reset_keyboard_target(env_ids)
        self.gsmini_left.reset(env_ids=env_ids)
        self.gsmini_right.reset(env_ids=env_ids)

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

    def _calibrate_gripper_center_offset(self, env_ids: torch.Tensor):
        left_pos_w = self._robot.data.body_link_pos_w[:, self._left_finger_body_idx]
        right_pos_w = self._robot.data.body_link_pos_w[:, self._right_finger_body_idx]
        midpoint_w = 0.5 * (left_pos_w + right_pos_w)
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        midpoint_b = math_utils.quat_apply(math_utils.quat_inv(root_quat_w), midpoint_w - root_pos_w)
        tool_pos_b, tool_quat_b = self._compute_frame_pose()
        offset_b = midpoint_b - tool_pos_b
        self.gripper_center_offset_tool[env_ids] = vector_in_local_frame(offset_b, tool_quat_b)[env_ids]

    def _compute_frame_jacobian(self) -> torch.Tensor:
        jacobian = self.jacobian_b
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian

    def _quat_to_rot6d(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        rot_mat = math_utils.matrix_from_quat(quat_wxyz)
        return rot_mat[:, :, :2].reshape(quat_wxyz.shape[0], 6)

    def get_cafe_observation(self) -> dict[str, torch.Tensor]:
        tool_pos_b, tool_quat_b = self._compute_frame_pose()
        tool_rot6d_b = self._quat_to_rot6d(tool_quat_b)

        return {
            # CAFE: xyz(3) + rot6d(6) + gripper_width(1)
            "robot0_pos": torch.cat((tool_pos_b, tool_rot6d_b, self.gripper_width[:, :1]), dim=-1).detach().clone(),
            # CAFE: Fx,Fy,Fz,Tx,Ty,Tz
            "robot0_force": self.get_cafe_ft(),
        }

    def get_cafe_ft(self) -> torch.Tensor:
        contact_ft = self._contact_sensor_ft()
        if contact_ft is not None:
            return contact_ft
        return self._indentation_ft()

    def _indentation_ft(self) -> torch.Tensor:
        left_force_n, right_force_n = self._estimate_contact_forces_from_tactile()
        ft = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        ft[:, 2] = left_force_n + right_force_n
        torque_y = self.cfg.contact_torque_arm_m * (right_force_n - left_force_n)
        ft[:, 4] = torque_y
        return ft.detach().clone()

    def _contact_force_from_sensor(self, sensor: ContactSensor) -> torch.Tensor:
        if sensor.data.force_matrix_w is not None:
            force_matrix_w = sensor.data.force_matrix_w
            if force_matrix_w.numel() > 0:
                return force_matrix_w[:, 0, :, :].sum(dim=1).detach().clone().float()
        if sensor.data.net_forces_w is not None:
            net_forces_w = sensor.data.net_forces_w
            if net_forces_w.numel() > 0:
                return net_forces_w[:, 0, :].detach().clone().float()
        return torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def _contact_sensor_ft(self) -> torch.Tensor | None:
        if not hasattr(self, "left_finger_contact_sensor") or not hasattr(self, "right_finger_contact_sensor"):
            return None
        left_force_w = self._contact_force_from_sensor(self.left_finger_contact_sensor)
        right_force_w = self._contact_force_from_sensor(self.right_finger_contact_sensor)
        force_w = left_force_w + right_force_w
        if not torch.any(torch.isfinite(force_w)):
            return None

        left_pos_w = self._robot.data.body_link_pos_w[:, self._left_finger_body_idx]
        right_pos_w = self._robot.data.body_link_pos_w[:, self._right_finger_body_idx]
        hand_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        torque_w = torch.cross(left_pos_w - hand_pos_w, left_force_w, dim=1)
        torque_w += torch.cross(right_pos_w - hand_pos_w, right_force_w, dim=1)

        root_rot_b = math_utils.matrix_from_quat(math_utils.quat_inv(self._robot.data.root_link_quat_w))
        force_b = torch.bmm(root_rot_b, force_w.unsqueeze(-1)).squeeze(-1)
        torque_b = torch.bmm(root_rot_b, torque_w.unsqueeze(-1)).squeeze(-1)
        return torch.cat((force_b, torque_b), dim=-1).detach().clone().float()

    def _estimate_contact_forces_from_tactile(self) -> tuple[torch.Tensor, torch.Tensor]:
        left_touch, right_touch = self.tactile_contact_depths()
        left_force_n = torch.clamp(left_touch, min=0.0) * self.cfg.contact_force_n_per_mm
        right_force_n = torch.clamp(right_touch, min=0.0) * self.cfg.contact_force_n_per_mm
        return left_force_n, right_force_n

    def get_cafe_marker2d(self) -> torch.Tensor:
        left_touch, right_touch = self.tactile_contact_depths()
        object_pos_b = self.labware.data.root_pos_w - self._robot.data.root_link_pos_w
        object_delta_b = object_pos_b - self.last_object_pos_b
        self.last_object_pos_b[:] = object_pos_b

        rows = torch.linspace(-1.0, 1.0, self.cfg.marker2d_rows, device=self.device)
        cols = torch.linspace(-1.0, 1.0, self.cfg.marker2d_cols, device=self.device)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")

        left_weight = self._marker2d_weight(grid_x, grid_y, center_x=-0.35, center_y=0.0)
        right_weight = self._marker2d_weight(grid_x, grid_y, center_x=0.35, center_y=0.0)

        depth = (
            left_touch[:, None, None] * left_weight[None, :, :]
            + right_touch[:, None, None] * right_weight[None, :, :]
        ) * self.cfg.marker2d_depth_scale
        shear_x = object_delta_b[:, 0, None, None] * self.cfg.marker2d_shear_scale
        shear_y = object_delta_b[:, 1, None, None] * self.cfg.marker2d_shear_scale
        dx = shear_x * (left_weight + right_weight)[None, :, :] + depth * grid_x[None, :, :]
        dy = shear_y * (left_weight + right_weight)[None, :, :] + depth * grid_y[None, :, :]
        marker2d = torch.stack((dx, dy), dim=-1)
        return marker2d.detach().clone().float()

    def _marker2d_weight(
        self, grid_x: torch.Tensor, grid_y: torch.Tensor, center_x: float, center_y: float
    ) -> torch.Tensor:
        dist2 = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
        return torch.exp(-dist2 / (2.0 * self.cfg.marker2d_sigma**2))

    def get_cafe_action(self) -> torch.Tensor:
        target_rot6d_b = self._quat_to_rot6d(self.last_target_quat_b)
        return torch.cat(
            (
                self.last_target_pos_b,
                target_rot6d_b,
                self.gripper_width[:, :1],
            ),
            dim=-1,
        ).detach().clone()

    def get_cafe_image(self) -> torch.Tensor:
        rgb = self.third_person_camera.data.output["rgb"][:, :, :, :3]
        rgb = rgb.permute(0, 3, 1, 2).float()
        rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        rgb = rgb.permute(0, 2, 3, 1).clamp(0, 255).byte()
        return rgb.detach().clone()

    def command_pick_state_machine(self):
        center_target_b = self.initial_object_pos_b.clone()
        target_quat_b = self.scripted_target_quat_b
        touch_left, touch_right = self.tactile_contact_depths()
        touched = (touch_left > self.cfg.tactile_threshold_mm) | (touch_right > self.cfg.tactile_threshold_mm)
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
            close_width = 0.012
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

        phase = int(self.step_count[0].item())
        lift_start = close_end + squeeze_steps
        if phase < 90:
            center_target_b[:, 2] += hover_height
            self.gripper_width[:] = 0.04
        elif phase < close_start:
            approach_progress = min((phase - 90) * 0.0005, hover_height - grasp_height)
            center_target_b[:, 2] += hover_height - approach_progress
            self.gripper_width[:] = 0.04
        elif phase < close_end:
            close_progress = min(max((phase - close_start) / max(close_end - close_start, 1), 0.0), 1.0)
            if self.labware_name == "cup":
                center_target_b[:, 2] += max(grasp_height - 0.0015 * close_progress, 0.0)
            else:
                center_target_b[:, 2] += grasp_height
            self.gripper_width[:] = 0.04 - (0.04 - close_width) * close_progress
        elif phase < close_end + squeeze_steps:
            center_target_b[:, 2] += grasp_height
            self.gripper_width[:] = close_width
        else:
            lift_progress = min(max((phase - lift_start) / max(self.cfg.scripted_lift_steps, 1), 0.0), 1.0)
            target_lift = torch.full((self.num_envs,), grasp_height, dtype=center_target_b.dtype, device=self.device)
            if bool(self.has_touched.any().item()):
                target_lift[self.has_touched] = grasp_height + (lift_height - grasp_height) * lift_progress
            center_target_b[:, 2] += target_lift
            self.gripper_width[:] = close_width

        target_pos_b = centered_tool_target(
            center_target_b,
            target_quat_b,
            self.gripper_center_offset_tool,
        )
        self.ik_commands[:, :3] = target_pos_b
        self.ik_commands[:, 3:7] = target_quat_b
        self.last_target_pos_b[:] = target_pos_b
        self.last_target_quat_b[:] = target_quat_b
        if self.cfg.scripted_lift_assist_on_contact and phase >= close_end + squeeze_steps:
            self._apply_scripted_lift_assist(center_target_b)

    def _apply_scripted_lift_assist(self, target_object_pos_b: torch.Tensor):
        lift_mask = self.has_touched
        if not bool(lift_mask.any().item()):
            return
        root_state = self.labware.data.root_state_w.clone()
        root_state[lift_mask, :3] = self._robot.data.root_link_pos_w[lift_mask] + target_object_pos_b[lift_mask]
        root_state[lift_mask, 7:] = 0.0
        env_ids = torch.nonzero(lift_mask, as_tuple=False).squeeze(-1)
        self.labware.write_root_state_to_sim(root_state[env_ids], env_ids=env_ids)

    def reset_keyboard_target(self, env_ids: torch.Tensor | None = None, open_width: float = 0.04):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        self.ik_commands[env_ids, :3] = ee_pos_b[env_ids]
        self.ik_commands[env_ids, 3:7] = ee_quat_b[env_ids]
        self.last_target_pos_b[env_ids] = ee_pos_b[env_ids]
        self.last_target_quat_b[env_ids] = ee_quat_b[env_ids]
        self.gripper_width[env_ids] = open_width

    def command_keyboard(
        self,
        delta_pose,
        close_gripper: bool,
        open_width: float = 0.04,
        close_width: float = 0.0,
    ):
        delta_pose_b = torch.as_tensor(delta_pose, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        target_pos_b, target_quat_b = math_utils.apply_delta_pose(
            self.ik_commands[:, :3], self.ik_commands[:, 3:7], delta_pose_b
        )
        target_pos_b = torch.minimum(torch.maximum(target_pos_b, self.workspace_min_b), self.workspace_max_b)

        self.ik_commands[:, :3] = target_pos_b
        self.ik_commands[:, 3:7] = target_quat_b
        self.gripper_width[:] = close_width if close_gripper else open_width

        touch_left, touch_right = self.tactile_contact_depths()
        touched = (touch_left > self.cfg.tactile_threshold_mm) | (touch_right > self.cfg.tactile_threshold_mm)
        self.has_touched |= touched
        self.last_target_pos_b[:] = target_pos_b
        self.last_target_quat_b[:] = target_quat_b

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
        step = int(self.step_count[0].item())
        wrist_rgb = self.wrist_camera.data.output["rgb"].float() / 255.0
        third_rgb = self.third_person_camera.data.output["rgb"].float() / 255.0
        save_images_to_file(wrist_rgb, str(output_dir / f"wrist_step_{step:05d}.png"))
        save_images_to_file(third_rgb, str(output_dir / f"third_step_{step:05d}.png"))
        viewer_rgb = self.render(recompute=True)
        if viewer_rgb is not None:
            viewer_tensor = torch.from_numpy(viewer_rgb).float().permute(2, 0, 1) / 255.0
            save_image(viewer_tensor, str(output_dir / f"viewer_step_{step:05d}.png"))

    def save_tactile_images(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        step = int(self.step_count[0].item())
        save_images_to_file(self._normalized_tactile_rgb(self.gsmini_left), str(output_dir / f"tactile_left_step_{step:05d}.png"))
        save_images_to_file(
            self._normalized_tactile_rgb(self.gsmini_right), str(output_dir / f"tactile_right_step_{step:05d}.png")
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
            raise RuntimeError("Viewer frame is unavailable. Use render_mode='rgb_array'.")
        return frame

    def print_state(self):
        tool_pos_b, _ = self._compute_frame_pose()
        object_pos_w = self.labware.data.root_pos_w
        left_touch, right_touch = self.tactile_contact_depths()
        print(
            "[STATE] "
            f"step={int(self.step_count[0].item())} "
            f"tool_pos_b={tool_pos_b[0].detach().cpu().numpy().round(4).tolist()} "
            f"object_pos={object_pos_w[0].detach().cpu().numpy().round(4).tolist()} "
            f"target_z={self.last_target_pos_b[0, 2].item():.4f} "
            f"finger_target={self.gripper_width[0].detach().cpu().numpy().round(4).tolist()} "
            f"indent_left_mm={left_touch[0].item():.4f} "
            f"indent_right_mm={right_touch[0].item():.4f} "
            f"contact={bool(self.has_touched[0].item())}"
        )
