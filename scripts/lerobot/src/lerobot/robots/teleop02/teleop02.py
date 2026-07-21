#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

import cv2
from mros.controller_msgs.msg import JointCmd, JointState
from mros.teleop_msgs.msg import TeleopMsg
from mros.sensor_msgs.msg import CompressedImage
from mros.hand_msgs.msg import HandCmd
from mros.hand_msgs.msg import HandMsg
from mros.std_msgs.msg import Float32Array, Float64Array
import numpy as np
from PIL import Image
import torch
from lerobot.cameras.utils import make_cameras_from_configs

# from lerobot.robots.teleop01.pytorch3d_utils import rotation_6d_to_matrix, matrix_to_rotation_6d
from real_world_rlWB.util.pytorch3d_utils import rotation_6d_to_matrix, matrix_to_rotation_6d
from scipy.spatial.transform import Rotation
import copy

from ..robot import Robot
from .config_teleop02 import Teleop02Config

import mros
import mros.sensor_msgs.msg.CompressedImage
import mros.std_msgs.msg.Float32Array
import mros.controller_msgs.msg.JointState
from mros.teleop_msgs.msg import KeyPoint

from mros.hand_msgs.msg import HandState

from real_world_rlWB.scripts.interpolate import interpolate_pose_waypoints
from real_world_rlWB.util.pytorch3d_utils import batch_quat_to_rot6d


from real_world_rlWB.util.basic_func import (
    _quat_wxyz_from_pose_orientation,
    _set_pose_orientation_wxyz,
    _quat_wxyz_to_rot6d,
    _rot6d_to_quat_wxyz,
    _quat_mul_wxyz,
    _quat_inv_wxyz,
)
logger = logging.getLogger(__name__)


class Teleop02(Robot):

    config_class = Teleop02Config
    name = "teleop02"
    STATE_JOINT_COLS = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "head_yaw_joint",
        "head_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_yaw_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
        # "right_hand_closed",
        "brainco2_hand_state_0",
        "brainco2_hand_state_1",
        "brainco2_hand_state_2",
        "brainco2_hand_state_3",
        "brainco2_hand_state_4",
        "brainco2_hand_state_5",
    ]
    # print(len(STATE_JOINT_COLS))
    ACTION_JOINT_COLS = [
        "base_rot6d_0",
        "base_rot6d_1",
        "base_rot6d_2",
        "base_rot6d_3",
        "base_rot6d_4",
        "base_rot6d_5",
        "head_cmd_x",
        "head_cmd_y",
        "head_cmd_z",
        "head_rot6d_0",
        "head_rot6d_1",
        "head_rot6d_2",
        "head_rot6d_3",
        "head_rot6d_4",
        "head_rot6d_5",
        "right_wrist_cmd_x",
        "right_wrist_cmd_y",
        "right_wrist_cmd_z",
        "right_wrist_rot6d_0",
        "right_wrist_rot6d_1",
        "right_wrist_rot6d_2",
        "right_wrist_rot6d_3",
        "right_wrist_rot6d_4",
        "right_wrist_rot6d_5",
        "brainco2_hand_cmd",
    ]

    # TOPICs for conecting WBT controller
    HEAD_RGB_TOPIC = "/head/color/image_raw/compressed"
    # LEFT_WRIST_RGB_TOPIC = "/left_wrist_camera/color/image_raw/compressed"
    VLA_OBSERVATION_TOPIC = "/vla/observation"
    # VLA_COMMAND_TOPIC = "/vla/command"
    JOINT_STATE_TOPIC = "/joint/state"
    FINGER_STATE_TOPIC = '/brainco2/hand/state'
    FINGER_CMD_TOPIC = '/brainco2/hand/cmd'    

    TELEOP_CMD_TOPIC = "/teleop_cmd"

    def __init__(self, config: Teleop02Config):
        super().__init__(config)
        mros.init('Teleop02Node')
        self.config = config
        self.left_pose = None
        self.input_name = None
        self.umi_flag = True
        self.left_reference_pose_mat = None
        self.head_reference_pose_mat = None
        # self.last_finger_state = np.zeros(12, dtype=np.float32)
        self.last_finger_state = np.zeros(6, dtype=np.float32)
        self.connect()

        self.manip_pose = np.array([0.24795642,-0.35561004,0.1557093,0.75742371,-0.32726956,-0.54519719,0.14820251])

        head_pose = KeyPoint()
        head_pose.name = "head"
        head_pose.pose.position.x = 0.030189851440564325  
        head_pose.pose.position.y = 0.014673380934470643
        head_pose.pose.position.z = 0.5973151388745949
        head_pose.pose.orientation.x = 0.008110045888411075   
        head_pose.pose.orientation.y = 0.1794541457153862
        head_pose.pose.orientation.z = -0.009025322555373375
        head_pose.pose.orientation.w = 0.9836915066696572
        self.head_pose = head_pose

        base_pose = KeyPoint()
        base_pose.name = "base"
        base_pose.pose.position.x = -0.1064145490527153  
        base_pose.pose.position.y = -0.03195652738213539
        base_pose.pose.position.z = 1.0081932544708252
        base_pose.pose.orientation.x = -0.014330258175869467 
        base_pose.pose.orientation.y = 0.008399610163173027 
        base_pose.pose.orientation.z = 0.011159406650108138 
        base_pose.pose.orientation.w = 0.9997997588982191
        self.base_pose = base_pose

        left_wrist_pose = KeyPoint()
        left_wrist_pose.name = "left_wrist"
        left_wrist_pose.pose.position.x = 0.10867027213060279 
        left_wrist_pose.pose.position.y = 0.2732764183532506 
        left_wrist_pose.pose.position.z = -0.10532293337296103
        left_wrist_pose.pose.orientation.x = 0.1644755273079153 
        left_wrist_pose.pose.orientation.y = 0.007869094694842272 
        left_wrist_pose.pose.orientation.z = -0.1694915166186797 
        left_wrist_pose.pose.orientation.w = 0.971678189556484
        self.left_wrist_pose = left_wrist_pose

        right_wrist_pose = KeyPoint()
        right_wrist_pose.name = "right_wrist"
        right_wrist_pose.pose.position.x = 0.236881480527492 
        right_wrist_pose.pose.position.y = -0.3158129131157036 
        right_wrist_pose.pose.position.z = 0.22425333151801985
        right_wrist_pose.pose.orientation.x = -0.31157179125153883 
        right_wrist_pose.pose.orientation.y = -0.488003570121542 
        right_wrist_pose.pose.orientation.z = 0.17049512715901663 
        right_wrist_pose.pose.orientation.w = 0.7973123265446029
        self.right_wrist_pose = right_wrist_pose

        left_hand = HandMsg()
        # left_hand.header.stamp = stamp
        left_hand.header.frame_id = ""
        left_hand.names = []
        left_hand.pos = [0.0, 1.5707, 0.0, 0.0, 0.0, 0.0]
        left_hand.vel = [0.0] * 6
        left_hand.current = [0.0] * 6
        left_hand.time = [100.0] * 6


        right_hand = HandMsg()
        # right_hand.header.stamp = stamp
        right_hand.header.frame_id = ""
        right_hand.names = []
        right_hand.pos = [0.0, 1.5707, 0.0, 0.0, 0.0, 0.0]
        right_hand.vel = [0.0] * 6
        right_hand.current = [0.0] * 6
        right_hand.time = [100.0] * 6

        self.left_hand = left_hand
        self.right_hand = right_hand





    @property
    def _motors_ft(self) -> dict[str, type]:
        # joint_name, joint_value =  self.controller.get_current_joint_state()
        # joint_name = joint_name[2:9]
        # finger_names = ["_thumb", "_thumb_aux", "_index", "_middle", "_ring", "_pinky"]
        # joint_name.extend([f"left{finger_name_item}" for finger_name_item in finger_names])
        # joint_name.extend([f"right{finger_name_item}" for finger_name_item in finger_names])
        self.input_name = self.STATE_JOINT_COLS
        return {f"{motor}.pos": float for motor in self.STATE_JOINT_COLS}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # return {"head": (480, 640, 3),}
        return {"head": (480, 640, 3)}
        # return {
        #     cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        # }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.ACTION_JOINT_COLS}


    @property
    def is_connected(self) -> bool:
        return True
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        print("init robot")
        # Subscribers
        self.head_rgb_subscriber = mros.subscribe(self.HEAD_RGB_TOPIC, CompressedImage, None)
        # self.left_wrist_rgb_subscriber = mros.subscribe(self.LEFT_WRIST_RGB_TOPIC, CompressedImage, None)
        self.vla_observation_subscriber = mros.subscribe(self.VLA_OBSERVATION_TOPIC, Float64Array, None)
        
        self.finger_state_subscriber = mros.subscribe(self.FINGER_STATE_TOPIC, HandState, None)
        self.joint_state_subscriber = mros.subscribe(self.JOINT_STATE_TOPIC, JointState, None)
        # Main function publishers
        # self.vla_command_publisher = mros.advertise(self.VLA_COMMAND_TOPIC, Float64Array, None)

        self.teleop_cmd_publisher = mros.advertise(self.TELEOP_CMD_TOPIC, TeleopMsg, None)
        self.finger_publisher = mros.advertise(self.FINGER_CMD_TOPIC, HandCmd, queue_size=10)

        print("complete init")

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated
    
    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        # (1) 读取 head 图像
        obs_dict = {}

        while True:
            head_rgb = self.head_rgb_subscriber.readMsgRT()
            if head_rgb:
                head_rgb = self.compressed_msg_to_numpy(head_rgb)
                head_rgb = head_rgb[:, :, ::-1].copy()
                obs_dict['head'] = head_rgb
                break
        
        # while True:
        #     left_wrist_rgb = self.left_wrist_rgb_subscriber.readMsgRT()
        #     if left_wrist_rgb:
        #         left_wrist_rgb = self.compressed_msg_to_numpy(left_wrist_rgb)
        #         left_wrist_rgb = left_wrist_rgb[:, :, ::-1].copy()
        #         obs_dict['left_wrist'] = left_wrist_rgb
        #         break

        # (2) 读取 joint 状态
        joint_state_msg = None
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            joint_state_msg = self.joint_state_subscriber.readMsgRT()
            if joint_state_msg is not None:
                break
            time.sleep(0.005)

        # Store into 0-30 index of obs_dict, the key can be obtained from self.STATE_JOINT_COLS
        if joint_state_msg is not None:
            # print(joint_state_msg.names, joint_state_msg.q)
            joint_state = np.asarray(joint_state_msg.q, dtype=np.float32)
            # print(joint_state.q, joint_state_msg.names)
            if joint_state.size < 31:
                raise ValueError(f"{self.JOINT_STATE_TOPIC} 数据长度不足 31，当前为 {joint_state.size}")
            for i in range(31):
                obs_dict[f'{self.STATE_JOINT_COLS[i]}.pos'] = joint_state[i]
        else:
            raise RuntimeError("尚未收到 /joint/state 消息，无法获取 joint 状态")


        # (3) 读取 finger 状态
        finger_state_msg = None
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            finger_state_msg = self.finger_state_subscriber.readMsgRT()
            if finger_state_msg is not None:
                break
            time.sleep(0.005)
        if finger_state_msg is not None:
            left_hand_state = finger_state_msg.hand_state[0].pos
            right_hand_state = finger_state_msg.hand_state[1].pos
            finger_state = np.asarray(right_hand_state)
            # print("finger_state",finger_state)

            self.last_finger_state = finger_state.copy()
        else:
            finger_state = self.last_finger_state.copy()

        for i in range(0, 6):
            # obs_dict[f'brainco1_hand_state_{i//2}.pos'] = finger_state[i]
            obs_dict[f'brainco2_hand_state_{i}.pos'] = finger_state[i]
        return obs_dict


    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        teleop_msg = TeleopMsg()
        teleop_msg.header.stamp = mros.Time()
        teleop_msg.header.frame_id = "pico_headset"
        teleop_msg.world.position.x = 0.0
        teleop_msg.world.position.y = 0.0
        teleop_msg.world.position.z = 0.0
        teleop_msg.world.orientation.x = 0.0
        teleop_msg.world.orientation.y = 0.0
        teleop_msg.world.orientation.z = 0.0
        teleop_msg.world.orientation.w = 1.0


        base_rot6d = np.array([action[f"base_rot6d_{idx}.pos"] for idx in range(6)], dtype=np.float32)
        head_pos = np.array([action[f"head_cmd_{idx}.pos"] for idx in ["x", "y", "z"]], dtype=np.float32)
        head_rot6d = np.array([action[f"head_rot6d_{idx}.pos"] for idx in range(6)], dtype=np.float32)
        rh_pos = np.array([action[f"right_wrist_cmd_{idx}.pos"] for idx in ["x", "y", "z"]], dtype=np.float32)
        rh_rot6d = np.array([action[f"right_wrist_rot6d_{idx}.pos"] for idx in range(6)], dtype=np.float32)

        finger_cmd = np.array([action[f"brainco2_hand_cmd.pos"]], dtype=np.float32)
        # action[24] = right_hand_closed 忽略

        #head
        new_head = copy.deepcopy(self.head_pose)
        # new_
        new_head.pose.position.x = head_pos[0]
        new_head.pose.position.y = head_pos[1]
        new_head.pose.position.z = head_pos[2]
        head_quat = _rot6d_to_quat_wxyz(head_rot6d)
        new_head.pose.orientation.x = head_quat[1]
        new_head.pose.orientation.y = head_quat[2]
        new_head.pose.orientation.z = head_quat[3]
        new_head.pose.orientation.w = head_quat[0]
        teleop_msg.anchors.append(new_head)

        #base
        new_base = copy.deepcopy(self.base_pose)
        base_quat = _rot6d_to_quat_wxyz(base_rot6d)
        new_base.pose.orientation.x = base_quat[1]
        new_base.pose.orientation.y = base_quat[2]
        new_base.pose.orientation.z = base_quat[3]
        new_base.pose.orientation.w = base_quat[0]
        teleop_msg.anchors.append(new_base)
        #left_wrist
        teleop_msg.anchors.append(self.left_wrist_pose)
        #right_wrist
        new_right_wrist = copy.deepcopy(self.right_wrist_pose)
        new_right_wrist.pose.position.x = rh_pos[0]
        new_right_wrist.pose.position.y = rh_pos[1]
        new_right_wrist.pose.position.z = rh_pos[2]
        right_wrist_quat = _rot6d_to_quat_wxyz(rh_rot6d)
        new_right_wrist.pose.orientation.x = right_wrist_quat[1]
        new_right_wrist.pose.orientation.y = right_wrist_quat[2]
        new_right_wrist.pose.orientation.z = right_wrist_quat[3]
        new_right_wrist.pose.orientation.w = right_wrist_quat[0]
        teleop_msg.anchors.append(new_right_wrist)

        self.teleop_cmd_publisher.publish(teleop_msg)

        # 根据 brainco2_hand_cmd 生成右手渐进闭合/张开的目标位姿
        # 大于 0.5 时每步在当前 state 基础上 +0.05 闭合，否则 -0.05 张开
        # hand_step = 0.2
        # hand_min = 0.0
        # hand_max = 1.5707
        # current_right_state = np.asarray(self.last_finger_state, dtype=np.float32).copy()
        # # print("current_right_state",current_right_state)
        # if current_right_state.size < 6:
        #     current_right_state = np.zeros(6, dtype=np.float32)
        # if float(finger_cmd[0]) > 0.2:
        #     target_right = current_right_state + hand_step
        # else:
        #     target_right = current_right_state - hand_step
        # target_right = np.clip(target_right, hand_min, hand_max).astype(np.float32)
        if float(finger_cmd[0]) > 0.4:
            target_right = np.array([1.5707, 1.5707, 1.5707, 1.5707, 1.5707, 1.5707])
        else:
            target_right = np.array([0, 1.5707, 0.0, 0.0, 0.0, 0.0])
        #do not change right hand dim 2
        target_right[1] = 1.5707

        stamp = mros.Time()

        finger_msg = HandCmd()
        finger_msg.header.stamp = stamp
        finger_msg.header.frame_id = ""
        finger_msg.hand_type = "branco2/hand"
        finger_msg.ctrl_mode = [1, 1]

        right_hand = HandMsg()
        right_hand.header.stamp = stamp
        right_hand.header.frame_id = ""
        right_hand.names = []
        right_hand.pos = target_right.tolist()
        right_hand.vel = [0.0] * 6
        right_hand.current = [0.0, 1.5707, 0.0, 0.0, 0.0, 0.0]
        right_hand.time = [100.0] * 6

        self.left_hand.header.stamp = stamp

        finger_msg.hand_cmd = [self.left_hand, right_hand]

        self.finger_publisher.publish(finger_msg)

        return 1


    def disconnect(self):
        pass
        return
        # self.left_arm.disconnect()
        # self.right_arm.disconnect()

        # for cam in self.cameras.values():
        #     cam.disconnect()
    
    def start_to_manipulation_pose(self,reverse:bool = False):
        #go to init position
        mid = np.array([0.0159696,-0.664002,0.35477, 0.7372773 , -0.6755902, 0, 0])

        init_pose= np.array([0.0159696,-0.264002,-0.02477,0.993405,-0.0891428,-0.0718367,0.00632769])
        manip_pose = self.manip_pose

        if not reverse:
            pose0 = init_pose
            pose1 = manip_pose
        else:
            pose0 = manip_pose
            pose1 = init_pose
        
        way_points = [pose0, mid, pose1]
        self.interpolate_pose_waypoints(way_points)
        #left and right hand to initial position
        stamp = mros.Time()
        finger_msg = HandCmd()
        finger_msg.header.stamp = stamp
        finger_msg.header.frame_id = ""
        finger_msg.hand_type = "branco2/hand"
        finger_msg.ctrl_mode = [1, 1]

        self.left_hand.header.stamp = stamp
        self.right_hand.header.stamp = stamp

        finger_msg.hand_cmd = [self.left_hand, self.right_hand]

        self.finger_publisher.publish(finger_msg)
    
    
    def direct_to_manipulation_pose(self):
        _, right_pose_mat, _ = self.get_current_pose_mats()
        right_xyz = right_pose_mat[:3, 3]
        right_R = right_pose_mat[:3, :3]
        right_quat = Rotation.from_matrix(right_R).as_quat(scalar_first = True)
        
        pose0 = np.array([right_xyz[0], right_xyz[1], right_xyz[2], right_quat[0], right_quat[1], right_quat[2], right_quat[3]], dtype=np.float32)
        pose1 = self.manip_pose
        way_points = [pose0, pose1]
        self.interpolate_pose_waypoints(way_points)
        #left and right hand to initial position
        stamp = mros.Time()
        finger_msg = HandCmd()
        finger_msg.header.stamp = stamp
        finger_msg.header.frame_id = ""
        finger_msg.hand_type = "branco2/hand"
        finger_msg.ctrl_mode = [1, 1]

        self.left_hand.header.stamp = stamp
        self.right_hand.header.stamp = stamp

        finger_msg.hand_cmd = [self.left_hand, self.right_hand]

        self.finger_publisher.publish(finger_msg)
    
    
    def interpolate_pose_waypoints(self, way_points: list[np.ndarray]) -> list[np.ndarray]:
        step = 50
        ts = np.linspace(0, 1, step)
        dt = 3 / step

        points = []
        for t in ts:
            points.append(interpolate_pose_waypoints(way_points, t))

        head_pose = self.head_pose
        head_quat_wxyz = np.array([
            head_pose.pose.orientation.w,
            head_pose.pose.orientation.x,
            head_pose.pose.orientation.y,
            head_pose.pose.orientation.z,
        ], dtype=np.float64)
        head_rot6d = batch_quat_to_rot6d(head_quat_wxyz).reshape(-1)

        base_pose = self.base_pose
        base_quat_wxyz = np.array([
            base_pose.pose.orientation.w,
            base_pose.pose.orientation.x,
            base_pose.pose.orientation.y,
            base_pose.pose.orientation.z,
        ], dtype=np.float64)
        base_rot6d = batch_quat_to_rot6d(base_quat_wxyz).reshape(-1)

        for point in points:
            xyz = point[0:3]
            quat = point[3:7]
            rot6d = batch_quat_to_rot6d(quat).reshape(-1)
            total_action = {}
            total_action["right_wrist_cmd_x.pos"] = float(xyz[0])
            total_action["right_wrist_cmd_y.pos"] = float(xyz[1])
            total_action["right_wrist_cmd_z.pos"] = float(xyz[2])
            for i in range(6):
                total_action[f"right_wrist_rot6d_{i}.pos"] = float(rot6d[i])

            # use origin head and base pose
            total_action["head_cmd_x.pos"] = float(head_pose.pose.position.x)
            total_action["head_cmd_y.pos"] = float(head_pose.pose.position.y)
            total_action["head_cmd_z.pos"] = float(head_pose.pose.position.z)
            for i in range(6):
                total_action[f"head_rot6d_{i}.pos"] = float(head_rot6d[i])
            for i in range(6):
                total_action[f"base_rot6d_{i}.pos"] = float(base_rot6d[i])

            total_action["brainco2_hand_cmd.pos"] = 0.0

            self.send_action(total_action)
            time.sleep(dt)






    def compressed_msg_to_numpy(self, msg: mros.sensor_msgs.msg.CompressedImage):
        """将 CompressedImage (jpeg/png) 转为 numpy BGR 图像。"""
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"解码压缩图像失败: {e}")
            return None
    
    def _xyz_xyzw_to_wxyz(self, xyz_xyzw: list[float] | torch.Tensor) -> list[float]:
        """
        将 [x, y, z, qx, qy, qz, qw] 转换为 [x, y, z, qw, qx, qy, qz]。
        """
        if torch.is_tensor(xyz_xyzw):
            pos = xyz_xyzw[..., :3].tolist()
            quat_xyzw = xyz_xyzw[..., 3:7].tolist()
        else:
            pos = xyz_xyzw[:3]
            quat_xyzw = xyz_xyzw[3:7]
        quat_wxyz = [quat_xyzw[3]] + quat_xyzw[:3]
        return pos + quat_wxyz


    def _pose_data_to_mat(self, pose_data: np.ndarray) -> np.ndarray:
        """将 [x, y, z, qw, qx, qy, qz] 转为 4x4 位姿矩阵。"""
        pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z = np.asarray(pose_data, dtype=np.float64)

        pose_mat = np.eye(4, dtype=np.float64)
        pose_mat[:3, :3] = Rotation.from_quat([quat_x, quat_y, quat_z, quat_w]).as_matrix()
        pose_mat[:3, 3] = [pos_x, pos_y, pos_z]
        return pose_mat

    def _pose_mat_to_wxyz_data(self, pose_mat: np.ndarray) -> np.ndarray:
        """将 4x4 位姿矩阵转为 [x, y, z, qw, qx, qy, qz]。"""
        pose_mat = np.asarray(pose_mat, dtype=np.float64)
        quat_xyzw = Rotation.from_matrix(pose_mat[:3, :3]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
        if quat_wxyz[0] < 0:
            quat_wxyz = -quat_wxyz
        return np.concatenate([pose_mat[:3, 3], quat_wxyz])

    def _relative_rot6d_to_mat(self, rot6d: np.ndarray) -> np.ndarray:
        rot6d_tensor = torch.from_numpy(np.asarray(rot6d, dtype=np.float32)).unsqueeze(0)
        return rotation_6d_to_matrix(rot6d_tensor).squeeze(0).numpy()

    def _rot6d_to_wxyz(self, rot6d: np.ndarray) -> np.ndarray:
        """将 6D 旋转表示转为 [qw, qx, qy, qz]。"""
        quat_xyzw = Rotation.from_matrix(self._relative_rot6d_to_mat(rot6d)).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
        if quat_wxyz[0] < 0:
            quat_wxyz = -quat_wxyz
        return quat_wxyz

    def _apply_relative_pose(
        self,
        current_pose_mat: np.ndarray,
        relative_pos: np.ndarray,
        relative_rot6d: np.ndarray,
    ) -> np.ndarray:
        """将相对位姿应用到当前位姿，返回绝对 4x4 位姿矩阵。"""
        relative_pose_mat = np.eye(4, dtype=np.float64)
        relative_pose_mat[:3, :3] = self._relative_rot6d_to_mat(relative_rot6d)
        relative_pose_mat[:3, 3] = np.asarray(relative_pos, dtype=np.float64)
        return np.asarray(current_pose_mat, dtype=np.float64) @ relative_pose_mat

    def get_current_pose_mats(
        self, vla_obs_data: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从 `/vla/observation` 读取 left/right/head 三个位姿矩阵。

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                `(left_pose_mat, right_pose_mat, head_pose_mat)`，均为 4x4 矩阵。
        """
        if vla_obs_data is None:
            msg = self.vla_observation_subscriber.readMsgRT()
            if msg is None:
                raise RuntimeError("尚未收到 /vla/observation 消息")
            vla_obs_data = msg.data

        vla_obs_data = np.asarray(vla_obs_data, dtype=np.float64)
        if vla_obs_data.size < 29:
            raise ValueError(f"/vla/observation 数据长度不足 29，当前为 {vla_obs_data.size}")

        head_pose_mat = self._pose_data_to_mat(vla_obs_data[8:15])
        left_pose_mat = self._pose_data_to_mat(vla_obs_data[15:22])
        right_pose_mat = self._pose_data_to_mat(vla_obs_data[22:29])
        return left_pose_mat, right_pose_mat, head_pose_mat

def rot6d_to_R(rot6d, eps=1e-8):
    rot6d = np.asarray(rot6d, dtype=np.float64)
    assert rot6d.shape == (6,)

    a1 = rot6d[:3]
    a2 = rot6d[3:]

    n1 = np.linalg.norm(a1)
    if n1 < eps:
        # 退化：给一个默认轴
        b1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        b1 = a1 / n1

    # 使 a2 与 b1 正交
    a2_ortho = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2_ortho)
    if n2 < eps:
        # 退化：挑一个和 b1 不平行的向量来构造 b2
        # 例如如果 b1 接近 x 轴，就用 y 轴；否则用 x 轴
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64) if abs(b1[0]) > 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        a2_ortho = tmp - np.dot(b1, tmp) * b1
        n2 = np.linalg.norm(a2_ortho)
    b2 = a2_ortho / max(n2, eps)

    b3 = np.cross(b1, b2)
    n3 = np.linalg.norm(b3)
    if n3 < eps:
        # 极端退化再兜底
        b3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        b3 = b3 / n3

    R = np.stack([b1, b2, b3], axis=1)  # columns

    # 修正 det(R) 变成 +1（保证在 SO(3)）
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0  # flip b3

    return R

def rotation_matrix_to_quaternion(R, eps=1e-12):
    R = np.asarray(R, dtype=np.float64)
    assert R.shape == (3, 3)

    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    trace = m00 + m11 + m22

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(max(1.0 + m00 - m11 - m22, eps))
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(max(1.0 + m11 - m00 - m22, eps))
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(max(1.0 + m22 - m00 - m11, eps))
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)

    # 强制单位化（非常关键）
    n = np.linalg.norm(q)
    if n < eps:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q = q / n

    # 可选：固定符号，减少跳变（w>=0）
    if q[3] < 0:
        q = -q

    return q

def rotation6d_to_quaternion(rot6d):
    R = rot6d_to_R(rot6d)
    return rotation_matrix_to_quaternion(R)



def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """归一化四元数并转成 3x3 旋转矩阵"""
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("zero-norm quaternion")
    q = q / norm
    x, y, z, w = q
    # 标准四元数转旋转矩阵
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return rot

def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """旋转矩阵取前两列拼成 6D 表示"""
    rot = quat_to_rotmat(quat)
    rot6d = rot[:, :2].reshape(-1)
    return rot6d.astype(np.float32)


ARUCO_DICT_NAME = "DICT_4X4_50"
MARKER_ID_0 = 0
MARKER_ID_1 = 1
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
from typing import List, Optional
from pathlib import Path

def estimate_gripper_width(img: np.ndarray) -> Optional[float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, _ARUCO_DICT, parameters=_ARUCO_PARAMS)
    if ids is None:
        return None

    marker_centers: list[np.ndarray] = []
    for idx, marker_id in enumerate(ids.flatten()):
        if marker_id in (MARKER_ID_0, MARKER_ID_1):
            pts = corners[idx][0]
            center = np.mean(pts, axis=0)  # (x, y)
            marker_centers.append(center)

    if len(marker_centers) >= 2:
        dist_pix = float(np.linalg.norm(marker_centers[0] - marker_centers[1]))
    elif len(marker_centers) == 1:
        dist_pix = float(abs(gray.shape[1] / 2 - marker_centers[0][0]) * 2)
    else:
        return None

    return dist_pix