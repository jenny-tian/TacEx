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
from typing import Any, Final

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
from .config_oli_roddy import oli_roddyConfig

import mros
import mros.sensor_msgs.msg.CompressedImage
import mros.std_msgs.msg.Float32Array
import mros.controller_msgs.msg.JointState
from mros.teleop_msgs.msg import KeyPoint

from mros.hand_msgs.msg import HandState

from real_world_rlWB.scripts.interpolate import interpolate_pose_pair, interpolate_pose_waypoints
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

# action 分段顺序：每段 pos(3)+rot6d(6)，最后 brainco2 标量（teleop anchor 名为 left_hand / right_hand）
_ACTION_BODY_PARTS: Final[tuple[str, ...]] = ("base", "torso", "head", "left_hand", "right_hand")


def _action_joint_col_names() -> list[str]:
    cols: list[str] = []
    for part in _ACTION_BODY_PARTS:
        cols.extend((f"{part}_cmd_x", f"{part}_cmd_y", f"{part}_cmd_z"))
        cols.extend(f"{part}_rot6d_{i}" for i in range(6))
    cols.append("brainco2_hand_cmd")
    return cols


def _pose9_from_action(action: dict[str, Any], part: str) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array([action[f"{part}_cmd_{k}.pos"] for k in ("x", "y", "z")], dtype=np.float32)
    rot6d = np.array([action[f"{part}_rot6d_{i}.pos"] for i in range(6)], dtype=np.float32)
    return pos, rot6d


def _write_pose9_to_action(
    target: dict[str, Any], part: str, pos_xyz: np.ndarray, rot6d: np.ndarray
) -> None:
    for k, v in zip(("x", "y", "z"), pos_xyz):
        target[f"{part}_cmd_{k}.pos"] = float(v)
    for i in range(6):
        target[f"{part}_rot6d_{i}.pos"] = float(rot6d[i])


def _keypoint_pose_to_quat_wxyz(kp: KeyPoint) -> np.ndarray:
    return np.array(
        [
            kp.pose.orientation.w,
            kp.pose.orientation.x,
            kp.pose.orientation.y,
            kp.pose.orientation.z,
        ],
        dtype=np.float64,
    )


def _teleop_anchors_by_name(msg: TeleopMsg) -> dict[str, KeyPoint]:
    out: dict[str, KeyPoint] = {}
    if not hasattr(msg, "anchors") or not msg.anchors:
        return out
    for a in msg.anchors:
        if not (hasattr(a, "name") and hasattr(a, "pose")):
            continue
        if a.name not in out:
            out[a.name] = a
    return out


def _teleop_keypoint_to_pose7(kp: KeyPoint) -> np.ndarray:
    p = kp.pose.position
    o = kp.pose.orientation
    return np.array([p.x, p.y, p.z, o.w, o.x, o.y, o.z], dtype=np.float64)


def _teleop_apply_pose7_to_keypoint(kp: KeyPoint, pose7: np.ndarray) -> None:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    p = kp.pose.position
    o = kp.pose.orientation
    p.x, p.y, p.z = float(pose7[0]), float(pose7[1]), float(pose7[2])
    o.w, o.x, o.y, o.z = (
        float(pose7[3]),
        float(pose7[4]),
        float(pose7[5]),
        float(pose7[6]),
    )


def _teleop_interpolate_last_to_first(
    first_msg: TeleopMsg, last_msg: TeleopMsg, num_steps: int
) -> list[TeleopMsg]:
    a_first = _teleop_anchors_by_name(first_msg)
    a_last = _teleop_anchors_by_name(last_msg)
    names = sorted(set(a_first.keys()) & set(a_last.keys()))
    if not names:
        return []

    last7 = {n: _teleop_keypoint_to_pose7(a_last[n]) for n in names}
    first7 = {n: _teleop_keypoint_to_pose7(a_first[n]) for n in names}

    out: list[TeleopMsg] = []
    for alpha in np.linspace(0.0, 1.0, num_steps):
        msg = copy.deepcopy(first_msg)
        by_name = _teleop_anchors_by_name(msg)
        for n in names:
            p = interpolate_pose_pair(last7[n], first7[n], alpha)
            _teleop_apply_pose7_to_keypoint(by_name[n], p)
        out.append(msg)
    return out


class oli_roddy(Robot):

    config_class = oli_roddyConfig
    name = "oli_roddy"
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
    ACTION_JOINT_COLS = _action_joint_col_names()

    # TOPICs for conecting WBT controller
    HEAD_RGB_TOPIC = "/head/color/image_raw/compressed"
    CHEST_RGB_TOPIC = "/chest/color/image_raw/compressed"
    # LEFT_WRIST_RGB_TOPIC = "/left_wrist_camera/color/image_raw/compressed"
    VLA_OBSERVATION_TOPIC = "/vla/observation"
    # VLA_COMMAND_TOPIC = "/vla/command"
    JOINT_STATE_TOPIC = "/joint/state"
    FINGER_STATE_TOPIC = '/brainco2/hand/state'
    FINGER_CMD_TOPIC = '/brainco2/hand/cmd'    

    TELEOP_CMD_TOPIC = "/teleop_cmd"

    def __init__(self, config: oli_roddyConfig):
        super().__init__(config)
        mros.init('oli_roddyNode')
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

        torso_pose = KeyPoint()
        torso_pose.name = "torso"
        torso_pose.pose.position.x = 0.0
        torso_pose.pose.position.y = 0.0
        torso_pose.pose.position.z = 1.0
        torso_pose.pose.orientation.w = 1.0
        torso_pose.pose.orientation.x = 0.0
        torso_pose.pose.orientation.y = 0.0
        torso_pose.pose.orientation.z = 0.0
        self.torso_pose = torso_pose

        left_hand_pose_kp = copy.deepcopy(left_wrist_pose)
        left_hand_pose_kp.name = "left_hand"
        self.left_hand_pose_kp = left_hand_pose_kp

        right_hand_pose_kp = copy.deepcopy(right_wrist_pose)
        right_hand_pose_kp.name = "right_hand"
        self.right_hand_pose_kp = right_hand_pose_kp

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

        self.last_cmd_msg = None
        self.start_cmd_msg = None
        #load first cmd from bag
        bag_path = "/home/limx/dataset/0502/050.bag"
        with mros.Bag(str(bag_path), "r") as bag:
            for topic, msg, t in bag.read_messages():
                if topic != self.TELEOP_CMD_TOPIC:
                    continue
                if not mros.ok():
                    break
                self.start_cmd_msg = copy.deepcopy(msg)
                break
        
        keep_name = ["base", "torso", "head", "left_hand", "right_hand"]
        for anc in self.start_cmd_msg.anchors:
            if anc.name not in keep_name:
                self.start_cmd_msg.anchors.remove(anc)




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
        self.chest_rgb_subscriber = mros.subscribe(self.CHEST_RGB_TOPIC, CompressedImage, None)
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
        i = 1
        while True:
            head_rgb = self.head_rgb_subscriber.readMsgRT()
            i+=1
            if head_rgb:
                head_rgb = self.compressed_msg_to_numpy(head_rgb)
                head_rgb = head_rgb[:, :, ::-1].copy()
                obs_dict['head'] = head_rgb
                break
        if self.config.using_chest_camera:
            while True:
                chest_rgb = self.chest_rgb_subscriber.readMsgRT()
                # print("wait chest rgb")
                if chest_rgb:
                    chest_rgb = self.compressed_msg_to_numpy(chest_rgb)
                    chest_rgb = chest_rgb[:, :, ::-1].copy()
                    obs_dict['chest'] = chest_rgb
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


        base_pos, base_rot6d = _pose9_from_action(action, "base")
        torso_pos, torso_rot6d = _pose9_from_action(action, "torso")
        head_pos, head_rot6d = _pose9_from_action(action, "head")
        lh_pos, lh_rot6d = _pose9_from_action(action, "left_hand")
        rh_pos, rh_rot6d = _pose9_from_action(action, "right_hand")

        finger_cmd = np.array([action["brainco2_hand_cmd.pos"]], dtype=np.float32)

        def _apply_pose_to_keypoint(kp: KeyPoint, pos: np.ndarray, rot6d: np.ndarray) -> None:
            kp.pose.position.x = float(pos[0])
            kp.pose.position.y = float(pos[1])
            kp.pose.position.z = float(pos[2])
            q = _rot6d_to_quat_wxyz(rot6d)
            kp.pose.orientation.x = q[1]
            kp.pose.orientation.y = q[2]
            kp.pose.orientation.z = q[3]
            kp.pose.orientation.w = q[0]

        # 顺序与 _ACTION_BODY_PARTS 一致，便于与数据集 / 控制器对齐
        new_base = copy.deepcopy(self.base_pose)
        _apply_pose_to_keypoint(new_base, base_pos, base_rot6d)
        teleop_msg.anchors.append(new_base)

        new_torso = copy.deepcopy(self.torso_pose)
        _apply_pose_to_keypoint(new_torso, torso_pos, torso_rot6d)
        teleop_msg.anchors.append(new_torso)

        new_head = copy.deepcopy(self.head_pose)
        _apply_pose_to_keypoint(new_head, head_pos, head_rot6d)
        teleop_msg.anchors.append(new_head)

        new_left_hand = copy.deepcopy(self.left_hand_pose_kp)
        _apply_pose_to_keypoint(new_left_hand, lh_pos, lh_rot6d)
        teleop_msg.anchors.append(new_left_hand)

        new_right_hand = copy.deepcopy(self.right_hand_pose_kp)
        _apply_pose_to_keypoint(new_right_hand, rh_pos, rh_rot6d)
        teleop_msg.anchors.append(new_right_hand)

        self.teleop_cmd_publisher.publish(teleop_msg)
        self.last_cmd_msg = copy.deepcopy(teleop_msg)

        self._publish_brainco2_hand_cmd(closed=float(finger_cmd[0]) > 0.4)

        return 1

    def _publish_brainco2_hand_cmd(self, closed: bool) -> None:
        if closed:
            target_right = np.array([1.5707, 1.5707, 1.5707, 1.5707, 1.5707, 1.5707])
        else:
            target_right = np.array([0, 1.5707, 0.0, 0.0, 0.0, 0.0])
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


    def disconnect(self):
        pass
        return
        # self.left_arm.disconnect()
        # self.right_arm.disconnect()

        # for cam in self.cameras.values():
        #     cam.disconnect()
    
    def to_start_pose(self, num_steps: int = 50, duration_s: float = 3.0) -> None:
        """回到默认起始 Teleop 位姿：无历史 cmd 则直接下发；否则从 last_cmd_msg 插值过渡。"""
        start_msg = self.start_cmd_msg
        if self.last_cmd_msg is None:
            start_msg.header.stamp = mros.Time()
            self.teleop_cmd_publisher.publish(start_msg)
            self._publish_brainco2_hand_cmd(closed=False)
            self.last_cmd_msg = copy.deepcopy(start_msg)
            return

        seq = _teleop_interpolate_last_to_first(start_msg, self.last_cmd_msg, num_steps)
        # if not seq:
        #     start_msg.header.stamp = mros.Time()
        #     self.teleop_cmd_publisher.publish(start_msg)
        #     self._publish_brainco2_hand_cmd(closed=False)
        #     self.last_cmd_msg = copy.deepcopy(start_msg)
        #     return

        dt = duration_s / float(num_steps)
        for msg in seq:
            msg.header.stamp = mros.Time()
            self.teleop_cmd_publisher.publish(msg)
            self._publish_brainco2_hand_cmd(closed=False)
            time.sleep(dt)
        self.last_cmd_msg = copy.deepcopy(start_msg)

    






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