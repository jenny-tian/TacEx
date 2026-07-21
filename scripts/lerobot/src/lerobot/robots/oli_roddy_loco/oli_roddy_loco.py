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
from mros.geometry_msgs.msg import Twist
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
from .config_oli_roddy_loco import oli_roddy_locoConfig

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

# action 分段顺序与 create_lerobot_dataset_roddy.py ACTION_NAMES 一致：
# 每段 pos(3)+rot6d(6)；base 段后接 continuous cmd_vel（x 线速度、z 角速度）；
# 新数据集最后接 left_hand_closed, right_hand_closed。
_ACTION_BODY_PARTS: Final[tuple[str, ...]] = ("base", "torso", "head", "left_hand", "right_hand")
_HAND_OPEN_POS: Final[tuple[float, ...]] = (0.0, 1.5707, 0.0, 0.0, 0.0, 0.0)
_HAND_CLOSED_POS: Final[tuple[float, ...]] = (1.5707, 1.5707, 1.5707, 1.5707, 1.5707, 1.5707)
_SUPPORTED_HAND_TYPES: Final[tuple[str, ...]] = ("brainco1", "brainco2")
_BRAINCO1_OPEN_POS: Final[tuple[float, ...]] = (0.0, 100.0, 0.0, 0.0, 0.0, 0.0)
_BRAINCO1_CLOSED_POS: Final[tuple[float, ...]] = (70.0, 100.0, 100.0, 100.0, 100.0, 100.0)
_BRAINCO1_CMD_MODE: Final[tuple[float, ...]] = (1.0, 1.0)


def _normalize_hand_type(hand_type: str) -> str:
    normalized = str(hand_type).strip().lower()
    if normalized not in _SUPPORTED_HAND_TYPES:
        raise ValueError(
            f"Unsupported hand_type={hand_type!r}; expected one of {_SUPPORTED_HAND_TYPES}."
        )
    return normalized


def _action_joint_col_names(using_left_hand: bool = True) -> list[str]:
    cols: list[str] = []
    for part in _ACTION_BODY_PARTS:
        cols.extend((f"{part}_cmd_x", f"{part}_cmd_y", f"{part}_cmd_z"))
        cols.extend(f"{part}_rot6d_{i}" for i in range(6))
        if part == "base":
            cols.extend(("base_vel_cmd_x", "base_ang_vel_cmd_z"))
    if using_left_hand:
        cols.append("left_hand_closed")
    cols.append("right_hand_closed")
    return cols


def _hand_state_col_names(
    using_left_hand: bool = True,
    hand_type: str = "brainco2",
) -> list[str]:
    prefix = _normalize_hand_type(hand_type)
    if using_left_hand:
        return [
            *(f"left_{prefix}_hand_state_{i}" for i in range(6)),
            *(f"right_{prefix}_hand_state_{i}" for i in range(6)),
        ]
    return [f"{prefix}_hand_state_{i}" for i in range(6)]


def _hand_msg_from_closed(closed: bool, stamp: mros.Time) -> HandMsg:
    target = np.array(_HAND_CLOSED_POS if closed else _HAND_OPEN_POS, dtype=np.float64)

    hand = HandMsg()
    hand.header.stamp = stamp
    hand.header.frame_id = ""
    hand.names = []
    hand.pos = target.tolist()
    hand.vel = [0.0] * 6
    hand.current = [0.0, 1.5707, 0.0, 0.0, 0.0, 0.0]
    hand.time = [100.0] * 6
    return hand


def _brainco1_interleaved_cmd(left_closed: bool, right_closed: bool) -> list[float]:
    left_pos = _BRAINCO1_CLOSED_POS if left_closed else _BRAINCO1_OPEN_POS
    right_pos = _BRAINCO1_CLOSED_POS if right_closed else _BRAINCO1_OPEN_POS
    data: list[float] = []
    for left_value, right_value in zip(left_pos, right_pos):
        data.extend((left_value, right_value))
    data.extend(_BRAINCO1_CMD_MODE)
    return data


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


def _teleop_msg_with_body_part_order(msg: TeleopMsg) -> TeleopMsg:
    ordered_msg = copy.deepcopy(msg)
    anchors_by_name = _teleop_anchors_by_name(ordered_msg)
    ordered_anchors: list[KeyPoint] = []
    missing: list[str] = []
    for name in _ACTION_BODY_PARTS:
        anchor = anchors_by_name.get(name)
        if anchor is None:
            missing.append(name)
            continue
        ordered_anchors.append(copy.deepcopy(anchor))
    if missing:
        print(f"[WARN] teleop_cmd missing anchors {missing}; keep original anchor order")
        return ordered_msg
    ordered_msg.anchors = ordered_anchors
    return ordered_msg


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


def _apply_xyz_quat_xyzw_to_keypoint(
    kp: KeyPoint, xyz: tuple[float, float, float], quat_xyzw: tuple[float, float, float, float]
) -> None:
    # Bag messages may share one pose object across anchors; detach before writing.
    if kp.pose is not None:
        kp.pose = copy.deepcopy(kp.pose)
    kp.pose.position.x = float(xyz[0])
    kp.pose.position.y = float(xyz[1])
    kp.pose.position.z = float(xyz[2])
    qx, qy, qz, qw = quat_xyzw
    kp.pose.orientation.x = float(qx)
    kp.pose.orientation.y = float(qy)
    kp.pose.orientation.z = float(qz)
    kp.pose.orientation.w = float(qw)


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

DEFAULT_PREPARE_BAG_PATH = "/home/limx/dataset/pick_toy_0626/start.bag"
DEFAULT_PICK_TOY_START_BAG_PATH = "/home/limx/dataset/pick_toy_0628/001.bag"
DEFAULT_PICK_FRUIT_START_BAG_PATH = "/home/limx/dataset/pick_fruit/002.bag"


class pick_toy_init:
    def __init__(
        self,
        robot: Any,
        prepare_bag_path: str = DEFAULT_PREPARE_BAG_PATH,
        start_bag_path: str = DEFAULT_PICK_TOY_START_BAG_PATH,
    ):
        self.robot = robot
        self.prepare_bag_path = prepare_bag_path
        self.start_bag_path = start_bag_path
        self.start_cmd_msg: TeleopMsg | None = None

    @staticmethod
    def _time_to_sec(t: Any) -> float:
        if hasattr(t, "toSec"):
            return float(t.toSec())
        return float(t)

    def _load_first_teleop_cmd_from_bag(self, bag_path: str) -> TeleopMsg | None:
        try:
            with mros.Bag(str(bag_path), "r") as bag:
                for topic, msg, _t in bag.read_messages():
                    if topic != self.robot.TELEOP_CMD_TOPIC:
                        continue
                    print(f"[pick_toy_init] Loaded start pose from {bag_path}")
                    return _teleop_msg_with_body_part_order(msg)
        except Exception as exc:
            print(f"[WARN] Failed to load start pose from bag {bag_path}: {exc}")
        return None

    def _get_start_cmd_msg(self) -> TeleopMsg | None:
        if self.start_cmd_msg is None:
            self.start_cmd_msg = self._load_first_teleop_cmd_from_bag(self.start_bag_path)
        return copy.deepcopy(self.start_cmd_msg) if self.start_cmd_msg is not None else None

    def prepare(self, keep_timing: bool = True, default_dt_s: float = 1.0 / 30.0) -> None:
        """Replay the task preparation bag once."""
        last_t: float | None = None
        last_cmd_msg: TeleopMsg | None = None
        teleop_count = 0
        vel_count = 0

        try:
            with mros.Bag(str(self.prepare_bag_path), "r") as bag:
                for topic, msg, t in bag.read_messages():
                    if not mros.ok():
                        break
                    if topic not in (self.robot.TELEOP_CMD_TOPIC, self.robot.CMD_VEL_TOPIC):
                        continue

                    current_t = self._time_to_sec(t)
                    if last_t is not None:
                        dt = max(0.0, current_t - last_t) if keep_timing else default_dt_s
                        if dt > 0:
                            time.sleep(dt)
                    last_t = current_t

                    if topic == self.robot.TELEOP_CMD_TOPIC:
                        out_msg = _teleop_msg_with_body_part_order(msg)
                        out_msg.header.stamp = mros.Time()
                        self.robot.teleop_cmd_publisher.publish(out_msg)
                        self.robot._publish_hand_cmd(
                            left_closed=False,
                            right_closed=False,
                        )
                        last_cmd_msg = copy.deepcopy(out_msg)
                        teleop_count += 1
                    elif topic == self.robot.CMD_VEL_TOPIC:
                        self.robot.cmd_vel_publisher.publish(copy.deepcopy(msg))
                        vel_count += 1
        except Exception as exc:
            print(f"[WARN] Failed to replay prepare bag {self.prepare_bag_path}: {exc}")
            return

        self.robot._publish_cmd_vel(0.0, 0.0)
        if last_cmd_msg is not None:
            self.robot.last_cmd_msg = last_cmd_msg
        print(
            f"[pick_toy_init] Prepared from {self.prepare_bag_path}: "
            f"{teleop_count} teleop_cmd, {vel_count} cmd_vel"
        )

    def _load_prepare_teleop_sequence(self) -> list[tuple[float, TeleopMsg]]:
        sequence: list[tuple[float, TeleopMsg]] = []
        try:
            with mros.Bag(str(self.prepare_bag_path), "r") as bag:
                for topic, msg, t in bag.read_messages():
                    if not mros.ok():
                        break
                    if topic != self.robot.TELEOP_CMD_TOPIC:
                        continue
                    sequence.append(
                        (
                            self._time_to_sec(t),
                            _teleop_msg_with_body_part_order(msg),
                        )
                    )
        except Exception as exc:
            print(f"[WARN] Failed to load prepare teleop_cmd from {self.prepare_bag_path}: {exc}")
            return []
        return sequence

    def _publish_teleop_only(self, msg: TeleopMsg) -> None:
        out_msg = copy.deepcopy(msg)
        out_msg.header.stamp = mros.Time()
        self.robot.teleop_cmd_publisher.publish(out_msg)
        self.robot._publish_hand_cmd(left_closed=False, right_closed=False)
        self.robot.last_cmd_msg = copy.deepcopy(out_msg)

    def reset_to_initial_pose(
        self,
        *,
        num_steps: int = 50,
        approach_duration_s: float = 2.0,
        keep_timing: bool = True,
        default_dt_s: float = 1.0 / 30.0,
    ) -> bool:
        """Return from prepare pose to the first pose of the prepare bag.

        This path intentionally publishes only ``/teleop_cmd`` (plus hand open
        commands) and skips all ``/sdk_cmd_vel`` from the bag.
        """
        teleop_sequence = self._load_prepare_teleop_sequence()
        if not teleop_sequence:
            print(f"[WARN] No teleop_cmd in prepare bag {self.prepare_bag_path}; skip reset_to_initial_pose")
            return False

        last_msg = copy.deepcopy(teleop_sequence[-1][1])
        current_msg = copy.deepcopy(self.robot.last_cmd_msg)

        if current_msg is None or num_steps <= 0:
            self._publish_teleop_only(last_msg)
        else:
            seq = _teleop_interpolate_last_to_first(last_msg, current_msg, num_steps)
            if not seq:
                self._publish_teleop_only(last_msg)
            else:
                dt = max(0.0, approach_duration_s) / float(num_steps)
                for msg in seq:
                    self._publish_teleop_only(msg)
                    if dt > 0:
                        time.sleep(dt)

        last_t: float | None = None
        replay_count = 0
        for current_t, msg in reversed(teleop_sequence):
            if last_t is not None:
                dt = max(0.0, last_t - current_t) if keep_timing else default_dt_s
                if dt > 0:
                    time.sleep(dt)
            self._publish_teleop_only(msg)
            last_t = current_t
            replay_count += 1

        print(
            f"[pick_toy_init] Reset to initial from {self.prepare_bag_path}: "
            f"{replay_count} teleop_cmd, 0 cmd_vel"
        )
        return True

    def to_start_pose(self, num_steps: int = 50, duration_s: float = 3.0) -> None:
        target_msg = self._get_start_cmd_msg()
        if target_msg is None:
            print("[WARN] pick_toy start_cmd_msg is None; skip to_start_pose")
            return

        self.robot.start_cmd_msg = copy.deepcopy(target_msg)
        current_msg = copy.deepcopy(self.robot.last_cmd_msg)
        # if current_msg is None:
        #     current_msg = self.robot._build_current_cmd_msg()
        if current_msg is None:
            print("[WARN] current_cmd_msg is None; publish pick_toy start pose directly")
            target_msg.header.stamp = mros.Time()
            self.robot.teleop_cmd_publisher.publish(target_msg)
            self.robot._publish_hand_cmd(left_closed=False, right_closed=False)
            self.robot._publish_cmd_vel(0.0, 0.0)
            self.robot.last_cmd_msg = copy.deepcopy(target_msg)
            return

        if num_steps <= 0:
            seq: list[TeleopMsg] = []
        else:
            seq = _teleop_interpolate_last_to_first(target_msg, current_msg, num_steps)

        if not seq:
            target_msg.header.stamp = mros.Time()
            self.robot.teleop_cmd_publisher.publish(target_msg)
            self.robot._publish_hand_cmd(left_closed=False, right_closed=False)
            self.robot._publish_cmd_vel(0.0, 0.0)
        else:
            dt = max(0.0, duration_s) / float(num_steps)
            self.robot._publish_teleop_sequence(seq, dt)

        self.robot.last_cmd_msg = copy.deepcopy(target_msg)


class oli_roddy_loco(Robot):

    config_class = oli_roddy_locoConfig
    name = "oli_roddy_loco"
    BODY_JOINT_COLS = [
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
    ]
    LEGACY_HAND_STATE_COLS = _hand_state_col_names(using_left_hand=False)
    LEFT_RIGHT_HAND_STATE_COLS = _hand_state_col_names(using_left_hand=True)
    STATE_JOINT_COLS = [*BODY_JOINT_COLS, *LEFT_RIGHT_HAND_STATE_COLS]
    # print(len(STATE_JOINT_COLS))
    ACTION_JOINT_COLS = _action_joint_col_names()

    # TOPICs for conecting WBT controller
    HEAD_RGB_TOPIC = "/head/color/image_raw/compressed"
    CHEST_RGB_TOPIC = "/chest/color/image_raw/compressed"
    RIGHT_WRIST_RGB_TOPIC = "/right_wrist_camera/color/image_raw/compressed"
    # LEFT_WRIST_RGB_TOPIC = "/left_wrist_camera/color/image_raw/compressed"
    VLA_OBSERVATION_TOPIC = "/vla/observation"
    # VLA_COMMAND_TOPIC = "/vla/command"
    JOINT_STATE_TOPIC = "/joint/state"
    FINGER_STATE_TOPIC = '/brainco2/hand/state'
    FINGER_CMD_TOPIC = '/brainco2/hand/cmd'    

    TELEOP_CMD_TOPIC = "/teleop_cmd"
    CMD_VEL_TOPIC = "/sdk_cmd_vel"

    def __init__(self, config: oli_roddy_locoConfig):
        super().__init__(config)
        mros.init('oli_roddy_locoNode')
        self.config = config
        self.using_left_hand = self.config.using_left_hand
        self.hand_type = _normalize_hand_type(self.config.hand_type)
        self.FINGER_STATE_TOPIC = f"/{self.hand_type}/hand/state"
        self.FINGER_CMD_TOPIC = f"/{self.hand_type}/hand/cmd"
        self.state_joint_cols = [
            *self.BODY_JOINT_COLS,
            *_hand_state_col_names(
                using_left_hand=self.using_left_hand,
                hand_type=self.hand_type,
            ),
        ]
        self.action_joint_cols = _action_joint_col_names(using_left_hand=self.using_left_hand)
        self.STATE_JOINT_COLS = self.state_joint_cols
        self.ACTION_JOINT_COLS = self.action_joint_cols
        self.left_pose = None
        self.input_name = None
        self.umi_flag = True
        self.left_reference_pose_mat = None
        self.head_reference_pose_mat = None
        self.last_left_finger_state = np.zeros(6, dtype=np.float32)
        self.last_right_finger_state = np.zeros(6, dtype=np.float32)
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
        left_hand.pos = list(_HAND_OPEN_POS)
        left_hand.vel = [0.0] * 6
        left_hand.current = [0.0] * 6
        left_hand.time = [100.0] * 6


        right_hand = HandMsg()
        # right_hand.header.stamp = stamp
        right_hand.header.frame_id = ""
        right_hand.names = []
        right_hand.pos = list(_HAND_OPEN_POS)
        right_hand.vel = [0.0] * 6
        right_hand.current = [0.0] * 6
        right_hand.time = [100.0] * 6

        self.left_hand = left_hand
        self.right_hand = right_hand

        self.last_cmd_msg = None
        self.start_cmd_msg = None
        # Current deployment hand poses (xyz + quat xyzw) used as the start of to_start_pose.
        self._current_left_hand_xyz = (0, 0.301273, -0.233775)
        self._current_left_hand_quat_xyzw = (0.0225799, 0.657747, 0.203116, 0.724984)
        self._current_right_hand_xyz = (0, -0.302342, -0.250169)
        self._current_right_hand_quat_xyzw = (-0.0623979, 0.672269, -0.162318, 0.719593)

        task_init_name = str(getattr(self.config, "task_init", "pick_toy") or "pick_toy")
        task_init_name = task_init_name.strip().lower().replace("-", "_")
        prepare_bag_path = getattr(self.config, "prepare_bag_path", None) or DEFAULT_PREPARE_BAG_PATH
        start_bag_path = getattr(self.config, "start_bag_path", None)
        if task_init_name == "pick_fruit":
            start_bag_path = start_bag_path or DEFAULT_PICK_FRUIT_START_BAG_PATH
        else:
            start_bag_path = start_bag_path or DEFAULT_PICK_TOY_START_BAG_PATH

        print(
            "[oli_roddy_loco] task_init="
            f"{task_init_name!r}, prepare_bag={prepare_bag_path!r}, "
            f"start_bag={start_bag_path!r}"
        )
        self.task_init = pick_toy_init(
            self,
            prepare_bag_path=prepare_bag_path,
            start_bag_path=start_bag_path,
        )
        self.start_cmd_msg = self.task_init._get_start_cmd_msg()




    @property
    def _motors_ft(self) -> dict[str, type]:
        # joint_name, joint_value =  self.controller.get_current_joint_state()
        # joint_name = joint_name[2:9]
        # finger_names = ["_thumb", "_thumb_aux", "_index", "_middle", "_ring", "_pinky"]
        # joint_name.extend([f"left{finger_name_item}" for finger_name_item in finger_names])
        # joint_name.extend([f"right{finger_name_item}" for finger_name_item in finger_names])
        self.input_name = self.state_joint_cols
        return {f"{motor}.pos": float for motor in self.state_joint_cols}

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
        return {f"{motor}.pos": float for motor in self.action_joint_cols}


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
        self.right_wrist_rgb_subscriber = mros.subscribe(self.RIGHT_WRIST_RGB_TOPIC, CompressedImage, None)
        self.vla_observation_subscriber = mros.subscribe(self.VLA_OBSERVATION_TOPIC, Float64Array, None)
        
        finger_state_msg_type = HandState if self.hand_type == "brainco2" else Float32Array
        finger_cmd_msg_type = HandCmd if self.hand_type == "brainco2" else Float32Array
        self.finger_state_subscriber = mros.subscribe(self.FINGER_STATE_TOPIC, finger_state_msg_type, None)
        self.joint_state_subscriber = mros.subscribe(self.JOINT_STATE_TOPIC, JointState, None)
        # Main function publishers
        # self.vla_command_publisher = mros.advertise(self.VLA_COMMAND_TOPIC, Float64Array, None)

        self.teleop_cmd_publisher = mros.advertise(self.TELEOP_CMD_TOPIC, TeleopMsg, None)
        self.finger_publisher = mros.advertise(self.FINGER_CMD_TOPIC, finger_cmd_msg_type, queue_size=10)
        self.cmd_vel_publisher = mros.advertise(self.CMD_VEL_TOPIC, Twist, queue_size=10)

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
        
        if self.config.using_right_wrist_camera:
            while True:
                right_wrist_rgb = self.right_wrist_rgb_subscriber.readMsgRT()
                if right_wrist_rgb:
                    right_wrist_rgb = self.compressed_msg_to_numpy(right_wrist_rgb)
                    right_wrist_rgb = right_wrist_rgb[:, :, ::-1].copy()
                    obs_dict['right_wrist'] = right_wrist_rgb
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

        # Store into 0-30 index of obs_dict, the key can be obtained from self.BODY_JOINT_COLS
        if joint_state_msg is not None:
            # print(joint_state_msg.names, joint_state_msg.q)
            joint_state = np.asarray(joint_state_msg.q, dtype=np.float32)
            # print(joint_state.q, joint_state_msg.names)
            if joint_state.size < 31:
                raise ValueError(f"{self.JOINT_STATE_TOPIC} 数据长度不足 31，当前为 {joint_state.size}")
            for i, joint_name in enumerate(self.BODY_JOINT_COLS):
                obs_dict[f'{joint_name}.pos'] = joint_state[i]
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
        if finger_state_msg is None:
            raise RuntimeError("尚未收到 hand state状态")

        left_hand_state, right_hand_state = self._extract_hand_state(finger_state_msg)
        self.last_left_finger_state = left_hand_state.copy()
        self.last_right_finger_state = right_hand_state.copy()

        if self.using_left_hand:
            for i in range(6):
                obs_dict[f'left_{self.hand_type}_hand_state_{i}.pos'] = left_hand_state[i]
                obs_dict[f'right_{self.hand_type}_hand_state_{i}.pos'] = right_hand_state[i]
        else:
            for i in range(6):
                obs_dict[f'{self.hand_type}_hand_state_{i}.pos'] = right_hand_state[i]
        return obs_dict

    def _extract_hand_state(self, finger_state_msg: Any) -> tuple[np.ndarray, np.ndarray]:
        if self.hand_type == "brainco2":
            if len(finger_state_msg.hand_state) < 2:
                raise ValueError(
                    f"{self.FINGER_STATE_TOPIC} hand_state 长度不足 2，"
                    f"当前为 {len(finger_state_msg.hand_state)}"
                )
            left_hand_state = np.asarray(finger_state_msg.hand_state[0].pos, dtype=np.float32)
            right_hand_state = np.asarray(finger_state_msg.hand_state[1].pos, dtype=np.float32)
        else:
            if not hasattr(finger_state_msg, "data") or finger_state_msg.data is None:
                raise ValueError(f"{self.FINGER_STATE_TOPIC} 缺少 data 字段")
            hand_state = np.asarray(finger_state_msg.data, dtype=np.float32)
            if hand_state.size < 12:
                raise ValueError(
                    f"{self.FINGER_STATE_TOPIC} data 长度不足 12，当前为 {hand_state.size}"
                )
            left_hand_state = hand_state[:12:2]
            right_hand_state = hand_state[1:12:2]

        if left_hand_state.size < 6 or right_hand_state.size < 6:
            raise ValueError(
                f"{self.FINGER_STATE_TOPIC} 左右手状态长度不足 6，"
                f"当前为 left={left_hand_state.size}, right={right_hand_state.size}"
            )
        return left_hand_state[:6].copy(), right_hand_state[:6].copy()


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

        left_finger_closed = False
        if self.using_left_hand:
            left_finger_closed = float(action["left_hand_closed.pos"]) > 0.4
        right_finger_closed = float(action["right_hand_closed.pos"]) > 0.4

        def _apply_pose_to_keypoint(
            kp: KeyPoint,
            pos: np.ndarray,
            rot6d: np.ndarray,
            quat_wxyz: np.ndarray | None = None,
        ) -> None:
            kp.pose.position.x = float(pos[0])
            kp.pose.position.y = float(pos[1])
            kp.pose.position.z = float(pos[2])
            q = (
                _rot6d_to_quat_wxyz(rot6d)
                if quat_wxyz is None
                else np.asarray(quat_wxyz, dtype=np.float64)
            )
            kp.pose.orientation.x = float(q[1])
            kp.pose.orientation.y = float(q[2])
            kp.pose.orientation.z = float(q[3])
            kp.pose.orientation.w = float(q[0])

        # 顺序与 _ACTION_BODY_PARTS 一致，便于与数据集 / 控制器对齐
        new_base = copy.deepcopy(self.base_pose)
        _apply_pose_to_keypoint(new_base, base_pos, base_rot6d)
        teleop_msg.anchors.append(new_base)

        base_q = _rot6d_to_quat_wxyz(base_rot6d)
        world_q_torso = _rot6d_to_quat_wxyz(torso_rot6d)
        torso_q = _quat_mul_wxyz(_quat_inv_wxyz(base_q), world_q_torso)

        new_torso = copy.deepcopy(self.torso_pose)
        _apply_pose_to_keypoint(new_torso, torso_pos, torso_rot6d, quat_wxyz=torso_q)
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

        self._publish_hand_cmd(
            left_closed=left_finger_closed,
            right_closed=right_finger_closed,
        )

        lin_x = float(action["base_vel_cmd_x.pos"])
        ang_z = float(action["base_ang_vel_cmd_z.pos"])
        self._publish_cmd_vel(lin_x, ang_z)

        return 1

    def _publish_cmd_vel(self, lin_x: float, ang_z: float) -> None:
        msg = Twist()
        msg.linear.x = lin_x
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = ang_z
        self.cmd_vel_publisher.publish(msg)

    def _publish_hand_cmd(self, left_closed: bool = False, right_closed: bool = False) -> None:
        if self.hand_type == "brainco1":
            self._publish_brainco1_hand_cmd(
                left_closed=left_closed,
                right_closed=right_closed,
            )
            return
        self._publish_brainco2_hand_cmd(
            left_closed=left_closed,
            right_closed=right_closed,
        )

    def _publish_brainco1_hand_cmd(self, left_closed: bool = False, right_closed: bool = False) -> None:
        finger_msg = Float32Array()
        finger_msg.data = _brainco1_interleaved_cmd(left_closed, right_closed)
        self.finger_publisher.publish(finger_msg)

    def _publish_brainco2_hand_cmd(self, left_closed: bool = False, right_closed: bool = False) -> None:
        stamp = mros.Time()

        finger_msg = HandCmd()
        finger_msg.header.stamp = stamp
        finger_msg.header.frame_id = ""
        finger_msg.hand_type = "branco2/hand"
        finger_msg.ctrl_mode = [1, 1]

        finger_msg.hand_cmd = [
            _hand_msg_from_closed(left_closed, stamp),
            _hand_msg_from_closed(right_closed, stamp),
        ]

        self.finger_publisher.publish(finger_msg)


    def _build_current_cmd_msg(self) -> TeleopMsg | None:
        """Build a teleop_cmd snapshot with the current deployment hand poses."""
        if self.start_cmd_msg is None:
            return None

        msg = copy.deepcopy(self.start_cmd_msg)
        hand_pose_by_name = {
            "left_hand": (self._current_left_hand_xyz, self._current_left_hand_quat_xyzw),
            "left_wrist": (self._current_left_hand_xyz, self._current_left_hand_quat_xyzw),
            "right_hand": (self._current_right_hand_xyz, self._current_right_hand_quat_xyzw),
            "right_wrist": (self._current_right_hand_xyz, self._current_right_hand_quat_xyzw),
        }
        for kp in msg.anchors:
            name = getattr(kp, "name", None)
            spec = hand_pose_by_name.get(name)
            if spec is None:
                continue
            xyz, quat_xyzw = spec
            _apply_xyz_quat_xyzw_to_keypoint(kp, xyz, quat_xyzw)
            # print(kp.name, kp.pose.position.x, kp.pose.position.y, kp.pose.position.z)
        return msg

    def _publish_teleop_sequence(self, seq: list[TeleopMsg], dt: float) -> None:
        for msg in seq:
            msg.header.stamp = mros.Time()
            self.teleop_cmd_publisher.publish(msg)
            self._publish_hand_cmd(left_closed=False, right_closed=False)
            self._publish_cmd_vel(0.0, 0.0)
            time.sleep(dt)

    @staticmethod
    def _load_start_cmd_from_bag(bag_path: str) -> TeleopMsg | None:
        """Load the first ``/teleop_cmd`` message from a rosbag as the episode start pose."""
        path = str(bag_path)
        if not path:
            return None
        target_index = 10
        current_index = 0
        try:
            with mros.Bag(path, "r") as bag:
                for topic, msg, _t in bag.read_messages():
                    if topic != oli_roddy_loco.TELEOP_CMD_TOPIC:
                        continue
                    current_index += 1
                    if current_index < target_index:
                        continue
                    if not mros.ok():
                        break
                    print(f"[oli_roddy_loco] Loaded start pose from {path}")
                    return _teleop_msg_with_body_part_order(msg)
        except Exception as exc:
            print(f"[WARN] Failed to load start pose from bag {path}: {exc}")
        return None

    def go_to_start_pose(
        self,
        *,
        settle_s: float = 0.0,
        num_steps: int = 50,
        duration_s: float = 3.0,
    ) -> None:
        """Interpolate from the current deployment hand pose to the bag start teleop_cmd."""
        if self.start_cmd_msg is None:
            print("[WARN] start_cmd_msg is None; skip go_to_start_pose")
            return

        current_msg = self._build_current_cmd_msg()
        # self.teleop_cmd_publisher.publish(current_msg)
        # return


        # return
        if current_msg is None:
            print("[WARN] current_cmd_msg is None; skip go_to_start_pose")
            return

        seq = _teleop_interpolate_last_to_first(self.start_cmd_msg, current_msg, num_steps)

        if not seq:
            msg = copy.deepcopy(self.start_cmd_msg)
            msg.header.stamp = mros.Time()
            self.teleop_cmd_publisher.publish(msg)
            self._publish_hand_cmd(left_closed=False, right_closed=False)
            self._publish_cmd_vel(0.0, 0.0)
        else:
            self._publish_teleop_sequence(seq, duration_s / float(num_steps))

        self.last_cmd_msg = copy.deepcopy(self.start_cmd_msg)
        if settle_s > 0:
            time.sleep(settle_s)

    def prepare(self) -> None:
        self.task_init.prepare()

    def reset_to_initial_pose(self) -> bool:
        print("reset to initial pose")
        return self.task_init.reset_to_initial_pose()

    def disconnect(self):
        pass
        return
        # self.left_arm.disconnect()
        # self.right_arm.disconnect()

        # for cam in self.cameras.values():
        #     cam.disconnect()
    
    def to_start_pose(self, num_steps: int = 50, duration_s: float = 3.0) -> None:
        print("go to start pose")
        self.task_init.to_start_pose(num_steps=num_steps, duration_s=duration_s)

    






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
