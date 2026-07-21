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

import numpy as np
from PIL import Image
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.dad03.RGBStrem import ColorStreamer
from lerobot.robots.dad03.dad03controller import ROBOT_IP, RobotController
# from lerobot.robots.dad03.RGBDStrem import DepthColorStreamer

from lerobot.robots.so100_follower import SO100Follower
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig

from ..robot import Robot
from .config_dad03 import Dad03Config
import torch
logger = logging.getLogger(__name__)


class Dad03(Robot):
    """
    [Bimanual SO-100 Follower Arms](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    This bimanual robot can also be easily adapted to use SO-101 follower arms, just replace the SO100Follower class with SO101Follower and SO100FollowerConfig with SO101FollowerConfig.
    """

    config_class = Dad03Config
    name = "bi_dad03"

    def __init__(self, config: Dad03Config):
        super().__init__(config)
        self.config = config
        self.connect()
        self.connected = False
        # left_arm_config = SO100FollowerConfig(
        #     id=f"{config.id}_left" if config.id else None,
        #     calibration_dir=config.calibration_dir,
        #     port=config.left_arm_port,
        #     disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
        #     max_relative_target=config.left_arm_max_relative_target,
        #     use_degrees=config.left_arm_use_degrees,
        #     cameras={},
        # )
        # right_arm_config = SO100FollowerConfig(
        #     id=f"{config.id}_right" if config.id else None,
        #     calibration_dir=config.calibration_dir,
        #     port=config.right_arm_port,
        #     disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
        #     max_relative_target=config.right_arm_max_relative_target,
        #     use_degrees=config.right_arm_use_degrees,
        #     cameras={},
        # )
        # self.left_arm = SO100Follower(left_arm_config)
        # self.right_arm = SO100Follower(right_arm_config)
        # self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        joint_name, joint_value =  self.controller.get_current_joint_state()
        joint_name = joint_name[2:9]
        finger_names = ["_thumb", "_thumb_aux", "_index", "_middle", "_ring", "_pinky"]
        joint_name.extend([f"left{finger_name_item}" for finger_name_item in finger_names])
        # joint_name.extend([f"right{finger_name_item}" for finger_name_item in finger_names])
        return {f"{motor}.pos": float for motor in joint_name}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {"top": (480, 640, 3),}
        # return {
        #     cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        # }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft
        # return {"state": self.controller.get_current_joint_state()[1]}

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, calibrate: bool = True) -> None:
        self.controller = RobotController(robot_ip=ROBOT_IP)
        time.sleep(0.5)
        self.head_rgbd_streamer = ColorStreamer(
        ns="/camera0",
        as_rgb=True,       # True 则输出 RGB；False 为 OpenCV 常用的 BGR
        start_immediately=True
    )
        self.controller.set_control_mode(1)
        time.sleep(0.5)
        self.controller.servoJ(data={
            "head_pitch": 0.9104045724868774,
            "head_yaw": 0.0021187369711697,
            "left": [0.2025694847106933,0.1247439384460449,-0.165397822856903,-0.5672013759613037,-0.0941750481724739,0.2631072998046875,-0.0833915546536445,],
            "right": [0.2882924079895019,-0.1422765254974365,0.1227461621165275,-0.4670491218566894,-0.050569511950016,0.2280561625957489,-0.0458136349916458,],
        })
        print("*"*50)
        self.controller.move_finger(data={
            "left_mode":1,
            "left_pos": [0]*6,
            "left_vel": [4.5]*6,
            "left_current": [0, 0, 0, 0, 0, 0],
            "left_time": [100]*6,
            "right_mode":1,
            "right_pos": [0]*6,
            "right_vel":  [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            "right_current": [0, 0, 0, 0, 0, 0],
            "right_time": [100]*6,
        })
        self.connected = True


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
        obs_dict = {}

        # # Add "left_" prefix
        # left_obs = self.left_arm.get_observation()
        # obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # # Add "right_" prefix
        # right_obs = self.right_arm.get_observation()
        # obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        # for cam_key, cam in self.cameras.items():
        #     start = time.perf_counter()
        #     obs_dict[cam_key] = cam.async_read()
        #     dt_ms = (time.perf_counter() - start) * 1e3
        #     logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        start = time.perf_counter()

        color_start_time = time.time()
        head_rgb = self.head_rgbd_streamer.get_latest_color()
        # video_hw=(480, 640)
        # head_rgb = np.asarray(Image.fromarray(head_rgb).resize((video_hw[1], video_hw[0])), dtype=np.uint8)
        # head_rgb, head_depth, head_K = self.head_rgbd_streamer.get_latest_color()
        obs_dict['top'] = head_rgb
        # print("color_time = " + str(time.time()-color_start_time))
        
        color_start_time = time.time()
        joint_name, joint_value =  self.controller.get_current_joint_state()
        joint_name = joint_name[2:9]
        joint_value = joint_value[2:9]
        # print("joint_time = " + str(time.time()-color_start_time))

        for idx, joint_name_item in enumerate(joint_name):
            obs_dict[f"{joint_name_item}.pos"] = joint_value[idx]

        leftfinger_pos, rightfinger_pos =  self.controller.get_current_finger_state()
        while len(leftfinger_pos) == 0:
            leftfinger_pos, rightfinger_pos =  self.controller.get_current_finger_state()
        finger_names = ["_thumb", "_thumb_aux", "_index", "_middle", "_ring", "_pinky"]
        for idx, finger_name_item in enumerate(finger_names):
            obs_dict[f"left{finger_name_item}.pos"] = leftfinger_pos[idx]
            # obs_dict[f"right{finger_name_item}.pos"] = rightfinger_pos[idx]


        dt_ms = (time.perf_counter() - start) * 1e3
        # logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        # print("observation time: ", dt_ms)

        return obs_dict

    def _left_action_keys_in_robot_order(self) -> list[str]:
        """Keys matching ``action_features`` / ``get_observation``, in servo order (7 arm + 6 finger).

        Do not build ``left_action`` from ``dict.items()`` iteration order — that order is not
        guaranteed to match joint order unless every producer inserts keys in exactly this order.
        """
        joint_name, _ = self.controller.get_current_joint_state()
        arm_names = joint_name[2:9]
        finger_suffixes = ["_thumb", "_thumb_aux", "_index", "_middle", "_ring", "_pinky"]
        return [f"{n}.pos" for n in arm_names] + [f"left{s}.pos" for s in finger_suffixes]

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Remove "left_" prefix
        # left_action = {
        #     key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        # }
        ordered_keys = self._left_action_keys_in_robot_order()
        left_action = [action[k] for k in ordered_keys]
        # Remove "right_" prefix
        # right_action = {
        #     key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        # }
        # right_action = [value for key, value in action.items() if key.startswith("right_")]

        left_action[-4] = min(left_action[-4] + (left_action[-4] / 0.8255) * 0.4, 1.5)
        left_action[-3] = min(left_action[-3] + (left_action[-3] / 1.32) * 0.2, 1.5)
        left_action[-2] = min(left_action[-2] + (left_action[-2] / 1.28) * 0.2, 1.5)
        left_action[-1] = min(left_action[-1] + (left_action[-1] / 1.22) * 0.2, 1.5)

        for index, now_action in enumerate(left_action):
            #check if is tensor
            if isinstance(now_action, torch.Tensor):
                now_action = now_action.item()
            left_action[index] = now_action

        self.controller.servoJ(data={
            "head_pitch": 0.9104045724868774,
            "head_yaw": 0.0021187369711697,
            "left": left_action[:7],
            "right": [0.2882924079895019,-0.1422765254974365,0.1227461621165275,-0.4670491218566894,-0.050569511950016,0.2280561625957489,-0.0458136349916458,],
        })
        self.controller.move_finger(data={
            "left_mode":1,
            "left_pos": left_action[7:],
            "left_vel": [4.5]*6,
            "left_current": [0, 0, 0, 0, 0, 0],
            "left_time": [100]*6,
            "right_mode":1,
            "right_pos": [0]*6,
            "right_vel":  [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            "right_current": [0, 0, 0, 0, 0, 0],
            "right_time": [100]*6,
        })
        # send_action_left = self.left_arm.send_action(left_action)
        # send_action_right = self.right_arm.send_action(right_action)
        # Add prefixes back
        # prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        # prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        # return {**prefixed_send_action_left, **prefixed_send_action_right}
        joint_dict = {}
        joint_name, joint_value =  self.controller.get_current_joint_state()
        joint_name = joint_name[2:]
        joint_value = joint_value[2:]

        for idx, joint_name_item in enumerate(joint_name):
            joint_dict[f"{joint_name_item}.pos"] = joint_value[idx]
        return joint_dict


    def disconnect(self):
        self.connected = False
        # self.left_arm.disconnect()
        # self.right_arm.disconnect()

        # for cam in self.cameras.values():
        #     cam.disconnect()
