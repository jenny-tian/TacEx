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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("oli_wbt")
@dataclass
class oli_wbtConfig(RobotConfig):
    left_arm_port: str = None
    right_arm_port: str = None

    using_chest_camera: bool = False

    # Optional
    left_arm_disable_torque_on_disconnect: bool = True
    left_arm_max_relative_target: float | dict[str, float] | None = None
    left_arm_use_degrees: bool = False
    right_arm_disable_torque_on_disconnect: bool = True
    right_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_use_degrees: bool = False

    # cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    using_right_wrist_camera: bool = False

    start_policy_checkpoint_path: str | None = None
    start_dataset_repo_id: str | None = None
    start_dataset_root: str | None = None
    start_pose_num_steps: int = 50
    start_pose_duration_s: float = 3.0

    finger_cmd_topic: str = "/brainco1/hand/cmd"
