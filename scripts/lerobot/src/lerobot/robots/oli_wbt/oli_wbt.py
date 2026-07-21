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

import json
import logging
import os
import time
from functools import cached_property
from pathlib import Path
from typing import Any, Final

import cv2
import mros
import numpy as np
from mros.controller_msgs.msg import JointCmd, JointState
from mros.sensor_msgs.msg import CompressedImage
from mros.std_msgs.msg import Float32Array
from mros.teleop_msgs.msg import KeyPoint, TeleopMsg
from scipy.spatial.transform import Rotation

from ..robot import Robot
from .config_oli_wbt import oli_wbtConfig

logger = logging.getLogger(__name__)


BODY_JOINT_COLS: Final[tuple[str, ...]] = (
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
)

HAND_STATE_COLS: Final[tuple[str, str]] = (
    "left_hand_closed",
    "right_hand_closed",
)

STATE_JOINT_COLS: Final[tuple[str, ...]] = (*BODY_JOINT_COLS, *HAND_STATE_COLS)

ACTION_JOINT_COLS: Final[tuple[str, ...]] = (
    *(f"joint_cmd_{joint_name}" for joint_name in BODY_JOINT_COLS),
    "base_pos_cmd_x",
    "base_pos_cmd_y",
    "base_pos_cmd_z",
    *(f"base_rot6d_{i}" for i in range(6)),
    "left_hand_closed",
    "right_hand_closed",
    "done",
)

DEFAULT_KP: Final[float] = 140.0
DEFAULT_KD: Final[float] = 4.0
INITIAL_BASE_Z: Final[float] = 0.9
OBS_TIMEOUT_S: Final[float] = 0.2
ACTION_KEY: Final[str] = "action"
MISSING_CONFIG_STRINGS: Final[set[str]] = {"", "none", "null"}
BRAINCO1_OPEN_POS: Final[tuple[float, ...]] = (0.0, 100.0, 0.0, 0.0, 0.0, 0.0)
BRAINCO1_CLOSED_POS: Final[tuple[float, ...]] = (70.0, 100.0, 100.0, 100.0, 100.0, 100.0)
BRAINCO1_CMD_MODE: Final[tuple[float, ...]] = (1.0, 1.0)
BRAINCO1_CLOSED_THRESHOLD: Final[float] = 20.0


def _rot6d_to_rotation(rot6d: np.ndarray) -> Rotation:
    """Convert a Zhou et al. 6D rotation into a scipy Rotation."""
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1, a2 = rot6d[:3], rot6d[3:6]

    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-8)
    b3 = np.cross(b1, b2)

    mat = np.stack([b1, b2, b3], axis=-2)
    return Rotation.from_matrix(mat)


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _action_value(action: dict[str, Any], name: str) -> float:
    key = f"{name}.pos"
    if key not in action:
        raise KeyError(f"Missing action key: {key}")
    return float(action[key])


def _is_missing_config_value(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in MISSING_CONFIG_STRINGS


def _action_feature_name(name: str) -> str:
    return name[:-4] if name.endswith(".pos") else name


def _as_numpy_vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _brainco1_interleaved_cmd(left_closed: bool, right_closed: bool) -> list[float]:
    left_pos = BRAINCO1_CLOSED_POS if left_closed else BRAINCO1_OPEN_POS
    right_pos = BRAINCO1_CLOSED_POS if right_closed else BRAINCO1_OPEN_POS
    data: list[float] = []
    for left_value, right_value in zip(left_pos, right_pos):
        data.extend((left_value, right_value))
    data.extend(BRAINCO1_CMD_MODE)
    return data


def _dataset_meta_exists(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _candidate_dataset_roots(configured_root: str | None) -> list[Path]:
    roots: list[Path] = []
    for value in (
        configured_root,
        os.environ.get("HF_LEROBOT_HOME"),
        os.environ.get("LEROBOT_HOME"),
        str(Path.home() / "lerobot_dataset"),
    ):
        if _is_missing_config_value(value):
            continue
        root = Path(str(value)).expanduser()
        if root not in roots:
            roots.append(root)
    return roots


class oli_wbt(Robot):
    """LeRobot wrapper for the Teleop02 WBT action space.

    Public feature names match the pick_shark LeRobot dataset. The final
    action dimension, ``done``, is accepted for model/dataset compatibility
    and is not sent to hardware.
    """

    config_class = oli_wbtConfig
    name = "oli_wbt"

    BODY_JOINT_COLS = list(BODY_JOINT_COLS)
    STATE_JOINT_COLS = list(STATE_JOINT_COLS)
    ACTION_JOINT_COLS = list(ACTION_JOINT_COLS)

    HEAD_RGB_TOPIC = "/head/color/image_raw/compressed"
    LEFT_WRIST_RGB_TOPIC = "/left_wrist_camera/color/image_raw/compressed"
    JOINT_STATE_TOPIC = "/joint/state"
    FINGER_STATE_TOPIC = "/brainco1/hand/state"
    FINGER_CMD_TOPIC = "/brainco1/hand/cmd"
    TELEOP_WBT_TOPIC = "/teleop_cmd_WBT"

    def __init__(self, config: oli_wbtConfig):
        super().__init__(config)
        self.config = config
        self.FINGER_CMD_TOPIC = str(config.finger_cmd_topic)

        self.last_finger_state = np.zeros(12, dtype=np.float32)
        self.last_finger_cmd = np.zeros(14, dtype=np.float32)

        self._accum_base_pos = np.array([0.0, 0.0, INITIAL_BASE_Z], dtype=np.float64)
        self._accum_base_yaw = 0.0
        self._accum_base_rot = Rotation.identity()
        self._start_pose_target: tuple[np.ndarray, float, float] | None = None
        self._is_connected = False

        self.connect()

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self.STATE_JOINT_COLS}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            "head": (480, 640, 3),
            "left_wrist": (480, 640, 3),
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self.ACTION_JOINT_COLS}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return

        mros.init("oli_wbtNode")

        self.head_rgb_subscriber = mros.subscribe(self.HEAD_RGB_TOPIC, CompressedImage, None)
        self.left_wrist_rgb_subscriber = mros.subscribe(
            self.LEFT_WRIST_RGB_TOPIC, CompressedImage, None
        )
        self.joint_state_subscriber = mros.subscribe(self.JOINT_STATE_TOPIC, JointState, None)
        self.finger_state_subscriber = mros.subscribe(
            self.FINGER_STATE_TOPIC, Float32Array, None
        )

        self.teleop_wbt_publisher = mros.advertise(self.TELEOP_WBT_TOPIC, TeleopMsg, None)
        self.finger_publisher = mros.advertise(
            self.FINGER_CMD_TOPIC, Float32Array, queue_size=10
        )

        self._is_connected = True
        logger.info("Connected oli_wbt robot wrapper")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def setup_motors(self) -> None:
        return

    def _read_msg(self, subscriber: Any, topic: str, timeout_s: float = OBS_TIMEOUT_S) -> Any:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            msg = subscriber.readMsgRT()
            if msg is not None:
                return msg
            time.sleep(0.005)
        raise RuntimeError(f"No message received from {topic} within {timeout_s:.3f}s")

    def _try_read_msg(self, subscriber: Any) -> Any | None:
        try:
            return subscriber.readMsgRT()
        except Exception as exc:
            logger.debug("Failed to read optional message: %s", exc)
            return None

    def _compressed_msg_to_numpy(self, msg: CompressedImage) -> np.ndarray:
        try:
            data = msg.data
            if isinstance(data, (bytes, bytearray)):
                buf = data
            elif isinstance(data, np.ndarray):
                buf = data.tobytes()
            else:
                buf = bytes(data)
            np_arr = np.frombuffer(buf, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise RuntimeError("Failed to decode compressed image") from exc

        if image is None:
            raise RuntimeError("Failed to decode compressed image")
        return image

    def _joint_state_vector(self, msg: JointState) -> np.ndarray:
        q = np.asarray(msg.q, dtype=np.float32)
        if q.size < len(BODY_JOINT_COLS):
            raise ValueError(
                f"{self.JOINT_STATE_TOPIC} q length is {q.size}, "
                f"expected at least {len(BODY_JOINT_COLS)}"
            )

        names = list(getattr(msg, "names", []) or [])
        if len(names) == q.size:
            by_name = dict(zip(names, q))
            if all(name in by_name for name in BODY_JOINT_COLS):
                return np.asarray([by_name[name] for name in BODY_JOINT_COLS], dtype=np.float32)

        return q[: len(BODY_JOINT_COLS)].copy()

    def _update_finger_state(self) -> None:
        msg = self._try_read_msg(self.finger_state_subscriber)
        if msg is None:
            return

        finger_state = np.asarray(getattr(msg, "data", []), dtype=np.float32)
        if finger_state.size >= 12:
            self.last_finger_state = finger_state[:12].copy()

    def _hand_closed_state(self) -> tuple[float, float]:
        left_cmd_avg = float(np.mean(self.last_finger_cmd[0:12:2]))
        right_cmd_avg = float(np.mean(self.last_finger_cmd[1:12:2]))
        left_hand_closed = 1.0 if left_cmd_avg > BRAINCO1_CLOSED_THRESHOLD else 0.0
        right_hand_closed = 1.0 if right_cmd_avg > BRAINCO1_CLOSED_THRESHOLD else 0.0
        return left_hand_closed, right_hand_closed

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("oli_wbt is not connected")

        head_msg = self._read_msg(self.head_rgb_subscriber, self.HEAD_RGB_TOPIC)
        left_wrist_msg = self._read_msg(
            self.left_wrist_rgb_subscriber, self.LEFT_WRIST_RGB_TOPIC
        )
        joint_state_msg = self._read_msg(self.joint_state_subscriber, self.JOINT_STATE_TOPIC)
        self._update_finger_state()

        head_rgb = self._compressed_msg_to_numpy(head_msg)[:, :, ::-1].copy()
        left_wrist_rgb = self._compressed_msg_to_numpy(left_wrist_msg)[:, :, ::-1].copy()
        joint_state = self._joint_state_vector(joint_state_msg)
        left_hand_closed, right_hand_closed = self._hand_closed_state()

        obs_dict: dict[str, Any] = {
            "head": head_rgb,
            "left_wrist": left_wrist_rgb,
        }
        for name, value in zip(BODY_JOINT_COLS, joint_state):
            obs_dict[f"{name}.pos"] = float(value)
        obs_dict["left_hand_closed.pos"] = left_hand_closed
        obs_dict["right_hand_closed.pos"] = right_hand_closed
        return obs_dict

    def _action_vector_from_dict(self, action: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [_action_value(action, name) for name in ACTION_JOINT_COLS],
            dtype=np.float64,
        )

    def _policy_train_config_path(self) -> Path | None:
        checkpoint_path = getattr(self.config, "start_policy_checkpoint_path", None)
        if _is_missing_config_value(checkpoint_path):
            return None

        base = Path(str(checkpoint_path)).expanduser()
        candidates = (
            base / "train_config.json",
            base / "pretrained_model" / "train_config.json",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        logger.warning(
            "No train_config.json found under start_policy_checkpoint_path=%r",
            str(checkpoint_path),
        )
        return None

    def _dataset_locator_from_policy_config(self) -> tuple[str | None, str | None]:
        train_config_path = self._policy_train_config_path()
        if train_config_path is None:
            return None, None

        try:
            with train_config_path.open("r", encoding="utf-8") as f:
                train_config = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", train_config_path, exc)
            return None, None

        dataset_cfg = train_config.get("dataset") or {}
        repo_id = dataset_cfg.get("repo_id") or train_config.get("dataset_repo_id")
        root = dataset_cfg.get("root") or train_config.get("dataset_root")
        repo_id = None if _is_missing_config_value(repo_id) else str(repo_id).strip()
        root = None if _is_missing_config_value(root) else str(root).strip()
        return repo_id, root

    def _resolve_start_dataset(self) -> tuple[str, str | None]:
        policy_repo_id, policy_root = self._dataset_locator_from_policy_config()

        repo_id = policy_repo_id
        if _is_missing_config_value(repo_id):
            repo_id = getattr(self.config, "start_dataset_repo_id", None)
        if _is_missing_config_value(repo_id):
            raise RuntimeError(
                "oli_wbt start pose needs a dataset. Set "
                "start_policy_checkpoint_path or start_dataset_repo_id."
            )

        root = getattr(self.config, "start_dataset_root", None)
        if _is_missing_config_value(root):
            root = policy_root

        repo_id_str = str(repo_id).strip()
        root_str = None if _is_missing_config_value(root) else str(root).strip()

        repo_path = Path(repo_id_str).expanduser()
        if repo_path.is_absolute():
            if _dataset_meta_exists(repo_path):
                return repo_path.name, str(repo_path)

            for candidate_root in _candidate_dataset_roots(root_str):
                candidate = candidate_root / repo_path.name
                if _dataset_meta_exists(candidate):
                    logger.warning(
                        "Configured start dataset %s does not exist; using local "
                        "dataset with the same name at %s",
                        repo_path,
                        candidate,
                    )
                    return candidate.name, str(candidate)

            return repo_path.name, str(repo_path)

        if root_str is not None:
            root_path = Path(root_str).expanduser()
            if _dataset_meta_exists(root_path):
                return repo_id_str, str(root_path)

            nested_root = root_path / repo_id_str
            if _dataset_meta_exists(nested_root):
                return repo_id_str, str(nested_root)

        return repo_id_str, root_str

    def _load_start_pose_target(self) -> tuple[np.ndarray, float, float]:
        if self._start_pose_target is not None:
            return self._start_pose_target

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        repo_id, root = self._resolve_start_dataset()
        dataset = LeRobotDataset(repo_id, root=root, episodes=[0], download_videos=False)

        action_feature = dataset.features.get(ACTION_KEY)
        if not isinstance(action_feature, dict) or "names" not in action_feature:
            raise RuntimeError(f"Dataset {repo_id!r} does not expose action feature names")

        action_names = [_action_feature_name(str(name)) for name in action_feature["names"]]
        if len(dataset.hf_dataset) <= 0:
            raise RuntimeError(f"Dataset {repo_id!r} episode 0 has no frames")

        first_frame = dataset.hf_dataset[0]
        if ACTION_KEY not in first_frame:
            raise RuntimeError(f"Dataset {repo_id!r} first frame has no {ACTION_KEY!r}")

        action_vec = _as_numpy_vector(first_frame[ACTION_KEY])
        if len(action_names) != action_vec.size:
            raise RuntimeError(
                f"Dataset {repo_id!r} action name count {len(action_names)} "
                f"does not match action dim {action_vec.size}"
            )

        action_by_name = dict(zip(action_names, action_vec))
        target_names = [f"joint_cmd_{name}" for name in BODY_JOINT_COLS]
        missing = [name for name in target_names if name not in action_by_name]
        if missing:
            raise RuntimeError(
                f"Dataset {repo_id!r} is missing WBT start joint targets: {missing}"
            )

        target_q = np.asarray([action_by_name[name] for name in target_names], dtype=np.float64)
        left_closed = float(action_by_name.get("left_hand_closed", 0.0))
        right_closed = float(action_by_name.get("right_hand_closed", 0.0))
        self._start_pose_target = (target_q, left_closed, right_closed)

        logger.info(
            "Loaded oli_wbt start pose from dataset repo_id=%r root=%r",
            repo_id,
            root,
        )
        return self._start_pose_target

    def _integrate_base_action(
        self, base_pos_action: np.ndarray, base_rot6d_action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        cos_yaw = np.cos(self._accum_base_yaw)
        sin_yaw = np.sin(self._accum_base_yaw)
        delta_x_body, delta_y_body = base_pos_action[:2]
        self._accum_base_pos[0] += cos_yaw * delta_x_body - sin_yaw * delta_y_body
        self._accum_base_pos[1] += sin_yaw * delta_x_body + cos_yaw * delta_y_body
        self._accum_base_pos[2] = base_pos_action[2]

        base_action_euler = _rot6d_to_rotation(base_rot6d_action).as_euler("ZYX")
        self._accum_base_yaw = _wrap_to_pi(self._accum_base_yaw + base_action_euler[0])
        self._accum_base_rot = Rotation.from_euler(
            "ZYX",
            [
                self._accum_base_yaw,
                base_action_euler[1],
                base_action_euler[2],
            ],
        )
        return self._accum_base_pos.copy(), self._accum_base_rot.as_quat()

    def _make_keypoint(
        self,
        name: str,
        pos: np.ndarray,
        quat_xyzw: np.ndarray,
    ) -> KeyPoint:
        kp = KeyPoint()
        kp.name = name
        kp.pose.position.x = float(pos[0])
        kp.pose.position.y = float(pos[1])
        kp.pose.position.z = float(pos[2])
        kp.pose.orientation.x = float(quat_xyzw[0])
        kp.pose.orientation.y = float(quat_xyzw[1])
        kp.pose.orientation.z = float(quat_xyzw[2])
        kp.pose.orientation.w = float(quat_xyzw[3])
        return kp

    def _publish_joint_command(
        self,
        joint_cmd_q: np.ndarray,
        *,
        base_pos: np.ndarray | None = None,
        base_quat_xyzw: np.ndarray | None = None,
        left_closed: float = 0.0,
        right_closed: float = 0.0,
    ) -> None:
        q = np.asarray(joint_cmd_q, dtype=np.float64).reshape(-1)
        if q.size != len(BODY_JOINT_COLS):
            raise ValueError(
                f"joint_cmd_q length is {q.size}, expected {len(BODY_JOINT_COLS)}"
            )

        if base_pos is None:
            base_pos = self._accum_base_pos.copy()
        if base_quat_xyzw is None:
            base_quat_xyzw = self._accum_base_rot.as_quat()

        teleop_msg = TeleopMsg()
        teleop_msg.header.stamp = mros.Time()
        teleop_msg.header.frame_id = "world"
        teleop_msg.world.orientation.w = 1.0

        joint_cmd = JointCmd()
        joint_cmd.names = list(BODY_JOINT_COLS)
        joint_cmd.q = q.astype(np.float32).tolist()
        joint_cmd.v = [0.0] * len(BODY_JOINT_COLS)
        joint_cmd.tau = [0.0] * len(BODY_JOINT_COLS)
        joint_cmd.kp = [DEFAULT_KP] * len(BODY_JOINT_COLS)
        joint_cmd.kd = [DEFAULT_KD] * len(BODY_JOINT_COLS)
        joint_cmd.mode = [0] * len(BODY_JOINT_COLS)
        joint_cmd.na = len(BODY_JOINT_COLS)
        teleop_msg.joint_cmd = joint_cmd
        teleop_msg.anchors = [
            self._make_keypoint("base_link", base_pos, base_quat_xyzw),
        ]

        self.teleop_wbt_publisher.publish(teleop_msg)
        self._publish_hand_cmd(left_closed, right_closed)

    def _publish_hand_cmd(self, left_closed: float, right_closed: float) -> None:
        finger_cmd = _brainco1_interleaved_cmd(
            left_closed=left_closed >= 0.5,
            right_closed=right_closed >= 0.5,
        )

        self.last_finger_cmd = np.asarray(finger_cmd, dtype=np.float32)
        msg = Float32Array()
        msg.data = finger_cmd
        self.finger_publisher.publish(msg)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("oli_wbt is not connected")

        action_vec = self._action_vector_from_dict(action)
        control_action = action_vec[:42]

        joint_cmd_q = control_action[:31]
        base_pos_action = control_action[31:34]
        base_rot6d_action = control_action[34:40]
        left_closed = float(control_action[40])
        right_closed = float(control_action[41])

        base_pos, base_quat_xyzw = self._integrate_base_action(
            base_pos_action, base_rot6d_action
        )

        self._publish_joint_command(
            joint_cmd_q,
            base_pos=base_pos,
            base_quat_xyzw=base_quat_xyzw,
            left_closed=left_closed,
            right_closed=right_closed,
        )

        return {
            f"{name}.pos": float(value)
            for name, value in zip(ACTION_JOINT_COLS, action_vec)
        }

    def to_start_pose(
        self,
        num_steps: int | None = None,
        duration_s: float | None = None,
    ) -> None:
        if not self.is_connected:
            raise RuntimeError("oli_wbt is not connected")

        target_q, target_left_closed, target_right_closed = self._load_start_pose_target()
        joint_state_msg = self._read_msg(self.joint_state_subscriber, self.JOINT_STATE_TOPIC)
        current_q = self._joint_state_vector(joint_state_msg).astype(np.float64)

        steps = (
            int(num_steps)
            if num_steps is not None
            else int(getattr(self.config, "start_pose_num_steps", 50))
        )
        duration = (
            float(duration_s)
            if duration_s is not None
            else float(getattr(self.config, "start_pose_duration_s", 3.0))
        )

        if steps <= 1:
            self._publish_joint_command(
                target_q,
                left_closed=target_left_closed,
                right_closed=target_right_closed,
            )
            return

        dt = max(0.0, duration) / float(steps)
        for q in np.linspace(current_q, target_q, steps):
            self._publish_joint_command(q, left_closed=0.0, right_closed=0.0)
            if dt > 0:
                time.sleep(dt)

        self._publish_joint_command(
            target_q,
            left_closed=target_left_closed,
            right_closed=target_right_closed,
        )

    def go_to_start_pose(
        self,
        *,
        settle_s: float = 0.0,
        num_steps: int | None = None,
        duration_s: float | None = None,
    ) -> None:
        self.to_start_pose(num_steps=num_steps, duration_s=duration_s)
        if settle_s > 0:
            time.sleep(settle_s)

    def reanchor_z(self, z: float) -> None:
        self._accum_base_pos[2] = float(z)

    def disconnect(self) -> None:
        self._is_connected = False
