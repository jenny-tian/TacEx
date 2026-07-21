import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# ------------------------------------------------------------- #
# --------------------------- Aruco Utils ----------------------#
# ------------------------------------------------------------- #
ARUCO_DICT_NAME = "DICT_4X4_50"
MARKER_ID_0 = 0
MARKER_ID_1 = 1
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
from typing import Optional


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


def xyz_xyzw_to_pose_mat(xyz_xyzw: list[float]) -> np.ndarray:
    """[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w] --> 4x4 pose matrix
    
    Args:
        xyz_xyzw: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
                  Note: quaternion format is [x, y, z, w] (scipy format)
    
    Returns:
        4x4 pose matrix
    """
    pose_mat = np.eye(4)
    pose_mat[:3, 3] = np.array([xyz_xyzw[0], xyz_xyzw[1], xyz_xyzw[2]])
    # scipy Rotation.from_quat expects [x, y, z, w]
    pose_mat[:3, :3] = Rotation.from_quat([xyz_xyzw[3], xyz_xyzw[4], xyz_xyzw[5], xyz_xyzw[6]]).as_matrix()
    return pose_mat


def xyz_wxyz_to_pose_mat(xyz_wxyz: list[float]) -> np.ndarray:
    """[pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z] --> 4x4 pose matrix
    
    Args:
        xyz_wxyz: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
                  Note: quaternion format is [w, x, y, z] (ROS format)
    
    Returns:
        4x4 pose matrix
    """
    pose_mat = np.eye(4)
    pose_mat[:3, 3] = np.array([xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2]])
    # Convert from [w, x, y, z] to [x, y, z, w] for scipy Rotation.from_quat
    pose_mat[:3, :3] = Rotation.from_quat([xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6], xyz_wxyz[3]]).as_matrix()
    return pose_mat


def pose_mat_to_xyz_xyzw(pose_mat: np.ndarray) -> list[float]:
    """4x4 pose matrix --> [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
    
    Args:
        pose_mat: 4x4 pose matrix
    
    Returns:
        [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        Note: quaternion format is [x, y, z, w] (scipy format)
    """
    pos_x = pose_mat[0, 3]
    pos_y = pose_mat[1, 3]
    pos_z = pose_mat[2, 3]

    rot_mat = pose_mat[:3, :3]
    quat = Rotation.from_matrix(rot_mat).as_quat()  # Returns [x, y, z, w]
    quat_x, quat_y, quat_z, quat_w = quat
    return [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]


def pose_mat_to_xyz_wxyz(pose_mat: np.ndarray) -> list[float]:
    """4x4 pose matrix --> [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
    
    Args:
        pose_mat: 4x4 pose matrix
    
    Returns:
        [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        Note: quaternion format is [w, x, y, z] (ROS format)
    """
    pos_x = pose_mat[0, 3]
    pos_y = pose_mat[1, 3]
    pos_z = pose_mat[2, 3]

    rot_mat = pose_mat[:3, :3]
    quat = Rotation.from_matrix(rot_mat).as_quat()  # Returns [x, y, z, w]
    quat_x, quat_y, quat_z, quat_w = quat
    # Convert from [x, y, z, w] to [w, x, y, z]
    return [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]


import numpy as np


# ============================================================
# 基础工具
# ============================================================

def make_T(R, t):
    """
    构造齐次变换矩阵

    R: (3,3) 旋转矩阵
    t: (3,)  平移向量

    返回:
        T ∈ SE(3)
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def change_frame(T_in_A, R_A_to_B):
    """
    纯 frame change（不引入任何物理运动）

    数学含义：
        已知同一个刚体的位姿 ^A T_obj
        给定坐标系旋转 R_A→B
        计算该刚体在 B 坐标系下的表示 ^B T_obj

    推导：
        ^B R_obj = R_A→B · ^A R_obj
        ^B p_obj = R_A→B · ^A p_obj

    注意：
        - 这里只允许旋转（frame 的原点重合）
        - 这是“表示变化”，不是“刚体运动”
    """
    T = np.eye(4)
    T[:3, :3] = R_A_to_B @ T_in_A[:3, :3]
    T[:3, 3]  = R_A_to_B @ T_in_A[:3, 3]
    return T


# ============================================================
# 主函数（等式推导版）
# ============================================================

def compute_target_wrist_pose(
    T_base_wrist_ctrl: np.ndarray,
    t_wrist_to_gripper_vla: np.ndarray,
    T_relative_gripper_vla: np.ndarray
) -> np.ndarray:

    # 用于 debug 的当前 wrist 旋转（base frame 下）
    R_base_ctrl = T_base_wrist_ctrl[:3, :3]

    # --------------------------------------------------
    # Step 0: define base ↔ vla relation
    # --------------------------------------------------
    # 约定：vla frame 初始姿态与 base frame 对齐
    # 因此 ^base R_vla = I, 只有纯坐标系变换（不依赖 wrist 当前姿态）
    R_base_vla = np.eye(3)
    R_base_to_vla = R_base_vla.T
    print("R_base_to_vla:", R_base_to_vla)

    # --------------------------------------------------
    # Step 1: base → vla (frame change)
    # --------------------------------------------------
    # ^vla T_wrist = R_base_to_vla · ^base T_wrist
    T_vla_wrist = change_frame(
        T_base_wrist_ctrl,
        R_base_to_vla
    )
    if True:
        rel_euler = Rotation.from_matrix(T_relative_gripper_vla[:3, :3]).as_euler("xyz", degrees=True)
        print("[DEBUG] T_relative_gripper_vla euler xyz (deg):", rel_euler)
        vla_wrist_euler = Rotation.from_matrix(T_vla_wrist[:3, :3]).as_euler("xyz", degrees=True)
        print("[DEBUG] T_vla_wrist euler xyz (deg):", vla_wrist_euler)


    # --------------------------------------------------
    # Step 2: wrist → gripper (body transform, offset)
    # --------------------------------------------------
    # ^vla T_gripper = ^vla T_wrist · ^wrist T_gripper
    T_wrist_to_gripper = make_T(
        np.eye(3),
        t_wrist_to_gripper_vla
    )
    T_vla_gripper = T_vla_wrist @ T_wrist_to_gripper


    # --------------------------------------------------
    # Step 3: apply relative motion (spatial motion in vla frame)
    # --------------------------------------------------
    # ^vla T_gripper_target
    #   = ^vla T_gripper' = (^vla T_relative) · (^vla T_gripper)
    # 如果相对位姿是在 vla/world frame 下定义，必须左乘
    T_vla_gripper_target = (
        T_relative_gripper_vla @ T_vla_gripper
    )


    # --------------------------------------------------
    # Step 4: remove offset (gripper → wrist)
    # --------------------------------------------------
    # ^vla T_wrist_target
    T_vla_wrist_target = (
        T_vla_gripper_target @
        np.linalg.inv(T_wrist_to_gripper)
    )


    # --------------------------------------------------
    # Step 5: vla → ctrl (frame change back)
    # --------------------------------------------------
    # ^base R_vla = R_base_vla
    # ^vla R_base = R_base_vla^T
    R_vla_to_base = R_base_vla
    print("R_vla_to_base:", R_vla_to_base)
    
    # ^base T_wrist^ctrl_target
    T_base_wrist_ctrl_target = change_frame(
        T_vla_wrist_target,
        R_vla_to_base
    )

    # Debug: 相对旋转在 base 下的实际轴角
    delta_rot = T_base_wrist_ctrl_target[:3, :3] @ R_base_ctrl.T
    rotvec = Rotation.from_matrix(delta_rot).as_rotvec()
    angle_deg = np.linalg.norm(rotvec) * 180.0 / np.pi
    axis = rotvec / (np.linalg.norm(rotvec) + 1e-12)
    print("[DEBUG] delta_rot axis (base):", axis, "angle_deg:", angle_deg)

    return T_base_wrist_ctrl_target
