import importlib.util
import math
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "tacex_tasks"
    / "lab_pick"
    / "grasp_geometry.py"
)
SPEC = importlib.util.spec_from_file_location("grasp_geometry", MODULE_PATH)
grasp_geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grasp_geometry)


def test_centered_tool_target_accounts_for_rotated_tool_offset():
    object_center = torch.tensor([0.5, 0.1, 0.02])
    yaw_quat = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    tool_offset = torch.tensor([0.02, 0.0, 0.10])

    target = grasp_geometry.centered_tool_target(
        object_center, yaw_quat, tool_offset
    )

    torch.testing.assert_close(
        target, torch.tensor([0.5, 0.08, -0.08]), atol=1e-6, rtol=0
    )


def test_vector_in_local_frame_undoes_frame_rotation():
    vector_b = torch.tensor([0.0, 0.02, 0.10])
    yaw_quat = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])

    vector_local = grasp_geometry.vector_in_local_frame(vector_b, yaw_quat)

    torch.testing.assert_close(
        vector_local, torch.tensor([0.02, 0.0, 0.10]), atol=1e-6, rtol=0
    )


def test_cup_gripper_quaternion_uses_normalized_nominal_orientation():
    nominal_quat = torch.tensor([2.0, 0.0, 0.0, 0.0])
    labware_yaw = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])

    result = grasp_geometry.yaw_aligned_gripper_quat(
        nominal_quat, labware_yaw, "cup"
    )

    torch.testing.assert_close(
        result, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6, rtol=0
    )


def test_slide_gripper_quaternion_tracks_labware_yaw():
    nominal_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    half_yaw = math.pi / 8.0
    labware_yaw = torch.tensor(
        [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]
    )

    result = grasp_geometry.yaw_aligned_gripper_quat(
        nominal_quat, labware_yaw, "slide"
    )

    torch.testing.assert_close(result, labware_yaw, atol=1e-6, rtol=0)
