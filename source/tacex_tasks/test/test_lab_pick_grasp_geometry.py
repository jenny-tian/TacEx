from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "tacex_tasks" / "lab_pick" / "grasp_geometry.py"
SPEC = importlib.util.spec_from_file_location("lab_pick_grasp_geometry", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geometry)


def test_centered_tool_target_subtracts_rotated_pad_midpoint_offset():
    object_center_b = torch.tensor([[0.5, 0.1, 0.02]])
    yaw_90 = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
    offset_tool = torch.tensor([[0.02, 0.0, 0.10]])

    target = geometry.centered_tool_target(object_center_b, yaw_90, offset_tool)

    torch.testing.assert_close(target, torch.tensor([[0.5, 0.08, -0.08]]), atol=1e-6, rtol=0.0)


def test_vector_in_tool_frame_round_trips_through_target_orientation():
    yaw_90 = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
    vector_b = torch.tensor([[0.0, 0.02, 0.10]])

    local = geometry.vector_in_local_frame(vector_b, yaw_90)

    torch.testing.assert_close(local, torch.tensor([[0.02, 0.0, 0.10]]), atol=1e-6, rtol=0.0)


def test_yaw_aligned_gripper_quat_preserves_nominal_for_cup():
    nominal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    labware = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])

    result = geometry.yaw_aligned_gripper_quat(nominal, labware, "cup")

    torch.testing.assert_close(result, nominal)


def test_yaw_aligned_gripper_quat_rotates_slide_about_base_z():
    nominal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    yaw = math.pi / 4
    labware = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])

    result = geometry.yaw_aligned_gripper_quat(nominal, labware, "slide")

    torch.testing.assert_close(result, labware, atol=1e-6, rtol=0.0)
