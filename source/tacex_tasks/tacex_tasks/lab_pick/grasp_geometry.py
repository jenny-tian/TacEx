from __future__ import annotations

import torch


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat(quat)
    xyz = quat[..., 1:]
    uv = torch.cross(xyz, vector, dim=-1)
    uuv = torch.cross(xyz, uv, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def vector_in_local_frame(vector_b: torch.Tensor, frame_quat_b: torch.Tensor) -> torch.Tensor:
    inverse = _normalize_quat(frame_quat_b).clone()
    inverse[..., 1:] *= -1.0
    return _quat_apply(inverse, vector_b)


def centered_tool_target(
    object_center_b: torch.Tensor,
    target_quat_b: torch.Tensor,
    gripper_center_offset_tool: torch.Tensor,
) -> torch.Tensor:
    return object_center_b - _quat_apply(target_quat_b, gripper_center_offset_tool)


def yaw_aligned_gripper_quat(
    nominal_quat_b: torch.Tensor,
    labware_quat_b: torch.Tensor,
    labware_name: str,
) -> torch.Tensor:
    if labware_name == "cup":
        return _normalize_quat(nominal_quat_b)

    quat = _normalize_quat(labware_quat_b)
    w, x, y, z = quat.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half_yaw = 0.5 * yaw
    zeros = torch.zeros_like(half_yaw)
    yaw_quat = torch.stack((torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)), dim=-1)
    return _normalize_quat(_quat_mul(yaw_quat, nominal_quat_b))
