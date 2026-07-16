import torch


def _normalize_quat(quat):
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(
        torch.finfo(quat.dtype).eps
    )


def _quat_mul(quat_a, quat_b):
    aw, ax, ay, az = quat_a.unbind(dim=-1)
    bw, bx, by, bz = quat_b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_apply(quat, vector):
    quat_vector = quat[..., 1:]
    uv = torch.cross(quat_vector, vector, dim=-1)
    uuv = torch.cross(quat_vector, uv, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def vector_in_local_frame(vector_b, frame_quat_b):
    frame_quat_b = _normalize_quat(frame_quat_b)
    frame_quat_conjugate = torch.cat(
        (frame_quat_b[..., :1], -frame_quat_b[..., 1:]), dim=-1
    )
    return _quat_apply(frame_quat_conjugate, vector_b)


def centered_tool_target(
    object_center_b, target_quat_b, gripper_center_offset_tool
):
    target_quat_b = _normalize_quat(target_quat_b)
    offset_b = _quat_apply(target_quat_b, gripper_center_offset_tool)
    return object_center_b - offset_b


def yaw_aligned_gripper_quat(
    nominal_quat_b, labware_quat_b, labware_name
):
    nominal_quat_b = _normalize_quat(nominal_quat_b)
    if labware_name == "cup":
        return nominal_quat_b

    labware_quat_b = _normalize_quat(labware_quat_b)
    w, x, y, z = labware_quat_b.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half_yaw = yaw / 2.0
    zeros = torch.zeros_like(half_yaw)
    yaw_quat = torch.stack(
        (torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)), dim=-1
    )
    return _normalize_quat(_quat_mul(yaw_quat, nominal_quat_b))
