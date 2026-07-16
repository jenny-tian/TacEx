# LabPick Centered Grasp and Dual-Camera Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the physical GelSight pad midpoint and gripper yaw with randomized labware while preserving wrist-camera data and adding synchronized third-person RGB data to LabPick records and failure diagnostics.

**Architecture:** Put quaternion and center-target calculations in a small Isaac-independent Torch module so the geometry is unit tested without launching Isaac Sim. `LabPickEnv` calibrates its physical pad midpoint at reset and consumes those helpers; `CafeRecordWriter` owns the backward-compatible two-camera directory schema; the collector captures both cameras once per rendered step and uses third-person RGB as the generic VLM failure image while retaining explicit views.

**Tech Stack:** Python 3.10, PyTorch, NumPy, Isaac Lab math/sensor APIs, pytest, Git/GitHub.

---

## File Map

- Create `source/tacex_tasks/tacex_tasks/lab_pick/grasp_geometry.py`: pure Torch quaternion, yaw-alignment, local-offset, and centered-target helpers.
- Create `source/tacex_tasks/test/test_lab_pick_grasp_geometry.py`: executable geometry unit tests without Isaac Sim startup.
- Modify `source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py`: calibrate pad midpoint, store reset-relative orientation, and command centered/yaw-aligned targets.
- Modify `source/tacex_tasks/tacex_tasks/lab_pick/bc_dataset.py`: store legacy wrist and new third-person raw/aligned streams.
- Modify `scripts/demos/lab_pick/collect_bc_dataset.py`: capture both cameras, pass both to the writer, and write two-view failure artifacts.
- Modify `source/tacex_tasks/test/test_lab_pick_static.py`: extend writer behavior and source integration checks.
- Modify `README.md`: document camera semantics and new failure artifacts.

## Execution Setup

Before Task 1, use `superpowers:using-git-worktrees` to create an isolated feature worktree from `main`. Because the repository was cloned sparsely, include all runtime packages required by the smoke test:

```bash
git sparse-checkout add source/tacex source/tacex_assets
```

Execute every task from the feature worktree. Push each task commit to the feature branch; merge only after all verification steps pass.

## Task 1: Pure Grasp Geometry Helpers

**Files:**
- Create: `source/tacex_tasks/tacex_tasks/lab_pick/grasp_geometry.py`
- Create: `source/tacex_tasks/test/test_lab_pick_grasp_geometry.py`

- [ ] **Step 1: Write failing tests for quaternion rotation and centered targeting**

Create `test_lab_pick_grasp_geometry.py` with an import-by-path fixture so importing the package does not start Isaac extensions:

```python
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
```

- [ ] **Step 2: Run the geometry tests and verify RED**

Run:

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  source/tacex_tasks/test/test_lab_pick_grasp_geometry.py -q
```

Expected: FAIL because `grasp_geometry.py` does not exist.

- [ ] **Step 3: Implement the minimal pure Torch geometry module**

Create `grasp_geometry.py` with normalized WXYZ quaternion operations and the three public helpers:

```python
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
```

- [ ] **Step 4: Run geometry tests and verify GREEN**

Run the command from Step 2.

Expected: `4 passed`.

- [ ] **Step 5: Commit and push the geometry unit**

```bash
git add source/tacex_tasks/tacex_tasks/lab_pick/grasp_geometry.py \
  source/tacex_tasks/test/test_lab_pick_grasp_geometry.py
git commit -m "feat: add LabPick grasp geometry helpers"
git push origin HEAD
```

## Task 2: Integrate Physical Pad-Center and Yaw Alignment

**Files:**
- Modify: `source/tacex_tasks/test/test_lab_pick_static.py`
- Modify: `source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py`

- [ ] **Step 1: Add failing source integration assertions**

Add a focused test:

```python
def test_lab_pick_scripted_grasp_targets_physical_pad_center_and_labware_yaw():
    env_source = read(TASK_ROOT / "lab_pick_env.py")
    assert "self.gripper_center_offset_tool" in env_source
    assert "def _calibrate_gripper_center_offset(" in env_source
    assert "left_pos_w = self._robot.data.body_link_pos_w[:, self._left_finger_body_idx]" in env_source
    assert "right_pos_w = self._robot.data.body_link_pos_w[:, self._right_finger_body_idx]" in env_source
    assert "yaw_aligned_gripper_quat(" in env_source
    assert "centered_tool_target(" in env_source
    assert "center_target_b = self.initial_object_pos_b.clone()" in env_source
    assert "target_pos_b = object_pos_b.clone()" not in env_source
```

- [ ] **Step 2: Run the new static test and verify RED**

Run:

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  source/tacex_tasks/test/test_lab_pick_static.py::test_lab_pick_scripted_grasp_targets_physical_pad_center_and_labware_yaw -q
```

Expected: FAIL at the first missing center-offset assertion.

- [ ] **Step 3: Add environment state and center calibration**

Import the helpers and initialize:

```python
from .grasp_geometry import centered_tool_target, vector_in_local_frame, yaw_aligned_gripper_quat

self.gripper_center_offset_tool = torch.zeros((self.num_envs, 3), device=self.device)
self.scripted_target_quat_b = self.nominal_ee_quat_b.clone()
```

Add calibration using the actual pad body midpoint:

```python
def _calibrate_gripper_center_offset(self, env_ids: torch.Tensor):
    left_pos_w = self._robot.data.body_link_pos_w[:, self._left_finger_body_idx]
    right_pos_w = self._robot.data.body_link_pos_w[:, self._right_finger_body_idx]
    midpoint_w = 0.5 * (left_pos_w + right_pos_w)
    root_pos_w = self._robot.data.root_link_pos_w
    root_quat_w = self._robot.data.root_link_quat_w
    midpoint_b = math_utils.quat_apply(math_utils.quat_inv(root_quat_w), midpoint_w - root_pos_w)
    tool_pos_b, tool_quat_b = self._compute_frame_pose()
    offset_b = midpoint_b - tool_pos_b
    self.gripper_center_offset_tool[env_ids] = vector_in_local_frame(offset_b, tool_quat_b)[env_ids]
```

- [ ] **Step 4: Store reset-relative labware yaw target**

In `_reset_idx`, after obtaining `nominal_ee_quat_b`, compute the labware quaternion in the robot base frame and cache the scripted orientation:

```python
labware_quat_b = math_utils.quat_mul(
    math_utils.quat_inv(self._robot.data.root_link_quat_w[env_ids]),
    self.labware_reset_quat_w[env_ids],
)
self.scripted_target_quat_b[env_ids] = yaw_aligned_gripper_quat(
    self.nominal_ee_quat_b[env_ids],
    labware_quat_b,
    self.labware_name,
)
self._calibrate_gripper_center_offset(env_ids)
```

- [ ] **Step 5: Refactor the scripted state machine around a center target**

Start each command from the stable reset center and cached orientation:

```python
center_target_b = self.initial_object_pos_b.clone()
target_quat_b = self.scripted_target_quat_b
```

Apply the existing phase-specific Z offsets to `center_target_b[:, 2]`. After the phase branch, compute and record the IK target:

```python
target_pos_b = centered_tool_target(
    center_target_b,
    target_quat_b,
    self.gripper_center_offset_tool,
)
self.ik_commands[:, :3] = target_pos_b
self.ik_commands[:, 3:7] = target_quat_b
self.last_target_pos_b[:] = target_pos_b
self.last_target_quat_b[:] = target_quat_b
```

Keep the existing phase timings, widths, lift condition, and optional lift-assist call unchanged.

The optional lift assist accepts an object-center target, not an IK tool target. Preserve that contract explicitly:

```python
if self.cfg.scripted_lift_assist_on_contact and phase >= close_end + squeeze_steps:
    self._apply_scripted_lift_assist(center_target_b)
```

- [ ] **Step 6: Run geometry and static tests**

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  source/tacex_tasks/test/test_lab_pick_grasp_geometry.py \
  source/tacex_tasks/test/test_lab_pick_static.py::test_lab_pick_scripted_grasp_targets_physical_pad_center_and_labware_yaw -q
```

Expected: all selected tests PASS.

- [ ] **Step 7: Commit and push environment integration**

```bash
git add source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py \
  source/tacex_tasks/test/test_lab_pick_static.py
git commit -m "feat: center LabPick grasps on tactile pads"
git push origin HEAD
```

## Task 3: Extend CafeRecordWriter With Third-Person RGB

**Files:**
- Modify: `source/tacex_tasks/test/test_lab_pick_static.py`
- Modify: `source/tacex_tasks/tacex_tasks/lab_pick/bc_dataset.py`

- [ ] **Step 1: Extend the real writer test first**

In `test_cafe_record_writer_outputs_forcecapture_cafe_directory`, add `third_rgb` to each sample and pass both images:

```python
"third_rgb": np.full((72, 128, 3), index + 10, dtype=np.uint8),
```

```python
writer.append_camera_sample(timestamp, sample["rgb"], sample["third_rgb"])
```

Add expected files and shape assertions:

```python
"camera/color/rgb.npy",
"camera/third/color/rgb.npy",
"camera/third/color/timestamps.npy",
"aligned_60Hz/third_rgb.npy",
```

```python
assert np.load(record_dir / "aligned_60Hz/third_rgb.npy").shape == (6, 72, 128, 3)
assert np.load(record_dir / "camera/third/color/rgb.npy").shape == (3, 72, 128, 3)
assert np.load(record_dir / "camera/third/color/timestamps.npy").shape == (3,)
assert writer.camera_rgb == []
assert writer.third_camera_rgb == []
```

- [ ] **Step 2: Run the writer test and verify RED**

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  source/tacex_tasks/test/test_lab_pick_static.py::test_cafe_record_writer_outputs_forcecapture_cafe_directory -q
```

Expected: FAIL because `append_camera_sample` accepts only wrist RGB and aligned samples do not write `third_rgb`.

- [ ] **Step 3: Add third-person writer state and aligned data**

In `CafeRecordWriter.__init__` add:

```python
self.third_camera_timestamps: list[float] = []
self.third_camera_rgb: list[np.ndarray] = []
```

In `append_aligned_sample` require and copy:

```python
"third_rgb": np.asarray(sample["third_rgb"], dtype=np.uint8),
```

Change the raw camera append signature:

```python
def append_camera_sample(self, timestamp: float, rgb: np.ndarray, third_rgb: np.ndarray):
    self.camera_timestamps.append(float(timestamp))
    self.camera_rgb.append(np.asarray(rgb, dtype=np.uint8))
    self.third_camera_timestamps.append(float(timestamp))
    self.third_camera_rgb.append(np.asarray(third_rgb, dtype=np.uint8))
```

- [ ] **Step 4: Write both streams during flush**

Add aligned output:

```python
np.save(aligned / "third_rgb.npy", self._stack_aligned("third_rgb", dtype=np.uint8))
```

Write the new raw path, using `(0, 720, 1280, 3)` only for an empty stream:

```python
third_color = self.record_dir / "camera" / "third" / "color"
np.save(third_color / "timestamps.npy", np.asarray(self.third_camera_timestamps, dtype=np.float64))
if self.third_camera_rgb:
    np.save(third_color / "rgb.npy", np.stack(self.third_camera_rgb, axis=0))
else:
    np.save(third_color / "rgb.npy", np.zeros((0, 720, 1280, 3), dtype=np.uint8))
```

Add `camera/third/color` to `_mkdirs` and clear both new lists in `clear_episode`.

- [ ] **Step 5: Run the writer test and full geometry tests**

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  source/tacex_tasks/test/test_lab_pick_static.py::test_cafe_record_writer_outputs_forcecapture_cafe_directory \
  source/tacex_tasks/test/test_lab_pick_grasp_geometry.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 6: Commit and push writer schema**

```bash
git add source/tacex_tasks/tacex_tasks/lab_pick/bc_dataset.py \
  source/tacex_tasks/test/test_lab_pick_static.py
git commit -m "feat: store LabPick third-person camera stream"
git push origin HEAD
```

## Task 4: Capture Both Cameras and Preserve Failure Views

**Files:**
- Modify: `source/tacex_tasks/test/test_lab_pick_static.py`
- Modify: `scripts/demos/lab_pick/collect_bc_dataset.py`

- [ ] **Step 1: Write failing collector integration assertions**

Update `test_lab_pick_collection_script_uses_forcecapture_cafe_record_layout`:

```python
assert 'env.wrist_camera.data.output["rgb"]' in script_source
assert 'env.third_person_camera.data.output["rgb"]' in script_source
assert '"third_rgb": third_rgb' in script_source
assert 'writer.append_camera_sample(next_camera_t, sample["rgb"], sample["third_rgb"])' in script_source
assert 'primary_rgb = np.asarray(sample["third_rgb"], dtype=np.uint8)' in script_source
assert 'f"{prefix}_wrist_rgb.npy"' in script_source
assert 'f"{prefix}_third_rgb.npy"' in script_source
```

- [ ] **Step 2: Run the collector static test and verify RED**

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  source/tacex_tasks/test/test_lab_pick_static.py::test_lab_pick_collection_script_uses_forcecapture_cafe_record_layout -q
```

Expected: FAIL because the collector currently reads and forwards only wrist RGB.

- [ ] **Step 3: Capture both rendered camera frames**

Change `_make_cafe_sample` to read both streams after the existing render call:

```python
wrist_rgb = env.wrist_camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy().astype(np.uint8)
third_rgb = env.third_person_camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy().astype(np.uint8)
```

Return:

```python
"rgb": wrist_rgb,
"third_rgb": third_rgb,
```

Pass both images to `append_camera_sample`.

- [ ] **Step 4: Write generic third-person and explicit two-view failure artifacts**

In `_write_frame_debug`, use:

```python
wrist_rgb = np.asarray(sample["rgb"], dtype=np.uint8)
third_rgb = np.asarray(sample["third_rgb"], dtype=np.uint8)
primary_rgb = np.asarray(sample["third_rgb"], dtype=np.uint8)
```

Retain generic `f"{prefix}_rgb.*"` outputs from `primary_rgb`. Additionally save:

```python
np.save(debug_dir / f"{prefix}_wrist_rgb.npy", wrist_rgb)
np.save(debug_dir / f"{prefix}_third_rgb.npy", third_rgb)
wrist_preview = _save_rgb_preview(debug_dir / f"{prefix}_wrist_rgb.png", wrist_rgb)
third_preview = _save_rgb_preview(debug_dir / f"{prefix}_third_rgb.png", third_rgb)
```

Add the explicit NPY and preview paths to the info text. Keep FT handling unchanged.

- [ ] **Step 5: Run collector and writer tests**

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  source/tacex_tasks/test/test_lab_pick_static.py::test_lab_pick_collection_script_uses_forcecapture_cafe_record_layout \
  source/tacex_tasks/test/test_lab_pick_static.py::test_cafe_record_writer_outputs_forcecapture_cafe_directory -q
```

Expected: both tests PASS.

- [ ] **Step 6: Commit and push collector integration**

```bash
git add scripts/demos/lab_pick/collect_bc_dataset.py \
  source/tacex_tasks/test/test_lab_pick_static.py
git commit -m "feat: collect LabPick wrist and third-person views"
git push origin HEAD
```

## Task 5: Documentation and Complete Static Verification

**Files:**
- Modify: `README.md`
- Modify: `source/tacex_tasks/test/test_lab_pick_static.py`

- [ ] **Step 1: Add failing README contract assertions**

Add a test that checks the documented paths and semantics:

```python
def test_readme_documents_lab_pick_dual_camera_records():
    readme = read(ROOT / "README.md")
    assert "camera/third/color/" in readme
    assert "third_rgb.npy" in readme
    assert "wrist camera" in readme.lower()
    assert "third-person camera" in readme.lower()
    assert "failure_frame_wrist_rgb" in readme
    assert "failure_frame_third_rgb" in readme
```

- [ ] **Step 2: Run the README test and verify RED**

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  source/tacex_tasks/test/test_lab_pick_static.py::test_readme_documents_lab_pick_dual_camera_records -q
```

Expected: FAIL because the new paths are not documented.

- [ ] **Step 3: Update README record layout and camera semantics**

Document the legacy wrist and new third-person paths:

```text
camera/color/rgb.npy                 # wrist RGB at camera_hz
camera/third/color/rgb.npy           # third-person RGB at camera_hz
aligned_60Hz/rgb.npy                 # aligned wrist RGB
aligned_60Hz/third_rgb.npy           # aligned third-person RGB
```

Document explicit failure frame files and state that generic `failure_frame_rgb`/`last_frame_rgb` now contain third-person RGB for VLM compatibility.

- [ ] **Step 4: Run all focused tests and syntax checks**

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  source/tacex_tasks/test/test_lab_pick_grasp_geometry.py \
  source/tacex_tasks/test/test_lab_pick_static.py -q
```

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m py_compile \
  source/tacex_tasks/tacex_tasks/lab_pick/grasp_geometry.py \
  source/tacex_tasks/tacex_tasks/lab_pick/bc_dataset.py \
  source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py \
  scripts/demos/lab_pick/collect_bc_dataset.py
```

Expected: all tests PASS and compilation exits `0`.

- [ ] **Step 5: Commit and push documentation**

```bash
git add README.md source/tacex_tasks/test/test_lab_pick_static.py
git commit -m "docs: describe LabPick dual-camera records"
git push origin HEAD
```

## Task 6: Isaac Lab Smoke Test and Slip-Rate Evidence

**Files:**
- No source changes expected.
- Runtime output: `/tmp/lab_pick_centered_dual_camera_smoke`

- [ ] **Step 1: Run a deterministic one-attempt smoke collection**

```bash
timeout 240s env \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/demos/lab_pick/collect_bc_dataset.py \
  --labware slide \
  --num_envs 1 \
  --num_demos 1 \
  --max_attempts 1 \
  --max_episode_steps 960 \
  --record_dir /tmp/lab_pick_centered_dual_camera_smoke \
  --seed 0 \
  --headless
```

Expected: one completed attempt with either a record or a failure debug directory; no camera or tensor-shape exception.

- [ ] **Step 2: Verify record shapes when a record was written**

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -c '
from pathlib import Path
import numpy as np
root = Path("/tmp/lab_pick_centered_dual_camera_smoke/record_000000")
if root.exists():
    wrist = np.load(root / "camera/color/rgb.npy", mmap_mode="r")
    third = np.load(root / "camera/third/color/rgb.npy", mmap_mode="r")
    aligned_wrist = np.load(root / "aligned_60Hz/rgb.npy", mmap_mode="r")
    aligned_third = np.load(root / "aligned_60Hz/third_rgb.npy", mmap_mode="r")
    assert wrist.shape[1:] == (480, 640, 3), wrist.shape
    assert third.shape[1:] == (720, 1280, 3), third.shape
    assert aligned_wrist.shape[0] == aligned_third.shape[0]
    print(wrist.shape, third.shape, aligned_wrist.shape, aligned_third.shape)
else:
    print("attempt produced failure debug instead of a record")
'
```

Expected: configured wrist/third resolutions and equal aligned counts, or an explicit failure-debug message.

- [ ] **Step 3: Run matched 20-attempt baseline and changed batches**

Create a detached baseline worktree at the last pre-feature production commit:

```bash
git worktree add --detach /tmp/TacEx-centered-grasp-baseline 2c85cd1
```

Run the baseline from `/tmp/TacEx-centered-grasp-baseline`:

```bash
timeout 1800s env \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/demos/lab_pick/collect_bc_dataset.py \
  --labware slide \
  --num_envs 1 \
  --num_demos 20 \
  --max_attempts 20 \
  --max_episode_steps 960 \
  --record_dir /tmp/lab_pick_baseline_seed0_20 \
  --seed 0 \
  --headless | tee /tmp/lab_pick_baseline_seed0_20.log
```

Run the changed branch from the feature worktree with the same arguments except output paths:

```bash
timeout 1800s env \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/demos/lab_pick/collect_bc_dataset.py \
  --labware slide \
  --num_envs 1 \
  --num_demos 20 \
  --max_attempts 20 \
  --max_episode_steps 960 \
  --record_dir /tmp/lab_pick_centered_seed0_20 \
  --seed 0 \
  --headless | tee /tmp/lab_pick_centered_seed0_20.log
```

Extract matched summaries:

```bash
rg '^\[SUMMARY\]' /tmp/lab_pick_baseline_seed0_20.log /tmp/lab_pick_centered_seed0_20.log
```

Expected: both runs report 20 attempts. Record the observed success/failure counts; do not claim reduced slip unless the changed run improves under these matched conditions.

- [ ] **Step 4: Inspect Git and remote synchronization**

```bash
git status --short --branch
git log -5 --oneline
git rev-parse HEAD
git rev-parse origin/main
```

Expected: clean worktree and identical local/remote commit IDs.
