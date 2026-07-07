# CAFE Strict Lab Pick Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LabPick data collection emit ForceCapture-CAFE-style raw record directories and aligned 60Hz arrays, with 30Hz RGB and 6D force/torque data.

**Architecture:** Keep the IsaacLab environment responsible for task state, reset randomization, success/failure, and per-step CAFE signals. Move CAFE file layout into a dedicated writer so the collection script only samples at target rates and flushes episodes. Preserve the existing HDF5 writer as a legacy path only if needed by old tests.

**Tech Stack:** Python, IsaacLab, PyTorch tensors, NumPy `.npy` arrays, imageio/PNG images, pytest static/unit tests.

---

### Task 1: Restore Clean Baseline

**Files:**
- Modify: `source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py`
- Modify: `source/tacex_tasks/test/test_lab_pick_static.py`

- [x] Remove the temporary tactile indentation break condition from `_get_dones`.
- [x] Remove static assertions that require `terminate_tactile_break_threshold_mm`.
- [x] Run `pytest source/tacex_tasks/test/test_lab_pick_static.py -q`; expected `7 passed`.

### Task 2: Add CAFE Strict Writer Tests

**Files:**
- Modify: `source/tacex_tasks/test/test_lab_pick_static.py`
- Modify: `source/tacex_tasks/tacex_tasks/lab_pick/bc_dataset.py`

- [ ] Write a failing test that constructs a CAFE record writer, appends synthetic samples at 60Hz/30Hz/90Hz/300Hz rates, flushes one episode, and asserts this layout:
  - `encoder/width.npy`
  - `encoder/timestamps.npy`
  - `tracker/xyz.npy`
  - `tracker/quat.npy`
  - `tracker/timestamps.npy`
  - `ftsensor/ft.npy`
  - `ftsensor/ft_compensated.npy`
  - `ftsensor/timestamps.npy`
  - `xense/marker2d.npy`
  - `xense/marker2d_flatten.npy`
  - `xense/timestamps.npy`
  - `camera/color/timestamps.npy`
  - `aligned_60Hz/xyz.npy`
  - `aligned_60Hz/quat.npy`
  - `aligned_60Hz/width.npy`
  - `aligned_60Hz/ft.npy`
  - `aligned_60Hz/marker2d.npy`
  - `aligned_60Hz/rgb.npy`
- [ ] Run that test and verify it fails because the strict writer does not exist.
- [ ] Implement `CafeRecordWriter` with a minimal append/flush API.
- [ ] Run the writer test and verify it passes.

### Task 3: Add 6D FT Environment Contract

**Files:**
- Modify: `source/tacex_tasks/test/test_lab_pick_static.py`
- Modify: `source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env_cfg.py`
- Modify: `source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py`

- [ ] Write a failing static test requiring:
  - `terminate_break_force_threshold_n`
  - `get_cafe_ft`
  - `force_norm = torch.linalg.norm(ft[:, :3], dim=1)`
  - `object_broken = force_norm > self.cfg.terminate_break_force_threshold_n`
  - `robot0_force` shape documented as 6D force/torque.
- [ ] Implement `get_cafe_ft()` using a 6D wrench source available from the robot articulation, with a zero fallback if the field is unavailable.
- [ ] Change `get_cafe_observation()` so `robot0_force` is `ft[:, 0:6]`.
- [ ] Include force-threshold break failure in `_get_dones`.
- [ ] Run the static test and verify it passes.

### Task 4: Update Collection Script

**Files:**
- Modify: `scripts/demos/lab_pick/collect_bc_dataset.py`
- Modify: `source/tacex_tasks/test/test_lab_pick_static.py`

- [ ] Write a failing static test requiring CLI flags `--record_dir`, `--camera_hz`, `--aligned_hz`, `--ft_hz`, `--tracker_hz`, and `CafeRecordWriter`.
- [ ] Replace default HDF5 output with record directory output while keeping `--dataset_file` as a deprecated alias if practical.
- [ ] Sample simulation steps at 120Hz, aligned state at 60Hz, camera at 30Hz, FT at 90Hz approximation, and tracker at 300Hz approximation/resampling from current pose.
- [ ] Flush each episode to `record_<index>` under the output directory.
- [ ] Run the static test and verify it passes.

### Task 5: Verify End to End

**Files:**
- No planned code changes.

- [ ] Run `pytest source/tacex_tasks/test/test_lab_pick_static.py -q`.
- [ ] Run `python -m py_compile` on modified Python files.
- [ ] Run one IsaacLab headless collection smoke with NVIDIA Vulkan/GL variables and `--num_demos 1 --max_episode_steps 120`.
- [ ] Inspect the generated record directory with Python and print key shapes and dtypes.
