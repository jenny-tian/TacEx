# Lab Pick Contact Sensor FT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LabPick's indentation-only FT estimate with IsaacLab ContactSensor-derived contact force vectors, and push only after one successful CAFE record is collected locally.

**Architecture:** Add left/right fingertip ContactSensor configurations filtered to the labware prim. Register the sensors in `LabPickEnv._setup_scene()`, read `force_matrix_w` or `net_forces_w`, transform the summed force/torque from world frame into robot base frame, and keep indentation-derived FT as a fallback if contact sensor data is unavailable. The CAFE writer and record layout remain unchanged.

**Tech Stack:** IsaacLab `ContactSensor`/`ContactSensorCfg`, PyTorch tensor math, pytest static tests, IsaacLab headless smoke data collection.

---

### Task 1: Contact Sensor Contract Test

**Files:**
- Modify: `source/tacex_tasks/test/test_lab_pick_static.py`
- Modify: `source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env_cfg.py`
- Modify: `source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py`

- [ ] Write a failing static test requiring `ContactSensor`, `ContactSensorCfg`, `left_finger_contact_sensor`, `right_finger_contact_sensor`, `_contact_force_from_sensor`, and `_contact_sensor_ft`.
- [ ] Run the test and verify it fails because contact sensors are not present.
- [ ] Add left/right contact sensor configs filtered to `/World/envs/env_.*/labware`.
- [ ] Instantiate the sensors in `_setup_scene()` and add them to `self.scene.sensors`.
- [ ] Run the test and verify it passes.

### Task 2: Contact Sensor FT Implementation

**Files:**
- Modify: `source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py`
- Modify: `source/tacex_tasks/test/test_lab_pick_static.py`

- [ ] Write a failing static test requiring `force_matrix_w`, `net_forces_w`, `math_utils.matrix_from_quat(math_utils.quat_inv`, and `torch.cross`.
- [ ] Run the test and verify it fails.
- [ ] Implement `_contact_force_from_sensor()` and `_contact_sensor_ft()`.
- [ ] Make `get_cafe_ft()` prefer contact sensor FT and fallback to `_indentation_ft()`.
- [ ] Run the test and verify it passes.

### Task 3: Successful Collection Gate

**Files:**
- No code changes unless validation exposes a runtime issue.

- [ ] Run static tests.
- [ ] Run py_compile on changed files.
- [ ] Run IsaacLab headless collection with enough steps and `--success_only`.
- [ ] Inspect the generated record and confirm `metadata["success"] == True`, `aligned_60Hz/ft.npy` shape is `(N,6)`, and at least one `Fx` or `Fy` sample is nonzero when contact is present.
- [ ] Only after that, commit and push.
