# Lab Pick Contact FT And Marker2D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LabPick's articulation-wrench FT and constant marker2d placeholders with contact-derived 6D wrench and GelSight-derived nonuniform marker displacement fields.

**Architecture:** Keep the CAFE record writer unchanged because its file layout is already aligned. Move the sensor semantics into `LabPickEnv`: `get_cafe_ft()` estimates a fingertip contact wrench from left/right GelSight indentation and gripper geometry, while `get_cafe_marker2d()` returns a `(num_envs, 14, 26, 2)` displacement field generated from indentation depth and fingertip/object relative motion. The collection script asks the environment for marker2d instead of constructing a constant placeholder.

**Tech Stack:** Python, PyTorch, NumPy, IsaacLab, pytest static tests, IsaacLab headless smoke tests.

---

### Task 1: Test Contact-Derived FT Contract

**Files:**
- Modify: `source/tacex_tasks/test/test_lab_pick_static.py`
- Modify: `source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env_cfg.py`
- Modify: `source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py`

- [ ] Write a failing static test requiring `contact_force_n_per_mm`, `contact_torque_arm_m`, `_estimate_contact_forces_from_tactile`, and no `body_incoming_joint_wrench_b` in `get_cafe_ft`.
- [ ] Run the test and confirm it fails.
- [ ] Implement `get_cafe_ft()` as `Fx,Fy,Fz,Tx,Ty,Tz` from left/right indentation-derived normal forces.
- [ ] Run the test and confirm it passes.

### Task 2: Test GelSight-Derived Marker2D Contract

**Files:**
- Modify: `source/tacex_tasks/test/test_lab_pick_static.py`
- Modify: `source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env_cfg.py`
- Modify: `source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py`

- [ ] Write a failing static test requiring `marker2d_rows`, `marker2d_cols`, `marker2d_sigma`, `get_cafe_marker2d`, and a Gaussian field expression.
- [ ] Run the test and confirm it fails.
- [ ] Implement `get_cafe_marker2d()` returning `(num_envs, 14, 26, 2)` based on left/right indentation, contact center, and object/tool relative lateral motion.
- [ ] Run the test and confirm it passes.

### Task 3: Use Environment Marker2D In CAFE Collection

**Files:**
- Modify: `scripts/demos/lab_pick/collect_bc_dataset.py`
- Modify: `source/tacex_tasks/test/test_lab_pick_static.py`

- [ ] Write a failing static test requiring `env.get_cafe_marker2d()` in the collection script and no constant `np.zeros((14, 26, 2))` marker placeholder.
- [ ] Run the test and confirm it fails.
- [ ] Update `_make_cafe_sample()` to use the environment marker2d tensor.
- [ ] Run the test and confirm it passes.

### Task 4: Verify End To End

**Files:**
- No planned code changes.

- [ ] Run `pytest source/tacex_tasks/test/test_lab_pick_static.py -q`.
- [ ] Run `python -m py_compile` on modified Python files.
- [ ] Run one IsaacLab headless collection smoke.
- [ ] Inspect output shapes and confirm `ft` is `(N,6)` and `marker2d_flatten` is `(N,728)`.
