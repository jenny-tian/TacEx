# LabPick Centered Grasp and Dual-Camera Collection Design

Date: 2026-07-16
Status: Approved for specification

## Objective

Reduce LabPick slip risk by aligning the physical midpoint between the two GelSight contact pads with the labware center and aligning the gripper yaw with non-axisymmetric labware. Extend CAFE-style collection with a synchronized third-person RGB stream while preserving the existing wrist-camera paths.

## Scope

This change covers the scripted LabPick collector and its `LabPickEnv` behavior for `slide`, `coverslip`, and `cup`. It extends `CafeRecordWriter`, failure debug artifacts, documentation, and focused tests.

It does not add image-based object detection, automatic force tuning, policy-model architecture changes, or an online VLM control loop.

## Centered Grasp Geometry

### Gripper center calibration

At environment reset, calculate the midpoint of the left and right GelSight pad body positions. Express the vector from the IK tool frame origin to that midpoint in the IK tool frame. Store this local vector per environment as the gripper-center offset.

For a desired tool orientation `target_quat_b`, rotate the stored local offset into the robot base frame. Compute the IK position target as:

```text
target_tool_pos_b = target_object_center_b - rotate(target_quat_b, gripper_center_offset_tool)
```

The target object's Z coordinate is then adjusted for hover, grasp, and lift phases using the existing phase schedule. This makes the physical pad midpoint, rather than the assumed tool origin, the alignment reference.

### Stable object target

Use the labware reset position as the XY grasp target throughout the scripted attempt. Do not chase the live object position once physics can move the object. This prevents contact-induced object motion from feeding back into the scripted approach target.

### Yaw alignment

At reset, derive the labware yaw relative to the robot base frame. For `slide` and `coverslip`, rotate the nominal end-effector orientation around the base-frame Z axis by this yaw. For the rotationally symmetric `cup`, retain the nominal orientation.

The aligned target quaternion is used throughout hover, approach, closing, squeeze, and lift phases and is recorded in the CAFE action.

### Grip schedule

The existing close widths, close timing, squeeze duration, lift height, and 6 N default break-force threshold remain unchanged. Center and yaw alignment are isolated first so their effect on slip rate can be measured without simultaneously changing gripping force.

## Dual-Camera Data Model

### Sample fields

Each collector sample contains:

- `rgb`: wrist RGB, retained as the legacy primary camera field.
- `third_rgb`: third-person RGB from `env.third_person_camera`.

Both images are captured after the same simulation render and share the sample timestamp.

### Raw and aligned output

Existing wrist-camera files remain unchanged:

```text
camera/color/rgb.npy
camera/color/timestamps.npy
aligned_60Hz/rgb.npy
```

New third-person files are added:

```text
camera/third/color/rgb.npy
camera/third/color/timestamps.npy
aligned_60Hz/third_rgb.npy
```

The third-person raw stream uses the configured camera rate. The aligned stream stores the third-person frame associated with each aligned sample. No implicit resize is performed, so wrist and third-person arrays retain their configured resolutions.

## Failure Debug and VLM Input

Failure artifacts retain the existing generic names for analyzer compatibility, but the generic RGB is changed to the third-person view because it provides a clearer view of gripper-object alignment and drops:

```text
failure_frame_rgb.npy/png
last_frame_rgb.npy/png
```

Each failure frame also stores explicitly named views:

```text
failure_frame_wrist_rgb.npy/png
failure_frame_third_rgb.npy/png
last_frame_wrist_rgb.npy/png
last_frame_third_rgb.npy/png
```

The existing VLM analyzer continues reading the generic `*_rgb` path and therefore receives the third-person image without an API or schema change. FT arrays and info files retain their current names and semantics.

## Compatibility

- Existing consumers of `camera/color/rgb.npy` and `aligned_60Hz/rgb.npy` continue receiving wrist images.
- Existing failure analyzer inputs remain present.
- Existing records without `third_rgb` remain readable; only newly written records contain the added stream.
- The metadata schema remains compatible. The README documents the additional files and identifies the camera associated with each path.

## Error Handling

- `CafeRecordWriter.append_aligned_sample` requires both `rgb` and `third_rgb` for newly collected samples. A missing stream fails immediately instead of silently writing unsynchronized data.
- Empty raw camera streams are written with zero frames and the configured per-stream fallback shape used by the writer.
- Camera timestamps are recorded independently in writer state, even though both streams currently use the same configured sampling schedule.

## Testing

### Geometry tests

Add testable quaternion and center-target helpers that do not require launching Isaac Sim. Verify:

- zero local center offset targets the object center;
- a known tool-local offset is rotated and subtracted correctly;
- zero labware yaw preserves the nominal quaternion;
- nonzero yaw rotates the gripper target around base Z;
- cup orientation remains nominal.

### Writer tests

Extend the real `CafeRecordWriter` filesystem test to verify:

- legacy wrist files remain present with their existing shapes;
- third-person raw RGB and timestamp files exist with expected shapes;
- `aligned_60Hz/third_rgb.npy` exists and matches aligned sample count;
- clearing or flushing an episode clears both camera buffers.

### Static integration checks

Verify that the collector reads both `wrist_camera` and `third_person_camera`, passes both fields to the writer, and failure debug writes the generic third-person image plus explicitly named wrist and third-person images.

### Verification

Run the focused static/unit test module and Python syntax compilation. If the Isaac Lab environment is available, run a short seeded slide collection and inspect one record for center alignment, both camera streams, and matching timestamps.

## Acceptance Criteria

1. For randomized slide and coverslip reset poses, the commanded physical pad midpoint targets the labware center and the commanded yaw follows the labware yaw.
2. Cup behavior keeps its nominal gripper orientation.
3. Existing wrist RGB paths and shapes remain compatible.
4. New third-person raw and aligned streams are written with synchronized sample counts and timestamps.
5. Failure analysis uses the third-person generic image while retaining both explicit views.
6. Focused tests and syntax checks pass.

