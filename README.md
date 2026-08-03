#

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
<!-- [![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/) -->
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

**Keywords:** tactile sensing, gelsight, isaaclab, vision-based-tactile-sensor, vbts, reinforcement learning

> [!note]
> **Preview Release**:
>
> The framework is under active development and currently in its beta phase.
> If you encounter bugs or have suggestions on how the framework can be improved, please tell us about them (e.g. via [Issues](https://github.com/DH-Ng/TacEx/issues)/[Discussions](https://github.com/DH-Ng/TacEx/discussions)).


# TacEx - Tactile Extension for Isaac Sim/Isaac Lab
**TacEx** brings **Vision-Based Tactile Sensor (VBTS)** into Isaac Sim/Lab.

This fork, `jenny-tian/TacEx`, adds an IsaacLab LabPick data-collection pipeline for generating ForceCapture-CAFE-compatible behavior cloning records. The original TacEx framework, citation, and acknowledgements are preserved below.

The framework integrates multiple simulation approaches for VBTS's and aims to be modular and extendable.
Components can be easily switched out, added and modified.

Currently, only the **GelSight Mini** is supported, but you can also easily add your own sensor (guide coming soon). We also plan to add more VBTS types later.

## **Main features**:
- [GPU accelerated Tactile RGB simulation](https://github.com/TimSchneider42/taxim) via [Taxim](https://github.com/Robo-Touch/Taxim)'s simulation approach
- Marker Motion Simulation via [FOTS](https://github.com/Rancho-zhao/FOTS)
- Integration of [UIPC](https://github.com/spiriMirror/libuipc) for GPU accelerated incremental potential contact to simulate FEM soft bodies, rigid bodies, cloth, etc. in a penetration-free and robust manner
- Marker Motion Simulation with FEM soft body based on the simulator used by the [ManiSkill-ViTac challenge](https://github.com/chuanyune/ManiSkill-ViTac2025) that leverages UIPC


Checkout the [website](https://sites.google.com/view/tacex) for showcases and the documentation for details, guides and tutorials.


## Installation
> [!NOTE]
> TacEx currently works with **Isaac Sim 4.5** and **IsaacLab 2.1.1**.
> The installation was tested on Ubuntu 22.04 with a 4090 GPU and Driver Version 550.163.01 + Cuda 12.4.

**0.** Make sure that you have **git-lfs**:

```bash
# Need it for the USD assets
git lfs install
```

**1.** Clone this repository and its submodules:
```bash
git clone --recurse-submodules https://github.com/jenny-tian/TacEx
cd TacEx
```

Then **install TacEx** [locally](docs/source/installation/Local-Installation.md)
or build a [Docker Container](docs/source/installation/Docker-Container-Setup.md).

## LabPick CAFE Data Collection

This fork includes a LabPick task for collecting slide/coverslip/cup manipulation demonstrations in a ForceCapture-CAFE-style record layout. The ForceCapture-CAFE repository is not vendored into this project; the data schema is matched for downstream compatibility.

### What is collected

Each demonstration is written as a `record_xxxxxx/` directory containing raw-style streams and aligned arrays:

```text
record_xxxxxx/
  metadata.npz
  encoder/
    width.npy
    timestamps.npy
  tracker/
    xyz.npy
    quat.npy
    timestamps.npy
  ftsensor/
    ft.npy
    ft_compensated.npy
    timestamps.npy
  xense/
    marker2d.npy
    marker2d_flatten.npy
    timestamps.npy
  aligned/
    xyz.npy
    quat.npy
    width.npy
    ft.npy
    marker2d.npy
    rgb.npy
    rgb_third.npy
    action.npy
    timestamps.npy
```

The default stream rates follow the ForceCapture-CAFE collection setup:

- RGB color: `30 Hz`, `480 x 640 x 3`, `uint8`
- aligned observations: `60 Hz`
- force/torque: `90 Hz`, 6D `Fx,Fy,Fz,Tx,Ty,Tz`
- tracker pose: `300 Hz`, `xyz + quat`
- tactile marker displacement: `60 Hz`, raw `(14, 26, 2)` and flattened `728`

In simulation, `ft` is generated from Isaac Lab `ContactSensor` readings on the left and right GelSight/fingertip pads. The exported 6D wrench is the net fingertip contact force and torque transformed into the robot base frame. `marker2d` is generated as a nonuniform GelSight-derived displacement field. These are physically motivated simulation signals, not real hardware sensor readings.

### Task criteria

- Labware reset pose is randomized each episode for better behavior-cloning generalization.
- A slide demonstration is marked successful when the labware is lifted at least `0.20 m` above its reset height.
- The scripted slide expert trajectory lifts to `0.25 m`.
- A demonstration is terminated as broken/failed if the net fingertip contact force exceeds `6 N`.

### Collect one slide demonstration

Run from the repository root:

```bash
timeout 240s env \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
  /home/tjx/miniforge3/envs/env_isaaclab/bin/python \
  scripts/demos/lab_pick/collect_bc_dataset.py \
  --labware slide \
  --num_envs 1 \
  --num_demos 1 \
  --max_episode_steps 960 \
  --record_dir /tmp/lab_pick_cafe_records \
  --success_only \
  --headless
```
If you have already started your virtual environmrnt, you can use the following command
```
  env __GLX_VENDOR_LIBRARY_NAME=nvidia \
    PYTHONUNBUFFERED=1 \
    python scripts/demos/lab_pick/collect_bc_dataset.py \
    --labware slide \
    --num_envs 1 \
    --num_demos 1 \
    --max_episode_steps 960 \
    --record_dir /tmp/lab_pick_cafe_records \
    --success_only 
```
or
```
env __GLX_VENDOR_LIBRARY_NAME=nvidia \
PYTHONUNBUFFERED=1 \
python scripts/demos/lab_pick/collect_bc_dataset.py \
  --labware slide \
  --num_envs 1 \
  --num_demos 50 \
  --record_dir ./dataset/ \
  --max_attempts 100 \
  --break_force_threshold_n 6.0 \
  --aligned_hz 30
```
Useful options:

- `--labware slide|coverslip|cup`
- `--num_demos 100`
- `--success_only`
- `--failure_only --max_attempts 10` to keep resetting until a failed attempt is recorded
- `--break_force_threshold_n 6.0` to explicitly set the break-force threshold for a run
- `--record_dir /path/to/output`
- `--aligned_hz 60 --camera_hz 30 --ft_hz 90 --tracker_hz 300`

Failed attempts are not stopped early. The script finishes the full `--max_episode_steps` episode, then writes a debug snapshot under `failed_attempts/attempt_xxxxxx/` with:

- `failure_frame_rgb.npy` and `failure_frame_rgb.png`/`.ppm`
- `failure_frame_ft.npy`
- `failure_frame_info.txt`, captured at the first frame that triggers the failure condition
- `last_frame_rgb.npy` and `last_frame_rgb.png`/`.ppm`
- `last_frame_ft.npy`
- `last_frame_info.txt`, including failure reason, final FT, force norm, torque norm, and the first failure step

### Train Flow Matching BC and DSRL-SAC

The commands below are the supported training path on this branch. They use the
Isaac Lab environment for simulation and Flow Matching training, and a separate
LeRobot environment only for optional dataset conversion.

Set the paths once from the repository root. Replace the Python paths if your
environments are installed elsewhere:

```bash
export TACEX_ROOT="$PWD"
export ISAAC_PYTHON="${TACEX_ISAAC_PYTHON:-/home/tjx/miniforge3/envs/env_isaaclab/bin/python}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$TACEX_ROOT/source/tacex:$TACEX_ROOT/source/tacex_assets:$TACEX_ROOT/source/tacex_tasks:$TACEX_ROOT/bc_policy${PYTHONPATH:+:$PYTHONPATH}"
```

For a remote desktop or TurboVNC session, run camera-enabled headless commands
with the graphics interposition variables removed:

```bash
unset LD_PRELOAD VGL_ISACTIVE VGL_DISPLAY DISPLAY
```

#### 1. Collect demonstrations

Collect at least 50 successful episodes for a smoke test; 200-500 episodes are
recommended for a useful BC policy:

```bash
env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
  "$ISAAC_PYTHON" scripts/demos/lab_pick/collect_bc_dataset.py \
  --labware slide \
  --num_demos 200 \
  --max_attempts 500 \
  --max_episode_steps 960 \
  --success_only \
  --record_dir datasets/lab_pick_slide_records \
  --headless
```

Check that the records contain aligned RGB, third-person RGB, state, width and
action arrays:

```bash
"$ISAAC_PYTHON" scripts/reinforcement_learning/dsrl/check_dsrl_ready.py \
  --records datasets/lab_pick_slide_records --min-success 50
```

#### 2. Convert records to the Flow Matching HDF5 format

The third-person camera is included explicitly because the dual-camera BC
policy consumes both `robot0_image` and `robot0_image_third`:

```bash
"$ISAAC_PYTHON" bc_policy/sim_robot/scripts/convert_records_to_hdf5.py \
  --input datasets/lab_pick_slide_records \
  --output datasets/lab_pick_slide_flow_matching.hdf5 \
  --success-only \
  --include-third-camera \
  --overwrite
```

#### 3. Train the dual-camera Flow Matching BC policy

This configuration matches the validated LabPick policy: phase conditioning,
ImageNet-normalized ImageNet-pretrained ResNet50 features, visual XY auxiliary
loss, and 32-step action chunks. The best checkpoint is written to
`outputs/lab_pick_flow_matching/best.pt`.

```bash
"$ISAAC_PYTHON" bc_policy/sim_robot/scripts/train_flow_matching.py \
  --dataset datasets/lab_pick_slide_flow_matching.hdf5 \
  --output-dir outputs/lab_pick_flow_matching \
  --success-only \
  --action-key high \
  --image-keys robot0_image,robot0_image_third \
  --n-state-obs-steps 2 \
  --n-image-obs-steps 2 \
  --n-action-steps 32 \
  --epochs 50 \
  --batch-size 32 \
  --num-workers 4 \
  --lr 1e-4 \
  --weight-decay 1e-6 \
  --warmup-steps 500 \
  --val-ratio 0.05 \
  --seed 42 \
  --normalizer-mode limits \
  --image-normalization imagenet \
  --pretrained-image-backbone resnet50_imagenet1k_v2 \
  --image-feature-dim 512 \
  --obs-feature-dim 512 \
  --transformer-layers 6 \
  --transformer-heads 8 \
  --transformer-embedding-dim 512 \
  --transformer-cond-layers 2 \
  --dropout 0.1 \
  --num-inference-steps 100 \
  --ode-solver euler \
  --include-phase \
  --visual-xy-loss-weight 1.0 \
  --ema-decay 0.999 \
  --amp \
  --save-every 10
```

For a short smoke test, add `--max-train-steps 20 --max-val-steps 5`.
The output contains `best.pt`, `last.pt` and `logs.jsonl`.

#### 4. Evaluate the frozen BC policy

Use the same camera, phase and action-chunk settings during evaluation:

```bash
env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
  "$ISAAC_PYTHON" scripts/demos/lab_pick/eval_flow_matching_policy.py \
  --checkpoint outputs/lab_pick_flow_matching/best.pt \
  --num_trials 50 \
  --seed 0 \
  --policy_seed 42 \
  --action_repeat 2 \
  --chunk_execute_steps 32 \
  --num_inference_steps 20 \
  --phase_horizon_steps 383 \
  --camera_warmup_steps 8 \
  --visual_xy_lock_phase 0.30 \
  --policy_camera third \
  --labware_random_xy 0.10 0.10 \
  --labware_random_yaw 0.7853981634 \
  --output logs/lab_pick_eval/bc_wide_seed0_49.json \
  --headless
```

#### 5. Train DSRL-SAC initialized from the BC policy

The DSRL launcher automatically selects
`TacEx-LabPick-Slide-DSRL-Base-v0`, enables cameras and runs headless. It
requires the isolated `.cache/lerobot_inference` dependencies described in
[`scripts/reinforcement_learning/dsrl/README.md`](scripts/reinforcement_learning/dsrl/README.md).

The recommended Flow Matching setup uses a 4-D normalized residual
`[dx, dy, dz, dwidth]` instead of the legacy 320-D initial-noise action. The
residual mean starts at zero, so the initial deterministic policy is the frozen BC, and the reset
distribution expands from +/-5 cm and +/-30 degrees to +/-10 cm and +/-45
degrees over the first 100k outer steps:

```bash
env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
  "$ISAAC_PYTHON" scripts/reinforcement_learning/dsrl/train_lab_pick_dsrl_sac.py \
  --dsrl_policy outputs/lab_pick_flow_matching/best.pt \
  --dsrl_policy_type flow_matching \
  --dsrl_action_mode residual \
  --dsrl_residual_position_scale_m 0.03 0.03 0.01 \
  --dsrl_residual_width_scale_m 0.002 \
  --dsrl_curriculum_steps 100000 \
  --dsrl_curriculum_start_xy_m 0.05 0.05 \
  --dsrl_curriculum_end_xy_m 0.10 0.10 \
  --dsrl_curriculum_start_yaw_deg 30 \
  --dsrl_curriculum_end_yaw_deg 45 \
  --timesteps 200000 \
  --seed 42
```

Checkpoints and TensorBoard logs are written below
`logs/skrl/lab_pick_slide/`. To continue a run, pass the saved checkpoint to
the underlying SKRL entry point, keep the residual settings unchanged, and set
`--dsrl_curriculum_start_step` to the already completed outer-step count:

```bash
env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
  "$ISAAC_PYTHON" scripts/reinforcement_learning/skrl/train_sac.py \
  --task TacEx-LabPick-Slide-DSRL-Base-v0 \
  --num_envs 1 \
  --headless --enable_cameras \
  --dsrl_policy outputs/lab_pick_flow_matching/best.pt \
  --dsrl_policy_type flow_matching \
  --dsrl_action_mode residual \
  --dsrl_residual_position_scale_m 0.03 0.03 0.01 \
  --dsrl_residual_width_scale_m 0.002 \
  --dsrl_curriculum_steps 100000 \
  --dsrl_curriculum_start_step <completed_outer_steps> \
  --dsrl_curriculum_start_xy_m 0.05 0.05 \
  --dsrl_curriculum_end_xy_m 0.10 0.10 \
  --dsrl_curriculum_start_yaw_deg 30 \
  --dsrl_curriculum_end_yaw_deg 45 \
  --checkpoint logs/skrl/lab_pick_slide/<run>/checkpoints/agent_<step>.pt \
  --timesteps 100000
```

Evaluate a residual checkpoint on the final wide distribution with the same
residual scales:

```bash
env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
  "$ISAAC_PYTHON" scripts/reinforcement_learning/dsrl/eval_lab_pick_dsrl_sac.py \
  --policy outputs/lab_pick_flow_matching/best.pt \
  --checkpoint logs/skrl/lab_pick_slide/<run>/checkpoints/agent_<step>.pt \
  --dsrl_action_mode residual \
  --dsrl_residual_position_scale_m 0.03 0.03 0.01 \
  --dsrl_residual_width_scale_m 0.002 \
  --labware_random_xy 0.10 0.10 \
  --labware_random_yaw 0.7853981634 \
  --num_trials 50 --seed 0 --policy_seed 42 \
  --output logs/lab_pick_eval/sac_residual_wide_seed0_49.json \
  --headless
```

Existing full-noise checkpoints remain supported by omitting
`--dsrl_action_mode residual`; `noise` is the compatibility default.

Use `systemd-run --user` for long unattended jobs. Do not terminate unrelated
Isaac or GPU processes when checking a running job; inspect the unit with:

```bash
systemctl --user status <unit>.service
journalctl --user -u <unit>.service -f
```

The older `scripts/bc_training/train_bc.py` examples are not the entry point
for this branch; use the Flow Matching command above.

### Analyze failed attempts with a VLM

After collection, analyze failed attempts as a separate post-processing step. This keeps simulation collection independent from network/API availability.

```bash
export OPENAI_API_KEY=...
export OPENAI_API_BASE=https://api.openai.com/v1

python scripts/demos/lab_pick/analyze_failed_attempts.py \
  --record_dir /tmp/lab_pick_cafe_records \
  --model gpt-4.1-mini \
  --frame auto \
  --break_force_threshold_n 6.0 \
  --skip_existing
```

For an OpenAI-compatible relay/proxy, set the relay base URL and use `--api_mode chat_completions` if the relay does not support the Responses API:

```bash
export OPENAI_API_KEY=<your-api-key>
export OPENAI_API_BASE=https://api.aiboys.xyz/v1

python scripts/demos/lab_pick/analyze_failed_attempts.py \
  --record_dir /tmp/lab_pick_cafe_records \
  --model gpt-4.1-mini \
  --api_mode chat_completions \
  --frame auto \
  --break_force_threshold_n 6.0 \
  --skip_existing
```

For an offline smoke test without calling the API:

```bash
python scripts/demos/lab_pick/analyze_failed_attempts.py \
  --record_dir /tmp/lab_pick_cafe_records \
  --frame auto \
  --break_force_threshold_n 6.0 \
  --dry_run
```

Each analyzed attempt writes:

```text
failed_attempts/attempt_xxxxxx/
  vlm_failure_analysis.json
  vlm_failure_analysis.txt
```

The full failed-attempt batch also writes:

```text
failed_attempts/failure_summary.csv
failed_attempts/failure_summary.json
```

With `--frame auto`, the VLM receives the first failure-triggering RGB frame and 6D FT vector (`failure_frame_*`). Older records without `failure_frame_*` automatically fall back to `last_frame_*`. The VLM returns a structured report with raw frame context, FT vector, force/torque norms, failure type, contact state, force assessment, risk level, visual reason, force reason, combined reason, evidence list, suggested safe force range, suggested next action, recommended policy change, recommended next test, and confidence. Suggested force ranges are clamped below the configured break threshold.

### Verify the LabPick CAFE pipeline

Static tests:

```bash
/home/tjx/miniforge3/envs/env_isaaclab/bin/python -m pytest source/tacex_tasks/test/test_lab_pick_static.py -q
```

Syntax check:

```bash
/home/tjx/miniforge3/envs/env_isaaclab/bin/python -m py_compile \
  source/tacex_tasks/tacex_tasks/lab_pick/bc_dataset.py \
  source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py \
  source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env_cfg.py \
  scripts/demos/lab_pick/collect_bc_dataset.py
```

### Related scripts

- `scripts/demos/lab_pick/collect_bc_dataset.py`: CAFE-compatible data collection.
- `scripts/demos/lab_pick/pick_labware.py`: scripted LabPick demo.
- `scripts/demos/lab_pick/pick_labware_keyboard.py`: keyboard-controlled LabPick demo.

Scripted Isaac Lab demo:

```bash
env \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
  /home/tjx/miniforge3/envs/env_isaaclab/bin/python \
  scripts/demos/lab_pick/pick_labware.py \
  --labware slide \
  --num_envs 1 \
  --duration 6 \
  --headless
```

Keyboard-controlled Isaac Lab demo:

```bash
env \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
  /home/tjx/miniforge3/envs/env_isaaclab/bin/python \
  scripts/demos/lab_pick/pick_labware_keyboard.py \
  --labware slide \
  --num_envs 1
```


## Contributing
Contributions of any kind are, of course, very welcome.
Be it suggestions, feedback, bug reports or pull requests.

Let's work together to advance tactile sensing in robotics!!!

## Citation
```bibtex
@article{nguyen2024tacexgelsighttactilesimulation,
      title={TacEx: GelSight Tactile Simulation in Isaac Sim -- Combining Soft-Body and Visuotactile Simulators},
      author={Duc Huy Nguyen and Tim Schneider and Guillaume Duret and Alap Kshirsagar and Boris Belousov and Jan Peters},
      year={2024},
      eprint={2411.04776},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.04776},
}
```

## Acknowledgements

TacEx is built upon code from
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/main)
- [Taxim](https://github.com/Robo-Touch/Taxim)
- [FOTS](https://github.com/Rancho-zhao/FOTS)
- [UIPC](https://github.com/spiriMirror/libuipc)
- [ManiSkill-ViTac challenge](https://github.com/chuanyune/ManiSkill-ViTac2025)
