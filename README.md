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
python scripts/demos/lab_pick/collect_bc_dataset.py \
  --labware slide \
  --num_envs 1 \
  --num_demos 2 \
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
