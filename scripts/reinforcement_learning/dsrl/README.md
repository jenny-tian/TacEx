# TacEx LabPick SAC / DSRL training

This pipeline keeps simulation and policy training isolated:

- `env_isaaclab`: Isaac Sim, LabPick, SKRL SAC, online DSRL rollouts.
- `tacex_lerobot`: dataset conversion, Diffusion Policy pretraining, model-only tests.
- `.cache/lerobot_inference`: pure Python LeRobot inference dependencies loaded by `env_isaaclab` without changing its core packages.

The workstation's TurboVNC session exports `LD_PRELOAD=libdlfaker.so:libvglfaker.so`. This breaks PyTorch/cuDNN dynamic loading. Camera-enabled headless Isaac commands must also unset `DISPLAY` so rendering uses direct Vulkan rather than the remote desktop's GLX context. The DSRL launcher performs all four cleanups automatically.

The commands below use `/usr/bin/env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY` for direct Vulkan headless execution.


## Phase 1: state SAC baseline

Task: `TacEx-LabPick-Slide-SAC-v0`

- observation: normalized 23-D privileged state;
- action: `[dx, dy, dz, gripper_width]` in a normalized 4-D space;
- success: slide lift at least `0.20 m`;
- break termination: fingertip force above `4 N` after contact;
- shaping: reach progress, lift progress, first/bilateral contact, success bonus, force and failure penalties.

After explicitly accepting the NVIDIA Omniverse EULA, smoke-test 200 steps:

```bash
/usr/bin/env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
OMNI_KIT_ACCEPT_EULA=YES \
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
/home/tjx/miniforge3/envs/env_isaaclab/bin/python \
  scripts/reinforcement_learning/skrl/train_sac.py \
  --task TacEx-LabPick-Slide-SAC-v0 \
  --num_envs 1 --timesteps 200 --headless --enable_cameras
```

Then run the full baseline, initially with one environment and later test 2-4 if GPU memory permits:

```bash
/usr/bin/env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
OMNI_KIT_ACCEPT_EULA=YES \
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
/home/tjx/miniforge3/envs/env_isaaclab/bin/python \
  scripts/reinforcement_learning/skrl/train_sac.py \
  --task TacEx-LabPick-Slide-SAC-v0 \
  --num_envs 1 --timesteps 1000000 --headless --enable_cameras
```

## Phase 2: collect successful demonstrations

Collect at least 50 successful new-format records. For Diffusion training, 200-500 successful episodes are preferred.

```bash
/usr/bin/env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
OMNI_KIT_ACCEPT_EULA=YES \
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
/home/tjx/miniforge3/envs/env_isaaclab/bin/python \
  scripts/demos/lab_pick/collect_bc_dataset.py \
  --labware slide --num_demos 200 --max_attempts 500 \
  --success_only --headless \
  --record_dir datasets/lab_pick_slide_dsrl_records
```

Validate them:

```bash
/home/tjx/miniforge3/envs/tacex_lerobot/bin/python \
  scripts/reinforcement_learning/dsrl/check_dsrl_ready.py \
  --records datasets/lab_pick_slide_dsrl_records --min-success 50
```

## Phase 3: convert to LeRobot v3

```bash
/home/tjx/miniforge3/envs/tacex_lerobot/bin/python \
  scripts/bc_training/create_lerobot_dataset.py \
  --input datasets/lab_pick_slide_dsrl_records \
  --output-root datasets/lab_pick_slide_lerobot \
  --repo-id local/tacex_lab_pick_slide \
  --success-only --overwrite
```

## Phase 4: pretrain a DDIM Diffusion Policy

DSRL requires DDIM so identical initial noise deterministically maps to the same action chunk.

```bash
conda activate tacex_lerobot
python scripts/bc_training/train_lab_pick_diffusion.py \
  --dataset-root datasets/lab_pick_slide_lerobot \
  --output-dir outputs/lab_pick_diffusion_ddim \
  --steps 100000 --batch-size 32 --video-backend pyav
```

Check the saved `pretrained_model` directory:

```bash
python scripts/reinforcement_learning/dsrl/check_dsrl_ready.py \
  --policy outputs/lab_pick_diffusion_ddim/checkpoints/last/pretrained_model
```

## Phase 5: DSRL-SAC online fine-tuning

The SAC actor outputs flattened DDIM initial noise with dimension `horizon * action_dim` (default `16 * 10 = 160`). The frozen Diffusion Policy decodes it into up to eight CAFE actions. Rewards are accumulated over that chunk.
The demonstrations are aligned at 60 Hz while LabPick physics runs at 120 Hz, so every decoded action is held for two physics steps (--dsrl_action_repeat 2).

```bash
OMNI_KIT_ACCEPT_EULA=YES \
/home/tjx/miniforge3/envs/env_isaaclab/bin/python \
  scripts/reinforcement_learning/dsrl/train_lab_pick_dsrl_sac.py \
  --dsrl_policy outputs/lab_pick_diffusion_ddim/checkpoints/last/pretrained_model \
  --timesteps 200000 --seed 42
```

Before online fine-tuning, evaluate the frozen BC checkpoint with the same randomized task and 60 Hz control timing:

    OMNI_KIT_ACCEPT_EULA=YES \
    /home/tjx/miniforge3/envs/env_isaaclab/bin/python \
      scripts/reinforcement_learning/dsrl/eval_lab_pick_diffusion_bc.py \
      --policy outputs/lab_pick_diffusion_ddim/checkpoints/last/pretrained_model \
      --num_trials 20 --seed 0 --action_repeat 2 --headless

### Flow Matching BC with a learned gate

For the sim_robot Flow Matching checkpoint, the gated variant adds one scalar
to the SAC action. The scalar controls a soft blend between native frozen-BC
Gaussian noise and SAC-proposed DSRL noise. The gate is capped so SAC remains
a residual correction instead of replacing the frozen BC policy.

```bash
TACEX_DSRL_TIMESTEPS=200000 \
  scripts/reinforcement_learning/dsrl/run_lab_pick_flow_gated_sac.sh
```

The default checkpoint is
`outputs/lab_pick_flow_bc100_scratch_rawclose_safe70_overforce24_pos6/best.pt`.
It was trained from scratch on 100 mixed demonstrations: 70 successful, 24
overforce-break, and 6 position-failure episodes. Sample weights were all 1.0,
and the demonstration category was not exposed to the policy
(`include_demo_mode=False`). A normal unconditioned 100-trial evaluation over
+/-10 cm and +/-45 degrees at the 4 N break threshold produced 65 successes,
22 overforce breakages, and 13 position/timeout failures. Thus 62.9% of the 35
failures were breakages. The evaluator did not request a failure mode. See the
checkpoint's `validation_summary.json` for the complete aggregate result.
Training starts
with a gate of `0.1`, temperature `0.5`, penalty `0.1`, and maximum gate `0.3`;
`DSRL/gate_mean` and `DSRL/gate_penalty` are written to the normal SKRL/IsaacLab
logs. Set `TACEX_DSRL_GATE_INIT`, `TACEX_DSRL_GATE_TEMPERATURE`,
`TACEX_DSRL_GATE_PENALTY`, or `TACEX_DSRL_GATE_MAX` to override the defaults.

The SAC actor and critics receive the frozen BC encoder's exact `global_cond`:
the same robot-state history, wrist-camera history, and third-person-camera
history used by Flow Matching. Simulator object position, object orientation,
and demonstration category are not included. The environment executes the
BC-predicted rot6d orientation instead of replacing it with simulator-aligned
yaw.

## Validation gates

1. SAC smoke test completes and writes a checkpoint without NaN/Inf.
2. Dataset checker finds at least 50 successful episodes with `aligned/rgb.npy`, `rgb_third.npy`, state and action arrays.
3. Diffusion checkpoint uses `noise_scheduler_type=DDIM` and includes policy pre/post processors.
4. `smoke_diffusion_noise.py` passes on CUDA and reports deterministic same-noise behavior.
5. DSRL outer-step rewards remain finite; break rate trends down rather than up.
6. Final evaluation uses at least 50 randomized seeds and reports success rate, break rate, peak force, episode length, and lift height.

Recommended comparison budgets:

- scripted policy: 50 episodes;
- state SAC baseline: 3 seeds, 50 evaluation episodes each;
- frozen Diffusion BC: 3 seeds, 50 evaluation episodes each;
- DSRL-SAC: 3 seeds, 50 evaluation episodes each.

For a resumable unattended run using the active `tacex-labpick-collect49.service`:

```bash
systemd-run --user --unit=tacex-labpick-dsrl-pipeline \
  --working-directory=/home/tjx/TacEx \
  /home/tjx/TacEx/scripts/reinforcement_learning/dsrl/run_lab_pick_pipeline.sh
```
