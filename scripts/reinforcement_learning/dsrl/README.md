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
- break termination: fingertip force above `6 N` after contact;
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

For DDIM Diffusion, the SAC actor can still output flattened initial noise with dimension `horizon * action_dim` (default `16 * 10 = 160`). The frozen Diffusion Policy decodes it into CAFE actions. Rewards are accumulated over that chunk.

For the 32-step Flow Matching policy, the recommended mode is instead a 4-D normalized residual `[dx, dy, dz, dwidth]`. This avoids a 320-D SAC action space while keeping the Flow Matching visual encoder and trajectory prior frozen. The demonstrations are aligned at 60 Hz while LabPick physics runs at 120 Hz, so every decoded action is held for two physics steps (`--dsrl_action_repeat 2`).
The residual penalty is measured against the zero residual prior, which is the frozen BC behavior. The recommended run keeps it at 5.0 during the 2,000-step warm-up and linearly decays it to 1.0 over the next 50,000 steps; it is disabled by default for backward compatibility.

Legacy DDIM noise training:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
/home/tjx/miniforge3/envs/env_isaaclab/bin/python \
  scripts/reinforcement_learning/dsrl/train_lab_pick_dsrl_sac.py \
  --dsrl_policy outputs/lab_pick_diffusion_ddim/checkpoints/last/pretrained_model \
  --timesteps 200000 --seed 42
```

Recommended Flow Matching residual training with a reset curriculum:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
/home/tjx/miniforge3/envs/env_isaaclab/bin/python \
  scripts/reinforcement_learning/dsrl/train_lab_pick_dsrl_sac.py \
  --dsrl_policy outputs/lab_pick_flow_matching/best.pt \
  --dsrl_policy_type flow_matching \
  --dsrl_action_mode residual \
  --dsrl_residual_position_scale_m 0.03 0.03 0.01 \
  --dsrl_residual_width_scale_m 0.002 \
  --dsrl_residual_penalty_scale 5.0 \
  --dsrl_residual_penalty_end_scale 1.0 \
  --dsrl_residual_penalty_decay_start_step 2000 \
  --dsrl_residual_penalty_decay_steps 50000 \
  --dsrl_curriculum_steps 100000 \
  --dsrl_curriculum_start_xy_m 0.05 0.05 \
  --dsrl_curriculum_end_xy_m 0.10 0.10 \
  --dsrl_curriculum_start_yaw_deg 30 \
  --dsrl_curriculum_end_yaw_deg 45 \
  --timesteps 200000 --seed 42
```

Omit `--dsrl_action_mode residual` for existing full-noise checkpoints. Residual and noise checkpoints have different actor output dimensions and must be evaluated with the mode used for training.

Before online fine-tuning, evaluate the frozen BC checkpoint with the same randomized task and 60 Hz control timing:

    OMNI_KIT_ACCEPT_EULA=YES \
    /home/tjx/miniforge3/envs/env_isaaclab/bin/python \
      scripts/reinforcement_learning/dsrl/eval_lab_pick_diffusion_bc.py \
      --policy outputs/lab_pick_diffusion_ddim/checkpoints/last/pretrained_model \
      --num_trials 20 --seed 0 --action_repeat 2 --headless

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
