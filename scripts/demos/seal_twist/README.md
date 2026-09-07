# SealTwist torque-window demo

This Isaac Lab task uses a Franka arm with two GelSight Mini fingertips to
tighten a bottle cap.  The cap must stop inside a `0.35-0.50 N m` terminal
torque window.  The scene uses physical pad/cap contacts plus an explicit
equivalent screw-and-elastomer reaction model.  This avoids unstable detailed
thread meshes while retaining the manipulation, tactile and torque-control
signals needed for learning.

## Collect one trajectory and video

From the repository root:

```bash
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
/home/limx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/demos/seal_twist/collect_dataset.py \
  --headless --device cpu --save_rgb --save_tactile --record_video \
  --output datasets/seal_twist/record_000000 \
  --video_path artifacts/seal_twist/seal_twist_expert.mp4
```

The record contains `metadata.json` and a compressed `trajectory.npz` with
robot pose/action, combined wrench, individual fingertip forces, cap pose,
angle/rate, seal torque/compression, phase, scene/wrist RGB and both GelSight
RGB streams. It also stores two `32 x 32` tactile force maps whose pixel sums
equal the measured left/right contact-force magnitudes. The metadata reports
whether the optional optical GelSight RGB stream changed during the episode.
The video overlay exposes the torque window and measured touch forces so
failures are visible without post-processing.

Omitting the three persistence flags runs a faster physical-contact-only check.
`--save_tactile` enables the more expensive optical GelSight RGB renderer;
without it, the trajectory still includes the two physical PhysX fingertip
force streams. `--record_stride 4` corresponds to 30 Hz with the default 120 Hz
simulation.

The command above uses CPU PhysX because this workstation has IOMMU enabled;
RTX still renders the scene and GelSight images on the GPU. On systems with a
working GPU PhysX setup, omit `--device cpu` for the usual CUDA path.

## Record the publication-style scene

```bash
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
/home/limx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/demos/seal_twist/collect_dataset.py \
  --headless --device cpu --record_video \
  --render_width 1280 --render_height 720 \
  --output datasets/seal_twist_paper/record_000000 \
  --video_path artifacts/seal_twist_paper/seal_twist_paper.mp4
```

This variant uses the same task physics with a navy laboratory backdrop,
accent lighting, shelving and restrained laboratory props for paper figures
and supplementary video.

## Collect data, train BC and evaluate failure modes

The complete bottle-cap BC experiment writes its outputs under
`exp_report/bottle`:

```bash
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
/home/limx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/demos/seal_twist/collect_bc_dataset.py \
  --headless --device cpu --num_episodes 48 --num_envs 8

python scripts/demos/seal_twist/validate_bc_dataset.py
python scripts/demos/seal_twist/train_state_bc.py \
  --epochs 100 --batch_size 1024 --device cuda

PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
/home/limx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/demos/seal_twist/eval_state_bc.py \
  --headless --device cpu --episodes 100 --num_envs 20 --seed 80000 \
  --report exp_report/bottle/evaluation/base/seed_80000.json
```

Repeat the last command with the frozen seeds listed in
`exp_report/bottle/EXPERIMENT_PROTOCOL.md`, then run
`python scripts/demos/seal_twist/summarize_bc_results.py`. The evaluator uses
per-episode minimum/maximum fingertip-force thresholds and physical cap/tool
alignment to produce mutually exclusive `under_force`, `over_force`, and
`alignment_failure` outcomes.

## Train and evaluate 6D-wrench Flow Matching BC

The Flow policy follows the conditional Flow Matching construction used by the
LabPick BC stack. Its 28-dimensional observation contains the complete
`[Fx, Fy, Fz, Tx, Ty, Tz]` wrench at indices `[10:16]`. A four-block,
512-wide residual MLP predicts the vector field for a four-action chunk.

The frozen checkpoint was trained as follows:

```bash
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
/home/limx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/demos/seal_twist/train_flow_bc.py \
  --device cuda --epochs 400 --horizon 4 --hidden-dim 512 --depth 4 \
  --ema-decay 0.99 --endpoint-loss-weight 2.0 \
  --checkpoint exp_report/bottle/checkpoints/flow_bc_6d_force_v5.pt \
  --report exp_report/bottle/training/flow_bc_6d_force_v5.json
```

For each frozen holdout seed (`80000`, `90000`, and `100000`), run:

```bash
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
/home/limx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/demos/seal_twist/eval_flow_bc.py \
  --headless --device cpu --policy-device cuda \
  --checkpoint exp_report/bottle/checkpoints/flow_bc_6d_force_v5.pt \
  --episodes 100 --num-envs 20 --seed SEED \
  --noise-mode zero --action-repeat 4 --chunk-execute-steps 4 \
  --force-feedback-target-n 7.7 \
  --force-feedback-gain-m-per-n 0.0003 \
  --force-feedback-max-step-m 0.0005 \
  --force-feedback-start-step 350 \
  --gripper-width-limits-m 0.0190 0.0400 \
  --report exp_report/bottle/evaluation/flow_6d_force/seed_SEED.json
```

Aggregate and audit the 300 episode records with:

```bash
python scripts/demos/seal_twist/summarize_flow_results.py
```

The three success rates are 57%, 44%, and 55% (156/300 overall, 52.0%;
Wilson 95% CI 46.4%-57.6%). The fixed 7.7 N tactile projection above uses
only observed contact force and does not read the randomized success
thresholds. See `exp_report/bottle/FLOW_6D_FORCE_RESULTS.md` for the audited
result and failure breakdown.
