#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

PYTHON=/home/tjx/miniforge3/envs/env_isaaclab/bin/python
RECORDS=datasets/lab_pick_slide_bc50_diverse_safe35_overforce12_pos3_xypm10_yawpm45
DATASET=/dev/shm/tacex_lab_pick_bc50_diverse_safe35_overforce12_pos3.hdf5
OUTPUT=outputs/lab_pick_flow_bc50_diverse_rawclose_safe35_overforce12_pos3

mkdir -p "${RECORDS}" "${OUTPUT}"

PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
  "${PYTHON}" scripts/demos/lab_pick/collect_bc_dataset.py \
  --labware slide --num_envs 1 --num_demos 50 --max_attempts 300 \
  --record_dir "${RECORDS}" --safe_demo_fraction 0.70 \
  --position_failure_demo_fraction 0.06 --require_expected_mode_outcome \
  --safe_close_width_m 0.0065 --overforce_close_width_m 0.0015 \
  --position_failure_offset_m 0.03 --break_force_threshold_n 4.8 \
  --labware_random_xy 0.10 0.10 --labware_random_yaw_degrees 45.0 \
  --max_episode_steps 960 --aligned_hz 60 --headless

"${PYTHON}" bc_policy/sim_robot/scripts/convert_records_to_hdf5.py \
  --input "${RECORDS}" --output "${DATASET}" --max-episodes 50 \
  --action-alignment auto --include-third-camera --overwrite

/usr/bin/env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
  OMNI_KIT_ACCEPT_EULA=YES \
  "${PYTHON}" bc_policy/sim_robot/scripts/train_flow_matching.py \
  --dataset "${DATASET}" --output-dir "${OUTPUT}" \
  --action-key high --image-keys robot0_image,robot0_image_third \
  --n-state-obs-steps 2 --n-image-obs-steps 2 --n-action-steps 32 \
  --epochs 3 --batch-size 32 --num-workers 4 \
  --lr 0.000003 --weight-decay 0.000001 --warmup-steps 50 \
  --val-ratio 0.10 --seed 42 --normalizer-mode limits \
  --image-feature-dim 512 --image-normalization none --obs-feature-dim 512 \
  --transformer-layers 6 --transformer-heads 8 --transformer-embedding-dim 512 \
  --transformer-cond-layers 2 --dropout 0.1 --num-inference-steps 100 \
  --ode-solver euler --ema-decay 0.999 --include-phase \
  --safe-sample-weight 1.25 --overforce-sample-weight 1.0 \
  --position-failure-sample-weight 0.25 \
  --safe-close-width-m 0.0 --safe-close-phase-weight 1.0 \
  --overforce-close-width-m 0.0 --visual-xy-loss-weight 1.0 \
  --init-checkpoint outputs/lab_pick_flow_bc50_strong48n_pos8_50_unconditioned_balanced/epoch_0001.pt \
  --amp --save-every 1
