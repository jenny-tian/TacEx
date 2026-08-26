#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec /usr/bin/env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
  OMNI_KIT_ACCEPT_EULA=YES \
  /home/tjx/miniforge3/envs/env_isaaclab/bin/python \
  bc_policy/sim_robot/scripts/train_flow_matching.py \
  --dataset /dev/shm/tacex_lab_pick_bc50_strong48n_pos8_50.hdf5 \
  --output-dir outputs/lab_pick_flow_bc50_unconditioned_success3_close55mm_overforce15_pos01_lr3e6 \
  --action-key high --image-keys robot0_image,robot0_image_third \
  --n-state-obs-steps 2 --n-image-obs-steps 2 --n-action-steps 32 \
  --epochs 3 --batch-size 32 --num-workers 4 \
  --lr 0.000003 --weight-decay 0.000001 --warmup-steps 50 \
  --val-ratio 0.10 --seed 42 --normalizer-mode limits \
  --image-feature-dim 512 --image-normalization none --obs-feature-dim 512 \
  --transformer-layers 6 --transformer-heads 8 --transformer-embedding-dim 512 \
  --transformer-cond-layers 2 --dropout 0.1 --num-inference-steps 100 \
  --ode-solver euler --ema-decay 0.999 --include-phase \
  --safe-sample-weight 3.0 --overforce-sample-weight 1.5 \
  --position-failure-sample-weight 0.1 \
  --safe-close-width-m 0.0055 --safe-close-phase 0.40 --safe-close-phase-weight 2.0 \
  --overforce-close-width-m 0.0015 \
  --visual-xy-loss-weight 1.0 \
  --init-checkpoint outputs/lab_pick_flow_bc50_strong48n_pos8_50_unconditioned_balanced/epoch_0001.pt \
  --amp --save-every 1
