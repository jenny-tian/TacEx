#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

PYTHON="${TACEX_ISAAC_PYTHON:-/home/tjx/miniforge3/envs/env_isaaclab/bin/python}"
SOURCE_RECORDS=datasets/lab_pick_slide_bc100_diverse_safe70_overforce24_pos6_xypm10_yawpm45
RECORDS=datasets/lab_pick_slide_bc200_hardyaw_safe140_overforce48_pos12_xypm10_yaw15to45
DATASET=/dev/shm/tacex_lab_pick_bc200_hardyaw_rotation.hdf5
OUTPUT=outputs/lab_pick_flow_bc200_hardyaw_rotation_scratch

mkdir -p "${RECORDS}" "${OUTPUT}"

# Reuse the first 100 records without duplicating their approximately 11 GB of image data.
for source in "${SOURCE_RECORDS}"/record_*; do
  target="${RECORDS}/$(basename "${source}")"
  if [[ ! -e "${target}" ]]; then
    ln -s "$(realpath "${source}")" "${target}"
  fi
done

current=$(find -L "${RECORDS}" -maxdepth 1 -type d -name 'record_*' | wc -l)
if (( current < 200 )); then
  /usr/bin/env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
    "${PYTHON}" scripts/demos/lab_pick/collect_bc_dataset.py \
    --labware slide --num_envs 1 --num_demos "$((200 - current))" --max_attempts 600 \
    --record_dir "${RECORDS}" --seed 20000 --safe_demo_fraction 0.70 \
    --position_failure_demo_fraction 0.06 --require_expected_mode_outcome \
    --safe_close_width_m 0.0065 --overforce_close_width_m 0.0015 \
    --position_failure_offset_m 0.03 --break_force_threshold_n 4.8 \
    --labware_random_xy 0.10 0.10 --labware_random_yaw_degrees 45.0 \
    --labware_random_yaw_min_abs_degrees 15.0 \
    --max_episode_steps 960 --aligned_hz 60 --headless
fi

"${PYTHON}" bc_policy/sim_robot/scripts/convert_records_to_hdf5.py \
  --input "${RECORDS}" --output "${DATASET}" --max-episodes 200 \
  --action-alignment auto --include-third-camera --overwrite

/usr/bin/env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
  OMNI_KIT_ACCEPT_EULA=YES \
  "${PYTHON}" bc_policy/sim_robot/scripts/train_flow_matching.py \
  --dataset "${DATASET}" --output-dir "${OUTPUT}" \
  --action-key high --image-keys robot0_image,robot0_image_third \
  --n-state-obs-steps 2 --n-image-obs-steps 2 --n-action-steps 32 \
  --epochs 40 --batch-size 32 --num-workers 4 \
  --lr 0.0001 --weight-decay 0.000001 --warmup-steps 500 \
  --val-ratio 0.10 --seed 42 --normalizer-mode limits \
  --image-feature-dim 512 --image-normalization none --obs-feature-dim 512 \
  --transformer-layers 6 --transformer-heads 8 --transformer-embedding-dim 512 \
  --transformer-cond-layers 2 --dropout 0.1 --num-inference-steps 100 \
  --ode-solver euler --ema-decay 0.999 --include-phase \
  --safe-sample-weight 1.0 --overforce-sample-weight 1.0 \
  --position-failure-sample-weight 1.0 \
  --safe-close-width-m 0.0 --safe-close-phase-weight 1.0 \
  --overforce-close-width-m 0.0 --visual-xy-loss-weight 1.0 \
  --rotation-loss-weight 1.5 --visual-rotation-loss-weight 1.0 \
  --amp --save-every 5
