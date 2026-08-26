#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

PYTHON=/home/tjx/miniforge3/envs/env_isaaclab/bin/python
BASE=outputs/lab_pick_flow_bc100_scratch_rawclose_safe70_overforce24_pos6
LOG_DIR=logs/lab_pick_bc100_scratch_best_4n_100_predicted_rotation

/usr/bin/env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
  "${PYTHON}" scripts/demos/lab_pick/eval_flow_matching_policy.py \
  --checkpoint "${BASE}/best.pt" --num_trials 100 --seed 3200 \
  --num_inference_steps 20 --chunk_execute_steps 32 --action_repeat 2 \
  --phase_horizon_steps 383 --visual_xy_lock_phase 0.30 \
  --break_force_threshold_n 4.0 --overforce_trial_fraction 0.0 \
  --position_failure_trial_fraction 0.0 --labware_random_xy 0.10 0.10 \
  --labware_random_yaw 0.7853981633974483 \
  --video_dir "${LOG_DIR}/videos" --video_every_n_steps 0 \
  --print_state_interval 0 --output "${LOG_DIR}/results.json" --headless
