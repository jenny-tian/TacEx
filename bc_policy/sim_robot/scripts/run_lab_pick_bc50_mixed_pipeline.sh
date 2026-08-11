#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${TACEX_BC50_PYTHON:-/home/tjx/miniforge3/envs/env_isaaclab/bin/python}"
RECORDS="${TACEX_BC50_RECORDS:-${REPO_ROOT}/datasets/lab_pick_slide_bc50_three_outcome_xypm10_yawpm45_200}"
DATASET="${TACEX_BC50_DATASET:-${REPO_ROOT}/datasets/lab_pick_slide_bc50_three_outcome_xypm10_yawpm45_200.hdf5}"
OUTPUT_DIR="${TACEX_BC50_OUTPUT:-${REPO_ROOT}/outputs/lab_pick_flow_bc50_three_outcome_xypm10_yawpm45}"
EVAL_DIR="${TACEX_BC50_EVAL_DIR:-${REPO_ROOT}/logs/lab_pick_flow_bc50_three_outcome_xypm10_yawpm45}"
TARGET_EPISODES="${TACEX_BC50_EPISODES:-200}"
EVAL_TRIALS="${TACEX_BC50_EVAL_TRIALS:-50}"
BREAK_FORCE_N="${TACEX_BC50_BREAK_FORCE_N:-3.8}"
SAFE_FRACTION="${TACEX_BC50_SAFE_FRACTION:-0.5}"
POSITION_FAILURE_FRACTION="${TACEX_BC50_POSITION_FAILURE_FRACTION:-0.25}"
SAFE_MODE_PROBABILITY="${TACEX_BC50_SAFE_MODE_PROBABILITY:-0.50}"
OVERFORCE_MODE_PROBABILITY="${TACEX_BC50_OVERFORCE_MODE_PROBABILITY:-0.25}"
POSITION_FAILURE_MODE_PROBABILITY="${TACEX_BC50_POSITION_FAILURE_MODE_PROBABILITY:-0.25}"
SAFE_WIDTH_M="${TACEX_BC50_SAFE_WIDTH_M:-0.0065}"
OVERFORCE_WIDTH_M="${TACEX_BC50_OVERFORCE_WIDTH_M:-0.0015}"
POSITION_FAILURE_OFFSET_M="${TACEX_BC50_POSITION_FAILURE_OFFSET_M:-0.03}"
INIT_CHECKPOINT="${TACEX_BC50_INIT_CHECKPOINT:-${REPO_ROOT}/outputs/lab_pick_flow_matching_causal_dual_phase_visual_xy_e5/best.pt}"

record_count() {
    if [[ ! -d "${RECORDS}" ]]; then
        echo 0
        return
    fi
    find "${RECORDS}" -maxdepth 1 -type d -name 'record_*' | wc -l
}

cd "${REPO_ROOT}"
mkdir -p "${RECORDS}" "${OUTPUT_DIR}" "${EVAL_DIR}"

current="$(record_count)"
if (( current < TARGET_EPISODES )); then
    remaining=$((TARGET_EPISODES - current))
    PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks     "${PYTHON}" scripts/demos/lab_pick/collect_bc_dataset.py         --labware slide         --num_envs 1         --num_demos "${remaining}"         --record_dir "${RECORDS}"         --safe_demo_fraction "${SAFE_FRACTION}"         --position_failure_demo_fraction "${POSITION_FAILURE_FRACTION}"         --safe_close_width_m "${SAFE_WIDTH_M}"         --overforce_close_width_m "${OVERFORCE_WIDTH_M}"         --position_failure_offset_m "${POSITION_FAILURE_OFFSET_M}"         --break_force_threshold_n "${BREAK_FORCE_N}"     --labware_random_xy 0.10 0.10         --labware_random_yaw_degrees 45.0         --max_episode_steps 960         --aligned_hz 60         --headless
fi

current="$(record_count)"
if (( current < TARGET_EPISODES )); then
    echo "[ERROR] collection stopped with only ${current}/${TARGET_EPISODES} records" >&2
    exit 1
fi

"${PYTHON}" bc_policy/sim_robot/scripts/convert_records_to_hdf5.py     --input "${RECORDS}"     --output "${DATASET}"     --max-episodes "${TARGET_EPISODES}"     --action-alignment auto     --include-third-camera     --overwrite

if [[ ! -f "${OUTPUT_DIR}/best.pt" ]]; then
    "${PYTHON}" bc_policy/sim_robot/scripts/train_flow_matching.py         --dataset "${DATASET}"         --output-dir "${OUTPUT_DIR}"         --action-key high         --image-keys robot0_image,robot0_image_third         --n-state-obs-steps 2         --n-image-obs-steps 2         --n-action-steps 32         --epochs 1         --batch-size 32         --num-workers 4         --lr 0.00002         --weight-decay 0.000001         --warmup-steps 100         --val-ratio 0.05         --seed 42         --normalizer-mode limits         --image-feature-dim 512         --image-normalization none         --obs-feature-dim 512         --transformer-layers 6         --transformer-heads 8         --transformer-embedding-dim 512         --transformer-cond-layers 2         --dropout 0.1         --num-inference-steps 100         --ode-solver euler         --ema-decay 0.999         --include-phase         --include-demo-mode         --safe-demo-mode-probability "${SAFE_MODE_PROBABILITY}"         --overforce-demo-mode-probability "${OVERFORCE_MODE_PROBABILITY}"         --position-failure-demo-mode-probability "${POSITION_FAILURE_MODE_PROBABILITY}"         --safe-sample-weight 1.0         --overforce-sample-weight 1.0         --position-failure-sample-weight 1.0         --overforce-close-width-m "${OVERFORCE_WIDTH_M}"         --visual-xy-loss-weight 1.0         --init-checkpoint "${INIT_CHECKPOINT}"         --preserve-init-normalization         --amp         --save-every 1
fi

PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks "${PYTHON}" scripts/demos/lab_pick/eval_flow_matching_policy.py     --checkpoint "${OUTPUT_DIR}/best.pt"     --num_trials "${EVAL_TRIALS}"     --seed 3200     --num_inference_steps 20     --chunk_execute_steps 32     --action_repeat 2     --phase_horizon_steps 383     --visual_xy_lock_phase 0.30     --break_force_threshold_n "${BREAK_FORCE_N}"     --overforce_trial_fraction 0.0     --position_failure_trial_fraction 0.0     --labware_random_xy 0.10 0.10     --labware_random_yaw 0.7853981633974483     --video_dir "${EVAL_DIR}/videos"     --video_every_n_steps 0     --print_state_interval 0     --output "${EVAL_DIR}/results.json"     --headless

jq -e '
    (.success_rate >= 0.40 and .success_rate <= 0.60)
    and (.broken > 0 and .position_failures > 0)
    and (.break_failure_fraction >= 0.30 and .break_failure_fraction <= 0.70)
' "${EVAL_DIR}/results.json" >/dev/null

echo "[PASS] mixed BC target met: ${EVAL_DIR}/results.json"
