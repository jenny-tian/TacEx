#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${TACEX_FLOW_PYTHON:-/home/tjx/miniforge3/envs/env_isaaclab/bin/python}"
RECORDS="${TACEX_FLOW_RECORDS:-${REPO_ROOT}/datasets/lab_pick_slide_dsrl_smoke}"
DATASET="${TACEX_FLOW_DATASET:-${REPO_ROOT}/datasets/lab_pick_slide_flow_matching_causal_third_200.hdf5}"
OUTPUT_DIR="${TACEX_FLOW_OUTPUT:-${REPO_ROOT}/outputs/lab_pick_flow_matching_causal_third_a32_s2_c2_200}"
COLLECT_UNIT="${TACEX_FLOW_COLLECT_UNIT:-tacex-labpick-collect150.service}"
TARGET_EPISODES="${TACEX_FLOW_TARGET_EPISODES:-200}"

record_count() {
    find "${RECORDS}" -maxdepth 1 -type d -name 'record_*' | wc -l
}

while systemctl --user is-active --quiet "${COLLECT_UNIT}"; do
    current="$(record_count)"
    echo "[WAIT] records=${current}/${TARGET_EPISODES} collector=${COLLECT_UNIT}"
    sleep 60
done

current="$(record_count)"
if (( current < TARGET_EPISODES )); then
    echo "[ERROR] collector stopped with only ${current}/${TARGET_EPISODES} records" >&2
    exit 1
fi

if [[ -f "${OUTPUT_DIR}/best.pt" ]]; then
    echo "[PASS] Flow Matching checkpoint already exists: ${OUTPUT_DIR}/best.pt"
    exit 0
fi

cd "${REPO_ROOT}"
"${PYTHON}" bc_policy/sim_robot/scripts/convert_records_to_hdf5.py \
    --input "${RECORDS}" \
    --output "${DATASET}" \
    --success-only \
    --max-episodes "${TARGET_EPISODES}" \
    --action-alignment auto \
    --include-third-camera \
    --overwrite

"${PYTHON}" bc_policy/sim_robot/scripts/train_flow_matching.py \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --success-only \
    --action-key high \
    --image-key robot0_image_third \
    --n-state-obs-steps 2 \
    --n-image-obs-steps 2 \
    --n-action-steps 32 \
    --epochs 50 \
    --batch-size 32 \
    --num-workers 4 \
    --lr 0.0001 \
    --weight-decay 0.000001 \
    --warmup-steps 500 \
    --val-ratio 0.05 \
    --seed 42 \
    --normalizer-mode limits \
    --image-feature-dim 512 \
    --obs-feature-dim 512 \
    --transformer-layers 6 \
    --transformer-heads 8 \
    --transformer-embedding-dim 512 \
    --transformer-cond-layers 2 \
    --dropout 0.1 \
    --num-inference-steps 100 \
    --ode-solver euler \
    --ema-decay 0.999 \
    --amp \
    --save-every 10

echo "[PASS] Flow Matching training complete: ${OUTPUT_DIR}/best.pt"
