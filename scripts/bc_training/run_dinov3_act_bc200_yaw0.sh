#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${TACEX_ISAAC_PYTHON:-python}"
RECORDS="${TACEX_BC200_RECORDS:-${REPO_ROOT}/datasets/lab_pick_slide_bc200_success_yaw0_seed270828}"
OUTPUT="${TACEX_BC200_OUTPUT:-${REPO_ROOT}/outputs/lab_pick_dinov3_act_bc200_yaw0_visualxy}"
REPORT_DIR="${TACEX_BC200_REPORT_DIR:-${REPO_ROOT}/reports/dinov3_act_bc200_yaw0}"
TARGET_EPISODES="${TACEX_BC200_EPISODES:-200}"
EVAL_TRIALS="${TACEX_BC200_EVAL_TRIALS:-100}"

cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${REPO_ROOT}/source/tacex:${REPO_ROOT}/source/tacex_assets:${REPO_ROOT}/source/tacex_tasks:${REPO_ROOT}/scripts/bc_training${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${RECORDS}" "${OUTPUT}" "${REPORT_DIR}"

successful="$(${PYTHON} - "${RECORDS}" <<'PY'
import sys
from pathlib import Path
import numpy as np

count = 0
for record in Path(sys.argv[1]).glob("record_*"):
    try:
        with np.load(record / "metadata.npz") as metadata:
            count += int(bool(metadata["success"]))
    except FileNotFoundError:
        pass
print(count)
PY
)"

if (( successful < TARGET_EPISODES )); then
  env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
    "${PYTHON}" scripts/demos/lab_pick/collect_bc_dataset.py \
    --labware slide --num_envs 1 --num_demos "$((TARGET_EPISODES - successful))" \
    --max_attempts "$((2 * (TARGET_EPISODES - successful)))" --max_episode_steps 960 \
    --success_only --record_dir "${RECORDS}" --seed 270828 \
    --safe_close_width_m 0.0065 --break_force_threshold_n 4.0 \
    --labware_random_xy 0.10 0.10 --labware_random_yaw_degrees 0.0 \
    --aligned_hz 60 --render_every_n_steps 4 --headless
fi

"${PYTHON}" scripts/bc_training/train_dinov3_act.py \
  --data-root "${RECORDS}" --output-dir "${OUTPUT}" --max-episodes "${TARGET_EPISODES}" \
  --success-only --require-yaw-zero --quat-order xyzw --precompute-features \
  --feature-grid-size 7 --condition-grid-size 4 \
  --state-obs-steps 2 --image-obs-steps 2 --chunk-size 32 \
  --epochs 30 --batch-size 128 --num-workers 8 --lr 0.0003 --amp

env -u LD_PRELOAD -u VGL_ISACTIVE -u VGL_DISPLAY -u DISPLAY \
  "${PYTHON}" scripts/bc_training/eval_dinov3_act_sim.py \
  --checkpoint "${OUTPUT}/best.pt" --num-trials "${EVAL_TRIALS}" --seed 3200 \
  --labware-random-xy 0.10 0.10 --labware-random-yaw-degrees 0.0 \
  --action-repeat 2 --chunk-execute-steps 32 --break-force-threshold-n 4.0 \
  --no-align-action-yaw \
  --output "${REPORT_DIR}/dinov3_act_bc200_eval.json" --no-record-video --headless

echo "DINOv3 BC200 training and evaluation completed."
echo "Evaluation: ${REPORT_DIR}/dinov3_act_bc200_eval.json"
