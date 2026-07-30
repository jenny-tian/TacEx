#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ISAAC_PYTHON="${TACEX_ISAAC_PYTHON:-/home/tjx/miniforge3/envs/env_isaaclab/bin/python}"
LEROBOT_PYTHON="${TACEX_LEROBOT_PYTHON:-/home/tjx/miniforge3/envs/tacex_lerobot/bin/python}"
RECORDS_DIR="${TACEX_RECORDS_DIR:-${REPO_ROOT}/datasets/lab_pick_slide_dsrl_smoke}"
DATASET_ROOT="${TACEX_DATASET_ROOT:-${REPO_ROOT}/datasets/lab_pick_slide_lerobot}"
DIFFUSION_OUTPUT="${TACEX_DIFFUSION_OUTPUT:-${REPO_ROOT}/outputs/lab_pick_diffusion_ddim}"
COLLECT_SERVICE="${TACEX_COLLECT_SERVICE:-tacex-labpick-collect49.service}"
DIFFUSION_STEPS="${TACEX_DIFFUSION_STEPS:-100000}"
DSRL_TIMESTEPS="${TACEX_DSRL_TIMESTEPS:-200000}"

cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
export PATH="$(dirname "${LEROBOT_PYTHON}"):/usr/bin:/bin:${PATH}"

echo "[PIPELINE] waiting for ${COLLECT_SERVICE} to finish"
while systemctl --user is-active --quiet "${COLLECT_SERVICE}"; do
  count=$(find "${RECORDS_DIR}" -maxdepth 1 -type d -name 'record_*' | wc -l)
  echo "[PIPELINE] records=${count}; collection still active"
  sleep 60
done

"${LEROBOT_PYTHON}" scripts/reinforcement_learning/dsrl/check_dsrl_ready.py \
  --records "${RECORDS_DIR}" --min-success 50

"${LEROBOT_PYTHON}" scripts/bc_training/create_lerobot_dataset.py \
  --input "${RECORDS_DIR}" \
  --output-root "${DATASET_ROOT}" \
  --repo-id local/tacex_lab_pick_slide \
  --success-only --overwrite

"${LEROBOT_PYTHON}" scripts/reinforcement_learning/dsrl/check_dsrl_ready.py \
  --records "${RECORDS_DIR}" --dataset-root "${DATASET_ROOT}" --min-success 50

"${REPO_ROOT}/scripts/bc_training/train_lab_pick_diffusion.py" \
  --dataset-root "${DATASET_ROOT}" \
  --repo-id local/tacex_lab_pick_slide \
  --output-dir "${DIFFUSION_OUTPUT}" \
  --steps "${DIFFUSION_STEPS}" --batch-size 32 --num-workers 4 --save-freq 20000 \
  --horizon 16 --action-steps 8 --down-dims 256 512 1024 \
  --inference-steps 10 --device cuda --video-backend pyav

POLICY_DIR="${DIFFUSION_OUTPUT}/checkpoints/last/pretrained_model"
"${LEROBOT_PYTHON}" scripts/reinforcement_learning/dsrl/check_dsrl_ready.py \
  --dataset-root "${DATASET_ROOT}" --policy "${POLICY_DIR}" --min-success 50
"${LEROBOT_PYTHON}" scripts/reinforcement_learning/dsrl/smoke_diffusion_noise.py --device cuda

"${ISAAC_PYTHON}" scripts/reinforcement_learning/dsrl/train_lab_pick_dsrl_sac.py \
  --dsrl_policy "${POLICY_DIR}" --timesteps "${DSRL_TIMESTEPS}" --seed 42

echo "[PIPELINE] completed successfully"
