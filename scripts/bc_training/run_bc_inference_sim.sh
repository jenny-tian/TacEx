#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/limx/github_repo/TacEx"
PYTHON_BIN="/home/limx/anaconda3/envs/env_isaaclab/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] ${PYTHON_BIN} is not executable. Please check the conda env_isaaclab environment." >&2
    exit 1
fi

cd "${REPO_ROOT}"

export PYTHONPATH="source/tacex:source/tacex_assets:source/tacex_tasks:${PYTHONPATH:-}"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export PYTHONUNBUFFERED=1
if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

DEFAULT_ARGS=(
    --checkpoint checkpoints/tacex_dinov3_fm_bc/best.pt
    --labware slide
    --num_envs 1
    --num_trials 20
    --max_episode_steps 960
    --aligned_hz 30
    --break_force_threshold_n 6.0
    --chunk_execute_steps 16
)

exec "${PYTHON_BIN}" scripts/bc_training/bc_inference_sim.py "${DEFAULT_ARGS[@]}" "$@"
