#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${TACEX_ISAAC_PYTHON:-/home/tjx/miniforge3/envs/env_isaaclab/bin/python}"
BC_POLICY="${TACEX_DSRL_BC_POLICY:-${REPO_ROOT}/outputs/lab_pick_flow_bc100_scratch_rawclose_safe70_overforce24_pos6/best.pt}"
TIMESTEPS="${TACEX_DSRL_TIMESTEPS:-200000}"
SEED="${TACEX_DSRL_SEED:-42}"
NOISE_MAGNITUDE="${TACEX_DSRL_NOISE_MAGNITUDE:-1.5}"
GATE_INIT="${TACEX_DSRL_GATE_INIT:-0.1}"
GATE_TEMPERATURE="${TACEX_DSRL_GATE_TEMPERATURE:-0.5}"
GATE_PENALTY="${TACEX_DSRL_GATE_PENALTY:-0.1}"
GATE_MAX="${TACEX_DSRL_GATE_MAX:-0.3}"

if [[ ! -f "${BC_POLICY}" ]]; then
  echo "Missing Flow Matching BC checkpoint: ${BC_POLICY}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES

exec "${PYTHON}" scripts/reinforcement_learning/dsrl/train_lab_pick_dsrl_sac.py \
  --dsrl_policy "${BC_POLICY}" \
  --dsrl_policy_type flow_matching \
  --dsrl_noise_magnitude "${NOISE_MAGNITUDE}" \
  --dsrl_gate \
  --dsrl_gate_init "${GATE_INIT}" \
  --dsrl_gate_temperature "${GATE_TEMPERATURE}" \
  --dsrl_gate_penalty "${GATE_PENALTY}" \
  --dsrl_gate_max "${GATE_MAX}" \
  --timesteps "${TIMESTEPS}" \
  --seed "${SEED}"
