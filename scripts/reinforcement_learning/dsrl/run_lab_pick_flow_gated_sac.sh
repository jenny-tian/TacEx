#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${TACEX_ISAAC_PYTHON:-/home/tjx/miniforge3/envs/env_isaaclab/bin/python}"
BC_POLICY="${TACEX_DSRL_BC_POLICY:-${REPO_ROOT}/outputs/lab_pick_flow_bc200_hardyaw_rotation_scratch/best.pt}"
TIMESTEPS="${TACEX_DSRL_TIMESTEPS:-200000}"
SEED="${TACEX_DSRL_SEED:-42}"
NOISE_MAGNITUDE="${TACEX_DSRL_NOISE_MAGNITUDE:-1.5}"
GATE_INIT="${TACEX_DSRL_GATE_INIT:-0.025}"
GATE_TEMPERATURE="${TACEX_DSRL_GATE_TEMPERATURE:-0.5}"
GATE_PENALTY="${TACEX_DSRL_GATE_PENALTY:-5.0}"
GATE_MIN="${TACEX_DSRL_GATE_MIN:-0.02}"
GATE_MAX="${TACEX_DSRL_GATE_MAX:-0.2}"

if [[ ! -f "${BC_POLICY}" ]]; then
  echo "Missing Flow Matching BC checkpoint: ${BC_POLICY}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES

exec "${PYTHON}" scripts/reinforcement_learning/dsrl/train_lab_pick_dsrl_sac.py \
  --dsrl_policy "${BC_POLICY}" \
  --dsrl_policy_type flow_matching \
  --dsrl_residual_mode physical \
  --dsrl_physical_residual_segments 4 \
  --dsrl_flow_chunk_execute_steps 16 \
  --dsrl_chunk_discount 1.0 \
  --dsrl_noise_magnitude "${NOISE_MAGNITUDE}" \
  --dsrl_gate \
  --dsrl_gate_init "${GATE_INIT}" \
  --dsrl_gate_temperature "${GATE_TEMPERATURE}" \
  --dsrl_gate_penalty "${GATE_PENALTY}" \
  --dsrl_gate_min "${GATE_MIN}" \
  --dsrl_gate_max "${GATE_MAX}" \
  --sac_terminal_timeouts \
  --sac_terminal_sample_fraction 0.25 \
  --sac_discount_factor 0.99 \
  --rl_success_reward 120 \
  --rl_failure_penalty 30 \
  --rl_timeout_penalty 40 \
  --rl_late_no_progress_penalty 0.25 \
  --rl_late_no_progress_onset 0.5 \
  --timesteps "${TIMESTEPS}" \
  --seed "${SEED}"
