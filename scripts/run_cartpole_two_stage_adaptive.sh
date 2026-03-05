#!/usr/bin/env bash
set -euo pipefail

# Two-stage cartpole flow-matching training:
# Stage 1: base FM pretrain
# Stage 2: adaptive rollout fine-tune initialized from Stage 1 best checkpoint
#
# Usage:
#   scripts/run_cartpole_two_stage_adaptive.sh [RUN_TAG] [STAGE1_EPOCHS] [STAGE2_EPOCHS]
# Example:
#   scripts/run_cartpole_two_stage_adaptive.sh expA 200 200

RUN_TAG="${1:-two_stage_$(date +%Y%m%d_%H%M%S)}"
STAGE1_EPOCHS="${2:-200}"
STAGE2_EPOCHS="${3:-200}"

EXP_ROOT="$(python -c 'from genMoPlan.utils.paths import get_experiments_path; print(get_experiments_path())')"
STAGE1_EXP="flow_matching/${RUN_TAG}_s1"
STAGE2_EXP="flow_matching/${RUN_TAG}_s2"
STAGE1_DIR="${EXP_ROOT}/cartpole_pybullet/${STAGE1_EXP}"

echo "[two-stage] run_tag=${RUN_TAG}"
echo "[two-stage] experiments_root=${EXP_ROOT}"
echo "[two-stage] stage1_dir=${STAGE1_DIR}"
echo "[two-stage] stage1_epochs=${STAGE1_EPOCHS} stage2_epochs=${STAGE2_EPOCHS}"

python scripts/train_trajectory.py \
  --dataset cartpole_pybullet \
  --method flow_matching \
  --variations stage1_base_fast stride_3_horizon_15_path_300 data_lim_100 \
  --num_epochs "${STAGE1_EPOCHS}" \
  --custom_exp_name "${STAGE1_EXP}" \
  --no_inference

python scripts/train_trajectory.py \
  --dataset cartpole_pybullet \
  --method flow_matching \
  --variations stage2_adaptive_ft stride_3_horizon_15_path_300 data_lim_100 \
  --num_epochs "${STAGE2_EPOCHS}" \
  --custom_exp_name "${STAGE2_EXP}" \
  --init_from "${STAGE1_DIR}" \
  --init_state_name best.pt

echo "[two-stage] done. stage1=${STAGE1_EXP} stage2=${STAGE2_EXP}"
