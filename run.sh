#!/usr/bin/env bash
# Usage: ./run.sh <method> <data_name> <data_dir> <exp_dir>
# method: rand | H_reg | variance | physics | combine | combine_active
# Example: ./run.sh combine lego /data/nerf_synthetic /exp/output

set -euo pipefail

METHOD=$1
DATA_NAME=$2
DATA_DIR=$3
EXP_DIR=$4

SCHEMA="v10seq1_inplace"
ITER=30000
METRIC="div"
SEED=0
CONFIG="configs/sim_config.json"

EXP_PATH="${EXP_DIR}/${METHOD}_${DATA_NAME}"
mkdir -p "${EXP_PATH}/logs"

CUDA_VISIBLE_DEVICES=0 \
python active_train.py \
  -s "${DATA_DIR}/${DATA_NAME}" \
  -m "${EXP_PATH}" \
  --method "${METHOD}" \
  --simulation_scene_config "${CONFIG}" \
  --schema "${SCHEMA}" \
  --iterations "${ITER}" \
  --metric_type "${METRIC}" \
  --seed "${SEED}" \
  2>&1 | tee "${EXP_PATH}/logs/train.log"
