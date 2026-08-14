#!/usr/bin/env bash
set -euo pipefail

# Example tree_mode inference entrypoint.
# Set ckpt_path in model_config/opt_tree_mode.yaml before running.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-model_config/opt_tree_mode.yaml}"
INPUT_DIR="${INPUT_DIR:-meshexample}"
OUT_DIR="${OUT_DIR:-inference_out}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

"${PYTHON}" src/main.py \
  --config "${CONFIG}" \
  --input_dir "${INPUT_DIR}" \
  --out_dir "${OUT_DIR}" \
  --input_type mesh \
  --mc \
  --mc_level 7 \
  "$@"
