#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-deepseek-r1}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}}"

TRAIN_INPUT="${ROOT_DIR}/slurp/dataset/slurp/train.jsonl"
DEVEL_INPUT="${ROOT_DIR}/slurp/dataset/slurp/devel.jsonl"
METADATA_FILE="${ROOT_DIR}/Experiment_3/slurp_metadata.json"

mkdir -p "${OUTPUT_DIR}"

# Train split
uv run python "${SCRIPT_DIR}/01_generate_oracle_cot.py" \
  --input_file "${TRAIN_INPUT}" \
  --metadata_file "${METADATA_FILE}" \
  --output_file "${OUTPUT_DIR}/oracle_cot_train.jsonl" \
  --model_name "${MODEL_NAME}"

# Devel split
uv run python "${SCRIPT_DIR}/01_generate_oracle_cot.py" \
  --input_file "${DEVEL_INPUT}" \
  --metadata_file "${METADATA_FILE}" \
  --output_file "${OUTPUT_DIR}/oracle_cot_devel.jsonl" \
  --model_name "${MODEL_NAME}"
