#!/bin/bash
set -e

# ============================================================
# Evaluation Script for Text-only Model
# ============================================================

# Default values
MODEL_PATH="${1:-outputs/text_only_stage1}"
TEST_FILE="${2:-slurp/dataset/slurp/test.jsonl}"
OUTPUT_DIR="${3:-inference_outputs}"

# Find latest checkpoint if directory given
if [ -d "$MODEL_PATH" ]; then
    CHECKPOINT=$(ls -d "$MODEL_PATH"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        MODEL_PATH="$CHECKPOINT"
    fi
fi

echo "============================================================"
echo " Evaluation"
echo "============================================================"
echo " Model:     $MODEL_PATH"
echo " Test File: $TEST_FILE"
echo " Output:    $OUTPUT_DIR"
echo "============================================================"

# Run evaluation
# --add_text_only is auto-detected from model config if trained with new code
uv run run_eval.py \
  --model_path "$MODEL_PATH" \
  --test_file "$TEST_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 32 \
  --num_beams 3 \
  --add_text_only

echo "Evaluation complete."
