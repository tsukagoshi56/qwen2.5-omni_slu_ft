#!/bin/bash
set -e

# Evaluate Audio Model
# Uses filename as key (no --load-gold)

MODEL_PATH="${1:-outputs/audio_text_mix}"
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
echo " Audio Evaluation (key: filename)"
echo "============================================================"
echo " Model:     $MODEL_PATH"
echo " Test File: $TEST_FILE"
echo "============================================================"

uv run run_eval.py \
  --model_path "$MODEL_PATH" \
  --test_file "$TEST_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 8 \
  --num_beams 3 \
  --no_transcript

# Run evaluation without --load-gold (uses filename as key)
PRED_FILE="$OUTPUT_DIR/$(basename "$MODEL_PATH")/predictions.jsonl"
python scripts/evaluation/evaluate.py \
  --gold-data "$TEST_FILE" \
  --prediction-file "$PRED_FILE"

echo "Evaluation complete."
