#!/bin/bash
set -e

# Evaluate Audio Model
# Uses filename as key (no --load-gold)

MODEL_PATH="${1:-outputs/audio_text_mix}"
TEST_FILE="${2:-slurp/dataset/slurp/test.jsonl}"
OUTPUT_DIR="${3:-inference_outputs}"
AUDIO_DIR="${4:-slurp/audio/slurp_real}"

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
# GPU settings
NUM_GPUS=2

# 1. Run inference (generate predictions.jsonl)
# Using torchrun for 2-GPU parallel inference
torchrun --nproc_per_node=$NUM_GPUS run_eval.py \
  --model_path "$MODEL_PATH" \
  --dataset slurp \
  --test_file "$TEST_FILE" \
  --audio_dir "$AUDIO_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 16 \
  --num_beams 3 \
  --no_transcript \
  --force_audio

# Run evaluation without --load-gold (uses filename as key)
PRED_FILE="$OUTPUT_DIR/$(basename "$MODEL_PATH")/predictions.jsonl"
python scripts/evaluation/evaluate.py \
  --gold-data "$TEST_FILE" \
  --prediction-file "$PRED_FILE"

echo "Evaluation complete."
