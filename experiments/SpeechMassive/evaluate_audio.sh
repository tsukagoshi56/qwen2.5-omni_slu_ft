#!/bin/bash
set -e

# Speech-MASSIVE Audio Evaluation
# Uses audio input for inference

MODEL_PATH="${1:-outputs/speech_massive_audio_text_mix}"
LANG_CODE="${2:-fr-FR}"
OUTPUT_DIR="${3:-inference_outputs}"

# Find latest checkpoint if directory given
if [ -d "$MODEL_PATH" ]; then
    CHECKPOINT=$(ls -d "$MODEL_PATH"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        MODEL_PATH="$CHECKPOINT"
    fi
fi

echo "============================================================"
echo " Speech-MASSIVE Audio Evaluation"
echo "============================================================"
echo " Model:    $MODEL_PATH"
echo " Language: $LANG_CODE"
echo "============================================================"

uv run run_eval.py \
  --model_path "$MODEL_PATH" \
  --dataset speech_massive \
  --massive_dataset_config "$LANG_CODE" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 8 \
  --num_beams 3 \
  --no_transcript \
  --force_audio


# Run evaluation (calculate metrics)
PRED_FILE="$OUTPUT_DIR/$(basename "$MODEL_PATH")/predictions.jsonl"
GOLD_FILE="$OUTPUT_DIR/$(basename "$MODEL_PATH")/gold.jsonl"
echo "Calculating metrics for $PRED_FILE..."
uv run python scripts/evaluation/evaluate.py \
  --gold-data "$GOLD_FILE" \
  --prediction-file "$PRED_FILE"

echo "Evaluation complete. Check $OUTPUT_DIR for predictions."
