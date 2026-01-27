#!/bin/bash
set -e

# Speech-MASSIVE Text-Only Evaluation
# Uses slurp_id-style key for gold text evaluation

MODEL_PATH="${1:-outputs/speech_massive_text_only}"
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
echo " Speech-MASSIVE Text-Only Evaluation"
echo "============================================================"
echo " Model:    $MODEL_PATH"
echo " Language: $LANG_CODE"
echo "============================================================"

# Note: Speech-MASSIVE evaluation requires custom handling
# The predictions are saved and can be manually evaluated

uv run run_eval.py \
  --model_path "$MODEL_PATH" \
  --dataset speech_massive \
  --massive_dataset_config "$LANG_CODE" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 32 \
  --num_beams 3 \
  --add_text_only

echo "Evaluation complete. Check $OUTPUT_DIR for predictions."
