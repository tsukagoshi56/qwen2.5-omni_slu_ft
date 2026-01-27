#!/bin/bash
set -e

# ============================================================
# Full Experiment Pipeline: Training → Evaluation
# ============================================================
# Runs the complete text-only fine-tuning experiment end-to-end.
# Individual steps can also be run separately using:
#   - experiments/text_only_stage1.sh
#   - experiments/evaluate.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_DIR="outputs/text_only_stage1"
TEST_FILE="slurp/dataset/slurp/test.jsonl"

# Paper Hyperparameters
NUM_EPOCHS=3
LEARNING_RATE=5e-6
WARMUP_RATIO=0.04
LR_SCHEDULER="cosine"

# H200 Optimized Batch Configuration
PER_DEVICE_BATCH=16
GRAD_ACCUMULATION=8

echo ""
echo "============================================================"
echo " EXPERIMENT: Text-only Fine-tuning (Paper 2509.15389v2)"
echo "============================================================"
echo " Model:          $MODEL_NAME"
echo " Output:         $OUTPUT_DIR"
echo " Test File:      $TEST_FILE"
echo " Global Batch:   $((PER_DEVICE_BATCH * GRAD_ACCUMULATION))"
echo "============================================================"
echo ""

# ============================================================
# STEP 1: Training
# ============================================================
echo "[STEP 1/2] Training..."
echo "------------------------------------------------------------"

uv run train_qwen2_audio_slurp.py \
  --model_name_or_path "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --add_text_only \
  --num_train_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --warmup_ratio $WARMUP_RATIO \
  --lr_scheduler_type $LR_SCHEDULER \
  --per_device_train_batch_size $PER_DEVICE_BATCH \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16

echo ""
echo "[STEP 1/2] Training complete."
echo ""

# ============================================================
# STEP 2: Evaluation
# ============================================================
echo "[STEP 2/2] Evaluation..."
echo "------------------------------------------------------------"

# Find latest checkpoint
CHECKPOINT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$CHECKPOINT" ]; then
    echo "No checkpoint found in $OUTPUT_DIR. Using final model."
    CHECKPOINT="$OUTPUT_DIR"
fi

echo "Using checkpoint: $CHECKPOINT"

uv run run_eval.py \
  --model_path "$CHECKPOINT" \
  --test_file "$TEST_FILE" \
  --output_dir "inference_outputs" \
  --batch_size 32 \
  --num_beams 3

echo ""
echo "============================================================"
echo " EXPERIMENT COMPLETE"
echo "============================================================"
echo " Model:       $CHECKPOINT"
echo " Predictions: inference_outputs/$(basename "$CHECKPOINT")/predictions.jsonl"
echo "============================================================"
