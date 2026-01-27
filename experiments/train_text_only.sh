#!/bin/bash
set -e

# Text-Only Fine-tuning
# Input: Ground-truth transcripts only (no audio)
# Output: SLU JSON labels

MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_DIR="outputs/text_only"

# Paper Hyperparameters
NUM_EPOCHS=3
LEARNING_RATE=5e-6
WARMUP_RATIO=0.04

# H200 optimized for 1 GPU (Total Batch 128 = 2 * 64)
PER_DEVICE_BATCH=2
GRAD_ACCUMULATION=64

echo "============================================================"
echo " Text-Only Fine-tuning"
echo "============================================================"
echo " Model:          $MODEL_NAME"
echo " Output:         $OUTPUT_DIR"
echo " Global Batch:   $((PER_DEVICE_BATCH * GRAD_ACCUMULATION))"
echo "============================================================"

uv run train_qwen2_audio_slurp.py \
  --model_name_or_path "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --train_text_only \
  --num_train_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --warmup_ratio $WARMUP_RATIO \
  --per_device_train_batch_size $PER_DEVICE_BATCH \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16

echo "Training complete. Model saved to: $OUTPUT_DIR"
