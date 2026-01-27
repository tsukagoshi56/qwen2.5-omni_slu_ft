#!/bin/bash
set -e

# Audio + Text Mixed Fine-tuning (Simple Mix)
# Input: Audio recordings + Ground-truth transcripts (combined dataset)
# Output: SLU JSON labels
# Note: --no_transcript removes transcript from prompt, so model learns from audio only

MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_DIR="outputs/audio_text_mix"

# Paper Hyperparameters
NUM_EPOCHS=3
LEARNING_RATE=5e-6
WARMUP_RATIO=0.04

# H200 Optimized Batch Configuration (reduced for audio memory)
PER_DEVICE_BATCH=8
GRAD_ACCUMULATION=16

echo "============================================================"
echo " Audio + Text Mixed Fine-tuning"
echo "============================================================"
echo " Model:          $MODEL_NAME"
echo " Output:         $OUTPUT_DIR"
echo " Global Batch:   $((PER_DEVICE_BATCH * GRAD_ACCUMULATION))"
echo "============================================================"

uv run train_qwen2_audio_slurp.py \
  --model_name_or_path "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --add_text_only \
  --use_all_recordings \
  --no_transcript \
  --num_train_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --warmup_ratio $WARMUP_RATIO \
  --per_device_train_batch_size $PER_DEVICE_BATCH \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16

echo "Training complete. Model saved to: $OUTPUT_DIR"
