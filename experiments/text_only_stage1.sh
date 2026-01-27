#!/bin/bash
set -e

# ============================================================
# Text-only Fine-tuning Stage 1 (Paper 2509.15389v2)
# ============================================================
# Target: Train LLM on gold transcripts only (no audio input)
# Frozen: Audio Encoder, Modality Adapter
# Trainable: LLM (full fine-tuning)
# ============================================================

MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_DIR="outputs/text_only_stage1"

# Paper Hyperparameters
NUM_EPOCHS=3
LEARNING_RATE=5e-6
WARMUP_RATIO=0.04
LR_SCHEDULER="cosine"

# H200 Optimized Batch Configuration
# Global Batch Size = 16 * 8 * 1 = 128 (matches paper's 8-GPU setup)
PER_DEVICE_BATCH=16
GRAD_ACCUMULATION=8

echo "============================================================"
echo " Text-only Fine-tuning Stage 1"
echo "============================================================"
echo " Model:          $MODEL_NAME"
echo " Output:         $OUTPUT_DIR"
echo " Epochs:         $NUM_EPOCHS"
echo " Learning Rate:  $LEARNING_RATE"
echo " Scheduler:      $LR_SCHEDULER"
echo " Global Batch:   $((PER_DEVICE_BATCH * GRAD_ACCUMULATION))"
echo "============================================================"

uv run train_qwen2_audio_slurp.py \
  --model_name_or_path "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --add_text_only \
  --num_train_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --warmup_ratio $WARMUP_RATIO \
  --per_device_train_batch_size $PER_DEVICE_BATCH \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16

echo "Training complete. Checkpoint saved to: $OUTPUT_DIR"
