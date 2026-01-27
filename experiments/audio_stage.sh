#!/bin/bash
set -e

# ============================================================
# Audio Fine-tuning (with audio input)
# ============================================================
# Target: Train LLM on audio input
# Frozen: Audio Encoder, Modality Adapter
# Trainable: LLM (full fine-tuning)
# ============================================================

MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_DIR="outputs/audio_stage"

# Paper Hyperparameters
NUM_EPOCHS=3
LEARNING_RATE=5e-6
WARMUP_RATIO=0.04

# H200 Optimized Batch Configuration
# Note: Audio requires more memory, reduce batch size if OOM
PER_DEVICE_BATCH=8
GRAD_ACCUMULATION=16

echo "============================================================"
echo " Audio Fine-tuning"
echo "============================================================"
echo " Model:          $MODEL_NAME"
echo " Output:         $OUTPUT_DIR"
echo " Epochs:         $NUM_EPOCHS"
echo " Learning Rate:  $LEARNING_RATE"
echo " Global Batch:   $((PER_DEVICE_BATCH * GRAD_ACCUMULATION))"
echo "============================================================"

uv run train_qwen2_audio_slurp.py \
  --model_name_or_path "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
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

echo "Training complete. Checkpoint saved to: $OUTPUT_DIR"
