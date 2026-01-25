#!/bin/bash
set -e

# Curriculum Learning Replication Script
# Based on: "For curriculum learning, the first two epochs are text-only... followed by a final epoch with both text and audio"

MODEL_PATH="Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_DIR="outputs/qwen2-audio-slurp-curriculum"
TEXT_ONLY_DIR="${OUTPUT_DIR}/stage1_text_only"
FINAL_DIR="${OUTPUT_DIR}/stage2_audio_text"

echo "=== Stage 1: Text-only Fine-tuning (2 Epochs) ==="
echo "Settings: LR=5e-6, Warmup=0.04"

uv run train_qwen2_audio_slurp.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$TEXT_ONLY_DIR" \
    --add_text_only \
    --num_train_epochs 2 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.04 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --bf16 \
    --debug_generation

echo "=== Stage 2: Audio + Text Fine-tuning (1 Epoch) ==="
echo "Settings: LR=3e-6, Warmup=0.02, Init from Stage 1"

uv run train_qwen2_audio_slurp.py \
    --model_name_or_path "$TEXT_ONLY_DIR" \
    --output_dir "$FINAL_DIR" \
    --num_train_epochs 1 \
    --learning_rate 3e-6 \
    --warmup_ratio 0.02 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --bf16 \
    --debug_generation

echo "Done! Final model saved in $FINAL_DIR"
