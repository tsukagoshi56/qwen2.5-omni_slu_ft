#!/bin/bash
set -e

# Server Training Script for audio_text_mix_e2e.py
# Uses torchrun for distributed training

NUM_GPUS=${1:-1}  # Default to 1 GPU, pass as first argument

echo "Starting training with ${NUM_GPUS} GPU(s)..."

torchrun --nproc_per_node=${NUM_GPUS} audio_text_mix_e2e.py \
    --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
    --output_dir outputs/qwen2-audio-text-mix \
    --num_train_epochs 2 \
    --batch_size 2 \
    --learning_rate 1e-5

echo "Training complete."
