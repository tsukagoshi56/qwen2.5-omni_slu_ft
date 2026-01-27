#!/bin/bash
set -e

# H200 Optimized Training Script
# Target: Global Batch Size 128 (Paper Paper 2509.15389v2)
# H200 VRAM (141GB) allows larger per-device batch size.
# Formulation: 16 (per_device) * 8 (grad_acc) * 1 (gpu) = 128

echo "Starting Text-only Fine-tuning on H200..."
echo "Config: Batch Size 16 * 8 = 128 (Paper Match)"

uv run train_qwen2_audio_slurp.py \
  --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
  --output_dir outputs/qwen2-text-slurp-paper-h200 \
  --add_text_only \
  --num_train_epochs 3 \
  --learning_rate 5e-6 \
  --warmup_ratio 0.04 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --bf16 \
  --logging_steps 1 \
  --save_steps 500

echo "Training complete."
