#!/bin/bash
set -e

# Audio-Text Mix Fine-tuning (Direct Mixing)
# Input: Mix of [Transcript -> JSON] and [Audio -> JSON]
# Output: SLU JSON labels

MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"

# Paper Hyperparameters for Direct Mixing
NUM_EPOCHS=3
LEARNING_RATE=4e-5
WARMUP_RATIO=0.04

# Multi-GPU Configuration
# Set NUM_GPUS to 1 or 2 as needed.
NUM_GPUS=2

# H200 optimized for Multi-GPU
# Target Global Batch: 128
# Formula: PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUMULATION = 128
# With 2 GPUs and Batch 16, Accum should be 4 (16*2*4=128)
# With 1 GPU  and Batch 32, Accum should be 4 (32*1*4=128)

PER_DEVICE_BATCH=16
GRAD_ACCUMULATION=4

GLOBAL_BATCH=$((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUMULATION))

# Generate specific output directory with timestamp and config
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_ROOT="outputs/audio_text_mix"
run_id="${TIMESTAMP}_ep${NUM_EPOCHS}_lr${LEARNING_RATE}_bs${GLOBAL_BATCH}_gpu${NUM_GPUS}"
OUTPUT_DIR="${OUTPUT_ROOT}/${run_id}"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo " Audio-Text Mix Fine-tuning"
echo "============================================================"
echo " Model:          $MODEL_NAME"
echo " Output Dir:     $OUTPUT_DIR"
echo " GPUs:           $NUM_GPUS"
echo " Global Batch:   $GLOBAL_BATCH"
echo "============================================================"

# Save a copy of this script to the output directory for reproducibility
cp "$0" "$OUTPUT_DIR/train_script_copy.sh"

# Use torchrun for multi-gpu support (works for 1 GPU too)
torchrun --nproc_per_node=$NUM_GPUS train_qwen2_audio_slurp.py \
  --model_name_or_path "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --add_text_only \
  --use_all_recordings \
  --partition_audio \
  --num_train_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --warmup_ratio $WARMUP_RATIO \
  --per_device_train_batch_size $PER_DEVICE_BATCH \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 \
  --ddp_find_unused_parameters False

echo "Training complete. Model saved to: $OUTPUT_DIR"
