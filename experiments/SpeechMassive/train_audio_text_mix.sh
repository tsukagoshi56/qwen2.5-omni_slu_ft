#!/bin/bash
set -e

# Speech-MASSIVE Audio + Text Mixed Training
# Input: Audio recordings + Text data (simple mix)

MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_DIR="outputs/speech_massive_audio_text_mix"

# Configuration
LANG_CODE="${1:-fr-FR}"
TRAIN_SPLIT="${2:-train}"
EVAL_SPLIT="${3:-validation}"

NUM_EPOCHS=3
LEARNING_RATE=5e-6
WARMUP_RATIO=0.04
PER_DEVICE_BATCH=8
GRAD_ACCUMULATION=16

echo "============================================================"
echo " Speech-MASSIVE Audio + Text Mixed Training"
echo "============================================================"
echo " Language:       $LANG_CODE"
echo " Model:          $MODEL_NAME"
echo " Output:         $OUTPUT_DIR"
echo " Global Batch:   $((PER_DEVICE_BATCH * GRAD_ACCUMULATION))"
echo "============================================================"

uv run train_qwen2_audio_slurp.py \
  --dataset speech_massive \
  --massive_dataset_config "$LANG_CODE" \
  --massive_train_split "$TRAIN_SPLIT" \
  --massive_eval_split "$EVAL_SPLIT" \
  --model_name_or_path "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --add_text_only \
  --no_transcript \
  --num_train_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --warmup_ratio $WARMUP_RATIO \
  --per_device_train_batch_size $PER_DEVICE_BATCH \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 \
  --debug_generation \
  --debug_generation_steps 10

echo "Training complete. Model saved to: $OUTPUT_DIR"
