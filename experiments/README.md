# Experiment Configuration

This folder contains scripts for reproducing the experiments from Paper 2509.15389v2.

## Training Configuration (Text-only Stage 1)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen2-Audio-7B-Instruct | Base model |
| Frozen | Audio Encoder, Modality Adapter | `audio_tower`, `multi_modal_projector` |
| Trainable | LLM (全パラメータ) | Full fine-tuning |
| Epochs | 3 | |
| Learning Rate | 5e-6 | Peak |
| LR Scheduler | Cosine | |
| Warmup Ratio | 0.04 | |
| Global Batch Size | 128 | `16 * 8 * 1` for single H200 |
| Precision | bfloat16 | |
| Input | Gold transcript only | `--add_text_only` |

## Scripts

### Full Experiment (Training + Evaluation)
```bash
bash experiments/run_experiment.sh
```

This runs the complete pipeline:
1. Text-only Training (Stage 1) 
2. Evaluation on test set

### Individual Steps

```bash
# Text-only Training (Stage 1)
bash experiments/text_only_stage1.sh

# Audio Training
bash experiments/audio_stage.sh

# Evaluation only (specify checkpoint)
bash experiments/evaluate.sh outputs/text_only_stage1/checkpoint-XXX
```

## Expected Results

After text-only training:
- **SLU-F1 (Text)**: ~0.85+ (using gold transcript)
- **SLU-F1 (Audio)**: Lower performance expected (model not trained on audio)

## Important Notes

1. **Re-training Required**: If you trained with the old code (before 2025-01-27), the model was trained on mixed audio+text data due to a bug. Re-train with the fixed code.

2. **Auto-detection**: The evaluation script automatically detects `text_only_mode` from the model config. No need to specify `--add_text_only` during evaluation if trained with the new code.
