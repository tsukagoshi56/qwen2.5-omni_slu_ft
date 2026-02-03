# Experiment: Rationale Fine-Tuning (FT)

This directory contains the training and inference scripts for the **Rationale-Conditioned SLU** experiment. 
It focuses on fine-tuning **Qwen2-Audio** to predict final SLU labels (Scenario, Action, Entities) given:
1. Input Audio
2. ASR N-best hypotheses (from context)
3. Rationale text (Chain-of-Thought reasoning)

## Script: `audio_text_mix_e2e_re.py`

This script handles the end-to-end fine-tuning process. It expects input data that already contains generated rationales and ASR n-best lists (produced by steps in `Experiment_Rationale`).

### Features
- **Prompting**: Constructs a prompt combining "ASR n-best hypotheses" and "Rationale" before asking for the "SLU" JSON label.
- **Audio Conditioning**: Uses the Qwen2-Audio encoder to listen to the raw audio.
- **Robustness**: Supports falling back to text-only training if audio files are missing (configurable).
- **Evaluation**: Performs full inference on the test set and calculates standard SLU metrics (Scenario Acc, Action Acc, Entity F1).

## Usage

### Single GPU Training

```bash
uv run Experiment_RationaleFT/audio_text_mix_e2e_re.py \
  --train_file /path/to/ASR_cot_train.jsonl \
  --eval_file /path/to/ASR_cot_devel.jsonl \
  --test_file slurp/dataset/slurp/test.jsonl \
  --audio_dir /path/to/slurp/audio/slurp_real \
  --output_dir outputs/qwen_rationale_label_ft
```

### Multi-GPU Training (Distributed)

Use `torchrun` to scale to multiple GPUs (e.g., 2 GPUs). Note that `--batch_size` is per-GPU.

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  Experiment_RationaleFT/audio_text_mix_e2e_re.py \
  --train_file /path/to/ASR_cot_train.jsonl \
  --eval_file /path/to/ASR_cot_devel.jsonl \
  --test_file slurp/dataset/slurp/test.jsonl \
  --audio_dir /path/to/slurp/audio/slurp_real \
  --output_dir outputs/qwen_rationale_label_ft_2gpu \
  --batch_size 8 \
  --num_train_epochs 3 \
  --export_label_eval
```

## Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--train_file` | Path to training JSONL (with rationales) | `.../ASR_cot_train.jsonl` |
| `--eval_file` | Path to validation JSONL | `.../ASR_cot_devel.jsonl` |
| `--test_file` | Path to test JSONL (SLURP format) | `slurp/dataset/slurp/test.jsonl` |
| `--audio_dir` | Root directory for resolving audio files | `.../slurp_real` |
| `--output_dir` | Directory to save model and predictions | `outputs/qwen_rationale_label_ft` |
| `--model_name_or_path` | Base model path | `Qwen/Qwen2-Audio-7B-Instruct` |
| `--batch_size` | Training batch size per device | `1` |
| `--export_label_eval` | Calculate and save label-only metrics after inference | `False` |
| `--add_text_only` | Augment training with text-only samples | `False` |
| `--strict_audio_missing` | Fail immediately if an audio file is not found | `False` |
| `--smoke` | Run a tiny "smoke test" with minimal data | `False` |

## Input Data Format

The training/eval JSONL files should contain records that have:
- `text`/`transcript`: (Optional) Ground truth text.
- `candidates`: List of ASR n-best strings.
- `rationale_text`: The reasoning text generated in the previous experiment stage.
- `final` / `target_obj`: The target SLU label `{"scenario": "...", "action": "...", "entities": [...]}`.
- `file` / `filename`: Path to the audio file relative to `audio_dir`.

## Output

The script produces:
- **Checkpoints**: Saved in `output_dir`.
- **`prediction.jsonl`**: Full output including generated text.
- **`prediction_labels_only.jsonl`**: (If `--export_label_eval`) JSONL with only the parsed SLU labels.
- **`metrics_label_only.json`**: (If `--export_label_eval`) Calculated accuracy and F1 scores.
