# Qwen2-Audio SLURP Finetune

Finetune Qwen2-Audio on SLURP or Speech-MASSIVE for Spoken Language Understanding.

## Requirements

- Python 3.10+ (recommended: 3.11)
- ~10GB disk space for full SLURP audio (~4GB for real-only)
- GPU with sufficient VRAM for Qwen2-Audio-7B

## Setup (uv)

This project uses `uv` for dependency management.

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (creates .venv automatically)
uv sync
```

## Data Preparation

The training script requires SLURP audio data. Use `prepare_data.py` to download and extract:

```bash
# Download and extract all audio (~10GB total)
uv run prepare_data.py

# Or download only slurp_real (~4GB, recommended for testing)
uv run prepare_data.py --real-only

# If tar files already downloaded, just extract
uv run prepare_data.py --skip-download

# Validate existing data
uv run prepare_data.py --validate-only
```

After preparation, you should have:
```
slurp/
├── audio/
│   └── slurp_real/      # Real recordings (~4GB)
│       └── *.flac
└── dataset/
    └── slurp/
        ├── train.jsonl
        ├── devel.jsonl
        └── test.jsonl
```

## Training

### Text-only Dry Run (no audio required)

```bash
uv run train_qwen2_audio_slurp.py \
  --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
  --add_text_only \
  --max_train_samples 4 \
  --max_steps 1 \
  --output_dir outputs/dry-run
```

### Full Training with Audio

```bash
### Full Training with Audio

```bash
uv run train_qwen2_audio_slurp.py \
  --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
  --output_dir outputs/qwen2-audio-slurp \
  --bf16
```

### Text-only Training (Gold Transcript)

Train using only the text transcripts (Gold Text). No audio files required.

```bash
uv run train_qwen2_audio_slurp.py \
  --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
  --add_text_only \
  --output_dir outputs/qwen2-text-slurp \
  --bf16
```

## Evaluation

Use `run_eval.py` to evaluate the trained model on the SLURP test set. This script performs inference and calculates SLU metrics (F1, SLU-F1).

### Single-GPU Evaluation

```bash
uv run run_eval.py \
  --model_path outputs/qwen2-audio-slurp/checkpoint-XXXX \
  --test_file slurp/dataset/slurp/test.jsonl \
  --batch_size 16 \
  --device cuda
```

### Multi-GPU Evaluation (Recommended)

To accelerate inference using multiple GPUs (e.g., 2x NVIDIA H200), use `torchrun`. This will automatically distribute the workload and merge the results.

```bash
# Example: Run on 2 GPUs
uv run torchrun --nproc_per_node=2 run_eval.py \
  --model_path outputs/qwen2-audio-slurp/checkpoint-XXXX \
  --test_file slurp/dataset/slurp/test.jsonl \
  --batch_size 16 \
  --num_beams 3 \
  --device cuda
```

- `--nproc_per_node`: Number of GPUs to use.
- `--model_path`: Path to the trained LoRA checkpoint or full model directory.
- `--batch_size`: Inference batch size per GPU.
- `--add_text_only`: Add this flag if evaluating a text-only model.
- `--no_transcript`: Use this flag for audio models to exclude transcript from prompt (Audio input only).

### Debugging Training Data

If you suspect issues with the training data format, use `debug_training_data.py` to inspect exactly what the model sees:

```bash
uv run debug_training_data.py --add_text_only --num_samples 3
```

To save the full processed training data (input prompts and targets) to a file for inspection:
```bash
uv run debug_training_data.py \
  --add_text_only \
  --num_samples 0 \
  --save_path processed_train.jsonl
```

This will output the prompt, raw data item, and the full processed text (input + target) for verification.

### Speech-MASSIVE Dataset

```bash
uv run train_qwen2_audio_slurp.py \
  --dataset speech_massive \
  --massive_dataset_config fr-FR \
  --massive_train_split train_115 \
  --output_dir outputs/speech-massive
```

## Rationale + Label Fine-tuning (ASR CoT)

For rationale-conditioned SLU fine-tuning with audio input, use:

`Experiment_RationaleFT/audio_text_mix_e2e_re.py`

This trainer consumes JSONL files that include:
- `candidates` (ASR n-best)
- `rationale_text`
- `final` (gold label: intent/entities)
- `filename` (audio file)

Default train/devel paths are set to:
- `/lustre/home/71200138/qwen_test/experiments/CoT_maker/ASR_cot_train.jsonl`
- `/lustre/home/71200138/qwen_test/experiments/CoT_maker/ASR_cot_devel.jsonl`

Example:

```bash
uv run Experiment_RationaleFT/audio_text_mix_e2e_re.py \
  --audio_dir /lustre/home/71200138/INTERSPEECH/experiment1/slurp/audio/slurp_real \
  --output_dir outputs/qwen_rationale_label_ft
```

Outputs:
- `prediction.jsonl`: full output (includes n-best/rationale/raw_output/target)
- `prediction_labels_only.jsonl`: label-only view for evaluation
- `metrics_label_only.json`: label-only metrics (scenario/action/intent acc + entity P/R/F1)

## Key Options

| Option | Description |
|--------|-------------|
| `--add_text_only` | Use text transcripts only (no audio) |
| `--batch_size` | Inference batch size (default: 1) |
| `--use_all_recordings` | Use all recordings per utterance (default: best WER) |
| `--no_include_transcript` | Remove transcript from prompt |
| `--push_to_hub` | Push trained model to Hugging Face Hub |
| `--bf16` / `--fp16` | Use mixed precision training |
| `--use_lora` | Enable LoRA (default: disabled, full finetune) |
| `--debug_generation` | Enable periodical generation logging during training |
| `--debug_generation_steps` | Steps interval for generation logging (default: 5) |

## Output Format (Paper 2509.15389v2)

This repository implements the format specified in Paper 2509.15389v2.

**Model Output (Raw LLM):**
```json
{"scenario": "alarm", "action": "set", "entities": [{"time": "eight o'clock"}]}
```

**Evaluation Output (`predictions.jsonl`):**
The inference script (`run_eval.py`) automatically converts the model output back to the standard SLURP evaluation format:
```json
// Text-only mode
{"slurp_id": "12345", "scenario": "alarm", "action": "set", "entities": [{"type": "time", "filler": "eight o'clock"}]}

// Audio mode
{"file": "audio-12345.flac", "scenario": "alarm", "action": "set", "entities": [{"type": "time", "filler": "eight o'clock"}]}
```

## Paper Compliance (2509.15389v2)

This codebase is aligned with the specifications from Paper 2509.15389v2.

### Quick Start (Recommended)

Use the pre-configured scripts in the `experiments/` folder:

```bash
# 1. Text-only Training (Stage 1)
bash experiments/text_only_stage1.sh

# 2. Evaluation (auto-detects text-only mode)
bash experiments/evaluate.sh
```

See [experiments/README.md](experiments/README.md) for detailed configuration.

### Training Configuration Summary

| Parameter | Value |
|-----------|-------|
| Frozen | Audio Encoder, Modality Adapter |
| Trainable | LLM (full fine-tuning) |
| Epochs | 3 |
| Learning Rate | 5e-6 (cosine decay) |
| Global Batch Size | 128 |
| Precision | bfloat16 |

## License

- Training code: MIT
- SLURP text data: CC BY 4.0
- SLURP audio data: CC BY-NC 4.0
