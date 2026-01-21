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
python prepare_data.py

# Or download only slurp_real (~4GB, recommended for testing)
python prepare_data.py --real-only

# If tar files already downloaded, just extract
python prepare_data.py --skip-download

# Validate existing data
python prepare_data.py --validate-only
```

After preparation, you should have:
```
slurp/
├── audio/
│   ├── slurp_real/      # Real recordings (~4GB)
│   │   └── *.flac
│   └── slurp_synth/     # Synthetic recordings (~6GB)
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
python train_qwen2_audio_slurp.py \
  --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
  --add_text_only \
  --max_train_samples 4 \
  --max_steps 1 \
  --output_dir outputs/dry-run
```

### Full Training with Audio

```bash
python train_qwen2_audio_slurp.py \
  --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
  --output_dir outputs/qwen2-audio-slurp \
  --bf16
```

### Speech-MASSIVE Dataset

```bash
python train_qwen2_audio_slurp.py \
  --dataset speech_massive \
  --massive_dataset_config fr-FR \
  --massive_train_split train_115 \
  --output_dir outputs/speech-massive
```

## Key Options

| Option | Description |
|--------|-------------|
| `--add_text_only` | Use text transcripts only (no audio) |
| `--use_all_recordings` | Use all recordings per utterance (default: best WER) |
| `--no_include_transcript` | Remove transcript from prompt |
| `--push_to_hub` | Push trained model to Hugging Face Hub |
| `--bf16` / `--fp16` | Use mixed precision training |
| `--use_lora` / `--no_lora` | Enable/disable LoRA (default: enabled) |

## Output Format

The model outputs JSON with scenario, action, and entities:

```json
{"scenario": "alarm", "action": "set", "entities": [{"time": "eight o'clock"}]}
```

## License

- Training code: MIT
- SLURP text data: CC BY 4.0
- SLURP audio data: CC BY-NC 4.0
