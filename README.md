# Qwen2-Audio SLURP Finetune

This repo contains a training script to finetune Qwen2-Audio on SLURP or
Speech-MASSIVE with the
prompt:

"""
Extract scenario, action, and entities (empty list if none) and
return a single-line JSON: {"scenario": "<string>", "action":
"<string>", "entities": [{"<entity_type>": "<entity_value>"}, ...]}
"""

The script can automatically download the SLURP repo and (optionally) the audio.
Speech-MASSIVE is loaded from Hugging Face datasets.

## Requirements

- Python 3.10+ (recommended: 3.11)
- macOS/Linux with enough disk space (audio is ~6GB)
- Access to Hugging Face to download Qwen2-Audio weights

## Setup (uv)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv
~/.local/bin/uv venv --python 3.11 .venv

# Install dependencies
~/.local/bin/uv pip install --python .venv \
  torch torchaudio transformers peft soundfile librosa accelerate datasets
```

## Quick Start (text-only dry run)

This runs on a very small subset without audio.

```bash
.venv/bin/python train_qwen2_audio_slurp.py \
  --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
  --add_text_only \
  --max_train_samples 4 \
  --max_eval_samples 2 \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --save_steps 1 \
  --eval_steps 1 \
  --max_length 512
```

## Full Training (audio + text)

This will download the SLURP repo and audio if missing.

```bash
.venv/bin/python train_qwen2_audio_slurp.py \
  --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
  --download_audio \
  --output_dir outputs/qwen2-audio-slurp \
  --bf16
```

## Speech-MASSIVE (audio + text)

Speech-MASSIVE uses intent and slot labels from MASSIVE.
We map `scenario_str` -> `scenario`, `intent_str` -> `action`, and slot labels
to `entities`.

```bash
.venv/bin/python train_qwen2_audio_slurp.py \
  --dataset speech_massive \
  --massive_dataset_config fr-FR \
  --massive_train_split train_115 \
  --massive_eval_split validation
```

For full training data, use `fr-FR` or `de-DE` with `--massive_train_split train`.

## Notes

- Use `--dataset slurp` (default) or `--dataset speech_massive`.
- `--download_slurp` is ON by default. Use `--no_download_slurp` to disable.
- Audio download is large; use `--download_audio` when you want it.
- If you are not authenticated with Hugging Face, run `huggingface-cli login`.
- The script looks for audio under `slurp/audio/` with `slurp_real/` and
  `slurp_synth/` subfolders.
- By default the prompt also includes the transcript (`Transcript: ...`).
  Use `--no_include_transcript` to remove it.

## Outputs

- Model checkpoints are written to `outputs/qwen2-audio-slurp` by default.
