# Experiments

This directory contains training and evaluation scripts for Qwen2-Audio SLU fine-tuning.

## Training Scripts

| Script | Description |
|--------|-------------|
| `train_text_only.sh` | Train using ground-truth transcripts only (no audio input) |
| `train_audio_text_mix.sh` | Train using both audio recordings and text data (simple mix) |

## Evaluation Scripts

| Script | Description | Key |
|--------|-------------|-----|
| `evaluate_text.sh` | Evaluate text-only model | `slurp_id` (uses `--load-gold`) |
| `evaluate_audio.sh` | Evaluate audio model | `file` (filename) |

## Usage

### Text-Only Experiment
```bash
# Train
bash experiments/train_text_only.sh

# Evaluate
bash experiments/evaluate_text.sh outputs/text_only
```

### Audio + Text Mixed Experiment
```bash
# Train
bash experiments/train_audio_text_mix.sh

# Evaluate
bash experiments/evaluate_audio.sh outputs/audio_text_mix
```

## Evaluation Key Note

- **Text-Only (Gold Transcript)**: Uses `slurp_id` as the matching key between predictions and gold data. Requires `--load-gold` flag.
- **Audio**: Uses `file` (filename) as the matching key. Does not use `--load-gold`.
