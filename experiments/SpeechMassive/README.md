# Speech-MASSIVE Experiments

Scripts for training and evaluating on the [Speech-MASSIVE](https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE) dataset.

## Setup

```bash
# Download and verify the dataset (default: en-US)
bash experiments/SpeechMassive/setup_data.sh en-US

# Other languages: fr-FR, de-DE, ja-JP, etc.
bash experiments/SpeechMassive/setup_data.sh ja-JP
```

## Training

```bash
# Text-Only Training (English)
bash experiments/SpeechMassive/train_text_only.sh en-US

# Audio + Text Mixed Training (English)
bash experiments/SpeechMassive/train_audio_text_mix.sh en-US
```

## Evaluation

```bash
# Text-Only Evaluation
bash experiments/SpeechMassive/evaluate_text.sh outputs/speech_massive_text_only en-US

# Audio Evaluation
bash experiments/SpeechMassive/evaluate_audio.sh outputs/speech_massive_audio_text_mix en-US
```

## Available Languages

The Speech-MASSIVE dataset supports multiple languages. Pass the language code as the first argument:

| Code | Language |
|------|----------|
| en-US | English |
| ja-JP | Japanese |
| fr-FR | French |
| de-DE | German |
| ... | (see HuggingFace for full list) |
