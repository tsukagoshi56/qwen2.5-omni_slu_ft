#!/bin/bash
set -e

# Speech-MASSIVE Data Setup
# Downloads and prepares the Speech-MASSIVE dataset from HuggingFace

# Configuration
LANG_CODE="${1:-fr-FR}"  # Default to French (en-US not available), can be changed
CACHE_DIR="${2:-~/.cache/huggingface/datasets}"

echo "============================================================"
echo " Speech-MASSIVE Data Setup"
echo "============================================================"
echo " Language: $LANG_CODE"
echo " Cache:    $CACHE_DIR"
echo "============================================================"

# Install required packages
# Install required packages
# NOTE: Dependencies are managed by uv in pyproject.toml
# pip install datasets soundfile

# Test dataset loading
uv run python -c "
from datasets import load_dataset, Audio

print('Loading Speech-MASSIVE dataset...')
dataset = load_dataset(
    'FBK-MT/Speech-MASSIVE',
    '$LANG_CODE',
    split='train',
    cache_dir='$CACHE_DIR',
    trust_remote_code=True
)
# Avoid automatic decoding to prevent torchcodec errors if not available
if 'audio' in dataset.column_names:
    dataset = dataset.cast_column('audio', Audio(decode=False))

print(f'Dataset loaded successfully!')
print(f'Number of examples: {len(dataset)}')
print(f'Features: {dataset.features}')
print(f'Sample: {dataset[0]}')
"

echo ""
echo "Data setup complete!"
echo "Available splits: train, validation, test"
