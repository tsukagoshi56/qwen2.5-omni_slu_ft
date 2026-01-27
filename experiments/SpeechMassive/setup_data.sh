#!/bin/bash
set -e

# Speech-MASSIVE Data Setup
# Downloads and prepares the Speech-MASSIVE dataset from HuggingFace

# Configuration
LANG_CODE="${1:-en-US}"  # Default to English, can be changed (e.g., fr-FR, de-DE, ja-JP)
CACHE_DIR="${2:-~/.cache/huggingface/datasets}"

echo "============================================================"
echo " Speech-MASSIVE Data Setup"
echo "============================================================"
echo " Language: $LANG_CODE"
echo " Cache:    $CACHE_DIR"
echo "============================================================"

# Install required packages
pip install datasets soundfile

# Test dataset loading
python -c "
from datasets import load_dataset, Audio

print('Loading Speech-MASSIVE dataset...')
dataset = load_dataset(
    'FBK-MT/Speech-MASSIVE',
    '$LANG_CODE',
    split='train',
    cache_dir='$CACHE_DIR',
    trust_remote_code=True
)

print(f'Dataset loaded successfully!')
print(f'Number of examples: {len(dataset)}')
print(f'Features: {dataset.features}')
print(f'Sample: {dataset[0]}')
"

echo ""
echo "Data setup complete!"
echo "Available splits: train, validation, test"
