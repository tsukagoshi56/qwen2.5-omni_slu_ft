# Build `test_FR.jsonl` (Speech-MASSIVE)

This folder provides a reproducible path to create a SLURP-style French test file:

- Input: Speech-MASSIVE test data (`fr-FR`, split `test`)
- Output: `test_FR.jsonl` usable as `--test_file` in this repository

## Files

- `build_test_fr_jsonl.py`: converter script
- `test_FR.jsonl`: generated output (create it with commands below)

## Option A: Build from local parquet files (zip extracted)

If you downloaded/extracted dataset files manually, run:

```bash
python Experiment_RationaleCompare/massive_test_fr/build_test_fr_jsonl.py \
  --source_dir /path/to/Speech-MASSIVE-test \
  --config fr-FR \
  --split test \
  --output_file Experiment_RationaleCompare/massive_test_fr/test_FR.jsonl
```

Expected parquet location pattern:

```text
/path/to/Speech-MASSIVE-test/fr-FR/test-*.parquet
```

## Option B: Build directly from HF dataset repo

```bash
python Experiment_RationaleCompare/massive_test_fr/build_test_fr_jsonl.py \
  --dataset_name FBK-MT/Speech-MASSIVE-test \
  --config fr-FR \
  --split test \
  --output_file Experiment_RationaleCompare/massive_test_fr/test_FR.jsonl
```

If your server must use cached files only:

```bash
python Experiment_RationaleCompare/massive_test_fr/build_test_fr_jsonl.py \
  --dataset_name FBK-MT/Speech-MASSIVE-test \
  --config fr-FR \
  --split test \
  --local_files_only \
  --cache_dir ~/.cache/huggingface/datasets \
  --output_file Experiment_RationaleCompare/massive_test_fr/test_FR.jsonl
```

## Quick checks

```bash
wc -l Experiment_RationaleCompare/massive_test_fr/test_FR.jsonl
head -n 2 Experiment_RationaleCompare/massive_test_fr/test_FR.jsonl
```

Confirm each row has at least:

- `slurp_id`
- `scenario`, `action`
- `tokens`, `entities`
- `recordings`

## Example usage in this repo

```bash
python Experiment_RationaleCompare/08_visualize_model_features.py \
  --pipeline multitask \
  --test_file Experiment_RationaleCompare/massive_test_fr/test_FR.jsonl \
  --audio_dir /path/to/audio_root \
  --model_name_or_path /path/to/checkpoint
```

Notes:

- `recordings.file` is taken from dataset audio path by default.
- Use `--recording_path_mode basename` if you prefer filename-only entries.
