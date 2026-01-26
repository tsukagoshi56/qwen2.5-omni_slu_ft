# Stable Version Walkthrough: Qwen2.5-SLURP Evaluation Fixes

This document summarizes the changes made to stabilize the evaluation and training pipeline for the Qwen2.5-Omni SLURP Fine-tuning project.

## 🚀 Key Fixes & Improvements

### 1. Robust Evaluation Metrics (`scripts/evaluation/metrics`)
- **Fixed `KeyError: type` / `KeyError: filler`**:
  - Modified `metrics.py` to transparently handle cases where the model generates entities missing "type" or "filler" keys.
  - Used `.get("type", "unknown")` instead of strict access.
- **Fixed `TypeError: unhashable type: 'list'`**:
  - Implemented aggressive string casting (`str(...)`) for entity types and fillers in `metrics.py`.
  - This prevents crashes when the model hallucinates lists (e.g., `["alarm"]`) instead of strings.
- **Added Debug Logging**:
  - Wrapped metric calculations in `try-except` blocks. If an unknown error occurs, it now prints the **exact conflicting data** (Gold vs Prediction) to the console before stopping/raising, aiding future debugging.
- **Fixed `wer()` argument error**:
  - Updated `distance.py` to use `reference=` instead of `truth=` to be compatible with modern `jiwer` versions.

### 2. Entity Format Compatibility (`run_eval.py` & `train_qwen2_audio_slurp.py`)
- **Automatic Format Conversion**:
  - Updated `run_eval.py` to automatically detect Compact Format entities (`[{"time": "7 am"}]`) produced by older checkpoints and convert them to Standard Format (`[{"type": "time", "filler": "7 am"}]`).
  - This ensures Entity F1 scores are calculated correctly (preventing 0.0 scores due to mismatch).
- **Training Target Alignment**:
  - Updated `train_qwen2_audio_slurp.py` so new models will natively generate Standard Format entities.
  - Updated the system `PROMPT` to explicitly request `{"type": ..., "filler": ...}` JSON structure.

### 3. Feature Extractor Loading (`run_eval.py`)
- **Robust Processor Loading**:
  - Fixed the `cannot load feature extractor` error common with checkpoints that lack `preprocessor_config.json`.
  - Added fallback logic:
    1. Try loading from checkpoint.
    2. If missing, read `config.json` → extract `_name_or_path` (Base Model).
    3. Load processor from the Base Model.

## 📌 Usage

To evaluate a model (even an old one):
```bash
python run_eval.py \
  --model_path outputs/your-checkpoint \
  --test_file slurp/dataset/slurp/test.jsonl \
  --device cuda
```
(Any format mismatch or missing config will be handled automatically.)

To train a new model:
```bash
python train_qwen2_audio_slurp.py ...
```
(It will now learn the correct output format from start.)
