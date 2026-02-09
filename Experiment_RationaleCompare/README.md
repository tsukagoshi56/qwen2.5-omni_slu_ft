# Rationale Comparison Pipeline (OR/SF/RD/Vanilla)

This folder implements the comparison pipeline for four methods:

- **Method 1: Vanilla SFT** (no CoT)
- **Method 2: Oracle Rationalization (OR‑CoT)** (gold text + gold JSON + DB definitions)
- **Method 3: Success‑Filtered Bootstrap (SF‑CoT)** (self‑generated, keep only correct)
- **Method 4: Reinforced Discovery (RD‑CoT)** (GRPO starting from SF‑CoT)

Key constraints (as requested):
- **Text inference** uses *gold transcript only*.
- **Audio inference** uses *audio only* (no transcript).
- **Test is audio‑only**.

All DB definitions are built from `Experiment_3/slurp_metadata.json`.

---

## Environment

Default API is DeepSeek (compatible with the existing scripts):

```bash
export DEEPSEEK_API_KEY=...
# Optional (priority: API_ENDPOINT > DEEPSEEK_BASE_URL > default)
export API_ENDPOINT=https://api.deepseek.com
# export DEEPSEEK_BASE_URL=https://api.deepseek.com
```

---

## 1) Oracle CoT generation (Method 2)

```bash
python Experiment_RationaleCompare/01_generate_oracle_cot.py \
  --input_file slurp/dataset/slurp/train.jsonl \
  --metadata_file Experiment_3/slurp_metadata.json \
  --output_file Experiment_RationaleCompare/oracle_cot.jsonl \
  --model_name deepseek-r1
```

- **Input**: gold transcript + gold JSON + DB definitions
- **Output**: `rationale_text=C/R/J`, `final=gold JSON`

---

## 2) Success‑Filtered CoT generation (Method 3)

```bash
python Experiment_RationaleCompare/02_generate_success_cot.py \
  --input_file slurp/dataset/slurp/train.jsonl \
  --metadata_file Experiment_3/slurp_metadata.json \
  --audio_dir slurp/slurp_real \
  --output_file Experiment_RationaleCompare/success_cot_raw.jsonl \
  --filtered_file Experiment_RationaleCompare/success_cot_filtered.jsonl \
  --modes text
```

- **Text mode**: uses gold transcript only (DeepSeek API default, same decoding defaults as 01)
- **Audio mode**: uses audio only (local Qwen2Audio). Enable with `--modes text,audio`.
- **Filtered** file keeps only correct predictions (configurable via `--success_match`)

By default, **text‑mode filtered samples omit `recordings`** to force text‑only inputs in SFT.

---

## 3) Prepare SFT JSONL

### Vanilla SFT (Method 1)
```bash
python Experiment_RationaleCompare/03_prepare_sft_jsonl.py \
  --input_files slurp/dataset/slurp/train.jsonl \
  --output_file Experiment_RationaleCompare/sft_vanilla_train.jsonl \
  --method vanilla
```

### Oracle SFT (Method 2)
```bash
python Experiment_RationaleCompare/03_prepare_sft_jsonl.py \
  --input_files Experiment_RationaleCompare/oracle_cot.jsonl \
  --output_file Experiment_RationaleCompare/sft_oracle_train.jsonl
```

### Success‑Filtered SFT (Method 3)
```bash
python Experiment_RationaleCompare/03_prepare_sft_jsonl.py \
  --input_files Experiment_RationaleCompare/success_cot_filtered.jsonl \
  --output_file Experiment_RationaleCompare/sft_success_train.jsonl
```

### Success‑CoT SFT (keep all outputs)
```bash
python Experiment_RationaleCompare/03_prepare_sft_jsonl.py \
  --input_files Experiment_RationaleCompare/success_cot_raw.jsonl \
  --output_file Experiment_RationaleCompare/sft_success_train.jsonl \
  --method sf-cot
```

`success_cot_raw.jsonl` includes `gold_label`; `03_prepare_sft_jsonl.py` will use it as `final` automatically.

---

## 4) SFT Training (audio_text_mix_e2e_re.py)

Use the existing trainer (audio + text mixed). Example:

```bash
python Experiment_RationaleFT/audio_text_mix_e2e_re.py \
  --train_file Experiment_RationaleCompare/sft_success_train.jsonl \
  --eval_file  Experiment_RationaleCompare/sft_success_train.jsonl \
  --test_file  slurp/dataset/slurp/test.jsonl \
  --audio_dir  slurp/slurp_real \
  --output_file outputs/qwen_rationale_label_ft/prediction.jsonl \
  --add_text_only
```

- `--add_text_only` lets the trainer include a text‑only copy of each audio sample.
- For **SF‑CoT**, the input file already contains both audio and text patterns.
- Training log is saved by default to `<output_dir>/train.log` (override with `--log_file`).

---

## 5) GRPO (Method 4)

```bash
python Experiment_RationaleCompare/04_run_grpo.py \
  --train_file Experiment_RationaleCompare/sft_success_train.jsonl \
  --metadata_file Experiment_3/slurp_metadata.json \
  --audio_dir slurp/slurp_real \
  --model_name_or_path outputs/qwen_rationale_label_ft \
  --ref_model_name_or_path outputs/qwen_rationale_label_ft \
  --output_dir outputs/grpo \
  --include_text \
  --group_size 4 \
  --max_new_tokens 4096 \
  --kl_beta 0.01
```

- Generates `group_size` samples per prompt and applies **GRPO** (group‑relative advantage).
- Reward is computed from parsed `J:` label (scenario/action/entities).
- KL penalty uses a frozen reference model.

---

## 6) Evaluation

Use the existing evaluation script (compatible with predictions from `audio_text_mix_e2e_re.py`):

```bash
python scripts/evaluation/evaluate.py \
  -g slurp/dataset/slurp/test.jsonl \
  -p outputs/qwen_rationale_label_ft/prediction.jsonl
```

---

## 7) GRPO Hyperparameter Sweep (Auto)

Use `05_sweep_grpo.py` to run many `04_run_grpo.py` jobs and rank them automatically.

### Smoke sweep (recommended first)
```bash
python Experiment_RationaleCompare/05_sweep_grpo.py \
  --config Experiment_RationaleCompare/sweep_configs/grpo_smoke_grid.json \
  --nproc_per_node 2
```

- Output root: `outputs/grpo_sweeps/<name_timestamp>/`
- Each run has its own `run.log` and output dir.
- Sweep summary is saved to:
  - `summary.json`
  - `summary.jsonl`
- Ranking is printed by `rank_by` (e.g., `test_intent_acc`).

### Small full sweep
```bash
python Experiment_RationaleCompare/05_sweep_grpo.py \
  --config Experiment_RationaleCompare/sweep_configs/grpo_full_small.json \
  --nproc_per_node 2
```

---

## Notes

- All scripts here are **generation/prep only**. Do **not** run them locally if the data/model is on the server.
- Audio inference always uses **audio only** (no transcript).
- Text inference always uses **gold transcript only**.
- API errors **fail fast** to avoid burning keys (no retries). There is a per-worker rate limit of `0.2s` after each API call.
