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
- `--asr_transcript` を付けると、text入力に `asr_hypotheses[0].text`（1-best）を使います。

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
- Success scoring uses a single scalar:
  - `success_score = format_ok + match_ok + has_gold_intent_candidate + slot_candidate_coverage` (0.0 to 4.0)
  - `correct` is `success_score >= 4.0`
- `--asr_transcript` を付けると、text入力に `asr_hypotheses[0].text`（1-best）を使います。

By default, **text‑mode filtered samples omit `recordings`** to force text‑only inputs in SFT.

### Re-score from existing raw output

If you already generated `success_cot_raw.jsonl`, you can recompute success metrics (including intent/slot candidates) without re-calling models:

```bash
python Experiment_RationaleCompare/02_generate_success_cot.py \
  --rescore_raw_file Experiment_RationaleCompare/success_cot_raw.jsonl
```

- If `--output_file` / `--filtered_file` are omitted in rescore mode, outputs are auto-written as:
  - `<raw>.rescored.jsonl`
  - `<raw>.rescored.filtered.jsonl`
- 同時に「難しい順（低スコア順）」のファイルも出力されます:
  - `<raw>.rescored.hard_first.jsonl`
- 低スコア順ファイルの出力先は `--difficulty_file` で上書きできます。
- You can override paths explicitly with `--output_file` and `--filtered_file`.

---

## 3) Prepare SFT JSONL

> **Important**: The rationale generation outputs (`01_`, `02_`) do not include audio file
> references (`recordings`). Use `--slurp_files` to enrich records by looking up `recordings`
> from the original SLURP JSONL files via `slurp_id`.

### Vanilla SFT (Method 1)
```bash
python Experiment_RationaleCompare/03_prepare_sft_jsonl.py \
  --input_files slurp/dataset/slurp/train.jsonl \
  --output_file Experiment_RationaleCompare/sft_vanilla_train.jsonl \
  --method vanilla \
  --slurp_files slurp/dataset/slurp/train.jsonl,slurp/dataset/slurp/devel.jsonl,slurp/dataset/slurp/test.jsonl
```

### Oracle SFT (Method 2)
```bash
python Experiment_RationaleCompare/03_prepare_sft_jsonl.py \
  --input_files Experiment_RationaleCompare/oracle_cot.jsonl \
  --output_file Experiment_RationaleCompare/sft_oracle_train.jsonl \
  --slurp_files slurp/dataset/slurp/train.jsonl,slurp/dataset/slurp/devel.jsonl,slurp/dataset/slurp/test.jsonl
```

### Success‑Filtered SFT (Method 3)
```bash
python Experiment_RationaleCompare/03_prepare_sft_jsonl.py \
  --input_files Experiment_RationaleCompare/success_cot_filtered.jsonl \
  --output_file Experiment_RationaleCompare/sft_success_train.jsonl \
  --slurp_files slurp/dataset/slurp/train.jsonl,slurp/dataset/slurp/devel.jsonl,slurp/dataset/slurp/test.jsonl
```

### Success‑CoT SFT (keep all outputs)
```bash
python Experiment_RationaleCompare/03_prepare_sft_jsonl.py \
  --input_files Experiment_RationaleCompare/success_cot_raw.jsonl \
  --output_file Experiment_RationaleCompare/sft_success_train.jsonl \
  --method sf-cot \
  --slurp_files slurp/dataset/slurp/train.jsonl,slurp/dataset/slurp/devel.jsonl,slurp/dataset/slurp/test.jsonl
```

`success_cot_raw.jsonl` includes `gold_label`; `03_prepare_sft_jsonl.py` will use it as `final` automatically.

`--slurp_files` omission results in empty `recordings`, causing audio items to be skipped during training.

---

## 4) SFT Training (audio_text_mix_e2e_re.py)

Use the existing trainer (audio + text mixed). Example:

```bash
python Experiment_RationaleCompare/audio_text_mix_e2e_re.py \
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

### Multitask SFT (CoT + Label) (`audio_text_mix_e2e_re_multitask.py`)

This variant trains two tasks together:
- **CoT task**: `C/R/J` output
- **Label task**: `J`-only output
- Loss is fixed to **`0.5 * L_cot + 0.5 * L_label`**.

Use separate files when you want filtered CoT only for the CoT branch:

```bash
python Experiment_RationaleCompare/audio_text_mix_e2e_re_multitask.py \
  --train_file Experiment_RationaleCompare/sft_vanilla_train.jsonl \
  --cot_train_file Experiment_RationaleCompare/sft_success_train.jsonl \
  --eval_file Experiment_RationaleCompare/sft_vanilla_train.jsonl \
  --cot_eval_file Experiment_RationaleCompare/sft_success_train.jsonl \
  --test_file slurp/dataset/slurp/test.jsonl \
  --audio_dir slurp/slurp_real \
  --output_file outputs/qwen_rationale_label_ft_multitask/prediction.jsonl \
  --add_text_only
```

- `--train_file`: label branch training file (`J`-only task).
- `--cot_train_file`: CoT branch training file (`C/R/J` task).
- `--eval_file` / `--cot_eval_file`: same split logic for validation.
- If `--cot_train_file` is omitted, the script auto-creates both tasks from `--train_file`.
- Test inference remains standard prediction output (no multitask duplication at test time).
- For test-time J-only generation, use `--no_cot` (equivalent to `--test_task_mode label`).
- For test-time candidates+label generation (no rationale), use `--candidates_only` (equivalent to `--test_task_mode candidates`).

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
- Default prompt style is `C/R/J`. Add `--no-cot` to switch to a direct `J`-only output format.
- Add `--candidates_only` to switch to `C+J` output format (candidates + final label, no `R`).
- Add `--label_cot` to use the same prompt wording/style as `audio_text_mix_e2e_re_multitask.py`:
  - `System: Predict SLU labels from transcript/audio.`
  - `Output Format` matches multitask SFT style (`C/R/J`, `C+J`, or `J`-only).
- In `--no-cot` mode, the experiment keeps the same System/DB/Input context and changes only the output format (for controlled comparison).
- Early stopping is enabled by default: after at least 1 epoch, training stops if eval does not improve for 3 consecutive evaluations (`--early_stopping_metric intent_acc`).
- Override with `--early_stopping_patience`, `--early_stopping_min_epochs`, `--early_stopping_metric`, or disable via `--no-early-stopping`.

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
