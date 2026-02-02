# Acoustic-Contrastive Rationale Distillation (SLU)

This folder implements the pipeline for:

- teacher-centric acoustic analysis with IPA (or IPA-aligned fallback),
- contrastive rationale generation without exposing IPA symbols,
- one-pass student distillation: `[Audio] -> [Natural Language Rationale] -> [Intent]`,
- boundary-focused evaluation (`RR_hier`, `BCS`, rationale faithfulness proxy).

## Step-by-step

Run from repository root: `qwen2 2.5-omni_slu_ft`.

### 1) Qwen3-Omni (vLLM) inference check with beam-search n-best

```bash
uv run Experiment_AcousticContrastiveDistill/1_prepare_contrastive_pairs.py \
  --api_base http://localhost:8000/v1 \
  --model Qwen/Qwen3-Omni-7B \
  --slurp_jsonl slurp/dataset/slurp/test.jsonl \
  --audio_dir slurp/slurp_real \
  --num_beams 5 \
  --nbest 5 \
  --limit 10
```

Outputs:
- `Experiment_AcousticContrastiveDistill/outputs/01_qwen3_omni_vllm_check.jsonl`

### 2) Install IPA corpus + compare IPA/ASR accuracy (both beam-search n-best)

This script downloads the corpus via Hugging Face `datasets.load_dataset(...)`, then:
- Whisper ASR n-best (beam search),
- Qwen3-Omni IPA+transcript n-best (beam search via vLLM),
- reports WER/PER and gap analysis.

```bash
uv run Experiment_AcousticContrastiveDistill/2_extract_or_align_ipa.py \
  --hf_dataset YOUR_IPA_CORPUS_NAME \
  --hf_split test \
  --audio_column audio \
  --text_column sentence \
  --ipa_column ipa \
  --omni_api_base http://localhost:8000/v1 \
  --omni_model Qwen/Qwen3-Omni-7B \
  --omni_num_beams 5 \
  --omni_nbest 5 \
  --asr_num_beams 5 \
  --asr_nbest 5
```

Outputs:
- `Experiment_AcousticContrastiveDistill/outputs/02_ipa_corpus_eval_samples.jsonl`
- `Experiment_AcousticContrastiveDistill/outputs/02_ipa_corpus_eval_report.json`

### 3) Generate acoustic-contrastive rationale

Template mode (no API):

```bash
uv run Experiment_AcousticContrastiveDistill/3_generate_acoustic_contrastive_rationale.py \
  --backend template
```

OpenAI-compatible API mode (e.g., DeepSeek-R1 endpoint):

```bash
OPENAI_API_KEY=... uv run Experiment_AcousticContrastiveDistill/3_generate_acoustic_contrastive_rationale.py \
  --backend openai_compatible \
  --api_base https://api.openai.com/v1 \
  --model deepseek-r1
```

Output:
- `Experiment_AcousticContrastiveDistill/outputs/03_teacher_rationales.jsonl`

### 4) Build distillation train/eval dataset

```bash
uv run Experiment_AcousticContrastiveDistill/4_build_distillation_dataset.py
```

Outputs:
- `Experiment_AcousticContrastiveDistill/outputs/04_distill_train.jsonl`
- `Experiment_AcousticContrastiveDistill/outputs/04_distill_eval.jsonl`

### 5) Train and predict with student ALM

Train:

```bash
uv run Experiment_AcousticContrastiveDistill/5_train_student_alm.py train \
  --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
  --output_dir outputs/acoustic_contrastive_student
```

Predict on eval split:

```bash
uv run Experiment_AcousticContrastiveDistill/5_train_student_alm.py predict \
  --model_name_or_path outputs/acoustic_contrastive_student \
  --input_file Experiment_AcousticContrastiveDistill/outputs/04_distill_eval.jsonl \
  --output_file Experiment_AcousticContrastiveDistill/outputs/05_student_predictions.jsonl
```

### 6) Evaluate boundary metrics

```bash
uv run Experiment_AcousticContrastiveDistill/6_evaluate_boundary_metrics.py \
  --reference_file Experiment_AcousticContrastiveDistill/outputs/01_contrastive_pairs.jsonl \
  --student_pred_file Experiment_AcousticContrastiveDistill/outputs/05_student_predictions.jsonl \
  --baseline_pred_file path/to/baseline_predictions.jsonl
```

Output:
- `Experiment_AcousticContrastiveDistill/outputs/06_eval_report.json`

## Expected prediction schema

For evaluation, each prediction row should contain:

- `id` or `slurp_id`
- `pred_intent` (preferred), or parsable JSON text in `prediction_text`
- optional score map for BCS:
  - `intent_logprobs` (log-prob),
  - or `intent_scores` / `score_map` (prob or log-prob).
