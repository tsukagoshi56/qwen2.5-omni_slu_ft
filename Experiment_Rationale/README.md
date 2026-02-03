# Experiment: Rationale-based SLU

This experiment tests if generating a "rationale" can improve Spoken Language Understanding (SLU) accuracy, especially when dealing with ambiguous ASR results.

The process involves generating n-best ASR hypotheses from audio and then using a model to reason about these hypotheses.

## Experimental Workflow

### Step 1: Run ASR to Generate N-best Hypotheses

This script runs Whisper (`openai/whisper-large-v3-turbo` by default) to perform ASR on the audio files you provide. It uses the `recordings[].file` field from the SLURP jsonl and resolves it under `--audio_dir` (e.g., `slurp/audio/slurp_real`). Audio decoding uses `ffmpeg_read`, matching the official evaluation script. It generates a JSONL file containing the n-best transcription hypotheses for each audio file. You can select which recording to use with `--recording_index` (default: 0). Set `--language` to control Whisper's decoding language. For diversity, you can use `--diversity_method group_beam` (default) with `--diversity_penalty` or `--diversity_method sampling_pool --sampling_pool 100` to sample and select diverse n-best; if group beam is unavailable, the script can fall back to sampling_pool when provided.

**Command (Single GPU):**
Run from the repository root (e.g., `qwen2 2.5-omni_slu_ft`). Errors are printed to the console by default.

```bash
uv run Experiment_Rationale/2_run_asr_on_audio.py
```

**Multi-GPU Execution (Fast):**
To speed up inference, use `torchrun` to parallelize across multiple GPUs. The script automatically splits the workload and merges the results into a single file at the end.

```bash
# Example: Run on 4 GPUs
torchrun --nproc_per_node=4 Experiment_Rationale/2_run_asr_on_audio.py
```
*(Note: If using `uv`, you might need `uv run torchrun ...` or ensure torchrun is in your path)*

**Quick test (process only 10 files):**
```bash
uv run Experiment_Rationale/2_run_asr_on_audio.py --limit 10
```

**Smoke test (run full Whisper with reduced samples):**
```bash
uv run Experiment_Rationale/2_run_asr_on_audio.py --smoke
```

### Step 2: Generate Rationale (DeepSeek API / Local Qwen2-Audio)

Use the rationale generator to connect gold labels with evidence from either n-best text or audio.
It outputs a JSONL file with short, structured rationales.

**DeepSeek API mode (default):**
`3_generate_rationale.py` uses DeepSeek API by default (`--local` is OFF).

```bash
export DEEPSEEK_API_KEY=your_api_key
# Optional (priority: API_ENDPOINT > DEEPSEEK_BASE_URL > default)
export API_ENDPOINT=https://api.deepseek.com
# export DEEPSEEK_BASE_URL=https://api.deepseek.com
```

**N-best mode with DeepSeek (default):**
```bash
uv run Experiment_Rationale/3_generate_rationale.py \
  --mode nbest \
  --model_name_or_path deepseek-r1 \
  --input_file Experiment_Rationale/real_asr_sampling_data.jsonl \
  --output_file Experiment_Rationale/rationale_output.jsonl
```

**Audio mode with DeepSeek (default):**
```bash
uv run Experiment_Rationale/3_generate_rationale.py \
  --mode audio \
  --model_name_or_path deepseek-r1 \
  --input_file Experiment_Rationale/real_asr_sampling_data.jsonl \
  --output_file Experiment_Rationale/rationale_output_audio.jsonl
```

**API parallel run (recommended for speed, no GPU required):**
`torchrun` launches multiple workers, and each worker sends API requests in parallel.
```bash
uv run torchrun --standalone --nproc_per_node=8 \
  Experiment_Rationale/3_generate_rationale.py \
  --mode nbest \
  --append_worker_suffix \
  --output_file Experiment_Rationale/rationale_output_parallel_api.jsonl

cat Experiment_Rationale/rationale_output_parallel_api.w*of8.jsonl \
  > Experiment_Rationale/rationale_output_parallel_api_merged.jsonl
```
Tip: If you hit API rate limits/timeouts, reduce `--nproc_per_node` (e.g., 8 -> 4).

**Local Qwen2-Audio mode (`--local`):**
```bash
uv run Experiment_Rationale/3_generate_rationale.py \
  --local \
  --mode nbest \
  --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda \
  --input_file Experiment_Rationale/real_asr_sampling_data.jsonl \
  --output_file Experiment_Rationale/rationale_output_local.jsonl
```

**N-best mode (text-only):**
```bash
uv run Experiment_Rationale/3_generate_rationale.py \
  --mode nbest \
  --input_file Experiment_Rationale/real_asr_sampling_data.jsonl \
  --output_file Experiment_Rationale/rationale_output.jsonl
```

**Audio mode (audio + gold labels):**
```bash
uv run Experiment_Rationale/3_generate_rationale.py \
  --mode audio \
  --input_file Experiment_Rationale/real_asr_sampling_data.jsonl \
  --output_file Experiment_Rationale/rationale_output_audio.jsonl
```

**Quick test (limit 10):**
```bash
uv run Experiment_Rationale/3_generate_rationale.py --mode nbest --limit 10
```

**Smoke test (run full rationale generation with reduced samples):**
```bash
uv run Experiment_Rationale/3_generate_rationale.py --mode nbest --smoke --output_mode full
```

**N-best ablation (k=1..5):**
```bash
uv run Experiment_Rationale/3_generate_rationale.py \
  --mode nbest \
  --ablation_1to5 \
  --output_file Experiment_Rationale/rationale_output_ablation.jsonl
```

This writes:
- `.../rationale_output_ablation.k1.jsonl`
- `.../rationale_output_ablation.k2.jsonl`
- `.../rationale_output_ablation.k3.jsonl`
- `.../rationale_output_ablation.k4.jsonl`
- `.../rationale_output_ablation.k5.jsonl`

You can also pass custom values with:
```bash
uv run Experiment_Rationale/3_generate_rationale.py --mode nbest --nbest_values 1,3,5
```

**Preview first 10 prompts/outputs:**
```bash
uv run Experiment_Rationale/3_generate_rationale.py --mode nbest --preview 10
```

**Limit mode (pretty JSON to stdout):**
```bash
uv run Experiment_Rationale/3_generate_rationale.py --mode nbest --limit 10 --limitmode
```

**2GPU parallel run (2 workers):**
```bash
CUDA_VISIBLE_DEVICES=0 uv run Experiment_Rationale/3_generate_rationale.py \
  --mode nbest --device cuda --num_workers 2 --worker_rank 0 --append_worker_suffix \
  --output_file Experiment_Rationale/rationale_output_parallel.jsonl &

CUDA_VISIBLE_DEVICES=1 uv run Experiment_Rationale/3_generate_rationale.py \
  --mode nbest --device cuda --num_workers 2 --worker_rank 1 --append_worker_suffix \
  --output_file Experiment_Rationale/rationale_output_parallel.jsonl &

wait
cat Experiment_Rationale/rationale_output_parallel.w0of2.jsonl \
    Experiment_Rationale/rationale_output_parallel.w1of2.jsonl \
    > Experiment_Rationale/rationale_output_parallel_merged.jsonl
```

**2GPU parallel run with `uv run torchrun`:**
```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  Experiment_Rationale/3_generate_rationale.py \
  --mode nbest --device cuda --append_worker_suffix \
  --output_file Experiment_Rationale/rationale_output_parallel.jsonl

cat Experiment_Rationale/rationale_output_parallel.w0of2.jsonl \
    Experiment_Rationale/rationale_output_parallel.w1of2.jsonl \
    > Experiment_Rationale/rationale_output_parallel_merged.jsonl
```

Note: The prompt includes a small few-shot example by default to encourage concise rejection reasons.
Output JSONL defaults to raw model outputs; use `--output_mode full` for metadata JSON.
`allowed_slot_types` is now passed as the full slot-type inventory (not sampled).
Intent labels are represented as `scenario_action` (underscore), e.g. `play_music`.

### Step 2-B: Generate Two-Stage Rationale (New)

`3_generate_rationale_two_stage.py` keeps the original script untouched and uses separated intent/slot processing:

- Stage 1-Intent (candidate generation):  
  Input = `n-best`/audio evidence + `reference_intent` + `intent_candidates`  
  Output = exactly 5 intent candidates (`topk_intents`)
- Stage 1-Slot (candidate generation):  
  Input = `n-best`/audio evidence + `reference_slot_types` + `allowed_slot_types`  
  Output = exactly 5 slot-type candidates (`topk_slot_types`)
- Stage 2-Intent (pruning):  
  Input = Stage1 `topk_intents` + evidence  
  Output = `intent_elimination`, `final_prediction`, `intent_rationalization`
- Stage 2-Slot (grounding):  
  Input = Stage1 `topk_slot_types` + evidence  
  Output = `slot_grounding`, `slot_rationalization`

The final JSON keeps the same top-level style as the original output and additionally stores both stage outputs under:
- `rationale.candidate_generation`
- `rationale.candidate_pruning`
and includes `rationale.topk_slot_types`.

**N-best mode (two-stage):**
```bash
uv run Experiment_Rationale/3_generate_rationale_two_stage.py \
  --mode nbest \
  --text_model_name_or_path Qwen/Qwen3-4B-Thinking-2507 \
  --input_file Experiment_Rationale/real_asr_sampling_data.jsonl \
  --output_file Experiment_Rationale/rationale_output_two_stage.jsonl
```

**Audio mode (two-stage):**
```bash
uv run Experiment_Rationale/3_generate_rationale_two_stage.py \
  --mode audio \
  --input_file Experiment_Rationale/real_asr_sampling_data.jsonl \
  --output_file Experiment_Rationale/rationale_output_two_stage_audio.jsonl
```

**Quick test (limit 10):**
```bash
uv run Experiment_Rationale/3_generate_rationale_two_stage.py --mode nbest --limit 10
```

**Preview prompts/outputs for all sub-stages:**
```bash
uv run Experiment_Rationale/3_generate_rationale_two_stage.py --mode nbest --preview 3
```

**2GPU parallel run (two-stage, 2 workers):**
```bash
CUDA_VISIBLE_DEVICES=0 uv run Experiment_Rationale/3_generate_rationale_two_stage.py \
  --mode nbest --device cuda --num_workers 2 --worker_rank 0 --append_worker_suffix \
  --output_file Experiment_Rationale/rationale_output_two_stage_parallel.jsonl &

CUDA_VISIBLE_DEVICES=1 uv run Experiment_Rationale/3_generate_rationale_two_stage.py \
  --mode nbest --device cuda --num_workers 2 --worker_rank 1 --append_worker_suffix \
  --output_file Experiment_Rationale/rationale_output_two_stage_parallel.jsonl &

wait
cat Experiment_Rationale/rationale_output_two_stage_parallel.w0of2.jsonl \
    Experiment_Rationale/rationale_output_two_stage_parallel.w1of2.jsonl \
    > Experiment_Rationale/rationale_output_two_stage_parallel_merged.jsonl
```

**2GPU parallel run with `uv run torchrun` (two-stage):**
```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  Experiment_Rationale/3_generate_rationale_two_stage.py \
  --mode nbest --device cuda --append_worker_suffix \
  --output_file Experiment_Rationale/rationale_output_two_stage_parallel.jsonl

cat Experiment_Rationale/rationale_output_two_stage_parallel.w0of2.jsonl \
    Experiment_Rationale/rationale_output_two_stage_parallel.w1of2.jsonl \
    > Experiment_Rationale/rationale_output_two_stage_parallel_merged.jsonl
```

**Notes:**
- Stage format checks and retry logic are applied independently to intent/slot stages.
- Parallel options are available in both scripts: `--num_workers`, `--worker_rank`, `--append_worker_suffix`.
- If `--append_worker_suffix` is not used, give each worker a different `--output_file`.
- In `--mode nbest`, two-stage script uses `--text_model_name_or_path` (default: `Qwen/Qwen3-4B-Thinking-2507`).
- In `--mode audio`, two-stage script uses `--model_name_or_path` (Qwen2-Audio family).
- `allowed_slot_types` uses the full slot-type inventory from metadata.
- If any sub-stage output is invalid after retries, deterministic fallback JSON is produced so processing can continue.

### Step 3: Fine-tune with Audio + N-best + Rationale + Gold Label

Use the rationale-conditioned trainer:

```bash
uv run Experiment_RationaleFT/audio_text_mix_e2e_re.py \
  --train_file /lustre/home/71200138/qwen_test/experiments/CoT_maker/ASR_cot_train.jsonl \
  --eval_file /lustre/home/71200138/qwen_test/experiments/CoT_maker/ASR_cot_devel.jsonl \
  --audio_dir /lustre/home/71200138/INTERSPEECH/experiment1/slurp/audio/slurp_real \
  --output_dir outputs/qwen_rationale_label_ft
```

2-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  Experiment_RationaleFT/audio_text_mix_e2e_re.py \
  --train_file /lustre/home/71200138/qwen_test/experiments/CoT_maker/ASR_cot_train.jsonl \
  --eval_file /lustre/home/71200138/qwen_test/experiments/CoT_maker/ASR_cot_devel.jsonl \
  --audio_dir /lustre/home/71200138/INTERSPEECH/experiment1/slurp/audio/slurp_real \
  --output_dir outputs/qwen_rationale_label_ft_2gpu \
  --batch_size 8
```

Notes:
- `--batch_size` is per GPU.
- Distributed prediction files are merged automatically on rank 0.

This script:
- uses audio input with prompt context from `candidates` + `rationale_text`
- trains on final SLU labels (`scenario`, `action`, `entities`)
- saves full predictions to `prediction.jsonl` (with n-best/rationale/raw output)
- evaluates using label-only extraction and saves:
  - `prediction_labels_only.jsonl`
  - `metrics_label_only.json`

### Step 3-B: Multi-task FT with `<ras>` + `<slu>` (New)

Use `audio_text_mix_e2e_multitask.py` to train two tasks together:
- `<ras>`: rationale text generation only
- `<slu>`: simple SLU JSON generation (`scenario`, `action`, `entities`)

It can also mix extra `<slu>` samples from gold text (`slurp train/devel`).

```bash
uv run Experiment_RationaleFT/audio_text_mix_e2e_multitask.py \
  --train_file /path/to/rationale_train.jsonl \
  --eval_file /path/to/rationale_eval.jsonl \
  --test_file slurp/dataset/slurp/test.jsonl \
  --gold_text_slu_file slurp/dataset/slurp/train.jsonl \
  --gold_text_slu_eval_file slurp/dataset/slurp/devel.jsonl \
  --audio_dir /path/to/slurp/audio/slurp_real \
  --output_dir outputs/qwen_multitask_ras_slu_ft
```

2-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  Experiment_RationaleFT/audio_text_mix_e2e_multitask.py \
  --train_file /path/to/rationale_train.jsonl \
  --eval_file /path/to/rationale_eval.jsonl \
  --test_file slurp/dataset/slurp/test.jsonl \
  --gold_text_slu_file slurp/dataset/slurp/train.jsonl \
  --gold_text_slu_eval_file slurp/dataset/slurp/devel.jsonl \
  --audio_dir /path/to/slurp/audio/slurp_real \
  --output_dir outputs/qwen_multitask_ras_slu_ft_2gpu \
  --batch_size 8
```

Useful options:
- `--disable_ras`: train only `<slu>`
- `--disable_rationale_slu`: keep `<ras>` + gold-text `<slu>`, but disable `<slu>` from rationale files
- `--gold_text_slu_limit`, `--gold_text_slu_eval_limit`: cap mixed gold-text SLU size
- `--train_audio_encoder`: enable audio encoder fine-tuning
- `--input_format {asr,ipa,arp}`: keep prompt context type compatible with the original script (default: `asr`)
