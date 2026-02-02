# Experiment: Rationale-based SLU

This experiment tests if generating a "rationale" can improve Spoken Language Understanding (SLU) accuracy, especially when dealing with ambiguous ASR results.

The process involves generating n-best ASR hypotheses from audio and then using a model to reason about these hypotheses.

## Experimental Workflow

### Step 1: Run ASR to Generate N-best Hypotheses

This script runs Whisper (`openai/whisper-large-v3-turbo` by default) to perform ASR on the audio files you provide. It uses the `recordings[].file` field from the SLURP jsonl and resolves it under `--audio_dir` (e.g., `slurp/slurp_real`). Audio decoding uses `ffmpeg_read`, matching the official evaluation script. It generates a JSONL file containing the n-best transcription hypotheses for each audio file. You can select which recording to use with `--recording_index` (default: 0). Set `--language` to control Whisper's decoding language. For diversity, you can use `--diversity_method group_beam` (default) with `--diversity_penalty` or `--diversity_method sampling_pool --sampling_pool 100` to sample and select diverse n-best; if group beam is unavailable, the script can fall back to sampling_pool when provided. By default it restricts to 2 GPUs via `--cuda_devices 0,1`.

**Command:**
Run from the repository root (e.g., `qwen2 2.5-omni_slu_ft`). Errors are printed to the console by default.

```bash
uv run Experiment_Rationale/2_run_asr_on_audio.py
```

**Quick test (process only 10 files):**
```bash
uv run Experiment_Rationale/2_run_asr_on_audio.py --limit 10
```

### Step 2: Generate Rationale with Qwen2-Audio

Use the rationale generator to connect gold labels with evidence from either n-best text or audio.
It outputs a JSONL file with short, structured rationales.

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

**Notes:**
- Stage format checks and retry logic are applied independently to intent/slot stages.
- Parallel options are available in both scripts: `--num_workers`, `--worker_rank`, `--append_worker_suffix`.
- If `--append_worker_suffix` is not used, give each worker a different `--output_file`.
- `allowed_slot_types` uses the full slot-type inventory from metadata.
- If any sub-stage output is invalid after retries, deterministic fallback JSON is produced so processing can continue.
