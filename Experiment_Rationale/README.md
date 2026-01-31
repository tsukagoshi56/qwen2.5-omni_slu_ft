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
