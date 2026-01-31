# Experiment: Rationale-based SLU

This experiment tests if generating a "rationale" can improve Spoken Language Understanding (SLU) accuracy, especially when dealing with ambiguous ASR results.

The process involves generating n-best ASR hypotheses from audio and then using a model to reason about these hypotheses.

## Experimental Workflow

### Step 1: Run ASR to Generate N-best Hypotheses

This script runs the `Qwen2-Audio` model to perform ASR on the audio files you provide. It uses the `recordings[].file` field from the SLURP jsonl and resolves it under `--audio_dir` (e.g., `slurp/slurp_real`). It generates a JSONL file containing the n-best transcription hypotheses for each audio file. You can select which recording to use with `--recording_index` (default: 0). For reproducibility, the script uses an explicit ASR prompt (`--prompt_text`) and a fixed seed (`--seed`). The default prompt explicitly asks for verbatim transcription only, and the system prompt (`--system_prompt`) is set to suppress instruction-following in the audio. Assistant-like replies are filtered by default; disable with `--no_filter_assistant_phrases` if needed. By default it restricts to 2 GPUs via `--cuda_devices 0,1`.

**Command:**
Run from the repository root (e.g., `qwen2 2.5-omni_slu_ft`). Errors are printed to the console by default.

```bash
uv run Experiment_Rationale/2_run_asr_on_audio.py
```

**Quick test (process only 10 files):**
```bash
uv run Experiment_Rationale/2_run_asr_on_audio.py --limit 10
```

### Step 2: Rationale Generation and SLU (Next Step)

*(This step is to be implemented after successfully generating the ASR hypotheses data.)*
