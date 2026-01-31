# Experiment: Rationale-based SLU

This experiment tests if generating a "rationale" can improve Spoken Language Understanding (SLU) accuracy, especially when dealing with ambiguous ASR results.

The process involves generating n-best ASR hypotheses from audio and then using a model to reason about these hypotheses.

## Experimental Workflow

### Step 1: Run ASR to Generate N-best Hypotheses

This script runs the `Qwen2-Audio` model to perform ASR on the audio files you provide. It uses the `recordings[].file` field from the SLURP jsonl and resolves it under `--audio_dir` (e.g., `slurp/slurp_real`). It generates a JSONL file containing the n-best transcription hypotheses for each audio file. You can select which recording to use with `--recording_index` (default: 0).

**Command:**
To execute and save all output and errors to log files, run the following command. If an error occurs, it will be saved in `asr_error.log`.

```bash
uv run qwen2-2.5-omni_slu_ft/Experiment_Rationale/2_run_asr_on_audio.py > asr_output.log 2> asr_error.log
```

**Quick test (process only 10 files):**
```bash
uv run qwen2-2.5-omni_slu_ft/Experiment_Rationale/2_run_asr_on_audio.py --limit 10 > asr_output.log 2> asr_error.log
```

### Step 2: Rationale Generation and SLU (Next Step)

*(This step is to be implemented after successfully generating the ASR hypotheses data.)*
