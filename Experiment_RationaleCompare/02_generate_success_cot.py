#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from common import (
    build_db_definitions,
    compare_labels,
    compute_reward,
    label_from_record,
    load_metadata,
    parse_j_from_output,
    read_jsonl,
    resolve_audio_path,
    write_jsonl,
)
from prompts import render_infer_audio_prompt, render_infer_text_prompt


_DEBUG = False


def _build_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package not available. Install it or use --text_local.")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("API_ENDPOINT") or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key, base_url=base_url)


def _call_api(client: Any, prompt: str, model_name: str, max_tokens: int, temperature: float, top_p: float) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return resp.choices[0].message.content or ""


def _merge_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("slurp_id", "")),
        str(row.get("mode", "")),
        str(row.get("method", "")),
    )


def _merge_worker_outputs(base_output_path: str, num_workers: int, cleanup: bool = False) -> int:
    root, ext = os.path.splitext(base_output_path)
    ext = ext or ".jsonl"
    merged: List[Dict[str, Any]] = []
    seen = set()
    for rank in range(num_workers):
        shard_path = f"{root}.w{rank}of{num_workers}{ext}"
        if not os.path.exists(shard_path):
            continue
        for row in read_jsonl(shard_path):
            key = _merge_key(row)
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
        if cleanup:
            try:
                os.remove(shard_path)
            except Exception:
                pass
    write_jsonl(base_output_path, merged)
    return len(merged)


def _strip_arg(args_list: List[str], flag: str, has_value: bool = True) -> List[str]:
    if flag not in args_list:
        return args_list
    cleaned: List[str] = []
    i = 0
    while i < len(args_list):
        if args_list[i] == flag:
            i += 1
            if has_value and i < len(args_list):
                i += 1
            continue
        cleaned.append(args_list[i])
        i += 1
    return cleaned


def _spawn_workers(num_workers: int, base_args: List[str]) -> List[subprocess.Popen]:
    procs: List[subprocess.Popen] = []
    base_args = _strip_arg(base_args, "--worker_rank", has_value=True)
    base_args = _strip_arg(base_args, "--merge_only", has_value=False)
    base_args = _strip_arg(base_args, "--no_spawn_workers", has_value=False)
    script_path = os.path.abspath(__file__)
    for rank in range(1, num_workers):
        cmd = [sys.executable, script_path] + base_args + ["--worker_rank", str(rank), "--no_spawn_workers"]
        procs.append(subprocess.Popen(cmd, env=os.environ.copy()))
    return procs


def _generate_audio_local(
    processor: AutoProcessor,
    model: Qwen2AudioForConditionalGeneration,
    audio_path: str,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    sr = processor.feature_extractor.sampling_rate
    audio, _ = librosa.load(audio_path, sr=sr)
    user_content = [
        {"type": "audio", "audio_url": "placeholder"},
        {"type": "text", "text": prompt},
    ]
    text_input = processor.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)


def _success_match_ok(match_mode: str, stats: Dict[str, Any]) -> bool:
    if match_mode == "intent":
        return bool(stats["intent_ok"])
    if match_mode == "scenario_action":
        return bool(stats["scenario_ok"] and stats["action_ok"])
    if match_mode == "full":
        return bool(stats["scenario_ok"] and stats["action_ok"] and stats["entity_f1"] == 1.0)
    return False


def _log_debug(message: str) -> None:
    if not _DEBUG:
        return
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Success-Filtered CoT (text and audio).")
    parser.add_argument("--input_file", type=str, default="slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--metadata_file", type=str, default="Experiment_3/slurp_metadata.json")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio/slurp_real")
    parser.add_argument("--output_file", type=str, default="Experiment_RationaleCompare/success_cot_raw.jsonl")
    parser.add_argument("--filtered_file", type=str, default="Experiment_RationaleCompare/success_cot_filtered.jsonl")
    parser.add_argument("--modes", type=str, default="text")
    parser.add_argument("--text_model_name", type=str, default="deepseekr1")
    parser.add_argument("--audio_model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recording_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_rank", type=int, default=0)
    parser.add_argument("--append_worker_suffix", action="store_true")
    parser.add_argument("--merge_workers", action="store_true", help="Merge worker shard outputs into output_file(s).")
    parser.add_argument("--merge_only", action="store_true", help="Only merge worker shard outputs and exit.")
    parser.add_argument("--merge_cleanup", action="store_true", help="Remove worker shard files after merge.")
    parser.add_argument("--no_spawn_workers", action="store_true", help="Do not auto-spawn worker processes.")
    parser.add_argument("--success_match", type=str, choices=["full", "scenario_action", "intent"], default="full")
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--debug", action="store_true", help="Print extra debug info.")
    parser.add_argument("--smoke", action="store_true", help="Process only 300 samples for debugging.")
    args = parser.parse_args()

    if str(args.text_model_name).strip() == "deepseekr1":
        args.text_model_name = "deepseek-r1"

    global _DEBUG
    _DEBUG = args.debug

    if args.num_workers > 1:
        # Always shard outputs and auto-merge when using multiple workers.
        args.append_worker_suffix = True
        args.merge_workers = True

    rng = random.Random(args.seed + args.worker_rank)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input_file) if not os.path.isabs(args.input_file) else args.input_file
    metadata_path = os.path.join(base_dir, args.metadata_file) if not os.path.isabs(args.metadata_file) else args.metadata_file
    audio_dir = os.path.join(base_dir, args.audio_dir) if not os.path.isabs(args.audio_dir) else args.audio_dir
    base_output_path = os.path.join(base_dir, args.output_file) if not os.path.isabs(args.output_file) else args.output_file
    base_filtered_path = os.path.join(base_dir, args.filtered_file) if not os.path.isabs(args.filtered_file) else args.filtered_file
    output_path = base_output_path
    filtered_path = base_filtered_path
    if args.append_worker_suffix and args.num_workers > 1:
        root, ext = os.path.splitext(base_output_path)
        output_path = f"{root}.w{args.worker_rank}of{args.num_workers}{ext or '.jsonl'}"
        root, ext = os.path.splitext(base_filtered_path)
        filtered_path = f"{root}.w{args.worker_rank}of{args.num_workers}{ext or '.jsonl'}"

    if args.merge_only:
        _merge_worker_outputs(base_output_path, args.num_workers, cleanup=args.merge_cleanup)
        _merge_worker_outputs(base_filtered_path, args.num_workers, cleanup=args.merge_cleanup)
        return

    worker_procs: List[subprocess.Popen] = []
    if (
        args.num_workers > 1
        and args.worker_rank == 0
        and (not args.no_spawn_workers)
        and (not args.merge_only)
    ):
        worker_procs = _spawn_workers(args.num_workers, sys.argv[1:])

    metadata = load_metadata(metadata_path)
    db_definitions = build_db_definitions(metadata)

    items = read_jsonl(input_path)
    if args.smoke:
        args.limit = 100
    if args.limit:
        items = items[: args.limit]
    if args.num_workers > 1:
        items = items[args.worker_rank :: args.num_workers]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    client = None
    if "text" in modes:
        client = _build_client()

    processor = None
    model = None
    device = torch.device(args.device)
    if "audio" in modes:
        processor = AutoProcessor.from_pretrained(args.audio_model_name_or_path, trust_remote_code=True)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.audio_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        model.eval()

    raw_rows: List[Dict[str, Any]] = []
    filtered_rows: List[Dict[str, Any]] = []

    for record in items:
        gold_text = str(record.get("sentence", "") or record.get("text", "") or "").strip()
        gold_label = label_from_record(record)
        recordings = record.get("recordings", []) if isinstance(record.get("recordings"), list) else []

        for mode in modes:
            output = ""
            if mode == "text":
                if not gold_text:
                    continue
                prompt = render_infer_text_prompt(db_definitions, gold_text)
                for attempt in range(args.retry + 1):
                    try:
                        output = _call_api(
                            client=client,
                            prompt=prompt,
                            model_name=args.text_model_name,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                        time.sleep(0.2)  # per-worker rate limit (~5 req/sec)
                        break
                    except Exception:
                        # Fail fast on any API error to avoid burning the key.
                        raise
            elif mode == "audio":
                if not recordings:
                    continue
                rec = recordings[args.recording_index] if args.recording_index < len(recordings) else recordings[0]
                filename = rec.get("file") if isinstance(rec, dict) else None
                if not filename:
                    continue
                audio_path = resolve_audio_path(audio_dir, filename)
                if not audio_path:
                    continue
                prompt = render_infer_audio_prompt(db_definitions)
                output = _generate_audio_local(
                    processor=processor,
                    model=model,
                    audio_path=audio_path,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                )
            else:
                continue

            pred_obj = parse_j_from_output(output) or {}
            pred_label = {
                "scenario": str(pred_obj.get("scenario", "")).strip(),
                "action": str(pred_obj.get("action", "")).strip(),
                "entities": pred_obj.get("entities", []) if isinstance(pred_obj.get("entities", []), list) else [],
            }
            stats = compare_labels(pred_label, gold_label)
            is_ok = _success_match_ok(args.success_match, stats)
            reward, _ = compute_reward(pred_label, gold_label)

            if args.debug:
                word_count = len((output or "").split())
                _log_debug(
                    f"[DEBUG] slurp_id={record.get('slurp_id')} mode={mode} words={word_count} "
                    f"correct={bool(is_ok)} reward={reward:.3f}"
                )
                _log_debug("[DEBUG] raw_output:")
                _log_debug(output or "")

            raw_row = {
                "slurp_id": record.get("slurp_id"),
                "sentence": gold_text,
                "recordings": recordings,
                "mode": mode,
                "method": "sf-cot",
                "rationale_text": output.strip(),
                "pred_label": pred_label,
                "gold_label": gold_label,
                "correct": bool(is_ok),
                "reward": reward,
            }
            raw_rows.append(raw_row)

            if is_ok:
                filtered_row = {
                    "slurp_id": record.get("slurp_id"),
                    "sentence": gold_text,
                    "final": gold_label,
                    "rationale_text": output.strip(),
                    "mode": mode,
                    "method": "sf-cot",
                }
                if mode == "audio":
                    filtered_row["recordings"] = recordings
                filtered_rows.append(filtered_row)

    write_jsonl(output_path, raw_rows)
    write_jsonl(filtered_path, filtered_rows)

    for proc in worker_procs:
        proc.wait()

    if args.merge_workers and args.num_workers > 1 and args.append_worker_suffix and args.worker_rank == 0:
        _merge_worker_outputs(base_output_path, args.num_workers, cleanup=args.merge_cleanup)
        _merge_worker_outputs(base_filtered_path, args.num_workers, cleanup=args.merge_cleanup)


if __name__ == "__main__":
    main()
