#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Success-Filtered CoT (text and audio).")
    parser.add_argument("--input_file", type=str, default="slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--metadata_file", type=str, default="Experiment_3/slurp_metadata.json")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio/slurp_real")
    parser.add_argument("--output_file", type=str, default="Experiment_RationaleCompare/success_cot_raw.jsonl")
    parser.add_argument("--filtered_file", type=str, default="Experiment_RationaleCompare/success_cot_filtered.jsonl")
    parser.add_argument("--modes", type=str, default="text,audio")
    parser.add_argument("--text_model_name", type=str, default="deepseek-r1")
    parser.add_argument("--audio_model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recording_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_rank", type=int, default=0)
    parser.add_argument("--append_worker_suffix", action="store_true")
    parser.add_argument("--success_match", type=str, choices=["full", "scenario_action", "intent"], default="full")
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    args = parser.parse_args()

    rng = random.Random(args.seed + args.worker_rank)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input_file) if not os.path.isabs(args.input_file) else args.input_file
    metadata_path = os.path.join(base_dir, args.metadata_file) if not os.path.isabs(args.metadata_file) else args.metadata_file
    audio_dir = os.path.join(base_dir, args.audio_dir) if not os.path.isabs(args.audio_dir) else args.audio_dir
    output_path = os.path.join(base_dir, args.output_file) if not os.path.isabs(args.output_file) else args.output_file
    filtered_path = os.path.join(base_dir, args.filtered_file) if not os.path.isabs(args.filtered_file) else args.filtered_file
    if args.append_worker_suffix and args.num_workers > 1:
        root, ext = os.path.splitext(output_path)
        output_path = f"{root}.w{args.worker_rank}of{args.num_workers}{ext or '.jsonl'}"
        root, ext = os.path.splitext(filtered_path)
        filtered_path = f"{root}.w{args.worker_rank}of{args.num_workers}{ext or '.jsonl'}"

    metadata = load_metadata(metadata_path)
    db_definitions = build_db_definitions(metadata)

    items = read_jsonl(input_path)
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
                        break
                    except Exception:
                        if attempt >= args.retry:
                            raise
                        time.sleep(args.retry_sleep)
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


if __name__ == "__main__":
    main()
