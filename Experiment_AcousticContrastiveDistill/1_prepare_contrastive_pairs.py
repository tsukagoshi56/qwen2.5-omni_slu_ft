#!/usr/bin/env python3
import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from common import pick_recording, read_jsonl, resolve_audio_path, write_jsonl


def infer_audio_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".wav"]:
        return "wav"
    if ext in [".mp3"]:
        return "mp3"
    if ext in [".flac"]:
        return "flac"
    return "wav"


def encode_audio_base64(path: str) -> Tuple[str, str]:
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8"), infer_audio_format(path)


def build_messages(
    prompt: str,
    audio_b64: str,
    audio_format: str,
    audio_payload_mode: str,
) -> List[Dict[str, Any]]:
    if audio_payload_mode == "input_audio":
        content = [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": audio_format}},
        ]
    else:
        # Some vLLM deployments follow audio_url style.
        data_uri = f"data:audio/{audio_format};base64,{audio_b64}"
        content = [
            {"type": "text", "text": prompt},
            {"type": "audio_url", "audio_url": {"url": data_uri}},
        ]
    return [{"role": "user", "content": content}]


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = text[start : end + 1]
    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def call_vllm_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    nbest: int,
    num_beams: int,
    max_tokens: int,
    timeout_sec: int,
) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "n": nbest,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        # vLLM beam parameters (ignored by servers that do not support them).
        "use_beam_search": True,
        "best_of": max(num_beams, nbest),
        "length_penalty": 1.0,
        "top_p": 1.0,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def collect_tasks_from_slurp(
    slurp_jsonl: str,
    audio_dir: str,
    recording_index: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    rows = read_jsonl(slurp_jsonl)
    tasks: List[Dict[str, Any]] = []
    for row in rows:
        slurp_id = str(row.get("slurp_id", "")).strip()
        rec = pick_recording(row.get("recordings", []) or [], index=recording_index)
        if not rec:
            continue
        audio_path = resolve_audio_path(audio_dir, rec)
        if not audio_path:
            continue
        tasks.append(
            {
                "id": slurp_id or os.path.splitext(os.path.basename(audio_path))[0],
                "audio_path": audio_path,
                "reference_text": str(row.get("sentence", "")).strip(),
            }
        )
        if limit and len(tasks) >= limit:
            break
    return tasks


def collect_tasks_from_files(audio_files: List[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for path in audio_files:
        p = str(path).strip()
        if not p or not os.path.exists(p):
            continue
        tasks.append({"id": os.path.splitext(os.path.basename(p))[0], "audio_path": p, "reference_text": ""})
        if limit and len(tasks) >= limit:
            break
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step-1 check: verify Qwen3-Omni vLLM audio inference with beam-search n-best."
    )
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-7B")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--audio_payload_mode", type=str, default="input_audio", choices=["input_audio", "audio_url"])
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--nbest", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--timeout_sec", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--sleep_sec", type=float, default=0.2)
    parser.add_argument("--audio_files", type=str, nargs="*", default=[])
    parser.add_argument("--slurp_jsonl", type=str, default="slurp/dataset/slurp/test.jsonl")
    parser.add_argument("--audio_dir", type=str, default="slurp/slurp_real")
    parser.add_argument("--recording_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--output_file",
        type=str,
        default="Experiment_AcousticContrastiveDistill/outputs/01_qwen3_omni_vllm_check.jsonl",
    )
    parser.add_argument("--fail_fast", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    prompt = args.prompt.strip() or (
        "You are an expert in speech and phonetics.\n"
        "From the input audio, produce strict JSON only:\n"
        "{\"transcript\": \"...\", \"ipa\": \"...\"}\n"
        "Use broad-phonetic IPA for ipa."
    )

    tasks = collect_tasks_from_files(args.audio_files, args.limit)
    if not tasks:
        tasks = collect_tasks_from_slurp(
            slurp_jsonl=args.slurp_jsonl,
            audio_dir=args.audio_dir,
            recording_index=args.recording_index,
            limit=args.limit,
        )
    if not tasks:
        raise SystemExit("No audio tasks found. Provide --audio_files or valid --slurp_jsonl/--audio_dir.")

    results: List[Dict[str, Any]] = []
    for task in tasks:
        sample_id = task["id"]
        audio_path = task["audio_path"]
        audio_b64, audio_format = encode_audio_base64(audio_path)
        messages = build_messages(
            prompt=prompt,
            audio_b64=audio_b64,
            audio_format=audio_format,
            audio_payload_mode=args.audio_payload_mode,
        )

        response_obj: Dict[str, Any] = {}
        last_error = ""
        for _ in range(args.max_retries + 1):
            try:
                response_obj = call_vllm_chat_completion(
                    api_base=args.api_base,
                    api_key=api_key,
                    model=args.model,
                    messages=messages,
                    nbest=args.nbest,
                    num_beams=args.num_beams,
                    max_tokens=args.max_tokens,
                    timeout_sec=args.timeout_sec,
                )
                last_error = ""
                break
            except urllib.error.HTTPError as e:
                last_error = f"http_error_{e.code}"
            except Exception as e:
                last_error = str(e)
            time.sleep(args.sleep_sec)

        choices = response_obj.get("choices", []) if response_obj else []
        hypotheses: List[Dict[str, Any]] = []
        for i, ch in enumerate(choices):
            content = str(ch.get("message", {}).get("content", "")).strip()
            parsed = extract_json_object(content) or {}
            hypotheses.append(
                {
                    "rank": i + 1,
                    "raw_text": content,
                    "transcript": str(parsed.get("transcript", "")).strip(),
                    "ipa": str(parsed.get("ipa", "")).strip(),
                    "finish_reason": ch.get("finish_reason"),
                }
            )

        ok = bool(hypotheses)
        row = {
            "id": sample_id,
            "audio_path": audio_path,
            "reference_text": task.get("reference_text", ""),
            "model": args.model,
            "num_beams": args.num_beams,
            "nbest": args.nbest,
            "ok": ok,
            "error": "" if ok else last_error,
            "hypotheses": hypotheses,
        }
        results.append(row)
        print(f"[{'OK' if ok else 'NG'}] {sample_id} hyps={len(hypotheses)} {('err=' + last_error) if not ok else ''}")

        if args.fail_fast and not ok:
            break

    write_jsonl(args.output_file, results)
    ok_count = sum(1 for r in results if r.get("ok"))
    print(f"[DONE] success={ok_count}/{len(results)} -> {args.output_file}")


if __name__ == "__main__":
    main()

