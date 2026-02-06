#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import threading
import time
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
except ImportError:
    ThreadPoolExecutor = None
    as_completed = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from common import (
    build_db_definitions,
    label_from_record,
    load_metadata,
    read_jsonl,
    write_jsonl,
)
from prompts import render_oracle_prompt


_thread_local = threading.local()
_error_lock = threading.Lock()
_DEBUG = False


def _canonicalize_model_name(model_name: str) -> str:
    value = str(model_name or "").strip()
    lower = value.lower()
    aliases = {
        "gpt4.1-mini": "gpt-4.1-mini",
        "gpt4.1": "gpt-4.1",
        "gpt4o-mini": "gpt-4o-mini",
        "gpt4o": "gpt-4o",
    }
    return aliases.get(lower, value)


def _is_deepseek_model(model_name: str) -> bool:
    return "deepseek" in str(model_name or "").strip().lower()


def _build_client(model_name: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package not available. Install it or use a local generator.")

    if _is_deepseek_model(model_name):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        base_url = os.environ.get("API_ENDPOINT") or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set.")
        return OpenAI(api_key=api_key, base_url=base_url)

    # Non-DeepSeek models: support OpenAI-compatible gateways (e.g., Bedrock proxy)
    # by allowing the same endpoint/key style used in DeepSeek mode.
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("BEDROCK_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
    )
    base_url = (
        os.environ.get("API_ENDPOINT")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
    )
    if not api_key:
        raise RuntimeError(
            "No API key found for non-DeepSeek model. "
            "Set one of OPENAI_API_KEY / BEDROCK_API_KEY / DEEPSEEK_API_KEY."
        )
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _get_thread_client(model_name: str) -> Any:
    client = getattr(_thread_local, "client", None)
    family = "deepseek" if _is_deepseek_model(model_name) else "openai"
    cached_family = getattr(_thread_local, "client_family", None)
    if client is None or cached_family != family:
        client = _build_client(model_name)
        _thread_local.client = client
        _thread_local.client_family = family
    return client


def _call_api(
    client: Any,
    prompt: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, Dict[str, Any]]:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    message = resp.choices[0].message
    content = message.content or ""
    reasoning_present = False
    if hasattr(message, "reasoning_content"):
        reasoning_present = bool((message.reasoning_content or "").strip())
    meta = {
        "model_requested": model_name,
        "model_responded": getattr(resp, "model", None),
        "response_id": getattr(resp, "id", None),
        "finish_reason": getattr(resp.choices[0], "finish_reason", None),
        "prompt_tokens": getattr(getattr(resp, "usage", None), "prompt_tokens", None),
        "completion_tokens": getattr(getattr(resp, "usage", None), "completion_tokens", None),
        "total_tokens": getattr(getattr(resp, "usage", None), "total_tokens", None),
        "reasoning_present": reasoning_present,
    }
    return content, meta


def _has_nonempty_c(output: str) -> bool:
    if not output:
        return False
    match = re.search(r"(?m)^\s*C:\s*(\S.+)$", output)
    return bool(match)


def _extract_crj(output: str) -> Tuple[str, bool]:
    if not output:
        return output, False
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    if not lines:
        return output, False

    # Find the last valid C->R->J sequence in order.
    c_idx = [i for i, ln in enumerate(lines) if re.match(r"^C:\s*", ln)]
    r_idx = [i for i, ln in enumerate(lines) if re.match(r"^R:\s*", ln)]
    j_idx = [i for i, ln in enumerate(lines) if re.match(r"^J:\s*", ln)]

    best = None
    for ci in c_idx:
        r_after = [ri for ri in r_idx if ri > ci]
        if not r_after:
            continue
        ri = r_after[0]
        j_after = [ji for ji in j_idx if ji > ri]
        if not j_after:
            continue
        ji = j_after[0]
        best = (ci, ri, ji)
    if best is None:
        return output, False

    ci, ri, ji = best
    c_line = lines[ci]
    r_line = lines[ri]
    j_line = lines[ji]
    return "\n".join([c_line, r_line, j_line]), True


def _normalize_intent(value: str) -> str:
    return value.replace(":", "_").strip()


def _reorder_candidates(values: List[str], order: List[str]) -> List[str]:
    order_map = {v: i for i, v in enumerate(order)}
    known = [v for v in values if v in order_map]
    unknown = [v for v in values if v not in order_map]
    known.sort(key=lambda v: order_map[v])
    return known + unknown


def _reorder_c_line(c_line: str, intent_order: List[str], slot_order: List[str]) -> str:
    if not c_line.startswith("C:"):
        return c_line
    content = c_line[2:].strip()
    parts = [p.strip() for p in content.split(";", 1)]
    intent_part = parts[0] if parts else ""
    slot_part = parts[1] if len(parts) > 1 else ""

    intents_raw = [x.strip() for x in intent_part.split("|") if x.strip()]
    intents_norm = [_normalize_intent(x) for x in intents_raw]
    intents_norm = _reorder_candidates(intents_norm, intent_order)
    intent_part_new = " | ".join(intents_norm) if intents_norm else intent_part

    slot_part_new = slot_part
    if slot_part:
        # If multiple slot types are present, reorder by slot_order.
        slot_chunks = [s.strip() for s in slot_part.split(",") if s.strip()]
        if len(slot_chunks) > 1:
            def slot_key(chunk: str) -> str:
                return chunk.split(":", 1)[0].strip()
            slot_chunks = _reorder_candidates(slot_chunks, slot_order=[s for s in slot_order])
            slot_part_new = ", ".join(slot_chunks)

    if slot_part_new:
        return f"C: {intent_part_new}; {slot_part_new}"
    return f"C: {intent_part_new}"


def _log_error(message: str) -> None:
    if not _DEBUG:
        return
    with _error_lock:
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message, flush=True)


def _append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
            sid = row.get("slurp_id")
            if sid in seen:
                continue
            seen.add(sid)
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


def _generate_with_retries(
    prompt: str,
    args: argparse.Namespace,
) -> Tuple[str, str, str, Dict[str, Any]]:
    last_exc: Optional[Exception] = None
    last_meta: Dict[str, Any] = {}
    for fmt_attempt in range(args.format_retries + 1):
        prompt_used = prompt
        if fmt_attempt > 0:
            prompt_used = (
                prompt
                + "\n\nReminder: Include a non-empty C line with at least one competing intent and one competing slot value."
            )
        output = ""
        for attempt in range(args.retry + 1):
            try:
                client = _get_thread_client(args.model_name)
                output, last_meta = _call_api(
                    client=client,
                    prompt=prompt_used,
                    model_name=args.model_name,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                time.sleep(0.2)  # per-worker rate limit (~5 req/sec)
                break
            except Exception as exc:
                last_exc = exc
                _log_error(
                    f"[ERROR] API call failed (attempt {attempt + 1}/{args.retry + 1}): {exc}"
                )
                # Fail fast on any API error to avoid burning the key.
                raise
        if not output:
            continue
        if args.skip_c_check or _has_nonempty_c(output):
            return output, prompt_used, "", last_meta
    if last_exc is not None and args.fail_on_empty:
        raise last_exc
    error_msg = "empty_output"
    if last_exc is not None:
        error_msg = f"api_error: {last_exc}"
    _log_error(f"[ERROR] Output empty after retries: {error_msg}")
    return output, prompt, error_msg, last_meta


def _run_single(
    idx: int,
    record: Dict[str, Any],
    db_definitions: str,
    intent_order: List[str],
    slot_order: List[str],
    args: argparse.Namespace,
) -> Optional[Tuple[int, Dict[str, Any], str, str, str]]:
    gold_text = str(record.get("sentence", "") or record.get("text", "") or "").strip()
    if not gold_text:
        return None
    gold_label = label_from_record(record)
    gold_json = json.dumps(gold_label, ensure_ascii=False)
    prompt = render_oracle_prompt(db_definitions, gold_text, gold_json)

    output, prompt_used, error_msg, api_meta = _generate_with_retries(prompt, args)
    cleaned, has_crj = _extract_crj(output)
    if has_crj:
        lines = cleaned.splitlines()
        if lines:
            lines[0] = _reorder_c_line(lines[0], intent_order, slot_order)
        output = "\n".join(lines)
    else:
        if not error_msg:
            error_msg = "missing_crj_lines"
    if args.debug:
        word_count = len((output or "").split())
        _log_error(
            f"[DEBUG] slurp_id={record.get('slurp_id')} words={word_count} tokens={api_meta}"
        )
        _log_error("[DEBUG] raw_output:")
        _log_error(output or "")

    result = {
        "slurp_id": record.get("slurp_id"),
        "sentence": gold_text,
        "recordings": record.get("recordings", []),
        "final": gold_label,
        "rationale_text": output.strip(),
        "method": "or-cot",
        "mode": "text",
        "model_name": args.model_name,
        "provider": "deepseek" if _is_deepseek_model(args.model_name) else "openai",
    }
    if error_msg:
        result["error"] = error_msg
    if api_meta:
        result["api_meta"] = api_meta
    return idx, result, prompt_used, gold_text, gold_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Oracle CoT rationales (gold text + gold JSON).")
    parser.add_argument("--input_file", type=str, default="slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--metadata_file", type=str, default="Experiment_3/slurp_metadata.json")
    parser.add_argument("--output_file", type=str, default="Experiment_RationaleCompare/oracle_cot.jsonl")
    parser.add_argument("--model_name", "--model", dest="model_name", type=str, default="deepseek-r1")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent API requests.")
    parser.add_argument("--preview", type=int, default=10, help="Print first N outputs to stdout.")
    parser.add_argument("--format_retries", type=int, default=0, help="Retry when C line is missing.")
    parser.add_argument("--skip_c_check", action="store_true", help="Do not enforce C line in output.")
    parser.add_argument("--fail_on_empty", action="store_true", help="Abort if output stays empty.")
    parser.add_argument("--error_file", type=str, default="", help="Optional jsonl to save error rows.")
    parser.add_argument("--save_every", type=int, default=50, help="Write partial outputs every N rows.")
    parser.add_argument("--resume", action="store_true", help="Skip slurp_id already in output_file.")
    parser.add_argument("--debug", action="store_true", help="Print extra debug info.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_rank", type=int, default=0)
    parser.add_argument("--append_worker_suffix", action="store_true")
    parser.add_argument("--merge_workers", action="store_true", help="Merge worker shard outputs into output_file.")
    parser.add_argument("--merge_only", action="store_true", help="Only merge worker shard outputs and exit.")
    parser.add_argument("--merge_cleanup", action="store_true", help="Remove worker shard files after merge.")
    parser.add_argument("--no_spawn_workers", action="store_true", help="Do not auto-spawn worker processes.")
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--smoke", action="store_true", help="Process only 300 samples for debugging.")
    args = parser.parse_args()
    args.model_name = _canonicalize_model_name(args.model_name)

    global _DEBUG
    _DEBUG = args.debug
    if args.debug:
        args.save_every = 0
        args.resume = False
    else:
        args.preview = 0

    if args.num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if args.worker_rank < 0 or args.worker_rank >= args.num_workers:
        raise ValueError("worker_rank out of range")

    if args.num_workers > 1:
        # Always shard outputs and auto-merge when using multiple workers.
        args.append_worker_suffix = True
        args.merge_workers = True

    if args.worker_rank == 0:
        provider = "deepseek" if _is_deepseek_model(args.model_name) else "openai"
        endpoint = (
            (os.environ.get("API_ENDPOINT") or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
            if provider == "deepseek"
            else (os.environ.get("API_ENDPOINT") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or "(default)")
        )
        print(
            f"[INFO] provider={provider} model={args.model_name} endpoint={endpoint} "
            f"(set by --model/--model_name)"
        )

    worker_procs: List[subprocess.Popen] = []
    if (
        args.num_workers > 1
        and args.worker_rank == 0
        and (not args.no_spawn_workers)
        and (not args.merge_only)
    ):
        worker_procs = _spawn_workers(args.num_workers, sys.argv[1:])

    rng = random.Random(args.seed + args.worker_rank)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input_file) if not os.path.isabs(args.input_file) else args.input_file
    metadata_path = os.path.join(base_dir, args.metadata_file) if not os.path.isabs(args.metadata_file) else args.metadata_file
    base_output_path = os.path.join(base_dir, args.output_file) if not os.path.isabs(args.output_file) else args.output_file
    output_path = base_output_path
    if args.append_worker_suffix and args.num_workers > 1:
        root, ext = os.path.splitext(base_output_path)
        output_path = f"{root}.w{args.worker_rank}of{args.num_workers}{ext or '.jsonl'}"

    if args.merge_only:
        _merge_worker_outputs(base_output_path, args.num_workers, cleanup=args.merge_cleanup)
        return

    metadata = load_metadata(metadata_path)
    db_definitions = build_db_definitions(metadata)
    intent_order = [
        _normalize_intent(str(x)) for x in metadata.get("intents", []) or [] if str(x).strip()
    ]
    slot_order = [str(x).strip() for x in metadata.get("slot_types", []) or [] if str(x).strip()]

    items = read_jsonl(input_path)
    if args.smoke:
        args.limit = 100
    if args.limit:
        items = items[: args.limit]
    if args.num_workers > 1:
        items = items[args.worker_rank :: args.num_workers]

    processed_ids = set()
    if args.resume and (not args.debug) and os.path.exists(output_path):
        try:
            for row in read_jsonl(output_path):
                processed_ids.add(row.get("slurp_id"))
        except Exception:
            processed_ids = set()
    if processed_ids:
        items = [it for it in items if it.get("slurp_id") not in processed_ids]

    results: List[Optional[Dict[str, Any]]] = [None] * len(items)
    error_rows: List[Dict[str, Any]] = []
    new_rows: List[Dict[str, Any]] = []
    preview_limit = max(0, int(args.preview))
    preview_lock = threading.Lock()
    preview_printed: List[bool] = [False] * len(items)

    if args.parallel > 1 and ThreadPoolExecutor is not None:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [
                executor.submit(_run_single, idx, record, db_definitions, intent_order, slot_order, args)
                for idx, record in enumerate(items)
            ]
            iterator = as_completed(futures) if as_completed is not None else futures
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="Oracle CoT", unit="sample")
            for fut in iterator:
                res = fut.result()
                if res is None:
                    continue
                idx, row, prompt, gold_text, gold_json = res
                results[idx] = row
                new_rows.append(row)
                if row.get("error"):
                    error_rows.append(row)
                    _log_error(f"[ERROR] slurp_id={row.get('slurp_id')} error={row.get('error')}")
                if (not args.debug) and args.save_every and len(new_rows) >= args.save_every:
                    _append_jsonl(output_path, new_rows)
                    new_rows = []
                if preview_limit and idx < preview_limit:
                    with preview_lock:
                        if not preview_printed[idx]:
                            preview_printed[idx] = True
                            if tqdm is not None:
                                tqdm.write(f"[PREVIEW {idx + 1}] slurp_id={row.get('slurp_id')}")
                                tqdm.write("SYSTEM PROMPT (FULL):")
                                tqdm.write(prompt)
                                tqdm.write(f"INPUT Transcript: {gold_text}")
                                tqdm.write(f"INPUT Target JSON: {gold_json}")
                                tqdm.write(f"OUTPUT (raw content):")
                                tqdm.write(row.get("rationale_text", ""))
                                if row.get("api_meta"):
                                    tqdm.write(f"TOKENS: {row['api_meta']}")
                            else:
                                print(f"[PREVIEW {idx + 1}] slurp_id={row.get('slurp_id')}", flush=True)
                                print("SYSTEM PROMPT (FULL):", flush=True)
                                print(prompt, flush=True)
                                print(f"INPUT Transcript: {gold_text}", flush=True)
                                print(f"INPUT Target JSON: {gold_json}", flush=True)
                                print("OUTPUT (raw content):", flush=True)
                                print(row.get("rationale_text", ""), flush=True)
                                if row.get("api_meta"):
                                    print(f"TOKENS: {row['api_meta']}", flush=True)
    else:
        iterator = items
        if tqdm is not None:
            iterator = tqdm(items, desc="Oracle CoT", unit="sample")
        for idx, record in enumerate(iterator):
            res = _run_single(idx, record, db_definitions, intent_order, slot_order, args)
            if res is None:
                continue
            idx, row, prompt, gold_text, gold_json = res
            results[idx] = row
            new_rows.append(row)
            if row.get("error"):
                error_rows.append(row)
                _log_error(f"[ERROR] slurp_id={row.get('slurp_id')} error={row.get('error')}")
            if (not args.debug) and args.save_every and len(new_rows) >= args.save_every:
                _append_jsonl(output_path, new_rows)
                new_rows = []
            if preview_limit and idx < preview_limit:
                if tqdm is not None:
                    tqdm.write(f"[PREVIEW {idx + 1}] slurp_id={row.get('slurp_id')}")
                    tqdm.write("SYSTEM PROMPT (FULL):")
                    tqdm.write(prompt)
                    tqdm.write(f"INPUT Transcript: {gold_text}")
                    tqdm.write(f"INPUT Target JSON: {gold_json}")
                    tqdm.write("OUTPUT (raw content):")
                    tqdm.write(row.get("rationale_text", ""))
                    if row.get("api_meta"):
                        tqdm.write(f"TOKENS: {row['api_meta']}")
                else:
                    print(f"[PREVIEW {idx + 1}] slurp_id={row.get('slurp_id')}", flush=True)
                    print("SYSTEM PROMPT (FULL):", flush=True)
                    print(prompt, flush=True)
                    print(f"INPUT Transcript: {gold_text}", flush=True)
                    print(f"INPUT Target JSON: {gold_json}", flush=True)
                    print("OUTPUT (raw content):", flush=True)
                    print(row.get("rationale_text", ""), flush=True)
                    if row.get("api_meta"):
                        print(f"TOKENS: {row['api_meta']}", flush=True)

    final_rows = [r for r in results if r is not None]
    if new_rows:
        _append_jsonl(output_path, new_rows)
        new_rows = []
    # If not resuming, keep full consolidated output.
    if not args.resume:
        write_jsonl(output_path, final_rows)
    if args.error_file:
        write_jsonl(args.error_file, error_rows)

    for proc in worker_procs:
        proc.wait()

    if args.merge_workers and args.num_workers > 1 and args.append_worker_suffix and args.worker_rank == 0:
        _merge_worker_outputs(base_output_path, args.num_workers, cleanup=args.merge_cleanup)


if __name__ == "__main__":
    main()
