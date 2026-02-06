#!/usr/bin/env python3
import argparse
import json
import os
import random
import threading
import time
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


def _build_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package not available. Install it or use a local generator.")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("API_ENDPOINT") or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key, base_url=base_url)


def _get_thread_client() -> Any:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = _build_client()
        _thread_local.client = client
    return client


def _call_api(client: Any, prompt: str, model_name: str, max_tokens: int, temperature: float, top_p: float) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return resp.choices[0].message.content or ""


def _run_single(
    idx: int,
    record: Dict[str, Any],
    db_definitions: str,
    args: argparse.Namespace,
) -> Optional[Tuple[int, Dict[str, Any], str, str, str]]:
    gold_text = str(record.get("sentence", "") or record.get("text", "") or "").strip()
    if not gold_text:
        return None
    gold_label = label_from_record(record)
    gold_json = json.dumps(gold_label, ensure_ascii=False)
    prompt = render_oracle_prompt(db_definitions, gold_text, gold_json)

    output = ""
    for attempt in range(args.retry + 1):
        try:
            client = _get_thread_client()
            output = _call_api(
                client=client,
                prompt=prompt,
                model_name=args.model_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            break
        except Exception:
            if attempt >= args.retry:
                raise
            time.sleep(args.retry_sleep)

    result = {
        "slurp_id": record.get("slurp_id"),
        "sentence": gold_text,
        "recordings": record.get("recordings", []),
        "final": gold_label,
        "rationale_text": output.strip(),
        "method": "or-cot",
        "mode": "text",
    }
    return idx, result, prompt, gold_text, gold_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Oracle CoT rationales (gold text + gold JSON).")
    parser.add_argument("--input_file", type=str, default="slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--metadata_file", type=str, default="Experiment_3/slurp_metadata.json")
    parser.add_argument("--output_file", type=str, default="Experiment_RationaleCompare/oracle_cot.jsonl")
    parser.add_argument("--model_name", type=str, default="deepseek-r1")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent API requests.")
    parser.add_argument("--preview", type=int, default=10, help="Print first N outputs to stdout.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_rank", type=int, default=0)
    parser.add_argument("--append_worker_suffix", action="store_true")
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if args.worker_rank < 0 or args.worker_rank >= args.num_workers:
        raise ValueError("worker_rank out of range")

    rng = random.Random(args.seed + args.worker_rank)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input_file) if not os.path.isabs(args.input_file) else args.input_file
    metadata_path = os.path.join(base_dir, args.metadata_file) if not os.path.isabs(args.metadata_file) else args.metadata_file
    output_path = os.path.join(base_dir, args.output_file) if not os.path.isabs(args.output_file) else args.output_file
    if args.append_worker_suffix and args.num_workers > 1:
        root, ext = os.path.splitext(output_path)
        output_path = f"{root}.w{args.worker_rank}of{args.num_workers}{ext or '.jsonl'}"

    metadata = load_metadata(metadata_path)
    db_definitions = build_db_definitions(metadata)

    items = read_jsonl(input_path)
    if args.limit:
        items = items[: args.limit]
    if args.num_workers > 1:
        items = items[args.worker_rank :: args.num_workers]

    results: List[Optional[Dict[str, Any]]] = [None] * len(items)
    preview_limit = max(0, int(args.preview))
    preview_lock = threading.Lock()
    preview_printed: List[bool] = [False] * len(items)

    if args.parallel > 1 and ThreadPoolExecutor is not None:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [
                executor.submit(_run_single, idx, record, db_definitions, args)
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
                if preview_limit and idx < preview_limit:
                    with preview_lock:
                        if not preview_printed[idx]:
                            preview_printed[idx] = True
                            if tqdm is not None:
                                tqdm.write(f"[PREVIEW {idx + 1}] slurp_id={row.get('slurp_id')}")
                                tqdm.write(f"INPUT Transcript: {gold_text}")
                                tqdm.write(f"INPUT Target JSON: {gold_json}")
                                tqdm.write("OUTPUT:")
                                tqdm.write(row.get("rationale_text", ""))
                            else:
                                print(f"[PREVIEW {idx + 1}] slurp_id={row.get('slurp_id')}", flush=True)
                                print(f"INPUT Transcript: {gold_text}", flush=True)
                                print(f"INPUT Target JSON: {gold_json}", flush=True)
                                print("OUTPUT:", flush=True)
                                print(row.get("rationale_text", ""), flush=True)
    else:
        iterator = items
        if tqdm is not None:
            iterator = tqdm(items, desc="Oracle CoT", unit="sample")
        for idx, record in enumerate(iterator):
            res = _run_single(idx, record, db_definitions, args)
            if res is None:
                continue
            idx, row, prompt, gold_text, gold_json = res
            results[idx] = row
            if preview_limit and idx < preview_limit:
                if tqdm is not None:
                    tqdm.write(f"[PREVIEW {idx + 1}] slurp_id={row.get('slurp_id')}")
                    tqdm.write(f"INPUT Transcript: {gold_text}")
                    tqdm.write(f"INPUT Target JSON: {gold_json}")
                    tqdm.write("OUTPUT:")
                    tqdm.write(row.get("rationale_text", ""))
                else:
                    print(f"[PREVIEW {idx + 1}] slurp_id={row.get('slurp_id')}", flush=True)
                    print(f"INPUT Transcript: {gold_text}", flush=True)
                    print(f"INPUT Target JSON: {gold_json}", flush=True)
                    print("OUTPUT:", flush=True)
                    print(row.get("rationale_text", ""), flush=True)

    final_rows = [r for r in results if r is not None]
    write_jsonl(output_path, final_rows)


if __name__ == "__main__":
    main()
