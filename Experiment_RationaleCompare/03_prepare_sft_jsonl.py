#!/usr/bin/env python3
import argparse
import os
from typing import Any, Dict, List, Tuple

from common import label_from_record, read_jsonl, write_jsonl
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _key_for_dedup(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("slurp_id", "")),
        str(row.get("mode", "")),
        str(row.get("method", "")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT JSONL for audio_text_mix_e2e_re.py.")
    parser.add_argument("--input_files", type=str, required=True, help="Comma-separated input jsonl files.")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--method", type=str, default="auto", help="auto|vanilla|or-cot|sf-cot")
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    files = [p.strip() for p in args.input_files.split(",") if p.strip()]
    rows: List[Dict[str, Any]] = []

    for path in files:
        items = read_jsonl(path)
        record_iter = items
        if tqdm is not None and not args.no_tqdm:
            record_iter = tqdm(
                items,
                total=len(items),
                desc=f"03_prepare_sft [{os.path.basename(path)}]",
                unit="row",
            )
        for record in record_iter:
            gold_label = record.get("gold_label")
            if isinstance(gold_label, dict):
                final_label = gold_label
            else:
                final_label = label_from_record(record)
            rationale_text = record.get("rationale_text", "")
            method = record.get("method")
            mode = record.get("mode")

            if args.method != "auto":
                method = args.method
                if method == "vanilla":
                    rationale_text = ""

            out = {
                "slurp_id": record.get("slurp_id"),
                "sentence": record.get("sentence") or record.get("text"),
                "recordings": record.get("recordings", []),
                "final": final_label,
                "rationale_text": rationale_text,
            }
            if method:
                out["method"] = method
            if mode:
                out["mode"] = mode

            rows.append(out)

    if args.dedup:
        seen = set()
        deduped = []
        row_iter = rows
        if tqdm is not None and not args.no_tqdm:
            row_iter = tqdm(rows, total=len(rows), desc="03_prepare_sft [dedup]", unit="row")
        for row in row_iter:
            key = _key_for_dedup(row)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        rows = deduped

    if args.limit is not None:
        rows = rows[: args.limit]

    write_jsonl(args.output_file, rows)


if __name__ == "__main__":
    main()
