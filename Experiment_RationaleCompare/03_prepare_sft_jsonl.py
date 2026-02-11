#!/usr/bin/env python3
import argparse
import json
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


def _build_slurp_map(slurp_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """Build a slurp_id -> record map from one or more SLURP JSONL files.

    The map stores ``recordings``, ``sentence``, and the raw record so that
    downstream enrichment can look up audio file references by slurp_id.
    """
    smap: Dict[str, Dict[str, Any]] = {}
    for path in slurp_files:
        if not os.path.exists(path):
            print(f"[WARN] SLURP file not found, skipping: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = str(rec.get("slurp_id", ""))
                if sid and sid not in smap:
                    smap[sid] = rec
    return smap


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT JSONL for audio_text_mix_e2e_re.py.")
    parser.add_argument("--input_files", type=str, required=True, help="Comma-separated input jsonl files.")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--slurp_files",
        type=str,
        default="",
        help="Comma-separated SLURP JSONL files (train/devel/test) to look up recordings by slurp_id.",
    )
    parser.add_argument("--method", type=str, default="auto", help="auto|vanilla|or-cot|sf-cot")
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    # Build slurp_id -> recordings lookup from original SLURP data.
    slurp_paths = [p.strip() for p in args.slurp_files.split(",") if p.strip()]
    slurp_map = _build_slurp_map(slurp_paths) if slurp_paths else {}
    if slurp_map:
        print(f"[INFO] Built SLURP lookup with {len(slurp_map)} entries from {len(slurp_paths)} file(s).")

    files = [p.strip() for p in args.input_files.split(",") if p.strip()]
    rows: List[Dict[str, Any]] = []
    enriched_count = 0

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

            # Get recordings: first from the record itself, then from SLURP lookup.
            recordings = record.get("recordings", [])
            sentence = record.get("sentence") or record.get("text") or ""
            slurp_id = str(record.get("slurp_id", ""))

            if (not recordings) and slurp_id and slurp_id in slurp_map:
                slurp_rec = slurp_map[slurp_id]
                recordings = slurp_rec.get("recordings", [])
                # Also fill sentence if missing.
                if not sentence:
                    sentence = slurp_rec.get("sentence", "")
                enriched_count += 1

            out = {
                "slurp_id": record.get("slurp_id"),
                "sentence": sentence,
                "recordings": recordings,
                "final": final_label,
                "rationale_text": rationale_text,
            }
            # Preserve audio file references from various field names.
            for key in ("file", "filename", "audio_file", "audio_filename"):
                val = record.get(key)
                if val:
                    out[key] = val
            if method:
                out["method"] = method
            if mode:
                out["mode"] = mode

            rows.append(out)

    if enriched_count:
        print(f"[INFO] Enriched {enriched_count} records with recordings from SLURP lookup.")

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
    print(f"[INFO] Wrote {len(rows)} records to {args.output_file}")


if __name__ == "__main__":
    main()
