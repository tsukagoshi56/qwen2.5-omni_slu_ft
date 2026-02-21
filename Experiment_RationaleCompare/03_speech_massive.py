#!/usr/bin/env python3
"""
Prepare Speech-MASSIVE rows by attaching audio file references from a separate CSV.

Typical use-case:
- You already have model output rows (JSONL/CSV) that include text/prompt and IDs,
  but no audio references.
- Another CSV has keys like: prompt, massive_id, speaker_id, file_name.
- This script joins them by massive_id and fills:
  - recordings: [{"file": "..."}]
  - file / filename / audio_file

Examples:
  python Experiment_RationaleCompare/03_speech_massive.py \
    --input_files Experiment_RationaleCompare/sm_fr_train_filtered.jsonl \
    --mapping_csv /path/to/massive_manifest.csv \
    --output_file Experiment_RationaleCompare/sm_fr_train_filtered_enriched.jsonl \
    --audio_root /path/to/audio_root
"""

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common import label_from_record, read_jsonl, write_jsonl


def _split_paths(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _pick_first_nonempty(row: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _extract_massive_id_from_slurp_id(slurp_id: str, dataset_config: str = "") -> str:
    sid = str(slurp_id or "").strip()
    if not sid:
        return ""
    return _normalize_massive_id(sid, dataset_config=dataset_config)


def _normalize_massive_id(value: Any, dataset_config: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    # Strip surrounding punctuation/quotes first.
    text = text.strip("\"'` ")

    # Exact config prefix, e.g. massive-fr-FR-3454
    cfg = str(dataset_config or "").strip()
    if cfg:
        prefix = f"massive-{cfg}-"
        if text.startswith(prefix):
            return text[len(prefix):].strip()

    # Generic prefix patterns:
    # - massive-fr-FR-3454
    # - massive_fr_FR_3454
    # - massive-fr_fr-3454
    patterns = (
        r"^massive-[^-]+-[^-]+-(.+)$",
        r"^massive_[^_]+_[^_]+_(.+)$",
        r"^massive-[^-]+_[^-]+-(.+)$",
        r"^massive_[^_]+-[^_]+_(.+)$",
    )
    for pat in patterns:
        m = re.match(pat, text)
        if m:
            return str(m.group(1)).strip()

    # Keep plain IDs as-is.
    return text


def _read_csv_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if isinstance(row, dict):
                parsed: Dict[str, Any] = {}
                for key, value in row.items():
                    text = str(value or "").strip()
                    if key in {"gold_label", "final", "pred_label", "target_label", "recordings", "entities"}:
                        if text and text[0] in {"{", "["}:
                            try:
                                parsed[key] = json.loads(text)
                                continue
                            except Exception:
                                pass
                    parsed[key] = value
                rows.append(parsed)
    return rows


def _read_input_rows(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _read_csv_rows(path)
    if ext in {".jsonl", ".json"}:
        return read_jsonl(path)
    # Fallback: try JSONL parser first, then CSV.
    try:
        return read_jsonl(path)
    except Exception:
        return _read_csv_rows(path)


def _write_csv_rows(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows_list:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            key_text = str(key)
            if key_text in seen:
                continue
            seen.add(key_text)
            fieldnames.append(key_text)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            out: Dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    out[key] = json.dumps(value, ensure_ascii=False)
                else:
                    out[key] = value
            writer.writerow(out)


def _write_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        _write_csv_rows(path, rows)
    else:
        write_jsonl(path, rows)


def _normalize_recordings(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            file_path = str(item.get("file", "")).strip()
            if file_path:
                out.append({"file": file_path})
        elif isinstance(item, str):
            file_path = str(item).strip()
            if file_path:
                out.append({"file": file_path})
    return out


def _to_sft_row_like_03(row: Dict[str, Any]) -> Dict[str, Any]:
    slurp_id = _pick_first_nonempty(row, ["slurp_id", "id", "massive_id"])
    sentence = _pick_first_nonempty(row, ["sentence", "text", "transcript", "prompt"])
    rationale_text = _pick_first_nonempty(row, ["rationale_text", "raw_output", "response", "assistant_text"])

    gold_label = row.get("gold_label")
    final_obj = row.get("final")
    if isinstance(gold_label, dict):
        final_label = gold_label
    elif isinstance(final_obj, dict):
        final_label = final_obj
    else:
        final_label = label_from_record(row)

    recordings = _normalize_recordings(row.get("recordings"))

    out = {
        "slurp_id": slurp_id,
        "sentence": sentence,
        "recordings": recordings,
        "final": final_label,
        "rationale_text": rationale_text,
    }
    for key in ("method", "mode", "dataset", "dataset_config", "dataset_split", "massive_id", "speaker_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            out[key] = value
    for key in ("file", "filename", "audio_file", "file_name"):
        value = row.get(key)
        if value is not None and str(value).strip():
            out[key] = value
    return out


def _has_recordings(row: Dict[str, Any]) -> bool:
    recs = row.get("recordings")
    if not isinstance(recs, list) or not recs:
        return False
    first = recs[0]
    if isinstance(first, dict):
        return bool(str(first.get("file", "")).strip())
    if isinstance(first, str):
        return bool(str(first).strip())
    return False


def _resolve_audio_file(file_name: str, audio_root: str) -> str:
    name = str(file_name or "").strip()
    if not name:
        return ""
    if os.path.isabs(name):
        return name
    root = str(audio_root or "").strip()
    if root:
        return os.path.join(root, name)
    return name


def _build_mapping(
    mapping_rows: List[Dict[str, Any]],
    mapping_massive_id_field: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    duplicates = 0
    missing_key = 0
    for row in mapping_rows:
        if not isinstance(row, dict):
            continue
        massive_id = _normalize_massive_id(
            row.get(mapping_massive_id_field, ""),
            dataset_config=str(row.get("dataset_config", "")).strip(),
        )
        if not massive_id:
            missing_key += 1
            continue
        if massive_id in mapping:
            duplicates += 1
            continue
        mapping[massive_id] = dict(row)
    return mapping, {
        "mapping_rows": len(mapping_rows),
        "mapping_unique_ids": len(mapping),
        "mapping_missing_id_rows": missing_key,
        "mapping_duplicate_id_rows": duplicates,
    }


def _dedup_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sid = str(row.get("slurp_id", "")).strip()
        mid = str(row.get("massive_id", "")).strip()
        mode = str(row.get("mode", "")).strip()
        key = (sid, mid, mode) if (sid or mid or mode) else (f"__row_{idx}", "", "")
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach Speech-MASSIVE audio references from CSV by massive_id."
    )
    parser.add_argument("--input_files", type=str, required=True, help="Comma-separated JSONL/CSV files to enrich.")
    parser.add_argument("--mapping_csv", type=str, required=True, help="CSV containing massive_id and file_name.")
    parser.add_argument("--output_file", type=str, required=True, help="Output .jsonl or .csv path.")
    parser.add_argument("--audio_root", type=str, default="", help="Optional root directory prepended to relative file_name.")

    parser.add_argument("--input_massive_id_field", type=str, default="massive_id")
    parser.add_argument("--mapping_massive_id_field", type=str, default="massive_id")
    parser.add_argument("--mapping_file_field", type=str, default="file_name")
    parser.add_argument("--mapping_speaker_field", type=str, default="speaker_id")
    parser.add_argument("--slurp_id_field", type=str, default="slurp_id")
    parser.add_argument("--dataset_config_field", type=str, default="dataset_config")
    parser.add_argument(
        "--output_style",
        type=str,
        choices=["03", "passthrough"],
        default="03",
        help="03: output 03_prepare_sft_jsonl-like rows. passthrough: keep all original fields.",
    )

    parser.add_argument("--overwrite_recordings", action="store_true", help="Overwrite existing recordings if present.")
    parser.add_argument("--strict_missing_map", action="store_true", help="Error if a row's massive_id is not found in mapping CSV.")
    parser.add_argument("--strict_missing_file", action="store_true", help="Error if mapping row exists but file_name is empty.")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate rows by (slurp_id, massive_id, mode).")
    parser.add_argument("--limit", type=int, default=None, help="Optional output row cap after processing.")
    args = parser.parse_args()

    input_paths = _split_paths(args.input_files)
    if not input_paths:
        raise ValueError("--input_files is empty.")
    if not os.path.exists(args.mapping_csv):
        raise FileNotFoundError(f"mapping_csv not found: {args.mapping_csv}")

    mapping_rows = _read_csv_rows(args.mapping_csv)
    mapping, mapping_stats = _build_mapping(mapping_rows, args.mapping_massive_id_field)
    if not mapping:
        raise RuntimeError(
            f"No valid mapping rows found from {args.mapping_csv} "
            f"(field={args.mapping_massive_id_field})."
        )

    all_rows: List[Dict[str, Any]] = []
    for path in input_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"input file not found: {path}")
        all_rows.extend(_read_input_rows(path))

    out_rows: List[Dict[str, Any]] = []
    missing_map_rows = 0
    missing_file_rows = 0
    attached_rows = 0
    kept_existing_recordings = 0

    for idx, row in enumerate(all_rows):
        if not isinstance(row, dict):
            continue
        out = dict(row)

        massive_id_raw = _pick_first_nonempty(
            out,
            [args.input_massive_id_field, "massive_id", "masssive_id", "id", "utt_id"],
        )
        massive_id = _normalize_massive_id(
            massive_id_raw,
            dataset_config=str(out.get(args.dataset_config_field, "")).strip(),
        )
        if not massive_id:
            slurp_id = str(out.get(args.slurp_id_field, "")).strip()
            dataset_config = str(out.get(args.dataset_config_field, "")).strip()
            massive_id = _extract_massive_id_from_slurp_id(slurp_id=slurp_id, dataset_config=dataset_config)
        if massive_id:
            out["massive_id"] = massive_id

        if not massive_id or massive_id not in mapping:
            missing_map_rows += 1
            out_rows.append(_to_sft_row_like_03(out) if args.output_style == "03" else out)
            continue

        map_row = mapping[massive_id]
        file_name = _pick_first_nonempty(map_row, [args.mapping_file_field, "file_name", "file", "filename"])
        if not file_name:
            missing_file_rows += 1
            out_rows.append(_to_sft_row_like_03(out) if args.output_style == "03" else out)
            continue

        audio_file = _resolve_audio_file(file_name=file_name, audio_root=args.audio_root)
        speaker_id = str(map_row.get(args.mapping_speaker_field, "")).strip()
        if speaker_id and (not str(out.get("speaker_id", "")).strip()):
            out["speaker_id"] = speaker_id

        out["file_name"] = file_name
        out["file"] = audio_file
        out["filename"] = audio_file
        out["audio_file"] = audio_file

        if args.overwrite_recordings or (not _has_recordings(out)):
            out["recordings"] = [{"file": audio_file}]
            attached_rows += 1
        else:
            kept_existing_recordings += 1

        out_rows.append(_to_sft_row_like_03(out) if args.output_style == "03" else out)

    if args.dedup:
        before = len(out_rows)
        out_rows = _dedup_rows(out_rows)
        print(f"[INFO] Dedup: {before} -> {len(out_rows)} rows")

    if args.limit is not None:
        out_rows = out_rows[: int(args.limit)]

    if args.strict_missing_map and missing_map_rows > 0:
        raise RuntimeError(
            f"strict_missing_map=True and {missing_map_rows} rows had no matching massive_id in mapping."
        )
    if args.strict_missing_file and missing_file_rows > 0:
        raise RuntimeError(
            f"strict_missing_file=True and {missing_file_rows} mapped rows had empty file_name."
        )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    _write_rows(args.output_file, out_rows)

    print(f"[INFO] Input rows: {len(all_rows)}")
    print(f"[INFO] Output rows: {len(out_rows)} -> {args.output_file}")
    print(
        "[INFO] Mapping stats: "
        f"rows={mapping_stats['mapping_rows']} "
        f"unique_ids={mapping_stats['mapping_unique_ids']} "
        f"missing_id_rows={mapping_stats['mapping_missing_id_rows']} "
        f"duplicate_id_rows={mapping_stats['mapping_duplicate_id_rows']}"
    )
    print(
        "[INFO] Attach stats: "
        f"attached_recordings={attached_rows} "
        f"kept_existing_recordings={kept_existing_recordings} "
        f"missing_map_rows={missing_map_rows} "
        f"missing_file_rows={missing_file_rows}"
    )


if __name__ == "__main__":
    main()
