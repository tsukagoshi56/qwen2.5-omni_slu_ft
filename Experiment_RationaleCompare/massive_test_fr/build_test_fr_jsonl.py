#!/usr/bin/env python3
"""
Build a SLURP-style FR test JSONL from Speech-MASSIVE test data.

This script supports two input modes:
1) Local parquet files (e.g., extracted zip from HF web download)
2) Hugging Face dataset loading (FBK-MT/Speech-MASSIVE-test)

Output format is compatible with scripts in this repository that expect
SLURP-like test.jsonl fields:
- slurp_id
- sentence
- scenario / action / intent
- tokens (list of {surface: ...})
- entities (list of {type, span})
- recordings (list of {file: ...})
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _normalize_intent_label(value: Any) -> str:
    return str(value or "").strip().replace(":", "_")


def _normalize_scenario_action(scenario_value: Any, intent_value: Any) -> Tuple[str, str, str]:
    scenario = str(scenario_value or "").strip()
    intent = _normalize_intent_label(intent_value)
    action = intent

    scenario_norm = _normalize_intent_label(scenario)
    if scenario_norm and intent.startswith(f"{scenario_norm}_"):
        action = intent[len(scenario_norm) + 1 :]
    elif (not scenario_norm) and "_" in intent:
        scenario2, action2 = intent.split("_", 1)
        if scenario2:
            scenario = scenario2
            action = action2

    return scenario, action, intent


def _extract_audio_path(record: Dict[str, Any]) -> str:
    direct = record.get("path")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    audio = record.get("audio")
    if isinstance(audio, dict):
        path = audio.get("path")
        if isinstance(path, str) and path.strip():
            return path.strip()
    return ""


def _split_bio_label(label: str, outside_labels: Sequence[str]) -> Tuple[str, str]:
    text = str(label or "").strip()
    if not text:
        return "", ""

    if text in outside_labels:
        return "", ""

    upper = text.upper()
    if upper.startswith("B-") or upper.startswith("I-"):
        return upper[0], text[2:]
    if upper.startswith("B_") or upper.startswith("I_"):
        return upper[0], text[2:]

    # If no BIO prefix exists, treat as continuation-type token label.
    return "I", text


def _labels_to_entity_spans(
    tokens: Sequence[str],
    labels: Sequence[str],
    outside_labels: Sequence[str],
) -> List[Dict[str, Any]]:
    outside_set = {str(x) for x in outside_labels}
    entities: List[Dict[str, Any]] = []

    current_type = ""
    current_span: List[int] = []

    def flush() -> None:
        nonlocal current_type, current_span
        if current_type and current_span:
            entities.append({"type": current_type, "span": list(current_span)})
        current_type = ""
        current_span = []

    for idx, (_, raw_label) in enumerate(zip(tokens, labels)):
        prefix, slot_type = _split_bio_label(str(raw_label), list(outside_set))
        if not slot_type:
            flush()
            continue

        if prefix == "B":
            flush()
            current_type = slot_type
            current_span = [idx]
            continue

        # prefix == "I" or fallback
        if current_type == slot_type and current_span:
            current_span.append(idx)
        else:
            flush()
            current_type = slot_type
            current_span = [idx]

    flush()
    return entities


def _load_records_from_parquet_glob(pattern: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched: {pattern}")

    ds = load_dataset("parquet", data_files=files, split="train")
    return [dict(row) for row in ds]


def _load_records_from_hf(
    dataset_name: str,
    config: str,
    split: str,
    cache_dir: Optional[str],
    local_files_only: bool,
) -> List[Dict[str, Any]]:
    from datasets import Audio, DownloadConfig, load_dataset

    kwargs: Dict[str, Any] = {
        "path": dataset_name,
        "name": config,
        "split": split,
        "cache_dir": cache_dir,
    }
    if local_files_only:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)
        kwargs["download_mode"] = "reuse_dataset_if_exists"

    ds = load_dataset(**kwargs)
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))
    return [dict(row) for row in ds]


def _iter_slurp_rows(
    records: Iterable[Dict[str, Any]],
    dataset_config: str,
    dataset_split: str,
    transcript_field: str,
    outside_labels: Sequence[str],
    recording_path_mode: str,
) -> Iterable[Dict[str, Any]]:
    for idx, record in enumerate(records):
        transcript = (
            record.get(transcript_field)
            or record.get("utt")
            or record.get("text")
            or ""
        )
        transcript = str(transcript).strip()

        scenario_raw = record.get("scenario_str") or record.get("scenario") or ""
        intent_raw = record.get("intent_str") or record.get("intent") or ""
        scenario, action, intent = _normalize_scenario_action(scenario_raw, intent_raw)

        token_values = record.get("tokens") if isinstance(record.get("tokens"), list) else []
        label_values = record.get("labels") if isinstance(record.get("labels"), list) else []
        tokens = [str(t) for t in token_values]
        labels = [str(l) for l in label_values]
        entities = _labels_to_entity_spans(tokens=tokens, labels=labels, outside_labels=outside_labels)

        audio_path = _extract_audio_path(record)
        if recording_path_mode == "basename" and audio_path:
            rec_file = os.path.basename(audio_path)
        else:
            rec_file = audio_path

        recordings = [{"file": rec_file}] if rec_file else []

        base_id = (
            record.get("id")
            or record.get("utt_id")
            or record.get("audio_id")
            or (os.path.basename(audio_path) if audio_path else "")
            or str(idx)
        )
        slurp_id = f"massive-{dataset_config}-{base_id}"

        yield {
            "slurp_id": slurp_id,
            "sentence": transcript,
            "text": transcript,
            "scenario": scenario,
            "action": action,
            "intent": intent,
            "tokens": [{"surface": tok} for tok in tokens],
            "entities": entities,
            "recordings": recordings,
            "dataset": "speech_massive",
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build test_FR.jsonl from Speech-MASSIVE test data.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="Experiment_RationaleCompare/massive_test_fr/test_FR.jsonl",
        help="Output path for SLURP-style JSONL.",
    )
    parser.add_argument("--config", type=str, default="fr-FR", help="Dataset config/language (default: fr-FR).")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test).")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="FBK-MT/Speech-MASSIVE-test",
        help="HF dataset repo used when --source_glob is not set.",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="HF datasets cache dir.")
    parser.add_argument(
        "--source_glob",
        type=str,
        default="",
        help=(
            "Parquet glob override. If set, load from local parquet files instead of HF dataset API. "
            "Example: '/data/Speech-MASSIVE-test/fr-FR/test-*.parquet'"
        ),
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="",
        help="Directory root containing '<config>/test-*.parquet'. Used only if --source_glob is empty.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Use local HF cache only (no remote download attempt).",
    )
    parser.add_argument(
        "--transcript_field",
        type=str,
        default="utt",
        help="Preferred transcript field name (default: utt).",
    )
    parser.add_argument(
        "--outside_labels",
        type=str,
        default="Other,O",
        help="Comma-separated outside labels for BIO parsing (default: Other,O).",
    )
    parser.add_argument(
        "--recording_path_mode",
        type=str,
        choices=["as_is", "basename"],
        default="as_is",
        help="How to save recordings.file from source audio path.",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for quick debug.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outside_labels = [x.strip() for x in str(args.outside_labels).split(",") if x.strip()]

    source_glob = str(args.source_glob or "").strip()
    source_dir = str(args.source_dir or "").strip()

    if not source_glob and source_dir:
        source_glob = os.path.join(source_dir, args.config, f"{args.split}-*.parquet")

    if source_glob:
        print(f"[INFO] Loading records from local parquet glob: {source_glob}")
        records = _load_records_from_parquet_glob(source_glob)
        source_desc = f"parquet:{source_glob}"
    else:
        print(
            "[INFO] Loading records from HF dataset: "
            f"name={args.dataset_name} config={args.config} split={args.split} "
            f"local_only={bool(args.local_files_only)}"
        )
        records = _load_records_from_hf(
            dataset_name=args.dataset_name,
            config=args.config,
            split=args.split,
            cache_dir=args.cache_dir,
            local_files_only=bool(args.local_files_only),
        )
        source_desc = f"hf:{args.dataset_name}/{args.config}:{args.split}"

    if args.max_samples is not None:
        records = records[: int(args.max_samples)]

    rows = list(
        _iter_slurp_rows(
            records=records,
            dataset_config=args.config,
            dataset_split=args.split,
            transcript_field=args.transcript_field,
            outside_labels=outside_labels,
            recording_path_mode=args.recording_path_mode,
        )
    )

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with_audio = sum(1 for r in rows if isinstance(r.get("recordings"), list) and len(r["recordings"]) > 0)
    with_entities = sum(1 for r in rows if isinstance(r.get("entities"), list) and len(r["entities"]) > 0)

    print(f"[DONE] Wrote {len(rows)} rows -> {args.output_file}")
    print(f"[INFO] Source: {source_desc}")
    print(f"[INFO] Rows with recordings: {with_audio}/{len(rows)}")
    print(f"[INFO] Rows with non-empty entities: {with_entities}/{len(rows)}")


if __name__ == "__main__":
    main()
