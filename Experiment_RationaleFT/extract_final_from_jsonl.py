#!/usr/bin/env python3
"""
Extract scenario/action/entities from model outputs in a JSONL file.

Usage:
  python Experiment_RationaleFT/extract_final_from_jsonl.py /path/to/input.jsonl

Input:
  - One JSONL file path only.

Output:
  - A NEW JSON file (never overwrites input file).
  - Auto path example: input.final_extracted.json
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


def pick_first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def clean_json_text(text: str) -> str:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text


def parse_entities(raw_entities: Any) -> List[Dict[str, str]]:
    entities: List[Dict[str, str]] = []
    if not isinstance(raw_entities, list):
        return entities
    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue
        ent_type = str(ent.get("type", "")).strip()
        filler = ent.get("filler")
        if filler is None:
            filler = ent.get("filter")
        if filler is None:
            filler = ent.get("value")
        if filler is None:
            filler = ""
        entities.append({"type": ent_type, "filler": str(filler)})
    return entities


def split_intent(intent: str) -> Tuple[str, str]:
    intent = (intent or "").strip()
    if not intent:
        return "", ""
    if "_" in intent:
        return tuple(x.strip() for x in intent.split("_", 1))
    if ":" in intent:
        return tuple(x.strip() for x in intent.split(":", 1))
    return "", intent


def decode_json_maybe_nested(text: str) -> Optional[Any]:
    current = text
    for _ in range(2):
        try:
            obj = json.loads(current)
        except Exception:
            return None
        if isinstance(obj, str):
            current = obj.strip()
            continue
        return obj
    return None


def extract_balanced_object(text: str, start_idx: int) -> Optional[str]:
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "{":
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    return None


def extract_all_json_objects(text: str, max_objects: int = 256) -> List[str]:
    objects: List[str] = []
    i = 0
    while i < len(text) and len(objects) < max_objects:
        if text[i] != "{":
            i += 1
            continue
        obj_text = extract_balanced_object(text, i)
        if obj_text:
            objects.append(obj_text)
            i += len(obj_text)
        else:
            i += 1
    return objects


def normalize_label_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    scenario = str(obj.get("scenario", "")).strip()
    action = str(obj.get("action", "")).strip()
    entities = parse_entities(obj.get("entities", []))

    if not scenario and not action:
        intent = obj.get("intent")
        if isinstance(intent, str):
            scenario, action = split_intent(intent)

    return {
        "scenario": scenario,
        "action": action,
        "entities": entities,
    }


def is_nonempty_label(label_obj: Dict[str, Any]) -> bool:
    return bool(label_obj.get("scenario") or label_obj.get("action") or label_obj.get("entities"))


def extract_from_final_keyword(raw_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    # Highest priority: object right after "final:" or "final_prediction:"
    matches = list(re.finditer(r"(?i)\bfinal(?:_prediction)?\b\s*[:=]\s*", raw_text))
    for m in reversed(matches):
        brace_pos = raw_text.find("{", m.end())
        if brace_pos < 0:
            continue
        obj_text = extract_balanced_object(raw_text, brace_pos)
        if not obj_text:
            continue
        parsed = decode_json_maybe_nested(obj_text)
        if isinstance(parsed, dict):
            label_obj = normalize_label_obj(parsed)
            if is_nonempty_label(label_obj):
                return label_obj, "final_keyword_object"
    return None, ""


def extract_label_from_text(raw_text: str) -> Tuple[Dict[str, Any], str]:
    default_obj = {"scenario": "", "action": "", "entities": []}
    text = clean_json_text(raw_text)

    # 1) Highest priority: object after "final:"
    final_obj, final_method = extract_from_final_keyword(text)
    if final_obj is not None:
        return final_obj, final_method

    # 2) Parse whole JSON, then prioritize wrappers.
    parsed = decode_json_maybe_nested(text)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("final"), dict):
            label_obj = normalize_label_obj(parsed["final"])
            if is_nonempty_label(label_obj):
                return label_obj, "top_level_final"
        if isinstance(parsed.get("final_prediction"), dict):
            label_obj = normalize_label_obj(parsed["final_prediction"])
            if is_nonempty_label(label_obj):
                return label_obj, "top_level_final_prediction"
        label_obj = normalize_label_obj(parsed)
        if is_nonempty_label(label_obj):
            return label_obj, "top_level"

    # 3) Fallback: scan embedded JSON objects.
    for obj_text in reversed(extract_all_json_objects(text)):
        parsed_obj = decode_json_maybe_nested(obj_text)
        if not isinstance(parsed_obj, dict):
            continue
        if isinstance(parsed_obj.get("final"), dict):
            label_obj = normalize_label_obj(parsed_obj["final"])
            if is_nonempty_label(label_obj):
                return label_obj, "embedded_final"
        if isinstance(parsed_obj.get("final_prediction"), dict):
            label_obj = normalize_label_obj(parsed_obj["final_prediction"])
            if is_nonempty_label(label_obj):
                return label_obj, "embedded_final_prediction"
        label_obj = normalize_label_obj(parsed_obj)
        if is_nonempty_label(label_obj):
            return label_obj, "embedded_object"

    return default_obj, "not_found"


def resolve_output_path(input_path: str) -> str:
    root, _ = os.path.splitext(input_path)
    candidate = f"{root}.final_extracted.json"
    if not os.path.exists(candidate):
        return candidate
    idx = 1
    while True:
        candidate = f"{root}.final_extracted.{idx}.json"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def detect_raw_text_field(row: Dict[str, Any]) -> Tuple[str, str]:
    ordered_keys = [
        "raw_output",
        "output",
        "prediction",
        "assistant_output",
        "rationale_raw",
        "rationale_text",
    ]
    for key in ordered_keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value, key
    return "", ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract final SLU labels from a JSONL file.")
    parser.add_argument("input_jsonl", type=str, help="Path to input JSONL file.")
    args = parser.parse_args()

    input_path = args.input_jsonl
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    extracted_rows: List[Dict[str, Any]] = []
    total = 0
    extracted = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                extracted_rows.append(
                    {
                        "line_no": line_no,
                        "scenario": "",
                        "action": "",
                        "entities": [],
                        "method": "invalid_jsonl_row",
                    }
                )
                continue

            if not isinstance(row, dict):
                extracted_rows.append(
                    {
                        "line_no": line_no,
                        "scenario": "",
                        "action": "",
                        "entities": [],
                        "method": "row_not_dict",
                    }
                )
                continue

            raw_text, raw_field = detect_raw_text_field(row)
            if not raw_text:
                raw_text = json.dumps(row, ensure_ascii=False)
                raw_field = "row_dump_fallback"

            label_obj, method = extract_label_from_text(raw_text)
            if is_nonempty_label(label_obj):
                extracted += 1

            out = {
                "line_no": line_no,
                "id": pick_first_nonempty(row.get("id"), row.get("sample_id"), row.get("uid")),
                "slurp_id": pick_first_nonempty(row.get("slurp_id")),
                "scenario": label_obj.get("scenario", ""),
                "action": label_obj.get("action", ""),
                "entities": label_obj.get("entities", []),
                "method": method,
                "raw_field": raw_field,
            }
            extracted_rows.append(out)

    output_path = resolve_output_path(input_path)
    output_obj = {
        "input_jsonl": input_path,
        "output_json": output_path,
        "total_rows": total,
        "extracted_rows": extracted,
        "rows": extracted_rows,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=2)

    print(f"[DONE] input: {input_path}")
    print(f"[DONE] output: {output_path}")
    print(f"[DONE] extracted: {extracted}/{total}")


if __name__ == "__main__":
    main()

