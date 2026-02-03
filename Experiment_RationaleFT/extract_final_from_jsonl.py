#!/usr/bin/env python3
"""
Extract scenario/action/entities from model outputs in a JSONL file.

Usage:
  python Experiment_RationaleFT/extract_final_from_jsonl.py /path/to/input.jsonl

Input:
  - One JSONL file path only.

Output:
  - A NEW JSONL file (never overwrites input file).
  - Auto path example: input.final_extracted.jsonl
  - Keeps each row format, and writes extracted labels into scenario/action/entities.
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

    def missing_or_error(value: Any) -> bool:
        text = str(value or "").strip().lower()
        return (not text) or text in {"error", "err", "unknown", "none", "null", "n/a", "na"}

    intent = obj.get("intent")
    if isinstance(intent, str) and (missing_or_error(scenario) or missing_or_error(action)):
        intent_scenario, intent_action = split_intent(intent)
        if missing_or_error(scenario) and intent_scenario:
            scenario = intent_scenario
        if missing_or_error(action) and intent_action:
            action = intent_action

    return {
        "scenario": scenario,
        "action": action,
        "entities": entities,
    }


def is_nonempty_label(label_obj: Dict[str, Any]) -> bool:
    return bool(label_obj.get("scenario") or label_obj.get("action") or label_obj.get("entities"))


def label_score(label_obj: Dict[str, Any]) -> int:
    score = 0
    if str(label_obj.get("scenario", "")).strip():
        score += 1
    if str(label_obj.get("action", "")).strip():
        score += 1
    if isinstance(label_obj.get("entities"), list) and label_obj.get("entities"):
        score += 1
    return score


def is_error_like(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"error", "err", "unknown", "none", "null", "n/a", "na"}


def is_missing_or_error(value: Any) -> bool:
    text = str(value or "").strip()
    return (not text) or is_error_like(text)


def merge_labels(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    merged = {
        "scenario": str(primary.get("scenario", "")).strip(),
        "action": str(primary.get("action", "")).strip(),
        "entities": parse_entities(primary.get("entities", [])),
    }
    sec_scenario = str(secondary.get("scenario", "")).strip()
    sec_action = str(secondary.get("action", "")).strip()
    if is_missing_or_error(merged["scenario"]) and sec_scenario and (not is_error_like(sec_scenario)):
        merged["scenario"] = sec_scenario
    if is_missing_or_error(merged["action"]) and sec_action and (not is_error_like(sec_action)):
        merged["action"] = sec_action
    if (not merged["entities"]) and isinstance(secondary.get("entities"), list):
        merged["entities"] = parse_entities(secondary.get("entities", []))
    return merged


def extract_last_group(pattern: str, text: str) -> str:
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not matches:
        return ""
    last = matches[-1]
    if isinstance(last, tuple):
        return str(last[0]).strip()
    return str(last).strip()


def extract_label_from_patterns(raw_text: str) -> Dict[str, Any]:
    text = raw_text or ""
    # Analyze the tail after the last "final" marker first.
    final_matches = list(re.finditer(r"(?i)\bfinal(?:_prediction)?\b\s*[:=]?", text))
    search_text = text[final_matches[-1].start():] if final_matches else text

    scenario = extract_last_group(r'"scenario"\s*[:=]\s*"([^"]+)"', search_text)
    if not scenario:
        scenario = extract_last_group(r"'scenario'\s*[:=]\s*'([^']+)'", search_text)
    if not scenario:
        scenario = extract_last_group(r"scenario\s*[:=]\s*([A-Za-z0-9_:-]+)", search_text)
    action = extract_last_group(r'"action"\s*[:=]\s*"([^"]+)"', search_text)
    if not action:
        action = extract_last_group(r"'action'\s*[:=]\s*'([^']+)'", search_text)
    if not action:
        action = extract_last_group(r"action\s*[:=]\s*([A-Za-z0-9_:-]+)", search_text)
    intent = extract_last_group(r'"intent"\s*[:=]\s*"([^"]+)"', search_text)
    if not intent:
        intent = extract_last_group(r"'intent'\s*[:=]\s*'([^']+)'", search_text)
    if not intent:
        intent = extract_last_group(r"intent\s*[:=]\s*([A-Za-z0-9_:-]+)", search_text)

    if (is_missing_or_error(scenario) or is_missing_or_error(action)) and intent:
        intent_scenario, intent_action = split_intent(intent)
        if is_missing_or_error(scenario) and intent_scenario:
            scenario = intent_scenario
        if is_missing_or_error(action) and intent_action:
            action = intent_action

    return {
        "scenario": str(scenario or "").strip(),
        "action": str(action or "").strip(),
        "entities": [],
    }


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
    candidate = f"{root}.final_extracted.jsonl"
    if not os.path.exists(candidate):
        return candidate
    idx = 1
    while True:
        candidate = f"{root}.final_extracted.{idx}.jsonl"
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

    total = 0
    extracted = 0
    updated = 0
    unresolved = 0
    unresolved_rows: List[Dict[str, Any]] = []
    output_path = resolve_output_path(input_path)

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
            try:
                row = json.loads(stripped)
            except Exception as exc:
                raise ValueError(f"Invalid JSONL row at line {line_no}: {exc}") from exc

            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at line {line_no} is not a JSON object.")

            raw_text, _ = detect_raw_text_field(row)
            if not raw_text:
                raw_text = json.dumps(row, ensure_ascii=False)

            label_obj, _ = extract_label_from_text(raw_text)
            row_dump_label, _ = extract_label_from_text(json.dumps(row, ensure_ascii=False))
            if label_score(row_dump_label) > label_score(label_obj):
                label_obj = row_dump_label
            # Final safety fallback: pattern analysis for rows still missing labels.
            if is_missing_or_error(label_obj.get("scenario")) or is_missing_or_error(label_obj.get("action")):
                pattern_label = extract_label_from_patterns(raw_text + "\n" + json.dumps(row, ensure_ascii=False))
                label_obj = merge_labels(label_obj, pattern_label)
            has_new_label = is_nonempty_label(label_obj)
            if has_new_label:
                extracted += 1

            out_row = dict(row)
            old_scenario = str(out_row.get("scenario", "")).strip()
            old_action = str(out_row.get("action", "")).strip()
            old_entities = out_row.get("entities", [])

            # Clear placeholder error labels before replacement.
            if is_error_like(old_scenario):
                out_row["scenario"] = ""
            if is_error_like(old_action):
                out_row["action"] = ""

            if label_obj.get("scenario"):
                out_row["scenario"] = label_obj["scenario"]
            if label_obj.get("action"):
                out_row["action"] = label_obj["action"]
            if has_new_label:
                out_row["entities"] = label_obj.get("entities", [])
            elif "entities" not in out_row:
                out_row["entities"] = parse_entities(old_entities)

            # Last check: if still empty/error, analyze full row text once more and patch fields.
            if is_missing_or_error(out_row.get("scenario")) or is_missing_or_error(out_row.get("action")):
                patch_label = extract_label_from_patterns(json.dumps(out_row, ensure_ascii=False))
                if is_missing_or_error(out_row.get("scenario")) and patch_label.get("scenario"):
                    out_row["scenario"] = patch_label["scenario"]
                if is_missing_or_error(out_row.get("action")) and patch_label.get("action"):
                    out_row["action"] = patch_label["action"]

            if is_missing_or_error(out_row.get("scenario")) or is_missing_or_error(out_row.get("action")):
                unresolved += 1
                if len(unresolved_rows) < 200:
                    unresolved_rows.append(
                        {
                            "line_no": line_no,
                            "id": pick_first_nonempty(out_row.get("id"), out_row.get("sample_id"), out_row.get("uid")),
                            "slurp_id": pick_first_nonempty(out_row.get("slurp_id")),
                            "scenario": str(out_row.get("scenario", "")).strip(),
                            "action": str(out_row.get("action", "")).strip(),
                            "raw_head": str(raw_text)[:500],
                        }
                    )

            new_scenario = str(out_row.get("scenario", "")).strip()
            new_action = str(out_row.get("action", "")).strip()
            new_entities = out_row.get("entities", [])
            if (old_scenario != new_scenario) or (old_action != new_action) or (old_entities != new_entities):
                updated += 1

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"[DONE] input: {input_path}")
    print(f"[DONE] output_jsonl: {output_path}")
    print(f"[DONE] extracted: {extracted}/{total}")
    print(f"[DONE] updated_rows: {updated}/{total}")
    print(f"[DONE] unresolved_rows: {unresolved}/{total}")
    if unresolved_rows:
        unresolved_path = output_path.replace(".jsonl", ".unresolved.jsonl")
        with open(unresolved_path, "w", encoding="utf-8") as f:
            for row in unresolved_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[DONE] unresolved_preview: {unresolved_path} (up to 200 rows)")


if __name__ == "__main__":
    main()
