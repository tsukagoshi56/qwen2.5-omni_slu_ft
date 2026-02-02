#!/usr/bin/env python3
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional


STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "for",
    "of",
    "on",
    "in",
    "at",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "with",
    "as",
    "about",
}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def compose_intent(scenario: str, action: str) -> str:
    s = str(scenario or "").strip()
    a = str(action or "").strip()
    if not s or not a:
        return ""
    return f"{s}_{a}"


def pick_recording(recordings: List[Dict[str, Any]], index: int = 0) -> Optional[str]:
    if not recordings:
        return None
    if index < len(recordings):
        file_name = recordings[index].get("file")
        if file_name:
            return file_name
    for rec in recordings:
        file_name = rec.get("file")
        if file_name:
            return file_name
    return None


def extract_slot_types(record: Dict[str, Any]) -> List[str]:
    results: List[str] = []
    seen = set()
    for ent in record.get("entities", []) or []:
        t = str(ent.get("type", "")).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        results.append(t)
    return results


def resolve_audio_path(audio_root: str, file_name: str) -> Optional[str]:
    if not file_name:
        return None
    candidates = [
        os.path.join(audio_root, file_name),
        os.path.join(audio_root, "slurp_real", file_name),
        os.path.join("slurp", "audio", "slurp_real", file_name),
        os.path.join("slurp", "slurp_real", file_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def tokenize_simple(text: str) -> List[str]:
    if not text:
        return []
    toks = re.findall(r"[a-z0-9']+", text.lower())
    return [t for t in toks if t and t not in STOPWORDS]
