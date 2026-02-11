#!/usr/bin/env python3
"""
Audio/text mixed SLU training and distributed inference for multitask outputs.

- Task A (CoT): C/R/J rationale output.
- Task B (Label): J-only output.
- Multitask training loss: 0.5 * L_cot + 0.5 * L_label.
"""

import argparse
import glob
import json
import logging
import os
import random
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from common import build_db_definitions, load_metadata

try:
    import librosa
except ImportError:
    librosa = None

try:
    import jiwer

    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_audio_or_raise(audio_path: str, sr: int) -> Tuple[Any, int]:
    if librosa is None:
        raise ModuleNotFoundError(
            "librosa is required for audio processing. "
            "Install librosa for train/eval/test with audio, or run --recover_only."
        )
    return librosa.load(audio_path, sr=sr)


SYSTEM_PROMPT_TEXT = (
    'System: Predict SLU labels from transcript.'
)
SYSTEM_PROMPT_AUDIO = (
    'System: Predict SLU labels from audio.'
)
OUTPUT_SCHEMA = (
    '{"Intent": "<scenario>_<action>", "entities": '
    '[{"type": "<entity_type>", "filler": "<entity_value>"}, ...]}'
)
PROMPT_OUTPUT_FORMAT = (
    "Output Format:\n"
    "C: Intent candidates: intent1 | intent2 | intent3; Slot candidates: slot_type1(value1|value2) | slot_type2\n"
    "R: label1!reason1; label2!reason2; ...\n"
    f"J: {OUTPUT_SCHEMA}"
)
PROMPT_OUTPUT_FORMAT_CANDIDATES_ONLY = (
    "Output Format:\n"
    "C: Intent candidates: intent1 | intent2 | intent3; Slot candidates: slot_type1(value1|value2) | slot_type2\n"
    f"J: {OUTPUT_SCHEMA}"
)
PROMPT_OUTPUT_FORMAT_LABEL_ONLY = (
    "Output Format:\n"
    f"J: {OUTPUT_SCHEMA}"
)
PROMPT_DB_DEFINITIONS = "Intents: (none)\nSlot Types: (none)"


def set_prompt_db_definitions(db_definitions: str) -> None:
    global PROMPT_DB_DEFINITIONS
    text = str(db_definitions or "").strip()
    PROMPT_DB_DEFINITIONS = text if text else "Intents: (none)\nSlot Types: (none)"


def setup_file_logging(log_path: str) -> None:
    if not log_path:
        return
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    abs_log_path = os.path.abspath(log_path)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            existing = os.path.abspath(getattr(handler, "baseFilename", ""))
            if existing == abs_log_path:
                return
    file_handler = logging.FileHandler(abs_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger.addHandler(file_handler)


def format_nbest(candidates: List[str], max_items: int = 5) -> str:
    if not candidates:
        return "- (none)"
    lines = []
    for idx, text in enumerate(candidates[:max_items], start=1):
        lines.append(f"- {idx}. {text}")
    return "\n".join(lines)


def normalize_rationale_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        # Keep short and stable for prompting.
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def candidate_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "transcript", "hypothesis", "value"):
            text = value.get(key)
            if text is not None:
                return str(text).strip()
    return str(value).strip()


def build_prompt_text(item: Dict[str, Any], include_transcript: bool = False) -> str:
    transcript = str(item.get("transcript", "") or "").strip()
    task_mode = str(item.get("task_mode", "cot") or "cot").strip().lower()
    if task_mode == "label":
        output_format = PROMPT_OUTPUT_FORMAT_LABEL_ONLY
    elif task_mode in ("candidates", "cand"):
        output_format = PROMPT_OUTPUT_FORMAT_CANDIDATES_ONLY
    else:
        output_format = PROMPT_OUTPUT_FORMAT

    if include_transcript and transcript:
        return (
            f"{SYSTEM_PROMPT_TEXT}\n\n"
            "[Input Data]\n"
            f"- Transcript: {transcript}\n\n"
            f"{output_format}"
        )
    return (
        f"{SYSTEM_PROMPT_AUDIO}\n\n"
        "[Input Data]\n"
        "- Audio: <AUDIO>\n\n"
        f"{output_format}"
    )


def build_training_target(rationale_text: str, final_json: str) -> str:
    rationale = (rationale_text or "").strip()
    if not rationale:
        return f"J: {final_json}"

    lines = [line.strip() for line in rationale.splitlines() if line.strip()]
    has_c = any(line.startswith("C:") for line in lines)
    has_r = any(line.startswith("R:") for line in lines)
    has_j = any(line.startswith("J:") for line in lines)

    if has_c and has_r and has_j:
        return "\n".join(lines)
    if has_j:
        return "\n".join(lines)
    return "\n".join(lines + [f"J: {final_json}"])


def build_label_only_target(final_json: str) -> str:
    return f"J: {final_json}"


def build_candidates_only_target(rationale_text: str, final_json: str) -> str:
    c_line = ""
    for line in (rationale_text or "").splitlines():
        s = line.strip()
        if s.startswith("C:"):
            c_line = s
            break
    if not c_line:
        c_line = "C: (none)"
    return "\n".join([c_line, f"J: {final_json}"])


# ==============================================================================
# 1. Data Loading
# ==============================================================================


def resolve_audio_path(
    audio_root: str,
    filename: str,
    return_searched_paths: bool = False,
) -> Any:
    if not filename:
        if return_searched_paths:
            return None, []
        return None

    filename = str(filename).strip()
    searched_paths: List[str] = []
    if os.path.isabs(filename):
        searched_paths.append(filename)
        if os.path.exists(filename):
            if return_searched_paths:
                return filename, searched_paths
            return filename

    basename = os.path.basename(filename)
    candidates = [
        os.path.join(audio_root, filename),
        os.path.join(audio_root, basename),
        os.path.join(audio_root, "slurp_real", filename),
        os.path.join(audio_root, "slurp_real", basename),
        os.path.join("slurp", "audio", "slurp_real", filename),
        os.path.join("slurp", "audio", "slurp_real", basename),
    ]
    for path in candidates:
        searched_paths.append(path)
        if os.path.exists(path):
            if return_searched_paths:
                return path, searched_paths
            return path

    # Last-resort fallback: build one-time basename index under audio_root.
    if audio_root and os.path.isdir(audio_root):
        if not hasattr(resolve_audio_path, "_audio_index"):
            index: Dict[str, str] = {}
            for root, _, files in os.walk(audio_root):
                for fn in files:
                    if fn not in index:
                        index[fn] = os.path.join(root, fn)
            resolve_audio_path._audio_index = index
        fallback = resolve_audio_path._audio_index.get(basename)
        if fallback and os.path.exists(fallback):
            searched_paths.append(f"[indexed] {fallback}")
            if return_searched_paths:
                return fallback, searched_paths
            return fallback

    if not hasattr(resolve_audio_path, "_debug_count"):
        resolve_audio_path._debug_count = 0
    if resolve_audio_path._debug_count < 10:
        logger.warning("Could not find %s. Checked: %s", filename, candidates)
        resolve_audio_path._debug_count += 1
    if return_searched_paths:
        return None, searched_paths
    return None


def parse_entities(raw_entities: Any) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    if not isinstance(raw_entities, list):
        return results

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
        results.append({"type": ent_type, "filler": str(filler)})
    return results


def get_dict_value_ci(obj: Dict[str, Any], *names: str) -> Any:
    if not isinstance(obj, dict):
        return None
    for name in names:
        if name in obj:
            return obj[name]
    lowered: Dict[str, Any] = {}
    for k, v in obj.items():
        lowered[str(k).strip().lower()] = v
    for name in names:
        key = str(name).strip().lower()
        if key in lowered:
            return lowered[key]
    return None


def intent_to_scenario_action(intent: str) -> Tuple[str, str]:
    intent = (intent or "").strip()
    if "_" in intent:
        scenario, action = intent.split("_", 1)
        return scenario, action
    return "", ""


def extract_target_obj(record: Dict[str, Any]) -> Dict[str, Any]:
    final_obj = record.get("final")
    if not isinstance(final_obj, dict):
        final_obj = {}

    scenario = str(get_dict_value_ci(final_obj, "scenario") or "").strip()
    action = str(get_dict_value_ci(final_obj, "action") or "").strip()

    intent = str(get_dict_value_ci(final_obj, "intent") or "").strip()
    if (not scenario or not action) and intent:
        inferred_scenario, inferred_action = intent_to_scenario_action(intent)
        scenario = scenario or inferred_scenario
        action = action or inferred_action

    entities = parse_entities(get_dict_value_ci(final_obj, "entities") or [])

    return {
        "scenario": scenario,
        "action": action,
        "entities": entities,
    }


def parse_json_like(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    for _ in range(2):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, str):
            text = obj.strip()
            continue
        return obj
    return None


def pick_first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", ""))
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)
    return str(content)


def extract_messages_texts(record: Dict[str, Any]) -> Tuple[str, str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return "", ""
    user_text = ""
    assistant_text = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        text = content_to_text(msg.get("content"))
        if role == "user" and not user_text:
            user_text = text
        elif role == "assistant" and not assistant_text:
            assistant_text = text
    return user_text, assistant_text


def extract_candidates_from_user_text(user_text: str) -> List[str]:
    if not user_text:
        return []
    results: List[str] = []
    for raw_line in user_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^(?:[-*]|\d+[.)])\s*(.+)$", line)
        if m:
            cand = m.group(1).strip()
            if cand and len(cand.split()) >= 2:
                results.append(cand)
    return results[:10]


def extract_filename_from_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text.strip():
        return ""

    patterns = [
        r'"filename"\s*:\s*"([^"]+)"',
        r'\\"filename\\"\s*:\s*\\"([^\\"]+)\\"',
        r"'filename'\s*:\s*'([^']+)'",
        r'filename\s*[:=]\s*["\']([^"\']+)["\']',
        r"(audio-[A-Za-z0-9_-]+\.(?:flac|wav|mp3|m4a|ogg))",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return str(m.group(1)).strip()
    return ""


def extract_sample_id(record: Dict[str, Any], fallback_index: int) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    rationale_obj = parse_json_like(record.get("rationale_text"))
    rationale_meta = (
        rationale_obj.get("meta", {})
        if isinstance(rationale_obj, dict) and isinstance(rationale_obj.get("meta"), dict)
        else {}
    )

    sample_id = pick_first_nonempty(
        record.get("id"),
        record.get("slurp_id"),
        record.get("uid"),
        record.get("uuid"),
        meta.get("id"),
        meta.get("slurp_id"),
        rationale_obj.get("id") if isinstance(rationale_obj, dict) else None,
        rationale_meta.get("id"),
        rationale_meta.get("slurp_id"),
    )
    if sample_id:
        return sample_id
    return f"row_{fallback_index}"


def extract_filename(record: Dict[str, Any]) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    if not meta:
        meta_obj = parse_json_like(record.get("meta"))
        if isinstance(meta_obj, dict):
            meta = meta_obj
    rationale_obj = parse_json_like(record.get("rationale_text"))
    rationale_meta = (
        rationale_obj.get("meta", {})
        if isinstance(rationale_obj, dict) and isinstance(rationale_obj.get("meta"), dict)
        else {}
    )

    recordings = record.get("recordings")
    rec_file = None
    if isinstance(recordings, list) and recordings and isinstance(recordings[0], dict):
        rec_file = recordings[0].get("file")
    audios = record.get("audios")
    audio0 = None
    if isinstance(audios, list) and audios:
        audio0 = audios[0]

    user_text, _ = extract_messages_texts(record)

    filename = pick_first_nonempty(
        record.get("filename"),
        record.get("file"),
        record.get("audio_filename"),
        record.get("audio_file"),
        meta.get("filename"),
        meta.get("file"),
        rationale_obj.get("filename") if isinstance(rationale_obj, dict) else None,
        rationale_obj.get("file") if isinstance(rationale_obj, dict) else None,
        rationale_meta.get("filename"),
        rationale_meta.get("file"),
        rec_file,
        audio0,
    )
    if filename:
        return filename

    # Last fallback for non-JSON rationale text containing filename snippets.
    return pick_first_nonempty(
        extract_filename_from_text(record.get("rationale_text")),
        extract_filename_from_text(record.get("rationale")),
        extract_filename_from_text(record.get("meta")),
        extract_filename_from_text(user_text),
        extract_filename_from_text(json.dumps(record, ensure_ascii=False)),
    )


def _normalize_loaded_obj_to_records(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, str):
        try:
            return _normalize_loaded_obj_to_records(json.loads(obj))
        except json.JSONDecodeError:
            return []
    if isinstance(obj, dict):
        for key in ("data", "items", "records", "examples"):
            maybe_list = obj.get(key)
            if isinstance(maybe_list, list):
                return [x for x in maybe_list if isinstance(x, dict)]

        # If this already looks like a single record, do not flatten dict values.
        record_like_keys = {
            "id",
            "slurp_id",
            "filename",
            "file",
            "audio_filename",
            "audio_file",
            "candidates",
            "final",
            "rationale_text",
            "meta",
            "recordings",
            "messages",
            "audios",
        }
        if any(k in obj for k in record_like_keys):
            return [obj]

        # Handle map-style datasets: {"1234": {...}, "1235": {...}, ...}
        # Keep this strict to avoid breaking one-sample dict records.
        keys = list(obj.keys())
        values = list(obj.values())
        if (
            len(obj) >= 2
            and all(isinstance(v, dict) for v in values)
            and all(re.fullmatch(r"[0-9]{1,8}", str(k)) for k in keys)
        ):
            return [v for v in values if isinstance(v, dict)]
        return [obj]
    if isinstance(obj, list):
        results: List[Dict[str, Any]] = []
        for x in obj:
            if isinstance(x, dict):
                results.append(x)
            elif isinstance(x, str):
                try:
                    parsed = json.loads(x)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    results.append(parsed)
        return results
    return []


def is_record_like_dict(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    record_like_keys = {
        "id",
        "slurp_id",
        "filename",
        "file",
        "audio_filename",
        "audio_file",
        "candidates",
        "final",
        "rationale_text",
        "meta",
        "recordings",
        "messages",
        "audios",
    }
    if any(k in record for k in record_like_keys):
        return True

    # Accept cases where only rationale text contains recoverable filename.
    if extract_filename(record):
        return True
    return False


def extract_target_obj_from_assistant(record: Dict[str, Any]) -> Dict[str, Any]:
    _, assistant_text = extract_messages_texts(record)
    parsed = parse_json_like(assistant_text)
    if isinstance(parsed, dict):
        if "scenario" in parsed or "action" in parsed or "entities" in parsed:
            return {
                "scenario": str(get_dict_value_ci(parsed, "scenario") or "").strip(),
                "action": str(get_dict_value_ci(parsed, "action") or "").strip(),
                "entities": parse_entities(get_dict_value_ci(parsed, "entities") or []),
            }
        if "final" in parsed:
            return extract_target_obj(parsed)
        intent = get_dict_value_ci(parsed, "intent")
        if intent is not None:
            scenario, action = intent_to_scenario_action(str(intent))
            return {
                "scenario": scenario,
                "action": action,
                "entities": parse_entities(get_dict_value_ci(parsed, "entities") or []),
            }
    return {"scenario": "", "action": "", "entities": []}


def load_rationale_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if not raw.strip():
        return []

    # 1) Try full-file JSON (supports JSON array/object files).
    try:
        obj = json.loads(raw)
        records = _normalize_loaded_obj_to_records(obj)
        if records:
            logger.info("Loaded %s as full JSON (%d records).", path, len(records))
            return records
    except json.JSONDecodeError:
        pass

    # 2) Fallback: parse as JSONL.
    records: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        for rec in _normalize_loaded_obj_to_records(obj):
            if is_record_like_dict(rec):
                records.append(rec)
    if records:
        logger.info("Loaded %s as JSONL (%d records).", path, len(records))
        return records

    # 3) Last fallback: parse as a stream of concatenated JSON objects.
    decoder = json.JSONDecoder()
    i = 0
    n = len(raw)
    while i < n:
        while i < n and raw[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, j = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            i += 1
            continue
        for rec in _normalize_loaded_obj_to_records(obj):
            if is_record_like_dict(rec):
                records.append(rec)
        i = j
    logger.info("Loaded %s as JSON stream fallback (%d records).", path, len(records))
    return records


def expand_multitask_items(base_item: Dict[str, Any], cot_task_mode: str = "cot") -> List[Dict[str, Any]]:
    target_obj = base_item.get("target_obj", {})
    final_json = json.dumps(target_obj, ensure_ascii=False)
    cot_mode = str(cot_task_mode or "cot").strip().lower()
    if cot_mode == "candidates":
        cot_target = build_candidates_only_target(base_item.get("rationale_text", ""), final_json)
    else:
        cot_target = base_item.get("target", build_label_only_target(final_json))
    cot_item = {
        **base_item,
        "task_mode": "candidates" if cot_mode == "candidates" else "cot",
        "task_id": 0,
        "target": cot_target,
    }
    label_item = {
        **base_item,
        "task_mode": "label",
        "task_id": 1,
        "target": build_label_only_target(final_json),
    }
    return [cot_item, label_item]


def build_task_item(base_item: Dict[str, Any], task_mode: str) -> Dict[str, Any]:
    mode = str(task_mode or "").strip().lower()
    target_obj = base_item.get("target_obj", {})
    final_json = json.dumps(target_obj, ensure_ascii=False)
    if mode == "label":
        return {
            **base_item,
            "task_mode": "label",
            "task_id": 1,
            "target": build_label_only_target(final_json),
        }
    if mode == "candidates":
        return {
            **base_item,
            "task_mode": "candidates",
            "task_id": 0,
            "target": build_candidates_only_target(base_item.get("rationale_text", ""), final_json),
        }
    return {
        **base_item,
        "task_mode": "cot",
        "task_id": 0,
        "target": base_item.get("target", build_label_only_target(final_json)),
    }


def build_multisource_multitask_items(
    label_items: List[Dict[str, Any]],
    cot_items: List[Dict[str, Any]],
    cot_task_mode: str = "cot",
) -> List[Dict[str, Any]]:
    mixed: List[Dict[str, Any]] = []
    mixed.extend(build_task_item(item, "label") for item in label_items)
    mixed.extend(build_task_item(item, cot_task_mode) for item in cot_items)
    return mixed


def build_items_from_rationale_jsonl(
    jsonl_path: str,
    audio_dir: str,
    add_text_only: bool = False,
    text_only: bool = False,
    max_samples: Optional[int] = None,
    allow_text_fallback_when_audio_missing: bool = True,
    print_audio_search_paths: bool = False,
    audio_search_print_limit: int = 100,
    strict_audio_missing: bool = False,
    multitask: bool = True,
    cot_task_mode: str = "cot",
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    fallback_text_items: List[Dict[str, Any]] = []
    if not os.path.exists(jsonl_path):
        logger.warning("JSONL file not found: %s", jsonl_path)
        return items

    records = load_rationale_records(jsonl_path)

    parsed_rows = 0
    rows_with_filename = 0
    rows_with_audio = 0
    rows_missing_audio = 0
    row_parse_errors = 0

    for idx, data in enumerate(records):
        if max_samples is not None and len(items) >= max_samples:
            break
        try:
            if not isinstance(data, dict):
                continue
            if not is_record_like_dict(data):
                continue
            parsed_rows += 1

            # Extract messages and audios first
            user_text, assistant_text = extract_messages_texts(data)

            raw_audios = data.get("audios")
            
            # Check if this is a pre-formatted record (has messages and audios)
            is_preformatted = (
                isinstance(raw_audios, list) 
                and len(raw_audios) > 0 
                and assistant_text 
                and "candidates" not in data  # Assuming pre-formatted data might not have top-level candidates/final
            )

            # If pre-formatted, trust the messages/audios directly
            if is_preformatted or (raw_audios and assistant_text):
                # Use the first audio path
                filename = raw_audios[0]

                # Try to resolve generic info for logging/eval (optional)
                sample_id = extract_sample_id(data, fallback_index=parsed_rows)
                candidates = []  # Not needed for training if target is pre-built

                target_obj = extract_target_obj(data)
                if (
                    not target_obj.get("scenario")
                    and not target_obj.get("action")
                    and not target_obj.get("entities")
                ):
                    target_obj = extract_target_obj_from_assistant(data)
                final_json = json.dumps(target_obj, ensure_ascii=False)

                rationale_text = normalize_rationale_text(data.get("rationale_text"))
                if not rationale_text:
                    rationale_text = assistant_text.strip()
                target_str = build_training_target(rationale_text, final_json)

                # Prioritize explicit transcript fields
                transcript = pick_first_nonempty(
                    data.get("transcript"),
                    data.get("text"),
                    data.get("sentence"),
                )
            else:
                # --- Original Logic for Raw/Component Data ---
                sample_id = extract_sample_id(data, fallback_index=parsed_rows)
                filename = extract_filename(data)
                
                candidates = data.get("candidates", [])
                if not isinstance(candidates, list):
                    candidates = []
                candidates = [candidate_to_text(c) for c in candidates]
                candidates = [c for c in candidates if c]

                if not candidates:
                    candidates = extract_candidates_from_user_text(user_text)

                # Prioritize explicit transcript fields, fallback to candidates[0]
                transcript = pick_first_nonempty(
                    data.get("transcript"),
                    data.get("text"),
                    data.get("sentence"),
                    candidates[0] if candidates else ""
                )
                rationale_text = normalize_rationale_text(data.get("rationale_text"))

                target_obj = extract_target_obj(data)
                if not target_obj.get("scenario") and not target_obj.get("action") and not target_obj.get("entities"):
                    target_obj = extract_target_obj_from_assistant(data)

                final_json = json.dumps(target_obj, ensure_ascii=False)
                target_str = build_training_target(rationale_text, final_json)
            
            # Common file resolution
            if filename:
                audio_path, searched_paths = resolve_audio_path(
                    audio_dir,
                    filename,
                    return_searched_paths=True,
                )
            else:
                audio_path, searched_paths = None, []

            base_item = {
                "id": sample_id,
                "slurp_id": sample_id,
                "file": filename,
                "audio_path": audio_path,
                "transcript": transcript,
                "candidates": candidates,
                "rationale_text": rationale_text,
                "target": target_str,
                "target_obj": target_obj,
                "prompt_text": user_text.strip() if user_text else "",
            }
            text_only_item = {**base_item, "audio_path": None}
            text_only_items = (
                expand_multitask_items(text_only_item, cot_task_mode=cot_task_mode)
                if multitask
                else [text_only_item]
            )
            fallback_text_items.extend(text_only_items)

            if text_only:
                items.extend(text_only_items)
                continue

            if add_text_only:
                items.extend(text_only_items)

            if audio_path:
                audio_items = (
                    expand_multitask_items(base_item, cot_task_mode=cot_task_mode)
                    if multitask
                    else [base_item]
                )
                items.extend(audio_items)
                rows_with_audio += 1
                if print_audio_search_paths and rows_with_audio <= audio_search_print_limit:
                    print(f"[AUDIO_OK] id={sample_id} file={filename}")
                    print(f"  resolved={audio_path}")
            else:
                rows_missing_audio += 1
                if rows_missing_audio <= audio_search_print_limit:
                    if not filename:
                        print(f"[AUDIO_NG] id={sample_id} file=<empty> (filename parse failed)")
                    else:
                        print(f"[AUDIO_NG] id={sample_id} file={filename} (not found)")
                    for p in searched_paths:
                        print(f"  searched: {p}")
                if strict_audio_missing:
                    raise RuntimeError(
                        f"Audio not found for id={sample_id}, file={filename}. "
                        f"Searched paths: {searched_paths}"
                    )
        except Exception as exc:
            row_parse_errors += 1
            if row_parse_errors <= 20:
                head = str(data)
                if len(head) > 300:
                    head = head[:300] + "...(truncated)"
                logger.error(
                    "Row parse error at index=%d type=%s error=%s record_head=%s",
                    idx,
                    type(data).__name__,
                    exc,
                    head,
                )
            continue

    if (not add_text_only) and len(items) == 0 and parsed_rows > 0 and allow_text_fallback_when_audio_missing:
        logger.warning(
            "No audio could be resolved from %s. Falling back to text-only items (%d rows).",
            jsonl_path,
            len(fallback_text_items),
        )
        items.extend(fallback_text_items)

    logger.info(
        (
            "Loaded %s -> %d items "
            "(parsed_rows=%d, rows_with_filename=%d, rows_with_audio=%d, rows_missing_audio=%d, row_parse_errors=%d)"
        ),
        jsonl_path,
        len(items),
        parsed_rows,
        rows_with_filename,
        rows_with_audio,
        rows_missing_audio,
        row_parse_errors,
    )
    return items


class MixedDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {**self.items[idx], "original_idx": idx}


# ==============================================================================
# 2. Sampler
# ==============================================================================


class DistributedHomogeneousBatchSampler(Sampler):
    def __init__(
        self,
        dataset: MixedDataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = False,
        seed: int = 0,
        shuffle: bool = True,
        total_epochs: int = 1,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.shuffle = shuffle
        self.total_epochs = max(1, total_epochs)

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package")
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

        all_audio = [i for i, item in enumerate(dataset.items) if item.get("audio_path") is not None]
        all_text = [i for i, item in enumerate(dataset.items) if item.get("audio_path") is None]

        self.local_audio_indices = all_audio[self.rank :: self.num_replicas]
        self.local_text_indices = all_text[self.rank :: self.num_replicas]

    def __iter__(self) -> Iterator[List[int]]:
        g_static = torch.Generator()
        g_static.manual_seed(self.seed)

        audio_indices_tensor = torch.tensor(self.local_audio_indices)
        if self.shuffle and len(audio_indices_tensor) > 0:
            perm_static = torch.randperm(len(audio_indices_tensor), generator=g_static)
            shuffled_audio = audio_indices_tensor[perm_static]
        else:
            shuffled_audio = audio_indices_tensor

        total_audio_count = len(shuffled_audio)
        chunk_size = max(1, total_audio_count // self.total_epochs) if total_audio_count else 0
        current_chunk_idx = self.epoch % self.total_epochs

        start_idx = current_chunk_idx * chunk_size
        if current_chunk_idx == self.total_epochs - 1:
            end_idx = total_audio_count
        else:
            end_idx = start_idx + chunk_size

        active_audio_indices = shuffled_audio[start_idx:end_idx]
        active_text_indices = torch.tensor(self.local_text_indices)

        g_dynamic = torch.Generator()
        g_dynamic.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            if len(active_audio_indices) > 0:
                audio_perm = torch.randperm(len(active_audio_indices), generator=g_dynamic)
                audio_idxs = active_audio_indices[audio_perm].tolist()
            else:
                audio_idxs = []
            if len(active_text_indices) > 0:
                text_perm = torch.randperm(len(active_text_indices), generator=g_dynamic)
                text_idxs = active_text_indices[text_perm].tolist()
            else:
                text_idxs = []
        else:
            audio_idxs = active_audio_indices.tolist()
            text_idxs = active_text_indices.tolist()

        batches: List[List[int]] = []
        for i in range(0, len(audio_idxs), self.batch_size):
            batch = audio_idxs[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        for i in range(0, len(text_idxs), self.batch_size):
            batch = text_idxs[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        total_audio = len(self.local_audio_indices)
        chunk_size = total_audio // self.total_epochs if self.total_epochs > 0 else total_audio
        current_audio_len = chunk_size
        current_text_len = len(self.local_text_indices)
        if self.drop_last:
            audio_batches = current_audio_len // self.batch_size
            text_batches = current_text_len // self.batch_size
        else:
            audio_batches = (current_audio_len + self.batch_size - 1) // self.batch_size
            text_batches = (current_text_len + self.batch_size - 1) // self.batch_size
        return audio_batches + text_batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch


# ==============================================================================
# 3. Collator
# ==============================================================================


@dataclass
class SmartCollator:
    processor: Any
    max_length: int = 512
    ignore_index: int = -100
    debug: bool = False
    _print_count: int = 0

    def __post_init__(self):
        self._print_count = 0

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        if len(batch) == 0:
            return {}
        is_audio_batch = batch[0].get("audio_path") is not None
        if is_audio_batch:
            return self._collate_audio(batch)
        return self._collate_text(batch)

    def _build_audio_chat(self, item: Dict[str, Any]) -> str:
        prompt_text = build_prompt_text(item)
        user_content = [
            {"type": "audio", "audio_url": "placeholder"},
            {"type": "text", "text": prompt_text},
        ]
        return self.processor.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_text_chat(self, item: Dict[str, Any]) -> str:
        prompt_text = build_prompt_text(item, include_transcript=True)
        user_content = [{"type": "text", "text": prompt_text}]
        return self.processor.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _collate_audio(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list, labels_list = [], []
        input_features_list, feature_mask_list = [], []
        task_ids: List[int] = []

        sr = self.processor.feature_extractor.sampling_rate
        eos_token = self.processor.tokenizer.eos_token or "<|endoftext|>"

        for item in batch:
            if item.get("audio_path") is None:
                continue
            try:
                audio, _ = load_audio_or_raise(item["audio_path"], sr=sr)
            except Exception:
                continue

            text_input = self._build_audio_chat(item)
            full_text = text_input + item["target"] + eos_token

            if self.debug and self._print_count < 5:
                print(f"\n[DEBUG Visualizer] Audio Sample ID: {item.get('id')}")
                print(f"[DEBUG Visualizer] Task: {item.get('task_mode', 'cot')}")
                print(f"[DEBUG Visualizer] Input Prompt:\n{text_input}")
                print(f"[DEBUG Visualizer] Target:\n{item['target']}")
                self._print_count += 1

            inputs = self.processor(
                text=full_text,
                audio=[audio],
                sampling_rate=sr,
                return_tensors="pt",
            )
            prompt_inputs = self.processor(
                text=text_input,
                audio=[audio],
                sampling_rate=sr,
                return_tensors="pt",
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]

            ids = inputs["input_ids"][0]
            lbs = ids.clone()
            lbs[:prompt_len] = self.ignore_index

            input_ids_list.append(ids)
            labels_list.append(lbs)
            task_ids.append(int(item.get("task_id", 0)))

            feat = inputs["input_features"]
            while feat.dim() > 2:
                feat = feat.squeeze(0)
            input_features_list.append(feat)

            if "feature_attention_mask" in inputs:
                f_mask = inputs["feature_attention_mask"]
                while f_mask.dim() > 1:
                    f_mask = f_mask.squeeze(0)
                feature_mask_list.append(f_mask)

        return {
            "input_ids": pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id,
            ),
            "labels": pad_sequence(
                labels_list,
                batch_first=True,
                padding_value=self.ignore_index,
            ),
            "attention_mask": pad_sequence(
                [torch.ones_like(ids) for ids in input_ids_list],
                batch_first=True,
                padding_value=0,
            ),
            "input_features": pad_sequence(
                input_features_list,
                batch_first=True,
                padding_value=0.0,
            ),
            "feature_attention_mask": (
                pad_sequence(feature_mask_list, batch_first=True, padding_value=0)
                if feature_mask_list
                else None
            ),
            "task_ids": torch.tensor(task_ids, dtype=torch.long),
        }

    def _collate_text(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list, labels_list = [], []
        task_ids: List[int] = []

        eos_token = self.processor.tokenizer.eos_token or "<|endoftext|>"
        for item in batch:
            if item.get("audio_path") is not None:
                continue
            text_input = self._build_text_chat(item)
            full_text = text_input + item["target"] + eos_token

            if self.debug and self._print_count < 5:
                print(f"\n[DEBUG Visualizer] Text Sample ID: {item.get('id')}")
                print(f"[DEBUG Visualizer] Task: {item.get('task_mode', 'cot')}")
                print(f"[DEBUG Visualizer] Input Prompt:\n{text_input}")
                print(f"[DEBUG Visualizer] Target:\n{item['target']}")
                self._print_count += 1

            inputs = self.processor.tokenizer(full_text, return_tensors="pt")
            prompt_inputs = self.processor.tokenizer(text_input, return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]

            ids = inputs["input_ids"][0]
            lbs = ids.clone()
            lbs[:prompt_len] = self.ignore_index
            input_ids_list.append(ids)
            labels_list.append(lbs)
            task_ids.append(int(item.get("task_id", 0)))

        return {
            "input_ids": pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id,
            ),
            "labels": pad_sequence(
                labels_list,
                batch_first=True,
                padding_value=self.ignore_index,
            ),
            "attention_mask": pad_sequence(
                [torch.ones_like(ids) for ids in input_ids_list],
                batch_first=True,
                padding_value=0,
            ),
            "task_ids": torch.tensor(task_ids, dtype=torch.long),
        }


# ==============================================================================
# 4. Trainer
# ==============================================================================


class CustomTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        batch_sampler = DistributedHomogeneousBatchSampler(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            drop_last=self.args.dataloader_drop_last,
            shuffle=True,
            total_epochs=int(self.args.num_train_epochs),
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        task_ids = inputs.pop("task_ids", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if task_ids is None or labels is None or not hasattr(outputs, "logits"):
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size(0), shift_labels.size(1))

        valid_mask = (shift_labels != -100).float()
        per_sample_loss = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)

        task_ids = task_ids.to(per_sample_loss.device)
        cot_mask = task_ids == 0
        label_mask = task_ids == 1

        if cot_mask.any() and label_mask.any():
            cot_loss = per_sample_loss[cot_mask].mean()
            label_loss = per_sample_loss[label_mask].mean()
            loss = 0.5 * cot_loss + 0.5 * label_loss
        elif cot_mask.any():
            loss = per_sample_loss[cot_mask].mean()
        elif label_mask.any():
            loss = per_sample_loss[label_mask].mean()
        else:
            loss = per_sample_loss.mean()

        return (loss, outputs) if return_outputs else loss


# ==============================================================================
# 5. Callback
# ==============================================================================


class SampleGenerationCallback(TrainerCallback):
    def __init__(self, eval_items, processor, model, num_samples: int = 3, max_new_tokens: int = 4096):
        self.eval_items = eval_items
        self.processor = processor
        self.model = model
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return

        logger.info("\n\n*** Validation Sample Generation (Audio) ***")
        audio_items = [item for item in self.eval_items if item.get("audio_path") is not None]
        if not audio_items:
            logger.info("No audio items found in validation set.")
            return

        samples = random.sample(audio_items, min(self.num_samples, len(audio_items)))
        device = self.model.device
        self.model.eval()
        sr = self.processor.feature_extractor.sampling_rate

        for item in samples:
            try:
                audio, _ = load_audio_or_raise(item["audio_path"], sr=sr)
                prompt_text = build_prompt_text(item)
                user_content = [
                    {"type": "audio", "audio_url": "placeholder"},
                    {"type": "text", "text": prompt_text},
                ]
                text_input = self.processor.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                inputs = self.processor(
                    text=text_input,
                    audio=[audio],
                    sampling_rate=sr,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

                input_len = inputs["input_ids"].shape[1]
                generated_text = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
                clean_pred = clean_json_text(generated_text)

                logger.info("-" * 60)
                logger.info("File:       %s", item.get("file"))
                logger.info("Transcript: %s", item.get("transcript"))
                logger.info("Prediction: %s", clean_pred)
            except Exception as exc:
                logger.error("Failed to generate sample for %s: %s", item.get("file"), exc)

        logger.info("-" * 60 + "\n")
        self.model.train()


# ==============================================================================
# 6. Inference + Evaluation helpers
# ==============================================================================


def clean_json_text(text: str) -> str:
    text = text.strip()
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def _extract_labeled_tail(text: str, labels: List[str]) -> str:
    if not isinstance(text, str):
        return ""
    for label in labels:
        # Accept `J:`, `j:`, `J ：` and similar variants.
        pattern = rf"(?is)(?:^|\n)\s*{re.escape(label)}\s*[:：]\s*(.+)$"
        m = re.search(pattern, text)
        if m:
            return str(m.group(1)).strip()
    return ""


def _parse_first_json_dict(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    if not candidate:
        return None

    decoder = json.JSONDecoder()
    for probe in (clean_json_text(candidate), candidate):
        probe = probe.strip()
        if not probe:
            continue
        # Some model outputs include doubled braces from prompt examples.
        normalized_probe = probe.replace("{{", "{").replace("}}", "}")
        for target in (probe, normalized_probe):
            try:
                obj = json.loads(target)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

            # Find the first decodable JSON object anywhere in the text.
            for m in re.finditer(r"\{", target):
                start = m.start()
                try:
                    obj, _ = decoder.raw_decode(target[start:])
                except Exception:
                    continue
                if isinstance(obj, dict):
                    return obj
    return None


def _is_error_label(label: Dict[str, Any]) -> bool:
    if not isinstance(label, dict):
        return True
    scenario = str(label.get("scenario", "") or "").strip().lower()
    action = str(label.get("action", "") or "").strip().lower()
    entities = parse_entities(label.get("entities", []))
    return scenario == "error" and action == "error" and len(entities) == 0


def _label_info_score(label: Dict[str, Any]) -> int:
    if not isinstance(label, dict):
        return -1
    scenario = str(label.get("scenario", "") or "").strip()
    action = str(label.get("action", "") or "").strip()
    entities = parse_entities(label.get("entities", []))
    return int(bool(scenario)) + int(bool(action)) + int(bool(entities))


def parse_prediction_label(raw_output: str) -> Dict[str, Any]:
    default_obj = {"scenario": "error", "action": "error", "entities": []}

    text = str(raw_output or "")
    probes: List[str] = []

    j_tail = _extract_labeled_tail(text, ["J", "SLU", "FINAL", "Final", "Output"])
    if j_tail:
        probes.append(j_tail)
    probes.append(text)

    parsed = None
    for probe in probes:
        parsed = _parse_first_json_dict(probe)
        if isinstance(parsed, dict):
            break

    if not isinstance(parsed, dict):
        return default_obj

    # Extraction Logic
    # 1. Check for "final" wrapper FIRST (To handle: "final": {"intent": "..."})
    for wrapper_key in ("final", "Final", "j", "J", "output", "prediction", "result"):
        wrapped = parsed.get(wrapper_key)
        if isinstance(wrapped, dict):
            parsed = wrapped

    scenario = get_dict_value_ci(parsed, "scenario")
    action = get_dict_value_ci(parsed, "action")
    entities = get_dict_value_ci(parsed, "entities", "slots") or []

    # 2. If scenario/action are missing, try to parse "intent"
    if not scenario and not action:
        intent = get_dict_value_ci(parsed, "intent")
        if isinstance(intent, str):
            intent = intent.strip()
            if "_" in intent:
                # Split by FIRST underscore
                scenario, action = intent.split("_", 1)
            else:
                # Fallback: keep scenario empty, use intent as action
                scenario = ""
                action = intent

    return {
        "scenario": str(scenario or "").strip(),
        "action": str(action or "").strip(),
        "entities": parse_entities(entities),
    }


def recover_prediction_file(input_path: str, output_path: str) -> Dict[str, int]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"recover input not found: {input_path}")

    total = 0
    changed = 0
    intent_key_recovered = 0
    rows: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue

            old_s = str(row.get("scenario", "") or "").strip()
            old_a = str(row.get("action", "") or "").strip()
            old_e = parse_entities(row.get("entities", []))

            candidate_outputs: List[str] = []
            for key in (
                "raw_output",
                "prediction",
                "pred_text",
                "model_output",
                "output",
                "response",
                "assistant",
                "assistant_text",
                "text",
            ):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    candidate_outputs.append(value)

            parsed_candidates: List[Dict[str, Any]] = []
            for text in candidate_outputs:
                parsed = parse_prediction_label(text)
                if not _is_error_label(parsed):
                    parsed_candidates.append(parsed)

            parsed_from_row = parse_prediction_label(json.dumps(row, ensure_ascii=False))
            if not _is_error_label(parsed_from_row):
                parsed_candidates.append(parsed_from_row)

            chosen = {}
            if parsed_candidates:
                chosen = sorted(parsed_candidates, key=_label_info_score, reverse=True)[0]

            new_s = str(chosen.get("scenario", "") or old_s).strip()
            new_a = str(chosen.get("action", "") or old_a).strip()
            new_e = chosen.get("entities") or old_e
            new_e = parse_entities(new_e)

            if (not old_s and not old_a) and (new_s or new_a):
                intent_key_recovered += 1

            row["scenario"] = new_s
            row["action"] = new_a
            row["entities"] = new_e
            row["pred_label"] = {"scenario": new_s, "action": new_a, "entities": new_e}

            if (old_s != new_s) or (old_a != new_a) or (old_e != new_e):
                changed += 1

            rows.append(row)

    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "num_rows": total,
        "num_changed": changed,
        "num_intent_key_recovered": intent_key_recovered,
    }


def calculate_wer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0
    if HAS_JIWER:
        return float(jiwer.wer(reference, hypothesis))
    return 0.0 if reference.strip() == hypothesis.strip() else 1.0



@dataclass
class InferenceCollator:
    processor: Any

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        self.processor.tokenizer.padding_side = "left"
        sr = self.processor.feature_extractor.sampling_rate

        texts = []
        audios = []
        valid_items = []

        for item in batch:
            if item.get("audio_path"):
                try:
                    audio, _ = load_audio_or_raise(item["audio_path"], sr=sr)
                    audios.append(audio)
                    prompt_text = build_prompt_text(item)
                    user_content = [
                        {"type": "audio", "audio_url": "placeholder"},
                        {"type": "text", "text": prompt_text},
                    ]
                except Exception as e:
                    logger.warning(f"Failed to load audio for {item.get('id')}: {e}")
                    continue
            else:
                prompt_text = build_prompt_text(item, include_transcript=True)
                user_content = [{"type": "text", "text": prompt_text}]

            text_input = self.processor.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text_input)
            valid_items.append(item)

        if not texts:
            return {}

        inputs = self.processor(
            text=texts,
            audio=audios if audios else None,
            sampling_rate=sr,
            padding=True,
            return_tensors="pt",
        )
        return {"net_inputs": inputs, "items": valid_items}


def _generate_batch(
    model,
    processor,
    batch_data: Dict[str, Any],
    device,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    if not batch_data:
        return []

    net_inputs = batch_data["net_inputs"]
    items = batch_data["items"]

    net_inputs = {k: v.to(device) for k, v in net_inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **net_inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = net_inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    raw_outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)

    results: List[Dict[str, Any]] = []
    for item, raw_output in zip(items, raw_outputs):
        pred_label = parse_prediction_label(raw_output)
        wer_score = calculate_wer(item.get("transcript", ""), raw_output)

        result_entry = {
            "scenario": pred_label["scenario"],
            "action": pred_label["action"],
            "entities": pred_label["entities"],
            "pred_label": pred_label,
            "file": item.get("file"),
            "slurp_id": item.get("slurp_id"),
            "id": item.get("id"),
            "wer": wer_score,
            "transcript": item.get("transcript", ""),
            "candidates": item.get("candidates", []),
            "rationale_text": item.get("rationale_text", ""),
            "raw_output": raw_output,
            "target": item.get("target", ""),
            "target_label": item.get("target_obj", {}),
            "type": "audio" if item.get("audio_path") else "text",
        }
        results.append(result_entry)

    return results


def run_distributed_inference(
    model,
    processor,
    items,
    output_path,
    device,
    rank,
    world_size,
    batch_size=1,
    max_new_tokens: int = 2048,
    num_workers: int = 0,
):
    model.eval()

    my_items = items[rank::world_size]
    # Split audio/text to keep clean batches
    my_audio_items = [x for x in my_items if x.get("audio_path") is not None]
    my_text_items = [x for x in my_items if x.get("audio_path") is None]

    local_results: List[Dict[str, Any]] = []
    processor.tokenizer.padding_side = "left"

    if rank == 0:
        logger.info("Starting Inference. Items: %d (Audio: %d, Text: %d), Batch size: %d",
                    len(my_items), len(my_audio_items), len(my_text_items), batch_size)

    # Audio Loader
    if my_audio_items:
        audio_loader = DataLoader(
            MixedDataset(my_audio_items),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=InferenceCollator(processor),
            drop_last=False,
            shuffle=False,
        )
        for i, batch_data in enumerate(audio_loader):
            if rank == 0 and i % 10 == 0:
                logger.info("Audio batch %d/%d", i + 1, len(audio_loader))
            try:
                local_results.extend(
                    _generate_batch(
                        model=model,
                        processor=processor,
                        batch_data=batch_data,
                        device=device,
                        max_new_tokens=max_new_tokens,
                    )
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                logger.error("Rank %d failed on audio batch %d: %s", rank, i, exc)

    # Text Loader
    if my_text_items:
        text_loader = DataLoader(
            MixedDataset(my_text_items),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=InferenceCollator(processor),
            drop_last=False,
            shuffle=False,
        )
        for i, batch_data in enumerate(text_loader):
            if rank == 0 and i % 10 == 0:
                logger.info("Text batch %d/%d", i + 1, len(text_loader))
            try:
                local_results.extend(
                    _generate_batch(
                        model=model,
                        processor=processor,
                        batch_data=batch_data,
                        device=device,
                        max_new_tokens=max_new_tokens,
                    )
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                logger.error("Rank %d failed on text batch %d: %s", rank, i, exc)

    temp_output_path = f"{output_path}.rank{rank}"
    try:
        with open(temp_output_path, "w", encoding="utf-8") as f:
            for res in local_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as exc:
        logger.error("Rank %d failed to save temp file: %s", rank, exc)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        logger.info("Merging results to %s", output_path)
        pattern = f"{output_path}.rank*"
        temp_files = sorted(glob.glob(pattern))
        with open(output_path, "w", encoding="utf-8") as outfile:
            for fname in temp_files:
                try:
                    with open(fname, "r", encoding="utf-8") as infile:
                        shutil.copyfileobj(infile, outfile)
                    os.remove(fname)
                except Exception as exc:
                    logger.error("Merge error %s: %s", fname, exc)


def save_label_only_predictions(full_prediction_path: str, label_only_path: str):
    rows = []
    with open(full_prediction_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "id": row.get("id"),
                    "file": row.get("file"),
                    "slurp_id": row.get("slurp_id"),
                    "scenario": row.get("scenario", ""),
                    "action": row.get("action", ""),
                    "entities": parse_entities(row.get("entities", [])),
                }
            )

    with open(label_only_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_entity(entity: Dict[str, Any]) -> Tuple[str, str]:
    if not isinstance(entity, dict):
        return "", ""
    ent_type = str(entity.get("type", "")).strip().lower()
    filler = str(entity.get("filler", "")).strip().lower()
    filler = re.sub(r"\s+", " ", filler)
    return ent_type, filler


def evaluate_prediction_file(prediction_path: str) -> Dict[str, float]:
    total = 0
    scenario_correct = 0
    action_correct = 0
    intent_correct = 0

    tp = 0
    fp = 0
    fn = 0

    with open(prediction_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue

            target_label = row.get("target_label")
            if not isinstance(target_label, dict):
                try:
                    target_label = json.loads(row.get("target", "{}"))
                except Exception:
                    target_label = {}

            pred_scenario = str(row.get("scenario", "")).strip()
            pred_action = str(row.get("action", "")).strip()
            gold_scenario = str(target_label.get("scenario", "")).strip()
            gold_action = str(target_label.get("action", "")).strip()

            total += 1
            scenario_correct += int(pred_scenario == gold_scenario)
            action_correct += int(pred_action == gold_action)
            intent_correct += int(
                (pred_scenario + "_" + pred_action) == (gold_scenario + "_" + gold_action)
            )

            pred_entities = {
                _normalize_entity(e) for e in parse_entities(row.get("entities", []))
            }
            gold_entities = {
                _normalize_entity(e) for e in parse_entities(target_label.get("entities", []))
            }

            tp += len(pred_entities & gold_entities)
            fp += len(pred_entities - gold_entities)
            fn += len(gold_entities - pred_entities)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    entity_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if total == 0:
        return {
            "num_samples": 0,
            "scenario_acc": 0.0,
            "action_acc": 0.0,
            "intent_acc": 0.0,
            "entity_precision": 0.0,
            "entity_recall": 0.0,
            "entity_f1": 0.0,
        }

    return {
        "num_samples": total,
        "scenario_acc": scenario_correct / total,
        "action_acc": action_correct / total,
        "intent_acc": intent_correct / total,
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": entity_f1,
    }


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        type=str,
        default="Experiment_RationaleCompare/sft_success_train.jsonl",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="Experiment_RationaleCompare/sft_success_train.jsonl",
    )
    parser.add_argument(
        "--cot_train_file",
        "--cot-train-file",
        dest="cot_train_file",
        type=str,
        default="",
        help=(
            "Optional CoT-only train file for multitask training. "
            "When set, --train_file is used for label task and this file is used for CoT task."
        ),
    )
    parser.add_argument(
        "--cot_eval_file",
        "--cot-eval-file",
        dest="cot_eval_file",
        type=str,
        default="",
        help=(
            "Optional CoT-only eval file for multitask validation. "
            "When set, --eval_file is used for label task and this file is used for CoT task."
        ),
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="slurp/dataset/slurp/test.jsonl",
        help="Path to test jsonl.",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/lustre/home/71200138/INTERSPEECH/experiment1/slurp/audio/slurp_real",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="Experiment_3/slurp_metadata.json",
        help="Metadata JSON used to build DB Definitions for prompts.",
    )

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_rationale_label_ft")
    parser.add_argument(
        "--output_file",
        "--output-file",
        "--outpt_file",
        "--outpt-file",
        dest="output_file",
        type=str,
        default="",
        help="Prediction output JSONL path (default: <output_dir>/prediction.jsonl).",
    )
    parser.add_argument(
        "--log_file",
        "--log-file",
        dest="log_file",
        type=str,
        default="",
        help="Training log file path (default: <output_dir>/train.log).",
    )
    parser.add_argument(
        "--recover_prediction_file",
        "--recover-prediction-file",
        dest="recover_prediction_file",
        type=str,
        default="",
        help="Existing prediction JSONL to recover/fix parser outputs from.",
    )
    parser.add_argument(
        "--recover_output_file",
        "--recover-output-file",
        dest="recover_output_file",
        type=str,
        default="",
        help="Recovered prediction JSONL path (default: <recover_input>.recovered.jsonl).",
    )
    parser.add_argument(
        "--recover_inplace",
        "--recover-inplace",
        dest="recover_inplace",
        action="store_true",
        help="Overwrite recover input file instead of writing a new output file.",
    )
    parser.add_argument(
        "--recover_only",
        "--recover-only",
        dest="recover_only",
        action="store_true",
        help="Run recovery mode only and exit (no train/inference).",
    )
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=None,
        help="Cap eval set size to speed up validation (None means no extra cap).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument(
        "--inference_num_workers",
        type=int,
        default=0,
        help="DataLoader workers for test inference (0 is safer to avoid deadlocks).",
    )
    parser.add_argument(
        "--train_audio_encoder",
        action="store_true",
        help="Enable training of audio_tower (audio encoder).",
    )
    parser.add_argument(
        "--export_label_eval",
        action="store_true",
        help="Also export label-only predictions and metrics after inference.",
    )
    parser.add_argument(
        "--train_candidates_only",
        "--train-candidates-only",
        "--no_r_train",
        "--no-r-train",
        dest="train_candidates_only",
        action="store_true",
        help="For multitask train/eval branches, use C+J targets/prompts (no R) instead of C/R/J.",
    )
    parser.add_argument(
        "--no_train_candidates_only",
        "--no-train-candidates-only",
        dest="train_candidates_only",
        action="store_false",
        help="Disable C+J train/eval mode and use standard C/R/J for CoT branch.",
    )
    parser.add_argument(
        "--test_task_mode",
        type=str,
        choices=["cot", "candidates", "label"],
        default="cot",
        help="Prompt/output mode used only for test inference (default: cot).",
    )
    parser.add_argument(
        "--candidates_only",
        "--candidates-only",
        dest="test_task_mode",
        action="store_const",
        const="candidates",
        help="Alias for --test_task_mode candidates (C+J generation at test time).",
    )
    parser.add_argument(
        "--no_cot",
        "--no-cot",
        dest="test_task_mode",
        action="store_const",
        const="label",
        help="Alias for --test_task_mode label (J-only generation at test time).",
    )
    parser.add_argument(
        "--with_cot",
        "--with-cot",
        dest="test_task_mode",
        action="store_const",
        const="cot",
        help="Alias for --test_task_mode cot (C/R/J generation at test time).",
    )
    parser.add_argument("--add_text_only", action="store_true", help="Also add text-only samples.")
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Use text-only samples for all splits (audio paths are ignored).",
    )
    parser.add_argument(
        "--no_text_fallback_when_audio_missing",
        action="store_true",
        help="Disable automatic text-only fallback when audio files cannot be resolved.",
    )
    parser.add_argument(
        "--print_audio_search_paths",
        action="store_true",
        help="Print searched audio paths to stdout.",
    )
    parser.add_argument(
        "--audio_search_print_limit",
        type=int,
        default=100,
        help="Maximum number of audio path debug prints per split.",
    )
    parser.add_argument(
        "--strict_audio_missing",
        action="store_true",
        help="Raise an error immediately when an audio file cannot be resolved.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run tiny smoke test.")

    args = parser.parse_args()
    cot_train_task_mode = "candidates" if args.train_candidates_only else "cot"

    if args.recover_only:
        recover_input = args.recover_prediction_file.strip()
        if not recover_input:
            raise ValueError("--recover_only requires --recover_prediction_file")
        if args.recover_inplace:
            recover_output = recover_input
        else:
            recover_output = args.recover_output_file.strip()
            if not recover_output:
                base, ext = os.path.splitext(recover_input)
                recover_output = f"{base}.recovered{ext}" if ext else f"{recover_input}.recovered.jsonl"
        stats = recover_prediction_file(recover_input, recover_output)
        logger.info(
            "Recover done: input=%s output=%s rows=%d changed=%d intent_key_recovered=%d",
            recover_input,
            recover_output,
            stats["num_rows"],
            stats["num_changed"],
            stats["num_intent_key_recovered"],
        )
        return


    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        log_path = args.log_file.strip() if args.log_file.strip() else os.path.join(args.output_dir, "train.log")
        setup_file_logging(log_path)
        logger.info("File logging enabled: %s", os.path.abspath(log_path))

    metadata = load_metadata(args.metadata_file)
    db_definitions = build_db_definitions(metadata)
    set_prompt_db_definitions(db_definitions)
    if rank == 0:
        logger.info("Using prompts with DB Definitions from: %s", args.metadata_file)
        if not os.path.exists(args.metadata_file):
            logger.warning("metadata_file not found: %s (using empty DB Definitions)", args.metadata_file)
        if args.text_only:
            logger.info("text_only=True: all splits will use text-only items.")

    if rank == 0:
        logger.info("Using test_file: %s", args.test_file)

    train_max_samples = args.max_samples
    eval_max_samples = args.eval_max_samples
    if eval_max_samples is None:
        eval_max_samples = (args.max_samples // 2) if args.max_samples else None

    if args.smoke:
        if rank == 0:
            logger.info("SMOKE MODE ON")
        # Increase smoke learn data to 2000, keep eval small
        train_max_samples = 2000
        eval_max_samples = 200
        args.num_train_epochs = 1

    cot_train_file = args.cot_train_file.strip()
    cot_eval_file = args.cot_eval_file.strip()

    if cot_train_file:
        if rank == 0:
            logger.info(
                "Using split multitask train files | label=%s | cot=%s",
                args.train_file,
                cot_train_file,
            )
        label_train_items = build_items_from_rationale_jsonl(
            args.train_file,
            args.audio_dir,
            add_text_only=args.add_text_only,
            text_only=args.text_only,
            max_samples=train_max_samples,
            allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
            print_audio_search_paths=args.print_audio_search_paths,
            audio_search_print_limit=args.audio_search_print_limit,
            strict_audio_missing=args.strict_audio_missing,
            multitask=False,
        )
        cot_train_items = build_items_from_rationale_jsonl(
            cot_train_file,
            args.audio_dir,
            add_text_only=args.add_text_only,
            text_only=args.text_only,
            max_samples=train_max_samples,
            allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
            print_audio_search_paths=args.print_audio_search_paths,
            audio_search_print_limit=args.audio_search_print_limit,
            strict_audio_missing=args.strict_audio_missing,
            multitask=False,
        )
        train_items = build_multisource_multitask_items(
            label_train_items,
            cot_train_items,
            cot_task_mode=cot_train_task_mode,
        )
    else:
        train_items = build_items_from_rationale_jsonl(
            args.train_file,
            args.audio_dir,
            add_text_only=args.add_text_only,
            text_only=args.text_only,
            max_samples=train_max_samples,
            allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
            print_audio_search_paths=args.print_audio_search_paths,
            audio_search_print_limit=args.audio_search_print_limit,
            strict_audio_missing=args.strict_audio_missing,
            multitask=True,
            cot_task_mode=cot_train_task_mode,
        )

    if cot_eval_file:
        if rank == 0:
            logger.info(
                "Using split multitask eval files | label=%s | cot=%s",
                args.eval_file,
                cot_eval_file,
            )
        label_eval_items = build_items_from_rationale_jsonl(
            args.eval_file,
            args.audio_dir,
            add_text_only=args.add_text_only,
            text_only=args.text_only,
            max_samples=eval_max_samples,
            allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
            print_audio_search_paths=args.print_audio_search_paths,
            audio_search_print_limit=args.audio_search_print_limit,
            strict_audio_missing=args.strict_audio_missing,
            multitask=False,
        )
        cot_eval_items = build_items_from_rationale_jsonl(
            cot_eval_file,
            args.audio_dir,
            add_text_only=args.add_text_only,
            text_only=args.text_only,
            max_samples=eval_max_samples,
            allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
            print_audio_search_paths=args.print_audio_search_paths,
            audio_search_print_limit=args.audio_search_print_limit,
            strict_audio_missing=args.strict_audio_missing,
            multitask=False,
        )
        eval_items = build_multisource_multitask_items(
            label_eval_items,
            cot_eval_items,
            cot_task_mode=cot_train_task_mode,
        )
    else:
        eval_items = build_items_from_rationale_jsonl(
            args.eval_file,
            args.audio_dir,
            add_text_only=args.add_text_only,
            text_only=args.text_only,
            max_samples=eval_max_samples,
            allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
            print_audio_search_paths=args.print_audio_search_paths,
            audio_search_print_limit=args.audio_search_print_limit,
            strict_audio_missing=args.strict_audio_missing,
            multitask=True,
            cot_task_mode=cot_train_task_mode,
        )

    if rank == 0:
        logger.info("Train items: %d | Eval items: %d", len(train_items), len(eval_items))
        train_cot = sum(1 for x in train_items if int(x.get("task_id", -1)) == 0)
        train_label = sum(1 for x in train_items if int(x.get("task_id", -1)) == 1)
        eval_cot = sum(1 for x in eval_items if int(x.get("task_id", -1)) == 0)
        eval_label = sum(1 for x in eval_items if int(x.get("task_id", -1)) == 1)
        logger.info("CoT branch mode for train/eval: %s", cot_train_task_mode)
        logger.info(
            "Multitask split | train: cot=%d label=%d | eval: cot=%d label=%d",
            train_cot,
            train_label,
            eval_cot,
            eval_label,
        )

    if len(train_items) == 0:
        raise RuntimeError("No train items loaded. Check train_file/audio_dir paths.")

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Match the original FT behavior: train both audio tower and multimodal projector.
    model.audio_tower.requires_grad_(True)
    model.multi_modal_projector.requires_grad_(True)
    if rank == 0:
        logger.info(
            "Trainability | audio_tower=%s, multi_modal_projector=%s",
            True,
            True,
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=1 if args.smoke else 10,
        eval_strategy="steps" if len(eval_items) > 0 else "no",
        eval_steps=2 if args.smoke else 50,
        save_strategy="no",
        save_total_limit=None,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
        report_to="none",
        disable_tqdm=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=MixedDataset(train_items),
        eval_dataset=MixedDataset(eval_items) if len(eval_items) > 0 else None,
        data_collator=SmartCollator(processor, debug=args.smoke),
        tokenizer=processor.tokenizer,
    )

    trainer.train()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    if world_size > 1:
        dist.barrier()

    test_max_samples = 10 if args.smoke else None
    if rank == 0 and args.smoke:
        logger.info("Loading only %d test items (smoke).", test_max_samples)

    test_items = build_items_from_rationale_jsonl(
        args.test_file,
        args.audio_dir,
        add_text_only=False,
        text_only=args.text_only,
        max_samples=test_max_samples,
        # Follow original script behavior for test: audio-only (no text fallback).
        allow_text_fallback_when_audio_missing=False if not args.text_only else True,
        print_audio_search_paths=args.print_audio_search_paths,
        audio_search_print_limit=args.audio_search_print_limit,
        strict_audio_missing=args.strict_audio_missing,
        multitask=False,
    )
    for item in test_items:
        item["task_mode"] = args.test_task_mode

    output_jsonl = args.output_file.strip() if args.output_file.strip() else os.path.join(args.output_dir, "prediction.jsonl")
    output_parent = os.path.dirname(output_jsonl)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    if rank == 0:
        logger.info("Test inference DataLoader workers: %d", args.inference_num_workers)
        logger.info("Test task mode: %s", args.test_task_mode)
        logger.info("Prediction output file: %s", output_jsonl)
    run_distributed_inference(
        model=model,
        processor=processor,
        items=test_items,
        output_path=output_jsonl,
        device=device,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_workers=args.inference_num_workers,
    )

    if world_size > 1:
        dist.barrier()

    if rank == 0 and args.export_label_eval:
        eval_output_dir = os.path.dirname(output_jsonl) or "."
        label_only_path = os.path.join(eval_output_dir, "prediction_labels_only.jsonl")
        save_label_only_predictions(output_jsonl, label_only_path)

        metrics = evaluate_prediction_file(output_jsonl)
        metrics_path = os.path.join(eval_output_dir, "metrics_label_only.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        logger.info("Label-only evaluation metrics: %s", json.dumps(metrics, ensure_ascii=False))
        logger.info("Saved full predictions: %s", output_jsonl)
        logger.info("Saved label-only predictions: %s", label_only_path)
        logger.info("Saved metrics: %s", metrics_path)
    elif rank == 0:
        logger.info("Saved predictions: %s", output_jsonl)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
