#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from common import (
    build_db_definitions,
    compare_labels,
    compute_reward,
    label_from_record,
    load_metadata,
    normalize_intent_label,
    parse_j_from_output,
    read_jsonl,
    resolve_audio_path,
    split_intent,
    write_jsonl,
)
from prompts import render_infer_audio_prompt, render_infer_text_prompt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

try:
    from train_qwen2_audio_slurp import build_massive_entities, load_speech_massive_split
except Exception:
    build_massive_entities = None
    load_speech_massive_split = None


_DEBUG = False
DEFAULT_OUTPUT_FILE = "Experiment_RationaleCompare/success_cot_raw.jsonl"
DEFAULT_FILTERED_FILE = "Experiment_RationaleCompare/success_cot_filtered.jsonl"


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
        raise RuntimeError("openai package not available. Install it or use --text_local.")
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


def _call_api(client: Any, prompt: str, model_name: str, max_tokens: int, temperature: float, top_p: float) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return resp.choices[0].message.content or ""


def _merge_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("slurp_id", "")),
        str(row.get("mode", "")),
        str(row.get("method", "")),
    )


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
            key = _merge_key(row)
            if key in seen:
                continue
            seen.add(key)
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


def _extract_asr_1best_text(record: Dict[str, Any]) -> str:
    hyps = record.get("asr_hypotheses")
    if not isinstance(hyps, list) or not hyps:
        return ""
    first = hyps[0]
    if isinstance(first, str):
        return first.strip()
    if not isinstance(first, dict):
        return ""
    for key in ("text", "transcript", "hypothesis", "value"):
        value = first.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _select_input_text(record: Dict[str, Any], use_asr_transcript: bool) -> Tuple[str, str]:
    if use_asr_transcript:
        return _extract_asr_1best_text(record), "asr_1best"
    text = str(record.get("sentence", "") or record.get("text", "") or "").strip()
    return text, "gold_transcript"


def _mojibake_score(text: str) -> int:
    markers = ("Ã", "Â", "â€", "â€™", "â€œ", "â€\x9d", "ã\x81", "ã\x82", "ã\x83", "ð\x9f", "�")
    score = sum(text.count(marker) for marker in markers)
    score += sum(1 for ch in text if 0x80 <= ord(ch) <= 0x9F)
    return score


def _locale_prefix(locale_hint: Optional[str]) -> str:
    text = str(locale_hint or "").strip()
    if not text:
        return ""
    return text.split("-", 1)[0].lower()


def _language_mismatch_penalty(text: str, locale_hint: Optional[str]) -> int:
    locale = _locale_prefix(locale_hint)
    if not locale:
        return 0
    japanese_chars = 0
    latin_chars = 0
    for ch in text:
        code = ord(ch)
        if (0x3040 <= code <= 0x30FF) or (0x4E00 <= code <= 0x9FFF):
            japanese_chars += 1
        elif ("a" <= ch.lower() <= "z"):
            latin_chars += 1
    if locale == "ja":
        # Japanese locale: do not penalize Japanese script.
        return 0
    # Non-Japanese locale: heavily penalize unexpected Japanese-script artifacts.
    if japanese_chars == 0:
        return 0
    return japanese_chars * 5 if latin_chars > 0 else japanese_chars * 3


def _mojibake_candidates(text: str) -> List[str]:
    seen = {text}
    candidates = [text]
    frontier = [text]
    for _ in range(2):
        next_frontier: List[str] = []
        for base in frontier:
            for enc in ("latin-1", "cp1252"):
                try:
                    candidate = base.encode(enc).decode("utf-8")
                except Exception:
                    continue
                if candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(candidate)
                next_frontier.append(candidate)
        if not next_frontier:
            break
        frontier = next_frontier
    return candidates


def _candidate_score(text: str, locale_hint: Optional[str]) -> int:
    score = _mojibake_score(text) * 4
    score += text.count("�") * 6
    score += _language_mismatch_penalty(text, locale_hint)
    score += sum(1 for ch in text if ord(ch) < 32 and ch not in ("\n", "\r", "\t")) * 2
    return score


def _maybe_fix_mojibake_text(value: Any, locale_hint: Optional[str] = None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    candidates = _mojibake_candidates(text)
    best_text = min(candidates, key=lambda cand: (_candidate_score(cand, locale_hint), abs(len(cand) - len(text))))
    return best_text


def _normalize_massive_scenario_action(scenario_value: Any, intent_value: Any) -> Tuple[str, str, str]:
    scenario = str(scenario_value or "").strip()
    intent = normalize_intent_label(str(intent_value or "").strip())
    action = intent
    scenario_norm = normalize_intent_label(scenario)
    if scenario_norm and intent.startswith(f"{scenario_norm}_"):
        action = intent[len(scenario_norm) + 1 :]
    elif (not scenario_norm) and intent:
        scenario2, action2 = split_intent(intent)
        if scenario2:
            scenario = scenario2
            action = action2
    return scenario, action, intent


def _unique_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _extract_massive_audio_path(record: Dict[str, Any]) -> str:
    direct = record.get("path")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    audio = record.get("audio")
    if isinstance(audio, dict):
        path = audio.get("path")
        if isinstance(path, str) and path.strip():
            return path.strip()
    return ""


def _massive_entities_to_common(
    tokens: Sequence[str],
    labels: Sequence[str],
    outside_label: str,
) -> List[Dict[str, str]]:
    if build_massive_entities is None:
        raise RuntimeError(
            "Speech-MASSIVE support requires train_qwen2_audio_slurp.py "
            "(build_massive_entities/load_speech_massive_split)."
        )
    raw_entities = build_massive_entities(tokens, labels, outside_label)
    entities: List[Dict[str, str]] = []
    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue
        slot_type = ""
        slot_value = ""
        if "type" in ent:
            slot_type = str(ent.get("type", "")).strip()
            filler = ent.get("filler")
            if filler is None:
                filler = ent.get("filter")
            if filler is None:
                filler = ent.get("value")
            slot_value = "" if filler is None else str(filler).strip()
        else:
            for key, value in ent.items():
                slot_type = str(key).strip()
                slot_value = "" if value is None else str(value).strip()
                break
        if not slot_type:
            continue
        entities.append({"type": slot_type, "filler": slot_value})
    return entities


def _normalize_speech_massive_record(
    record: Dict[str, Any],
    dataset_config: str,
    split: str,
    index: int,
    transcript_field: str,
    outside_label: str,
) -> Dict[str, Any]:
    raw_transcript = (
        record.get(transcript_field)
        or record.get("utt")
        or record.get("text")
        or ""
    )
    transcript = _maybe_fix_mojibake_text(raw_transcript, locale_hint=dataset_config)
    raw_scenario = record.get("scenario_str") or record.get("scenario") or ""
    raw_intent = record.get("intent_str") or record.get("intent") or ""
    scenario, action, intent = _normalize_massive_scenario_action(raw_scenario, raw_intent)
    token_values = record.get("tokens") if isinstance(record.get("tokens"), list) else []
    label_values = record.get("labels") if isinstance(record.get("labels"), list) else []
    tokens = [_maybe_fix_mojibake_text(tok, locale_hint=dataset_config) for tok in token_values]
    labels = [str(lbl) for lbl in label_values]
    entities = _massive_entities_to_common(tokens, labels, outside_label) if (tokens and labels) else []
    audio_path = _extract_massive_audio_path(record)
    recordings = [{"file": audio_path}] if audio_path else []

    base_id = (
        record.get("id")
        or record.get("utt_id")
        or record.get("audio_id")
        or os.path.basename(audio_path)
        or str(index)
    )
    slurp_id = f"massive-{dataset_config}-{base_id}"
    if _DEBUG:
        raw_transcript_text = str(raw_transcript or "").strip()
        if raw_transcript_text and raw_transcript_text != transcript:
            _log_debug(f"[DEBUG] fixed mojibake transcript for {slurp_id}")
        raw_intent_text = normalize_intent_label(str(raw_intent or "").strip())
        if raw_intent_text and raw_intent_text != action:
            _log_debug(
                f"[DEBUG] normalized Speech-MASSIVE intent/action for {slurp_id}: "
                f"scenario={scenario!r}, intent={raw_intent_text!r}, action={action!r}"
            )

    return {
        "slurp_id": slurp_id,
        "sentence": transcript,
        "text": transcript,
        "recordings": recordings,
        "scenario": scenario,
        "action": action,
        "intent": intent,
        "entities": entities,
        "tokens": [{"surface": tok} for tok in tokens],
        "massive_tokens": tokens,
        "massive_labels": labels,
        "dataset": "speech_massive",
        "dataset_config": dataset_config,
        "dataset_split": split,
    }


def _load_speech_massive_items(
    dataset_name: str,
    dataset_configs: Sequence[str],
    split: str,
    cache_dir: Optional[str],
    transcript_field: str,
    outside_label: str,
) -> List[Dict[str, Any]]:
    if load_speech_massive_split is None:
        raise RuntimeError(
            "Speech-MASSIVE support requires train_qwen2_audio_slurp.py "
            "(load_speech_massive_split)."
        )

    items: List[Dict[str, Any]] = []
    for dataset_config in dataset_configs:
        ds = load_speech_massive_split(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            cache_dir=cache_dir,
        )
        for idx, row in enumerate(ds):
            if not isinstance(row, dict):
                continue
            items.append(
                _normalize_speech_massive_record(
                    record=row,
                    dataset_config=dataset_config,
                    split=split,
                    index=idx,
                    transcript_field=transcript_field,
                    outside_label=outside_label,
                )
            )
    return items


def _build_metadata_from_items(items: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    scenarios: List[str] = []
    actions: List[str] = []
    intents: List[str] = []
    slot_types: List[str] = []
    for item in items:
        gold = label_from_record(item)
        scenario = str(gold.get("scenario", "")).strip()
        action = str(gold.get("action", "")).strip()
        if scenario:
            scenarios.append(scenario)
        if action:
            actions.append(action)
        if scenario or action:
            intents.append(normalize_intent_label(f"{scenario}_{action}".strip("_")))
        entities = gold.get("entities", []) if isinstance(gold.get("entities"), list) else []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            slot_type = str(ent.get("type", "")).strip()
            if slot_type:
                slot_types.append(slot_type)
    return {
        "scenarios": _unique_keep_order(scenarios),
        "actions": _unique_keep_order(actions),
        "intents": _unique_keep_order(intents),
        "slot_types": _unique_keep_order(slot_types),
    }


def _generate_audio_local(
    processor: AutoProcessor,
    model: Qwen2AudioForConditionalGeneration,
    audio_path: str,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    sr = processor.feature_extractor.sampling_rate
    audio, _ = librosa.load(audio_path, sr=sr)
    user_content = [
        {"type": "audio", "audio_url": "placeholder"},
        {"type": "text", "text": prompt},
    ]
    text_input = processor.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)


def _success_match_ok(match_mode: str, stats: Dict[str, Any]) -> bool:
    if match_mode == "intent":
        return bool(stats["intent_ok"])
    if match_mode == "scenario_action":
        return bool(stats["scenario_ok"] and stats["action_ok"])
    if match_mode == "full":
        return bool(stats["scenario_ok"] and stats["action_ok"])
    return False


def _log_debug(message: str) -> None:
    if not _DEBUG:
        return
    print(message, flush=True)


def _normalize_pred_label(pred_obj: Any) -> Dict[str, Any]:
    if not isinstance(pred_obj, dict):
        pred_obj = {}
    raw_intent = pred_obj.get("intent")
    if raw_intent is None:
        raw_intent = pred_obj.get("Intent")
    intent = normalize_intent_label(str(raw_intent or "").strip())
    scenario = str(pred_obj.get("scenario", "")).strip()
    action = str(pred_obj.get("action", "")).strip()
    if (not scenario or not action) and intent:
        scenario2, action2 = split_intent(intent)
        scenario = scenario or scenario2
        action = action or action2
    if not intent and (scenario or action):
        intent = normalize_intent_label(f"{scenario}_{action}".strip("_"))

    entities: List[Dict[str, str]] = []
    raw_entities = pred_obj.get("entities", [])
    if isinstance(raw_entities, list):
        for ent in raw_entities:
            if not isinstance(ent, dict):
                continue
            ent_type = str(ent.get("type", "")).strip()
            filler = ent.get("filler")
            if filler is None:
                filler = ent.get("filter")
            if filler is None:
                filler = ent.get("value")
            if not ent_type:
                # Accept compact entity form: {"slot_type": "slot_value"}.
                compact_items = [
                    (str(k).strip(), v)
                    for k, v in ent.items()
                    if str(k).strip() and str(k).strip().lower() not in {"type", "filler", "filter", "value"}
                ]
                if compact_items:
                    ent_type, filler = compact_items[0]
            filler = "" if filler is None else str(filler).strip()
            entities.append({"type": ent_type, "filler": filler})
    return {
        "intent": intent,
        "scenario": scenario,
        "action": action,
        "entities": entities,
    }


def _extract_prefixed_line(text: str, prefix: str) -> str:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    target = prefix.upper()
    for ln in lines:
        if re.match(rf"^{re.escape(target)}\s*", ln.upper()):
            body = ln.split(":", 1)[1].strip() if ":" in ln else ""
            if body:
                return f"{prefix} {body}"
            break
    return f"{prefix} (none)"


def _extract_intent_candidates(text: str) -> List[str]:
    c_line = _extract_prefixed_line(text, "C:")
    if c_line == "C: (none)":
        return []
    c_body = c_line.split(":", 1)[1].strip() if ":" in c_line else ""
    if not c_body:
        return []
    intent_part = c_body.split(";", 1)[0].strip()
    lowered = intent_part.lower()
    for prefix in ("intent candidates:", "intent candidate:", "intent:", "intents:"):
        if lowered.startswith(prefix):
            intent_part = intent_part[len(prefix):].strip()
            break
    values = [normalize_intent_label(x.strip()) for x in intent_part.split("|") if x.strip()]
    return [v for v in values if v and v != "(none)"]


def _extract_slot_candidates(text: str) -> List[str]:
    c_line = _extract_prefixed_line(text, "C:")
    if c_line == "C: (none)":
        return []
    c_body = c_line.split(":", 1)[1].strip() if ":" in c_line else ""
    if not c_body:
        return []
    slot_part = c_body
    if ";" in c_body:
        maybe_slot = c_body.split(";", 1)[1].strip()
        if maybe_slot:
            slot_part = maybe_slot
    lowered = slot_part.lower()
    for prefix in ("slot candidates:", "slot candidate:", "slots:", "slot:"):
        if lowered.startswith(prefix):
            slot_part = slot_part[len(prefix):].strip()
            break
    if not slot_part:
        return []
    # Remove value-candidate parentheses before splitting by "|"
    # e.g., "artist(adele|sia) | album" -> "artist | album"
    slot_part = re.sub(r"\([^)]*\)", "", slot_part)
    values = [x.strip().lower() for x in slot_part.split("|") if x.strip()]
    cleaned = [v for v in values if v and v not in {"(none)", "none"}]
    return cleaned


def _gold_intent_from_label(gold_label: Dict[str, Any]) -> str:
    return normalize_intent_label(f"{gold_label.get('scenario', '')}_{gold_label.get('action', '')}")


def _gold_slot_types_from_label(gold_label: Dict[str, Any]) -> List[str]:
    entities = gold_label.get("entities", []) if isinstance(gold_label, dict) else []
    slots: List[str] = []
    if isinstance(entities, list):
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            slot = str(ent.get("type", "")).strip().lower()
            if slot:
                slots.append(slot)
    if not slots:
        return ["__none__"]
    # unique keep order
    seen = set()
    uniq: List[str] = []
    for s in slots:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _slot_candidate_coverage(gold_slot_types: List[str], slot_candidates: List[str]) -> float:
    gold_set = {str(x).strip().lower() for x in gold_slot_types if str(x).strip() and str(x).strip().lower() != "__none__"}
    cand_set = {str(x).strip().lower() for x in slot_candidates if str(x).strip() and str(x).strip().lower() != "__none__"}
    if not gold_set:
        return 1.0 if not cand_set else 0.0
    return float(len(gold_set & cand_set) / max(1, len(gold_set)))


def _success_score(
    format_ok: bool,
    match_ok: bool,
    has_gold_intent_candidate: bool,
    slot_candidate_coverage: float,
) -> float:
    # Simple formula:
    # success_score = format_ok + match_ok + has_gold_intent_candidate + slot_candidate_coverage (range: 0..4)
    return float(
        (1 if format_ok else 0)
        + (1 if match_ok else 0)
        + (1 if has_gold_intent_candidate else 0)
        + float(slot_candidate_coverage)
    )


def _raw_sort_key(row: Dict[str, Any]) -> Tuple[int, float, float]:
    return (
        1 if row.get("correct") else 0,
        float(row.get("success_score", 0.0)),
        float(row.get("reward", 0.0)),
    )


def _difficulty_sort_key(row: Dict[str, Any]) -> Tuple[float, float, int]:
    # Hard-first ordering for analysis:
    # lower success score -> lower reward -> incorrect first
    return (
        float(row.get("success_score", 0.0)),
        float(row.get("reward", 0.0)),
        0 if not bool(row.get("correct")) else 1,
    )


def _make_filtered_row(raw_row: Dict[str, Any], coverage_fallback: bool = False) -> Dict[str, Any]:
    filtered_row = {
        "slurp_id": raw_row.get("slurp_id"),
        "sentence": raw_row.get("sentence", ""),
        "final": raw_row.get("gold_label", {}),
        "rationale_text": str(raw_row.get("rationale_text", "")).strip(),
        "mode": raw_row.get("mode", ""),
        "method": "sf-cot",
        "reward": float(raw_row.get("reward", 0.0)),
        "success_score": float(raw_row.get("success_score", 0.0)),
        "format_ok": bool(raw_row.get("format_ok", False)),
        "match_ok": bool(raw_row.get("match_ok", False)),
        "has_gold_intent_candidate": bool(raw_row.get("has_gold_intent_candidate", False)),
        "has_gold_slot_candidates": bool(raw_row.get("has_gold_slot_candidates", False)),
        "slot_candidate_coverage": float(raw_row.get("slot_candidate_coverage", 0.0)),
        "correct": bool(raw_row.get("correct", False)),
        "coverage_fallback": bool(coverage_fallback),
    }
    recordings = raw_row.get("recordings")
    if raw_row.get("mode") == "audio" and isinstance(recordings, list):
        filtered_row["recordings"] = recordings
    for key in ("dataset", "dataset_config", "dataset_split"):
        value = raw_row.get(key)
        if value is not None and str(value).strip():
            filtered_row[key] = value
    return filtered_row


def _format_model_output(raw_output: str, no_cot: bool = False) -> Tuple[str, Dict[str, Any], bool]:
    pred_obj = parse_j_from_output(raw_output)
    pred_label = _normalize_pred_label(pred_obj)
    j_obj = {
        "Intent": pred_label.get("intent", ""),
        "entities": pred_label.get("entities", []),
    }
    j_line = "J: " + json.dumps(j_obj, ensure_ascii=False)
    has_j = pred_obj is not None
    if no_cot:
        return j_line, pred_label, has_j

    c_line = _extract_prefixed_line(raw_output, "C:")
    r_line = _extract_prefixed_line(raw_output, "R:")
    has_c = c_line != "C: (none)"
    has_r = r_line != "R: (none)"
    formatted = "\n".join([c_line, r_line, j_line])
    return formatted, pred_label, (has_c and has_r and has_j)


def _enrich_raw_row_for_success(raw_row: Dict[str, Any], success_match: str) -> Dict[str, Any]:
    row = dict(raw_row)
    source_output = str(row.get("raw_output", "") or row.get("rationale_text", "") or "")
    formatted_output, pred_label_from_output, format_ok = _format_model_output(source_output, no_cot=False)

    gold_label = row.get("gold_label")
    if not isinstance(gold_label, dict):
        gold_label = row.get("final") if isinstance(row.get("final"), dict) else {}

    pred_label = row.get("pred_label")
    if not isinstance(pred_label, dict):
        pred_label = {}
    # Prefer parsed J from output when available.
    if pred_label_from_output:
        pred_label = pred_label_from_output

    stats = compare_labels(pred_label, gold_label)
    gold_intent = _gold_intent_from_label(gold_label)
    gold_slot_types = _gold_slot_types_from_label(gold_label)
    intent_candidates = _extract_intent_candidates(formatted_output)
    slot_candidates = _extract_slot_candidates(formatted_output)
    has_gold_intent_candidate = bool(gold_intent) and (gold_intent in intent_candidates)
    slot_cov = _slot_candidate_coverage(gold_slot_types, slot_candidates)
    has_gold_slot_candidates = bool(slot_cov >= 1.0 - 1e-8)
    match_ok = _success_match_ok(success_match, stats)
    score = _success_score(format_ok, match_ok, has_gold_intent_candidate, slot_cov)
    is_ok = bool(score >= 4.0)
    reward, _ = compute_reward(pred_label, gold_label)

    row["rationale_text"] = formatted_output.strip()
    row["raw_output"] = source_output.strip()
    row["pred_label"] = pred_label
    row["gold_label"] = gold_label
    row["correct"] = bool(is_ok)
    row["match_ok"] = bool(match_ok)
    row["success_score"] = float(score)
    row["success_formula"] = "success_score = format_ok + match_ok + has_gold_intent_candidate + slot_candidate_coverage"
    row["has_gold_intent_candidate"] = bool(has_gold_intent_candidate)
    row["has_gold_slot_candidates"] = bool(has_gold_slot_candidates)
    row["slot_candidate_coverage"] = float(slot_cov)
    row["gold_intent"] = gold_intent
    row["gold_slot_types"] = gold_slot_types
    row["intent_candidates"] = intent_candidates
    row["slot_candidates"] = slot_candidates
    row["format_ok"] = bool(format_ok)
    row["reward"] = float(reward)
    row["method"] = str(row.get("method", "sf-cot") or "sf-cot")
    return row


def _build_filtered_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Build success list from good rows first.
    filtered_rows = [_make_filtered_row(row, coverage_fallback=False) for row in raw_rows if row.get("correct")]

    # Coverage completion: ensure at least one example for each gold intent and each gold slot type.
    required_intents = sorted(
        {str(row.get("gold_intent", "")).strip() for row in raw_rows if str(row.get("gold_intent", "")).strip()}
    )
    required_slots = sorted({slot for row in raw_rows for slot in (row.get("gold_slot_types", []) or [])})

    selected_keys = {(str(r.get("slurp_id")), str(r.get("mode"))) for r in filtered_rows}

    def current_intents() -> set:
        values = set()
        for row in filtered_rows:
            gold = row.get("final", {})
            intent = _gold_intent_from_label(gold if isinstance(gold, dict) else {})
            if intent:
                values.add(intent)
        return values

    def current_slots() -> set:
        values = set()
        for row in filtered_rows:
            gold = row.get("final", {})
            values.update(_gold_slot_types_from_label(gold if isinstance(gold, dict) else {}))
        return values

    def best_row_for(intent: Optional[str] = None, slot_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for row in raw_rows:
            key = (str(row.get("slurp_id")), str(row.get("mode")))
            if key in selected_keys:
                continue
            if intent is not None and row.get("gold_intent") != intent:
                continue
            if slot_type is not None and slot_type not in (row.get("gold_slot_types", []) or []):
                continue
            candidates.append(row)
        if not candidates:
            return None
        return sorted(candidates, key=_raw_sort_key, reverse=True)[0]

    # Fill missing intents first.
    for intent in required_intents:
        if intent in current_intents():
            continue
        row = best_row_for(intent=intent)
        if row is None:
            continue
        selected_keys.add((str(row.get("slurp_id")), str(row.get("mode"))))
        filtered_rows.append(_make_filtered_row(row, coverage_fallback=not bool(row.get("correct"))))

    # Then fill missing slot types.
    for slot in required_slots:
        if slot in current_slots():
            continue
        row = best_row_for(slot_type=slot)
        if row is None:
            continue
        selected_keys.add((str(row.get("slurp_id")), str(row.get("mode"))))
        filtered_rows.append(_make_filtered_row(row, coverage_fallback=not bool(row.get("correct"))))

    # Keep "good ones first" ordering.
    filtered_rows.sort(
        key=lambda row: (
            1 if row.get("correct") else 0,
            float(row.get("success_score", 0.0)),
            float(row.get("reward", 0.0)),
        ),
        reverse=True,
    )
    return filtered_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Success-Filtered CoT (text and audio).")
    parser.add_argument("--dataset", type=str, choices=["slurp", "speech_massive"], default="slurp")
    parser.add_argument("--input_file", type=str, default="slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--metadata_file", type=str, default="Experiment_3/slurp_metadata.json")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio/slurp_real")
    parser.add_argument("--massive_dataset_name", type=str, default="FBK-MT/Speech-MASSIVE")
    parser.add_argument("--massive_dataset_config", type=str, default="it-IT")
    parser.add_argument("--massive_split", type=str, default="train_115")
    parser.add_argument("--massive_cache_dir", type=str, default=None)
    parser.add_argument("--massive_transcript_field", type=str, default="utt")
    parser.add_argument("--massive_outside_label", type=str, default="Other")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--filtered_file", type=str, default=DEFAULT_FILTERED_FILE)
    parser.add_argument(
        "--difficulty_file",
        type=str,
        default="",
        help="(rescore mode) Save an additional hard-first JSONL sorted by low success_score.",
    )
    parser.add_argument(
        "--rescore_raw_file",
        type=str,
        default=None,
        help="Recompute success metrics from an existing raw jsonl and output to new files.",
    )
    parser.add_argument("--modes", type=str, default="text")
    parser.add_argument("--text_model_name", "--model", dest="text_model_name", type=str, default="deepseekr1")
    parser.add_argument("--audio_model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recording_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", dest="do_sample", action="store_true", help="Enable sampling.")
    parser.add_argument(
        "--no_do_sample",
        "--no-do-sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_rank", type=int, default=0)
    parser.add_argument("--append_worker_suffix", action="store_true")
    parser.add_argument("--merge_workers", action="store_true", help="Merge worker shard outputs into output_file(s).")
    parser.add_argument("--merge_only", action="store_true", help="Only merge worker shard outputs and exit.")
    parser.add_argument("--merge_cleanup", action="store_true", help="Remove worker shard files after merge.")
    parser.add_argument("--no_spawn_workers", action="store_true", help="Do not auto-spawn worker processes.")
    parser.add_argument("--success_match", type=str, choices=["full", "scenario_action", "intent"], default="full")
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument(
        "--asr_transcript",
        "--asr-transcript",
        dest="asr_transcript",
        action="store_true",
        help='Use asr_hypotheses[0]["text"] (1-best) for text-mode input instead of sentence/text.',
    )
    parser.add_argument("--debug", action="store_true", help="Print extra debug info.")
    parser.add_argument("--smoke", action="store_true", help="Process only 300 samples for debugging.")
    parser.set_defaults(do_sample=True)
    args = parser.parse_args()

    args.text_model_name = _canonicalize_model_name(args.text_model_name)
    if str(args.text_model_name).strip().lower() == "deepseekr1":
        args.text_model_name = "deepseek-r1"

    global _DEBUG
    _DEBUG = args.debug

    if args.num_workers > 1:
        # Always shard outputs and auto-merge when using multiple workers.
        args.append_worker_suffix = True
        args.merge_workers = True

    rng = random.Random(args.seed + args.worker_rank)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input_file) if not os.path.isabs(args.input_file) else args.input_file
    metadata_path = os.path.join(base_dir, args.metadata_file) if not os.path.isabs(args.metadata_file) else args.metadata_file
    audio_dir = os.path.join(base_dir, args.audio_dir) if not os.path.isabs(args.audio_dir) else args.audio_dir
    rescore_raw_path = (
        os.path.join(base_dir, args.rescore_raw_file)
        if (args.rescore_raw_file and not os.path.isabs(args.rescore_raw_file))
        else args.rescore_raw_file
    )

    if args.rescore_raw_file and args.output_file == DEFAULT_OUTPUT_FILE:
        root, ext = os.path.splitext(str(rescore_raw_path or "rescore_raw.jsonl"))
        base_output_path = f"{root}.rescored{ext or '.jsonl'}"
    else:
        base_output_path = os.path.join(base_dir, args.output_file) if not os.path.isabs(args.output_file) else args.output_file

    if args.rescore_raw_file and args.filtered_file == DEFAULT_FILTERED_FILE:
        root, ext = os.path.splitext(str(rescore_raw_path or "rescore_raw.jsonl"))
        base_filtered_path = f"{root}.rescored.filtered{ext or '.jsonl'}"
    else:
        base_filtered_path = os.path.join(base_dir, args.filtered_file) if not os.path.isabs(args.filtered_file) else args.filtered_file

    if args.difficulty_file.strip():
        base_difficulty_path = (
            os.path.join(base_dir, args.difficulty_file)
            if not os.path.isabs(args.difficulty_file)
            else args.difficulty_file
        )
    elif args.rescore_raw_file:
        root, ext = os.path.splitext(base_output_path)
        base_difficulty_path = f"{root}.hard_first{ext or '.jsonl'}"
    else:
        base_difficulty_path = ""

    output_path = base_output_path
    filtered_path = base_filtered_path
    difficulty_path = base_difficulty_path
    if args.append_worker_suffix and args.num_workers > 1:
        root, ext = os.path.splitext(base_output_path)
        output_path = f"{root}.w{args.worker_rank}of{args.num_workers}{ext or '.jsonl'}"
        root, ext = os.path.splitext(base_filtered_path)
        filtered_path = f"{root}.w{args.worker_rank}of{args.num_workers}{ext or '.jsonl'}"
        if base_difficulty_path:
            root, ext = os.path.splitext(base_difficulty_path)
            difficulty_path = f"{root}.w{args.worker_rank}of{args.num_workers}{ext or '.jsonl'}"

    if args.merge_only:
        _merge_worker_outputs(base_output_path, args.num_workers, cleanup=args.merge_cleanup)
        _merge_worker_outputs(base_filtered_path, args.num_workers, cleanup=args.merge_cleanup)
        if base_difficulty_path:
            _merge_worker_outputs(base_difficulty_path, args.num_workers, cleanup=args.merge_cleanup)
        return

    if args.rescore_raw_file:
        rows = read_jsonl(str(rescore_raw_path))
        if args.smoke:
            args.limit = 100
        if args.limit:
            rows = rows[: args.limit]
        if args.num_workers > 1:
            rows = rows[args.worker_rank :: args.num_workers]

        row_iter = rows
        if tqdm is not None:
            desc = "02_rescore_success_cot"
            if args.num_workers > 1:
                desc = f"{desc} [w{args.worker_rank}/{args.num_workers}]"
            row_iter = tqdm(
                rows,
                total=len(rows),
                desc=desc,
                disable=(args.num_workers > 1 and args.worker_rank != 0),
            )

        raw_rows = [_enrich_raw_row_for_success(row, args.success_match) for row in row_iter]
        raw_rows.sort(key=_raw_sort_key, reverse=True)
        filtered_rows = _build_filtered_rows(raw_rows)
        difficulty_rows = sorted(raw_rows, key=_difficulty_sort_key)
        write_jsonl(output_path, raw_rows)
        write_jsonl(filtered_path, filtered_rows)
        if difficulty_path:
            write_jsonl(difficulty_path, difficulty_rows)

        if args.merge_workers and args.num_workers > 1 and args.append_worker_suffix and args.worker_rank == 0:
            _merge_worker_outputs(base_output_path, args.num_workers, cleanup=args.merge_cleanup)
            _merge_worker_outputs(base_filtered_path, args.num_workers, cleanup=args.merge_cleanup)
            if base_difficulty_path:
                _merge_worker_outputs(base_difficulty_path, args.num_workers, cleanup=args.merge_cleanup)
        return

    worker_procs: List[subprocess.Popen] = []
    if (
        args.num_workers > 1
        and args.worker_rank == 0
        and (not args.no_spawn_workers)
        and (not args.merge_only)
    ):
        worker_procs = _spawn_workers(args.num_workers, sys.argv[1:])

    if args.dataset == "speech_massive":
        dataset_configs = [cfg.strip() for cfg in args.massive_dataset_config.split(",") if cfg.strip()]
        if not dataset_configs:
            raise ValueError("massive_dataset_config must contain at least one config (e.g., it-IT).")
        items = _load_speech_massive_items(
            dataset_name=args.massive_dataset_name,
            dataset_configs=dataset_configs,
            split=args.massive_split,
            cache_dir=args.massive_cache_dir,
            transcript_field=args.massive_transcript_field,
            outside_label=args.massive_outside_label,
        )
        metadata = _build_metadata_from_items(items)
    else:
        items = read_jsonl(input_path)
        metadata = load_metadata(metadata_path)
    db_definitions = build_db_definitions(metadata)

    if args.smoke:
        args.limit = 100
    if args.limit:
        items = items[: args.limit]
    if args.num_workers > 1:
        items = items[args.worker_rank :: args.num_workers]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    client = None
    if "text" in modes:
        client = _build_client(args.text_model_name)

    processor = None
    model = None
    device = torch.device(args.device)
    if "audio" in modes:
        processor = AutoProcessor.from_pretrained(args.audio_model_name_or_path, trust_remote_code=True)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.audio_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        model.eval()

    raw_rows: List[Dict[str, Any]] = []
    filtered_rows: List[Dict[str, Any]] = []

    record_iter = items
    if tqdm is not None:
        desc = "02_generate_success_cot"
        if args.num_workers > 1:
            desc = f"{desc} [w{args.worker_rank}/{args.num_workers}]"
        record_iter = tqdm(
            items,
            total=len(items),
            desc=desc,
            disable=(args.num_workers > 1 and args.worker_rank != 0),
        )

    for record in record_iter:
        gold_text, text_source = _select_input_text(record, args.asr_transcript)
        gold_label = label_from_record(record)
        recordings = record.get("recordings", []) if isinstance(record.get("recordings"), list) else []

        for mode in modes:
            output = ""
            formatted_output = ""
            if mode == "text":
                if not gold_text:
                    continue
                prompt = render_infer_text_prompt(db_definitions, gold_text)
                for attempt in range(args.retry + 1):
                    try:
                        output = _call_api(
                            client=client,
                            prompt=prompt,
                            model_name=args.text_model_name,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                        time.sleep(0.2)  # per-worker rate limit (~5 req/sec)
                        break
                    except Exception:
                        # Fail fast on any API error to avoid burning the key.
                        raise
            elif mode == "audio":
                if not recordings:
                    continue
                rec = recordings[args.recording_index] if args.recording_index < len(recordings) else recordings[0]
                filename = rec.get("file") if isinstance(rec, dict) else None
                if not filename:
                    continue
                audio_path = resolve_audio_path(audio_dir, filename)
                if not audio_path:
                    continue
                prompt = render_infer_audio_prompt(db_definitions)
                output = _generate_audio_local(
                    processor=processor,
                    model=model,
                    audio_path=audio_path,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                )
            else:
                continue

            formatted_output, pred_label, format_ok = _format_model_output(output, no_cot=False)
            stats = compare_labels(pred_label, gold_label)
            gold_intent = _gold_intent_from_label(gold_label)
            gold_slot_types = _gold_slot_types_from_label(gold_label)
            intent_candidates = _extract_intent_candidates(formatted_output)
            slot_candidates = _extract_slot_candidates(formatted_output)
            has_gold_intent_candidate = bool(gold_intent) and (gold_intent in intent_candidates)
            slot_cov = _slot_candidate_coverage(gold_slot_types, slot_candidates)
            has_gold_slot_candidates = bool(slot_cov >= 1.0 - 1e-8)
            match_ok = _success_match_ok(args.success_match, stats)
            score = _success_score(format_ok, match_ok, has_gold_intent_candidate, slot_cov)
            is_ok = bool(score >= 4.0)
            reward, _ = compute_reward(pred_label, gold_label)

            if args.debug:
                word_count = len((output or "").split())
                _log_debug(
                    f"[DEBUG] slurp_id={record.get('slurp_id')} mode={mode} words={word_count} "
                    f"format_ok={bool(format_ok)} correct={bool(is_ok)} "
                    f"match_ok={bool(match_ok)} has_gold_intent_candidate={bool(has_gold_intent_candidate)} "
                    f"has_gold_slot_candidates={bool(has_gold_slot_candidates)} "
                    f"slot_candidate_coverage={slot_cov:.3f} success_score={score:.3f} reward={reward:.3f}"
                )
                _log_debug("[DEBUG] raw_output:")
                _log_debug(output or "")
                _log_debug("[DEBUG] formatted_output:")
                _log_debug(formatted_output or "")

            raw_row = {
                "slurp_id": record.get("slurp_id"),
                "sentence": gold_text,
                "input_text_source": text_source,
                "recordings": recordings,
                "dataset": record.get("dataset", args.dataset),
                "dataset_config": record.get("dataset_config", ""),
                "dataset_split": record.get("dataset_split", ""),
                "mode": mode,
                "method": "sf-cot",
                "rationale_text": formatted_output.strip(),
                "raw_output": output.strip(),
                "format_ok": bool(format_ok),
                "pred_label": pred_label,
                "gold_label": gold_label,
                "correct": bool(is_ok),
                "match_ok": bool(match_ok),
                "success_score": score,
                "success_formula": "success_score = format_ok + match_ok + has_gold_intent_candidate + slot_candidate_coverage",
                "has_gold_intent_candidate": bool(has_gold_intent_candidate),
                "has_gold_slot_candidates": bool(has_gold_slot_candidates),
                "slot_candidate_coverage": float(slot_cov),
                "gold_intent": gold_intent,
                "gold_slot_types": gold_slot_types,
                "intent_candidates": intent_candidates,
                "slot_candidates": slot_candidates,
                "reward": reward,
            }
            raw_rows.append(raw_row)

    raw_rows.sort(key=_raw_sort_key, reverse=True)
    filtered_rows = _build_filtered_rows(raw_rows)

    write_jsonl(output_path, raw_rows)
    write_jsonl(filtered_path, filtered_rows)

    for proc in worker_procs:
        proc.wait()

    if args.merge_workers and args.num_workers > 1 and args.append_worker_suffix and args.worker_rank == 0:
        _merge_worker_outputs(base_output_path, args.num_workers, cleanup=args.merge_cleanup)
        _merge_worker_outputs(base_filtered_path, args.num_workers, cleanup=args.merge_cleanup)


if __name__ == "__main__":
    main()
