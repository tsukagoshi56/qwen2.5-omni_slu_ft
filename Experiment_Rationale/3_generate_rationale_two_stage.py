#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import librosa
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# ----------------------------
# Small helpers
# ----------------------------

STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","to","for","of","on","in","at","by","from",
    "is","are","was","were","be","been","being","do","does","did","have","has","had",
    "i","you","he","she","it","we","they","me","my","your","his","her","its","our","their",
    "this","that","these","those","there","here","now","please","can","could","would","should",
    "what","which","who","whom","when","where","why","how","with","as","about","up","down",
}

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path: str, items: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def write_raw_jsonl(path: str, outputs: List[Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

def resolve_audio_path(audio_root: str, filename: str) -> Optional[str]:
    if not filename:
        return None
    candidates = [
        os.path.join(audio_root, filename),
        os.path.join(audio_root, "slurp_real", filename),
        os.path.join("slurp", "audio", "slurp_real", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def load_audio(path: str, target_sr: int = 16000) -> Optional[List[float]]:
    try:
        audio, _ = librosa.load(path, sr=target_sr, mono=True)
        return audio
    except Exception as exc:
        print(f"[WARN] Failed to load audio: {path} ({exc})")
        return None

def build_entities(record: Dict[str, Any]) -> List[Dict[str, str]]:
    tokens = record.get("tokens", []) or []
    raw_entities = record.get("entities", []) or []
    results: List[Dict[str, str]] = []
    for ent in raw_entities:
        ent_type = ent.get("type", "")
        filler = ent.get("filler")
        if not filler:
            span = ent.get("span") or []
            if isinstance(span, list) and span:
                words = []
                for idx in span:
                    if isinstance(idx, int) and idx < len(tokens):
                        words.append(tokens[idx].get("surface", ""))
                filler = " ".join([w for w in words if w])
        if filler is None:
            filler = ""
        results.append({"type": ent_type, "filler": filler})
    return results

def extract_slot_types(entities: List[Dict[str, str]]) -> List[str]:
    seen = set()
    slot_types: List[str] = []
    for ent in entities:
        t = ent.get("type", "")
        if not t or t in seen:
            continue
        seen.add(t)
        slot_types.append(t)
    return slot_types

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    toks = re.findall(r"[a-z0-9']+", text.lower())
    return [t for t in toks if t and t not in STOPWORDS]

def summarize_nbest(hyp_texts: List[str], max_tokens: int = 6) -> Dict[str, List[str]]:
    if not hyp_texts:
        return {"stable": [], "unstable": []}
    hyp_tokens = [set(tokenize(t)) for t in hyp_texts if t]
    if not hyp_tokens:
        return {"stable": [], "unstable": []}
    counts = Counter()
    for tok_set in hyp_tokens:
        counts.update(tok_set)
    n = len(hyp_tokens)
    stable_thresh = max(1, math.ceil(n * 0.6))
    unstable_thresh = max(1, math.floor(n * 0.4))
    stable = [t for t, c in counts.items() if c >= stable_thresh]
    unstable = [t for t, c in counts.items() if c <= unstable_thresh]
    stable.sort(key=lambda t: (-counts[t], t))
    unstable.sort(key=lambda t: (counts[t], t))
    return {
        "stable": stable[:max_tokens],
        "unstable": unstable[:max_tokens],
    }

def format_interpretations(nbest_texts: List[str], max_items: int = 5) -> str:
    if not nbest_texts:
        return "interpretation_1: (none)"
    lines = []
    for i, text in enumerate(nbest_texts[:max_items]):
        lines.append(f"interpretation_{i+1}: {text}")
    return "\n".join(lines)

def load_metadata(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        return {"scenarios": [], "actions": [], "intents": [], "slot_types": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "scenarios": data.get("scenarios", []),
        "actions": data.get("actions", []),
        "intents": data.get("intents", []),
        "slot_types": data.get("slot_types", []),
    }

def load_clusters(path: str) -> Dict[str, Dict[str, List[str]]]:
    if not os.path.exists(path):
        return {"scenarios": {}, "actions": {}, "slot_types": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    def build_map(section: Dict[str, List[str]]) -> Dict[str, List[str]]:
        label_map = {}
        for _, members in section.items():
            for m in members:
                label_map[m] = members
        return label_map
    return {
        "scenarios": build_map(data.get("scenarios", {})),
        "actions": build_map(data.get("actions", {})),
        "slot_types": build_map(data.get("slot types", {})),
    }

def load_confusing_pairs(path: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("actions", []):
        pair = item.get("pair") or []
        sim = item.get("similarity", 0.0)
        if len(pair) != 2:
            continue
        a, b = pair
        mapping[a].append((b, sim))
        mapping[b].append((a, sim))
    result: Dict[str, List[str]] = {}
    for key, vals in mapping.items():
        vals.sort(key=lambda x: -x[1])
        result[key] = [v[0] for v in vals]
    return result

def pick_recording(recordings: List[Dict[str, Any]], index: int = 0) -> Optional[str]:
    if not recordings:
        return None
    if index < len(recordings):
        return recordings[index].get("file")
    return recordings[0].get("file")

def select_candidates_topk(
    gold: str,
    candidates: List[str],
    k: int,
    rng: random.Random,
) -> List[str]:
    k = max(1, k)
    ordered: List[str] = []
    if gold:
        ordered.append(gold)
    pool = [c for c in candidates if c != gold]
    rng.shuffle(pool)
    ordered.extend(pool[: max(0, k - len(ordered))])
    return ordered[:k]

def build_full_intent_candidates(
    reference_intent: str,
    intent_inventory: List[str],
) -> List[str]:
    ordered: List[str] = []
    if reference_intent:
        ordered.append(reference_intent)
    for intent in intent_inventory:
        if intent and intent not in ordered:
            ordered.append(intent)
    return ordered

def select_slot_types_topk(
    gold_types: List[str],
    slot_types: List[str],
    k: int,
    rng: random.Random,
) -> List[str]:
    k = max(1, k)
    ordered: List[str] = []
    for t in gold_types:
        if t not in ordered:
            ordered.append(t)
    pool = [t for t in slot_types if t not in ordered]
    rng.shuffle(pool)
    ordered.extend(pool[: max(0, k - len(ordered))])
    return ordered[:k]

def select_slot_candidates_topk(
    gold_types: List[str],
    slot_types: List[str],
    k: int,
    rng: random.Random,
) -> List[str]:
    return select_slot_types_topk(gold_types, slot_types, k, rng)

def compose_intent(scenario: str, action: str) -> str:
    if not scenario or not action:
        return ""
    return f"{scenario}:{action}"

def load_intents_from_slurp_splits(paths: List[str]) -> List[str]:
    intents = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        for record in read_jsonl(path):
            intent = compose_intent(
                str(record.get("scenario", "")).strip(),
                str(record.get("action", "")).strip(),
            )
            if intent:
                intents.add(intent)
    return sorted(intents)

def load_slurp_map(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        for record in read_jsonl(path):
            slurp_id = str(record.get("slurp_id", "")).strip()
            if slurp_id and slurp_id not in mapping:
                mapping[slurp_id] = record
    return mapping

def split_intent(intent: str) -> Tuple[str, str]:
    if not intent or ":" not in intent:
        return "", ""
    scenario, action = intent.split(":", 1)
    return scenario.strip(), action.strip()

def postprocess_rationale_output(
    parsed: Optional[Dict[str, Any]],
    fallback_intent: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return parsed

    final_prediction = parsed.get("final_prediction")
    if not isinstance(final_prediction, dict):
        final_prediction = {}

    intent = str(final_prediction.get("intent", "")).strip()
    if not intent:
        topk_intents = parsed.get("topk_intents", [])
        if isinstance(topk_intents, list):
            for cand in topk_intents:
                if isinstance(cand, dict) and cand.get("intent"):
                    intent = str(cand["intent"]).strip()
                    break
    if not intent:
        intent = fallback_intent

    scenario_from_intent, action_from_intent = split_intent(intent)
    if intent:
        final_prediction["intent"] = intent
    if scenario_from_intent:
        final_prediction["scenario"] = scenario_from_intent
    if action_from_intent:
        final_prediction["action"] = action_from_intent

    parsed["final_prediction"] = final_prediction
    return parsed

def validate_topk_intents(
    parsed: Optional[Dict[str, Any]],
    intent_candidates: List[str],
    reference_intent: str,
) -> Tuple[bool, str]:
    if not isinstance(parsed, dict):
        return False, "output is not valid JSON object"
    topk_intents = parsed.get("topk_intents")
    if not isinstance(topk_intents, list):
        return False, "topk_intents is not a list"
    if len(topk_intents) != 5:
        return False, f"topk_intents must contain exactly 5 items, got {len(topk_intents)}"

    normalized: List[str] = []
    for idx, item in enumerate(topk_intents):
        if not isinstance(item, dict):
            return False, f"topk_intents[{idx}] is not an object"
        intent = str(item.get("intent", "")).strip()
        if not intent:
            return False, f"topk_intents[{idx}].intent is empty"
        normalized.append(intent)

    if len(set(normalized)) != 5:
        return False, "topk_intents must contain 5 unique intents"
    invalid = [intent for intent in normalized if intent not in intent_candidates]
    if invalid:
        return False, f"topk_intents contains intents outside candidates: {invalid}"
    if reference_intent and normalized.count(reference_intent) != 1:
        return False, "topk_intents must include reference_intent exactly once"
    return True, ""

def build_retry_prompt(base_prompt: str, previous_output: str, error_reason: str, stage_name: str) -> str:
    return (
        f"Your previous output violated {stage_name} constraints.\n"
        f"Violation: {error_reason}\n"
        "Regenerate the COMPLETE JSON from scratch.\n"
        "Hard constraints:\n"
        "- Output JSON ONLY.\n\n"
        "ORIGINAL TASK:\n"
        f"{base_prompt}\n\n"
        "PREVIOUS OUTPUT (invalid):\n"
        f"{previous_output}"
    )

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        return None

def to_topk_dicts(intents: List[str]) -> List[Dict[str, str]]:
    return [{"intent": intent} for intent in intents[:5]]

def extract_topk_intents(parsed: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(parsed, dict):
        return []
    topk_intents = parsed.get("topk_intents")
    if not isinstance(topk_intents, list):
        return []
    results: List[str] = []
    for item in topk_intents:
        if not isinstance(item, dict):
            continue
        intent = str(item.get("intent", "")).strip()
        if intent:
            results.append(intent)
    return results

def validate_stage2_output(
    parsed: Optional[Dict[str, Any]],
    stage1_topk: List[str],
    reference_intent: str,
) -> Tuple[bool, str]:
    if not isinstance(parsed, dict):
        return False, "output is not valid JSON object"
    expected_topk = stage1_topk[:5]
    expected_set = set(expected_topk)
    if len(expected_topk) != 5:
        return False, "stage1 topk does not contain 5 intents"

    topk_now = extract_topk_intents(parsed)
    if len(topk_now) != 5:
        return False, "topk_intents must contain exactly 5 items"
    if set(topk_now) != expected_set:
        return False, "stage2 topk_intents must match stage1 candidates"

    intent_elimination = parsed.get("intent_elimination")
    if not isinstance(intent_elimination, list):
        return False, "intent_elimination is not a list"
    if len(intent_elimination) != 4:
        return False, f"intent_elimination must contain exactly 4 items, got {len(intent_elimination)}"

    non_reference = [intent for intent in expected_topk if intent != reference_intent]
    seen = set()
    for idx, item in enumerate(intent_elimination):
        if not isinstance(item, dict):
            return False, f"intent_elimination[{idx}] is not an object"
        intent = str(item.get("intent", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not intent:
            return False, f"intent_elimination[{idx}].intent is empty"
        if intent not in non_reference:
            return False, f"intent_elimination[{idx}].intent must be one of the non-reference topk intents"
        if intent in seen:
            return False, "intent_elimination contains duplicated intents"
        seen.add(intent)
        if not reason:
            return False, f"intent_elimination[{idx}].reason is empty"

    final_prediction = parsed.get("final_prediction")
    if not isinstance(final_prediction, dict):
        return False, "final_prediction is not an object"
    final_intent = str(final_prediction.get("intent", "")).strip()
    if not final_intent:
        return False, "final_prediction.intent is empty"
    if final_intent not in expected_set:
        return False, "final_prediction.intent must be included in stage1 topk"
    return True, ""

def build_stage1_prompt_nbest(
    gold_intent: str,
    gold_slot_types: List[str],
    intent_candidates: List[str],
    slot_candidates: List[str],
    nbest_texts: List[str],
    stable_tokens: List[str],
    unstable_tokens: List[str],
    use_fewshot: bool = False,
) -> str:
    intent_note = f"intent_candidates ({len(intent_candidates)}):"
    slot_note = f"allowed_slot_types ({len(slot_candidates)}):"
    interpretations_text = format_interpretations(nbest_texts)
    fewshot = ""
    if use_fewshot:
        fewshot = (
            "EXAMPLE INPUT:\n"
            "reference_intent: play:music\n"
            "reference_slot_types: [\"song_name\"]\n"
            "intent_candidates (5): [\"play:music\",\"music:query\",\"alarm:set\",\"weather:query\",\"qa:factoid\"]\n"
            "allowed_slot_types (5): [\"song_name\",\"artist_name\",\"time\",\"date\",\"place_name\"]\n"
            "interpretations:\n"
            "interpretation_1: play yesterday\n"
            "interpretation_2: play yester day\n"
            "interpretation_3: please play yesterday\n"
            "stable_tokens: [\"play\",\"yesterday\"]\n"
            "unstable_tokens: []\n"
            "decision_pivots: [\"play\",\"yesterday\"]\n"
            "EXAMPLE OUTPUT:\n"
            "{\n"
            "  \"topk_intents\": [\n"
            "    {\"intent\": \"play:music\"},\n"
            "    {\"intent\": \"music:query\"},\n"
            "    {\"intent\": \"alarm:set\"},\n"
            "    {\"intent\": \"weather:query\"},\n"
            "    {\"intent\": \"qa:factoid\"}\n"
            "  ]\n"
            "}\n\n"
        )
    return (
        "You are a teacher model in STAGE-1 candidate generation.\n"
        "Use ONLY the information provided below. Output ENGLISH ONLY. Output JSON ONLY.\n\n"
        "==================================================\n"
        "GENERAL RULES\n"
        "==================================================\n"
        "- topk_intents MUST contain exactly 5 unique intents from intent_candidates.\n"
        "- topk_intents MUST include reference_intent exactly once.\n"
        "- Output ONLY topk_intents in JSON.\n\n"
        "==================================================\n"
        "INPUT (REFERENCE + CANDIDATES + INTERPRETATIONS)\n"
        "==================================================\n"
        f"reference_intent: {gold_intent}\n"
        f"reference_slot_types: {json.dumps(gold_slot_types, ensure_ascii=False)}\n"
        f"{intent_note} {json.dumps(intent_candidates, ensure_ascii=False)}\n"
        f"{slot_note} {json.dumps(slot_candidates, ensure_ascii=False)}\n"
        "interpretations:\n"
        f"{interpretations_text}\n"
        f"stable_tokens: {json.dumps(stable_tokens, ensure_ascii=False)}\n"
        f"unstable_tokens: {json.dumps(unstable_tokens, ensure_ascii=False)}\n"
        "decision_pivots: []\n\n"
        "==================================================\n"
        "REASONING PROCEDURE (FOLLOW IN ORDER)\n"
        "==================================================\n"
        "Step 1: TOP-5 INTENT CANDIDATES\n"
        "- Use ONLY intent_candidates.\n"
        "- Produce EXACTLY 5 unique intents and include reference_intent exactly once.\n"
        "- Focus on intents only in Stage-1.\n\n"
        "==================================================\n"
        "OUTPUT FORMAT (STRICT JSON)\n"
        "==================================================\n"
        "{\n"
        "  \"topk_intents\": [\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"}\n"
        "  ]\n"
        "}\n\n"
        f"{fewshot}"
        "Now produce the JSON for the given INPUT."
    )


def build_stage1_prompt_audio(
    gold_intent: str,
    gold_slot_types: List[str],
    intent_candidates: List[str],
    slot_candidates: List[str],
    use_fewshot: bool = False,
) -> str:
    intent_note = f"intent_candidates ({len(intent_candidates)}):"
    slot_note = f"allowed_slot_types ({len(slot_candidates)}):"
    fewshot = ""
    if use_fewshot:
        fewshot = (
            "EXAMPLE INPUT:\n"
            "reference_intent: alarm:set\n"
            "reference_slot_types: [\"time\"]\n"
            "intent_candidates (5): [\"alarm:set\",\"alarm:query\",\"calendar:set\",\"datetime:query\",\"general:greet\"]\n"
            "allowed_slot_types (5): [\"time\",\"date\",\"location\",\"song_name\",\"person\"]\n"
            "interpretations:\n"
            "interpretation_1: (audio only)\n"
            "stable_tokens: []\n"
            "unstable_tokens: []\n"
            "decision_pivots: [\"time\"]\n"
            "EXAMPLE OUTPUT:\n"
            "{\n"
            "  \"topk_intents\": [\n"
            "    {\"intent\": \"alarm:set\"},\n"
            "    {\"intent\": \"alarm:query\"},\n"
            "    {\"intent\": \"calendar:set\"},\n"
            "    {\"intent\": \"datetime:query\"},\n"
            "    {\"intent\": \"general:greet\"}\n"
            "  ]\n"
            "}\n\n"
        )
    return (
        "You are a teacher model in STAGE-1 candidate generation.\n"
        "Use ONLY the information provided below. Output ENGLISH ONLY. Output JSON ONLY.\n\n"
        "==================================================\n"
        "GENERAL RULES\n"
        "==================================================\n"
        "- topk_intents MUST contain exactly 5 unique intents from intent_candidates.\n"
        "- topk_intents MUST include reference_intent exactly once.\n"
        "- Output ONLY topk_intents in JSON.\n\n"
        "==================================================\n"
        "INPUT (REFERENCE + CANDIDATES)\n"
        "==================================================\n"
        f"reference_intent: {gold_intent}\n"
        f"reference_slot_types: {json.dumps(gold_slot_types, ensure_ascii=False)}\n"
        f"{intent_note} {json.dumps(intent_candidates, ensure_ascii=False)}\n"
        f"{slot_note} {json.dumps(slot_candidates, ensure_ascii=False)}\n"
        "interpretations:\n"
        "interpretation_1: (audio only)\n"
        "stable_tokens: []\n"
        "unstable_tokens: []\n"
        "decision_pivots: []\n\n"
        "==================================================\n"
        "REASONING PROCEDURE (FOLLOW IN ORDER)\n"
        "==================================================\n"
        "Step 1: TOP-5 INTENT CANDIDATES\n"
        "- Use ONLY intent_candidates.\n"
        "- Produce EXACTLY 5 unique intents and include reference_intent exactly once.\n"
        "- Focus on intents only in Stage-1.\n\n"
        "==================================================\n"
        "OUTPUT FORMAT (STRICT JSON)\n"
        "==================================================\n"
        "{\n"
        "  \"topk_intents\": [\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"}\n"
        "  ]\n"
        "}\n\n"
        f"{fewshot}"
        "Now produce the JSON for the given INPUT."
    )

def build_stage2_prompt_nbest(
    gold_intent: str,
    gold_slot_types: List[str],
    slot_candidates: List[str],
    nbest_texts: List[str],
    stable_tokens: List[str],
    unstable_tokens: List[str],
    stage1_topk: List[str],
    use_fewshot: bool = False,
) -> str:
    topk_json = json.dumps(to_topk_dicts(stage1_topk), ensure_ascii=False)
    interpretations_text = format_interpretations(nbest_texts)
    fewshot = ""
    if use_fewshot:
        fewshot = (
            "EXAMPLE INPUT:\n"
            "reference_intent: play:music\n"
            "reference_slot_types: [\"song_name\"]\n"
            "stage1_topk_intents: [{\"intent\":\"play:music\"},{\"intent\":\"music:query\"},{\"intent\":\"alarm:set\"},{\"intent\":\"weather:query\"},{\"intent\":\"qa:factoid\"}]\n"
            "interpretations:\n"
            "interpretation_1: play yesterday\n"
            "interpretation_2: please play yesterday\n"
            "stable_tokens: [\"play\",\"yesterday\"]\n"
            "unstable_tokens: []\n"
            "EXAMPLE OUTPUT:\n"
            "{\n"
            "  \"interpretation_uncertainty_analysis\": {\n"
            "    \"stable_cues\": [\"play\",\"yesterday\"],\n"
            "    \"unstable_cues\": [],\n"
            "    \"decision_pivots\": [\"play\",\"yesterday\"]\n"
            "  },\n"
            "  \"topk_intents\": [\n"
            "    {\"intent\": \"play:music\"},\n"
            "    {\"intent\": \"music:query\"},\n"
            "    {\"intent\": \"alarm:set\"},\n"
            "    {\"intent\": \"weather:query\"},\n"
            "    {\"intent\": \"qa:factoid\"}\n"
            "  ],\n"
            "  \"intent_elimination\": [\n"
            "    {\"intent\": \"music:query\", \"reason\": \"command style favors execution over info retrieval\"},\n"
            "    {\"intent\": \"alarm:set\", \"reason\": \"no alarm-time request cues appear in hypotheses\"},\n"
            "    {\"intent\": \"weather:query\", \"reason\": \"lexical focus is media playback rather than weather\"},\n"
            "    {\"intent\": \"qa:factoid\", \"reason\": \"utterance asks action, not a factual answer\"}\n"
            "  ],\n"
            "  \"final_prediction\": {\"intent\": \"play:music\", \"scenario\": \"play\", \"action\": \"music\"},\n"
            "  \"slot_grounding\": [\n"
            "    {\"slot_type\": \"song_name\", \"supported\": true, \"best_span\": \"yesterday\", \"source_hypothesis\": \"interpretation_1\"}\n"
            "  ],\n"
            "  \"final_rationalization\": \"stable playback cues support play:music with song_name grounding\"\n"
            "}\n\n"
        )
    return (
        "You are a teacher model in STAGE-2 candidate pruning and rationale generation.\n"
        "Use ONLY the information provided below. Output ENGLISH ONLY. Output JSON ONLY.\n\n"
        "==================================================\n"
        "GENERAL RULES\n"
        "==================================================\n"
        "- Use stage1_topk_intents as fixed candidates.\n"
        "- topk_intents in output must contain the same 5 intents as stage1_topk_intents.\n"
        "- intent_elimination MUST contain exactly 4 non-reference intents from stage1_topk_intents.\n"
        "- final_prediction.intent MUST be one of stage1_topk_intents.\n"
        "- Do NOT invent slot types outside allowed_slot_types.\n\n"
        "==================================================\n"
        "INPUT\n"
        "==================================================\n"
        f"reference_intent: {gold_intent}\n"
        f"reference_slot_types: {json.dumps(gold_slot_types, ensure_ascii=False)}\n"
        f"allowed_slot_types ({len(slot_candidates)}): {json.dumps(slot_candidates, ensure_ascii=False)}\n"
        f"stage1_topk_intents: {topk_json}\n"
        "interpretations:\n"
        f"{interpretations_text}\n"
        f"stable_tokens: {json.dumps(stable_tokens, ensure_ascii=False)}\n"
        f"unstable_tokens: {json.dumps(unstable_tokens, ensure_ascii=False)}\n"
        "decision_pivots: []\n\n"
        "==================================================\n"
        "OUTPUT FORMAT (STRICT JSON)\n"
        "==================================================\n"
        "{\n"
        "  \"interpretation_uncertainty_analysis\": {\n"
        "    \"stable_cues\": [],\n"
        "    \"unstable_cues\": [],\n"
        "    \"decision_pivots\": []\n"
        "  },\n"
        "  \"topk_intents\": [\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"}\n"
        "  ],\n"
        "  \"intent_elimination\": [\n"
        "    {\"intent\": \"\", \"reason\": \"\"},\n"
        "    {\"intent\": \"\", \"reason\": \"\"},\n"
        "    {\"intent\": \"\", \"reason\": \"\"},\n"
        "    {\"intent\": \"\", \"reason\": \"\"}\n"
        "  ],\n"
        "  \"final_prediction\": {\"intent\": \"\", \"scenario\": \"\", \"action\": \"\"},\n"
        "  \"slot_grounding\": [\n"
        "    {\"slot_type\": \"\", \"supported\": true, \"best_span\": \"\", \"source_hypothesis\": \"\"}\n"
        "  ],\n"
        "  \"final_rationalization\": \"\"\n"
        "}\n\n"
        f"{fewshot}"
        "Now produce the JSON for the given INPUT."
    )

def build_stage2_prompt_audio(
    gold_intent: str,
    gold_slot_types: List[str],
    slot_candidates: List[str],
    stage1_topk: List[str],
    use_fewshot: bool = False,
) -> str:
    topk_json = json.dumps(to_topk_dicts(stage1_topk), ensure_ascii=False)
    fewshot = ""
    if use_fewshot:
        fewshot = (
            "EXAMPLE INPUT:\n"
            "reference_intent: alarm:set\n"
            "reference_slot_types: [\"time\"]\n"
            "stage1_topk_intents: [{\"intent\":\"alarm:set\"},{\"intent\":\"alarm:query\"},{\"intent\":\"calendar:set\"},{\"intent\":\"datetime:query\"},{\"intent\":\"general:greet\"}]\n"
            "EXAMPLE OUTPUT:\n"
            "{\n"
            "  \"interpretation_uncertainty_analysis\": {\n"
            "    \"stable_cues\": [\"wake-up\"],\n"
            "    \"unstable_cues\": [],\n"
            "    \"decision_pivots\": [\"time\"]\n"
            "  },\n"
            "  \"topk_intents\": [\n"
            "    {\"intent\": \"alarm:set\"},\n"
            "    {\"intent\": \"alarm:query\"},\n"
            "    {\"intent\": \"calendar:set\"},\n"
            "    {\"intent\": \"datetime:query\"},\n"
            "    {\"intent\": \"general:greet\"}\n"
            "  ],\n"
            "  \"intent_elimination\": [\n"
            "    {\"intent\": \"alarm:query\", \"reason\": \"prosody indicates command execution, not a query\"},\n"
            "    {\"intent\": \"calendar:set\", \"reason\": \"missing event-centric content for calendar creation\"},\n"
            "    {\"intent\": \"datetime:query\", \"reason\": \"request sets a time rather than asking current time\"},\n"
            "    {\"intent\": \"general:greet\", \"reason\": \"utterance has task parameters, not social greeting\"}\n"
            "  ],\n"
            "  \"final_prediction\": {\"intent\": \"alarm:set\", \"scenario\": \"alarm\", \"action\": \"set\"},\n"
            "  \"slot_grounding\": [\n"
            "    {\"slot_type\": \"time\", \"supported\": true, \"best_span\": \"time\", \"source_hypothesis\": \"interpretation_1\"}\n"
            "  ],\n"
            "  \"final_rationalization\": \"audio command cues support alarm:set with time slot\"\n"
            "}\n\n"
        )
    return (
        "You are a teacher model in STAGE-2 candidate pruning and rationale generation.\n"
        "Use ONLY the information provided below. Output ENGLISH ONLY. Output JSON ONLY.\n\n"
        "==================================================\n"
        "GENERAL RULES\n"
        "==================================================\n"
        "- Use stage1_topk_intents as fixed candidates.\n"
        "- topk_intents in output must contain the same 5 intents as stage1_topk_intents.\n"
        "- intent_elimination MUST contain exactly 4 non-reference intents from stage1_topk_intents.\n"
        "- final_prediction.intent MUST be one of stage1_topk_intents.\n"
        "- source_hypothesis must be interpretation_1 or \"none\".\n\n"
        "==================================================\n"
        "INPUT\n"
        "==================================================\n"
        f"reference_intent: {gold_intent}\n"
        f"reference_slot_types: {json.dumps(gold_slot_types, ensure_ascii=False)}\n"
        f"allowed_slot_types ({len(slot_candidates)}): {json.dumps(slot_candidates, ensure_ascii=False)}\n"
        f"stage1_topk_intents: {topk_json}\n"
        "interpretations:\n"
        "interpretation_1: (audio only)\n"
        "stable_tokens: []\n"
        "unstable_tokens: []\n"
        "decision_pivots: []\n\n"
        "==================================================\n"
        "OUTPUT FORMAT (STRICT JSON)\n"
        "==================================================\n"
        "{\n"
        "  \"interpretation_uncertainty_analysis\": {\n"
        "    \"stable_cues\": [],\n"
        "    \"unstable_cues\": [],\n"
        "    \"decision_pivots\": []\n"
        "  },\n"
        "  \"topk_intents\": [\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"},\n"
        "    {\"intent\": \"\"}\n"
        "  ],\n"
        "  \"intent_elimination\": [\n"
        "    {\"intent\": \"\", \"reason\": \"\"},\n"
        "    {\"intent\": \"\", \"reason\": \"\"},\n"
        "    {\"intent\": \"\", \"reason\": \"\"},\n"
        "    {\"intent\": \"\", \"reason\": \"\"}\n"
        "  ],\n"
        "  \"final_prediction\": {\"intent\": \"\", \"scenario\": \"\", \"action\": \"\"},\n"
        "  \"slot_grounding\": [\n"
        "    {\"slot_type\": \"\", \"supported\": true, \"best_span\": \"\", \"source_hypothesis\": \"\"}\n"
        "  ],\n"
        "  \"final_rationalization\": \"\"\n"
        "}\n\n"
        f"{fewshot}"
        "Now produce the JSON for the given INPUT."
    )

def run_model_once(
    processor: AutoProcessor,
    model: Qwen2AudioForConditionalGeneration,
    mode: str,
    prompt: str,
    device: str,
    gen_kwargs: Dict[str, Any],
    audio: Optional[List[float]],
    sampling_rate: int,
) -> str:
    if mode == "nbest":
        user_content = [{"type": "text", "text": prompt}]
        text_input = processor.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=text_input, return_tensors="pt")
    else:
        user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": prompt}]
        text_input = processor.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=text_input, audio=[audio], sampling_rate=sampling_rate, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate two-stage rationale with Qwen2-Audio (stage1 candidates + stage2 pruning).")
    parser.add_argument("--mode", type=str, choices=["nbest", "audio"], default="nbest")
    parser.add_argument("--input_file", type=str, default="Experiment_Rationale/real_asr_sampling_data.jsonl")
    parser.add_argument("--slurp_file", type=str, default="slurp/dataset/slurp/test.jsonl", help="Additional fallback SLURP file for missing fields.")
    parser.add_argument("--slurp_train_file", type=str, default="slurp/dataset/slurp/train.jsonl", help="SLURP train split used to build intent inventory.")
    parser.add_argument("--slurp_devel_file", type=str, default="slurp/dataset/slurp/devel.jsonl", help="SLURP devel split used to build intent inventory.")
    parser.add_argument("--slurp_test_file", type=str, default="slurp/dataset/slurp/test.jsonl", help="SLURP test split used to build intent inventory.")
    parser.add_argument("--audio_dir", type=str, default="slurp/slurp_real")
    parser.add_argument("--metadata_file", type=str, default="Experiment_3/slurp_metadata.json")
    parser.add_argument("--clusters_file", type=str, default="Experiment_3/slurp_clusters.json")
    parser.add_argument("--confusing_pairs_file", type=str, default="Experiment_3/slurp_confusing_pairs.json")
    parser.add_argument("--output_file", type=str, default="Experiment_Rationale/rationale_output_two_stage.jsonl")
    parser.add_argument("--output_mode", type=str, default="raw", choices=["raw", "full"], help="raw: write compact records with raw outputs; full: write full metadata JSON.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recording_index", type=int, default=0)
    parser.add_argument("--num_hypotheses", type=int, default=5)
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidates for slot types.")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview", type=int, default=0, help="Print prompt and output for first N samples.")
    parser.add_argument("--limitmode", action="store_true", help="Print pretty JSON results to stdout.")
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--use_fewshot", action="store_true", help="Enable built-in few-shot exemplars in prompts.")
    parser.add_argument("--format_retries", type=int, default=2, help="Retry count when topk_intents format constraints are violated.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input_file) if not os.path.isabs(args.input_file) else args.input_file
    slurp_path = os.path.join(base_dir, args.slurp_file) if not os.path.isabs(args.slurp_file) else args.slurp_file
    slurp_train_path = os.path.join(base_dir, args.slurp_train_file) if not os.path.isabs(args.slurp_train_file) else args.slurp_train_file
    slurp_devel_path = os.path.join(base_dir, args.slurp_devel_file) if not os.path.isabs(args.slurp_devel_file) else args.slurp_devel_file
    slurp_test_path = os.path.join(base_dir, args.slurp_test_file) if not os.path.isabs(args.slurp_test_file) else args.slurp_test_file
    audio_root = os.path.join(base_dir, args.audio_dir) if not os.path.isabs(args.audio_dir) else args.audio_dir
    metadata_path = os.path.join(base_dir, args.metadata_file) if not os.path.isabs(args.metadata_file) else args.metadata_file
    clusters_path = os.path.join(base_dir, args.clusters_file) if not os.path.isabs(args.clusters_file) else args.clusters_file
    confusing_path = os.path.join(base_dir, args.confusing_pairs_file) if not os.path.isabs(args.confusing_pairs_file) else args.confusing_pairs_file
    output_path = os.path.join(base_dir, args.output_file) if not os.path.isabs(args.output_file) else args.output_file

    rng = random.Random(args.seed)

    metadata = load_metadata(metadata_path)
    # clusters/confusing_pairs args are kept for compatibility (currently unused)

    split_paths = [slurp_train_path, slurp_devel_path, slurp_test_path]
    intent_inventory = load_intents_from_slurp_splits(split_paths)
    if intent_inventory:
        print(f"[INFO] Loaded {len(intent_inventory)} intents from train/devel/test splits.")
    else:
        intent_inventory = metadata["intents"]
        print(f"[WARN] Could not build intents from splits. Falling back to metadata intents ({len(intent_inventory)}).")

    slurp_map = load_slurp_map(split_paths + [slurp_path])

    items = read_jsonl(input_path)
    if args.limit:
        items = items[: args.limit]

    if not items:
        print(f"[ERROR] No input items found: {input_path}")
        return

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    torch_dtype = torch.bfloat16 if "cuda" in args.device else torch.float32
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(args.device)
    model.eval()

    results: List[Dict[str, Any]] = []
    raw_outputs: List[Any] = []
    sr = processor.feature_extractor.sampling_rate

    for idx, item in enumerate(tqdm(items, desc="Generating two-stage rationale", unit="sample")):
        slurp_id = str(item.get("slurp_id", ""))
        fallback = slurp_map.get(slurp_id)
        if fallback:
            for key in ["scenario", "action", "entities", "tokens", "recordings", "sentence"]:
                if item.get(key) in (None, "", [], {}):
                    item[key] = fallback.get(key)

        if args.mode == "nbest" and not item.get("asr_hypotheses"):
            continue

        scenario = item.get("scenario", "")
        action = item.get("action", "")
        gold_intent = compose_intent(scenario, action)
        input_text = str(item.get("sentence", "") or "")
        gold_entities = build_entities(item)
        gold_slot_types = extract_slot_types(gold_entities)

        # n-best texts
        nbest_texts: List[str] = []
        if args.mode == "nbest":
            for h in item.get("asr_hypotheses", [])[: args.num_hypotheses]:
                if isinstance(h, dict):
                    txt = h.get("text", "")
                else:
                    txt = str(h)
                if txt:
                    nbest_texts.append(txt.strip())
        stable_unstable = summarize_nbest(nbest_texts)

        intent_candidates = build_full_intent_candidates(gold_intent, intent_inventory)

        slot_candidates = select_slot_candidates_topk(
            gold_types=gold_slot_types,
            slot_types=metadata["slot_types"],
            k=args.num_candidates,
            rng=rng,
        )

        audio = None
        if args.mode == "audio":
            filename = pick_recording(item.get("recordings", []), args.recording_index)
            audio_path = resolve_audio_path(audio_root, filename) if filename else None
            audio = load_audio(audio_path, target_sr=sr) if audio_path else None
            if audio is None:
                continue

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": processor.tokenizer.pad_token_id,
        }
        if args.do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            })

        if args.mode == "nbest":
            stage1_prompt = build_stage1_prompt_nbest(
                gold_intent=gold_intent,
                gold_slot_types=gold_slot_types,
                intent_candidates=intent_candidates,
                slot_candidates=slot_candidates,
                nbest_texts=nbest_texts,
                stable_tokens=stable_unstable["stable"],
                unstable_tokens=stable_unstable["unstable"],
                use_fewshot=args.use_fewshot,
            )
        else:
            stage1_prompt = build_stage1_prompt_audio(
                gold_intent=gold_intent,
                gold_slot_types=gold_slot_types,
                intent_candidates=intent_candidates,
                slot_candidates=slot_candidates,
                use_fewshot=args.use_fewshot,
            )

        stage1_generated = ""
        stage1_parsed: Optional[Dict[str, Any]] = None
        stage1_topk_valid = False
        stage1_validation_error = ""
        max_attempts = max(1, args.format_retries + 1)
        for attempt in range(max_attempts):
            current_prompt = stage1_prompt if attempt == 0 else build_retry_prompt(
                stage1_prompt,
                stage1_generated,
                stage1_validation_error,
                stage_name="stage1",
            )
            stage1_generated = run_model_once(
                processor=processor,
                model=model,
                mode=args.mode,
                prompt=current_prompt,
                device=args.device,
                gen_kwargs=gen_kwargs,
                audio=audio,
                sampling_rate=sr,
            )
            stage1_parsed = extract_json(stage1_generated)
            stage1_topk_valid, stage1_validation_error = validate_topk_intents(stage1_parsed, intent_candidates, gold_intent)
            if stage1_topk_valid:
                break

        stage1_topk_intents = extract_topk_intents(stage1_parsed)
        if not stage1_topk_valid or len(stage1_topk_intents) != 5 or len(set(stage1_topk_intents)) != 5:
            stage1_topk_intents = select_candidates_topk(
                gold=gold_intent,
                candidates=intent_candidates,
                k=5,
                rng=rng,
            )
            stage1_parsed = {
                "topk_intents": to_topk_dicts(stage1_topk_intents),
                "fallback_used": True,
                "fallback_reason": stage1_validation_error or "stage1 format violation",
            }

        if args.mode == "nbest":
            stage2_prompt = build_stage2_prompt_nbest(
                gold_intent=gold_intent,
                gold_slot_types=gold_slot_types,
                slot_candidates=slot_candidates,
                nbest_texts=nbest_texts,
                stable_tokens=stable_unstable["stable"],
                unstable_tokens=stable_unstable["unstable"],
                stage1_topk=stage1_topk_intents,
                use_fewshot=args.use_fewshot,
            )
        else:
            stage2_prompt = build_stage2_prompt_audio(
                gold_intent=gold_intent,
                gold_slot_types=gold_slot_types,
                slot_candidates=slot_candidates,
                stage1_topk=stage1_topk_intents,
                use_fewshot=args.use_fewshot,
            )

        stage2_generated = ""
        stage2_parsed: Optional[Dict[str, Any]] = None
        stage2_valid = False
        stage2_validation_error = ""
        for attempt in range(max_attempts):
            current_prompt = stage2_prompt if attempt == 0 else build_retry_prompt(
                stage2_prompt,
                stage2_generated,
                stage2_validation_error,
                stage_name="stage2",
            )
            stage2_generated = run_model_once(
                processor=processor,
                model=model,
                mode=args.mode,
                prompt=current_prompt,
                device=args.device,
                gen_kwargs=gen_kwargs,
                audio=audio,
                sampling_rate=sr,
            )
            stage2_parsed = extract_json(stage2_generated)
            stage2_valid, stage2_validation_error = validate_stage2_output(stage2_parsed, stage1_topk_intents, gold_intent)
            if stage2_valid:
                break

        if not stage2_valid:
            stage2_parsed = {
                "interpretation_uncertainty_analysis": {
                    "stable_cues": stable_unstable["stable"][:5] if args.mode == "nbest" else [],
                    "unstable_cues": stable_unstable["unstable"][:5] if args.mode == "nbest" else [],
                    "decision_pivots": stable_unstable["stable"][:3] if args.mode == "nbest" else [],
                },
                "topk_intents": to_topk_dicts(stage1_topk_intents),
                "intent_elimination": [
                    {"intent": intent, "reason": "less consistent with utterance evidence than reference_intent"}
                    for intent in stage1_topk_intents
                    if intent != gold_intent
                ][:4],
                "final_prediction": {
                    "intent": gold_intent,
                    "scenario": scenario,
                    "action": action,
                },
                "slot_grounding": [
                    {
                        "slot_type": slot_type,
                        "supported": True,
                        "best_span": "",
                        "source_hypothesis": "none",
                    }
                    for slot_type in gold_slot_types
                ],
                "final_rationalization": "fallback rationale generated because stage2 output was invalid",
                "fallback_used": True,
                "fallback_reason": stage2_validation_error or "stage2 format violation",
            }

        stage2_parsed = postprocess_rationale_output(stage2_parsed, fallback_intent=gold_intent)
        stage2_parsed["topk_intents"] = to_topk_dicts(stage1_topk_intents)

        combined_rationale = {
            "candidate_generation": stage1_parsed,
            "candidate_pruning": stage2_parsed,
            "interpretation_uncertainty_analysis": stage2_parsed.get("interpretation_uncertainty_analysis", {}),
            "topk_intents": to_topk_dicts(stage1_topk_intents),
            "intent_elimination": stage2_parsed.get("intent_elimination", []),
            "final_prediction": stage2_parsed.get("final_prediction", {}),
            "slot_grounding": stage2_parsed.get("slot_grounding", []),
            "final_rationalization": stage2_parsed.get("final_rationalization", ""),
        }
        combined_raw_output = json.dumps(
            {
                "stage1_raw_output": stage1_generated,
                "stage2_raw_output": stage2_generated,
            },
            ensure_ascii=False,
        )

        result = {
            "slurp_id": slurp_id,
            "mode": args.mode,
            "input_text": input_text,
            "gold_intent": gold_intent,
            "gold_scenario": scenario,
            "gold_action": action,
            "intent_candidates": intent_candidates,
            "slot_candidates": slot_candidates,
            "nbest": nbest_texts if args.mode == "nbest" else [],
            "nbest_summary": stable_unstable if args.mode == "nbest" else {},
            "rationale": combined_rationale,
            "topk_valid": stage1_topk_valid,
            "topk_validation_error": "" if stage1_topk_valid else stage1_validation_error,
            "stage2_valid": stage2_valid,
            "stage2_validation_error": "" if stage2_valid else stage2_validation_error,
            "raw_output": combined_raw_output,
        }
        if args.save_raw:
            result["rationale_raw"] = {
                "stage1_prompt": stage1_prompt,
                "stage1_output": stage1_generated,
                "stage2_prompt": stage2_prompt,
                "stage2_output": stage2_generated,
            }
        raw_record = {
            "slurp_id": slurp_id,
            "input_text": input_text,
            "gold_intent": gold_intent,
            "raw_output": combined_raw_output,
            "topk_valid": stage1_topk_valid,
            "topk_validation_error": "" if stage1_topk_valid else stage1_validation_error,
            "stage2_valid": stage2_valid,
            "stage2_validation_error": "" if stage2_valid else stage2_validation_error,
        }
        raw_outputs.append(raw_record)
        if args.output_mode == "full":
            results.append(result)

        if args.preview and idx < args.preview:
            print("=" * 80)
            print(f"[PREVIEW] {idx+1} / {args.preview} | slurp_id={slurp_id} | mode={args.mode}")
            print("-" * 80)
            print("STAGE1 PROMPT:")
            print(stage1_prompt)
            print("-" * 80)
            print("STAGE1 OUTPUT:")
            print(stage1_generated)
            print("-" * 80)
            print("STAGE2 PROMPT:")
            print(stage2_prompt)
            print("-" * 80)
            print("STAGE2 OUTPUT:")
            print(stage2_generated)
        if args.limitmode:
            limit_view = {
                "slurp_id": slurp_id,
                "input_text": input_text,
                "gold_intent": gold_intent,
                "stage1_prompt": stage1_prompt,
                "stage1_raw_output": stage1_generated,
                "stage2_prompt": stage2_prompt,
                "stage2_raw_output": stage2_generated,
            }
            print(json.dumps(limit_view, ensure_ascii=False, indent=2))
            print("")

    if args.output_mode == "full":
        write_jsonl(output_path, results)
    else:
        write_raw_jsonl(output_path, raw_outputs)
    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    main()
