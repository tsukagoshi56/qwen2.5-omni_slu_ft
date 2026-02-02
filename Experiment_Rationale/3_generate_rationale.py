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

def append_worker_suffix(path: str, worker_rank: int, num_workers: int) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}.w{worker_rank}of{num_workers}{ext or '.jsonl'}"

def shard_items_by_worker(
    items: List[Dict[str, Any]],
    num_workers: int,
    worker_rank: int,
) -> List[Dict[str, Any]]:
    if num_workers <= 1:
        return items
    return [item for idx, item in enumerate(items) if idx % num_workers == worker_rank]

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

def parse_nbest_schedule(
    nbest_values: str,
    default_num_hypotheses: int,
    ablation_1to5: bool,
) -> List[int]:
    if ablation_1to5:
        return [1, 2, 3, 4, 5]
    if not nbest_values:
        return [max(1, int(default_num_hypotheses))]

    schedule: List[int] = []
    for token in nbest_values.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            k = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid n-best value: {token}") from exc
        if k < 1:
            raise ValueError(f"n-best value must be >= 1, got {k}")
        if k not in schedule:
            schedule.append(k)
    if not schedule:
        raise ValueError("No valid n-best values were provided.")
    return schedule

def resolve_output_path_for_k(output_path: str, k: int, schedule: List[int]) -> str:
    if len(schedule) == 1:
        return output_path
    if "{k}" in output_path:
        return output_path.format(k=k)
    root, ext = os.path.splitext(output_path)
    return f"{root}.k{k}{ext or '.jsonl'}"

def build_smoke_rationale_output(
    gold_intent: str,
    gold_slot_types: List[str],
    intent_candidates: List[str],
    nbest_texts: List[str],
) -> Dict[str, Any]:
    ordered_intents: List[str] = []
    if gold_intent:
        ordered_intents.append(gold_intent)
    for cand in intent_candidates:
        if cand and cand not in ordered_intents:
            ordered_intents.append(cand)
        if len(ordered_intents) >= 5:
            break
    topk = ordered_intents[:5]
    intent_elimination = [
        {
            "intent": intent,
            "reason": "smoke mode placeholder rationale for I/O validation",
        }
        for intent in topk
        if intent != gold_intent
    ][:4]
    source_hypothesis = "interpretation_1" if nbest_texts else "none"
    slot_grounding = []
    for slot_type in gold_slot_types:
        slot_grounding.append(
            {
                "slot_type": slot_type,
                "supported": bool(nbest_texts),
                "best_span": nbest_texts[0] if nbest_texts else "",
                "source_hypothesis": source_hypothesis,
            }
        )
    scenario, action = split_intent(gold_intent)
    return {
        "interpretation_uncertainty_analysis": {
            "stable_cues": [],
            "unstable_cues": [],
            "decision_pivots": [],
        },
        "topk_intents": [{"intent": intent} for intent in topk],
        "intent_elimination": intent_elimination,
        "final_prediction": {
            "intent": gold_intent,
            "scenario": scenario,
            "action": action,
        },
        "slot_grounding": slot_grounding,
        "final_rationalization": "smoke mode placeholder rationale",
    }

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
    normalized_reference = normalize_intent_label(reference_intent)
    if normalized_reference:
        ordered.append(normalized_reference)
    for intent in intent_inventory:
        normalized_intent = normalize_intent_label(intent)
        if normalized_intent and normalized_intent not in ordered:
            ordered.append(normalized_intent)
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

def build_full_slot_candidates(
    reference_slot_types: List[str],
    slot_inventory: List[str],
) -> List[str]:
    ordered: List[str] = []
    for slot_type in reference_slot_types:
        if slot_type and slot_type not in ordered:
            ordered.append(slot_type)
    for slot_type in slot_inventory:
        if slot_type and slot_type not in ordered:
            ordered.append(slot_type)
    return ordered

def normalize_intent_label(intent: str) -> str:
    return str(intent or "").strip().replace(":", "_")

def compose_intent(scenario: str, action: str) -> str:
    if not scenario or not action:
        return ""
    return f"{scenario}_{action}"

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
    if not intent:
        return "", ""
    if "_" in intent:
        scenario, action = intent.split("_", 1)
    elif ":" in intent:
        scenario, action = intent.split(":", 1)
    else:
        return "", ""
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

    intent = normalize_intent_label(final_prediction.get("intent", ""))
    if not intent:
        topk_intents = parsed.get("topk_intents", [])
        if isinstance(topk_intents, list):
            for cand in topk_intents:
                if isinstance(cand, dict) and cand.get("intent"):
                    intent = normalize_intent_label(cand["intent"])
                    break
    if not intent:
        intent = normalize_intent_label(fallback_intent)

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
        intent = normalize_intent_label(item.get("intent", ""))
        if not intent:
            return False, f"topk_intents[{idx}].intent is empty"
        normalized.append(intent)

    if len(set(normalized)) != 5:
        return False, "topk_intents must contain 5 unique intents"
    normalized_candidates = {normalize_intent_label(intent) for intent in intent_candidates}
    invalid = [intent for intent in normalized if intent not in normalized_candidates]
    if invalid:
        return False, f"topk_intents contains intents outside candidates: {invalid}"
    normalized_reference = normalize_intent_label(reference_intent)
    if normalized_reference and normalized.count(normalized_reference) != 1:
        return False, "topk_intents must include reference_intent exactly once"
    return True, ""

def build_retry_prompt(base_prompt: str, previous_output: str, error_reason: str) -> str:
    return (
        "Your previous output violated format constraints.\n"
        f"Violation: {error_reason}\n"
        "Regenerate the COMPLETE JSON from scratch.\n"
        "Hard constraints:\n"
        "- topk_intents must have exactly 5 unique items.\n"
        "- each intent in topk_intents must come from intent_candidates.\n"
        "- topk_intents must include reference_intent exactly once.\n"
        "- intent_elimination should explain only the 4 non-reference intents from topk_intents.\n"
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

def build_prompt_nbest(
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
            "reference_intent: play_music\n"
            "reference_slot_types: [\"song_name\"]\n"
            "intent_candidates (5): [\"play_music\",\"music_query\",\"alarm_set\",\"weather_query\",\"qa_factoid\"]\n"
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
            "  \"interpretation_uncertainty_analysis\": {\n"
            "    \"stable_cues\": [\"play\",\"yesterday\"],\n"
            "    \"unstable_cues\": [],\n"
            "    \"decision_pivots\": [\"play\",\"yesterday\"]\n"
            "  },\n"
            "  \"topk_intents\": [\n"
            "    {\"intent\": \"play_music\"},\n"
            "    {\"intent\": \"music_query\"},\n"
            "    {\"intent\": \"alarm_set\"},\n"
            "    {\"intent\": \"weather_query\"},\n"
            "    {\"intent\": \"qa_factoid\"}\n"
            "  ],\n"
            "  \"intent_elimination\": [\n"
            "    {\"intent\": \"music_query\", \"reason\": \"utterance is command style; no interrogative pattern for retrieval\"},\n"
            "    {\"intent\": \"alarm_set\", \"reason\": \"lexical focus is media playback, not alarm scheduling parameters\"},\n"
            "    {\"intent\": \"weather_query\", \"reason\": \"content refers to a track name, not meteorological information request\"},\n"
            "    {\"intent\": \"qa_factoid\", \"reason\": \"intent asks execution of playback action, not fact-seeking answer\"}\n"
            "  ],\n"
            "  \"final_prediction\": {\n"
            "    \"intent\": \"play_music\",\n"
            "    \"scenario\": \"play\",\n"
            "    \"action\": \"music\"\n"
            "  },\n"
            "  \"slot_grounding\": [\n"
            "    {\"slot_type\": \"song_name\", \"supported\": true, \"best_span\": \"yesterday\", \"source_hypothesis\": \"interpretation_1\"}\n"
            "  ],\n"
            "  \"final_rationalization\": \"stable play,yesterday supports play-music with song_name despite minor uncertainty\"\n"
            "}\n\n"
        )
    return (
        "You are a teacher model whose role is NOT to predict labels, but to rationalize GIVEN reference labels.\n"
        "Use ONLY the information provided below. Output ENGLISH ONLY. Output JSON ONLY.\n\n"
        "==================================================\n"
        "GENERAL RULES\n"
        "==================================================\n"
        "- Do NOT invent intents or slot types outside the provided candidates.\n"
        "- Do NOT hallucinate slot values not supported by the utterance.\n"
        "- Do NOT output the reference labels in the JSON.\n"
        "- topk_intents MUST contain exactly 5 unique intents from intent_candidates.\n"
        "- topk_intents MUST include reference_intent exactly once.\n"
        "- intent_elimination MUST have exactly 4 items, each for a non-reference intent in topk_intents.\n"
        "- Do NOT repeat the same elimination wording pattern; avoid generic \"no ... cue\" only responses.\n"
        "- The maximum TOP-K for intents is fixed to 5.\n\n"
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
        "Step 1: INTERPRETATION UNCERTAINTY ANALYSIS\n"
        "- List stable_cues, unstable_cues, decision_pivots (<=5 each).\n"
        "- Do NOT reference intent/slot labels.\n"
        "Step 2: TOP-5 INTENT CANDIDATES\n"
        "- Use ONLY intent_candidates.\n"
        "- Produce EXACTLY 5 unique intents and include reference_intent exactly once.\n"
        "Step 3: INTENT ELIMINATION\n"
        "- Eliminate ONLY the 4 non-reference intents in topk_intents.\n"
        "- Each reason must be specific and non-repetitive.\n"
        "- Prefer different evidence dimensions across reasons (speech-act mismatch, domain mismatch, argument/slot mismatch, target mismatch).\n"
        "Step 4: FINAL INTENT RESOLUTION\n"
        "- Select one intent from topk_intents and split it into scenario/action by the first '_'.\n"
        "Step 5: SLOT GROUNDING\n"
        "- For EACH reference_slot_type, mark supported and give best_span and source_hypothesis.\n"
        "- source_hypothesis must be interpretation_1..interpretation_5 or \"none\".\n"
        "Step 6: FINAL RATIONALIZATION\n"
        "- One concise sentence linking cues to reference labels.\n\n"
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


def build_prompt_audio(
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
            "reference_intent: alarm_set\n"
            "reference_slot_types: [\"time\"]\n"
            "intent_candidates (5): [\"alarm_set\",\"alarm_query\",\"calendar_set\",\"datetime_query\",\"general_greet\"]\n"
            "allowed_slot_types (5): [\"time\",\"date\",\"location\",\"song_name\",\"person\"]\n"
            "interpretations:\n"
            "interpretation_1: (audio only)\n"
            "stable_tokens: []\n"
            "unstable_tokens: []\n"
            "decision_pivots: [\"time\"]\n"
            "EXAMPLE OUTPUT:\n"
            "{\n"
            "  \"interpretation_uncertainty_analysis\": {\n"
            "    \"stable_cues\": [\"time\"],\n"
            "    \"unstable_cues\": [],\n"
            "    \"decision_pivots\": [\"time\"]\n"
            "  },\n"
            "  \"topk_intents\": [\n"
            "    {\"intent\": \"alarm_set\"},\n"
            "    {\"intent\": \"alarm_query\"},\n"
            "    {\"intent\": \"calendar_set\"},\n"
            "    {\"intent\": \"datetime_query\"},\n"
            "    {\"intent\": \"general_greet\"}\n"
            "  ],\n"
            "  \"intent_elimination\": [\n"
            "    {\"intent\": \"alarm_query\", \"reason\": \"prosodic and command framing indicate execution request, not information retrieval\"},\n"
            "    {\"intent\": \"calendar_set\", \"reason\": \"requested parameter is wake-up time, missing event-specific arguments\"},\n"
            "    {\"intent\": \"datetime_query\", \"reason\": \"user is assigning a time target rather than asking current date/time\"},\n"
            "    {\"intent\": \"general_greet\", \"reason\": \"utterance carries task intent and parameters instead of social salutation\"}\n"
            "  ],\n"
            "  \"final_prediction\": {\n"
            "    \"intent\": \"alarm_set\",\n"
            "    \"scenario\": \"alarm\",\n"
            "    \"action\": \"set\"\n"
            "  },\n"
            "  \"slot_grounding\": [\n"
            "    {\"slot_type\": \"time\", \"supported\": true, \"best_span\": \"time\", \"source_hypothesis\": \"interpretation_1\"}\n"
            "  ],\n"
            "  \"final_rationalization\": \"audio cues about time align with alarm set and time slot\"\n"
            "}\n\n"
        )
    return (
        "You are a teacher model whose role is NOT to predict labels, but to rationalize GIVEN reference labels.\n"
        "Use ONLY the information provided below. Output ENGLISH ONLY. Output JSON ONLY.\n\n"
        "==================================================\n"
        "GENERAL RULES\n"
        "==================================================\n"
        "- Do NOT invent intents or slot types outside the provided candidates.\n"
        "- Do NOT hallucinate slot values not supported by the utterance.\n"
        "- Do NOT output the reference labels in the JSON.\n"
        "- topk_intents MUST contain exactly 5 unique intents from intent_candidates.\n"
        "- topk_intents MUST include reference_intent exactly once.\n"
        "- intent_elimination MUST have exactly 4 items, each for a non-reference intent in topk_intents.\n"
        "- Do NOT repeat the same elimination wording pattern; avoid generic \"no ... cue\" only responses.\n"
        "- The maximum TOP-K for intents is fixed to 5.\n\n"
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
        "Step 1: INTERPRETATION UNCERTAINTY ANALYSIS\n"
        "- List stable_cues, unstable_cues, decision_pivots (<=5 each).\n"
        "- Do NOT reference intent/slot labels.\n"
        "Step 2: TOP-5 INTENT CANDIDATES\n"
        "- Use ONLY intent_candidates.\n"
        "- Produce EXACTLY 5 unique intents and include reference_intent exactly once.\n"
        "Step 3: INTENT ELIMINATION\n"
        "- Eliminate ONLY the 4 non-reference intents in topk_intents.\n"
        "- Each reason must be specific and non-repetitive.\n"
        "- Prefer different evidence dimensions across reasons (speech-act mismatch, domain mismatch, argument/slot mismatch, target mismatch).\n"
        "Step 4: FINAL INTENT RESOLUTION\n"
        "- Select one intent from topk_intents and split it into scenario/action by the first '_'.\n"
        "Step 5: SLOT GROUNDING\n"
        "- For EACH reference_slot_type, mark supported and give best_span and source_hypothesis.\n"
        "- source_hypothesis must be interpretation_1 or \"none\".\n"
        "Step 6: FINAL RATIONALIZATION\n"
        "- One concise sentence linking cues to reference labels.\n\n"
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

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate rationale with Qwen2-Audio (nbest or audio mode).")
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
    parser.add_argument("--output_file", type=str, default="Experiment_Rationale/rationale_output.jsonl")
    parser.add_argument("--output_mode", type=str, default="raw", choices=["raw", "full"], help="raw: write compact records with raw outputs; full: write full metadata JSON.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recording_index", type=int, default=0)
    parser.add_argument("--num_hypotheses", type=int, default=5)
    parser.add_argument("--num_candidates", type=int, default=5, help="(Compatibility) ignored: all slot types are always used as candidates.")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ablation_1to5", action="store_true", help="Run n-best ablation with k=1..5 (nbest mode only).")
    parser.add_argument("--nbest_values", type=str, default="", help="Comma-separated n-best sizes, e.g. '1,2,3,4,5' (nbest mode only).")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data-parallel workers (processes).")
    parser.add_argument("--worker_rank", type=int, default=0, help="Rank of this worker in [0, num_workers).")
    parser.add_argument("--append_worker_suffix", action="store_true", help="Append .w{rank}of{num_workers} to output filename.")
    parser.add_argument("--preview", type=int, default=0, help="Print prompt and output for first N samples.")
    parser.add_argument("--limitmode", action="store_true", help="Print pretty JSON results to stdout.")
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--use_fewshot", action="store_true", help="Enable built-in few-shot exemplars in prompts.")
    parser.add_argument("--format_retries", type=int, default=2, help="Retry count when topk_intents format constraints are violated.")
    parser.add_argument("--smoke", action="store_true", help="Skip model inference and emit deterministic placeholder rationales.")
    parser.add_argument("--smoke_limit", type=int, default=3, help="Number of samples processed in --smoke mode.")
    args = parser.parse_args()

    # torchrun compatibility: auto-map worker/device from distributed env vars.
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if env_world_size > 1 and args.num_workers == 1 and args.worker_rank == 0:
        args.num_workers = env_world_size
        args.worker_rank = env_rank
    if args.device == "cuda" and env_world_size > 1:
        args.device = f"cuda:{env_local_rank}"
        if torch.cuda.is_available():
            torch.cuda.set_device(env_local_rank)

    if args.num_workers < 1:
        print(f"[ERROR] num_workers must be >= 1, got {args.num_workers}")
        return
    if args.worker_rank < 0 or args.worker_rank >= args.num_workers:
        print(f"[ERROR] worker_rank must be in [0, {args.num_workers}), got {args.worker_rank}")
        return
    if args.smoke_limit < 1:
        print(f"[ERROR] smoke_limit must be >= 1, got {args.smoke_limit}")
        return
    if args.mode != "nbest" and (args.ablation_1to5 or args.nbest_values):
        print("[ERROR] --ablation_1to5 and --nbest_values are supported only in --mode nbest.")
        return

    try:
        nbest_schedule = parse_nbest_schedule(args.nbest_values, args.num_hypotheses, args.ablation_1to5)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return

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
    if args.num_workers > 1 and args.append_worker_suffix:
        output_path = append_worker_suffix(output_path, args.worker_rank, args.num_workers)

    metadata = load_metadata(metadata_path)
    # clusters/confusing_pairs args are kept for compatibility (currently unused)

    split_paths = [slurp_train_path, slurp_devel_path, slurp_test_path]
    intent_inventory = load_intents_from_slurp_splits(split_paths)
    if intent_inventory:
        print(f"[INFO] Loaded {len(intent_inventory)} intents from train/devel/test splits.")
    else:
        intent_inventory = [normalize_intent_label(intent) for intent in metadata["intents"] if normalize_intent_label(intent)]
        print(f"[WARN] Could not build intents from splits. Falling back to metadata intents ({len(intent_inventory)}).")

    slurp_map = load_slurp_map(split_paths + [slurp_path])

    items = read_jsonl(input_path)
    if args.limit:
        items = items[: args.limit]
    if args.smoke:
        items = items[: args.smoke_limit]
    total_items = len(items)
    items = shard_items_by_worker(items, args.num_workers, args.worker_rank)
    if args.num_workers > 1:
        print(
            f"[INFO] Worker {args.worker_rank}/{args.num_workers} processing "
            f"{len(items)} / {total_items} items."
        )
        if not args.append_worker_suffix:
            print("[WARN] append_worker_suffix is off. Make sure each worker uses a different output_file.")

    if not items:
        print(f"[ERROR] No input items found for this worker: {input_path}")
        return

    processor = None
    model = None
    sr = 16000
    if not args.smoke:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        torch_dtype = torch.bfloat16 if "cuda" in args.device else torch.float32
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(args.device)
        model.eval()
        sr = processor.feature_extractor.sampling_rate
    else:
        print(f"[SMOKE] Enabled. Model inference skipped (smoke_limit={args.smoke_limit}).")

    if args.mode != "nbest":
        nbest_schedule = [args.num_hypotheses]
    print(f"[INFO] n-best schedule: {nbest_schedule}")
    for nbest_k in nbest_schedule:
        current_output_path = resolve_output_path_for_k(output_path, nbest_k, nbest_schedule)
        print(f"[INFO] Start generation for nbest_k={nbest_k} -> {current_output_path}")
        results: List[Dict[str, Any]] = []
        raw_outputs: List[Any] = []

        for idx, item in enumerate(tqdm(items, desc=f"Generating rationale (k={nbest_k})", unit="sample")):
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

            nbest_texts: List[str] = []
            if args.mode == "nbest":
                for h in item.get("asr_hypotheses", [])[:nbest_k]:
                    txt = h.get("text", "") if isinstance(h, dict) else str(h)
                    if txt:
                        nbest_texts.append(txt.strip())
            stable_unstable = summarize_nbest(nbest_texts)

            intent_candidates = build_full_intent_candidates(gold_intent, intent_inventory)
            slot_candidates = build_full_slot_candidates(
                reference_slot_types=gold_slot_types,
                slot_inventory=metadata["slot_types"],
            )

            audio = None
            if args.mode == "audio":
                filename = pick_recording(item.get("recordings", []), args.recording_index)
                audio_path = resolve_audio_path(audio_root, filename) if filename else None
                audio = load_audio(audio_path, target_sr=sr) if audio_path else None
                if audio is None:
                    continue

            if args.mode == "nbest":
                prompt = build_prompt_nbest(
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
                prompt = build_prompt_audio(
                    gold_intent=gold_intent,
                    gold_slot_types=gold_slot_types,
                    intent_candidates=intent_candidates,
                    slot_candidates=slot_candidates,
                    use_fewshot=args.use_fewshot,
                )

            generated = ""
            parsed: Optional[Dict[str, Any]] = None
            topk_valid = False
            validation_error = ""

            if args.smoke:
                parsed = build_smoke_rationale_output(
                    gold_intent=gold_intent,
                    gold_slot_types=gold_slot_types,
                    intent_candidates=intent_candidates,
                    nbest_texts=nbest_texts,
                )
                generated = json.dumps(parsed, ensure_ascii=False)
                topk_valid, validation_error = validate_topk_intents(parsed, intent_candidates, gold_intent)
            else:
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

                base_prompt = prompt
                max_attempts = max(1, args.format_retries + 1)
                for attempt in range(max_attempts):
                    current_prompt = base_prompt if attempt == 0 else build_retry_prompt(base_prompt, generated, validation_error)
                    if args.mode == "nbest":
                        user_content = [{"type": "text", "text": current_prompt}]
                        text_input = processor.apply_chat_template(
                            [{"role": "user", "content": user_content}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        inputs = processor(text=text_input, return_tensors="pt")
                    else:
                        user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": current_prompt}]
                        text_input = processor.apply_chat_template(
                            [{"role": "user", "content": user_content}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        inputs = processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")

                    inputs = {k: v.to(args.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, **gen_kwargs)
                    input_len = inputs["input_ids"].shape[1]
                    generated = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
                    parsed = extract_json(generated)
                    topk_valid, validation_error = validate_topk_intents(parsed, intent_candidates, gold_intent)
                    if topk_valid:
                        break

            parsed = postprocess_rationale_output(parsed, fallback_intent=gold_intent)

            result = {
                "slurp_id": slurp_id,
                "mode": args.mode,
                "nbest_k": nbest_k if args.mode == "nbest" else None,
                "smoke_mode": args.smoke,
                "input_text": input_text,
                "gold_intent": gold_intent,
                "gold_scenario": scenario,
                "gold_action": action,
                "intent_candidates": intent_candidates,
                "slot_candidates": slot_candidates,
                "nbest": nbest_texts if args.mode == "nbest" else [],
                "nbest_summary": stable_unstable if args.mode == "nbest" else {},
                "rationale": parsed,
                "topk_valid": topk_valid,
                "topk_validation_error": "" if topk_valid else validation_error,
                "raw_output": generated,
            }
            if args.save_raw:
                result["rationale_raw"] = generated
            raw_record = {
                "slurp_id": slurp_id,
                "nbest_k": nbest_k if args.mode == "nbest" else None,
                "smoke_mode": args.smoke,
                "input_text": input_text,
                "gold_intent": gold_intent,
                "slot_candidates": slot_candidates,
                "raw_output": generated,
                "topk_valid": topk_valid,
                "topk_validation_error": "" if topk_valid else validation_error,
            }
            raw_outputs.append(raw_record)
            if args.output_mode == "full":
                results.append(result)

            if args.preview and idx < args.preview:
                print("=" * 80)
                print(f"[PREVIEW] {idx+1} / {args.preview} | slurp_id={slurp_id} | mode={args.mode} | nbest_k={nbest_k}")
                print("-" * 80)
                print("PROMPT:")
                print(prompt)
                print("-" * 80)
                print("OUTPUT:")
                print(generated)
            if args.limitmode:
                limit_view = {
                    "slurp_id": slurp_id,
                    "nbest_k": nbest_k if args.mode == "nbest" else None,
                    "input_text": input_text,
                    "gold_intent": gold_intent,
                    "slot_candidates": slot_candidates,
                    "prompt": prompt,
                    "raw_output": generated,
                }
                print(json.dumps(limit_view, ensure_ascii=False, indent=2))
                print("")

        if args.output_mode == "full":
            write_jsonl(current_output_path, results)
        else:
            write_raw_jsonl(current_output_path, raw_outputs)
        print(f"Done. Saved to {current_output_path}")

if __name__ == "__main__":
    main()
