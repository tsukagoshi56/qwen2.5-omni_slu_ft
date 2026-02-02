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
    use_fewshot: bool = True,
) -> str:
    intent_note = f"intent_candidates ({len(intent_candidates)}):"
    slot_note = f"allowed_slot_types ({len(slot_candidates)}):"
    interpretations_text = format_interpretations(nbest_texts)
    fewshot = ""
    if use_fewshot:
        fewshot = (
            "EXAMPLE INPUT:\n"
            "reference_intent: music:play\n"
            "reference_slot_types: [\"song_name\"]\n"
            "intent_candidates (5): [\"music:play\",\"music:query\",\"alarm:set\",\"weather:query\",\"qa:factoid\"]\n"
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
            "    {\"intent\": \"music:play\"},\n"
            "    {\"intent\": \"music:query\"},\n"
            "    {\"intent\": \"alarm:set\"},\n"
            "    {\"intent\": \"weather:query\"},\n"
            "    {\"intent\": \"qa:factoid\"}\n"
            "  ],\n"
            "  \"intent_elimination\": [\n"
            "    {\"intent\": \"music:query\", \"reason\": \"utterance is command style; no interrogative pattern for retrieval\"},\n"
            "    {\"intent\": \"alarm:set\", \"reason\": \"lexical focus is media playback, not alarm scheduling parameters\"},\n"
            "    {\"intent\": \"weather:query\", \"reason\": \"content refers to a track name, not meteorological information request\"},\n"
            "    {\"intent\": \"qa:factoid\", \"reason\": \"intent asks execution of playback action, not fact-seeking answer\"}\n"
            "  ],\n"
            "  \"final_prediction\": {\n"
            "    \"intent\": \"music:play\",\n"
            "    \"scenario\": \"music\",\n"
            "    \"action\": \"play\"\n"
            "  },\n"
            "  \"slot_grounding\": [\n"
            "    {\"slot_type\": \"song_name\", \"supported\": true, \"best_span\": \"yesterday\", \"source_hypothesis\": \"interpretation_1\"}\n"
            "  ],\n"
            "  \"final_rationalization\": \"stable play,yesterday supports music-play with song_name despite minor uncertainty\"\n"
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
        "- Select one intent from topk_intents and split it into scenario/action by the first ':'.\n"
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
    use_fewshot: bool = True,
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
            "  \"interpretation_uncertainty_analysis\": {\n"
            "    \"stable_cues\": [\"time\"],\n"
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
            "    {\"intent\": \"alarm:query\", \"reason\": \"prosodic and command framing indicate execution request, not information retrieval\"},\n"
            "    {\"intent\": \"calendar:set\", \"reason\": \"requested parameter is wake-up time, missing event-specific arguments\"},\n"
            "    {\"intent\": \"datetime:query\", \"reason\": \"user is assigning a time target rather than asking current date/time\"},\n"
            "    {\"intent\": \"general:greet\", \"reason\": \"utterance carries task intent and parameters instead of social salutation\"}\n"
            "  ],\n"
            "  \"final_prediction\": {\n"
            "    \"intent\": \"alarm:set\",\n"
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
        "- Select one intent from topk_intents and split it into scenario/action by the first ':'.\n"
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
    parser.add_argument("--slurp_file", type=str, default="slurp/dataset/slurp/test.jsonl")
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
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidates for intent/slot types.")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview", type=int, default=0, help="Print prompt and output for first N samples.")
    parser.add_argument("--limitmode", action="store_true", help="Print pretty JSON results to stdout.")
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--format_retries", type=int, default=2, help="Retry count when topk_intents format constraints are violated.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input_file) if not os.path.isabs(args.input_file) else args.input_file
    slurp_path = os.path.join(base_dir, args.slurp_file) if not os.path.isabs(args.slurp_file) else args.slurp_file
    audio_root = os.path.join(base_dir, args.audio_dir) if not os.path.isabs(args.audio_dir) else args.audio_dir
    metadata_path = os.path.join(base_dir, args.metadata_file) if not os.path.isabs(args.metadata_file) else args.metadata_file
    clusters_path = os.path.join(base_dir, args.clusters_file) if not os.path.isabs(args.clusters_file) else args.clusters_file
    confusing_path = os.path.join(base_dir, args.confusing_pairs_file) if not os.path.isabs(args.confusing_pairs_file) else args.confusing_pairs_file
    output_path = os.path.join(base_dir, args.output_file) if not os.path.isabs(args.output_file) else args.output_file

    rng = random.Random(args.seed)

    metadata = load_metadata(metadata_path)
    # clusters/confusing_pairs args are kept for compatibility (currently unused)

    slurp_map = {}
    if os.path.exists(slurp_path):
        slurp_map = {str(d.get("slurp_id")): d for d in read_jsonl(slurp_path)}

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

    for idx, item in enumerate(tqdm(items, desc="Generating rationale", unit="sample")):
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

        intent_candidates = select_candidates_topk(
            gold=gold_intent,
            candidates=metadata["intents"],
            k=args.num_candidates,
            rng=rng,
        )

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

        if args.mode == "nbest":
            prompt = build_prompt_nbest(
                gold_intent=gold_intent,
                gold_slot_types=gold_slot_types,
                intent_candidates=intent_candidates,
                slot_candidates=slot_candidates,
                nbest_texts=nbest_texts,
                stable_tokens=stable_unstable["stable"],
                unstable_tokens=stable_unstable["unstable"],
            )
        else:
            prompt = build_prompt_audio(
                gold_intent=gold_intent,
                gold_slot_types=gold_slot_types,
                intent_candidates=intent_candidates,
                slot_candidates=slot_candidates,
            )

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
        generated = ""
        parsed: Optional[Dict[str, Any]] = None
        topk_valid = False
        validation_error = ""
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
            "input_text": input_text,
            "gold_intent": gold_intent,
            "raw_output": generated,
            "topk_valid": topk_valid,
            "topk_validation_error": "" if topk_valid else validation_error,
        }
        raw_outputs.append(raw_record)
        if args.output_mode == "full":
            results.append(result)

        if args.preview and idx < args.preview:
            print("=" * 80)
            print(f"[PREVIEW] {idx+1} / {args.preview} | slurp_id={slurp_id} | mode={args.mode}")
            print("-" * 80)
            print("PROMPT:")
            print(prompt)
            print("-" * 80)
            print("OUTPUT:")
            print(generated)
        if args.limitmode:
            limit_view = {
                "slurp_id": slurp_id,
                "input_text": input_text,
                "gold_intent": gold_intent,
                "prompt": prompt,
                "raw_output": generated,
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
