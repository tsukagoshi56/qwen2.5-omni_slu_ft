#!/usr/bin/env python3
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                items.append(obj)
    return items


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_metadata(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        return {"scenarios": [], "actions": [], "intents": [], "slot_types": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "scenarios": data.get("scenarios", []) or [],
        "actions": data.get("actions", []) or [],
        "intents": data.get("intents", []) or [],
        "slot_types": data.get("slot_types", []) or [],
    }


def normalize_intent_label(intent: str) -> str:
    return str(intent or "").strip().replace(":", "_")


def build_db_definitions(metadata: Dict[str, List[str]]) -> str:
    def unique_keep_order(values: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for v in values:
            text = str(v).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    def fmt(label: str, values: List[str]) -> str:
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        return f"{label}: " + (", ".join(cleaned) if cleaned else "(none)")

    intents = [normalize_intent_label(x) for x in unique_keep_order(metadata.get("intents", []) or [])]
    slot_types = unique_keep_order(metadata.get("slot_types", []) or [])

    parts = [
        fmt("Intents", intents),
        fmt("Slot Types", slot_types),
    ]
    return "\n".join(parts)


def _entities_from_slurp(tokens: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    if not tokens or not entities:
        return results
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        ent_type = str(ent.get("type", "")).strip()
        span = ent.get("span") or []
        if not isinstance(span, list) or not span:
            continue
        words = []
        for idx in span:
            if isinstance(idx, int) and 0 <= idx < len(tokens):
                words.append(str(tokens[idx].get("surface", "")).lower())
        filler = " ".join([w for w in words if w]).strip()
        results.append({"type": ent_type, "filter": filler})
    return results


def parse_entities(raw_entities: Any, tokens: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    if isinstance(raw_entities, list) and raw_entities:
        for ent in raw_entities:
            if not isinstance(ent, dict):
                continue
            ent_type = str(ent.get("type", "")).strip()
            filler = ent.get("filler")
            if filler is None:
                filler = ent.get("filter")
            if filler is None:
                filler = ent.get("value")
            if filler is None and tokens is not None:
                span = ent.get("span") or []
                if isinstance(span, list) and span:
                    words = []
                    for idx in span:
                        if isinstance(idx, int) and 0 <= idx < len(tokens):
                            words.append(str(tokens[idx].get("surface", "")).lower())
                    filler = " ".join([w for w in words if w]).strip()
            if filler is None:
                filler = ""
            filler = str(filler)
            results.append({"type": ent_type, "filter": filler})
        return results
    if tokens is not None and isinstance(raw_entities, list):
        return _entities_from_slurp(tokens, raw_entities)
    return results


def split_intent(intent: str) -> Tuple[str, str]:
    intent = normalize_intent_label(intent)
    if not intent:
        return "", ""
    if "_" in intent:
        scenario, action = intent.split("_", 1)
        return scenario, action
    return "", intent


def label_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    final_obj = record.get("final") if isinstance(record.get("final"), dict) else None
    if final_obj:
        scenario = str(final_obj.get("scenario", "")).strip()
        action = str(final_obj.get("action", "")).strip()
        intent = str(final_obj.get("intent", "")).strip()
        if (not scenario or not action) and intent:
            scenario2, action2 = split_intent(intent)
            scenario = scenario or scenario2
            action = action or action2
        tokens = record.get("tokens") if isinstance(record.get("tokens"), list) else []
        entities = parse_entities(final_obj.get("entities", []), tokens=tokens)
        return {"scenario": scenario, "action": action, "entities": entities}

    scenario = str(record.get("scenario", "")).strip()
    action = str(record.get("action", "")).strip()
    intent = str(record.get("intent", "")).strip()
    if (not scenario or not action) and intent:
        scenario2, action2 = split_intent(intent)
        scenario = scenario or scenario2
        action = action or action2
    tokens = record.get("tokens") if isinstance(record.get("tokens"), list) else []
    entities = parse_entities(record.get("entities", []), tokens=tokens)
    return {"scenario": scenario, "action": action, "entities": entities}


def normalize_entity(entity: Dict[str, Any]) -> Tuple[str, str]:
    ent_type = str(entity.get("type", "")).strip().lower()
    filler = entity.get("filler")
    if filler is None:
        filler = entity.get("filter", "")
    filler = str(filler).strip().lower()
    filler = re.sub(r"\s+", " ", filler)
    return ent_type, filler


def entity_f1(pred_entities: List[Dict[str, Any]], gold_entities: List[Dict[str, Any]]) -> float:
    pred = {normalize_entity(e) for e in pred_entities}
    gold = {normalize_entity(e) for e in gold_entities}
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compare_labels(pred: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    def unpack(label: Dict[str, Any]) -> Tuple[str, str]:
        scenario = str(label.get("scenario", "")).strip()
        action = str(label.get("action", "")).strip()
        intent = str(label.get("intent", "")).strip()
        if (not scenario or not action) and intent:
            scenario2, action2 = split_intent(intent)
            scenario = scenario or scenario2
            action = action or action2
        return scenario, action

    pred_scenario, pred_action = unpack(pred if isinstance(pred, dict) else {})
    gold_scenario, gold_action = unpack(gold if isinstance(gold, dict) else {})

    scenario_ok = pred_scenario == gold_scenario
    action_ok = pred_action == gold_action
    intent_ok = (pred_scenario + "_" + pred_action) == (gold_scenario + "_" + gold_action)
    f1 = entity_f1(pred.get("entities", []) or [], gold.get("entities", []) or [])

    return {
        "scenario_ok": scenario_ok,
        "action_ok": action_ok,
        "intent_ok": intent_ok,
        "entity_f1": f1,
    }


def compute_reward(
    pred: Dict[str, Any],
    gold: Dict[str, Any],
    w_scenario: float = 1.0,
    w_action: float = 1.0,
    w_intent: float = 0.5,
    w_entity: float = 1.0,
) -> Tuple[float, Dict[str, Any]]:
    stats = compare_labels(pred, gold)
    reward = 0.0
    reward += w_scenario * (1.0 if stats["scenario_ok"] else 0.0)
    reward += w_action * (1.0 if stats["action_ok"] else 0.0)
    reward += w_intent * (1.0 if stats["intent_ok"] else 0.0)
    reward += w_entity * stats["entity_f1"]
    return reward, stats


def clean_json_text(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = clean_json_text(text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def parse_j_from_output(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    if "J:" in text:
        tail = text.rsplit("J:", 1)[-1].strip()
        return parse_json_block(tail)
    return parse_json_block(text)


def resolve_audio_path(audio_root: str, filename: str) -> Optional[str]:
    if not filename:
        return None
    filename = str(filename).strip()
    if os.path.isabs(filename) and os.path.exists(filename):
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
        if os.path.exists(path):
            return path
    return None
