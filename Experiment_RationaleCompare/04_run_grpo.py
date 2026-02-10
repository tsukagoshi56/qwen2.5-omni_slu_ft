#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import librosa
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from common import (
    compare_labels,
    compute_reward,
    label_from_record,
    normalize_intent_label,
    parse_j_from_output,
    read_jsonl,
    resolve_audio_path,
    split_intent,
)
DEFAULT_ONLY_GRPO_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"


@dataclass
class GrpoItem:
    slurp_id: Any
    sentence: str
    audio_path: Optional[str]
    gold_label: Dict[str, Any]
    mode: str


class GrpoDataset(Dataset):
    def __init__(self, items: List[GrpoItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> GrpoItem:
        return self.items[idx]


def collate_grpo_items(batch: List[GrpoItem]) -> List[GrpoItem]:
    # Keep dataclass items as-is; default_collate cannot stack custom classes.
    return batch


def build_items(input_file: str, audio_dir: str, include_text: bool) -> List[GrpoItem]:
    records = read_jsonl(input_file)
    items: List[GrpoItem] = []
    for record in records:
        gold_label = label_from_record(record)
        sentence = str(record.get("sentence") or record.get("text") or "").strip()
        recordings = record.get("recordings", []) if isinstance(record.get("recordings"), list) else []

        audio_path = None
        if recordings:
            rec = recordings[0] if isinstance(recordings[0], dict) else None
            filename = rec.get("file") if rec else None
            audio_path = resolve_audio_path(audio_dir, filename) if filename else None
            if audio_path:
                items.append(
                    GrpoItem(
                        slurp_id=record.get("slurp_id"),
                        sentence=sentence,
                        audio_path=audio_path,
                        gold_label=gold_label,
                        mode="audio",
                    )
                )

        if include_text and sentence:
            items.append(
                GrpoItem(
                    slurp_id=record.get("slurp_id"),
                    sentence=sentence,
                    audio_path=None,
                    gold_label=gold_label,
                    mode="text",
                )
            )

    return items


def _record_key(item: GrpoItem, fallback_idx: int) -> Any:
    return item.slurp_id if item.slurp_id is not None else f"__row_{fallback_idx}"


def _label_intent_key(gold_label: Dict[str, Any]) -> str:
    scenario = str(gold_label.get("scenario", "")).strip().lower()
    action = str(gold_label.get("action", "")).strip().lower()
    key = f"{scenario}_{action}".strip("_")
    return key or "unknown_intent"


def _label_slot_types(gold_label: Dict[str, Any]) -> List[str]:
    slots: List[str] = []
    entities = gold_label.get("entities", []) if isinstance(gold_label, dict) else []
    if isinstance(entities, list):
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            slot = str(ent.get("type", "")).strip().lower()
            if slot:
                slots.append(slot)
    if not slots:
        slots.append("__none__")
    return slots


def _label_intent_slot_combos(gold_label: Dict[str, Any]) -> List[Tuple[str, str]]:
    intent = _label_intent_key(gold_label if isinstance(gold_label, dict) else {})
    slot_types = _label_slot_types(gold_label if isinstance(gold_label, dict) else {})
    return sorted({(intent, slot) for slot in slot_types})


def _expand_items_by_intent_slot_combo(
    items: List[GrpoItem],
    per_combo_count: int,
    seed: int,
    shuffle_output: bool = True,
) -> Tuple[List[GrpoItem], Dict[str, int]]:
    if per_combo_count <= 0:
        return items, {
            "num_combos": 0,
            "requested_per_combo": 0,
            "selected_record_instances": 0,
            "selected_unique_records": 0,
            "total_items_after": len(items),
        }

    record_to_items: Dict[Any, List[GrpoItem]] = defaultdict(list)
    record_to_gold: Dict[Any, Dict[str, Any]] = {}
    for idx, item in enumerate(items):
        key = _record_key(item, idx)
        record_to_items[key].append(item)
        if key not in record_to_gold:
            record_to_gold[key] = item.gold_label if isinstance(item.gold_label, dict) else {}

    combo_to_records: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
    for key, gold_label in record_to_gold.items():
        combos = _label_intent_slot_combos(gold_label)
        for combo in combos:
            combo_to_records[combo].append(key)

    rng = random.Random(seed)
    selected_record_counts: Counter = Counter()
    for combo in sorted(combo_to_records.keys()):
        record_keys = combo_to_records[combo]
        if not record_keys:
            continue
        if len(record_keys) >= per_combo_count:
            chosen = rng.sample(record_keys, per_combo_count)
        else:
            # Do not oversample rare buckets; keep all available examples as-is.
            chosen = list(record_keys)
        selected_record_counts.update(chosen)

    expanded_items: List[GrpoItem] = []
    for key, count in selected_record_counts.items():
        record_items = record_to_items.get(key, [])
        if not record_items:
            continue
        for _ in range(count):
            expanded_items.extend(record_items)

    if shuffle_output:
        rng.shuffle(expanded_items)
    stats = {
        "num_combos": len(combo_to_records),
        "requested_per_combo": per_combo_count,
        "selected_record_instances": int(sum(selected_record_counts.values())),
        "selected_unique_records": int(len(selected_record_counts)),
        "total_items_after": len(expanded_items),
    }
    return expanded_items, stats


def _select_balanced_record_keys(
    items: List[GrpoItem],
    target_records: int,
    seed: int,
    per_intent_limit: int = 0,
) -> set:
    groups: Dict[Any, Dict[str, Any]] = {}
    for idx, item in enumerate(items):
        key = _record_key(item, idx)
        if key not in groups:
            intent = _label_intent_key(item.gold_label if isinstance(item.gold_label, dict) else {})
            slot_types = _label_slot_types(item.gold_label if isinstance(item.gold_label, dict) else {})
            groups[key] = {
                "intent": intent,
                "slot_types": list(set(slot_types)),
            }

    intent_freq = Counter(group["intent"] for group in groups.values())
    total_records = len(groups)
    if per_intent_limit > 0:
        max_by_intent = sum(min(per_intent_limit, cnt) for cnt in intent_freq.values())
        if target_records <= 0:
            target_records = max_by_intent
        else:
            target_records = min(target_records, max_by_intent)
    if target_records <= 0 or target_records >= total_records:
        return set(groups.keys())

    slot_freq: Counter = Counter()
    for group in groups.values():
        slot_freq.update(group["slot_types"])

    rng = random.Random(seed)
    remaining = set(groups.keys())
    selected: set = set()
    selected_intent_counts: Counter = Counter()
    uncovered_intents = set(intent_freq.keys())
    uncovered_slots = set(slot_freq.keys())

    # Phase 1: greedy minimum coverage over intents and slot types.
    while (uncovered_intents or uncovered_slots) and remaining and len(selected) < target_records:
        best_score = None
        best_keys: List[Any] = []
        for key in remaining:
            group = groups[key]
            if per_intent_limit > 0 and selected_intent_counts[group["intent"]] >= per_intent_limit:
                continue
            new_intent = 1 if group["intent"] in uncovered_intents else 0
            new_slots = len(set(group["slot_types"]) & uncovered_slots)
            rarity_bonus = (1.0 / max(1, intent_freq[group["intent"]])) + sum(
                1.0 / max(1, slot_freq[s]) for s in group["slot_types"]
            )
            score = (1000.0 * new_intent) + (100.0 * new_slots) + rarity_bonus
            if best_score is None or score > best_score:
                best_score = score
                best_keys = [key]
            elif score == best_score:
                best_keys.append(key)
        if not best_keys:
            break
        chosen = rng.choice(best_keys)
        selected.add(chosen)
        remaining.remove(chosen)
        group = groups[chosen]
        selected_intent_counts[group["intent"]] += 1
        uncovered_intents.discard(group["intent"])
        uncovered_slots.difference_update(group["slot_types"])

    # Phase 2: expand with balanced intent counts and slot coverage smoothing.
    selected_slot_counts: Counter = Counter()
    for key in selected:
        selected_slot_counts.update(groups[key]["slot_types"])

    while len(selected) < target_records and remaining:
        intents_to_keys: Dict[str, List[Any]] = defaultdict(list)
        for key in remaining:
            intent = groups[key]["intent"]
            if per_intent_limit > 0 and selected_intent_counts[intent] >= per_intent_limit:
                continue
            intents_to_keys[intent].append(key)
        if not intents_to_keys:
            break

        min_intent_count = min(selected_intent_counts[intent] for intent in intents_to_keys.keys())
        candidate_intents = [i for i in intents_to_keys.keys() if selected_intent_counts[i] == min_intent_count]
        max_pool = max(len(intents_to_keys[i]) for i in candidate_intents)
        candidate_intents = [i for i in candidate_intents if len(intents_to_keys[i]) == max_pool]
        chosen_intent = rng.choice(candidate_intents)

        best_score = None
        best_keys = []
        for key in intents_to_keys[chosen_intent]:
            group = groups[key]
            slot_balance_gain = sum(1.0 / (1.0 + selected_slot_counts[s]) for s in group["slot_types"])
            slot_rarity_bonus = sum(1.0 / max(1, slot_freq[s]) for s in group["slot_types"])
            score = slot_balance_gain + (0.1 * slot_rarity_bonus)
            if best_score is None or score > best_score:
                best_score = score
                best_keys = [key]
            elif score == best_score:
                best_keys.append(key)
        if not best_keys:
            break
        chosen = rng.choice(best_keys)
        selected.add(chosen)
        remaining.remove(chosen)
        group = groups[chosen]
        selected_intent_counts[group["intent"]] += 1
        selected_slot_counts.update(group["slot_types"])

    return selected


def build_chat_input(processor: AutoProcessor, prompt: str, audio: bool) -> str:
    if audio:
        user_content = [
            {"type": "audio", "audio_url": "placeholder"},
            {"type": "text", "text": prompt},
        ]
    else:
        user_content = [{"type": "text", "text": prompt}]
    return processor.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_grpo_prompt(mode: str, sentence: str, db_definitions: str, no_cot: bool = False) -> str:
    # Compact prompt for GRPO: instruction + input + output format only.
    # DB Definitions and detailed Rules are intentionally omitted.
    _ = db_definitions
    output_schema = (
        '{"Intent": "<scenario>_<action>", "entities": '
        '[{"type": "<entity_type>", "filler": "<entity_value>"}, ...]}'
    )
    if mode == "audio":
        if no_cot:
            return (
                "System: Predict SLU labels from audio.\n\n"
                "[Input Data]\n"
                "- Audio: <AUDIO>\n\n"
                "Output Format:\n"
                f"J: {output_schema}"
            )
        return (
            "System: Predict SLU labels from audio.\n\n"
            "[Input Data]\n"
            "- Audio: <AUDIO>\n\n"
            "Output Format:\n"
            "C: Intent candidates: intent1 | intent2 | intent3; Slot candidates: slot_type1(value1|value2) | slot_type2\n"
            "R: label1!reason1; label2!reason2; ...\n"
            f"J: {output_schema}"
        )
    text = str(sentence or "").strip()
    if no_cot:
        return (
            "System: Predict SLU labels from transcript.\n\n"
            "[Input Data]\n"
            f"- Transcript: {text}\n\n"
            "Output Format:\n"
            f"J: {output_schema}"
        )
    return (
        "System: Predict SLU labels from transcript.\n\n"
        "[Input Data]\n"
        f"- Transcript: {text}\n\n"
        "Output Format:\n"
        "C: Intent candidates: intent1 | intent2 | intent3; Slot candidates: slot_type1(value1|value2) | slot_type2\n"
        "R: label1!reason1; label2!reason2; ...\n"
        f"J: {output_schema}"
    )


def prepare_inputs(
    processor: AutoProcessor,
    prompt_text: str,
    full_text: str,
    audio: Optional[torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if audio is None:
        inputs = processor(text=full_text, return_tensors="pt")
        prompt_inputs = processor(text=prompt_text, return_tensors="pt")
    else:
        sr = processor.feature_extractor.sampling_rate
        inputs = processor(text=full_text, audio=[audio], sampling_rate=sr, return_tensors="pt")
        prompt_inputs = processor(text=prompt_text, audio=[audio], sampling_rate=sr, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
    inputs["_prompt_len"] = prompt_inputs["input_ids"].shape[1]
    return inputs


def model_forward(model: Qwen2AudioForConditionalGeneration, inputs: Dict[str, torch.Tensor]):
    kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        # Disable KV cache for training/logprob passes to reduce memory.
        "use_cache": False,
    }
    if "input_features" in inputs:
        kwargs["input_features"] = inputs["input_features"]
    if "feature_attention_mask" in inputs:
        kwargs["feature_attention_mask"] = inputs["feature_attention_mask"]
    return model(**kwargs)


def compute_logprob_sum(
    model: Qwen2AudioForConditionalGeneration,
    inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    prompt_len = int(inputs.get("_prompt_len", 0))
    outputs = model_forward(model, inputs)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    input_ids = inputs["input_ids"]

    target_ids = input_ids[:, 1:]
    log_probs = log_probs[:, :-1, :]
    start = max(prompt_len - 1, 0)
    if start >= log_probs.shape[1]:
        return torch.tensor(0.0, device=input_ids.device)
    token_logprobs = log_probs[0, start:, :].gather(1, target_ids[0, start:].unsqueeze(-1)).squeeze(-1)
    return token_logprobs.sum()


def generate_samples(
    model: Qwen2AudioForConditionalGeneration,
    processor: AutoProcessor,
    prompt_text: str,
    audio: Optional[torch.Tensor],
    device: torch.device,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> List[str]:
    text_input = build_chat_input(processor, prompt_text, audio is not None)
    if audio is None:
        inputs = processor(text=text_input, return_tensors="pt")
    else:
        sr = processor.feature_extractor.sampling_rate
        inputs = processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs: List[str] = []
    for _ in range(group_size):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        input_len = inputs["input_ids"].shape[1]
        generated_text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        outputs.append(generated_text)
    return outputs


def _shorten(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


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


def _format_model_output(raw_output: str, no_cot: bool) -> Tuple[str, Dict[str, Any], bool]:
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


def _levenshtein_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def _word_distance(truth: str, hypothesis: str) -> float:
    ref = [w for w in str(truth or "").strip().split() if w]
    hyp = [w for w in str(hypothesis or "").strip().split() if w]
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(_levenshtein_distance(ref, hyp)) / float(len(ref))


def _char_distance(truth: str, hypothesis: str) -> float:
    ref = list(str(truth or ""))
    hyp = list(str(hypothesis or ""))
    denom = max(len(ref), len(hyp))
    if denom == 0:
        return 0.0
    return float(_levenshtein_distance(ref, hyp)) / float(denom)


def _entity_label_filler(entity: Dict[str, Any]) -> Tuple[str, str]:
    # Align with the label normalization used by compare_labels/entity_f1.
    # This avoids artificial zero SLU-F1 caused by case/whitespace/key-format drift.
    label = str(entity.get("type", "unknown")).strip().lower() or "unknown"
    filler = entity.get("filler")
    if filler is None:
        filler = entity.get("filter", "")
    filler = " ".join(str(filler).strip().lower().split())
    return label, filler


def _span_distance_counts(
    gold_entities: List[Dict[str, Any]],
    pred_entities: List[Dict[str, Any]],
    distance_kind: str,
) -> Tuple[float, float, float]:
    if distance_kind == "word":
        dist_fn = _word_distance
    else:
        dist_fn = _char_distance

    gold_labels: List[str] = []
    gold_fillers: List[str] = []
    for ent in gold_entities:
        lbl, fil = _entity_label_filler(ent if isinstance(ent, dict) else {})
        gold_labels.append(lbl)
        gold_fillers.append(fil)

    pred_labels: List[str] = []
    pred_fillers: List[str] = []
    for ent in pred_entities:
        lbl, fil = _entity_label_filler(ent if isinstance(ent, dict) else {})
        pred_labels.append(lbl)
        pred_fillers.append(fil)

    tp = 0.0
    fp = 0.0
    fn = 0.0

    for pred_label, pred_filler in zip(pred_labels, pred_fillers):
        candidate_idxs = [i for i, lbl in enumerate(gold_labels) if lbl == pred_label]
        if candidate_idxs:
            best_idx = candidate_idxs[0]
            best_dist = dist_fn(gold_fillers[best_idx], pred_filler)
            for idx in candidate_idxs[1:]:
                d = dist_fn(gold_fillers[idx], pred_filler)
                if d < best_dist:
                    best_idx = idx
                    best_dist = d
            tp += 1.0
            fp += float(best_dist)
            fn += float(best_dist)
            gold_labels.pop(best_idx)
            gold_fillers.pop(best_idx)
        else:
            fp += 1.0

    fn += float(len(gold_labels))
    return tp, fp, fn


def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
    precision = 0.0 if (tp == 0.0 and fp == 0.0) else tp / (tp + fp)
    recall = 0.0 if (tp == 0.0 and fn == 0.0) else tp / (tp + fn)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2.0 * ((precision * recall) / (precision + recall))


def _debug_write_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_debug_section(title: str, content: str) -> None:
    print(f"[DEBUG-SECTION-BEGIN] {title}")
    print(content if content is not None else "")
    print(f"[DEBUG-SECTION-END] {title}")


def _debug_print_dataset(items: List[GrpoItem], preview_items: int) -> None:
    audio_count = sum(1 for x in items if x.mode == "audio")
    text_count = sum(1 for x in items if x.mode == "text")
    print(f"[DEBUG] dataset_size={len(items)} audio_items={audio_count} text_items={text_count}")
    for idx, item in enumerate(items[:preview_items]):
        print(
            "[DEBUG] item_preview "
            f"idx={idx} slurp_id={item.slurp_id} mode={item.mode} "
            f"audio_path={item.audio_path} "
            f"sentence_len={len(item.sentence)} "
            f"gold={json.dumps(item.gold_label, ensure_ascii=False)}"
        )


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank_world() -> Tuple[int, int]:
    if _is_distributed():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if not isinstance(exc, RuntimeError):
        return False
    return "out of memory" in msg or "cuda out of memory" in msg


def _recover_from_oom() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_model(
    model,
    processor: AutoProcessor,
    items: List[GrpoItem],
    db_definitions: str,
    no_cot: bool,
    device: torch.device,
    max_new_tokens: int,
    rank: int,
    world_size: int,
    debug: bool = False,
    debug_max_chars: int = 1200,
    collect_predictions: bool = False,
    preview_count: int = 0,
    preview_prefix: str = "[EVAL-PREVIEW]",
    show_progress: bool = False,
    progress_desc: str = "eval",
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    eval_model = _unwrap_model(model)
    was_training = eval_model.training
    eval_model.eval()

    local_items = items[rank::world_size] if world_size > 1 else items
    local_count = 0.0
    reward_sum = 0.0
    scenario_sum = 0.0
    action_sum = 0.0
    intent_sum = 0.0
    entity_f1_sum = 0.0
    slu_tp_sum = 0.0
    slu_fp_sum = 0.0
    slu_fn_sum = 0.0
    local_prediction_rows: List[Dict[str, Any]] = []
    local_oom_skips = 0.0

    iterable = local_items
    if show_progress and rank == 0 and tqdm is not None:
        iterable = tqdm(local_items, desc=progress_desc, unit="sample")

    with torch.no_grad():
        for idx, item in enumerate(iterable):
            try:
                audio = None
                if item.audio_path:
                    try:
                        sr = processor.feature_extractor.sampling_rate
                        audio, _ = librosa.load(item.audio_path, sr=sr)
                    except Exception:
                        continue

                prompt = build_grpo_prompt(
                    item.mode,
                    item.sentence,
                    db_definitions=db_definitions,
                    no_cot=no_cot,
                )
                text_input = build_chat_input(processor, prompt, audio is not None)
                if audio is None:
                    inputs = processor(text=text_input, return_tensors="pt")
                else:
                    sr = processor.feature_extractor.sampling_rate
                    inputs = processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                output_ids = eval_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
                input_len = inputs["input_ids"].shape[1]
                generated_text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
                formatted_text, pred_label, format_ok = _format_model_output(generated_text, no_cot=no_cot)
                reward, _ = compute_reward(pred_label, item.gold_label)
                stats = compare_labels(pred_label, item.gold_label)
            except RuntimeError as exc:
                if _is_oom_error(exc):
                    local_oom_skips += 1.0
                    _recover_from_oom()
                    if rank == 0:
                        print(
                            f"[WARN][EVAL-OOM] skipped sample slurp_id={item.slurp_id} mode={item.mode}: {exc}"
                        )
                    continue
                raise

            local_count += 1.0
            reward_sum += float(reward)
            scenario_sum += 1.0 if stats["scenario_ok"] else 0.0
            action_sum += 1.0 if stats["action_ok"] else 0.0
            intent_sum += 1.0 if stats["intent_ok"] else 0.0
            entity_f1_sum += float(stats["entity_f1"])
            w_tp, w_fp, w_fn = _span_distance_counts(
                item.gold_label.get("entities", []) or [],
                pred_label.get("entities", []) or [],
                distance_kind="word",
            )
            c_tp, c_fp, c_fn = _span_distance_counts(
                item.gold_label.get("entities", []) or [],
                pred_label.get("entities", []) or [],
                distance_kind="char",
            )
            slu_tp_sum += (w_tp + c_tp)
            slu_fp_sum += (w_fp + c_fp)
            slu_fn_sum += (w_fn + c_fn)

            if collect_predictions:
                local_prediction_rows.append(
                    {
                        "scenario": pred_label["scenario"],
                        "action": pred_label["action"],
                        "entities": pred_label["entities"],
                        "pred_label": pred_label,
                        "file": os.path.basename(item.audio_path) if item.audio_path else None,
                        "slurp_id": item.slurp_id,
                        "id": item.slurp_id,
                        "wer": 0.0,
                        "transcript": item.sentence,
                        "candidates": [],
                        "rationale_text": formatted_text,
                        "format_ok": bool(format_ok),
                        "raw_output": generated_text,
                        "target": "",
                        "target_label": item.gold_label,
                        "type": "audio" if item.audio_path else "text",
                    }
                )

            if debug and rank == 0 and idx < 2:
                print(f"[EVAL-DEBUG] mode={item.mode} slurp_id={item.slurp_id}")
                _print_debug_section("eval.input_prompt", prompt)
                _print_debug_section("eval.output_raw", generated_text)
                _print_debug_section("eval.output_formatted", formatted_text)
                print(f"[EVAL-DEBUG] gold={json.dumps(item.gold_label, ensure_ascii=False)}")
                print(
                    f"[EVAL-DEBUG] pred={json.dumps(pred_label, ensure_ascii=False)} "
                    f"format_ok={bool(format_ok)} reward={reward:.4f}"
                )
            if rank == 0 and idx < max(0, preview_count):
                print(
                    f"{preview_prefix} idx={idx} mode={item.mode} slurp_id={item.slurp_id} "
                    f"reward={reward:.4f}"
                )
                print(f"{preview_prefix} input_prompt:")
                print(_shorten(prompt, debug_max_chars))
                print(f"{preview_prefix} output_raw:")
                print(_shorten(generated_text, debug_max_chars))
                print(f"{preview_prefix} output_formatted:")
                print(_shorten(formatted_text, debug_max_chars))
                print(f"{preview_prefix} gold={json.dumps(item.gold_label, ensure_ascii=False)}")
                print(f"{preview_prefix} pred={json.dumps(pred_label, ensure_ascii=False)}")

    metrics_tensor = torch.tensor(
        [
            local_count,
            reward_sum,
            scenario_sum,
            action_sum,
            intent_sum,
            entity_f1_sum,
            slu_tp_sum,
            slu_fp_sum,
            slu_fn_sum,
        ],
        device=device,
        dtype=torch.float32,
    )
    if world_size > 1 and _is_distributed():
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    oom_tensor = torch.tensor([local_oom_skips], device=device, dtype=torch.float32)
    if world_size > 1 and _is_distributed():
        dist.all_reduce(oom_tensor, op=dist.ReduceOp.SUM)
    total_oom_skips = int(oom_tensor[0].item())
    if rank == 0 and total_oom_skips > 0:
        print(f"[WARN][EVAL-OOM] skipped {total_oom_skips} samples due to OOM")

    total = float(metrics_tensor[0].item())
    if total <= 0:
        result = {
            "num_samples": 0.0,
            "reward_mean": 0.0,
            "scenario_acc": 0.0,
            "action_acc": 0.0,
            "intent_acc": 0.0,
            "entity_f1_mean": 0.0,
            "slu_f1": 0.0,
        }
    else:
        slu_f1_value = _f1_from_counts(
            float(metrics_tensor[6].item()),
            float(metrics_tensor[7].item()),
            float(metrics_tensor[8].item()),
        )
        result = {
            "num_samples": total,
            "reward_mean": float(metrics_tensor[1].item() / total),
            "scenario_acc": float(metrics_tensor[2].item() / total),
            "action_acc": float(metrics_tensor[3].item() / total),
            "intent_acc": float(metrics_tensor[4].item() / total),
            "entity_f1_mean": float(metrics_tensor[5].item() / total),
            "slu_f1": float(slu_f1_value),
        }

    prediction_rows: List[Dict[str, Any]] = []
    if collect_predictions:
        if world_size > 1 and _is_distributed():
            gathered: List[List[Dict[str, Any]]] = [None] * world_size
            dist.all_gather_object(gathered, local_prediction_rows)
            if rank == 0:
                for chunk in gathered:
                    if chunk:
                        prediction_rows.extend(chunk)
        else:
            prediction_rows = local_prediction_rows

    if was_training:
        eval_model.train()
    return result, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO fine-tuning after SF-CoT.")
    parser.add_argument(
        "--train_file",
        type=str,
        default="slurp/dataset/slurp/training.json",
        help="Training data path (default: slurp/dataset/slurp/training.json).",
    )
    parser.add_argument("--eval_file", type=str, default="", help="Optional eval jsonl path.")
    parser.add_argument(
        "--test_file",
        type=str,
        default="slurp/dataset/slurp/test.jsonl",
        help="Test jsonl path used for final audio-only evaluation and prediction.jsonl export.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="Experiment_3/slurp_metadata.json",
        help="Metadata JSON used to build DB Definitions for prompts.",
    )
    parser.add_argument("--audio_dir", type=str, default="slurp/slurp_real")
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--ref_model_name_or_path", type=str, default="")
    parser.add_argument(
        "--only_grpo",
        "--only-grpo",
        dest="only_grpo",
        action="store_true",
        help=(
            "Run GRPO directly from the original base model "
            f"({DEFAULT_ONLY_GRPO_MODEL}) for both policy and ref "
            "(no SFT prerequisite in this script)."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="outputs/grpo")
    parser.add_argument(
        "--no_cot",
        "--no-cot",
        dest="no_cot",
        action="store_true",
        help="Use direct J-only prompting (no C/R lines) for controlled GRPO comparison.",
    )
    parser.add_argument(
        "--include_text",
        dest="include_text",
        action="store_true",
        help="Include text-mode samples in training (default: enabled).",
    )
    parser.add_argument(
        "--no_include_text",
        "--no-include-text",
        dest="include_text",
        action="store_false",
        help="Disable text-mode samples and train with audio-mode samples only.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--shuffle_train",
        dest="shuffle_train",
        action="store_true",
        help="Shuffle training order each epoch (default: disabled; keep top-to-bottom order).",
    )
    parser.add_argument(
        "--no_shuffle_train",
        "--no-shuffle-train",
        dest="shuffle_train",
        action="store_false",
        help="Disable training shuffle and consume samples in file order.",
    )
    parser.add_argument(
        "--balanced_train_records",
        "--balanced-train-records",
        dest="balanced_train_records",
        type=int,
        default=0,
        help=(
            "If >0, pick a balanced training subset by slurp_id count "
            "(intent/slot coverage aware) before training. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--balanced_per_intent",
        "--balanced-per-intent",
        dest="balanced_per_intent",
        type=int,
        default=0,
        help=(
            "If >0, select roughly N records per intent (slot-aware), "
            "instead of controlling by global record count."
        ),
    )
    parser.add_argument(
        "--balanced_per_intent_slot_combo",
        "--balanced-per-intent-slot-combo",
        "--per_combo",
        "--per-combo",
        dest="balanced_per_intent_slot_combo",
        type=int,
        default=0,
        help=(
            "If >0, treat each (intent, slot_type) pair as one bucket and use up to N records "
            "per bucket (rare buckets keep all available records, no oversampling)."
        ),
    )
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Maximum number of GRPO training steps. 0 means no step cap (use epochs only).",
    )
    parser.add_argument("--eval_every", type=int, default=100, help="Run eval every N global steps (0 disables).")
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=None,
        help="Cap eval items for faster validation (default: use --smoke_eval_samples even outside --smoke).",
    )
    parser.add_argument("--test_every", type=int, default=0, help="Run test every N global steps (0 disables).")
    parser.add_argument(
        "--test_max_samples",
        type=int,
        default=None,
        help="Cap test items for faster validation (default: full test set; in --smoke use --smoke_test_samples).",
    )
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--kl_beta", type=float, default=0.01)
    parser.add_argument("--advantage_normalize", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reward_w_scenario", type=float, default=1.0)
    parser.add_argument("--reward_w_action", type=float, default=1.0)
    parser.add_argument("--reward_w_intent", type=float, default=0.5)
    parser.add_argument("--reward_w_entity", type=float, default=1.0)
    parser.add_argument(
        "--early_stopping",
        dest="early_stopping",
        action="store_true",
        help="Enable early stopping based on eval metric.",
    )
    parser.add_argument(
        "--no_early_stopping",
        "--no-early-stopping",
        dest="early_stopping",
        action="store_false",
        help="Disable early stopping.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Stop after N consecutive evals without improvement.",
    )
    parser.add_argument(
        "--early_stopping_min_epochs",
        type=int,
        default=1,
        help="Run at least this many epochs before early stopping can trigger.",
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        choices=["reward_mean", "intent_acc", "scenario_acc", "action_acc", "entity_f1_mean", "slu_f1"],
        default="intent_acc",
        help="Eval metric used for early stopping.",
    )
    parser.add_argument("--debug", action="store_true", help="Print rich debug information.")
    parser.add_argument("--debug_preview_items", type=int, default=5, help="Dataset preview rows in debug.")
    parser.add_argument("--debug_preview_steps", type=int, default=3, help="Training steps to trace in debug.")
    parser.add_argument("--debug_preview_samples", type=int, default=3, help="Generated samples to show per item.")
    parser.add_argument("--debug_max_chars", type=int, default=1200, help="Max chars per debug text field.")
    parser.add_argument(
        "--debug_output_file",
        type=str,
        default="",
        help="Optional JSONL path for debug traces (default: <output_dir>/grpo_debug_trace.jsonl).",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a tiny sanity-check setting.")
    parser.add_argument("--smoke_train_samples", type=int, default=200)
    parser.add_argument("--smoke_eval_samples", type=int, default=32)
    parser.add_argument("--smoke_test_samples", type=int, default=32)
    parser.add_argument(
        "--skip_split_check",
        action="store_true",
        help="Skip startup existence check for slurp train/devel/test split files.",
    )
    parser.add_argument(
        "--allow_empty_db",
        action="store_true",
        help="Allow empty/missing DB Definitions (default: disabled; DB is required).",
    )
    parser.set_defaults(include_text=True, no_cot=False, early_stopping=False, shuffle_train=False)

    # Accept both --snake_case and --kebab-case flags.
    normalized_argv: List[str] = []
    for token in sys.argv[1:]:
        if token.startswith("--"):
            if "=" in token:
                key, value = token.split("=", 1)
                token = f"--{key[2:].replace('-', '_')}={value}"
            else:
                token = f"--{token[2:].replace('-', '_')}"
        normalized_argv.append(token)

    args = parser.parse_args(normalized_argv)
    if args.early_stopping_patience < 1:
        raise ValueError("--early_stopping_patience must be >= 1")
    if args.early_stopping_min_epochs < 0:
        raise ValueError("--early_stopping_min_epochs must be >= 0")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
    if args.balanced_train_records < 0:
        raise ValueError("--balanced_train_records must be >= 0")
    if args.balanced_per_intent < 0:
        raise ValueError("--balanced_per_intent must be >= 0")
    if args.balanced_per_intent_slot_combo < 0:
        raise ValueError("--balanced_per_intent_slot_combo must be >= 0")
    if args.max_steps < 0:
        raise ValueError("--max_steps must be >= 0")
    if args.smoke:
        args.num_train_epochs = 1
        args.group_size = min(args.group_size, 2)
        args.max_new_tokens = min(args.max_new_tokens, 96)
        args.log_every = 1
        if args.eval_file and args.eval_every <= 0:
            args.eval_every = 10
        if args.test_file and args.test_every <= 0:
            args.test_every = 10

    forced_only_grpo_base = False
    if args.only_grpo:
        requested_policy = str(args.model_name_or_path).strip()
        requested_ref = str(args.ref_model_name_or_path).strip()
        forced_only_grpo_base = (
            (requested_policy not in ("", DEFAULT_ONLY_GRPO_MODEL))
            or (requested_ref not in ("", DEFAULT_ONLY_GRPO_MODEL))
        )
        # In only-GRPO mode, always start both policy/ref from the original base model.
        args.model_name_or_path = DEFAULT_ONLY_GRPO_MODEL
        args.ref_model_name_or_path = DEFAULT_ONLY_GRPO_MODEL
    if not str(args.model_name_or_path).strip():
        raise ValueError(
            "--model_name_or_path is required unless --only-grpo is set "
            f"(then it defaults to {DEFAULT_ONLY_GRPO_MODEL})."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank != -1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        rank, world_size = _get_rank_world()
    else:
        rank, world_size = 0, 1

    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, args.train_file) if not os.path.isabs(args.train_file) else args.train_file
    metadata_path = (
        os.path.join(base_dir, args.metadata_file)
        if (args.metadata_file and not os.path.isabs(args.metadata_file))
        else args.metadata_file
    )
    eval_path = (
        os.path.join(base_dir, args.eval_file)
        if (args.eval_file and not os.path.isabs(args.eval_file))
        else args.eval_file
    )
    test_path = (
        os.path.join(base_dir, args.test_file)
        if (args.test_file and not os.path.isabs(args.test_file))
        else args.test_file
    )
    audio_dir = os.path.join(base_dir, args.audio_dir) if not os.path.isabs(args.audio_dir) else args.audio_dir
    output_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train_file not found: {train_path}")

    if not args.skip_split_check:
        default_splits = {
            "train": os.path.join(base_dir, "slurp/dataset/slurp/train.jsonl"),
            "devel": os.path.join(base_dir, "slurp/dataset/slurp/devel.jsonl"),
            "test": os.path.join(base_dir, "slurp/dataset/slurp/test.jsonl"),
        }
        missing = [name for name, path in default_splits.items() if not os.path.exists(path)]
        if missing:
            detail = ", ".join(f"{name}={default_splits[name]}" for name in missing)
            raise FileNotFoundError(
                "Required SLURP split files are missing at startup: "
                f"{detail} (set --skip_split_check to bypass)."
            )
        if rank == 0:
            print(
                "[CHECK] dataset splits found: "
                + ", ".join(f"{name}={path}" for name, path in default_splits.items())
            )

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    if distributed:
        dist.barrier()

    if args.debug and rank == 0:
        if args.debug_output_file:
            debug_output_path = (
                args.debug_output_file
                if os.path.isabs(args.debug_output_file)
                else os.path.join(base_dir, args.debug_output_file)
            )
        else:
            debug_output_path = os.path.join(output_dir, "grpo_debug_trace.jsonl")
        os.makedirs(os.path.dirname(debug_output_path), exist_ok=True)
        if os.path.exists(debug_output_path):
            os.remove(debug_output_path)
    else:
        debug_output_path = ""

    # GRPO prompt in this script is intentionally compact and does not inject DB Definitions.
    db_definitions = ""
    metadata_exists = bool(metadata_path) and os.path.exists(metadata_path)
    if rank == 0 and (not metadata_exists) and args.debug:
        print(f"[DEBUG] metadata_file not found: {metadata_path} (unused in compact prompt mode)")

    items = build_items(train_path, audio_dir, include_text=args.include_text)
    if args.balanced_per_intent_slot_combo > 0:
        total_items_before = len(items)
        unique_records_before = len({_record_key(x, i) for i, x in enumerate(items)})
        items, combo_stats = _expand_items_by_intent_slot_combo(
            items=items,
            per_combo_count=args.balanced_per_intent_slot_combo,
            seed=args.seed,
            shuffle_output=args.shuffle_train,
        )
        if rank == 0:
            print(
                "[GRPO] intent-slot combo balancing enabled: "
                f"per_combo={args.balanced_per_intent_slot_combo} "
                f"combos={combo_stats['num_combos']} "
                f"record_instances={combo_stats['selected_record_instances']} "
                f"unique_records={combo_stats['selected_unique_records']}/{unique_records_before} "
                f"items={combo_stats['total_items_after']}/{total_items_before}"
            )
    elif args.balanced_train_records > 0 or args.balanced_per_intent > 0:
        total_items_before = len(items)
        unique_records_before = len({_record_key(x, i) for i, x in enumerate(items)})
        # When per-intent mode is requested, let per_intent_limit determine effective size.
        effective_target_records = 0 if args.balanced_per_intent > 0 else args.balanced_train_records
        selected_keys = _select_balanced_record_keys(
            items=items,
            target_records=effective_target_records,
            seed=args.seed,
            per_intent_limit=args.balanced_per_intent,
        )
        items = [x for i, x in enumerate(items) if _record_key(x, i) in selected_keys]
        if rank == 0:
            unique_records_after = len({_record_key(x, i) for i, x in enumerate(items)})
            print(
                "[GRPO] balanced subset enabled: "
                f"requested_total={args.balanced_train_records} "
                f"per_intent={args.balanced_per_intent} "
                f"selected_records={unique_records_after}/{unique_records_before} "
                f"items={len(items)}/{total_items_before}"
            )
    if args.smoke:
        items = items[: max(1, args.smoke_train_samples)]

    eval_items: List[GrpoItem] = []
    if args.eval_file:
        if not os.path.exists(eval_path):
            if rank == 0:
                print(f"[WARN] eval_file not found: {eval_path} (skip eval)")
            args.eval_file = ""
        else:
            # Eval should use audio-only inputs for consistent ASR+SLU validation.
            eval_items = build_items(eval_path, audio_dir, include_text=False)
            eval_cap = args.eval_max_samples
            if eval_cap is None:
                eval_cap = args.smoke_eval_samples
            if eval_cap is not None:
                eval_items = eval_items[: max(0, eval_cap)]

    test_items: List[GrpoItem] = []
    if args.test_file:
        if not os.path.exists(test_path):
            if rank == 0:
                print(f"[WARN] test_file not found: {test_path} (skip final test/prediction export)")
            args.test_file = ""
        else:
            # Test should use audio-only inputs for final report comparability.
            test_items = build_items(test_path, audio_dir, include_text=False)
            test_cap = args.test_max_samples
            if test_cap is None and args.smoke:
                test_cap = args.smoke_test_samples
            if test_cap is not None:
                test_items = test_items[: max(0, test_cap)]

    if args.debug and rank == 0:
        print("[DEBUG] ===== Run Config =====")
        print(f"[DEBUG] distributed={distributed} rank={rank} world_size={world_size} local_rank={local_rank}")
        print(
            f"[DEBUG] train_path={train_path} metadata_path={metadata_path} audio_dir={audio_dir} output_dir={output_dir}"
        )
        print(
            f"[DEBUG] model={args.model_name_or_path} ref_model={args.ref_model_name_or_path or args.model_name_or_path}"
        )
        print(
            "[DEBUG] hyperparams "
            f"batch_size={args.batch_size} group_size={args.group_size} max_new_tokens={args.max_new_tokens} "
            f"temperature={args.temperature} top_p={args.top_p} do_sample={args.do_sample} "
            f"lr={args.learning_rate} kl_beta={args.kl_beta} grad_accum_steps={args.grad_accum_steps} "
            f"include_text={args.include_text} no_cot={args.no_cot} smoke={args.smoke} "
            f"shuffle_train={args.shuffle_train} "
            f"balanced_train_records={args.balanced_train_records} "
            f"balanced_per_intent={args.balanced_per_intent} "
            f"balanced_per_intent_slot_combo={args.balanced_per_intent_slot_combo} "
            f"early_stopping={args.early_stopping} es_metric={args.early_stopping_metric} "
            f"es_patience={args.early_stopping_patience} es_min_epochs={args.early_stopping_min_epochs}"
        )
        if args.eval_file:
            print(
                f"[DEBUG] eval_file={eval_path} eval_items={len(eval_items)} "
                f"eval_every={args.eval_every} eval_max_samples={args.eval_max_samples}"
            )
        if args.test_file:
            print(
                f"[DEBUG] test_file={test_path} test_items={len(test_items)} "
                f"test_every={args.test_every} test_max_samples={args.test_max_samples}"
            )
        print(f"[DEBUG] debug_output_file={debug_output_path}")
        print("[DEBUG] prompt_mode=compact (no DB Definitions, no detailed Rules)")
        _debug_print_dataset(items, preview_items=max(0, args.debug_preview_items))

    dataset = GrpoDataset(items)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=args.shuffle_train)
        if distributed
        else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None and args.shuffle_train),
        sampler=sampler,
        collate_fn=collate_grpo_items,
    )
    if rank == 0:
        total_batches = len(dataloader) * args.num_train_epochs
        effective_total_steps = min(total_batches, args.max_steps) if args.max_steps > 0 else total_batches
        optimizer_steps = math.ceil(effective_total_steps / max(1, args.grad_accum_steps))
        mode_label = "ONLY_GRPO" if args.only_grpo else "SFT_INIT+GRPO"
        print(
            f"[GRPO] mode={mode_label} total_batches={total_batches} "
            f"effective_steps={effective_total_steps} max_steps={args.max_steps} "
            f"grad_accum_steps={args.grad_accum_steps} optimizer_steps~={optimizer_steps}"
        )
        print(f"[GRPO] prompt_style={'J_ONLY' if args.no_cot else 'C_R_J'}")
        if args.only_grpo:
            print(
                f"[GRPO] only_grpo=True -> force base model for both policy/ref: "
                f"{DEFAULT_ONLY_GRPO_MODEL}"
            )
            if forced_only_grpo_base:
                print(
                    "[GRPO] note: user-specified --model_name_or_path / "
                    "--ref_model_name_or_path were ignored in only_grpo mode."
                )
        if args.eval_file:
            print(
                f"[GRPO] eval enabled: eval_file={eval_path} "
                f"eval_items={len(eval_items)} eval_every={args.eval_every}"
            )
        if args.test_file:
            print(
                f"[GRPO] test enabled: test_file={test_path} "
                f"test_items={len(test_items)} test_every={args.test_every}"
            )

    if distributed:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            # Audio/Text batches can activate different submodules per step.
            # Enable unused-parameter detection to avoid DDP reduction errors.
            find_unused_parameters=True,
        )
    model.train()

    ref_path = args.ref_model_name_or_path or args.model_name_or_path
    if rank == 0 and args.only_grpo:
        print(
            f"[GRPO] only_grpo=True: policy_init={args.model_name_or_path} ref_model={ref_path}"
        )
    ref_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        ref_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    best_eval_metric = float("-inf")
    no_improve_count = 0
    early_stopped = False
    reached_max_steps = False
    for epoch in range(args.num_train_epochs):
        if args.max_steps > 0 and global_step >= args.max_steps:
            reached_max_steps = True
            break
        if sampler is not None and args.shuffle_train:
            sampler.set_epoch(epoch)
        accum_steps = 0
        for batch_idx, batch in enumerate(dataloader):
            if args.max_steps > 0 and global_step >= args.max_steps:
                reached_max_steps = True
                break
            batch_loss = torch.tensor(0.0, device=device)
            sample_count = 0
            batch_had_oom = False
            reward_values: List[float] = []
            advantage_values: List[float] = []
            kl_values: List[float] = []
            logprob_values: List[float] = []
            ref_logprob_values: List[float] = []
            sample_loss_values: List[float] = []

            for item in batch:
                try:
                    audio = None
                    if item.audio_path:
                        sr = processor.feature_extractor.sampling_rate
                        audio, _ = librosa.load(item.audio_path, sr=sr)

                    prompt = build_grpo_prompt(
                        item.mode,
                        item.sentence,
                        db_definitions=db_definitions,
                        no_cot=args.no_cot,
                    )

                    debug_step = args.debug and rank == 0 and (global_step < args.debug_preview_steps)
                    if debug_step:
                        chat_prompt = build_chat_input(processor, prompt, audio is not None)
                        print(
                            f"[DEBUG][step={global_step}] item slurp_id={item.slurp_id} mode={item.mode} "
                            f"audio_used={audio is not None}"
                        )
                        _print_debug_section("train.prompt_raw", prompt)
                        _print_debug_section("train.chat_prompt_raw", chat_prompt)

                    samples = generate_samples(
                        model=_unwrap_model(model),
                        processor=processor,
                        prompt_text=prompt,
                        audio=audio,
                        device=device,
                        group_size=args.group_size,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=args.do_sample,
                    )

                    rewards: List[float] = []
                    pred_labels: List[Dict[str, Any]] = []
                    formatted_samples: List[str] = []
                    format_ok_values: List[bool] = []
                    for text in samples:
                        formatted_text, pred_label, format_ok = _format_model_output(text, no_cot=args.no_cot)
                        reward, _ = compute_reward(
                            pred_label,
                            item.gold_label,
                            w_scenario=args.reward_w_scenario,
                            w_action=args.reward_w_action,
                            w_intent=args.reward_w_intent,
                            w_entity=args.reward_w_entity,
                        )
                        rewards.append(reward)
                        reward_values.append(float(reward))
                        pred_labels.append(pred_label)
                        formatted_samples.append(formatted_text)
                        format_ok_values.append(bool(format_ok))

                    if debug_step:
                        preview_n = min(args.debug_preview_samples, len(samples))
                        for i in range(preview_n):
                            print(
                                f"[DEBUG][step={global_step}] sample#{i} reward={rewards[i]:.4f} "
                                f"format_ok={format_ok_values[i]} "
                                f"pred={json.dumps(pred_labels[i], ensure_ascii=False)}"
                            )
                            _print_debug_section(f"train.sample_raw#{i}", samples[i])
                            _print_debug_section(f"train.sample_formatted#{i}", formatted_samples[i])

                    mean_reward = sum(rewards) / max(len(rewards), 1)
                    if args.advantage_normalize:
                        variance = sum((r - mean_reward) ** 2 for r in rewards) / max(len(rewards), 1)
                        std = math.sqrt(variance) if variance > 0 else 1.0
                    else:
                        std = 1.0

                    for sample_idx, (sample_text, reward) in enumerate(zip(samples, rewards)):
                        advantage = (reward - mean_reward) / (std + 1e-6)
                        advantage_values.append(float(advantage))
                        prompt_text = build_chat_input(processor, prompt, audio is not None)
                        full_text = prompt_text + sample_text

                        inputs = prepare_inputs(
                            processor=processor,
                            prompt_text=prompt_text,
                            full_text=full_text,
                            audio=audio,
                            device=device,
                        )
                        logprob = compute_logprob_sum(model, inputs)

                        with torch.no_grad():
                            ref_inputs = prepare_inputs(
                                processor=processor,
                                prompt_text=prompt_text,
                                full_text=full_text,
                                audio=audio,
                                device=device,
                            )
                            ref_logprob = compute_logprob_sum(ref_model, ref_inputs)

                        kl = logprob - ref_logprob
                        loss = -(advantage * logprob) + args.kl_beta * kl
                        batch_loss += loss
                        sample_count += 1
                        logprob_values.append(float(logprob.item()))
                        ref_logprob_values.append(float(ref_logprob.item()))
                        kl_values.append(float(kl.item()))
                        sample_loss_values.append(float(loss.item()))

                        if debug_step and sample_idx < args.debug_preview_samples:
                            trace_row = {
                                "global_step": global_step,
                                "epoch": epoch,
                                "slurp_id": item.slurp_id,
                                "mode": item.mode,
                                "sample_idx": sample_idx,
                                "reward": float(reward),
                                "mean_reward": float(mean_reward),
                                "std_reward": float(std),
                                "advantage": float(advantage),
                                "logprob": float(logprob.item()),
                                "ref_logprob": float(ref_logprob.item()),
                                "kl": float(kl.item()),
                                "loss": float(loss.item()),
                                "gold_label": item.gold_label,
                                "pred_label": pred_labels[sample_idx] if sample_idx < len(pred_labels) else {},
                                "format_ok": bool(format_ok_values[sample_idx]) if sample_idx < len(format_ok_values) else False,
                                "prompt": _shorten(prompt, args.debug_max_chars),
                                "sample_raw": _shorten(sample_text, args.debug_max_chars),
                                "sample_formatted": _shorten(
                                    formatted_samples[sample_idx], args.debug_max_chars
                                )
                                if sample_idx < len(formatted_samples)
                                else "",
                            }
                            _debug_write_jsonl(debug_output_path, trace_row)
                            print(
                                f"[DEBUG][step={global_step}] sample#{sample_idx} "
                                f"adv={advantage:.4f} logprob={logprob.item():.4f} "
                                f"ref={ref_logprob.item():.4f} kl={kl.item():.4f} loss={loss.item():.4f}"
                            )
                except RuntimeError as exc:
                    if _is_oom_error(exc):
                        batch_had_oom = True
                        _recover_from_oom()
                        if rank == 0:
                            print(
                                f"[WARN][TRAIN-OOM] step={global_step} batch_idx={batch_idx} "
                                f"slurp_id={item.slurp_id} mode={item.mode} -> skip batch"
                            )
                        break
                    raise

            skip_batch = batch_had_oom or (sample_count == 0)
            if world_size > 1 and _is_distributed():
                skip_tensor = torch.tensor([1 if skip_batch else 0], device=device, dtype=torch.int32)
                dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
                skip_batch = bool(skip_tensor[0].item())
            if skip_batch:
                optimizer.zero_grad(set_to_none=True)
                _recover_from_oom()
                if rank == 0 and (not batch_had_oom):
                    print(
                        f"[WARN][TRAIN-OOM] step={global_step} batch_idx={batch_idx} "
                        "skipped because another rank reported OOM."
                    )
                continue
            batch_loss = batch_loss / sample_count
            backward_had_oom = False
            try:
                (batch_loss / args.grad_accum_steps).backward()
            except RuntimeError as exc:
                if _is_oom_error(exc):
                    backward_had_oom = True
                    _recover_from_oom()
                    if rank == 0:
                        print(
                            f"[WARN][TRAIN-OOM] step={global_step} batch_idx={batch_idx} "
                            "OOM during backward -> skip batch"
                        )
                else:
                    raise

            if world_size > 1 and _is_distributed():
                bw_oom_tensor = torch.tensor([1 if backward_had_oom else 0], device=device, dtype=torch.int32)
                dist.all_reduce(bw_oom_tensor, op=dist.ReduceOp.MAX)
                backward_had_oom = bool(bw_oom_tensor[0].item())

            if backward_had_oom:
                optimizer.zero_grad(set_to_none=True)
                accum_steps = 0
                _recover_from_oom()
                continue

            accum_steps += 1

            is_last_batch = batch_idx == (len(dataloader) - 1)
            if accum_steps >= args.grad_accum_steps or (is_last_batch and accum_steps > 0):
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_steps = 0

            if rank == 0 and args.log_every and global_step % args.log_every == 0:
                if reward_values:
                    reward_mean = sum(reward_values) / len(reward_values)
                    reward_min = min(reward_values)
                    reward_max = max(reward_values)
                else:
                    reward_mean = reward_min = reward_max = 0.0
                adv_mean = (sum(advantage_values) / len(advantage_values)) if advantage_values else 0.0
                kl_mean = (sum(kl_values) / len(kl_values)) if kl_values else 0.0
                logprob_mean = (sum(logprob_values) / len(logprob_values)) if logprob_values else 0.0
                ref_logprob_mean = (
                    (sum(ref_logprob_values) / len(ref_logprob_values)) if ref_logprob_values else 0.0
                )
                sample_loss_mean = (
                    (sum(sample_loss_values) / len(sample_loss_values)) if sample_loss_values else 0.0
                )
                print(
                    f"[GRPO] step={global_step} loss={batch_loss.item():.4f} samples={sample_count} "
                    f"reward_mean={reward_mean:.4f} reward_min={reward_min:.4f} reward_max={reward_max:.4f} "
                    f"adv_mean={adv_mean:.4f} kl_mean={kl_mean:.4f} "
                    f"logprob_mean={logprob_mean:.4f} ref_logprob_mean={ref_logprob_mean:.4f} "
                    f"sample_loss_mean={sample_loss_mean:.4f}"
                )

            if rank == 0 and args.save_every and global_step > 0 and global_step % args.save_every == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                _unwrap_model(model).save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)

            if args.eval_file and args.eval_every > 0 and global_step > 0 and global_step % args.eval_every == 0:
                eval_metrics, _ = evaluate_model(
                    model=model,
                    processor=processor,
                    items=eval_items,
                    db_definitions=db_definitions,
                    no_cot=args.no_cot,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    rank=rank,
                    world_size=world_size,
                    debug=args.debug,
                    debug_max_chars=args.debug_max_chars,
                    preview_count=(5 if args.smoke else 0),
                    preview_prefix="[GRPO-EVAL-SMOKE]",
                )
                if rank == 0:
                    print(
                        "[GRPO-EVAL] "
                        f"step={global_step} n={int(eval_metrics['num_samples'])} "
                        f"reward_mean={eval_metrics['reward_mean']:.4f} "
                        f"intent_acc={eval_metrics['intent_acc']:.4f} "
                        f"scenario_acc={eval_metrics['scenario_acc']:.4f} "
                        f"action_acc={eval_metrics['action_acc']:.4f} "
                        f"entity_f1_mean={eval_metrics['entity_f1_mean']:.4f} "
                        f"slu_f1={eval_metrics['slu_f1']:.4f}"
                    )
                if args.early_stopping:
                    metric_value = float(eval_metrics.get(args.early_stopping_metric, float("-inf")))
                    improved = metric_value > best_eval_metric
                    if improved:
                        best_eval_metric = metric_value
                        no_improve_count = 0
                    else:
                        if (epoch + 1) > args.early_stopping_min_epochs:
                            no_improve_count += 1
                    if rank == 0:
                        stage = (
                            "warmup"
                            if (epoch + 1) <= args.early_stopping_min_epochs
                            else "active"
                        )
                        print(
                            "[GRPO-ES] "
                            f"metric={args.early_stopping_metric} value={metric_value:.4f} "
                            f"best={best_eval_metric:.4f} no_improve={no_improve_count} "
                            f"patience={args.early_stopping_patience} stage={stage} epoch={epoch + 1}"
                        )
                    if (
                        (epoch + 1) > args.early_stopping_min_epochs
                        and no_improve_count >= args.early_stopping_patience
                    ):
                        early_stopped = True

            if args.test_file and args.test_every > 0 and global_step > 0 and global_step % args.test_every == 0:
                test_metrics, _ = evaluate_model(
                    model=model,
                    processor=processor,
                    items=test_items,
                    db_definitions=db_definitions,
                    no_cot=args.no_cot,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    rank=rank,
                    world_size=world_size,
                    debug=args.debug,
                    debug_max_chars=args.debug_max_chars,
                    show_progress=True,
                    progress_desc=f"test@step{global_step}",
                )
                if rank == 0:
                    print(
                        "[GRPO-TEST] "
                        f"step={global_step} n={int(test_metrics['num_samples'])} "
                        f"reward_mean={test_metrics['reward_mean']:.4f} "
                        f"intent_acc={test_metrics['intent_acc']:.4f} "
                        f"scenario_acc={test_metrics['scenario_acc']:.4f} "
                        f"action_acc={test_metrics['action_acc']:.4f} "
                        f"entity_f1_mean={test_metrics['entity_f1_mean']:.4f} "
                        f"slu_f1={test_metrics['slu_f1']:.4f}"
                    )

            global_step += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                reached_max_steps = True
                break
            if early_stopped:
                break

        if early_stopped:
            break
        if reached_max_steps:
            break

    if rank == 0 and early_stopped:
        print(
            "[GRPO] Early stopping triggered "
            f"(metric={args.early_stopping_metric}, patience={args.early_stopping_patience}, "
            f"min_epochs={args.early_stopping_min_epochs}, best={best_eval_metric:.4f})"
        )
    if rank == 0 and reached_max_steps:
        print(f"[GRPO] Reached max_steps={args.max_steps}; stopping training loop.")

    if args.eval_file:
        final_eval, _ = evaluate_model(
            model=model,
            processor=processor,
            items=eval_items,
            db_definitions=db_definitions,
            no_cot=args.no_cot,
            device=device,
            max_new_tokens=args.max_new_tokens,
            rank=rank,
            world_size=world_size,
            debug=args.debug,
            debug_max_chars=args.debug_max_chars,
            preview_count=(5 if args.smoke else 0),
            preview_prefix="[GRPO-EVAL-FINAL-SMOKE]",
        )
        if rank == 0:
            print(
                "[GRPO-EVAL-FINAL] "
                f"n={int(final_eval['num_samples'])} "
                f"reward_mean={final_eval['reward_mean']:.4f} "
                f"intent_acc={final_eval['intent_acc']:.4f} "
                f"scenario_acc={final_eval['scenario_acc']:.4f} "
                f"action_acc={final_eval['action_acc']:.4f} "
                f"entity_f1_mean={final_eval['entity_f1_mean']:.4f} "
                f"slu_f1={final_eval['slu_f1']:.4f}"
            )

    if args.test_file:
        final_test, prediction_rows = evaluate_model(
            model=model,
            processor=processor,
            items=test_items,
            db_definitions=db_definitions,
            no_cot=args.no_cot,
            device=device,
            max_new_tokens=args.max_new_tokens,
            rank=rank,
            world_size=world_size,
            debug=args.debug,
            debug_max_chars=args.debug_max_chars,
            collect_predictions=True,
            show_progress=True,
            progress_desc="test-final",
        )
        if rank == 0:
            print(
                "[GRPO-TEST-FINAL] "
                f"n={int(final_test['num_samples'])} "
                f"reward_mean={final_test['reward_mean']:.4f} "
                f"intent_acc={final_test['intent_acc']:.4f} "
                f"scenario_acc={final_test['scenario_acc']:.4f} "
                f"action_acc={final_test['action_acc']:.4f} "
                f"entity_f1_mean={final_test['entity_f1_mean']:.4f} "
                f"slu_f1={final_test['slu_f1']:.4f}"
            )
            prediction_path = os.path.join(output_dir, "prediction.jsonl")
            with open(prediction_path, "w", encoding="utf-8") as f:
                for row in prediction_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[GRPO-TEST-FINAL] saved predictions: {prediction_path}")

    if distributed:
        dist.barrier()
    if rank == 0:
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        _unwrap_model(model).save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
