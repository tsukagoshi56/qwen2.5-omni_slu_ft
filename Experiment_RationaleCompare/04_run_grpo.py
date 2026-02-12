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
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor
try:
    from transformers import Qwen2AudioForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen2AudioForConditionalGeneration = None
try:
    from peft import LoraConfig, TaskType, get_peft_model
    HAS_PEFT = True
except Exception:  # pragma: no cover
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    HAS_PEFT = False
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
PARAM_DEBUG_EVAL_SAMPLES = 200
AUDIO_MODULE_NAME_HINTS = (
    "audio_tower",
    "audio_encoder",
    "speech_encoder",
    "audio_backbone",
    "multi_modal_projector",
    "multimodal_projector",
    "audio_projector",
    "mm_projector",
)


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


def load_audio_model_from_pretrained(
    model_name_or_path: str,
    *,
    torch_dtype: torch.dtype,
    trust_remote_code: bool = True,
):
    attempts: List[Tuple[str, Any]] = [
        ("AutoModelForCausalLM", AutoModelForCausalLM),
        ("AutoModel", AutoModel),
    ]
    if Qwen2AudioForConditionalGeneration is not None:
        attempts.append(("Qwen2AudioForConditionalGeneration", Qwen2AudioForConditionalGeneration))

    errors: List[str] = []
    for loader_name, loader_cls in attempts:
        try:
            return loader_cls.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            errors.append(f"{loader_name}: {exc}")

    detail = " | ".join(errors) if errors else "no loader available"
    raise RuntimeError(
        f"Failed to load model '{model_name_or_path}' with audio-capable loaders. Details: {detail}"
    )


def get_audio_sampling_rate_or_raise(processor: Any, model_name_or_path: str) -> int:
    feature_extractor = getattr(processor, "feature_extractor", None)
    sampling_rate = getattr(feature_extractor, "sampling_rate", None) if feature_extractor is not None else None
    if sampling_rate is None:
        raise ValueError(
            f"Model '{model_name_or_path}' is not audio-ready in this script. "
            "Audio input is mandatory, so use an audio-capable checkpoint."
        )
    try:
        return int(sampling_rate)
    except Exception as exc:
        raise ValueError(
            f"Invalid audio sampling rate '{sampling_rate}' for model '{model_name_or_path}'."
        ) from exc


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


def _sample_items_by_slurp_id_ratio(
    items: List[GrpoItem],
    ratio: float,
    seed: int,
) -> Tuple[List[GrpoItem], Dict[str, int]]:
    if not items:
        return items, {"unique_ids": 0, "selected_ids": 0, "items_before": 0, "items_after": 0}

    item_keys: List[Any] = [_record_key(item, idx) for idx, item in enumerate(items)]
    unique_keys = sorted(
        set(item_keys),
        key=lambda x: (type(x).__name__, str(x)),
    )
    unique_count = len(unique_keys)

    if ratio >= 1.0:
        return (
            items,
            {
                "unique_ids": unique_count,
                "selected_ids": unique_count,
                "items_before": len(items),
                "items_after": len(items),
            },
        )
    if ratio <= 0.0:
        return (
            [],
            {
                "unique_ids": unique_count,
                "selected_ids": 0,
                "items_before": len(items),
                "items_after": 0,
            },
        )

    selected_count = int(unique_count * ratio)
    if selected_count == 0:
        selected_count = 1
    selected_count = min(selected_count, unique_count)

    rng = random.Random(seed)
    selected_keys = set(rng.sample(unique_keys, selected_count))
    sampled_items = [item for item, key in zip(items, item_keys) if key in selected_keys]

    return (
        sampled_items,
        {
            "unique_ids": unique_count,
            "selected_ids": selected_count,
            "items_before": len(items),
            "items_after": len(sampled_items),
        },
    )


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


def build_grpo_prompt(
    mode: str,
    sentence: str,
    db_definitions: str,
    no_cot: bool = False,
    label_cot: bool = False,
    candidates_only: bool = False,
) -> str:
    # Compact prompt for GRPO: instruction + input + output format only.
    # DB Definitions and detailed Rules are intentionally omitted.
    _ = db_definitions
    output_schema = (
        '{"Intent": "<scenario>_<action>", "entities": '
        '[{"type": "<entity_type>", "filler": "<entity_value>"}, ...]}'
    )
    label_only = bool(no_cot)
    cand_only = bool(candidates_only) and (not label_only)
    if label_cot:
        output_format = (
            "Output Format:\n"
            f"J: {output_schema}"
            if label_only
            else (
                (
                    "Output Format:\n"
                    "C: Intent candidates: intent1 | intent2 | intent3; Slot candidates: slot_type1(value1|value2) | slot_type2\n"
                    f"J: {output_schema}"
                )
                if cand_only
                else (
                    "Output Format:\n"
                    "C: Intent candidates: intent1 | intent2 | intent3; Slot candidates: slot_type1(value1|value2) | slot_type2\n"
                    "R: label1!reason1; label2!reason2; ...\n"
                    f"J: {output_schema}"
                )
            )
        )
        if mode == "audio":
            return (
                "System: Predict SLU labels from audio.\n\n"
                "[Input Data]\n"
                "- Audio: <AUDIO>\n\n"
                f"{output_format}"
            )
        text = str(sentence or "").strip()
        return (
            "System: Predict SLU labels from transcript.\n\n"
            "[Input Data]\n"
            f"- Transcript: {text}\n\n"
            f"{output_format}"
        )

    if mode == "audio":
        if label_only:
            return (
                "System: Predict SLU labels from audio.\n\n"
                "[Input Data]\n"
                "- Task: LABEL_ONLY\n"
                "- Audio: <AUDIO>\n\n"
                "Output Format:\n"
                f"J: {output_schema}"
            )
        if cand_only:
            return (
                "System: Predict SLU labels from audio.\n\n"
                "[Input Data]\n"
                "- Task: CANDIDATES_LABEL\n"
                "- Audio: <AUDIO>\n\n"
                "Output Format:\n"
                "C: Intent candidates: intent1 | intent2 | intent3; Slot candidates: slot_type1(value1|value2) | slot_type2\n"
                f"J: {output_schema}"
            )
        return (
            "System: Predict SLU labels from audio.\n\n"
            "[Input Data]\n"
            "- Task: COT_LABEL\n"
            "- Audio: <AUDIO>\n\n"
            "Output Format:\n"
            "C: Intent candidates: intent1 | intent2 | intent3; Slot candidates: slot_type1(value1|value2) | slot_type2\n"
            "R: label1!reason1; label2!reason2; ...\n"
            f"J: {output_schema}"
        )
    text = str(sentence or "").strip()
    if label_only:
        return (
            "System: Predict SLU labels from transcript.\n\n"
            "[Input Data]\n"
            "- Task: LABEL_ONLY\n"
            f"- Transcript: {text}\n\n"
            "Output Format:\n"
            f"J: {output_schema}"
        )
    if cand_only:
        return (
            "System: Predict SLU labels from transcript.\n\n"
            "[Input Data]\n"
            "- Task: CANDIDATES_LABEL\n"
            f"- Transcript: {text}\n\n"
            "Output Format:\n"
            "C: Intent candidates: intent1 | intent2 | intent3; Slot candidates: slot_type1(value1|value2) | slot_type2\n"
            f"J: {output_schema}"
        )
    return (
        "System: Predict SLU labels from transcript.\n\n"
        "[Input Data]\n"
        "- Task: COT_LABEL\n"
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


def model_forward(model: Any, inputs: Dict[str, torch.Tensor]):
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


def compute_token_logprobs(
    model: Any,
    inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return per-token log-probabilities for the response tokens (excluding prompt)."""
    prompt_len = int(inputs.get("_prompt_len", 0))
    outputs = model_forward(model, inputs)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    input_ids = inputs["input_ids"]

    target_ids = input_ids[:, 1:]
    log_probs = log_probs[:, :-1, :]
    start = max(prompt_len - 1, 0)
    if start >= log_probs.shape[1]:
        return torch.tensor([0.0], device=input_ids.device)
    token_logprobs = log_probs[0, start:, :].gather(1, target_ids[0, start:].unsqueeze(-1)).squeeze(-1)
    return token_logprobs


def generate_samples(
    model: Any,
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

    # Switch to eval mode for generation to disable dropout etc.
    was_training = model.training
    model.eval()

    outputs: List[str] = []
    with torch.no_grad():
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

    # Restore previous mode.
    if was_training:
        model.train()
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
    slot_part = re.sub(r"\([^)]*\)", "", slot_part)
    values = [x.strip().lower() for x in slot_part.split("|") if x.strip()]
    return [v for v in values if v and v not in {"(none)", "none"}]


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
    # unique keep order
    seen = set()
    uniq: List[str] = []
    for s in slots:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _candidate_bonus(
    formatted_output: str,
    gold_label: Dict[str, Any],
    w_intent_candidate: float,
    w_slot_candidate: float,
    w_intent_count: float = 0.0,
    intent_count_target: int = 3,
) -> Tuple[float, bool, bool, int, float]:
    if (w_intent_candidate <= 0.0) and (w_slot_candidate <= 0.0) and (w_intent_count <= 0.0):
        return 0.0, False, False, 0, 0.0

    intent_candidates = _extract_intent_candidates(formatted_output)
    slot_candidates = _extract_slot_candidates(formatted_output)
    intent_count = len(intent_candidates)

    gold_intent = _gold_intent_from_label(gold_label)
    gold_slots = set(_gold_slot_types_from_label(gold_label))
    cand_slots = {str(x).strip().lower() for x in slot_candidates if str(x).strip()}

    has_intent = bool(gold_intent) and (gold_intent in intent_candidates)
    has_slot = bool(gold_slots & cand_slots) if gold_slots else False

    bonus = 0.0
    if has_intent:
        bonus += float(w_intent_candidate)
    if has_slot:
        bonus += float(w_slot_candidate)

    # Encourage C: intent candidate list length to be close to target (default=3).
    target = max(1, int(intent_count_target))
    proximity = max(0.0, 1.0 - (abs(intent_count - target) / float(target)))
    count_bonus = float(w_intent_count) * float(proximity) if w_intent_count > 0.0 else 0.0
    bonus += count_bonus

    return bonus, has_intent, has_slot, intent_count, count_bonus


def _extract_rationale_pairs(text: str) -> List[Tuple[str, str]]:
    r_line = _extract_prefixed_line(text, "R:")
    if r_line == "R: (none)":
        return []
    body = r_line.split(":", 1)[1].strip() if ":" in r_line else ""
    if not body:
        return []
    parts = [p.strip() for p in body.split(";") if p.strip()]
    pairs: List[Tuple[str, str]] = []
    for part in parts:
        if "!" in part:
            label, reason = part.split("!", 1)
            pairs.append((label.strip(), reason.strip()))
        else:
            pairs.append((part.strip(), ""))
    return pairs


def _rationale_candidate_coverage(
    formatted_output: str,
    gold_label: Optional[Dict[str, Any]] = None,
) -> Tuple[float, bool, int, int]:
    intent_candidates = set(_extract_intent_candidates(formatted_output))
    slot_candidates = {str(x).strip().lower() for x in _extract_slot_candidates(formatted_output) if str(x).strip()}

    # Exclude the accepted (gold) intent from the denominator because R only
    # lists rejection reasons for non-accepted candidates.
    if gold_label is not None:
        gold_intent_key = _gold_intent_from_label(gold_label)
        intent_candidates.discard(gold_intent_key)

    total_candidates = len(intent_candidates) + len(slot_candidates)
    if total_candidates == 0:
        return 1.0, True, 0, 0

    covered_intents: set = set()
    covered_slots: set = set()
    rationale_pairs = _extract_rationale_pairs(formatted_output)
    for raw_label, raw_reason in rationale_pairs:
        reason = str(raw_reason or "").strip()
        if not reason:
            continue
        label = str(raw_label or "").strip()
        intent_key = normalize_intent_label(label)
        slot_key = re.sub(r"\([^)]*\)", "", label).strip().lower()
        slot_key = re.sub(r"^(slot|slots)\s*:\s*", "", slot_key)
        if intent_key in intent_candidates:
            covered_intents.add(intent_key)
        if slot_key in slot_candidates:
            covered_slots.add(slot_key)

    covered_candidates = len(covered_intents) + len(covered_slots)
    coverage = float(covered_candidates) / float(total_candidates)
    full_coverage = covered_candidates == total_candidates
    return coverage, full_coverage, covered_candidates, total_candidates


def _is_j_fully_correct(stats: Dict[str, Any]) -> bool:
    if not isinstance(stats, dict):
        return False
    if not bool(stats.get("scenario_ok", False)):
        return False
    if not bool(stats.get("action_ok", False)):
        return False
    return True


def _apply_cot_reward_adjustment(
    reward: float,
    *,
    format_ok: bool,
    cot_only: bool,
    cot_format_bonus: float,
    cot_format_penalty: float,
) -> float:
    if not cot_only:
        return float(reward)
    adjusted = float(reward)
    if format_ok:
        adjusted += float(cot_format_bonus)
    else:
        adjusted -= float(cot_format_penalty)
    return adjusted


def _format_model_output(
    raw_output: str,
    no_cot: bool,
    candidates_only: bool = False,
) -> Tuple[str, Dict[str, Any], bool]:
    pred_obj = parse_j_from_output(raw_output)
    pred_label = _normalize_pred_label(pred_obj)
    j_obj = {
        "Intent": pred_label.get("intent", ""),
        "entities": pred_label.get("entities", []),
    }
    j_line = "J: " + json.dumps(j_obj, ensure_ascii=False)
    has_j = pred_obj is not None
    label_only = bool(no_cot)
    cand_only = bool(candidates_only) and (not label_only)
    if label_only:
        return j_line, pred_label, has_j

    c_line = _extract_prefixed_line(raw_output, "C:")
    r_line = _extract_prefixed_line(raw_output, "R:")
    has_c = c_line != "C: (none)"
    has_r = r_line != "R: (none)"
    if cand_only:
        formatted = "\n".join([c_line, j_line])
        return formatted, pred_label, (has_c and has_j)
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


def _parse_lora_targets(targets: str) -> List[str]:
    return [x.strip() for x in str(targets or "").split(",") if x.strip()]


def _resolve_lora_targets(
    model: Any,
    base_targets: List[str],
    include_audio_tower: bool,
) -> Any:
    if not include_audio_tower:
        return base_targets

    audio_linear_names: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        lname = str(name).lower()
        if any(hint in lname for hint in AUDIO_MODULE_NAME_HINTS):
            audio_linear_names.append(name)

    if not audio_linear_names:
        return base_targets

    pattern_parts: List[str] = []
    if base_targets:
        escaped_suffix = "|".join(re.escape(x) for x in base_targets)
        pattern_parts.append(rf".*\.(?:{escaped_suffix})")
    escaped_audio = "|".join(re.escape(x) for x in sorted(set(audio_linear_names)))
    pattern_parts.append(rf"(?:{escaped_audio})")
    return rf"(?:{'|'.join(pattern_parts)})"


def _pick_param_debug_eval_items(
    items: List[GrpoItem],
    *,
    enabled: bool,
    sample_size: int,
    seed: int,
    eval_index: int,
) -> List[GrpoItem]:
    if not enabled:
        return items
    if sample_size <= 0 or len(items) <= sample_size:
        return items
    rng = random.Random(f"param_debug_eval:{seed}:{eval_index}")
    return rng.sample(items, sample_size)


def evaluate_model(
    model,
    processor: AutoProcessor,
    items: List[GrpoItem],
    db_definitions: str,
    no_cot: bool,
    label_cot: bool,
    candidates_only: bool,
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
    reward_w_intent_candidate: float = 0.0,
    reward_w_slot_candidate: float = 0.0,
    reward_w_c_intent_count: float = 0.0,
    reward_c_intent_target: int = 3,
    reward_w_rationale_coverage: float = 0.25,
    cot_only: bool = False,
    cot_format_bonus: float = 0.0,
    cot_format_penalty: float = 0.25,
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
    rationale_cov_sum = 0.0
    rationale_full_sum = 0.0
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
                    label_cot=label_cot,
                    candidates_only=candidates_only,
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
                formatted_text, pred_label, format_ok = _format_model_output(
                    generated_text,
                    no_cot=no_cot,
                    candidates_only=candidates_only,
                )
                reward, stats = compute_reward(pred_label, item.gold_label)
                cand_bonus, has_int_cand, has_slot_cand, c_intent_count, c_count_bonus = _candidate_bonus(
                    formatted_output=formatted_text,
                    gold_label=item.gold_label,
                    w_intent_candidate=reward_w_intent_candidate,
                    w_slot_candidate=reward_w_slot_candidate,
                    w_intent_count=reward_w_c_intent_count,
                    intent_count_target=reward_c_intent_target,
                )
                rationale_cov, rationale_full, rationale_cov_numer, rationale_cov_denom = _rationale_candidate_coverage(
                    formatted_text, gold_label=item.gold_label
                )
                j_fully_correct = _is_j_fully_correct(stats)
                rationale_reward_term = (
                    float(reward_w_rationale_coverage) * float(rationale_cov) if j_fully_correct else 0.0
                )
                reward += cand_bonus
                reward += rationale_reward_term
                reward = _apply_cot_reward_adjustment(
                    reward,
                    format_ok=bool(format_ok),
                    cot_only=bool(cot_only),
                    cot_format_bonus=float(cot_format_bonus),
                    cot_format_penalty=float(cot_format_penalty),
                )
            except RuntimeError as exc:
                if _is_oom_error(exc):
                    local_oom_skips += 1.0
                    _recover_from_oom()
                    if rank == 0:
                        print(
                            f"[WARN][EVAL-OOM] skipped sample slurp_id={item.slurp_id} "
                            f"mode={item.mode}: {exc}"
                        )
                    continue
                raise

            local_count += 1.0
            # Print eval sample every 10 samples for debugging rationale quality
            if rank == 0 and int(local_count) % 10 == 1:
                print(
                    f"[EVAL-SAMPLE {int(local_count)}] reward={reward:.3f} "
                    f"format_ok={format_ok} rationale_cov={rationale_cov:.2f} "
                    f"j_correct={j_fully_correct}\n"
                    f"  GOLD: {json.dumps(item.gold_label, ensure_ascii=False)}\n"
                    f"  OUTPUT: {generated_text[:500]}",
                    flush=True,
                )
            reward_sum += float(reward)
            scenario_sum += 1.0 if stats["scenario_ok"] else 0.0
            action_sum += 1.0 if stats["action_ok"] else 0.0
            intent_sum += 1.0 if stats["intent_ok"] else 0.0
            entity_f1_sum += float(stats["entity_f1"])
            rationale_cov_sum += float(rationale_cov)
            rationale_full_sum += 1.0 if rationale_full else 0.0
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
                        "has_gold_intent_candidate": bool(has_int_cand),
                        "has_gold_slot_candidate": bool(has_slot_cand),
                        "c_intent_count": int(c_intent_count),
                        "c_count_bonus": float(c_count_bonus),
                        "rationale_candidate_coverage": float(rationale_cov),
                        "rationale_candidate_full_coverage": bool(rationale_full),
                        "rationale_covered_candidates": int(rationale_cov_numer),
                        "rationale_total_candidates": int(rationale_cov_denom),
                        "j_fully_correct_for_rationale": bool(j_fully_correct),
                        "rationale_reward_term": float(rationale_reward_term),
                        "candidate_bonus": float(cand_bonus),
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
                    f"format_ok={bool(format_ok)} has_intent_cand={bool(has_int_cand)} "
                    f"has_slot_cand={bool(has_slot_cand)} c_intent_count={int(c_intent_count)} "
                    f"c_count_bonus={c_count_bonus:.4f} cand_bonus={cand_bonus:.4f} "
                    f"rat_cov={rationale_cov:.3f} rat_full={bool(rationale_full)} "
                    f"j_ok={bool(j_fully_correct)} r_term={rationale_reward_term:.4f} reward={reward:.4f}"
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
            rationale_cov_sum,
            rationale_full_sum,
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
            "rationale_candidate_coverage_mean": 0.0,
            "rationale_candidate_full_rate": 0.0,
            "slu_f1": 0.0,
        }
    else:
        slu_f1_value = _f1_from_counts(
            float(metrics_tensor[8].item()),
            float(metrics_tensor[9].item()),
            float(metrics_tensor[10].item()),
        )
        result = {
            "num_samples": total,
            "reward_mean": float(metrics_tensor[1].item() / total),
            "scenario_acc": float(metrics_tensor[2].item() / total),
            "action_acc": float(metrics_tensor[3].item() / total),
            "intent_acc": float(metrics_tensor[4].item() / total),
            "entity_f1_mean": float(metrics_tensor[5].item() / total),
            "rationale_candidate_coverage_mean": float(metrics_tensor[6].item() / total),
            "rationale_candidate_full_rate": float(metrics_tensor[7].item() / total),
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
            "Run GRPO directly from base model(s) without SFT prerequisite. "
            f"If --model_name_or_path is empty, it defaults to {DEFAULT_ONLY_GRPO_MODEL}."
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
        "--candidates_only",
        "--candidates-only",
        dest="candidates_only",
        action="store_true",
        help="Use C+J prompting (intent/slot candidates + final J), without R rationale.",
    )
    parser.add_argument(
        "--no_candidates_only",
        "--no-candidates-only",
        dest="candidates_only",
        action="store_false",
        help="Disable C+J mode and use the normal C/R/J or J-only style.",
    )
    parser.add_argument(
        "--cot_only",
        "--cot-only",
        dest="cot_only",
        action="store_true",
        default=True,
        help="Force CoT-style training (C/R/J). Applies CoT format reward shaping. (default: True)",
    )
    parser.add_argument(
        "--no_cot_only",
        "--no-cot-only",
        dest="cot_only",
        action="store_false",
        help="Disable CoT format reward shaping.",
    )
    parser.add_argument(
        "--cot_format_bonus",
        type=float,
        default=0.0,
        help="Extra reward when C/R/J format is valid under --cot_only.",
    )
    parser.add_argument(
        "--cot_format_penalty",
        type=float,
        default=0.25,
        help="Penalty when C/R/J format is broken under --cot_only.",
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
    parser.add_argument(
        "--label_cot",
        "--label-COT",
        "--label_CoT",
        dest="label_cot",
        action="store_true",
        help="Use multitask.py-aligned prompt text/format style (no Task tag line).",
    )
    parser.add_argument(
        "--no_label_cot",
        "--no-label-COT",
        "--no_label_CoT",
        dest="label_cot",
        action="store_false",
        help="Use GRPO native prompt text/format style.",
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
        "--train_id_sample_ratio",
        "--train-id-sample-ratio",
        dest="train_id_sample_ratio",
        type=float,
        default=1.0,
        help=(
            "Randomly keep only this ratio of unique training slurp_id values "
            "before other balancing/smoke filtering (0.0-1.0)."
        ),
    )
    parser.add_argument(
        "--train_id_sample_seed",
        "--train-id-sample-seed",
        dest="train_id_sample_seed",
        type=int,
        default=None,
        help="Seed for train slurp_id subsampling (default: --seed).",
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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", dest="do_sample", action="store_true", default=True, help="Enable sampling (default: True).")
    parser.add_argument(
        "--no_do_sample",
        "--no-do-sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling.",
    )
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--use_lora",
        "--use-lora",
        dest="use_lora",
        action="store_true",
        help="Enable LoRA adapters for policy model fine-tuning.",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora_audio_tower",
        "--lora-audio-tower",
        dest="lora_audio_tower",
        action="store_true",
        help="Include audio-related linear layers in LoRA targets.",
    )
    parser.add_argument(
        "--no_lora_audio_tower",
        "--no-lora-audio-tower",
        dest="lora_audio_tower",
        action="store_false",
        help="Do not include audio-related linear layers in LoRA targets.",
    )
    parser.add_argument(
        "--lora_target_modules",
        "--lora-target-modules",
        dest="lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names to apply LoRA to.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Maximum number of GRPO training steps. 0 means no step cap (use epochs only).",
    )
    parser.add_argument("--eval_every", type=int, default=500, help="Run eval every N global steps (0 disables).")
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
        "--reward_w_intent_candidate",
        type=float,
        default=0.25,
        help="Bonus reward when gold intent appears in C: intent candidates.",
    )
    parser.add_argument(
        "--reward_w_slot_candidate",
        type=float,
        default=0.25,
        help="Bonus reward when any gold slot type appears in C: slot candidates.",
    )
    parser.add_argument(
        "--reward_w_c_intent_count",
        type=float,
        default=0.25,
        help="Bonus weight for keeping C: intent candidate count close to --reward_c_intent_target.",
    )
    parser.add_argument(
        "--reward_c_intent_target",
        type=int,
        default=3,
        help="Target number of intent candidates in C: for count-shaping reward.",
    )
    parser.add_argument(
        "--reward_w_rationale_coverage",
        type=float,
        default=0.25,
        help="Bonus weight multiplied by rationale coverage over candidates in C (0.0 disables).",
    )
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
        "--eval_preview_count",
        type=int,
        default=3,
        help="Number of real model outputs to print for each eval call (set 0 to disable).",
    )
    parser.add_argument(
        "--test_preview_count",
        type=int,
        default=3,
        help="Number of real model outputs to print for each test-eval call (set 0 to disable).",
    )
    parser.add_argument(
        "--param_debug",
        "--param-debug",
        dest="param_debug",
        action="store_true",
        help=(
            "Quick GRPO parameter-debug preset: eval every 50 steps "
            f"with a fixed random {PARAM_DEBUG_EVAL_SAMPLES}-sample eval subset."
        ),
    )
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
    parser.set_defaults(
        include_text=True,
        label_cot=False,
        no_cot=False,
        candidates_only=False,
        early_stopping=False,
        shuffle_train=False,
        do_sample=True,
        param_debug=False,
        use_lora=False,
        lora_audio_tower=True,
    )

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
    if args.train_id_sample_ratio < 0.0 or args.train_id_sample_ratio > 1.0:
        raise ValueError("--train_id_sample_ratio must satisfy 0.0 <= value <= 1.0")
    if args.balanced_train_records < 0:
        raise ValueError("--balanced_train_records must be >= 0")
    if args.balanced_per_intent < 0:
        raise ValueError("--balanced_per_intent must be >= 0")
    if args.balanced_per_intent_slot_combo < 0:
        raise ValueError("--balanced_per_intent_slot_combo must be >= 0")
    if args.eval_preview_count < 0:
        raise ValueError("--eval_preview_count must be >= 0")
    if args.test_preview_count < 0:
        raise ValueError("--test_preview_count must be >= 0")
    if args.reward_c_intent_target <= 0:
        raise ValueError("--reward_c_intent_target must be >= 1")
    if args.max_steps < 0:
        raise ValueError("--max_steps must be >= 0")
    if args.lora_r < 1:
        raise ValueError("--lora_r must be >= 1")
    if args.lora_alpha < 1:
        raise ValueError("--lora_alpha must be >= 1")
    if args.lora_dropout < 0 or args.lora_dropout >= 1:
        raise ValueError("--lora_dropout must satisfy 0 <= value < 1")
    if args.no_cot and args.candidates_only:
        raise ValueError("--no_cot and --candidates_only cannot be used together")
    if args.cot_only and (args.no_cot or args.candidates_only):
        raise ValueError("--cot_only cannot be used with --no_cot or --candidates_only")
    if args.cot_only:
        args.no_cot = False
        args.candidates_only = False
    if args.smoke:
        args.num_train_epochs = 1
        args.group_size = min(args.group_size, 2)
        args.max_new_tokens = min(args.max_new_tokens, 96)
        args.log_every = 1
        if args.eval_file and args.eval_every <= 0:
            args.eval_every = 10
        if args.test_file and args.test_every <= 0:
            args.test_every = 10
    if args.param_debug:
        if args.eval_file:
            args.eval_every = 500
        # Keep full eval pool; the run will draw one fixed random subset.
        args.eval_max_samples = None
    if args.use_lora and not HAS_PEFT:
        raise ImportError(
            "--use_lora was specified, but peft is not installed. "
            "Install peft in this environment and retry."
        )

    if args.only_grpo:
        requested_policy = str(args.model_name_or_path).strip()
        requested_ref = str(args.ref_model_name_or_path).strip()
        if not requested_policy:
            requested_policy = DEFAULT_ONLY_GRPO_MODEL
        if not requested_ref:
            requested_ref = requested_policy
        args.model_name_or_path = requested_policy
        args.ref_model_name_or_path = requested_ref
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
    sample_seed = args.seed if args.train_id_sample_seed is None else int(args.train_id_sample_seed)
    items, sample_stats = _sample_items_by_slurp_id_ratio(
        items=items,
        ratio=args.train_id_sample_ratio,
        seed=sample_seed,
    )
    if rank == 0:
        print(
            "[GRPO] train slurp_id subsampling: "
            f"ratio={args.train_id_sample_ratio:.4f} seed={sample_seed} "
            f"selected_ids={sample_stats['selected_ids']}/{sample_stats['unique_ids']} "
            f"items={sample_stats['items_after']}/{sample_stats['items_before']}"
        )

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
            if args.param_debug:
                eval_cap = None
            elif eval_cap is None:
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

    # In --param_debug mode, keep one fixed eval subset for the whole run.
    param_debug_eval_items = eval_items
    if args.eval_file and args.param_debug:
        param_debug_eval_items = _pick_param_debug_eval_items(
            eval_items,
            enabled=True,
            sample_size=PARAM_DEBUG_EVAL_SAMPLES,
            seed=args.seed,
            eval_index=0,
        )

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
            f"use_lora={args.use_lora} lora_r={args.lora_r} lora_alpha={args.lora_alpha} "
            f"lora_dropout={args.lora_dropout} lora_audio_tower={args.lora_audio_tower} "
            f"include_text={args.include_text} no_cot={args.no_cot} "
            f"candidates_only={args.candidates_only} label_cot={args.label_cot} "
            f"cot_only={args.cot_only} "
            f"cot_format_bonus={args.cot_format_bonus} cot_format_penalty={args.cot_format_penalty} "
            f"reward_w_c_intent_count={args.reward_w_c_intent_count} "
            f"reward_c_intent_target={args.reward_c_intent_target} "
            f"reward_w_rationale_coverage={args.reward_w_rationale_coverage} "
            f"eval_preview_count={args.eval_preview_count} "
            f"test_preview_count={args.test_preview_count} "
            f"smoke={args.smoke} "
            f"param_debug={args.param_debug} "
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
                f"eval_every={args.eval_every} eval_max_samples={args.eval_max_samples} "
                f"param_debug_eval_subset={PARAM_DEBUG_EVAL_SAMPLES if args.param_debug else 'off'}"
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
        prompt_style = "J_ONLY" if args.no_cot else ("C_J" if args.candidates_only else "C_R_J")
        print(f"[GRPO] prompt_style={prompt_style}")
        print(
            "[GRPO] c_intent_count_reward "
            f"target={args.reward_c_intent_target} weight={args.reward_w_c_intent_count}"
        )
        print(
            "[GRPO] rationale_coverage_reward "
            f"weight={args.reward_w_rationale_coverage}"
        )
        if args.only_grpo:
            print(
                "[GRPO] only_grpo=True -> train from base model(s) without SFT prerequisite: "
                f"policy={args.model_name_or_path} ref={args.ref_model_name_or_path or args.model_name_or_path}"
            )
        if args.eval_file:
            print(
                f"[GRPO] eval enabled: eval_file={eval_path} "
                f"eval_items={len(eval_items)} eval_every={args.eval_every}"
            )
            if args.param_debug:
                print(
                    "[GRPO] param_debug=True -> eval uses one fixed random subset "
                    f"for this run: n={len(param_debug_eval_items)}"
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
    _ = get_audio_sampling_rate_or_raise(processor, args.model_name_or_path)

    model = load_audio_model_from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    if args.use_lora:
        lora_base_targets = _parse_lora_targets(args.lora_target_modules)
        if not lora_base_targets:
            raise ValueError("--lora_target_modules must contain at least one module name")
        resolved_targets = _resolve_lora_targets(
            model=model,
            base_targets=lora_base_targets,
            include_audio_tower=args.lora_audio_tower,
        )
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=resolved_targets,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        if rank == 0:
            print(
                "[GRPO] LoRA enabled: "
                f"r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} "
                f"audio_tower={args.lora_audio_tower} "
                f"base_targets={','.join(lora_base_targets)} "
                f"resolved_target_type={'regex' if isinstance(resolved_targets, str) else 'list'}"
            )
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()
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
    ref_model = load_audio_model_from_pretrained(
        ref_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for optimizer.")
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    best_eval_metric = float("-inf")
    no_improve_count = 0
    early_stopped = False
    reached_max_steps = False
    reward_running_sum = 0.0
    reward_running_count = 0
    if args.eval_file:
        init_eval_items = param_debug_eval_items if args.param_debug else eval_items
        init_eval_metrics, _ = evaluate_model(
            model=model,
            processor=processor,
            items=init_eval_items,
            db_definitions=db_definitions,
            no_cot=args.no_cot,
            label_cot=args.label_cot,
            candidates_only=args.candidates_only,
            device=device,
            max_new_tokens=args.max_new_tokens,
            rank=rank,
            world_size=world_size,
            debug=args.debug,
            debug_max_chars=args.debug_max_chars,
            preview_count=args.eval_preview_count,
            preview_prefix="[GRPO-EVAL-INIT-PREVIEW]",
            reward_w_intent_candidate=args.reward_w_intent_candidate,
            reward_w_slot_candidate=args.reward_w_slot_candidate,
            reward_w_c_intent_count=args.reward_w_c_intent_count,
            reward_c_intent_target=args.reward_c_intent_target,
            reward_w_rationale_coverage=args.reward_w_rationale_coverage,
            cot_only=args.cot_only,
            cot_format_bonus=args.cot_format_bonus,
            cot_format_penalty=args.cot_format_penalty,
        )
        if rank == 0:
            print(
                "[GRPO-EVAL-INIT] "
                f"n={int(init_eval_metrics['num_samples'])} "
                f"reward_mean={init_eval_metrics['reward_mean']:.4f} "
                f"intent_acc={init_eval_metrics['intent_acc']:.4f} "
                f"scenario_acc={init_eval_metrics['scenario_acc']:.4f} "
                f"action_acc={init_eval_metrics['action_acc']:.4f} "
                f"entity_f1_mean={init_eval_metrics['entity_f1_mean']:.4f} "
                f"rat_cov={init_eval_metrics['rationale_candidate_coverage_mean']:.4f} "
                f"rat_full={init_eval_metrics['rationale_candidate_full_rate']:.4f} "
                f"slu_f1={init_eval_metrics['slu_f1']:.4f}"
            )
            if args.param_debug:
                print(f"[GRPO-EVAL-INIT] param_debug_subset={len(init_eval_items)}")
        if args.early_stopping:
            init_metric_value = float(
                init_eval_metrics.get(args.early_stopping_metric, float("-inf"))
            )
            if math.isfinite(init_metric_value):
                best_eval_metric = init_metric_value
                no_improve_count = 0
                if rank == 0:
                    print(
                        "[GRPO-ES] "
                        f"initial metric={args.early_stopping_metric} value={init_metric_value:.4f}"
                    )
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
            # Diversity tracking per group.
            diversity_unique_outputs: List[int] = []
            diversity_unique_rewards: List[int] = []
            diversity_unique_intents: List[int] = []
            diversity_reward_spread: List[float] = []

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
                        label_cot=args.label_cot,
                        candidates_only=args.candidates_only,
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
                    candidate_bonus_values: List[float] = []
                    has_intent_candidate_values: List[bool] = []
                    has_slot_candidate_values: List[bool] = []
                    c_intent_count_values: List[int] = []
                    c_count_bonus_values: List[float] = []
                    rationale_cov_values: List[float] = []
                    rationale_full_values: List[bool] = []
                    rationale_cov_count_values: List[int] = []
                    rationale_total_count_values: List[int] = []
                    j_fully_correct_values: List[bool] = []
                    rationale_reward_values: List[float] = []
                    for text in samples:
                        formatted_text, pred_label, format_ok = _format_model_output(
                            text,
                            no_cot=args.no_cot,
                            candidates_only=args.candidates_only,
                        )
                        reward, stats = compute_reward(
                            pred_label,
                            item.gold_label,
                            w_scenario=args.reward_w_scenario,
                            w_action=args.reward_w_action,
                            w_intent=args.reward_w_intent,
                            w_entity=args.reward_w_entity,
                        )
                        cand_bonus, has_int_cand, has_slot_cand, c_intent_count, c_count_bonus = _candidate_bonus(
                            formatted_output=formatted_text,
                            gold_label=item.gold_label,
                            w_intent_candidate=args.reward_w_intent_candidate,
                            w_slot_candidate=args.reward_w_slot_candidate,
                            w_intent_count=args.reward_w_c_intent_count,
                            intent_count_target=args.reward_c_intent_target,
                        )
                        rationale_cov, rationale_full, rationale_cov_numer, rationale_cov_denom = _rationale_candidate_coverage(
                            formatted_text, gold_label=item.gold_label
                        )
                        j_fully_correct = _is_j_fully_correct(stats)
                        rationale_reward_term = (
                            float(args.reward_w_rationale_coverage) * float(rationale_cov) if j_fully_correct else 0.0
                        )
                        reward += cand_bonus
                        reward += rationale_reward_term
                        reward = _apply_cot_reward_adjustment(
                            reward,
                            format_ok=bool(format_ok),
                            cot_only=bool(args.cot_only),
                            cot_format_bonus=float(args.cot_format_bonus),
                            cot_format_penalty=float(args.cot_format_penalty),
                        )
                        rewards.append(reward)
                        reward_values.append(float(reward))
                        pred_labels.append(pred_label)
                        formatted_samples.append(formatted_text)
                        format_ok_values.append(bool(format_ok))
                        candidate_bonus_values.append(float(cand_bonus))
                        has_intent_candidate_values.append(bool(has_int_cand))
                        has_slot_candidate_values.append(bool(has_slot_cand))
                        c_intent_count_values.append(int(c_intent_count))
                        c_count_bonus_values.append(float(c_count_bonus))
                        rationale_cov_values.append(float(rationale_cov))
                        rationale_full_values.append(bool(rationale_full))
                        rationale_cov_count_values.append(int(rationale_cov_numer))
                        rationale_total_count_values.append(int(rationale_cov_denom))
                        j_fully_correct_values.append(bool(j_fully_correct))
                        rationale_reward_values.append(float(rationale_reward_term))

                    if debug_step:
                        preview_n = min(args.debug_preview_samples, len(samples))
                        for i in range(preview_n):
                            print(
                                f"[DEBUG][step={global_step}] sample#{i} reward={rewards[i]:.4f} "
                                f"format_ok={format_ok_values[i]} "
                                f"has_intent_cand={has_intent_candidate_values[i]} "
                                f"has_slot_cand={has_slot_candidate_values[i]} "
                                f"c_intent_count={c_intent_count_values[i]} "
                                f"c_count_bonus={c_count_bonus_values[i]:.4f} "
                                f"cand_bonus={candidate_bonus_values[i]:.4f} "
                                f"rat_cov={rationale_cov_values[i]:.3f} "
                                f"rat_full={rationale_full_values[i]} "
                                f"j_ok={j_fully_correct_values[i]} "
                                f"r_term={rationale_reward_values[i]:.4f} "
                                f"pred={json.dumps(pred_labels[i], ensure_ascii=False)}"
                            )
                            _print_debug_section(f"train.sample_raw#{i}", samples[i])
                            _print_debug_section(f"train.sample_formatted#{i}", formatted_samples[i])

                    if rank == 0 and global_step % 10 == 0:
                        print(
                            f"[TRAIN-SAMPLE][step={global_step}] slurp_id={item.slurp_id} mode={item.mode} "
                            f"reward={rewards[0]:.4f} gold={json.dumps(item.gold_label, ensure_ascii=False)}"
                        )
                        print(f"[TRAIN-SAMPLE] raw_output: {samples[0]}")

                    mean_reward = sum(rewards) / max(len(rewards), 1)
                    if args.advantage_normalize:
                        variance = sum((r - mean_reward) ** 2 for r in rewards) / max(len(rewards), 1)
                        std = math.sqrt(variance) if variance > 0 else 1.0
                    else:
                        std = 1.0

                    # Track group diversity.
                    n_unique_out = len(set(samples))
                    n_unique_rew = len(set(rewards))
                    pred_intents = [pl.get("intent", "") for pl in pred_labels]
                    n_unique_int = len(set(pred_intents))
                    r_spread = max(rewards) - min(rewards) if rewards else 0.0
                    diversity_unique_outputs.append(n_unique_out)
                    diversity_unique_rewards.append(n_unique_rew)
                    diversity_unique_intents.append(n_unique_int)
                    diversity_reward_spread.append(r_spread)

                    if debug_step:
                        print(
                            f"[DEBUG][step={global_step}] diversity: "
                            f"unique_outputs={n_unique_out}/{len(samples)} "
                            f"unique_rewards={n_unique_rew}/{len(rewards)} "
                            f"unique_intents={n_unique_int} "
                            f"reward_spread={r_spread:.4f}"
                        )

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
                        token_lps = compute_token_logprobs(model, inputs)

                        with torch.no_grad():
                            ref_inputs = prepare_inputs(
                                processor=processor,
                                prompt_text=prompt_text,
                                full_text=full_text,
                                audio=audio,
                                device=device,
                            )
                            ref_token_lps = compute_token_logprobs(ref_model, ref_inputs)

                        # Align lengths (rare edge-case where tokenization differs).
                        min_len = min(token_lps.shape[0], ref_token_lps.shape[0])
                        token_lps = token_lps[:min_len]
                        ref_token_lps = ref_token_lps[:min_len]

                        # Per-token KL divergence: KL(π_θ || π_ref) ≈ log π_θ - log π_ref
                        per_token_kl = token_lps - ref_token_lps

                        # Mean log-probability (length-normalised).
                        logprob_mean = token_lps.mean()
                        ref_logprob_mean = ref_token_lps.mean()
                        kl_mean_val = per_token_kl.mean()

                        # GRPO loss: -advantage * mean(log π_θ) + β * mean(per-token KL)
                        # advantage is a Python float (no grad), so this is correct REINFORCE.
                        loss = -(advantage * logprob_mean) + args.kl_beta * kl_mean_val
                        batch_loss += loss
                        sample_count += 1
                        logprob_values.append(float(logprob_mean.item()))
                        ref_logprob_values.append(float(ref_logprob_mean.item()))
                        kl_values.append(float(kl_mean_val.item()))
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
                                "logprob": float(logprob_mean.item()),
                                "ref_logprob": float(ref_logprob_mean.item()),
                                "kl": float(kl_mean_val.item()),
                                "loss": float(loss.item()),
                                "gold_label": item.gold_label,
                                "pred_label": pred_labels[sample_idx] if sample_idx < len(pred_labels) else {},
                                "format_ok": bool(format_ok_values[sample_idx]) if sample_idx < len(format_ok_values) else False,
                                "has_gold_intent_candidate": (
                                    bool(has_intent_candidate_values[sample_idx])
                                    if sample_idx < len(has_intent_candidate_values)
                                    else False
                                ),
                                "has_gold_slot_candidate": (
                                    bool(has_slot_candidate_values[sample_idx])
                                    if sample_idx < len(has_slot_candidate_values)
                                    else False
                                ),
                                "c_intent_count": (
                                    int(c_intent_count_values[sample_idx])
                                    if sample_idx < len(c_intent_count_values)
                                    else 0
                                ),
                                "c_count_bonus": (
                                    float(c_count_bonus_values[sample_idx])
                                    if sample_idx < len(c_count_bonus_values)
                                    else 0.0
                                ),
                                "candidate_bonus": (
                                    float(candidate_bonus_values[sample_idx])
                                    if sample_idx < len(candidate_bonus_values)
                                    else 0.0
                                ),
                                "rationale_candidate_coverage": (
                                    float(rationale_cov_values[sample_idx])
                                    if sample_idx < len(rationale_cov_values)
                                    else 0.0
                                ),
                                "rationale_candidate_full_coverage": (
                                    bool(rationale_full_values[sample_idx])
                                    if sample_idx < len(rationale_full_values)
                                    else False
                                ),
                                "rationale_covered_candidates": (
                                    int(rationale_cov_count_values[sample_idx])
                                    if sample_idx < len(rationale_cov_count_values)
                                    else 0
                                ),
                                "rationale_total_candidates": (
                                    int(rationale_total_count_values[sample_idx])
                                    if sample_idx < len(rationale_total_count_values)
                                    else 0
                                ),
                                "j_fully_correct_for_rationale": (
                                    bool(j_fully_correct_values[sample_idx])
                                    if sample_idx < len(j_fully_correct_values)
                                    else False
                                ),
                                "rationale_reward_term": (
                                    float(rationale_reward_values[sample_idx])
                                    if sample_idx < len(rationale_reward_values)
                                    else 0.0
                                ),
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
                                f"adv={advantage:.4f} logprob={logprob_mean.item():.4f} "
                                f"ref={ref_logprob_mean.item():.4f} kl={kl_mean_val.item():.4f} loss={loss.item():.4f}"
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

            if rank == 0 and reward_values:
                reward_running_sum += float(sum(reward_values))
                reward_running_count += len(reward_values)

            if rank == 0 and args.log_every and global_step % args.log_every == 0:
                if reward_values:
                    reward_mean = sum(reward_values) / len(reward_values)
                    reward_min = min(reward_values)
                    reward_max = max(reward_values)
                else:
                    reward_mean = reward_min = reward_max = 0.0
                reward_mean_cum = (
                    reward_running_sum / reward_running_count if reward_running_count > 0 else 0.0
                )
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
                    f"reward_mean_cum={reward_mean_cum:.4f} reward_seen={reward_running_count} "
                    f"adv_mean={adv_mean:.4f} kl_mean={kl_mean:.4f} "
                    f"logprob_mean={logprob_mean:.4f} ref_logprob_mean={ref_logprob_mean:.4f} "
                    f"sample_loss_mean={sample_loss_mean:.4f}"
                )
                # Log diversity metrics.
                if diversity_unique_outputs:
                    avg_uniq_out = sum(diversity_unique_outputs) / len(diversity_unique_outputs)
                    avg_uniq_rew = sum(diversity_unique_rewards) / len(diversity_unique_rewards)
                    avg_uniq_int = sum(diversity_unique_intents) / len(diversity_unique_intents)
                    avg_r_spread = sum(diversity_reward_spread) / len(diversity_reward_spread)
                    print(
                        f"[GRPO-DIV] step={global_step} "
                        f"avg_unique_outputs={avg_uniq_out:.1f}/{args.group_size} "
                        f"avg_unique_rewards={avg_uniq_rew:.1f}/{args.group_size} "
                        f"avg_unique_intents={avg_uniq_int:.1f} "
                        f"avg_reward_spread={avg_r_spread:.4f}"
                    )

            if rank == 0 and args.save_every and global_step > 0 and global_step % args.save_every == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                _unwrap_model(model).save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)

            if args.eval_file and args.eval_every > 0 and global_step > 0 and global_step % args.eval_every == 0:
                eval_items_step = param_debug_eval_items if args.param_debug else eval_items
                eval_metrics, _ = evaluate_model(
                    model=model,
                    processor=processor,
                    items=eval_items_step,
                    db_definitions=db_definitions,
                    no_cot=args.no_cot,
                    label_cot=args.label_cot,
                    candidates_only=args.candidates_only,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    rank=rank,
                    world_size=world_size,
                    debug=args.debug,
                    debug_max_chars=args.debug_max_chars,
                    preview_count=args.eval_preview_count,
                    preview_prefix="[GRPO-EVAL-PREVIEW]",
                    reward_w_intent_candidate=args.reward_w_intent_candidate,
                    reward_w_slot_candidate=args.reward_w_slot_candidate,
                    reward_w_c_intent_count=args.reward_w_c_intent_count,
                    reward_c_intent_target=args.reward_c_intent_target,
                    reward_w_rationale_coverage=args.reward_w_rationale_coverage,
                    cot_only=args.cot_only,
                    cot_format_bonus=args.cot_format_bonus,
                    cot_format_penalty=args.cot_format_penalty,
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
                        f"rat_cov={eval_metrics['rationale_candidate_coverage_mean']:.4f} "
                        f"rat_full={eval_metrics['rationale_candidate_full_rate']:.4f} "
                        f"slu_f1={eval_metrics['slu_f1']:.4f}"
                    )
                    if args.param_debug:
                        print(f"[GRPO-EVAL] step={global_step} param_debug_subset={len(eval_items_step)}")
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
                    label_cot=args.label_cot,
                    candidates_only=args.candidates_only,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    rank=rank,
                    world_size=world_size,
                    debug=args.debug,
                    debug_max_chars=args.debug_max_chars,
                    show_progress=True,
                    progress_desc=f"test@step{global_step}",
                    preview_count=args.test_preview_count,
                    preview_prefix="[GRPO-TEST-PREVIEW]",
                    reward_w_intent_candidate=args.reward_w_intent_candidate,
                    reward_w_slot_candidate=args.reward_w_slot_candidate,
                    reward_w_c_intent_count=args.reward_w_c_intent_count,
                    reward_c_intent_target=args.reward_c_intent_target,
                    reward_w_rationale_coverage=args.reward_w_rationale_coverage,
                    cot_only=args.cot_only,
                    cot_format_bonus=args.cot_format_bonus,
                    cot_format_penalty=args.cot_format_penalty,
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
                        f"rat_cov={test_metrics['rationale_candidate_coverage_mean']:.4f} "
                        f"rat_full={test_metrics['rationale_candidate_full_rate']:.4f} "
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
        final_eval_items = param_debug_eval_items if args.param_debug else eval_items
        final_eval, _ = evaluate_model(
            model=model,
            processor=processor,
            items=final_eval_items,
            db_definitions=db_definitions,
            no_cot=args.no_cot,
            label_cot=args.label_cot,
            candidates_only=args.candidates_only,
            device=device,
            max_new_tokens=args.max_new_tokens,
            rank=rank,
            world_size=world_size,
            debug=args.debug,
            debug_max_chars=args.debug_max_chars,
            preview_count=args.eval_preview_count,
            preview_prefix="[GRPO-EVAL-FINAL-PREVIEW]",
            reward_w_intent_candidate=args.reward_w_intent_candidate,
            reward_w_slot_candidate=args.reward_w_slot_candidate,
            reward_w_c_intent_count=args.reward_w_c_intent_count,
            reward_c_intent_target=args.reward_c_intent_target,
            reward_w_rationale_coverage=args.reward_w_rationale_coverage,
            cot_only=args.cot_only,
            cot_format_bonus=args.cot_format_bonus,
            cot_format_penalty=args.cot_format_penalty,
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
                f"rat_cov={final_eval['rationale_candidate_coverage_mean']:.4f} "
                f"rat_full={final_eval['rationale_candidate_full_rate']:.4f} "
                f"slu_f1={final_eval['slu_f1']:.4f}"
            )
            if args.param_debug:
                print(f"[GRPO-EVAL-FINAL] param_debug_subset={len(final_eval_items)}")

    if args.test_file:
        final_test, prediction_rows = evaluate_model(
            model=model,
            processor=processor,
            items=test_items,
            db_definitions=db_definitions,
            no_cot=args.no_cot,
            label_cot=args.label_cot,
            candidates_only=args.candidates_only,
            device=device,
            max_new_tokens=args.max_new_tokens,
            rank=rank,
            world_size=world_size,
            debug=args.debug,
            debug_max_chars=args.debug_max_chars,
            collect_predictions=True,
            show_progress=True,
            progress_desc="test-final",
            preview_count=args.test_preview_count,
            preview_prefix="[GRPO-TEST-FINAL-PREVIEW]",
            reward_w_intent_candidate=args.reward_w_intent_candidate,
            reward_w_slot_candidate=args.reward_w_slot_candidate,
            reward_w_c_intent_count=args.reward_w_c_intent_count,
            reward_c_intent_target=args.reward_c_intent_target,
            reward_w_rationale_coverage=args.reward_w_rationale_coverage,
            cot_only=args.cot_only,
            cot_format_bonus=args.cot_format_bonus,
            cot_format_penalty=args.cot_format_penalty,
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
                f"rat_cov={final_test['rationale_candidate_coverage_mean']:.4f} "
                f"rat_full={final_test['rationale_candidate_full_rate']:.4f} "
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
