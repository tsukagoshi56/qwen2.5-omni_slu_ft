#!/usr/bin/env python3
"""
Audio+N-best+Rationale training and distributed inference for SLU labels.

- Train target format is kept as:
  {"scenario": "...", "action": "...", "entities": [{"type": "...", "filler": "..."}]}
- Input prompt (audio mode) uses ASR n-best and rationale text.
- Test output JSONL keeps full context (n-best/rationale/target/raw output).
- Evaluation uses label-only extraction from prediction JSONL.
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

import librosa
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

try:
    import jiwer

    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT_HEADER = (
    "Predict the final SLU label from speech, ASR n-best, and rationale.\n"
    "Output JSON only with keys: scenario, action, entities."
)


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
    candidates = item.get("candidates", []) or []
    rationale_text = item.get("rationale_text", "") or ""
    transcript = item.get("transcript", "") or ""

    blocks = [PROMPT_HEADER]
    if include_transcript and transcript:
        blocks.append(f"Transcript:\n{transcript}")
    blocks.append("ASR n-best hypotheses:\n" + format_nbest(candidates))
    blocks.append("Rationale:\n" + (rationale_text if rationale_text else "(none)"))
    blocks.append(
        "Return only: {\"scenario\":\"...\",\"action\":\"...\",\"entities\":[{\"type\":\"...\",\"filler\":\"...\"}]}"
    )
    return "\n\n".join(blocks)


# ==============================================================================
# 1. Data Loading
# ==============================================================================


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

    if not hasattr(resolve_audio_path, "_debug_count"):
        resolve_audio_path._debug_count = 0
    if resolve_audio_path._debug_count < 10:
        logger.warning("Could not find %s. Checked: %s", filename, candidates)
        resolve_audio_path._debug_count += 1
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

    scenario = str(final_obj.get("scenario", "")).strip()
    action = str(final_obj.get("action", "")).strip()

    intent = str(final_obj.get("intent", "")).strip()
    if (not scenario or not action) and intent:
        inferred_scenario, inferred_action = intent_to_scenario_action(intent)
        scenario = scenario or inferred_scenario
        action = action or inferred_action

    entities = parse_entities(final_obj.get("entities", []))

    return {
        "scenario": scenario,
        "action": action,
        "entities": entities,
    }


def build_items_from_rationale_jsonl(
    jsonl_path: str,
    audio_dir: str,
    add_text_only: bool = False,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.exists(jsonl_path):
        logger.warning("JSONL file not found: %s", jsonl_path)
        return items

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if max_samples is not None and len(items) >= max_samples:
            break
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, str):
            # Some files contain JSON-encoded JSON strings.
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                continue
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            # Some rows can be a JSON string/array. Skip safely.
            continue

        sample_id = str(data.get("id", ""))
        filename = data.get("filename")
        if not filename:
            meta = data.get("meta")
            if isinstance(meta, dict):
                filename = meta.get("filename")
        filename = str(filename or "").strip()

        candidates = data.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        candidates = [candidate_to_text(c) for c in candidates]
        candidates = [c for c in candidates if c]

        transcript = candidates[0] if candidates else ""
        rationale_text = normalize_rationale_text(data.get("rationale_text"))
        target_obj = extract_target_obj(data)
        target_str = json.dumps(target_obj, ensure_ascii=False)

        audio_path = resolve_audio_path(audio_dir, filename) if filename else None

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
        }

        if add_text_only:
            items.append({**base_item, "audio_path": None})

        if audio_path:
            items.append(base_item)

    logger.info("Loaded %s -> %d items", jsonl_path, len(items))
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

        sr = self.processor.feature_extractor.sampling_rate
        eos_token = self.processor.tokenizer.eos_token or "<|endoftext|>"

        for item in batch:
            if item.get("audio_path") is None:
                continue
            try:
                audio, _ = librosa.load(item["audio_path"], sr=sr)
            except Exception:
                continue

            text_input = self._build_audio_chat(item)
            full_text = text_input + item["target"] + eos_token

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
        }

    def _collate_text(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list, labels_list = [], []

        eos_token = self.processor.tokenizer.eos_token or "<|endoftext|>"
        for item in batch:
            if item.get("audio_path") is not None:
                continue
            text_input = self._build_text_chat(item)
            full_text = text_input + item["target"] + eos_token

            inputs = self.processor.tokenizer(full_text, return_tensors="pt")
            prompt_inputs = self.processor.tokenizer(text_input, return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]

            ids = inputs["input_ids"][0]
            lbs = ids.clone()
            lbs[:prompt_len] = self.ignore_index
            input_ids_list.append(ids)
            labels_list.append(lbs)

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


# ==============================================================================
# 5. Callback
# ==============================================================================


class SampleGenerationCallback(TrainerCallback):
    def __init__(self, eval_items, processor, model, num_samples: int = 3):
        self.eval_items = eval_items
        self.processor = processor
        self.model = model
        self.num_samples = num_samples

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
                audio, _ = librosa.load(item["audio_path"], sr=sr)
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
                    output_ids = self.model.generate(**inputs, max_new_tokens=128)

                input_len = inputs["input_ids"].shape[1]
                generated_text = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
                clean_pred = clean_json_text(generated_text)

                logger.info("-" * 60)
                logger.info("File:       %s", item.get("file"))
                logger.info("N-best[0]:  %s", (item.get("candidates") or [""])[0])
                logger.info("Rationale:  %s", (item.get("rationale_text") or "")[:120])
                logger.info("Target:     %s", item.get("target"))
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


def parse_prediction_label(raw_output: str) -> Dict[str, Any]:
    default_obj = {"scenario": "error", "action": "error", "entities": []}

    json_str = clean_json_text(raw_output)
    try:
        parsed = json.loads(json_str)
    except Exception:
        # Last fallback: extract first {...} block.
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if not match:
            return default_obj
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return default_obj

    if not isinstance(parsed, dict):
        return default_obj

    return {
        "scenario": str(parsed.get("scenario", "")).strip(),
        "action": str(parsed.get("action", "")).strip(),
        "entities": parse_entities(parsed.get("entities", [])),
    }


def calculate_wer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0
    if HAS_JIWER:
        return float(jiwer.wer(reference, hypothesis))
    return 0.0 if reference.strip() == hypothesis.strip() else 1.0


def _run_batch_inference(
    model,
    processor,
    batch_items: List[Dict[str, Any]],
    device,
    sr: int,
    is_audio: bool,
) -> List[Dict[str, Any]]:
    texts: List[str] = []
    audios: List[Any] = []

    for item in batch_items:
        if is_audio:
            audio, _ = librosa.load(item["audio_path"], sr=sr)
            audios.append(audio)
            prompt_text = build_prompt_text(item)
            user_content = [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": prompt_text},
            ]
        else:
            prompt_text = build_prompt_text(item, include_transcript=True)
            user_content = [{"type": "text", "text": prompt_text}]

        text_input = processor.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(text_input)

    if is_audio:
        inputs = processor(
            text=texts,
            audio=audios,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(text=texts, return_tensors="pt", padding=True)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    raw_outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)

    results: List[Dict[str, Any]] = []
    for item, raw_output in zip(batch_items, raw_outputs):
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


def run_distributed_inference(model, processor, items, output_path, device, rank, world_size, batch_size=1):
    model.eval()

    my_items = items[rank::world_size]
    my_audio_items = [x for x in my_items if x.get("audio_path") is not None]
    my_text_items = [x for x in my_items if x.get("audio_path") is None]

    local_results: List[Dict[str, Any]] = []
    processor.tokenizer.padding_side = "left"

    if rank == 0:
        logger.info("Starting Inference. Items: %d, Batch size: %d", len(my_items), batch_size)

    sr = processor.feature_extractor.sampling_rate

    def iter_batches(data: List[Dict[str, Any]]) -> Iterator[List[Dict[str, Any]]]:
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    for i, batch_items in enumerate(iter_batches(my_audio_items)):
        if rank == 0 and i % 10 == 0:
            logger.info("Audio batch %d/%d", i + 1, (len(my_audio_items) + batch_size - 1) // batch_size)
        try:
            local_results.extend(
                _run_batch_inference(
                    model=model,
                    processor=processor,
                    batch_items=batch_items,
                    device=device,
                    sr=sr,
                    is_audio=True,
                )
            )
        except Exception as exc:
            logger.error("Rank %d failed on audio batch %d: %s", rank, i, exc)

    for i, batch_items in enumerate(iter_batches(my_text_items)):
        if rank == 0 and i % 10 == 0:
            logger.info("Text batch %d/%d", i + 1, (len(my_text_items) + batch_size - 1) // batch_size)
        try:
            local_results.extend(
                _run_batch_inference(
                    model=model,
                    processor=processor,
                    batch_items=batch_items,
                    device=device,
                    sr=sr,
                    is_audio=False,
                )
            )
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
        default="/lustre/home/71200138/qwen_test/experiments/CoT_maker/ASR_cot_train.jsonl",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="/lustre/home/71200138/qwen_test/experiments/CoT_maker/ASR_cot_devel.jsonl",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="If omitted, eval_file is used.",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/lustre/home/71200138/INTERSPEECH/experiment1/slurp/audio/slurp_real",
    )

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_rationale_label_ft")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--add_text_only", action="store_true", help="Also add text-only samples.")
    parser.add_argument("--smoke", action="store_true", help="Run tiny smoke test.")

    args = parser.parse_args()
    if args.test_file is None:
        args.test_file = args.eval_file

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

    if args.smoke:
        if rank == 0:
            logger.info("SMOKE MODE ON")
        args.max_samples = 32
        args.num_train_epochs = 1

    train_items = build_items_from_rationale_jsonl(
        args.train_file,
        args.audio_dir,
        add_text_only=args.add_text_only,
        max_samples=args.max_samples,
    )
    eval_items = build_items_from_rationale_jsonl(
        args.eval_file,
        args.audio_dir,
        add_text_only=args.add_text_only,
        max_samples=(args.max_samples // 2) if args.max_samples else None,
    )

    if rank == 0:
        logger.info("Train items: %d | Eval items: %d", len(train_items), len(eval_items))

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

    # Keep the original lightweight FT setup.
    model.audio_tower.requires_grad_(False)
    model.multi_modal_projector.requires_grad_(False)

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
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=MixedDataset(train_items),
        eval_dataset=MixedDataset(eval_items) if len(eval_items) > 0 else None,
        data_collator=SmartCollator(processor),
        tokenizer=processor.tokenizer,
    )

    if len(eval_items) > 0:
        trainer.add_callback(
            SampleGenerationCallback(
                eval_items=eval_items,
                processor=processor,
                model=model,
                num_samples=3,
            )
        )

    trainer.train()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    if world_size > 1:
        dist.barrier()

    test_max_samples = 500 if args.smoke else None
    if rank == 0 and args.smoke:
        logger.info("Loading only %d test items (smoke).", test_max_samples)

    test_items = build_items_from_rationale_jsonl(
        args.test_file,
        args.audio_dir,
        add_text_only=False,
        max_samples=test_max_samples,
    )

    output_jsonl = os.path.join(args.output_dir, "prediction.jsonl")
    run_distributed_inference(
        model=model,
        processor=processor,
        items=test_items,
        output_path=output_jsonl,
        device=device,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
    )

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        label_only_path = os.path.join(args.output_dir, "prediction_labels_only.jsonl")
        save_label_only_predictions(output_jsonl, label_only_path)

        metrics = evaluate_prediction_file(output_jsonl)
        metrics_path = os.path.join(args.output_dir, "metrics_label_only.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        logger.info("Label-only evaluation metrics: %s", json.dumps(metrics, ensure_ascii=False))
        logger.info("Saved full predictions: %s", output_jsonl)
        logger.info("Saved label-only predictions: %s", label_only_path)
        logger.info("Saved metrics: %s", metrics_path)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
