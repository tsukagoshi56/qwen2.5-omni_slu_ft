#!/usr/bin/env python3
"""
Audio-Text Mix Training & Distributed Inference (SLURP Format)
Hierarchical Output-Side Semantic Candidates (SBERT Clustering)
==============================================================

- クラスタリング（SBERT + KMeans）で以下をグルーピングして “出力側” に候補を提示:
  1) Scenario（例: general query / mail query ...）
  2) Action
  3) Slot Type（entity type）
  4) (optional) intent_label = scenario_action も追加で提示可能

- 学習ターゲット例（階層）:
  Scenario Group: ['x', 'y', ...] Selection: <scenario>
  Action Group:   ['a', 'b', ...] Selection: <action>
  Slot Group ['slotA','slotB',...] Selection: <slot_type> | ...
  Answer: {"scenario":..., "action":..., "entities":[...]}

- Audio/Text 混在学習、DDP、推論（distributed）対応。
"""

import argparse
import json
import pickle
import os
import random
import re
import glob
import shutil
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Iterator, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.nn.utils.rnn import pad_sequence

import librosa
from transformers import AutoProcessor, TrainingArguments, Trainer, TrainerCallback

# Model class
try:
    from transformers import Qwen2AudioForConditionalGeneration
    MODEL_CLS = Qwen2AudioForConditionalGeneration
except Exception:
    from transformers import AutoModelForCausalLM
    MODEL_CLS = AutoModelForCausalLM

# SBERT & clustering
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

# WER
try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------
BASE_PROMPT_TEXT = (
    "You are a hierarchical voice assistant. Analyze the user's request. "
    "First, identify the semantic group of the intent. "
    "Then, select the specific label (scenario/action/slots) from that group. "
    "Finally, output the answer as a JSON object."
)
ASR_PROMPT_TEXT = "Transcribe the audio."

# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------
def resolve_audio_path(audio_root: str, filename: str) -> Optional[str]:
    candidates = [
        os.path.join(audio_root, filename),
        os.path.join(audio_root, "slurp_real", filename),
        os.path.join("slurp", "audio", "slurp_real", filename),
        os.path.join("slurp", "slurp_real", filename),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def calculate_wer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0
    if HAS_JIWER:
        return jiwer.wer(reference, hypothesis)
    ref = reference.split()
    hyp = hypothesis.split()
    return 1.0 if ref != hyp else 0.0

def extract_json_from_mixed_output(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if "Answer:" in text:
        parts = text.split("Answer:", 1)
        potential = parts[1].strip() if len(parts) > 1 else ""
        match2 = re.search(r"(\{.*\})", potential, re.DOTALL)
        if match2:
            return match2.group(1).strip()
        return potential
    match3 = re.findall(r"(\{.*\})", text, re.DOTALL)
    if match3:
        return match3[-1].strip()
    return text

def _normalize_asr_list(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for v in value:
        if not isinstance(v, str):
            continue
        v = re.sub(r"\s+", " ", v.strip())
        if v:
            out.append(v)
    return list(dict.fromkeys(out))

def load_asr_cache(cache_path: str) -> Dict[str, List[str]]:
    if not cache_path or not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r") as f:
            raw = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load ASR cache: {e}")
        return {}
    if not isinstance(raw, dict):
        return {}
    cache: Dict[str, List[str]] = {}
    for k, v in raw.items():
        cache[str(k)] = _normalize_asr_list(v)
    return cache

def save_asr_cache(cache_path: str, cache: Dict[str, List[str]]) -> None:
    if not cache_path:
        return
    try:
        dir_path = os.path.dirname(cache_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save ASR cache: {e}")

def build_asr_cache(
    jsonl_paths: List[str],
    audio_dir: str,
    model,
    processor,
    device: torch.device,
    n_best: int,
) -> Dict[str, List[str]]:
    n_best = max(1, int(n_best))
    cache: Dict[str, List[str]] = {}
    sr = processor.feature_extractor.sampling_rate
    processor.tokenizer.padding_side = "left"
    was_training = model.training
    model.eval()

    for path in jsonl_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                slurp_id = d.get("slurp_id", -1)
                sid = str(slurp_id)
                if sid in cache:
                    continue
                recs = d.get("recordings", []) or []
                audio_path = None
                for rec in recs:
                    if not isinstance(rec, dict):
                        continue
                    fname = rec.get("file", "")
                    audio_path = resolve_audio_path(audio_dir, fname)
                    if audio_path:
                        break
                if not audio_path:
                    cache[sid] = []
                    continue

                audio, _ = librosa.load(audio_path, sr=sr)
                user_content = [
                    {"type": "audio", "audio_url": "placeholder"},
                    {"type": "text", "text": ASR_PROMPT_TEXT},
                ]
                text_input = processor.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        num_return_sequences=n_best,
                        temperature=0.8,
                        top_p=0.95,
                        use_cache=True,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )

                input_len = inputs["input_ids"].shape[1]
                hyps: List[str] = []
                for k in range(out_ids.shape[0]):
                    hyp = processor.decode(out_ids[k][input_len:], skip_special_tokens=True).strip()
                    hyp = re.sub(r"\s+", " ", hyp)
                    if hyp:
                        hyps.append(hyp)
                cache[sid] = list(dict.fromkeys(hyps))

    if was_training:
        model.train()
    return cache

# ----------------------------------------------------------------------
# 1) Clustering
# ----------------------------------------------------------------------
def _identity_group_map(items: List[str]) -> Dict[str, List[str]]:
    return {x: [x] for x in items}

def get_clusters(items: List[str], n_clusters: int, name: str, rank: int = 0) -> Dict[str, List[str]]:
    """
    items を SBERT で埋め込み → KMeans でクラスタリング。
    戻り値: item -> sorted(cluster_members)
    """
    items = [x for x in items if x]
    if not items:
        return {}

    # Adjust
    n_clusters = max(1, min(n_clusters, len(items)))

    if not HAS_SBERT:
        if rank == 0:
            logger.warning(f"[{name}] sentence-transformers not found. Falling back to identity groups.")
        return _identity_group_map(items)

    if rank == 0:
        logger.info(f"Clustering {len(items)} {name} into {n_clusters} groups...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    cleaned = [i.replace("_", " ") for i in items]
    embeddings = model.encode(cleaned, show_progress_bar=(rank == 0))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    groups: Dict[str, List[str]] = {}
    for item, cid in zip(items, cluster_labels):
        groups.setdefault(str(cid), []).append(item)

    # sort members for deterministic prompting
    for cid in list(groups.keys()):
        groups[cid] = sorted(groups[cid])

    item_to_group = {item: groups[str(cid)] for item, cid in zip(items, cluster_labels)}
    return item_to_group

def scan_metadata(jsonl_paths: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    scenarios, actions, slot_types, intent_labels = set(), set(), set(), set()

    for path in jsonl_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                s = d.get("scenario")
                a = d.get("action")
                if s:
                    scenarios.add(s)
                if a:
                    actions.add(a)
                if s and a:
                    intent_labels.add(f"{s}_{a}")
                for e in d.get("entities", []) or []:
                    t = (e or {}).get("type")
                    if t:
                        slot_types.add(t)

    return (
        sorted(list(scenarios)),
        sorted(list(actions)),
        sorted(list(slot_types)),
        sorted(list(intent_labels)),
    )

def build_all_clusters(
    jsonl_paths: List[str],
    n_scenario_clusters: int,
    n_action_clusters: int,
    n_slot_clusters: int,
    n_intent_clusters: int,
    include_intent_label_group: bool,
    rank: int = 0,
) -> Dict[str, Dict[str, List[str]]]:
    """
    戻り値:
      {
        "scenario": s_map,
        "action":   a_map,
        "slot":     sl_map,
        "intent":   intent_map (optional)
      }
    """
    scenarios, actions, slots, intents = scan_metadata(jsonl_paths)

    s_map = get_clusters(scenarios, n_scenario_clusters, "Scenarios", rank=rank)
    a_map = get_clusters(actions, n_action_clusters, "Actions", rank=rank)
    sl_map = get_clusters(slots, n_slot_clusters, "Slot Types", rank=rank)

    out = {"scenario": s_map, "action": a_map, "slot": sl_map}

    if include_intent_label_group:
        intent_map = get_clusters(intents, n_intent_clusters, "Intent Labels (scenario_action)", rank=rank)
        out["intent"] = intent_map

    return out

def ddp_broadcast_object(obj: Any, rank: int, world_size: int) -> Any:
    """
    rank0 で作った Python object を全 rank へ配布。
    broadcast_object_list を使い、NCCL バックエンドでも動作するように device を指定する。
    """
    if world_size <= 1:
        return obj

    # Determine device for NCCL backend
    backend = dist.get_backend()
    if backend == "nccl":
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
        except Exception:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Use broadcast_object_list with device parameter (PyTorch >= 1.8)
    obj_list = [obj] if rank == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0, device=device)
    return obj_list[0]

def maybe_load_or_build_clusters(
    cache_path: str,
    jsonl_paths: List[str],
    n_scenario_clusters: int,
    n_action_clusters: int,
    n_slot_clusters: int,
    n_intent_clusters: int,
    include_intent_label_group: bool,
    rank: int,
    world_size: int,
) -> Dict[str, Dict[str, List[str]]]:
    """
    rank0 で cluster cache をロード/生成し、DDPで全 rank に配る。
    """
    clusters = None
    if rank == 0 and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                clusters = json.load(f)
            logger.info(f"Loaded cluster cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load cluster cache: {e}")

    clusters = ddp_broadcast_object(clusters, rank, world_size)

    if clusters is None:
        if rank == 0:
            clusters = build_all_clusters(
                jsonl_paths=jsonl_paths,
                n_scenario_clusters=n_scenario_clusters,
                n_action_clusters=n_action_clusters,
                n_slot_clusters=n_slot_clusters,
                n_intent_clusters=n_intent_clusters,
                include_intent_label_group=include_intent_label_group,
                rank=rank,
            )
            if cache_path:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "w") as f:
                        json.dump(clusters, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved cluster cache: {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to save cluster cache: {e}")

        clusters = ddp_broadcast_object(clusters, rank, world_size)

    return clusters

# ----------------------------------------------------------------------
# 2) Build items (targets)
# ----------------------------------------------------------------------
def _quote_list(xs: List[str]) -> str:
    return ", ".join([f"'{x}'" for x in xs])

def build_items_from_slurp(
    jsonl_path: str,
    audio_dir: str,
    clusters: Dict[str, Dict[str, List[str]]],
    asr_cache: Optional[Dict[str, List[str]]] = None,
    add_text_only: bool = True,
    max_samples: Optional[int] = None,
    include_intent_label_group: bool = True,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not jsonl_path or not os.path.exists(jsonl_path):
        return items

    s_map = clusters.get("scenario", {})
    a_map = clusters.get("action", {})
    sl_map = clusters.get("slot", {})
    intent_map = clusters.get("intent", {}) if include_intent_label_group else {}

    with open(jsonl_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if max_samples is not None and len(items) >= max_samples:
            break
        try:
            d = json.loads(line)
        except Exception:
            continue

        s = d.get("scenario") or ""
        a = d.get("action") or ""
        transcript = d.get("sentence", "") or ""
        slurp_id = d.get("slurp_id", -1)
        asr_list = []
        if asr_cache is not None:
            asr_list = asr_cache.get(str(slurp_id), [])
        if asr_list:
            transcript = random.choice(asr_list)

        # entities normalize
        raw_entities = d.get("entities", []) or []
        tokens = d.get("tokens", []) or []
        processed_entities = []
        for e in raw_entities:
            if not isinstance(e, dict):
                continue
            filler = e.get("filler")
            if (not filler) and ("span" in e) and tokens:
                try:
                    span = e["span"]
                    if isinstance(span, list) and len(span) > 0:
                        selected = tokens[span[0] : span[-1] + 1]
                        filler = " ".join([t.get("surface", "") for t in selected if isinstance(t, dict)])
                except Exception:
                    pass
            if filler is None:
                filler = ""
            processed_entities.append({"type": e.get("type", ""), "filler": filler})

        # groups
        s_members = s_map.get(s, [s]) if s else [""]
        a_members = a_map.get(a, [a]) if a else [""]

        # slot groups (unique types)
        slot_types = sorted(list({(e.get("type") or "") for e in processed_entities if (e.get("type") or "")}))
        slot_info_parts = []
        for st in slot_types:
            members = sl_map.get(st, [st])
            slot_info_parts.append(f"Slot Group [{_quote_list(members)}] Selection: {st}")
        slot_target_str = " | ".join(slot_info_parts) if slot_info_parts else "Slot: None"

        # optional intent label group (scenario_action)
        intent_label = f"{s}_{a}" if s and a else ""
        intent_block = ""
        if include_intent_label_group and intent_label:
            i_members = intent_map.get(intent_label, [intent_label])
            intent_block = (
                f"Intent Group: [{_quote_list(i_members)}] Selection: {intent_label}\n"
            )

        json_obj = {"scenario": s, "action": a, "entities": processed_entities}
        json_str = json.dumps(json_obj, ensure_ascii=False)

        full_target = (
            f"{intent_block}"
            f"Scenario Group: [{_quote_list(s_members)}] Selection: {s}\n"
            f"Action Group: [{_quote_list(a_members)}] Selection: {a}\n"
            f"{slot_target_str}\n"
            f"Answer: {json_str}"
        )

        item_base = {
            "slurp_id": slurp_id,
            "transcript": transcript,
            "target": full_target,
            "system_prompt": BASE_PROMPT_TEXT,
        }

        if add_text_only:
            items.append({**item_base, "file": None, "audio_path": None})

        recs = d.get("recordings", []) or []
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            fname = rec.get("file", "")
            audio_path = resolve_audio_path(audio_dir, fname)
            if audio_path:
                items.append({**item_base, "file": fname, "audio_path": audio_path})
                break

    return items

# ----------------------------------------------------------------------
# 3) Dataset / Sampler / Collator
# ----------------------------------------------------------------------
class MixedDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {**self.items[idx], "original_idx": idx}

class DistributedHomogeneousBatchSampler(Sampler):
    """
    audio / text を分けた後、rankごとに分割し、batch単位で混ぜる
    """
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
        self.shuffle = shuffle
        self.total_epochs = max(1, int(total_epochs))
        self.epoch = 0

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires torch.distributed")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires torch.distributed")
            rank = dist.get_rank()

        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

        all_audio = [i for i, it in enumerate(dataset.items) if it.get("audio_path") is not None]
        all_text = [i for i, it in enumerate(dataset.items) if it.get("audio_path") is None]

        self.local_audio_indices = all_audio[self.rank :: self.num_replicas]
        self.local_text_indices = all_text[self.rank :: self.num_replicas]

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        g_static = torch.Generator()
        g_static.manual_seed(self.seed)

        audio_tensor = torch.tensor(self.local_audio_indices, dtype=torch.long)
        if self.shuffle and len(audio_tensor) > 0:
            perm = torch.randperm(len(audio_tensor), generator=g_static)
            shuffled_audio = audio_tensor[perm]
        else:
            shuffled_audio = audio_tensor

        total_audio = len(shuffled_audio)
        chunk_size = max(1, total_audio // self.total_epochs)
        chunk_idx = self.epoch % self.total_epochs
        start = chunk_idx * chunk_size
        end = total_audio if chunk_idx == (self.total_epochs - 1) else (start + chunk_size)
        active_audio = shuffled_audio[start:end]

        text_tensor = torch.tensor(self.local_text_indices, dtype=torch.long)

        g_dyn = torch.Generator()
        g_dyn.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            if len(active_audio) > 0:
                active_audio = active_audio[torch.randperm(len(active_audio), generator=g_dyn)]
            if len(text_tensor) > 0:
                text_tensor = text_tensor[torch.randperm(len(text_tensor), generator=g_dyn)]

        audio_idxs = active_audio.tolist()
        text_idxs = text_tensor.tolist()

        batches: List[List[int]] = []
        for i in range(0, len(audio_idxs), self.batch_size):
            b = audio_idxs[i : i + self.batch_size]
            if len(b) == self.batch_size or (not self.drop_last):
                batches.append(b)
        for i in range(0, len(text_idxs), self.batch_size):
            b = text_idxs[i : i + self.batch_size]
            if len(b) == self.batch_size or (not self.drop_last):
                batches.append(b)

        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self) -> int:
        total_audio = len(self.local_audio_indices)
        chunk_size = total_audio // self.total_epochs
        current_audio_len = chunk_size
        current_text_len = len(self.local_text_indices)

        if self.drop_last:
            audio_batches = current_audio_len // self.batch_size
            text_batches = current_text_len // self.batch_size
        else:
            audio_batches = (current_audio_len + self.batch_size - 1) // self.batch_size
            text_batches = (current_text_len + self.batch_size - 1) // self.batch_size

        return audio_batches + text_batches

@dataclass
class SmartCollator:
    processor: Any
    max_length: int = 512
    ignore_index: int = -100
    base_prompt: str = BASE_PROMPT_TEXT

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {}
        is_audio_batch = (batch[0].get("audio_path") is not None)
        if is_audio_batch:
            return self._collate_audio(batch)
        return self._collate_text(batch)

    def _collate_audio(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list, labels_list = [], []
        input_features_list, feature_mask_list = [], []

        sr = self.processor.feature_extractor.sampling_rate
        eos_token = self.processor.tokenizer.eos_token or "<|endoftext|>"

        for item in batch:
            audio_path = item.get("audio_path")
            if not audio_path:
                continue
            audio, _ = librosa.load(audio_path, sr=sr)

            prompt_text = item.get("system_prompt", self.base_prompt)
            user_content = [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": prompt_text},
            ]
            text_input = self.processor.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )

            full_text = text_input + (item.get("target", "") or "") + eos_token

            inputs = self.processor(text=full_text, audio=[audio], sampling_rate=sr, return_tensors="pt")
            prompt_inputs = self.processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")
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
                fmask = inputs["feature_attention_mask"]
                while fmask.dim() > 1:
                    fmask = fmask.squeeze(0)
                feature_mask_list.append(fmask)

        out = {
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
                [torch.ones_like(x) for x in input_ids_list],
                batch_first=True,
                padding_value=0,
            ),
            "input_features": pad_sequence(
                input_features_list,
                batch_first=True,
                padding_value=0.0,
            ),
        }
        out["feature_attention_mask"] = (
            pad_sequence(feature_mask_list, batch_first=True, padding_value=0) if feature_mask_list else None
        )
        return out

    def _collate_text(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list, labels_list = [], []
        eos_token = self.processor.tokenizer.eos_token or "<|endoftext|>"

        for item in batch:
            if item.get("audio_path") is not None:
                continue
            prompt_text = item.get("system_prompt", self.base_prompt)
            transcript = item.get("transcript", "") or ""

            user_content = [{"type": "text", "text": f"{transcript}\n{prompt_text}"}]
            text_input = self.processor.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )

            full_text = text_input + (item.get("target", "") or "") + eos_token

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
                [torch.ones_like(x) for x in input_ids_list],
                batch_first=True,
                padding_value=0,
            ),
        }

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

# ----------------------------------------------------------------------
# 4) Callback: sample generation on eval
# ----------------------------------------------------------------------
class SampleGenerationCallback(TrainerCallback):
    def __init__(self, eval_items, processor, model, num_samples=3):
        self.eval_items = eval_items
        self.processor = processor
        self.model = model
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return

        logger.info("\n\n*** Validation Sample Generation (Audio) ***")
        audio_items = [it for it in self.eval_items if it.get("audio_path") is not None]
        if not audio_items:
            return

        samples = random.sample(audio_items, min(self.num_samples, len(audio_items)))
        device = self.model.device
        self.model.eval()
        sr = self.processor.feature_extractor.sampling_rate

        for item in samples:
            try:
                audio_path = item.get("audio_path")
                if not audio_path:
                    continue
                audio, _ = librosa.load(audio_path, sr=sr)
                prompt_text = item.get("system_prompt", BASE_PROMPT_TEXT)

                user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": prompt_text}]
                text_input = self.processor.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                inputs = self.processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_new_tokens=256)

                input_len = inputs["input_ids"].shape[1]
                generated_text = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

                logger.info("-" * 60)
                logger.info(f"File:       {item.get('file', 'Unknown')}")
                logger.info(f"Target:     {item.get('target', 'N/A')}")
                logger.info(f"Prediction: {generated_text}")
            except Exception as e:
                logger.error(f"Sample generation failed: {e}")

        logger.info("-" * 60 + "\n")
        self.model.train()

# ----------------------------------------------------------------------
# 5) Distributed inference
# ----------------------------------------------------------------------
def run_distributed_inference(
    model,
    processor,
    items: List[Dict[str, Any]],
    output_path: str,
    device: torch.device,
    rank: int,
    world_size: int,
    batch_size: int = 1,
):
    model.eval()
    my_items = items[rank::world_size]
    my_items.sort(key=lambda x: (1 if x.get("audio_path") else 0, x.get("slurp_id", 0)))

    local_results = []
    processor.tokenizer.padding_side = "left"
    sr = processor.feature_extractor.sampling_rate

    if rank == 0:
        logger.info(f"Starting Inference. Items per rank: ~{len(my_items)}")

    for i in range(0, len(my_items), batch_size):
        batch_items = my_items[i : i + batch_size]
        texts, audios = [], []
        has_audio = False

        try:
            for it in batch_items:
                audio_path = it.get("audio_path")
                transcript = it.get("transcript", "")
                prompt_text = it.get("system_prompt", BASE_PROMPT_TEXT)

                if audio_path:
                    has_audio = True
                    audio, _ = librosa.load(audio_path, sr=sr)
                    audios.append(audio)
                    user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": prompt_text}]
                    text_input = processor.apply_chat_template(
                        [{"role": "user", "content": user_content}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    texts.append(text_input)
                else:
                    user_content = [{"type": "text", "text": f"{transcript}\n{prompt_text}"}]
                    text_input = processor.apply_chat_template(
                        [{"role": "user", "content": user_content}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    texts.append(text_input)

            if has_audio:
                inputs = processor(text=texts, audio=audios, sampling_rate=sr, return_tensors="pt", padding=True)
            else:
                inputs = processor(text=texts, return_tensors="pt", padding=True)

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]
            raw_outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for j, raw_output in enumerate(raw_outputs):
                original = batch_items[j]
                json_str = extract_json_from_mixed_output(raw_output)

                parsed_obj = {"scenario": "error", "action": "error", "entities": []}
                try:
                    if json_str:
                        parsed_obj = json.loads(json_str)
                except Exception:
                    pass

                wer = calculate_wer(original.get("transcript", ""), raw_output)
                local_results.append(
                    {
                        "scenario": parsed_obj.get("scenario", ""),
                        "action": parsed_obj.get("action", ""),
                        "entities": parsed_obj.get("entities", []),
                        "file": original.get("file"),
                        "slurp_id": original.get("slurp_id"),
                        "wer": wer,
                        "transcript": original.get("transcript", ""),
                        "raw_output": raw_output,
                        "extracted_json": json_str,
                        "target": original.get("target", ""),
                    }
                )
        except Exception as e:
            logger.error(f"Rank {rank} failed on batch {i}: {e}")

    tmp = f"{output_path}.rank{rank}"
    with open(tmp, "w") as f:
        for r in local_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        with open(output_path, "w") as out:
            for fname in sorted(glob.glob(f"{output_path}.rank*")):
                with open(fname, "r") as inp:
                    shutil.copyfileobj(inp, out)
                os.remove(fname)

# ----------------------------------------------------------------------
# 6) Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--eval_file", type=str, default="slurp/dataset/slurp/devel.jsonl")
    parser.add_argument("--test_file", type=str, default="slurp/dataset/slurp/test.jsonl")
    parser.add_argument("--audio_dir", type=str, default="slurp/slurp_real")
    parser.add_argument("--asr_cache_path", type=str, default="")
    parser.add_argument("--asr_n_best", type=int, default=5)

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_hierarchical_sbert")

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--grad_accum", type=int, default=4)

    # clustering knobs
    parser.add_argument("--scenario_clusters", type=int, default=6)
    parser.add_argument("--action_clusters", type=int, default=15)
    parser.add_argument("--slot_clusters", type=int, default=18)
    parser.add_argument("--intent_clusters", type=int, default=20)
    parser.add_argument("--include_intent_label_group", action="store_true", help="Also show Group for scenario_action label.")

    # compute knobs
    parser.add_argument("--freeze_audio_tower", action="store_true", help="Freeze audio_tower + projector if available.")
    parser.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")

    args = parser.parse_args()

    # DDP setup
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

    # smoke overrides
    if args.smoke:
        args.max_samples = 32
        args.num_train_epochs = 3
        args.eval_steps = 2
        args.logging_steps = 1
        args.save_strategy = "no"

    os.makedirs(args.output_dir, exist_ok=True)

    # cluster cache path
    cluster_cache_path = os.path.join(args.output_dir, "clusters.cache.json")

    # 1) clusters (rank0 build or load + broadcast)
    all_jsonl = [args.train_file, args.eval_file, args.test_file]
    clusters = maybe_load_or_build_clusters(
        cache_path=cluster_cache_path,
        jsonl_paths=all_jsonl,
        n_scenario_clusters=args.scenario_clusters,
        n_action_clusters=args.action_clusters,
        n_slot_clusters=args.slot_clusters,
        n_intent_clusters=args.intent_clusters,
        include_intent_label_group=args.include_intent_label_group,
        rank=rank,
        world_size=world_size,
    )

    # 2) processor + model
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = MODEL_CLS.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True,
    ).to(device)

    if args.freeze_audio_tower:
        # Qwen2-Audio has these; safe-guard for other models
        if hasattr(model, "audio_tower"):
            model.audio_tower.requires_grad_(False)
        if hasattr(model, "multi_modal_projector"):
            model.multi_modal_projector.requires_grad_(False)

    # 3) ASR cache (load/build)
    asr_cache: Dict[str, List[str]] = {}
    if args.asr_cache_path:
        if rank == 0 and (not os.path.exists(args.asr_cache_path)):
            logger.info(f"Building ASR cache (n_best={args.asr_n_best}) -> {args.asr_cache_path}")
            asr_cache = build_asr_cache(
                jsonl_paths=all_jsonl,
                audio_dir=args.audio_dir,
                model=model,
                processor=processor,
                device=device,
                n_best=args.asr_n_best,
            )
            save_asr_cache(args.asr_cache_path, asr_cache)
        if world_size > 1:
            dist.barrier()
        if not asr_cache:
            asr_cache = load_asr_cache(args.asr_cache_path)

    # 4) build dataset items
    train_items = build_items_from_slurp(
        args.train_file,
        args.audio_dir,
        clusters,
        asr_cache=asr_cache,
        add_text_only=True,
        max_samples=args.max_samples,
        include_intent_label_group=args.include_intent_label_group,
    )
    eval_items = build_items_from_slurp(
        args.eval_file,
        args.audio_dir,
        clusters,
        asr_cache=asr_cache,
        add_text_only=True,
        max_samples=(args.max_samples // 2 if args.max_samples else None),
        include_intent_label_group=args.include_intent_label_group,
    )

    if rank == 0:
        logger.info(f"Train items: {len(train_items)} | Eval items: {len(eval_items)}")
        if train_items:
            logger.info("\n--- Target format example ---\n" + train_items[0]["target"] + "\n")

    # 5) training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy if len(eval_items) > 0 else "no",
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        report_to="none",
        remove_unused_columns=False,  # important for multimodal keys
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
        trainer.add_callback(SampleGenerationCallback(eval_items=eval_items, processor=processor, model=model))

    trainer.train()

    # save
    if rank == 0:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    if world_size > 1:
        dist.barrier()

    # 6) inference on test (audio-only by default)
    test_items = build_items_from_slurp(
        args.test_file,
        args.audio_dir,
        clusters,
        asr_cache=asr_cache,
        add_text_only=False,
        max_samples=(500 if args.smoke else None),
        include_intent_label_group=args.include_intent_label_group,
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
        dist.destroy_process_group()

    if rank == 0:
        logger.info(f"Done. Predictions saved to: {output_jsonl}")

if __name__ == "__main__":
    main()
