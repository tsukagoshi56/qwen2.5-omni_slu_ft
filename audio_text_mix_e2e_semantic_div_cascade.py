#!/usr/bin/env python3
"""
Audio-Text Mix Training & Distributed Inference (Slurp Format)
Output-Side Semantic Candidates Ablation (SBERT Enhanced & Fully Robust)
"""

import argparse
import json
import os
import random
import re
import torch
import numpy as np
import logging
import glob
import shutil
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Iterator
import librosa

from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist

try:
    from transformers import Qwen2AudioForConditionalGeneration
    MODEL_CLS = Qwen2AudioForConditionalGeneration
except Exception:
    from transformers import AutoModelForCausalLM
    MODEL_CLS = AutoModelForCausalLM

# --- SBERTのインポート ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt
BASE_PROMPT_TEXT = (
    "You are a voice assistant. Analyze the user's spoken request. "
    "First, identify the semantic group of the intent. "
    "Then, select the specific intent from that group. "
    "Finally, output the answer as a JSON object."
)
ASR_PROMPT_TEXT = "Transcribe the audio."

# ==============================================================================
# 1. Data Loading & SBERT Semantic Graph
# ==============================================================================

def resolve_audio_path(audio_root: str, filename: str) -> Optional[str]:
    candidates = [
        os.path.join(audio_root, filename),
        os.path.join(audio_root, "slurp_real", filename),
        os.path.join("slurp", "audio", "slurp_real", filename),
    ]
    for path in candidates:
        if os.path.exists(path): return path
    return None

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

def ddp_broadcast_object(obj: Any, rank: int, world_size: int) -> Any:
    if world_size <= 1:
        return obj
    backend = dist.get_backend()
    if backend == "nccl":
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
        except Exception:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    obj_list = [obj] if rank == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0, device=device)
    return obj_list[0]

def collect_all_labels(jsonl_paths: List[str]) -> List[str]:
    unique_labels = set()
    logger.info("Scanning datasets to collect all unique labels...")
    for path in jsonl_paths:
        if not os.path.exists(path): continue
        with open(path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    s = data.get("scenario")
                    a = data.get("action")
                    if s and a: unique_labels.add(f"{s}_{a}")
                except: pass
    # 修正：必ずソートしてリストを返す（全プロセスで順序を一致させる）
    return sorted(list(unique_labels))

def build_hierarchical_groups(all_labels: List[str], n_clusters: int = 20, rank: int = 0) -> Dict[str, List[str]]:
    """
    SBERTでラベルをベクトル化し、K-Meansでクラスタリングしてグループを作成する。
    Returns: label_to_group dict mapping each label to its cluster members
    """
    if rank == 0:
        logger.info(f"Clustering {len(all_labels)} labels into {n_clusters} groups...")
    
    if not HAS_SBERT:
        raise ImportError("Clustering requires sentence-transformers. Please install it.")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    cleaned_labels = [label.replace("_", " ") for label in all_labels]
    embeddings = model.encode(cleaned_labels)

    # クラスタリングの実行
    # n_clustersはラベル総数に応じて調整（例: ラベル数の1/4〜1/5程度）
    # random_state固定により計算結果を全GPUで一致させる
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # クラスタID -> [ラベルリスト] の辞書を作成
    groups = {}
    for label, cluster_id in zip(all_labels, cluster_labels):
        cid = str(cluster_id)
        if cid not in groups:
            groups[cid] = []
        groups[cid].append(label)
    
    # 修正：グループ内のラベルもソートして、プロンプト内の並び順の決定論性を高める
    label_to_group = {label: sorted(groups[str(cid)]) for label, cid in zip(all_labels, cluster_labels)}
    
    # 修正：ログ出力を Rank 0 のみに制限し、全クラスタサイズを表示
    if rank == 0:
        all_sizes = [len(g) for g in groups.values()]
        logger.info(f"Successfully created {len(groups)} clusters.")
        logger.info(f"Cluster sizes: {all_sizes}")
    
    return label_to_group

def build_items_from_slurp(
    jsonl_path,
    audio_dir,
    label_to_group,
    asr_cache: Optional[Dict[str, List[str]]] = None,
    asr_n_best: int = 1,
    add_text_only: bool = True,
    max_samples: Optional[int] = None,
):
    items = []
    if not os.path.exists(jsonl_path): return items

    with open(jsonl_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if max_samples is not None and len(items) >= max_samples: break
        data = json.loads(line)
        slurp_id = data.get("slurp_id", -1)

        current_label = f"{data.get('scenario')}_{data.get('action')}"
        
        # --- Hierarchical Clustering Logic ---
        # label_to_group は build_hierarchical_groups 内で既に sorted 済みです
        group_members = label_to_group.get(current_label, [current_label])
        
        # 修正：シャッフルを廃止し、常に同じ順序（アルファベット順）で提示する
        # これにより、同じグループに属するデータはすべて同じ Group 文字列を持つことになります
        group_str = ", ".join([f"'{m}'" for m in group_members])

        # Entity Processing
        raw_entities = data.get("entities", [])
        tokens = data.get("tokens", [])
        processed_entities = []
        for e in raw_entities:
            filler = e.get("filler")
            if not filler and "span" in e and tokens:
                try:
                    span = e["span"]
                    if isinstance(span, list) and len(span) > 0:
                        selected = tokens[span[0]:span[-1]+1]
                        filler = " ".join([t.get("surface", "") for t in selected])
                except: pass
            if filler is None: filler = ""
            processed_entities.append({"type": e.get("type"), "filler": filler})

        # JSON Object
        json_obj = {
            "scenario": data.get("scenario", ""),
            "action": data.get("action", ""),
            "entities": processed_entities
        }
        json_str = json.dumps(json_obj, ensure_ascii=False)
        
        # 2. 階層的ターゲット文字列の構築
        # Group -> Selection -> Answer の順に推論させる
        full_target_str = (
            f"Group: [{group_str}]\n"
            f"Selection: {current_label}\n"
            f"Answer: {json_str}"
        )
        
        reference_transcript = data.get("sentence", "")
        asr_list: List[str] = []
        if asr_cache is not None:
            asr_list = asr_cache.get(str(slurp_id), [])
        if asr_list:
            asr_list = asr_list[: max(1, int(asr_n_best))]
        else:
            asr_list = [reference_transcript] if reference_transcript else []
        if not asr_list:
            continue

        file_name = None
        for rec in data.get("recordings", []) or []:
            if isinstance(rec, dict):
                fname = rec.get("file")
                if fname:
                    file_name = fname
                    break

        if add_text_only:
            for idx, asr_text in enumerate(asr_list):
                item_base = {
                    "slurp_id": slurp_id,
                    "transcript": asr_text,
                    "reference_transcript": reference_transcript,
                    "asr_rank": idx,
                    "asr_n_best": len(asr_list),
                    "target": full_target_str,
                    "system_prompt": BASE_PROMPT_TEXT
                }
                items.append({**item_base, "file": file_name, "audio_path": None})
    return items

class MixedDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return {**self.items[idx], "original_idx": idx}

# ==============================================================================
# 2. Sampler
# ==============================================================================
class DistributedHomogeneousBatchSampler(Sampler):
    def __init__(self, dataset: MixedDataset, batch_size: int, 
                 num_replicas: Optional[int] = None, rank: Optional[int] = None, 
                 drop_last: bool = False, seed: int = 0, shuffle: bool = True, total_epochs: int = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.shuffle = shuffle
        self.total_epochs = total_epochs
        if num_replicas is None:
            if not torch.distributed.is_available(): raise RuntimeError("Requires distributed package")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available(): raise RuntimeError("Requires distributed package")
            rank = torch.distributed.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        
        all_audio = [i for i, item in enumerate(dataset.items) if item.get("audio_path") is not None]
        all_text = [i for i, item in enumerate(dataset.items) if item.get("audio_path") is None]
        
        self.local_audio_indices = all_audio[self.rank::self.num_replicas]
        self.local_text_indices = all_text[self.rank::self.num_replicas]

    def __iter__(self) -> Iterator[List[int]]:
        g_static = torch.Generator()
        g_static.manual_seed(self.seed)
        audio_indices_tensor = torch.tensor(self.local_audio_indices)
        if self.shuffle:
            perm_static = torch.randperm(len(audio_indices_tensor), generator=g_static)
            shuffled_audio = audio_indices_tensor[perm_static]
        else: shuffled_audio = audio_indices_tensor

        total_audio_count = len(shuffled_audio)
        actual_epochs = max(1, self.total_epochs)
        chunk_size = max(1, total_audio_count // actual_epochs)
        current_chunk_idx = self.epoch % actual_epochs
        start_idx = current_chunk_idx * chunk_size
        if current_chunk_idx == actual_epochs - 1: end_idx = total_audio_count
        else: end_idx = start_idx + chunk_size
            
        active_audio_indices = shuffled_audio[start_idx:end_idx]
        active_text_indices = torch.tensor(self.local_text_indices)

        g_dynamic = torch.Generator()
        g_dynamic.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            audio_perm = torch.randperm(len(active_audio_indices), generator=g_dynamic)
            text_perm = torch.randperm(len(active_text_indices), generator=g_dynamic)
            audio_idxs = active_audio_indices[audio_perm].tolist()
            text_idxs = active_text_indices[text_perm].tolist()
        else:
            audio_idxs = active_audio_indices.tolist()
            text_idxs = active_text_indices.tolist()
        
        batches = []
        for i in range(0, len(audio_idxs), self.batch_size):
            batch = audio_idxs[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last: batches.append(batch)
        for i in range(0, len(text_idxs), self.batch_size):
            batch = text_idxs[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last: batches.append(batch)
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(batches)
        for batch in batches: yield batch

    def __len__(self):
        total_audio = len(self.local_audio_indices)
        actual_epochs = max(1, self.total_epochs)
        chunk_size = total_audio // actual_epochs
        current_audio_len = chunk_size 
        current_text_len = len(self.local_text_indices)
        audio_batches = (current_audio_len + self.batch_size - 1) // self.batch_size
        text_batches = (current_text_len + self.batch_size - 1) // self.batch_size
        if self.drop_last:
            audio_batches = current_audio_len // self.batch_size
            text_batches = current_text_len // self.batch_size
        return audio_batches + text_batches
    def set_epoch(self, epoch: int): self.epoch = epoch

# ==============================================================================
# 3. Collator
# ==============================================================================
@dataclass
class SmartCollator:
    processor: Any
    max_length: int = 512
    ignore_index: int = -100
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        if len(batch) == 0: return {}
        return self._collate_text(batch)

    def _collate_audio(self, batch):
        input_ids_list, labels_list, input_features_list, feature_mask_list = [], [], [], []
        sr = self.processor.feature_extractor.sampling_rate
        eos_token = self.processor.tokenizer.eos_token or "<|endoftext|>"

        for item in batch:
            if item.get("audio_path") is None: continue 
            audio, _ = librosa.load(item["audio_path"], sr=sr)
            prompt_text = item.get("system_prompt", BASE_PROMPT_TEXT)
            user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": prompt_text}]
            
            text_input = self.processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            
            # 【修正】 .get("target", "") を使用
            full_text = text_input + item.get("target", "") + eos_token
            
            inputs = self.processor(text=full_text, audio=[audio], sampling_rate=sr, return_tensors="pt")
            prompt_inputs = self.processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            ids = inputs["input_ids"][0]
            lbs = ids.clone()
            lbs[:prompt_len] = self.ignore_index
            
            input_ids_list.append(ids)
            labels_list.append(lbs)
            feat = inputs["input_features"]
            while feat.dim() > 2: feat = feat.squeeze(0)
            input_features_list.append(feat)
            if "feature_attention_mask" in inputs:
                f_mask = inputs["feature_attention_mask"]
                while f_mask.dim() > 1: f_mask = f_mask.squeeze(0)
                feature_mask_list.append(f_mask)
        
        return {
            "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index),
            "attention_mask": pad_sequence([torch.ones_like(ids) for ids in input_ids_list], batch_first=True, padding_value=0),
            "input_features": pad_sequence(input_features_list, batch_first=True, padding_value=0.0),
            "feature_attention_mask": pad_sequence(feature_mask_list, batch_first=True, padding_value=0) if feature_mask_list else None
        }

    def _collate_text(self, batch):
        input_ids_list, labels_list = [], []
        eos_token = self.processor.tokenizer.eos_token or "<|endoftext|>"

        for item in batch:
            if item.get("audio_path") is not None: continue
            prompt_text = item.get("system_prompt", BASE_PROMPT_TEXT)
            
            # 【修正】 .get("transcript", "") を使用
            transcript = item.get("transcript", "")
            user_content = [{"type": "text", "text": f"{transcript}\n{prompt_text}"}]
            
            text_input = self.processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            
            # 【修正】 .get("target", "") を使用
            full_text = text_input + item.get("target", "") + eos_token
            
            inputs = self.processor.tokenizer(full_text, return_tensors="pt")
            prompt_inputs = self.processor.tokenizer(text_input, return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            ids = inputs["input_ids"][0]
            lbs = ids.clone()
            lbs[:prompt_len] = self.ignore_index
            input_ids_list.append(ids)
            labels_list.append(lbs)
            
        return {
            "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index),
            "attention_mask": pad_sequence([torch.ones_like(ids) for ids in input_ids_list], batch_first=True, padding_value=0),
        }

class CustomTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None: raise ValueError("Trainer: training requires a train_dataset.")
        batch_sampler = DistributedHomogeneousBatchSampler(
            self.train_dataset, batch_size=self.args.train_batch_size,
            num_replicas=self.args.world_size, rank=self.args.process_index,
            drop_last=self.args.dataloader_drop_last, shuffle=True, total_epochs=int(self.args.num_train_epochs)
        )
        return DataLoader(
            self.train_dataset, batch_sampler=batch_sampler,
            collate_fn=self.data_collator, num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# ==============================================================================
# 5. Callbacks
# ==============================================================================
class SampleGenerationCallback(TrainerCallback):
    def __init__(self, eval_items, processor, model, num_samples=3):
        self.eval_items = eval_items
        self.processor = processor
        self.model = model
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, **kwargs):
        if args.process_index != 0: return

        logger.info("\n\n*** Validation Sample Generation (Text) ***")
        text_items = [item for item in self.eval_items if item.get("transcript") is not None]
        if not text_items: return

        samples = random.sample(text_items, min(self.num_samples, len(text_items)))
        device = self.model.device
        self.model.eval()

        for item in samples:
            try:
                prompt_text = item.get("system_prompt", BASE_PROMPT_TEXT)
                transcript = item.get("transcript", "")

                user_content = [{"type": "text", "text": f"{transcript}\n{prompt_text}"}]
                text_input = self.processor.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                inputs = self.processor.tokenizer(text_input, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_new_tokens=256)
                
                input_len = inputs["input_ids"].shape[1]
                generated_text = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
                
                logger.info(f"-" * 60)
                logger.info(f"File:       {item.get('file', 'Unknown')}")
                logger.info(f"ASR Rank:   {item.get('asr_rank', 'N/A')}")
                logger.info(f"ASR Text:   {item.get('transcript', '')}")
                # 【修正】 .get("target", "") を使用
                logger.info(f"Target:     {item.get('target', 'N/A')}")
                logger.info(f"Prediction: {generated_text}")
            except Exception as e:
                logger.error(f"Sample generation failed: {e}")
        logger.info("-" * 60 + "\n")
        self.model.train()

# ==============================================================================
# 6. Inference
# ==============================================================================
def extract_json_from_mixed_output(text: str) -> str:
    text = text.strip()
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match: return match.group(1)
    if "Answer:" in text:
        parts = text.split("Answer:", 1)
        if len(parts) > 1:
            potential = parts[1].strip()
            match = re.search(r'(\{.*\})', potential, re.DOTALL)
            if match: return match.group(1)
            return potential
    match = re.findall(r'(\{.*\})', text, re.DOTALL)
    if match: return match[-1]
    return text

def calculate_wer(reference: str, hypothesis: str) -> float:
    if not reference: return 0.0
    if HAS_JIWER: return jiwer.wer(reference, hypothesis)
    else:
        ref = reference.split(); hyp = hypothesis.split()
        return 1.0 if ref != hyp else 0.0

def run_distributed_inference(model, processor, items, output_path, device, rank, world_size, batch_size=1):
    model.eval()
    my_items = items[rank::world_size]
    # 【修正】 ソート時も .get() を使用
    my_items.sort(key=lambda x: (1 if x.get("audio_path") else 0, x.get("slurp_id", 0)))
    
    local_results = []
    processor.tokenizer.padding_side = "left"
    if rank == 0: logger.info(f"Starting Inference. Items: {len(my_items)}")

    for i in range(0, len(my_items), batch_size):
        batch_items = my_items[i : i + batch_size]
        if rank == 0 and i % 10 == 0: logger.info(f"Processing {i}/{len(my_items)}...")
        texts = []

        try:
            for item in batch_items:
                transcript = item.get("transcript", "")
                prompt_text = item.get("system_prompt", BASE_PROMPT_TEXT)
                user_content = [{"type": "text", "text": f"{transcript}\n{prompt_text}"}]
                text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
                texts.append(text_input)

            inputs = processor.tokenizer(texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=256, use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
            
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]
            raw_outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for j, raw_output in enumerate(raw_outputs):
                original_item = batch_items[j]
                json_str = extract_json_from_mixed_output(raw_output)
                parsed_obj = {"scenario": "error", "action": "error", "entities": []}
                try:
                    if json_str: parsed_obj = json.loads(json_str)
                except: pass

                reference_transcript = original_item.get("reference_transcript", original_item.get("transcript", ""))
                asr_transcript = original_item.get("transcript", "")
                wer_score = calculate_wer(reference_transcript, asr_transcript)
                result_entry = {
                    "scenario": parsed_obj.get("scenario", ""),
                    "action": parsed_obj.get("action", ""),
                    "entities": parsed_obj.get("entities", []),
                    "file": original_item.get("file"),
                    "slurp_id": original_item.get("slurp_id"),
                    "wer": wer_score,
                    "transcript": asr_transcript,
                    "reference_transcript": reference_transcript,
                    "asr_rank": original_item.get("asr_rank"),
                    "asr_n_best": original_item.get("asr_n_best"),
                    "raw_output": raw_output,
                    "extracted_json": json_str,
                    # 【修正】 .get("target", "") を使用
                    "target": original_item.get("target", "")
                }
                local_results.append(result_entry)
        except Exception as e: logger.error(f"Rank {rank} failed on batch {i}: {e}")

    temp_output_path = f"{output_path}.rank{rank}"
    with open(temp_output_path, "w") as f:
        for res in local_results: f.write(json.dumps(res, ensure_ascii=False) + "\n")
    if world_size > 1: dist.barrier()
    if rank == 0:
        with open(output_path, "w") as outfile:
            for fname in glob.glob(f"{output_path}.rank*"):
                with open(fname, "r") as infile: shutil.copyfileobj(infile, outfile)
                os.remove(fname)

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
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_output_sbert")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test.")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank(); world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0; world_size = 1; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke: args.max_samples = 32; args.num_train_epochs = 3; args.report_to = "none"
    if args.eval_file is None: args.eval_file = args.train_file.replace("train.jsonl", "devel.jsonl")
    if args.test_file is None: args.test_file = args.train_file.replace("train.jsonl", "test.jsonl")

    # 1. Collect & Hierarchical Clustering
    all_jsonl = [args.train_file, args.eval_file, args.test_file]
    all_labels = collect_all_labels(all_jsonl)
    # 修正：rank を引数に追加してログ制御
    label_to_group = build_hierarchical_groups(all_labels, n_clusters=20, rank=rank)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = MODEL_CLS.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    if hasattr(model, "audio_tower"):
        model.audio_tower.requires_grad_(False)
    if hasattr(model, "multi_modal_projector"):
        model.multi_modal_projector.requires_grad_(False)

    # 2. Build ASR cache (audio -> n-best transcripts)
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
    else:
        if rank == 0:
            logger.info(f"Building ASR cache in-memory (n_best={args.asr_n_best})")
            asr_cache = build_asr_cache(
                jsonl_paths=all_jsonl,
                audio_dir=args.audio_dir,
                model=model,
                processor=processor,
                device=device,
                n_best=args.asr_n_best,
            )
        asr_cache = ddp_broadcast_object(asr_cache, rank, world_size)

    # 3. Build Dataset (ASR text only)
    train_items = build_items_from_slurp(
        args.train_file,
        args.audio_dir,
        label_to_group,
        asr_cache=asr_cache,
        asr_n_best=args.asr_n_best,
        max_samples=args.max_samples,
    )
    eval_items = build_items_from_slurp(
        args.eval_file,
        args.audio_dir,
        label_to_group,
        asr_cache=asr_cache,
        asr_n_best=args.asr_n_best,
        max_samples=args.max_samples // 2 if args.max_samples else None,
    )
    if rank == 0:
        print(f"Train: {len(train_items)} | Eval: {len(eval_items)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4, learning_rate=args.learning_rate, bf16=True,
        logging_steps=1 if args.smoke else 10, eval_strategy="steps" if len(eval_items) > 0 else "no",
        eval_steps=2 if args.smoke else 50, save_strategy="no", report_to="none"
    )

    trainer = CustomTrainer(
        model=model, args=training_args,
        train_dataset=MixedDataset(train_items), eval_dataset=MixedDataset(eval_items) if len(eval_items) > 0 else None,
        data_collator=SmartCollator(processor), tokenizer=processor.tokenizer,
    )
    if len(eval_items) > 0:
        trainer.add_callback(SampleGenerationCallback(eval_items=eval_items, processor=processor, model=model))

    trainer.train()

    if rank == 0:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
    if world_size > 1: dist.barrier()

    test_max_samples = 500 if args.smoke else None
    test_items = build_items_from_slurp(
        args.test_file,
        args.audio_dir,
        label_to_group,
        asr_cache=asr_cache,
        asr_n_best=args.asr_n_best,
        max_samples=test_max_samples,
        add_text_only=True,
    )
    output_jsonl = os.path.join(args.output_dir, "prediction.jsonl")
    run_distributed_inference(
        model=model, processor=processor, items=test_items,
        output_path=output_jsonl, device=device, rank=rank,
        world_size=world_size, batch_size=args.batch_size
    )   
    if world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    main()
