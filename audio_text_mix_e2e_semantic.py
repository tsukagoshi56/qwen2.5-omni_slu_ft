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
    Qwen2AudioForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist

# --- SBERTのインポート ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
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
    "First, list 3 likely intent candidates based on the audio. "
    "Then, select the correct action and output the final answer as a JSON object."
)

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
    return list(unique_labels)

def build_semantic_graph(all_labels: List[str]) -> Dict[str, List[str]]:
    """SBERTを用いて類似ラベル（Hard Negatives）のグラフを構築"""
    logger.info("Building semantic similarity graph...")
    graph = {}
    
    cleaned_labels = [label.replace("_", " ") for label in all_labels]

    if HAS_SBERT:
        model_name = 'all-MiniLM-L6-v2'
        logger.info(f"Loading SBERT model: {model_name}")
        model = SentenceTransformer(model_name)
        
        embeddings = model.encode(cleaned_labels)
        sim_matrix = cosine_similarity(embeddings)
        
        for i, target in enumerate(all_labels):
            scores = sim_matrix[i]
            candidates = []
            for j, score in enumerate(scores):
                if i == j: continue
                candidates.append((all_labels[j], score))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            graph[target] = [c[0] for c in candidates[:10]]
            
    else:
        logger.warning("sentence-transformers not found. Falling back to simple rule-based matching.")
        for target in all_labels:
            parts = target.split('_')
            t_s, t_a = parts[0], parts[1:]
            candidates = []
            for other in all_labels:
                if other == target: continue
                o_parts = other.split('_')
                score = 0
                if t_s == o_parts[0]: score += 10
                if t_a == o_parts[1:]: score += 5
                if score > 0: candidates.append((other, score))
            candidates.sort(key=lambda x: x[1], reverse=True)
            graph[target] = [c[0] for c in candidates]

    return graph

def build_items_from_slurp(jsonl_path, audio_dir, all_labels, semantic_graph, add_text_only=True, max_samples=None):
    items = []
    if not os.path.exists(jsonl_path): return items
    
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if max_samples is not None and len(items) >= max_samples: break
        data = json.loads(line)
        slurp_id = data.get("slurp_id", -1)
        
        current_label = f"{data.get('scenario')}_{data.get('action')}"
        
        # --- Semantic Candidates Logic ---
        rng = random.Random(slurp_id)
        
        distractors = []
        similar_candidates = semantic_graph.get(current_label, [])
        
        if len(similar_candidates) >= 2:
            pool = similar_candidates[:5] 
            distractors = rng.sample(pool, 2)
        elif len(similar_candidates) == 1:
            distractors = [similar_candidates[0]]
        
        while len(distractors) < 2:
            fallback = rng.choice(all_labels)
            if fallback != current_label and fallback not in distractors:
                distractors.append(fallback)

        candidates = [current_label] + distractors
        rng.shuffle(candidates)
        
        cand_str = ", ".join([f"'{c}'" for c in candidates])

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
        
        # Chain-of-Generation Target
        full_target_str = (
            f"Candidates: [{cand_str}]\n"
            f"Answer: {json_str}"
        )
        
        transcript = data.get("sentence", "")
        item_base = {
            "slurp_id": slurp_id,
            "transcript": transcript,
            "target": full_target_str,
            "system_prompt": BASE_PROMPT_TEXT
        }
        
        if add_text_only:
            items.append({**item_base, "file": None, "audio_path": None})
        if data.get("recordings"):
            for rec in data["recordings"]:
                path = resolve_audio_path(audio_dir, rec.get("file", ""))
                if path:
                    items.append({**item_base, "file": rec.get("file"), "audio_path": path})
                    break 
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
        is_audio_batch = (batch[0].get("audio_path") is not None)
        if is_audio_batch: return self._collate_audio(batch)
        else: return self._collate_text(batch)

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

        logger.info("\n\n*** Validation Sample Generation (Audio) ***")
        audio_items = [item for item in self.eval_items if item.get("audio_path") is not None]
        if not audio_items: return

        samples = random.sample(audio_items, min(self.num_samples, len(audio_items)))
        device = self.model.device
        self.model.eval()
        sr = self.processor.feature_extractor.sampling_rate

        for item in samples:
            try:
                audio_path = item.get("audio_path")
                if not audio_path: continue
                audio, _ = librosa.load(audio_path, sr=sr)
                prompt_text = item.get("system_prompt", BASE_PROMPT_TEXT)
                
                user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": prompt_text}]
                text_input = self.processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
                
                inputs = self.processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_new_tokens=256)
                
                input_len = inputs["input_ids"].shape[1]
                generated_text = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
                
                logger.info(f"-" * 60)
                logger.info(f"File:       {item.get('file', 'Unknown')}")
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
    sr = processor.feature_extractor.sampling_rate

    for i in range(0, len(my_items), batch_size):
        batch_items = my_items[i : i + batch_size]
        if rank == 0 and i % 10 == 0: logger.info(f"Processing {i}/{len(my_items)}...")
        texts, audios = [], []
        has_audio = False

        try:
            for item in batch_items:
                audio_path = item.get("audio_path")
                transcript = item.get("transcript", "")
                prompt_text = item.get("system_prompt", BASE_PROMPT_TEXT)
                if audio_path:
                    has_audio = True
                    audio, _ = librosa.load(audio_path, sr=sr)
                    audios.append(audio)
                    user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": prompt_text}]
                    text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
                    texts.append(text_input)
                else:
                    user_content = [{"type": "text", "text": f"{transcript}\n{prompt_text}"}]
                    text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
                    texts.append(text_input)

            if has_audio: inputs = processor(text=texts, audio=audios, sampling_rate=sr, return_tensors="pt", padding=True)
            else: inputs = processor(text=texts, return_tensors="pt", padding=True)
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

                wer_score = calculate_wer(original_item.get("transcript", ""), raw_output)
                result_entry = {
                    "scenario": parsed_obj.get("scenario", ""),
                    "action": parsed_obj.get("action", ""),
                    "entities": parsed_obj.get("entities", []),
                    "file": original_item.get("file"),
                    "slurp_id": original_item.get("slurp_id"),
                    "wer": wer_score,
                    "transcript": original_item.get("transcript", ""),
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

    # 1. Collect & SBERT
    all_jsonl = [args.train_file, args.eval_file, args.test_file]
    all_labels = collect_all_labels(all_jsonl)
    semantic_graph = build_semantic_graph(all_labels)

    # 2. Build Dataset
    train_items = build_items_from_slurp(args.train_file, args.audio_dir, all_labels, semantic_graph, max_samples=args.max_samples)
    eval_items = build_items_from_slurp(args.eval_file, args.audio_dir, all_labels, semantic_graph, max_samples=args.max_samples // 2 if args.max_samples else None)
    if rank == 0: print(f"Train: {len(train_items)} | Eval: {len(eval_items)}")

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.audio_tower.requires_grad_(False)
    model.multi_modal_projector.requires_grad_(False)

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
        args.test_file, args.audio_dir, all_labels, semantic_graph,
        max_samples=test_max_samples, add_text_only=False 
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