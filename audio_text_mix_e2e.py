#!/usr/bin/env python3
"""
Audio-Text Mix Training (Distributed Homogeneous Batching - FIXED)
=========================================================
Fixes:
- Applies Homogeneous Batching to EVALUATION as well (prevents crashes).
-Clarifies Test data loading.
"""

import argparse
import json
import os
import random
import torch
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Iterator
import librosa

from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT = """You are a voice assistant. Analyze the user's spoken request and output a JSON object with:
- "scenario": the general intent category
- "action": the specific action within that scenario
- "entities": a list of extracted entities as {"type": ..., "filler": ...}

Output only valid JSON, no extra text."""

# ==============================================================================
# 1. Data Loading
# ==============================================================================

def resolve_audio_path(audio_root: str, filename: str) -> Optional[str]:
    candidates = [
        os.path.join(audio_root, filename),
        os.path.join(audio_root, "slurp_real", filename),
        os.path.join("slurp", "audio", "slurp_real", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def build_items_from_slurp(jsonl_path, audio_dir, add_text_only=True, max_samples=None):
    items = []
    if not os.path.exists(jsonl_path):
        return items
    
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if max_samples and len(items) >= max_samples:
            break
        data = json.loads(line)
        target = json.dumps({
            "scenario": data.get("scenario", ""),
            "action": data.get("action", ""),
            "entities": [{"type": e.get("type"), "filler": e.get("filler")} for e in data.get("entities", [])]
        }, ensure_ascii=False)
        transcript = data.get("sentence", "")
        
        # Text Item
        if add_text_only:
            items.append({"audio_path": None, "transcript": transcript, "target": target})
        
        # Audio Item
        if data.get("recordings"):
            path = resolve_audio_path(audio_dir, data["recordings"][0].get("file", ""))
            if path:
                items.append({"audio_path": path, "transcript": transcript, "target": target})
    return items

class MixedDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return {**self.items[idx], "original_idx": idx}

# ==============================================================================
# 2. Distributed Homogeneous Batch Sampler
# ==============================================================================

class DistributedHomogeneousBatchSampler(Sampler):
    def __init__(self, dataset: MixedDataset, batch_size: int, 
                 num_replicas: Optional[int] = None, 
                 rank: Optional[int] = None, 
                 drop_last: bool = False,
                 seed: int = 0,
                 shuffle: bool = True): # Added shuffle flag
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.shuffle = shuffle

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

        all_audio = [i for i, item in enumerate(dataset.items) if item["audio_path"] is not None]
        all_text = [i for i, item in enumerate(dataset.items) if item["audio_path"] is None]

        # Split for DDP
        self.local_audio_indices = all_audio[self.rank::self.num_replicas]
        self.local_text_indices = all_text[self.rank::self.num_replicas]

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        audio_indices = torch.tensor(self.local_audio_indices)
        text_indices = torch.tensor(self.local_text_indices)
        
        if self.shuffle:
            audio_perm = torch.randperm(len(audio_indices), generator=g)
            text_perm = torch.randperm(len(text_indices), generator=g)
            audio_idxs = audio_indices[audio_perm].tolist()
            text_idxs = text_indices[text_perm].tolist()
        else:
            audio_idxs = audio_indices.tolist()
            text_idxs = text_indices.tolist()
        
        batches = []
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

    def __len__(self):
        audio_len = len(self.local_audio_indices)
        text_len = len(self.local_text_indices)
        audio_batches = (audio_len + self.batch_size - 1) // self.batch_size
        text_batches = (text_len + self.batch_size - 1) // self.batch_size
        if self.drop_last:
            audio_batches = audio_len // self.batch_size
            text_batches = text_len // self.batch_size
        return audio_batches + text_batches
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch

# ==============================================================================
# 3. Smart Collator
# ==============================================================================

@dataclass
class SmartCollator:
    processor: Any
    max_length: int = 512
    ignore_index: int = -100
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Critical: Check modality. If mixed, this logic fails. Sampler prevents mixing.
        if len(batch) == 0: return {}
        is_audio_batch = (batch[0].get("audio_path") is not None)
        
        if is_audio_batch:
            return self._collate_audio(batch)
        else:
            return self._collate_text(batch)

    def _collate_audio(self, batch):
        input_ids_list, labels_list, input_features_list, feature_mask_list = [], [], [], []
        sr = self.processor.feature_extractor.sampling_rate
        
        for item in batch:
            if item["audio_path"] is None: 
                # Safety fallback just in case
                continue 
            audio, _ = librosa.load(item["audio_path"], sr=sr)
            user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": PROMPT}]
            text_input = self.processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            full_text = text_input + item["target"]
            
            inputs = self.processor(text=full_text, audio=[audio], return_tensors="pt")
            prompt_inputs = self.processor(text=text_input, audio=[audio], return_tensors="pt")
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
        for item in batch:
            if item["audio_path"] is not None: continue # Safety fallback
            user_content = [{"type": "text", "text": f"{item['transcript']}\n{PROMPT}"}]
            text_input = self.processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            full_text = text_input + item["target"]
            
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

# ==============================================================================
# 4. Custom Trainer with EVAL Support
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
            shuffle=True
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Overridden to use Homogeneous Batching during Evaluation too.
        Without this, standard random sampler mixes Audio/Text and crashes SmartCollator.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # Use drop_last=False for eval so we don't lose data
        # shuffle=False is usually better for eval determinism, but 
        # mixing audio/text batches order is fine as long as batches are pure.
        batch_sampler = DistributedHomogeneousBatchSampler(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            drop_last=False, 
            shuffle=False 
        )

        return DataLoader(
            eval_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# ==============================================================================
# 5. Final Evaluation Logic
# ==============================================================================

def evaluate_model(model, processor, items, device):
    """
    Runs simple inference on Rank 0 only.
    """
    model.eval()
    results = []
    print(f"Evaluating on {len(items)} items (Rank 0)...")
    
    for i, item in enumerate(items):
        audio_path = item.get("audio_path")
        transcript = item.get("transcript", "")
        
        if audio_path:
            # Load audio
            audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
            user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": PROMPT}]
            text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_input, audio=[audio], return_tensors="pt")
        else:
            user_content = [{"type": "text", "text": f"{transcript}\n{PROMPT}"}]
            text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_input, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128)
            
        input_len = inputs["input_ids"].shape[1]
        decoded = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        
        results.append({"target": item["target"], "prediction": decoded, "type": "audio" if audio_path else "text"})
        
        if i < 3:
            print(f"[{results[-1]['type'].upper()}] Pred: {decoded} | Target: {item['target']}")

    acc = sum(1 for r in results if r["target"].strip() == r["prediction"].strip()) / len(results)
    print(f"\nFinal Accuracy: {acc:.2%}")

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--eval_file", type=str, default=None, help="If not set, uses train_file path replaced with devel.jsonl")
    parser.add_argument("--test_file", type=str, default=None, help="If not set, uses train_file path replaced with test.jsonl")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio/slurp_real")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_smart_batch_ddp")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device = torch.device(f"cuda:{local_rank}") if local_rank != -1 else "cuda"

    # --- Data Paths ---
    if args.eval_file is None:
        args.eval_file = args.train_file.replace("train.jsonl", "devel.jsonl")
    if args.test_file is None:
        args.test_file = args.train_file.replace("train.jsonl", "test.jsonl")

    # --- Load Data ---
    train_items = build_items_from_slurp(args.train_file, args.audio_dir, max_samples=args.max_samples)
    eval_items = build_items_from_slurp(args.eval_file, args.audio_dir, max_samples=args.max_samples // 10 if args.max_samples else None)
    
    if local_rank in [-1, 0]:
        print(f"Train: {len(train_items)} | Eval: {len(eval_items)}")
        if len(eval_items) == 0:
            print(f"WARNING: Eval file {args.eval_file} not found or empty.")

    # --- Model & Processor ---
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.audio_tower.requires_grad_(False)
    model.multi_modal_projector.requires_grad_(False)

    # --- Training ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps" if len(eval_items) > 0 else "no",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none"
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=MixedDataset(train_items),
        eval_dataset=MixedDataset(eval_items) if len(eval_items) > 0 else None,
        data_collator=SmartCollator(processor),
        tokenizer=processor.tokenizer,
    )

    trainer.train()

    # --- Final Test (Rank 0 Only) ---
    if local_rank in [-1, 0]:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        
        print("\n=== Running Final Test on Rank 0 ===")
        test_items = build_items_from_slurp(args.test_file, args.audio_dir, max_samples=100)
        if test_items:
            # Note: We use the raw model (or unwrapped) for generation to handle simple inference
            evaluate_model(model, processor, test_items, device)
        else:
            print(f"Test file {args.test_file} not found.")

    # Wait for Rank 0 to finish testing before killing other processes
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dist.barrier()

if __name__ == "__main__":
    main()