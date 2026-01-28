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
        slurp_id = data.get("slurp_id")
        
        # Text Item
        if add_text_only:
            items.append({
                "audio_path": None, 
                "transcript": transcript, 
                "target": target,
                "slurp_id": slurp_id
            })
        
        # Audio Item
        if data.get("recordings"):
            path = resolve_audio_path(audio_dir, data["recordings"][0].get("file", ""))
            if path:
                items.append({
                    "audio_path": path, 
                    "transcript": transcript, 
                    "target": target,
                    "slurp_id": slurp_id
                })
    return items

# ... (Existing Dataset and Sampler classes remain unchanged) ...

# ==============================================================================
# 5. Final Evaluation Logic (Multi-GPU DDP)
# ==============================================================================

def calculate_wer(reference, hypothesis):
    try:
        import jiwer
        return jiwer.wer(reference, hypothesis)
    except ImportError:
        # Fallback simple WER
        r = reference.split()
        h = hypothesis.split()
        d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
        d = d.reshape((len(r) + 1, len(h) + 1))
        for i in range(len(r) + 1): d[i][0] = i
        for j in range(len(h) + 1): d[0][j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)
        return d[len(r)][len(h)] / float(len(r)) if len(r) > 0 else 0.0

def evaluate_model(model, processor, items, device, output_dir):
    """
    Runs inference on ALL items using DDP slicing to speed up processing.
    Saves predictions in 'prediction.jsonl' format with extended fields.
    """
    from tqdm import tqdm
    import torch.distributed as dist
    import re
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 1. Slice items for this GPU
    total = len(items)
    chunk_size = (total + world_size - 1) // world_size
    if local_rank != -1:
        start_idx = local_rank * chunk_size
        end_idx = min(start_idx + chunk_size, total)
        my_items = items[start_idx:end_idx]
    else:
        my_items = items # Single GPU mode

    model.eval()
    results = []
    
    if local_rank in [-1, 0]:
        print(f"Starting evaluation on {total} items ({len(my_items)} per GPU)...")
        iterator = tqdm(my_items, desc="Evaluating")
    else:
        iterator = my_items
    
    for i, item in enumerate(iterator):
        audio_path = item.get("audio_path")
        transcript = item.get("transcript", "")
        slurp_id = item.get("slurp_id", "")
        
        if audio_path:
            # Load audio
            audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
            user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": PROMPT}]
            text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_input, audio=[audio], return_tensors="pt")
            
            # Key for output
            file_key = os.path.basename(audio_path)
            eval_type = "audio"
        else:
            user_content = [{"type": "text", "text": f"{transcript}\n{PROMPT}"}]
            text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_input, return_tensors="pt")
            file_key = f"text_{slurp_id}_{i}"
            eval_type = "text"

        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128)
            
        input_len = inputs["input_ids"].shape[1]
        decoded = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        
        # Parse JSON output
        parsed_json = {}
        try:
            # Try to find JSON object in the output
            match = re.search(r'\{.*\}', decoded, re.DOTALL)
            if match:
                json_str = match.group(0)
                parsed_json = json.loads(json_str)
            else:
                parsed_json = json.loads(decoded)
        except:
            parsed_json = {} # Falied to parse

        # Calculate WER (Reference: transcript, Hypothesis: decoded raw text)
        # Note: This might be high since model outputs JSON, but requested by user.
        wer_score = calculate_wer(transcript, decoded)
        
        # Format output
        res = {
            "scenario": parsed_json.get("scenario", ""),
            "action": parsed_json.get("action", ""),
            "entities": parsed_json.get("entities", []),
            "file": file_key,
            "slurp_id": slurp_id,
            "wer": wer_score,
            "transcript": transcript,
            "raw_output": decoded,
            "target": item["target"], # Original target JSON string
            "type": eval_type
        }
        results.append(res)

    # 2. Save Rank Output
    rank = local_rank if local_rank != -1 else 0
    rank_file = os.path.join(output_dir, f"predictions_rank_{rank}.jsonl")
    
    with open(rank_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # 3. Barrier & Merge
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        
    if rank == 0:
        final_file = os.path.join(output_dir, "predictions.jsonl")
        print(f"Merging results to {final_file}...")
        
        with open(final_file, "w") as f_out:
            for r in range(world_size):
                rf = os.path.join(output_dir, f"predictions_rank_{r}.jsonl")
                if os.path.exists(rf):
                    with open(rf, "r") as f_in:
                        f_out.write(f_in.read())
                    os.remove(rf)
        print("Done.")
    
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

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
    parser.add_argument("--only_eval", action="store_true", help="Skip training and run evaluation only")
    
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device = torch.device(f"cuda:{local_rank}") if local_rank != -1 else "cuda"

    # --- Data Paths ---
    if args.eval_file is None:
        args.eval_file = args.train_file.replace("train.jsonl", "devel.jsonl")
    if args.test_file is None:
        args.test_file = args.train_file.replace("train.jsonl", "test.jsonl")

    # --- Model & Processor ---
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.audio_tower.requires_grad_(False)
    model.multi_modal_projector.requires_grad_(False)

    # --- Training ---
    if not args.only_eval:
        # Load Data for Training
        train_items = build_items_from_slurp(args.train_file, args.audio_dir, max_samples=args.max_samples)
        eval_items = build_items_from_slurp(args.eval_file, args.audio_dir, max_samples=args.max_samples // 10 if args.max_samples else None)
        
        if local_rank in [-1, 0]:
            print(f"Train: {len(train_items)} | Eval: {len(eval_items)}")
            if len(eval_items) == 0:
                print(f"WARNING: Eval file {args.eval_file} not found or empty.")

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
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
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

        # Save model after training
        if local_rank in [-1, 0]:
            trainer.save_model(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print("\n=== Training Complete. Model Saved. ===")

        # Wait for save
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            dist.barrier()
    else:
        print("Skipping training (--only_eval set). Loading model/processor directly.")

    # --- Final Test ---
    if local_rank in [-1, 0]:
        print("\n=== Running Final Test on ALL Data (Multi-GPU) ===")
        
    # Load ALL test items (no max_samples limit for final test)
    test_items = build_items_from_slurp(args.test_file, args.audio_dir, max_samples=None)
    
    if test_items:
        evaluate_model(model, processor, test_items, device, args.output_dir)
    elif local_rank == 0:
        print(f"Test file {args.test_file} not found. Skipping evaluation.")

    # Cleanup
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dist.barrier()

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
# 5. Final Evaluation Logic (Multi-GPU DDP)
# ==============================================================================

def evaluate_model(model, processor, items, device, output_dir):
    """
    Runs inference on ALL items using DDP slicing to speed up processing.
    Saves predictions in 'prediction.jsonl' format.
    """
    from tqdm import tqdm
    import torch.distributed as dist
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 1. Slice items for this GPU
    total = len(items)
    chunk_size = (total + world_size - 1) // world_size
    if local_rank != -1:
        start_idx = local_rank * chunk_size
        end_idx = min(start_idx + chunk_size, total)
        my_items = items[start_idx:end_idx]
    else:
        my_items = items # Single GPU mode

    model.eval()
    results = []
    
    if local_rank in [-1, 0]:
        print(f"Starting evaluation on {total} items ({len(my_items)} per GPU)...")
        iterator = tqdm(my_items, desc="Evaluating")
    else:
        iterator = my_items
    
    for i, item in enumerate(iterator):
        audio_path = item.get("audio_path")
        transcript = item.get("transcript", "")
        
        if audio_path:
            # Load audio
            audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
            user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": PROMPT}]
            text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_input, audio=[audio], return_tensors="pt")
            
            # Key for output
            file_key = os.path.basename(audio_path)
        else:
            user_content = [{"type": "text", "text": f"{transcript}\n{PROMPT}"}]
            text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_input, return_tensors="pt")
            file_key = item.get("slurp_id", f"text_{i}")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128)
            
        input_len = inputs["input_ids"].shape[1]
        decoded = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        
        # Format matching run_eval.py / prediction.jsonl
        res = {
            "file": file_key,
            "prediction": decoded,
            "target": item["target"],
            "type": "audio" if audio_path else "text"
        }
        results.append(res)

    # 2. Save Rank Output
    rank = local_rank if local_rank != -1 else 0
    rank_file = os.path.join(output_dir, f"predictions_rank_{rank}.jsonl")
    
    with open(rank_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # 3. Barrier & Merge
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        
    if rank == 0:
        final_file = os.path.join(output_dir, "predictions.jsonl")
        print(f"Merging results to {final_file}...")
        
        with open(final_file, "w") as f_out:
            for r in range(world_size):
                rf = os.path.join(output_dir, f"predictions_rank_{r}.jsonl")
                if os.path.exists(rf):
                    with open(rf, "r") as f_in:
                        f_out.write(f_in.read())
                    os.remove(rf)
        print("Done.")
    
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

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
        save_total_limit=2, # Keep last 2 checkpoints
        load_best_model_at_end=True, # Load best model at end
        metric_for_best_model="loss", # We don't have compute_metrics set up for Trainer, so use validation loss
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

    # --- Final Test ---
    if local_rank in [-1, 0]:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print("\n=== Running Final Test on ALL Data (Multi-GPU) ===")
    
    # Wait for save to complete
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dist.barrier()
        
    # Load ALL test items (no max_samples limit for final test)
    test_items = build_items_from_slurp(args.test_file, args.audio_dir, max_samples=None)
    
    if test_items:
        evaluate_model(model, processor, test_items, device, args.output_dir)
    elif local_rank == 0:
        print(f"Test file {args.test_file} not found.")

    # Cleanup
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dist.barrier()

if __name__ == "__main__":
    main()