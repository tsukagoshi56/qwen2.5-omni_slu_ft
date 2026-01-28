#!/usr/bin/env python3
"""
Audio-Text Mix Training (Homogeneous Batching Strategy)
=========================================================
Features:
- "Homogeneous Batching": Creates batches that are either purely Audio or purely Text.
- Eliminates the need for dummy audio processing.
- Prevents "found 0 <|AUDIO|>" errors naturally.

Usage:
    python audio_text_mix_smart_batch.py --max_samples 50 --batch_size 2 --num_train_epochs 2
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT = """You are a voice assistant. Analyze the user's spoken request and output a JSON object with:
- "scenario": the general intent category
- "action": the specific action within that scenario
- "entities": a list of extracted entities as {"type": ..., "filler": ...}

Output only valid JSON, no extra text."""

# ==============================================================================
# 1. Data Loading (Standard)
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
        # Return index too, helpful for debugging
        return {**self.items[idx], "original_idx": idx}

# ==============================================================================
# 2. Homogeneous Batch Sampler (Key Logic)
# ==============================================================================

class HomogeneousBatchSampler(Sampler):
    """
    A sampler that groups data by modality (Audio vs Text).
    It ensures each batch contains ONLY audio items OR ONLY text items.
    """
    def __init__(self, dataset: MixedDataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by type
        self.audio_indices = [i for i, item in enumerate(dataset.items) if item["audio_path"] is not None]
        self.text_indices = [i for i, item in enumerate(dataset.items) if item["audio_path"] is None]

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within their groups
        audio_idxs = list(self.audio_indices)
        text_idxs = list(self.text_indices)
        random.shuffle(audio_idxs)
        random.shuffle(text_idxs)
        
        # Create batches
        batches = []
        
        # Audio batches
        for i in range(0, len(audio_idxs), self.batch_size):
            batch = audio_idxs[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
                
        # Text batches
        for i in range(0, len(text_idxs), self.batch_size):
            batch = text_idxs[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        # Shuffle the ORDER of batches (so the model sees mixed modalities over time)
        random.shuffle(batches)
        
        for batch in batches:
            yield batch

    def __len__(self):
        # Calculate approximate number of batches
        audio_len = len(self.audio_indices)
        text_len = len(self.text_indices)
        
        audio_batches = (audio_len + self.batch_size - 1) // self.batch_size
        text_batches = (text_len + self.batch_size - 1) // self.batch_size
        
        if self.drop_last:
            audio_batches = audio_len // self.batch_size
            text_batches = text_len // self.batch_size
            
        return audio_batches + text_batches

# ==============================================================================
# 3. Smart Collator (Switches Modes)
# ==============================================================================

@dataclass
class SmartCollator:
    processor: Any
    max_length: int = 512
    ignore_index: int = -100
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Check the modality of the batch (Sampler ensures they are all same type)
        is_audio_batch = (batch[0].get("audio_path") is not None)
        
        if is_audio_batch:
            return self._collate_audio(batch)
        else:
            return self._collate_text(batch)

    def _collate_audio(self, batch):
        """Process a batch of AUDIO items."""
        input_ids_list = []
        labels_list = []
        input_features_list = []
        feature_mask_list = []
        
        sr = self.processor.feature_extractor.sampling_rate
        
        for item in batch:
            # Load audio
            audio, _ = librosa.load(item["audio_path"], sr=sr)
            
            # Create Prompt with <|audio|> placeholder
            user_content = [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": PROMPT}
            ]
            text_input = self.processor.apply_chat_template(
                [{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True
            )
            full_text = text_input + item["target"]
            
            # Process
            inputs = self.processor(
                text=full_text, audio=[audio], return_tensors="pt"
            )
            prompt_inputs = self.processor(
                text=text_input, audio=[audio], return_tensors="pt"
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            # Labels
            ids = inputs["input_ids"][0]
            lbs = ids.clone()
            lbs[:prompt_len] = self.ignore_index
            
            input_ids_list.append(ids)
            labels_list.append(lbs)
            
            # Features (Remove batch dim 1)
            feat = inputs["input_features"]
            while feat.dim() > 2: feat = feat.squeeze(0)
            input_features_list.append(feat)
            
            if "feature_attention_mask" in inputs:
                f_mask = inputs["feature_attention_mask"]
                while f_mask.dim() > 1: f_mask = f_mask.squeeze(0)
                feature_mask_list.append(f_mask)
        
        # Pad Everything
        batch_out = {
            "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index),
            "attention_mask": pad_sequence([torch.ones_like(ids) for ids in input_ids_list], batch_first=True, padding_value=0),
            "input_features": pad_sequence(input_features_list, batch_first=True, padding_value=0.0)
        }
        if feature_mask_list:
            batch_out["feature_attention_mask"] = pad_sequence(feature_mask_list, batch_first=True, padding_value=0)
            
        return batch_out

    def _collate_text(self, batch):
        """Process a batch of TEXT-ONLY items (No input_features)."""
        input_ids_list = []
        labels_list = []
        
        for item in batch:
            # Pure text prompt (NO audio placeholder)
            user_content = [{"type": "text", "text": f"{item['transcript']}\n{PROMPT}"}]
            
            text_input = self.processor.apply_chat_template(
                [{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True
            )
            full_text = text_input + item["target"]
            
            # Process text only
            inputs = self.processor.tokenizer(full_text, return_tensors="pt")
            prompt_inputs = self.processor.tokenizer(text_input, return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            ids = inputs["input_ids"][0]
            lbs = ids.clone()
            lbs[:prompt_len] = self.ignore_index
            
            input_ids_list.append(ids)
            labels_list.append(lbs)
            
        batch_out = {
            "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index),
            "attention_mask": pad_sequence([torch.ones_like(ids) for ids in input_ids_list], batch_first=True, padding_value=0),
            # NO input_features here!
        }
        return batch_out

# ==============================================================================
# 4. Custom Trainer (To inject Sampler)
# ==============================================================================

class CustomTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """Override to use HomogeneousBatchSampler"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
            
        train_dataset = self.train_dataset
        
        # Create our custom sampler
        batch_sampler = HomogeneousBatchSampler(
            train_dataset,
            batch_size=self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last
        )

        return DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio/slurp_real")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_smart_batch")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--bf16", action="store_true", default=True)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # Model
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True
    ).to(device)
    
    # Freeze
    model.audio_tower.requires_grad_(False)
    model.multi_modal_projector.requires_grad_(False)
    
    # Data
    train_items = build_items_from_slurp(args.train_file, args.audio_dir, max_samples=args.max_samples)
    train_dataset = MixedDataset(train_items)
    
    print(f"Loaded {len(train_items)} items.")
    
    # Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        logging_steps=5,
        save_steps=100,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0 # Simplifies debugging
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=SmartCollator(processor),
        tokenizer=processor.tokenizer,
    )
    
    print("Starting training with Homogeneous Batching...")
    trainer.train()
    
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()