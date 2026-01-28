#!/usr/bin/env python3
"""
Audio-Text Mix End-to-End Training Script (Batch Supported)
=========================================================
Qwen2-Audio Mixed Input Training - Robust Batching Version

Features:
- Supports batch_size > 1 by padding text-only items with dummy silence features.
- Auto-skips corrupted or extremely short audio files.
- Consistent processor usage.

Usage:
    # Quick test (Batch size 2, 2 epochs)
    python audio_text_mix_e2e_v2.py --max_samples 20 --num_train_epochs 2 --batch_size 2
    
    # Full training
    python audio_text_mix_e2e_v2.py --num_train_epochs 3 --batch_size 4
"""

import argparse
import json
import os
import random
import torch
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import librosa

from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# ==============================================================================
# Constants & Setup
# ==============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT = """You are a voice assistant. Analyze the user's spoken request and output a JSON object with:
- "scenario": the general intent category
- "action": the specific action within that scenario
- "entities": a list of extracted entities as {"type": ..., "filler": ...}

Output only valid JSON, no extra text."""

# ==============================================================================
# Data Loading
# ==============================================================================

def resolve_audio_path(audio_root: str, filename: str) -> Optional[str]:
    """Check multiple possible locations for audio files."""
    candidates = [
        os.path.join(audio_root, filename),
        os.path.join(audio_root, "slurp_real", filename),
        os.path.join("slurp", "audio", "slurp_real", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def build_items_from_slurp(
    jsonl_path: str,
    audio_dir: str,
    add_text_only: bool = True,
    max_samples: Optional[int] = None
) -> List[Dict]:
    """Build dataset items from SLURP jsonl file."""
    items = []
    
    if not os.path.exists(jsonl_path):
        logger.error(f"File not found: {jsonl_path}")
        return items
    
    logger.info(f"Loading data from {jsonl_path}...")
    
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if max_samples and len(items) >= max_samples:
            break
            
        data = json.loads(line)
        
        # Build target JSON
        entities = [{"type": e.get("type"), "filler": e.get("filler")} for e in data.get("entities", [])]
        target = json.dumps({
            "scenario": data.get("scenario", ""),
            "action": data.get("action", ""),
            "entities": entities
        }, ensure_ascii=False)
        
        transcript = data.get("sentence", "")
        slurp_id = data.get("slurp_id", str(len(items)))
        
        # 1. Add Text-Only Item
        if add_text_only:
            items.append({
                "audio_path": None,
                "transcript": transcript,
                "target": target,
                "slurp_id": f"{slurp_id}_text",
            })
        
        # 2. Add Audio Item
        recordings = data.get("recordings", [])
        if recordings:
            audio_file = recordings[0].get("file", "")
            audio_path = resolve_audio_path(audio_dir, audio_file)
            
            if audio_path:
                items.append({
                    "audio_path": audio_path,
                    "transcript": transcript,
                    "target": target,
                    "slurp_id": f"{slurp_id}_audio",
                })
    
    return items

class MixedDataset(Dataset):
    def __init__(self, items: List[Dict]):
        self.items = items
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]

# ==============================================================================
# Robust Collator (Supports Batching)
# ==============================================================================

@dataclass
class AudioTextCollator:
    processor: Any
    max_length: int = 512
    ignore_index: int = -100
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collates mixed audio/text items. 
        Critical Strategy for Batching:
        - Even text-only items get a 'dummy' audio feature (silence).
        - However, text-only items do NOT get the <|audio|> token in their prompt.
        - This ensures all tensor shapes align (Batch, Seq, Dim) for 'input_features'.
        """
        
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        input_features_list = []
        feature_mask_list = []
        
        # Sampling rate from processor
        sr = self.processor.feature_extractor.sampling_rate
        
        for item in batch:
            audio_path = item.get("audio_path")
            transcript = item.get("transcript", "")
            target = item.get("target", "")
            
            audio_np = None
            is_valid_audio = False
            
            # --- 1. Load Audio (or create dummy) ---
            if audio_path:
                try:
                    # Load audio
                    audio_np, _ = librosa.load(audio_path, sr=sr)
                    # Check duration (too short audio causes Qwen crash)
                    if len(audio_np) < sr * 0.1:  # < 0.1s
                        logger.warning(f"Audio too short, treating as text-only: {audio_path}")
                        audio_np = None
                    else:
                        is_valid_audio = True
                except Exception as e:
                    logger.warning(f"Failed to load {audio_path}: {e}. Treating as text-only.")
                    audio_np = None
            
            # Create dummy silence for text-only items so batching works
            if audio_np is None:
                # 0.5s silence
                audio_np = np.zeros(int(sr * 0.5), dtype=np.float32)
                is_valid_audio = False
            
            # --- 2. Construct Prompt ---
            if is_valid_audio:
                # Audio Prompt: Contains placeholder
                user_content = [
                    {"type": "audio", "audio_url": "placeholder_path"}, # path string doesn't matter here
                    {"type": "text", "text": PROMPT}
                ]
            else:
                # Text-Only Prompt: NO audio placeholder
                # We still pass features, but the model won't look at them because the prompt doesn't link to them.
                user_content = [
                    {"type": "text", "text": f"User: {transcript}\n\n{PROMPT}"}
                ]

            messages = [{"role": "user", "content": user_content}]
            
            # Apply chat template
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = prompt_text + target
            
            # --- 3. Tokenize & Feature Extraction ---
            # We process one by one to avoid padding issues at this stage
            # We treat EVERYTHING as having audio input (dummy or real)
            inputs = self.processor(
                text=full_text,
                audio=[audio_np], # List of 1
                return_tensors="pt",
                padding="longest", 
            )
            
            # Calculate prompt length for masking labels
            prompt_inputs = self.processor(
                text=prompt_text,
                audio=[audio_np],
                return_tensors="pt",
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            # --- 4. Prepare Tensors ---
            curr_input_ids = inputs["input_ids"][0]
            curr_att_mask = inputs["attention_mask"][0]
            
            # Labels: mask prompt
            curr_labels = curr_input_ids.clone()
            curr_labels[:prompt_len] = self.ignore_index
            
            # Audio Features
            # inputs['input_features'] shape is usually (1, Seq, Dim) or (1, 1, Seq, Dim)
            # We want to collect (Seq, Dim) and pad them later
            feat = inputs["input_features"]
            feat_mask = inputs.get("feature_attention_mask")
            
            # Squeeze batch dims to get (Seq, Dim)
            while feat.dim() > 2:
                feat = feat.squeeze(0)
            if feat_mask is not None:
                while feat_mask.dim() > 1:
                    feat_mask = feat_mask.squeeze(0)
            
            input_ids_list.append(curr_input_ids)
            attention_mask_list.append(curr_att_mask)
            labels_list.append(curr_labels)
            input_features_list.append(feat) # (Seq, Dim)
            if feat_mask is not None:
                feature_mask_list.append(feat_mask)

        # --- 5. Batch Padding ---
        # Pad Text Tokens
        padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        padded_att_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index)
        
        # Pad Audio Features manually (pad_sequence pads dim 0, which is time/seq)
        # input_features: List of (Time, Dim) -> (Batch, MaxTime, Dim)
        padded_features = pad_sequence(input_features_list, batch_first=True, padding_value=0.0)
        
        batch_out = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_att_mask,
            "labels": padded_labels,
            "input_features": padded_features,
        }
        
        if feature_mask_list:
            padded_feat_mask = pad_sequence(feature_mask_list, batch_first=True, padding_value=0)
            batch_out["feature_attention_mask"] = padded_feat_mask
            
        return batch_out

# ==============================================================================
# Evaluation (Inference)
# ==============================================================================

def evaluate_model(model, processor, items, max_samples=10, device="cuda"):
    """Evaluate using generate(). Keep batch_size=1 for safety in eval."""
    model.eval()
    results = []
    
    print(f"\nEvaluating on {min(len(items), max_samples)} samples...")
    
    for i, item in enumerate(tqdm(items[:max_samples])):
        audio_path = item.get("audio_path")
        transcript = item.get("transcript", "")
        
        # Prepare inputs
        if audio_path and os.path.exists(audio_path):
            audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
            user_content = [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": PROMPT}
            ]
            inputs = processor(
                text=processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True),
                audio=[audio],
                return_tensors="pt"
            )
        else:
            # Text only inference
            user_content = [{"type": "text", "text": f"{transcript}\n{PROMPT}"}]
            inputs = processor(
                text=processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True),
                return_tensors="pt"
            )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            
        input_len = inputs["input_ids"].shape[1]
        pred = processor.decode(gen_ids[0][input_len:], skip_special_tokens=True)
        
        results.append({"target": item["target"], "prediction": pred})
        
        if i < 3:
            print(f"\nTarget: {item['target']}\nPred:   {pred}")

    acc = sum(1 for r in results if r["target"].strip() == r["prediction"].strip()) / len(results)
    print(f"\nAccuracy: {acc:.2%}")
    return acc

# ==============================================================================
# Main Training
# ==============================================================================

def train_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Load Processor & Model
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    ).to(device)
    
    # Freeze audio encoder parts to save memory/prevent instability
    model.audio_tower.requires_grad_(False)
    model.multi_modal_projector.requires_grad_(False)
    
    # 2. Data
    train_items = build_items_from_slurp(args.train_file, args.audio_dir, max_samples=args.max_samples)
    if not train_items:
        raise ValueError("No training data found.")
    
    # Shuffle for mixed batches
    random.shuffle(train_items)
    
    train_dataset = MixedDataset(train_items)
    collator = AudioTextCollator(processor=processor, max_length=args.max_length)
    
    # 3. Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size, # Now supports > 1
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=5,
        save_steps=100,
        bf16=args.bf16,
        remove_unused_columns=False, # Essential for custom collator keys
        report_to="none",
        dataloader_num_workers=4,    # Speed up loading
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    # 4. Quick Eval
    test_file = args.train_file.replace("train.jsonl", "test.jsonl")
    test_items = build_items_from_slurp(test_file, args.audio_dir, max_samples=10)
    evaluate_model(model, processor, test_items, device=device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio/slurp_real")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_mix_v2")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bf16", action="store_true", default=True)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)

if __name__ == "__main__":
    main()