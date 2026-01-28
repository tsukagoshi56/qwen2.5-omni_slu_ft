#!/usr/bin/env python3
"""
Audio-Text Mix End-to-End Training and Evaluation Script
=========================================================
This script provides a self-contained implementation for training and evaluating
the Qwen2-Audio model on SLURP dataset with audio+text mixed input.

Features:
- Consistent processor usage throughout training and inference
- Small-scale testing support (--max_samples for quick validation)
- Single file, easy to debug

Usage:
    # Quick test with 10 samples, 2 epochs
    python audio_text_mix_e2e.py --max_samples 10 --num_train_epochs 2
    
    # Full training
    python audio_text_mix_e2e.py --num_train_epochs 3
"""

import argparse
import json
import os
import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
import librosa

# ==============================================================================
# Constants
# ==============================================================================

PROMPT = """You are a voice assistant. Analyze the user's spoken request and output a JSON object with:
- "scenario": the general intent category
- "action": the specific action within that scenario
- "entities": a list of extracted entities as {"type": ..., "filler": ...}

Output only valid JSON, no extra text."""

SAMPLING_RATE = 16000

# ==============================================================================
# Data Loading
# ==============================================================================

def load_audio(audio_path: str, target_sr: int = SAMPLING_RATE) -> np.ndarray:
    """Load audio file and resample to target sampling rate."""
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio


def resolve_audio_path(audio_root: str, filename: str) -> Optional[str]:
    """Check multiple possible locations for audio files."""
    parent_dir = os.path.dirname(audio_root)
    candidates = [
        os.path.join(audio_root, filename),
        os.path.join(audio_root, "slurp_real", filename),
        # Also check if slurp_real is sibling to audio_root
        os.path.join(parent_dir, "slurp_real", filename),
        # Explicitly check slurp/slurp_real relative to CWD
        os.path.join("slurp", "slurp_real", filename),
        # Also try audio subdirectory
        os.path.join("slurp", "audio", "slurp_real", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def build_items_from_slurp(
    jsonl_path: str,
    audio_dir: str,
    add_text_only: bool = True,  # Also add text-only items
    max_samples: Optional[int] = None
) -> List[Dict]:
    """Build dataset items from SLURP jsonl file.
    
    Creates both audio items (audio_path set) and text items (audio_path=None).
    Audio items: use audio + prompt
    Text items: use transcript + prompt
    """
    items = []
    missing_audio = 0
    
    if not os.path.exists(jsonl_path):
        print(f"ERROR: jsonl file not found: {jsonl_path}")
        return items
    
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f):
            data = json.loads(line)
            
            # Build target JSON
            entities = []
            for ent in data.get("entities", []):
                entities.append({
                    "type": ent.get("type", ""),
                    "filler": ent.get("filler", "")
                })
            
            target = json.dumps({
                "scenario": data.get("scenario", ""),
                "action": data.get("action", ""),
                "entities": entities
            }, ensure_ascii=False)
            
            transcript = data.get("sentence", "")
            slurp_id = data.get("slurp_id", str(len(items)))
            
            # Add text-only item
            if add_text_only:
                items.append({
                    "audio_path": None,  # No audio - text only
                    "transcript": transcript,
                    "target": target,
                    "slurp_id": slurp_id,
                })
            
            # Add audio item (first recording)
            recordings = data.get("recordings", [])
            if recordings:
                rec = recordings[0]
                audio_file = rec.get("file", "")
                audio_path = resolve_audio_path(audio_dir, audio_file)
                
                if audio_path is not None:
                    items.append({
                        "audio_path": audio_path,
                        "transcript": transcript,
                        "target": target,
                        "slurp_id": slurp_id,
                    })
                else:
                    missing_audio += 1
                    if line_num < 5:
                        print(f"  Missing audio: {audio_file}")
            
            if max_samples and len(items) >= max_samples:
                break
    
    if missing_audio:
        print(f"Warning: {missing_audio} recordings missing from {audio_dir}")
    
    return items


class MixedDataset(Dataset):
    """Dataset that mixes audio and text items, with optional audio partitioning.
    
    - Text items: use transcript + prompt
    - Audio items: use audio + prompt
    - partition_audio: if True, only use 1/N audio items per epoch
    """
    
    def __init__(
        self, 
        items: List[Dict], 
        partition_audio: bool = True,
        total_epochs: int = 3
    ):
        # Separate text and audio items
        self.text_items = [x for x in items if x.get("audio_path") is None]
        self.audio_items = [x for x in items if x.get("audio_path") is not None]
        self.partition_audio = partition_audio
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Shuffle audio items stably for partitioning
        if self.partition_audio and len(self.audio_items) > 0:
            rng = random.Random(42)
            rng.shuffle(self.audio_items)
        
        self._rebuild_items()
        
        print(f"MixedDataset: {len(self.text_items)} text items, {len(self.audio_items)} audio items total")
    
    def set_epoch(self, epoch: int):
        """Called by Trainer to update epoch for audio partitioning."""
        self.current_epoch = epoch
        self._rebuild_items()
    
    def _rebuild_items(self):
        """Rebuild the item list based on current epoch."""
        if not self.partition_audio or not self.audio_items:
            # No partitioning - use all items
            self.items = self.text_items + self.audio_items
        else:
            # Partition audio: use 1/N of audio items per epoch
            num_audio = len(self.audio_items)
            chunk_size = (num_audio + self.total_epochs - 1) // self.total_epochs
            start_idx = (self.current_epoch % self.total_epochs) * chunk_size
            end_idx = min(start_idx + chunk_size, num_audio)
            
            current_audio_batch = self.audio_items[start_idx:end_idx]
            self.items = self.text_items + current_audio_batch
            
            print(f"Epoch {self.current_epoch}: {len(self.text_items)} text + {len(current_audio_batch)} audio (idx {start_idx}-{end_idx})")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]


# ==============================================================================
# Collator
# ==============================================================================

@dataclass
class AudioTextCollator:
    """Collator that uses processor consistently for both training and inference."""
    
    processor: Any
    max_length: int = 512
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch for training."""
        
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_input_features = []
        all_feature_attention_mask = []
        
        for item in batch:
            audio_path = item.get("audio_path")
            transcript = item.get("transcript", "")
            target = item.get("target", "")
            
            if audio_path is not None:
                # Audio item: use audio + prompt
                audio_np = load_audio(audio_path)
                
                # Build conversation with audio_url placeholder
                user_content = [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": PROMPT}
                ]
                messages = [{"role": "user", "content": user_content}]
                
                prompt_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                full_text = prompt_text + target
                
                # Process with processor (official pattern: audio= with list)
                prompt_inputs = self.processor(
                    text=prompt_text,
                    audio=[audio_np],
                    return_tensors="pt",
                    padding=True,
                )
                full_inputs = self.processor(
                    text=full_text,
                    audio=[audio_np],
                    return_tensors="pt",
                    padding=True,
                )
                
                input_ids = full_inputs["input_ids"].squeeze(0)
                attention_mask = full_inputs["attention_mask"].squeeze(0)
                input_features = full_inputs.get("input_features")
                feature_attention_mask = full_inputs.get("feature_attention_mask")
                
            else:
                # Text-only item: use transcript + prompt
                user_content = [
                    {"type": "text", "text": f"{transcript}\n{PROMPT}"}
                ]
                messages = [{"role": "user", "content": user_content}]
                
                prompt_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                full_text = prompt_text + target
                
                # Text-only: use tokenizer directly
                prompt_inputs = self.processor.tokenizer(
                    prompt_text, return_tensors="pt"
                )
                full_inputs = self.processor.tokenizer(
                    full_text, return_tensors="pt",
                    truncation=True, max_length=self.max_length
                )
                
                input_ids = full_inputs["input_ids"].squeeze(0)
                attention_mask = full_inputs["attention_mask"].squeeze(0)
                input_features = None
                feature_attention_mask = None
            
            # Create labels (mask prompt, only predict target)
            prompt_len = prompt_inputs["input_ids"].size(1)
            
            labels = input_ids.clone()
            labels[:prompt_len] = -100  # Mask prompt tokens
            
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
            
            if input_features is not None:
                all_input_features.append(input_features.squeeze(0))
            if feature_attention_mask is not None:
                all_feature_attention_mask.append(feature_attention_mask.squeeze(0))
        
        # Pad to same length
        max_len = max(ids.size(0) for ids in all_input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        pad_token_id = self.processor.tokenizer.pad_token_id or 0
        
        for ids, mask, labels in zip(all_input_ids, all_attention_mask, all_labels):
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])
            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)
            padded_labels.append(labels)
        
        result = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }
        
        if all_input_features:
            result["input_features"] = torch.stack(all_input_features)
        if all_feature_attention_mask:
            result["feature_attention_mask"] = torch.stack(all_feature_attention_mask)
        
        return result


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_model(
    model,
    processor,
    items: List[Dict],
    max_samples: int = 10,
    device: str = "cuda"
) -> Dict:
    """Run inference and evaluate results."""
    
    model.eval()
    results = []
    
    eval_items = items[:max_samples] if max_samples else items
    
    print(f"\n{'='*60}")
    print(f"Evaluating on {len(eval_items)} samples")
    print(f"{'='*60}\n")
    
    for i, item in enumerate(tqdm(eval_items, desc="Evaluating")):
        # Load audio
        audio = load_audio(item["audio_path"])
        
        # Build conversation with audio_url placeholder
        user_content = [
            {"type": "audio", "audio_url": item["audio_path"]},
            {"type": "text", "text": PROMPT}
        ]
        messages = [{"role": "user", "content": user_content}]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process with processor (official pattern: audio= with list)
        inputs = processor(
            text=text,
            audio=[audio],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        
        # Decode (remove input tokens)
        input_len = inputs["input_ids"].size(1)
        new_ids = generated_ids[0, input_len:]
        response = processor.decode(new_ids, skip_special_tokens=True)
        
        results.append({
            "target": item["target"],
            "prediction": response,
        })
        
        # Print first few samples
        if i < 5:
            print(f"\n--- Sample {i+1} ---")
            print(f"Target: {item['target']}")
            print(f"Pred:   {response}")
    
    # Calculate accuracy (simple exact match for now)
    correct = sum(1 for r in results if r["target"].strip() == r["prediction"].strip())
    accuracy = correct / len(results) if results else 0
    
    print(f"\n{'='*60}")
    print(f"Results: {correct}/{len(results)} exact match ({accuracy*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return {"accuracy": accuracy, "results": results}


# ==============================================================================
# Training
# ==============================================================================

def train_model(args):
    """Main training function."""
    
    print(f"\n{'='*60}")
    print("Audio-Text Mix End-to-End Training")
    print(f"{'='*60}\n")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load processor and model
    print(f"Loading model from: {args.model_name_or_path}")
    
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    )
    
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Tokenizer len: {len(processor.tokenizer)}")
    
    # Freeze audio components
    if hasattr(model, "audio_tower"):
        for param in model.audio_tower.parameters():
            param.requires_grad = False
        print("Froze audio_tower")
    if hasattr(model, "multi_modal_projector"):
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False
        print("Froze multi_modal_projector")
    
    # Load training data
    print(f"\nLoading training data from: {args.train_file}")
    train_items = build_items_from_slurp(
        args.train_file,
        args.audio_dir,
        add_text_only=True,  # Include text-only items
        max_samples=args.max_samples
    )
    print(f"Loaded {len(train_items)} training items")
    
    if len(train_items) == 0:
        raise ValueError(f"No training samples loaded! Check paths:\n  train_file: {args.train_file}\n  audio_dir: {args.audio_dir}")
    
    # Load validation data from devel.jsonl
    devel_file = args.train_file.replace("train.jsonl", "devel.jsonl")
    print(f"\nLoading validation data from: {devel_file}")
    eval_items = build_items_from_slurp(
        devel_file,
        args.audio_dir,
        add_text_only=True,
        max_samples=args.eval_samples if args.max_samples else None
    )
    print(f"Loaded {len(eval_items)} validation items")
    
    # Load test data from test.jsonl
    test_file = args.train_file.replace("train.jsonl", "test.jsonl")
    print(f"\nLoading test data from: {test_file}")
    test_items = build_items_from_slurp(
        test_file,
        args.audio_dir,
        add_text_only=True,
        max_samples=args.eval_samples if args.max_samples else None
    )
    print(f"Loaded {len(test_items)} test items")
    
    # Create datasets with audio partitioning
    train_dataset = MixedDataset(
        train_items, 
        partition_audio=True,
        total_epochs=args.num_train_epochs
    )
    eval_dataset = MixedDataset(eval_items, partition_audio=False) if eval_items else None
    
    # Collator
    collator = AudioTextCollator(processor=processor, max_length=args.max_length)
    
    # Force batch_size=1 for mixed audio/text to avoid audio token mismatch in batches
    # (audio items have input_features, text items don't)
    if args.batch_size != 1:
        print(f"Warning: batch_size={args.batch_size} but mixed audio/text requires batch_size=1. Forcing batch_size=1.")
        args.batch_size = 1
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,  # Must be 1 for mixed batches
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=50,
        bf16=args.bf16,
        remove_unused_columns=False,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save
    print(f"\nSaving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    # Evaluate using the SAME processor
    print("\nRunning evaluation with the same processor...")
    model.to(device)
    eval_results = evaluate_model(
        model=model,
        processor=processor,  # Same processor!
        items=eval_items,
        max_samples=args.eval_samples,
        device=device,
    )
    
    print("\nDone!")
    return model, processor


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Audio-Text Mix E2E Training")
    
    # Data
    parser.add_argument("--train_file", type=str, default="slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio/slurp_real")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--eval_samples", type=int, default=10, help="Number of samples for evaluation")
    
    # Model
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/audio_text_mix_e2e")
    parser.add_argument("--max_length", type=int, default=512)
    
    # Training
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--bf16", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    train_model(args)


if __name__ == "__main__":
    main()
