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


def build_items_from_slurp(
    jsonl_path: str,
    audio_dir: str,
    max_samples: Optional[int] = None
) -> List[Dict]:
    """Build dataset items from SLURP jsonl file."""
    items = []
    with open(jsonl_path, "r") as f:
        for line in f:
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
            
            # Get audio files
            recordings = data.get("recordings", [])
            if not recordings:
                continue
                
            # Use first recording
            rec = recordings[0]
            audio_file = rec.get("file", "")
            audio_path = os.path.join(audio_dir, audio_file)
            
            if not os.path.exists(audio_path):
                continue
            
            items.append({
                "audio_path": audio_path,
                "transcript": data.get("sentence", ""),
                "target": target,
                "slurp_id": data.get("slurp_id", str(len(items))),
            })
            
            if max_samples and len(items) >= max_samples:
                break
    
    return items


class AudioTextDataset(Dataset):
    """Simple dataset for audio-text mix training."""
    
    def __init__(self, items: List[Dict]):
        self.items = items
    
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
            # Load audio
            audio = load_audio(item["audio_path"])
            
            # Build conversation with chat template
            user_content = [
                {"type": "audio", "audio": item["audio_path"]},
                {"type": "text", "text": PROMPT}
            ]
            messages = [{"role": "user", "content": user_content}]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Add target for training
            target = item.get("target", "")
            full_text = text + target
            
            # Process with processor
            inputs = self.processor(
                text=full_text,
                audios=[audio],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
            input_features = inputs.get("input_features")
            feature_attention_mask = inputs.get("feature_attention_mask")
            
            # Create labels (mask prompt, only predict target)
            # Find where target starts by tokenizing just the prompt
            prompt_inputs = self.processor(
                text=text,
                audios=[audio],
                return_tensors="pt",
            )
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
        
        # Build conversation (same as training)
        user_content = [
            {"type": "audio", "audio": item["audio_path"]},
            {"type": "text", "text": PROMPT}
        ]
        messages = [{"role": "user", "content": user_content}]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process input
        inputs = processor(
            text=text,
            audios=[audio],
            return_tensors="pt",
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
    
    # Load data
    print(f"\nLoading data from: {args.train_file}")
    train_items = build_items_from_slurp(
        args.train_file,
        args.audio_dir,
        max_samples=args.max_samples
    )
    print(f"Loaded {len(train_items)} training samples")
    
    # Split train/eval
    eval_size = min(len(train_items) // 5, 50)
    eval_items = train_items[:eval_size]
    train_items = train_items[eval_size:]
    
    print(f"Train: {len(train_items)}, Eval: {len(eval_items)}")
    
    train_dataset = AudioTextDataset(train_items)
    eval_dataset = AudioTextDataset(eval_items) if eval_items else None
    
    # Collator
    collator = AudioTextCollator(processor=processor, max_length=args.max_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
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
