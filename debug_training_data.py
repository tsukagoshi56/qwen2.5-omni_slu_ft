#!/usr/bin/env python
"""Debug script to inspect training data format."""
import argparse
import json
from train_qwen2_audio_slurp import (
    build_items,
    SlurpDataset,
    Qwen2AudioCollator,
    CollatorConfig,
    PROMPT,
    resolve_slurp_root,
    resolve_data_dir,
)
from transformers import AutoProcessor

def main():
    parser = argparse.ArgumentParser(description="Debug training data")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--data_dir", default="slurp/dataset/slurp")
    parser.add_argument("--audio_dir", default="slurp/audio")
    parser.add_argument("--train_file", default="train.jsonl")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to display")
    parser.add_argument("--add_text_only", action="store_true")
    parser.add_argument("--save_path", default=None, help="Save all processed items to this JSONL path")
    args = parser.parse_args()

    print("=" * 60)
    print("PROMPT used in training:")
    print("=" * 60)
    print(PROMPT)
    print()

    # Load processor
    print("Loading processor...")
    try:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load processor from {args.model_name_or_path}: {e}")
        return

    # Build items
    slurp_root = resolve_slurp_root(args.data_dir, args.audio_dir, None)
    data_dir = resolve_data_dir(slurp_root, args.data_dir)
    train_path = f"{data_dir}/{args.train_file}"
    
    print(f"Loading data from: {train_path}")
    items = build_items(train_path, args.audio_dir, use_all_recordings=False, add_text_only=args.add_text_only)
    
    print(f"Total items: {len(items)}")
    print()

    # Show raw items
    print("=" * 60)
    print(f"RAW DATA (first {args.num_samples} items):")
    print("=" * 60)
    for i, item in enumerate(items[:args.num_samples]):
        print(f"\n--- Item {i+1} ---")
        print(f"slurp_id: {item.get('slurp_id')}")
        print(f"transcript: {item.get('transcript')}")
        print(f"audio_path: {item.get('audio_path')}")
        print(f"target: {item.get('target')}")
    
    print()
    print("=" * 60)
    print("PROCESSED PROMPT (what model sees during training):")
    print("=" * 60)
    
    # Build collator config
    config = CollatorConfig(
        include_transcript=True,  # Default
        max_length=2048,
        audio_sampling_rate=processor.feature_extractor.sampling_rate if hasattr(processor, 'feature_extractor') else None
    )
    collator = Qwen2AudioCollator(processor, config)
    
    # Show processed items sample
    for i, item in enumerate(items[:args.num_samples]):
        print(f"\n--- Processed Item {i+1} ---")
        
        # Build the prompt text
        prompt_text = collator.build_prompt(item["transcript"])
        print(f"Prompt text: {prompt_text[:200]}...")
        
        # Build message structure
        user_content = []
        if item.get("audio_path") and not args.add_text_only:
            user_content.append({"type": "audio", "audio": item["audio_path"]})
        user_content.append({"type": "text", "text": prompt_text})
        
        messages = [{"role": "user", "content": user_content}]
        full_messages = messages + [
            {"role": "assistant", "content": [{"type": "text", "text": item["target"]}]}
        ]
        
        # Apply chat template
        full_text = processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        
        print(f"\nFULL TEXT (input + expected output):")
        print("-" * 40)
        print(full_text)
        print("-" * 40)
        
        print(f"\nEXPECTED OUTPUT (target):")
        print(item["target"])

    # Save all items
    if args.save_path:
        print(f"\nSaving all {len(items)} processed inputs to {args.save_path}...")
        with open(args.save_path, "w", encoding="utf-8") as f:
            for item in items:
                prompt_text = collator.build_prompt(item["transcript"])
                user_content = []
                if item.get("audio_path") and not args.add_text_only:
                    user_content.append({"type": "audio", "audio": item["audio_path"]})
                user_content.append({"type": "text", "text": prompt_text})
                
                messages = [{"role": "user", "content": user_content}]
                full_messages = messages + [
                    {"role": "assistant", "content": [{"type": "text", "text": item["target"]}]}
                ]
                
                full_text = processor.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False
                )
                
                dump_obj = {
                    "slurp_id": item.get("slurp_id"),
                    "transcript": item.get("transcript"),
                    "full_text_input": full_text,
                    "target": item.get("target")
                }
                f.write(json.dumps(dump_obj, ensure_ascii=False) + "\n")
        print("Save complete.")

if __name__ == "__main__":
    main()
