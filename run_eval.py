import argparse
import json
import logging
import os
import subprocess
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, AutoModelForCausalLM

# Import helper functions from training script
try:
    from train_qwen2_audio_slurp import (
        PROMPT,
        build_items,
        SlurpDataset,
        load_audio_input,
    )
except ImportError:
    raise ImportError("Could not import from train_qwen2_audio_slurp.py. Make sure it is in the same directory.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_json(text: str) -> dict:
    """Extract JSON object from the model output."""
    try:
        # Look for the JSON part starting with '{'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            return json.loads(json_str)
    except Exception as e:
        logger.warning(f"Failed to parse JSON: {e} | Text: {text}")
    
    # Return empty valid structure if failed
    return {"scenario": "none", "action": "none", "entities": []}

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on SLURP dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or checkpoint")
    parser.add_argument("--test_file", type=str, default="slurp/dataset/slurp/test.jsonl", help="Path to test data (jsonl)")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio", help="Path to audio directory")
    parser.add_argument("--output_file", type=str, default="predictions.jsonl", help="Output predictions file")
    parser.add_argument("--gold_file", type=str, default=None, help="Path to gold file for evaluation script (defaults to test_file)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (for dry run)")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size (currently only 1 supported securely)")
    parser.add_argument("--add_text_only", action="store_true", help="Use text transcript instead of audio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Load Processor and Model
    logger.info(f"Loading model from {args.model_path}...")
    
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load model directly without AutoConfig to avoid "unrecognized model" error
    try:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16, 
            device_map=args.device,
            trust_remote_code=True
        )
    except Exception as e:
        logger.info(f"Qwen2AudioForConditionalGeneration failed: {e}")
        logger.info("Falling back to AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16, 
            device_map=args.device,
            trust_remote_code=True
        )

    # Load Dataset
    logger.info(f"Loading dataset from {args.test_file}...")
    items = build_items(args.test_file, args.audio_dir, use_all_recordings=False, add_text_only=args.add_text_only)
    
    if args.max_samples:
        logger.info(f"Limiting to {args.max_samples} samples for dry run.")
        items = items[:args.max_samples]
    
    logger.info(f"Num examples: {len(items)}")
    
    predictions = []
    
    # Inference Loop
    model.eval()
    with torch.no_grad():
        for item in tqdm(items):
            # Prepare inputs
            transcript = item.get("transcript", "")
            prompt_text = f"{PROMPT}\nTranscript: {transcript}" if args.add_text_only else PROMPT
            
            messages = []
            content = []
            
            if not args.add_text_only:
                audio_input = item.get("audio") or item.get("audio_path")
                if audio_input:
                    # Load and process audio
                    # Note: We process one by one for simplicity and correctness with Qwen2Audio
                    # For batched inference, padding handling is complex.
                    audio_tensor = load_audio_input(audio_input, processor.feature_extractor.sampling_rate)
                    content.append({"type": "audio", "audio": audio_input}) # Processor will handle path or we need to pass tensor? 
                    # Actually apply_chat_template expects paths or specific format.
                    # Let's check how processor handles inputs.
                    # Qwen2AudioProcessor.apply_chat_template prepares text prompt.
                    # Then we call processor(images/audios=..., text=...)
                    pass
            
            content.append({"type": "text", "text": prompt_text})
            messages.append({"role": "user", "content": content})
            
            # Apply chat template
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Prepare model inputs
            # If using audio, we need to load it. 
            # item["audio"] is typically a path string in our build_items
            audios = []
            if not args.add_text_only:
                 # Check if the inputs need to be loaded raw or if processor handles paths
                 # Qwen2Audio processor usually handles list of audios. 
                 # But let's look at how the training script does it: 
                 # It calls processor(text=..., audios=..., ...)
                 # We'll load the audio tensor manually to be safe/consistent with training script utils
                 audio_path = item.get("audio") or item.get("audio_path")
                 if audio_path:
                     # We use the tensor loaded by helper or pass path? 
                     # The processor CAN accept paths, but let's stick to what we know works.
                     # Actually, to use 'audios' arg in processor, we often pass the raw waveform list.
                     if isinstance(audio_path, str):
                        import librosa 
                        # Use our load_audio helper
                        from train_qwen2_audio_slurp import load_audio
                        wav = load_audio(audio_path, target_sr=processor.feature_extractor.sampling_rate)
                        audios.append(wav.numpy()) # processor expects numpy or tensor
            
            inputs = processor(
                text=[text],
                audios=audios if audios else None,
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(model.device)
            
            # Generate
            # max_new_tokens=128 should be enough for JSON output
            generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            
            # Decode
            # Only decode the new tokens
            generated_ids = generated_ids[:, inputs.input_ids.size(1):]
            response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Parse JSON
            parsed = extract_json(response_text)
            
            # Add to predictions
            # evaluate.py expects: {"file": filename or "slurp_id": id, "scenario": ..., ...}
            # key depends on load_gold=False (implies 'file' key) or True (implies 'slurp_id' key)
            # We'll include both if possible, but evaluate.py logic:
            # result[example.pop("slurp_id" if load_gold else "file")]
            # So we should output slurp_id if we plan to use --load-gold, which is better for direct comparison.
            # However, build_items returns items with 'slurp_id'.
            
            pred_entry = {
                "slurp_id": item["slurp_id"],
                "file": item.get("audio_path", "unknown"), # evaluate.py uses 'file' if load_gold=False
                "scenario": parsed.get("scenario", ""),
                "action": parsed.get("action", ""),
                "entities": parsed.get("entities", [])
            }
            # Write line immediately (or collect) - let's write to file at the end
            predictions.append(pred_entry)
            
    # Save predictions
    logger.info(f"Saving predictions to {args.output_file}...")
    with open(args.output_file, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
            
    # Run Evaluation Script
    gold_file = args.gold_file if args.gold_file else args.test_file
    logger.info(f"Running evaluation against {gold_file}...")
    
    # We use --load-gold because we are predicting against the gold jsonl (test.jsonl)
    # matching by slurp_id is more robust than filename.
    eval_cmd = [
        "python", "slurp/scripts/evaluation/evaluate.py",
        "--gold-data", gold_file,
        "--prediction-file", args.output_file,
        "--load-gold"
    ]
    
    subprocess.run(eval_cmd, check=False)

if __name__ == "__main__":
    main()
