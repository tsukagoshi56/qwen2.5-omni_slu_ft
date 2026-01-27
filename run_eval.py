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
        # 1. Try standard JSON extraction
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback to repair or regex if standard parsing fails
        import re
        try:
            # Common error: keys without quotes? (e.g. {scenario: "foo"})
            # Or missing commas. 
            # For now, let's try a simple regex to capture the main fields if JSON fails
            scenario_match = re.search(r'"scenario":\s*"([^"]+)"', text)
            action_match = re.search(r'"action":\s*"([^"]+)"', text)
            
            # If standard key-value pairs aren't found, try looser regex
            if not scenario_match:
                 scenario_match = re.search(r'scenario\W+([a-zA-Z_]+)', text)
            if not action_match:
                 action_match = re.search(r'action\W+([a-zA-Z_]+)', text)

            scenario = scenario_match.group(1) if scenario_match else "none"
            action = action_match.group(1) if action_match else "none"
            
            # Entities are harder to parse with regex, return empty list for safety on fallback
            return {"scenario": scenario, "action": action, "entities": []}
            
        except Exception as e:
            logger.warning(f"Failed to parse JSON (fallback also failed): {e} | Text: {text}")

    except Exception as e:
        logger.warning(f"Failed to parse JSON: {e} | Text: {text}")
    
    # Return empty valid structure if failed
    return {"scenario": "none", "action": "none", "entities": []}

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on SLURP dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or checkpoint")
    parser.add_argument("--test_file", type=str, default="slurp/dataset/slurp/test.jsonl", help="Path to test data (jsonl)")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio", help="Path to audio directory")
    parser.add_argument("--output_dir", type=str, default="inference_outputs", help="Base directory for output predictions")
    parser.add_argument("--output_file", type=str, default=None, help="Specific output filename (optional, overrides auto-naming)")
    parser.add_argument("--gold_file", type=str, default=None, help="Path to gold file for evaluation script (defaults to test_file)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (for dry run)")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--num_beams", type=int, default=3, help="Beam search size (default 3 per paper)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--add_text_only", action="store_true", help="Use text transcript instead of audio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_flash_attn", action="store_true", default=False, help="Use Flash Attention 2 if available")
    parser.add_argument("--debug", action="store_true", default=False, help="Print raw model output for first N samples")
    parser.add_argument("--debug_count", type=int, default=5, help="Number of samples to show in debug mode")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate (increase if output is truncated)")
    
    args = parser.parse_args()
    
    # Load Processor and Model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
        
    logger.info(f"Loading model from {args.model_path}...")

    # Determine Output Path
    model_name = os.path.basename(os.path.normpath(args.model_path))
    if args.output_file:
        output_file = args.output_file
    else:
        output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "predictions.jsonl")
    
    logger.info(f"Predictions will be saved to: {output_file}")
    
    # Check if it's a PEFT adapter
    is_adapter = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": args.device,
        "trust_remote_code": True,
    }
    
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 enabled.")

    if is_adapter:
        try:
            from peft import PeftConfig, PeftModel
            peft_config = PeftConfig.from_pretrained(args.model_path)
            base_model_path = peft_config.base_model_name_or_path
            logger.info(f"Detected LoRA adapter. Base model: {base_model_path}")
            
            # Load processor (prefer adapter path, fallback to base)
            try:
                processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
            except Exception:
                logger.info(f"Failed to load processor from {args.model_path}, trying base model {base_model_path}")
                processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
            
            # Load base model
            try:
                model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    base_model_path,
                    **model_kwargs
                )
            except Exception as e:
                logger.info(f"Qwen2AudioForConditionalGeneration failed for base model: {e}")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    **model_kwargs
                )
                
            # Load adapter
            logger.info(f"Loading LoRA adapter from {args.model_path}...")
            model = PeftModel.from_pretrained(model, args.model_path)
            logger.info("LoRA adapter loaded successfully.")
            
        except ImportError:
            logger.error("peft is required to load LoRA adapters. Please install it with 'pip install peft'.")
            raise
    else:
        # Standard loading logic
        processor = None
        processor_load_paths = [args.model_path]
        
        # If path looks like a checkpoint, try parent directory first
        if "checkpoint-" in os.path.basename(os.path.normpath(args.model_path)):
            parent_dir = os.path.dirname(os.path.normpath(args.model_path))
            processor_load_paths.insert(0, parent_dir)
            logger.info(f"Detected checkpoint path. Will try parent directory first: {parent_dir}")
        
        for load_path in processor_load_paths:
            try:
                logger.info(f"Attempting to load processor from: {load_path}")
                processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
                logger.info(f"Successfully loaded processor from: {load_path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load processor from {load_path}: {e}")
                continue
        
        # Fallback: Try reading _name_or_path from config.json to find base model
        if processor is None:
            try:
                config_path = os.path.join(args.model_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    base_model = config_dict.get("_name_or_path")
                    if base_model and base_model != args.model_path:
                        logger.info(f"Fallback: Loading processor from base model in config.json: {base_model}")
                        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
            except Exception as e2:
                logger.warning(f"Fallback (config.json) also failed: {e2}")
        
        if processor is None:
            raise RuntimeError(f"Could not load processor from any path. Tried: {processor_load_paths}")
        try:
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                args.model_path,
                **model_kwargs
            )
        except Exception as e:
            logger.info(f"Qwen2AudioForConditionalGeneration failed: {e}")
            logger.info("Falling back to AutoModelForCausalLM")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, 
                **model_kwargs
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
    
    # Set padding side to left for batch generation
    if processor.tokenizer.padding_side != "left":
        logger.info("Setting tokenizer padding_side to 'left' for batch inference")
        processor.tokenizer.padding_side = "left"
    
    from train_qwen2_audio_slurp import load_audio
    from torch.utils.data import DataLoader

    class EvalCollator:
        def __init__(self, processor, add_text_only):
            self.processor = processor
            self.add_text_only = add_text_only

        def __call__(self, batch):
            batch_texts = []
            batch_audios = []
            batch_items = []
            
            for item in batch:
                transcript = item.get("transcript", "")
                transcript = item.get("transcript", "")
                # Always include transcript in prompt to match training distribution, unless specifically not wanted?
                # The training script (train_qwen2_audio_slurp.py) creates PROMPT + "\nTranscript: " + transcript if include_transcript is True (default).
                # So we should mirror that here.
                prompt_text = f"{PROMPT}\nTranscript: {transcript}"
                
                content = []
                if not self.add_text_only:
                    audio_path = item.get("audio") or item.get("audio_path")
                    if audio_path:
                        content.append({"type": "audio", "audio": audio_path})
                
                content.append({"type": "text", "text": prompt_text})
                messages = [{"role": "user", "content": content}]
                
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_texts.append(text)
                batch_items.append(item)
                
                if not self.add_text_only:
                    audio_path = item.get("audio") or item.get("audio_path")
                    if audio_path and isinstance(audio_path, str):
                        wav = load_audio(audio_path, target_sr=self.processor.feature_extractor.sampling_rate)
                        batch_audios.append(wav.numpy())

            inputs = self.processor(
                text=batch_texts,
                audios=batch_audios if batch_audios else None,
                return_tensors="pt",
                padding=True
            )
            return inputs, batch_items

    dataset = SlurpDataset(items)
    collator = EvalCollator(processor, args.add_text_only)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collator
    )

    for inputs, batch_items in tqdm(dataloader):
        inputs = inputs.to(model.device)
        
        # Generate for the whole batch
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, do_sample=False)
        
        # Decode
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        responses = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Parse and collect predictions
        for item, response_text in zip(batch_items, responses):
            # Debug mode: print raw output for first N samples
            if args.debug and len(predictions) < args.debug_count:
                logger.info(f"=== DEBUG Sample {len(predictions)+1} ===")
                logger.info(f"Transcript: {item.get('transcript', 'N/A')[:100]}...")
                logger.info(f"Raw Model Output: {response_text}")
                logger.info("=" * 40)
            
            parsed = extract_json(response_text)
            
            # Convert entities from compact format {"type": "filler"} to standard format [{"type": "type", "filler": "filler"}]
            # This handles the mismatch between training target format and evaluation expectation
            # Check if using paper v2 format from model config
            use_paper_format = False
            if hasattr(model, "config") and getattr(model.config, "slurp_fmt", None) == "paper_v2":
                use_paper_format = True
            elif hasattr(model, "peft_config") and "base_model_name_or_path" in model.peft_config:
                 # If adapter, we might need to check base model config or assume based on something else
                 # For now, rely on base model config loaded into 'model'
                 if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
                     if getattr(model.base_model.config, "slurp_fmt", None) == "paper_v2":
                         use_paper_format = True

            entities_list = parsed.get("entities", [])
            standard_entities = []
            
            if use_paper_format:
                 # Convert [{"date": "thursday"}] -> [{"type": "date", "filler": "thursday"}]
                 for ent in entities_list:
                     if isinstance(ent, dict):
                         for k, v in ent.items():
                             standard_entities.append({"type": str(k), "filler": str(v)})
            else:
                # Existing logic for backward compat
                for ent in entities_list:
                    if "type" in ent and "filler" in ent:
                        standard_entities.append(ent)
                    else:
                        for k, v in ent.items():
                            standard_entities.append({"type": str(k), "filler": str(v)})

            pred_entry = {
                "scenario": parsed.get("scenario", ""),
                "action": parsed.get("action", ""),
                "entities": standard_entities
            }
            if args.add_text_only:
                pred_entry["slurp_id"] = str(item["slurp_id"])
            else:
                 # Audio mode uses file key
                 audio_path = item.get("audio") or item.get("audio_path")
                 if audio_path:
                     pred_entry["file"] = os.path.basename(audio_path)
                 else:
                     pred_entry["file"] = "unknown"
            predictions.append(pred_entry)
            
    # Save predictions
    logger.info(f"Saving predictions to {output_file}...")
    with open(output_file, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
            
    # Run Evaluation Script
    gold_file = args.gold_file if args.gold_file else args.test_file
    logger.info(f"Running evaluation against {gold_file}...")
    
    # We use --load-gold because we are predicting against the gold jsonl (test.jsonl)
    # matching by slurp_id is more robust than filename.
    eval_cmd = [
        "python", "scripts/evaluation/evaluate.py",
        "--gold-data", gold_file,
        "--prediction-file", output_file,
        "--load-gold"
    ]
    
    subprocess.run(eval_cmd, check=False)

if __name__ == "__main__":
    main()
