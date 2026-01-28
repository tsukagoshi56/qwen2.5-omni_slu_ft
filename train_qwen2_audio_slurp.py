import argparse
import json
import random
import logging
import os
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, AutoModelForCausalLM

# Import helper functions from training script
try:
    from train_qwen2_audio_slurp import (
        PROMPT,
        build_items,
        SlurpDataset,
        load_audio_input,
        load_speech_massive_split,
        SpeechMassiveDataset,
    )
except ImportError:
    raise ImportError("Could not import from train_qwen2_audio_slurp.py. Make sure it is in the same directory.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_json_robust(text: str) -> dict:
    """
    Robust JSON extraction that handles markdown code blocks and incomplete strings.
    """
    # 1. Remove Markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # 2. Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Try finding the outermost bracket pair
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 4. Regex fallback (Last resort for malformed JSON)
    try:
        scenario_match = re.search(r'"scenario":\s*"([^"]+)"', text)
        action_match = re.search(r'"action":\s*"([^"]+)"', text)
        
        scenario = scenario_match.group(1) if scenario_match else "none"
        action = action_match.group(1) if action_match else "none"
        
        # Simple entity extraction regex (limited support)
        entities = []
        # Try to find entities block
        return {"scenario": scenario, "action": action, "entities": entities}
    except Exception:
        pass

    # Return structure indicating failure, but keeping raw output for debug
    return {"scenario": "error", "action": "error", "entities": [], "error": "json_parse_failed"}

def bio_to_spans(tokens, labels):
    """Convert bio labels to spans for evaluation (SpeechMassive helper)."""
    spans = []
    current_type = None
    current_span = []
    
    for i, label in enumerate(labels):
        if label in ["O", "Other"]:
            if current_type:
                spans.append({"type": current_type, "span": current_span})
                current_type = None
                current_span = []
            continue
            
        if label.startswith("B-"):
            if current_type:
                spans.append({"type": current_type, "span": current_span})
            current_type = label[2:]
            current_span = [i]
        elif label.startswith("I-"):
            if current_type == label[2:]:
                current_span.append(i)
            else:
                if current_type:
                     spans.append({"type": current_type, "span": current_span})
                current_type = label[2:]
                current_span = [i]
                
    if current_type:
        spans.append({"type": current_type, "span": current_span})
    return spans

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on SLURP dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or checkpoint")
    parser.add_argument("--dataset", type=str, default="slurp", choices=["slurp", "speech_massive"], help="Dataset to evaluate on")
    parser.add_argument("--test_file", type=str, default="slurp/dataset/slurp/test.jsonl", help="Path to test data (jsonl) for SLURP")
    parser.add_argument("--massive_dataset_config", type=str, default="en-US", help="Config for SpeechMassive (e.g. en-US)")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio", help="Path to audio directory")
    parser.add_argument("--output_dir", type=str, default="inference_outputs", help="Base directory for output predictions")
    parser.add_argument("--output_file", type=str, default=None, help="Specific output filename")
    parser.add_argument("--num_samples", type=int, default=None, help="Randomly select N samples")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (first N)")
    parser.add_argument("--add_text_only", action="store_true", help="Use text transcript instead of audio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Match training debug length")
    
    args = parser.parse_args()
    
    # --- Setup Device ---
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # --- Load Processor & Model ---
    logger.info(f"Loading processor and model from {args.model_path}...")
    
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, fix_mistral_regex=True)
    except Exception as e:
        logger.warning(f"Failed to load processor from checkpoint: {e}. Trying base model Qwen/Qwen2-Audio-7B-Instruct...")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True)

    # Ensure clean generation config
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Load Model
    # Try loading as Qwen2Audio, fallback to AutoModel
    try:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True
        )
    except Exception:
        # Support for PEFT/LoRA if loaded via AutoModel logic or if structure differs
        try:
            from peft import PeftModel, PeftConfig
            config = PeftConfig.from_pretrained(args.model_path)
            base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=args.device,
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, args.model_path)
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map=args.device,
                trust_remote_code=True
            )

    model.eval()
    logger.info("Model loaded successfully.")

    # --- Prepare Output ---
    model_name = os.path.basename(os.path.normpath(args.model_path))
    if args.output_file:
        output_file_path = args.output_file
    else:
        out_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)
        output_file_path = os.path.join(out_dir, "predictions.jsonl")

    # --- Load Data ---
    if args.dataset == "speech_massive":
        logger.info(f"Loading SpeechMassive dataset ({args.massive_dataset_config})...")
        massive_data = load_speech_massive_split("FBK-MT/Speech-MASSIVE", args.massive_dataset_config, "test", None)
        dataset = SpeechMassiveDataset(massive_data, "utt", "Other", args.add_text_only, False)
        # Generate Gold file logic omitted for brevity, focusing on inference
    else:
        logger.info(f"Loading SLURP dataset from {args.test_file}...")
        items = build_items(
            args.test_file, 
            args.audio_dir, 
            use_all_recordings=(not args.add_text_only), 
            add_text_only=args.add_text_only,
            train_text_only=args.add_text_only
        )
        dataset = SlurpDataset(items)

    # Subset Logic
    if args.num_samples and len(dataset) > args.num_samples:
        indices = random.sample(range(len(dataset)), args.num_samples)
        dataset = [dataset[i] for i in indices] # Treat as list
    elif args.max_samples:
        dataset = [dataset[i] for i in range(min(len(dataset), args.max_samples))]

    logger.info(f"Starting inference on {len(dataset)} samples...")
    predictions = []

    # --- Inference Loop ---
    # STRICT SETTING: Disable batching logic to prevent padding issues with Audio
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        
        transcript = item.get("transcript", "")
        
        # Prompt Logic: Must match training exactly
        # Training Logic: 
        #   If Audio: PROMPT only
        #   If Text-only: Transcript + PROMPT
        has_audio = (not args.add_text_only) and (item.get("audio_path") or item.get("audio"))
        
        if args.add_text_only:
             prompt_text = f"{transcript}\n{PROMPT}" if transcript else PROMPT
        else:
             prompt_text = PROMPT

        # Build Chat Message
        user_content = []
        audio_input = None
        
        if has_audio:
            # Determine source
            src = item.get("audio_path") or item.get("audio")
            audio_ref = item.get("audio_ref") or (src if isinstance(src, str) else "audio")
            
            user_content.append({"type": "audio", "audio": audio_ref}) # Placeholder for template
            
            # Load actual audio
            try:
                # Use fixed SR 16000
                audio_input = load_audio_input(src, 16000)
            except Exception as e:
                logger.error(f"Error loading audio for item {i}: {e}")
                continue
        
        user_content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": user_content}]

        # Apply Template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Prepare Inputs
        inputs = {}
        if audio_input is not None:
            # Audio Feature Extraction
            audio_np = audio_input.numpy() if isinstance(audio_input, torch.Tensor) else audio_input
            if audio_np.ndim > 1: audio_np = audio_np.flatten()
            
            # Use padding="max_length" to match training collator behavior
            # Default training max_length for audio is usually 3000 (30s) in Qwen processor
            audio_features = processor.feature_extractor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length", 
                return_attention_mask=True,
            )
            
            # Tokenize Text (NO PADDING here, as batch_size=1)
            # Important: Do not set padding=True here!
            text_tokens = processor.tokenizer(text, return_tensors="pt", padding=False)
            
            inputs["input_ids"] = text_tokens["input_ids"]
            inputs["attention_mask"] = text_tokens["attention_mask"]
            inputs["input_features"] = audio_features["input_features"]
            if "attention_mask" in audio_features:
                inputs["feature_attention_mask"] = audio_features["attention_mask"]
        else:
            inputs = processor.tokenizer(text, return_tensors="pt", padding=False)

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                # CRITICAL: Use Greedy Search (num_beams=1) for structured JSON
                num_beams=1,
                do_sample=False,
                repetition_penalty=1.0, # Keep neutral unless looping occurs
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )

        # Decode
        # Slice off input tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids_new = generated_ids[:, input_len:]
        
        response_text = processor.batch_decode(generated_ids_new, skip_special_tokens=True)[0]

        # Parse JSON
        parsed = extract_json_robust(response_text)
        
        # Normalize Entities
        entities_list = parsed.get("entities", [])
        standard_entities = []
        if isinstance(entities_list, list):
            for ent in entities_list:
                if isinstance(ent, dict):
                    # Flatten {"type": "value"} structure
                    for k, v in ent.items():
                        standard_entities.append({"type": str(k), "filler": str(v)})

        # Save Entry
        # Use filename for key if audio mode, slurp_id if text mode
        if args.add_text_only:
            key_field = "slurp_id"
            key_val = str(item.get("slurp_id", f"id_{i}"))
        else:
            key_field = "file"
            src = item.get("audio_path") or item.get("audio")
            key_val = os.path.basename(src) if isinstance(src, str) else f"audio_{i}"

        pred_entry = {
            key_field: key_val,
            "scenario": parsed.get("scenario", "none"),
            "action": parsed.get("action", "none"),
            "entities": standard_entities,
            "raw_output": response_text # Keep for debugging
        }
        predictions.append(pred_entry)

        # Debug print first few
        if i < 3:
            print(f"\n--- Sample {i} ---")
            print(f"Target: {item.get('target', '')}")
            print(f"Pred:   {response_text}")

    # Write Output
    logger.info(f"Saving predictions to {output_file_path}...")
    with open(output_file_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("Done.")

if __name__ == "__main__":
    main()