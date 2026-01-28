import argparse
import json
import random
import logging
import os
import subprocess
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, AutoModelForCausalLM

# Import helper functions from training script
try:
    from train_qwen2_audio_slurp import (
        PROMPT,
        build_items,
        SlurpDataset,
        load_audio,
        load_audio_input,
        load_speech_massive_split,
        SpeechMassiveDataset,
        resolve_slurp_root
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

def bio_to_spans(tokens, labels):
    """Convert bio labels to spans for evaluation."""
    spans = []
    current_type = None
    current_span = []
    
    for i, label in enumerate(labels):
        if label == "O" or label == "Other":
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
                # Mismatch or new start without B (shouldn't happen in valid BIO but handle it)
                if current_type:
                     spans.append({"type": current_type, "span": current_span})
                current_type = label[2:]
                current_span = [i]
                
    if current_type:
        spans.append({"type": current_type, "span": current_span})
        
    # Convert to format expected by evaluate.py/util.py logic:
    # Entities: [{"type": ..., "span": [...]}]
    # But util.py expects "entities" list in "gold". 
    # And tokens list of dicts.
    return spans

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on SLURP dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or checkpoint")
    parser.add_argument("--dataset", type=str, default="slurp", choices=["slurp", "speech_massive"], help="Dataset to evaluate on")
    parser.add_argument("--test_file", type=str, default="slurp/dataset/slurp/test.jsonl", help="Path to test data (jsonl) for SLURP")
    parser.add_argument("--massive_dataset_config", type=str, default="en-US", help="Config for SpeechMassive (e.g. en-US)")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio", help="Path to audio directory")
    parser.add_argument("--output_dir", type=str, default="inference_outputs", help="Base directory for output predictions")
    parser.add_argument("--output_file", type=str, default=None, help="Specific output filename (optional, overrides auto-naming)")
    parser.add_argument("--gold_file", type=str, default=None, help="Path to gold file for evaluation script (defaults to test_file)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (for dry run - first N)")
    parser.add_argument("--num_samples", type=int, default=None, help="Randomly select N samples for testing")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search size (1=greedy to match training)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--add_text_only", action="store_true", help="Use text transcript instead of audio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_flash_attn", action="store_true", default=False, help="Use Flash Attention 2 if available")
    parser.add_argument("--debug", action="store_true", default=False, help="Print raw model output for first N samples")
    parser.add_argument("--debug_count", type=int, default=5, help="Number of samples to show in debug mode")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens to generate (128 matches training)")
    
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty (1.0 = no penalty)")
    parser.add_argument("--no_transcript", action="store_true", help="Do NOT include transcript in the prompt (Audio -> JSON only)")
    parser.add_argument("--force_audio", action="store_true", help="Force audio evaluation even if model config suggests text_only_mode")
    
    args = parser.parse_args()
    
    # --- DDP Initialization for torchrun ---
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        # Force device to local rank
        args.device = f"cuda:{local_rank}"
        if local_rank == 0:
            logger.info(f"Initialized DDP. World Size: {world_size}")
    else:
        # Default behavior
        pass

    # Load Processor and Model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
        
    if local_rank <= 0:
        logger.info(f"Loading model from {args.model_path}...")

    # Determine Output Path
    model_name = os.path.basename(os.path.normpath(args.model_path))
    if args.output_file:
        output_file = args.output_file
    else:
        output_dir = os.path.join(args.output_dir, model_name)
        # Only rank 0 creates directory
        if local_rank <= 0:
            os.makedirs(output_dir, exist_ok=True)
        # Barrier to ensure dir exists
        if is_distributed:
            torch.distributed.barrier()
            
        output_file = os.path.join(output_dir, "predictions.jsonl")

    # If distributed, append rank to output file so processes don't overwrite
    if is_distributed:
        output_file = f"{output_file}.rank{local_rank}"

    if local_rank <= 0:
        logger.info(f"Predictions will be saved to: {output_file}")
    
    # Check if it's a PEFT adapter
    is_adapter = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        # In DDP, strict device mapping avoids OOM on rank 0
        "device_map": args.device if not is_distributed else {"": local_rank},
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
                processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, fix_mistral_regex=True)
            except Exception:
                logger.info(f"Failed to load processor from {args.model_path}, trying base model {base_model_path}")
                processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True, fix_mistral_regex=True)
            
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
        # Standard loading logic: Load both processor and model from the same path
        # This assumes args.model_path points to the directory containing both model and saved processor
        processor = None
        try:
            logger.info(f"Loading processor from: {args.model_path}")
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, fix_mistral_regex=True)
        except Exception as e:
            logger.warning(f"Failed to load full processor from {args.model_path}: {e}")
            # Maybe it's a checkpoint with only tokenizer, try to load tokenizer and feature_extractor separately
            try:
                from transformers import AutoTokenizer, AutoFeatureExtractor
                logger.info("Trying to load tokenizer from checkpoint and feature_extractor from base model...")
                
                # Detect base model from config
                config_path = os.path.join(args.model_path, "config.json")
                base_model_name = "Qwen/Qwen2-Audio-7B-Instruct"  # Default
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config_data = json.load(f)
                        base_model_name = config_data.get("_name_or_path", base_model_name)
                
                logger.info(f"Loading feature_extractor from base model: {base_model_name}")
                processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True, fix_mistral_regex=True)
                
                # Now replace the tokenizer with the one from checkpoint
                logger.info(f"Loading tokenizer from checkpoint: {args.model_path}")
                tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
                processor.tokenizer = tokenizer
                logger.info("Successfully loaded processor with checkpoint tokenizer and base feature_extractor")
            except Exception as e2:
                raise RuntimeError(f"Failed to load processor: {e} | Also failed fallback: {e2}")

        try:
            logger.info(f"Loading model from: {args.model_path}")
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

    logger.info(f"processor tokenizer name_or_path: {processor.tokenizer.name_or_path}")
    logger.info(f"model vocab_size: {getattr(model.config, 'vocab_size', None)}")
    logger.info(f"tokenizer len: {len(processor.tokenizer)}")
    logger.info(f"tokenizer vocab_size (base): {processor.tokenizer.vocab_size}")
    logger.info(f"tokenizer added_tokens_encoder: {processor.tokenizer.added_tokens_encoder}")
    logger.info(f"tokenizer pad_token: {processor.tokenizer.pad_token} (id={processor.tokenizer.pad_token_id})")
    logger.info(f"tokenizer eos_token: {processor.tokenizer.eos_token} (id={processor.tokenizer.eos_token_id})")
    logger.info(f"tokenizer bos_token: {processor.tokenizer.bos_token} (id={getattr(processor.tokenizer, 'bos_token_id', None)})")

    # Critical: Only EXPAND embeddings if tokenizer is larger than model
    # NEVER shrink - if model vocab > tokenizer, keep model's embeddings (they were trained)
    model_vocab = getattr(model.config, 'vocab_size', None)
    tokenizer_len = len(processor.tokenizer)
    if model_vocab is not None and tokenizer_len > model_vocab:
        logger.warning(f"Vocab size mismatch! Model: {model_vocab}, Tokenizer: {tokenizer_len}")
        logger.info(f"Resizing model embeddings to {tokenizer_len} to match tokenizer...")
        model.resize_token_embeddings(tokenizer_len)
    elif model_vocab is not None and tokenizer_len < model_vocab:
        logger.warning(f"Model vocab ({model_vocab}) > Tokenizer len ({tokenizer_len}). Keeping model's larger vocab size (trained embeddings preserved).")


    
    # Ensure pad_token is set (crucial for batch generation)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {processor.tokenizer.pad_token}")

    if args.add_text_only:
        logger.info("Using text-only mode (--add_text_only flag provided)")
    elif args.force_audio:
        logger.info("Using audio mode (--force_audio flag provided)")

    # Load Dataset
    if args.dataset == "speech_massive":
        logger.info(f"Loading SpeechMassive dataset ({args.massive_dataset_config})...")
        massive_data = load_speech_massive_split(
            "FBK-MT/Speech-MASSIVE", args.massive_dataset_config, "test", None 
        )
        dataset_obj = SpeechMassiveDataset(
            massive_data, "utt", "Other", args.add_text_only, False
        )
        items = dataset_obj # SpeechMassiveDataset behaves like list of items in some ways but it's a Dataset
        
        # Build gold file for SpeechMassive
        gold_file_path = os.path.join(os.path.dirname(output_file), "gold.jsonl")
        logger.info(f"Generating gold file for SpeechMassive at {gold_file_path}")
        with open(gold_file_path, "w") as f:
            for i in range(len(dataset_obj)):
                item = dataset_obj[i]
                # Reconstruct tokens info
                tokens = item.get("tokens", [])
                labels = item.get("labels", [])
                
                # If audio is derived from HF, audio_ref is path
                audio_path = item.get("audio_ref")
                if not audio_path and isinstance(item.get("audio"), dict):
                     audio_path = item["audio"].get("path")
                
                file_key = os.path.basename(audio_path) if audio_path else f"unknown_{i}"

                gold_entry = {
                    "tokens": [{"surface": t} for t in tokens],
                    "scenario": item.get("scenario", "none"),
                    "action": item.get("action", "none"),
                    "entities": bio_to_spans(tokens, labels),
                    "recordings": [{"file": file_key}]
                }
                # Fallback for text-only mode if needed? 
                # Evaluator util.py expects 'recordings' list if load_gold=False
                f.write(json.dumps(gold_entry) + "\n")
        
        # For compatibility with loop below
        dataset = dataset_obj 
        
    else:
        logger.info(f"Loading dataset from {args.test_file}...")
        # Audio mode: use all recordings (each file is a separate prediction keyed by filename)
        # Text-only mode: one item per slurp_id (prediction keyed by slurp_id)
        use_all = not args.add_text_only
        # For text-only evaluation, set train_text_only=True to prevent audio items from being added
        items = build_items(
            args.test_file, 
            args.audio_dir, 
            use_all_recordings=use_all, 
            add_text_only=args.add_text_only,
            train_text_only=args.add_text_only  # Prevents audio items when doing text-only eval
        )
        dataset = SlurpDataset(items)

    if args.num_samples:
        if local_rank <= 0:
            logger.info(f"Randomly selecting {args.num_samples} samples.")
        
        # Ensure reproducibility for random selection
        random.seed(42)
        
        if isinstance(items, list):
            if len(items) > args.num_samples:
                items = random.sample(items, args.num_samples)
            dataset = SlurpDataset(items)
        else:
            # Subset torch dataset (SpeechMassive)
            total_len = len(dataset)
            num_to_select = min(total_len, args.num_samples)
            indices = random.sample(range(total_len), num_to_select)
            dataset = torch.utils.data.Subset(dataset, indices)
            # Update items if it was just an alias to dataset
            items = dataset

    elif args.max_samples:
        if local_rank <= 0:
            logger.info(f"Limiting to {args.max_samples} samples for dry run.")
        if isinstance(items, list):
            items = items[:args.max_samples]
            dataset = SlurpDataset(items)
        else:
            # Subset torch dataset
            dataset = torch.utils.data.Subset(dataset, range(args.max_samples))
            items = dataset
    
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=False
        )
        if local_rank <= 0:
            logger.info(f"Num examples per rank: {len(sampler)}")
    else:
        sampler = None
    
    if local_rank <= 0:
        logger.info(f"Num examples: {len(dataset)}")
    
    predictions = []
    
    # Inference Loop
    model.eval()
    
    # Set padding side to right to match training configuration (critical for RoPE models)
    if processor.tokenizer.padding_side != "right":
        if local_rank <= 0:
            logger.info("Setting tokenizer padding_side to 'right' to match training")
        processor.tokenizer.padding_side = "right"
    
    # Remove EvalCollator and DataLoader
    # We will process items one by one to ensure exact match with training logic and avoid batching issues
    
    logger.info(f"Starting inference (no DataLoader, sequential processing)...")
    
    # Pre-calculate audio sampling rate
    audio_sr = processor.feature_extractor.sampling_rate if hasattr(processor, 'feature_extractor') else 16000

    # Manual iteration wrapper
    # local_rank 0 の場合のみ表示を出し、それ以外は静かに処理
    data_iterator = enumerate(dataset)
    if local_rank <= 0:
        data_iterator = tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating")
    
    # Storage for predictions (we will append to this list)
    # Note: 'predictions' list is already defined above
    
    # Handle DDP sharding if necessary (manual sharded loop)
    my_indices = range(len(dataset))
    if is_distributed:
        total_len = len(dataset)
        chunk_size = (total_len + world_size - 1) // world_size
        start_idx = local_rank * chunk_size
        end_idx = min(start_idx + chunk_size, total_len)
        my_indices = range(start_idx, end_idx)

    for i in my_indices:
        item = dataset[i]
        # --- Logic exactly matching SampleGenerationCallback in train_qwen2_audio_slurp.py ---
        
        transcript = item.get("transcript", "")
        target = item.get("target", "{}") # Default to empty JSON if missing
        
        # Determine prompt
        if args.add_text_only:
             prompt_text = f"{transcript}\n{PROMPT}" if transcript else PROMPT
        else:
            # Audio mode: do NOT include transcript in prompt (matches training logic)
            prompt_text = PROMPT
        
        # Load audio
        audio_input = item.get("audio_path") or item.get("audio")
        audio = None
        if not args.add_text_only and audio_input:
            try:
                # Always use 16000 for Qwen2-Audio consistency
                audio = load_audio_input(audio_input, 16000)
            except Exception as e:
                logger.warning(f"Failed to load audio for sample {i}: {e}")

        # Build text WITHOUT chat template (matches SampleGenerationCallback in training)
        if audio is not None:
            text = f"<|AUDIO|>\n{prompt_text}"
        else:
            text = f"{transcript}\n{prompt_text}" if transcript else prompt_text
        
        # Prepare inputs (matches SampleGenerationCallback logic)
        if audio is not None:
            audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
            if audio_np.ndim > 1:
                audio_np = audio_np.flatten() # Ensure 1D
            
            # Use processor directly like in training's SampleGenerationCallback
            inputs = processor(
                text=text,
                audios=[audio_np],
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = processor.tokenizer(text, return_tensors="pt", padding=True)
            
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Debug: print input details
        if i == 0 and local_rank <= 0:
            logger.info(f"DEBUG: input_ids shape: {inputs['input_ids'].shape}")
            logger.info(f"DEBUG: input keys: {list(inputs.keys())}")
            if 'input_features' in inputs:
                logger.info(f"DEBUG: input_features shape: {inputs['input_features'].shape}")
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
                repetition_penalty=args.repetition_penalty
            )
        
        # Match training's SampleGenerationCallback decode logic exactly
        generated_ids_list = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs['input_ids'], generated_ids)
        ]
        response_text = processor.batch_decode(generated_ids_list, skip_special_tokens=True)[0]
        
        # Debug: print generation details
        if i == 0 and local_rank <= 0:
            logger.info(f"DEBUG: generated_ids shape: {generated_ids.shape}")
            logger.info(f"DEBUG: generated_ids_list[0] length: {len(generated_ids_list[0])}")
            logger.info(f"DEBUG: raw response: {repr(response_text)}")
        
        # --- Display similar to SampleGenerationCallback ---
        if local_rank <= 0:
            print(f"\n--- Sample {i+1} ---", flush=True)
            print(f"Input (len={len(transcript)}): {transcript}", flush=True)
            print(f"Target: {target}", flush=True)
            print(f"Pred:   {response_text}", flush=True)

        # --- JSON Post-processing and saving ---
        parsed = extract_json(response_text)
        
        # Determine format (paper_v2 etc)
        use_paper_format = False
        if hasattr(model, "config") and getattr(model.config, "slurp_fmt", None) == "paper_v2":
            use_paper_format = True
        
        entities_list = parsed.get("entities", [])
        standard_entities = []
        for ent in entities_list:
            if isinstance(ent, dict):
                # Try to normalize based on format
                if "type" in ent and "filler" in ent:
                    standard_entities.append({"type": str(ent["type"]), "filler": str(ent["filler"]).lower()})
                else:
                    for k, v in ent.items():
                        standard_entities.append({"type": str(k), "filler": str(v).lower()})

        if args.add_text_only:
            pred_entry = {
                "slurp_id": str(item.get("slurp_id", "unknown")),
                "scenario": parsed.get("scenario", "none"),
                "action": parsed.get("action", "none"),
                "entities": standard_entities,
                "raw_output": response_text
            }
        else:
            audio_path = item.get("audio_path") or (audio_input if isinstance(audio_input, str) else "audio")
            file_key = os.path.basename(audio_path) if audio_path else f"unknown_{i}"
            pred_entry = {
                "file": file_key,
                "scenario": parsed.get("scenario", "none"),
                "action": parsed.get("action", "none"),
                "entities": standard_entities,
                "raw_output": response_text
            }

        predictions.append(pred_entry)
            
    # Save predictions
    # Save predictions
    logger.info(f"Saving predictions to {output_file}...")
    with open(output_file, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
            
    if is_distributed:
        torch.distributed.barrier()
        # Rank 0 merges files
        if local_rank == 0:
            logger.info("Merging DDP output files...")
            # Reconstruct base output filename (remove .rank0)
            base_output = output_file.replace(".rank0", "")
            
            all_lines = []
            for r in range(world_size):
                rank_file = f"{base_output}.rank{r}"
                if os.path.exists(rank_file):
                    with open(rank_file, "r") as rf:
                        all_lines.extend(rf.readlines())
                    # cleanup
                    os.remove(rank_file)
            
            with open(base_output, "w") as f_out:
                for line in all_lines:
                    f_out.write(line)
            logger.info(f"Merged output saved to {base_output}")
            
    # Evaluation is now handled by external shell scripts to properly manage keys (slurp_id vs file)
    if local_rank <= 0:
        logger.info(f"Predictions saved. Please run evaluation script manually.")

if __name__ == "__main__":
    main()
