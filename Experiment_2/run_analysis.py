#!/usr/bin/env python3
"""
Confusable Pairs Analysis for Speech LLM (Qwen2-Audio)

This script performs inference while extracting:
- Logits (output probabilities)
- Hidden States (final and intermediate layers)
- Attention Weights (Audio-to-Token)

And generates:
- Confusion Matrix with top confusable pairs
- Sample classification (Success, Ambiguous Success, Ambiguous Failure, Fatal Failure)

Features are saved to disk for later visualization without re-running inference.
"""

import argparse
import json
import random
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, AutoModelForCausalLM

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from train_qwen2_audio_slurp import (
        PROMPT,
        build_items,
        SlurpDataset,
        load_audio_input,
        resolve_slurp_root,
    )
except ImportError:
    raise ImportError("Could not import from train_qwen2_audio_slurp.py. Make sure it is in the parent directory.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict:
    """Extract JSON object from the model output."""
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        import re
        try:
            scenario_match = re.search(r'"scenario":\s*"([^"]+)"', text)
            action_match = re.search(r'"action":\s*"([^"]+)"', text)
            
            if not scenario_match:
                scenario_match = re.search(r'scenario\W+([a-zA-Z_]+)', text)
            if not action_match:
                action_match = re.search(r'action\W+([a-zA-Z_]+)', text)

            scenario = scenario_match.group(1) if scenario_match else "none"
            action = action_match.group(1) if action_match else "none"
            
            return {"scenario": scenario, "action": action, "entities": []}
        except Exception:
            pass
    except Exception:
        pass
    
    return {"scenario": "none", "action": "none", "entities": []}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute prediction entropy from logits."""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy


def get_top_k_probs(logits: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get top-k probabilities and their indices."""
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k, dim=-1)
    return top_probs, top_indices


def classify_sample(
    is_correct: bool,
    confidence: float,
    margin: float,
    high_conf_threshold: float = 0.8,
    low_conf_threshold: float = 0.5,
    margin_threshold: float = 0.3
) -> str:
    """
    Classify sample into categories:
    - success: High confidence, correct
    - ambiguous_success: Low confidence but correct
    - ambiguous_failure: Low margin, incorrect (confused with similar label)
    - fatal_failure: High confidence, incorrect (completely wrong)
    """
    if is_correct:
        if confidence >= high_conf_threshold:
            return "success"
        else:
            return "ambiguous_success"
    else:
        if margin < margin_threshold:
            return "ambiguous_failure"
        else:
            return "fatal_failure"


class AnalysisCollator:
    """
    Collator for analysis that properly handles audio using feature_extractor and tokenizer separately.
    Matches the robust logic in run_eval.py (EvalCollator) to support batching.
    """
    
    def __init__(self, processor, add_text_only=False):
        self.processor = processor
        self.add_text_only = add_text_only

    def __call__(self, batch):
        batch_items = batch
        batch_texts = []
        batch_audios = []
        
        for item in batch:
            transcript = item.get("transcript", "")
            
            # Determine prompt text
            if self.add_text_only:
                if transcript:
                    prompt_text = f"{transcript}\n{PROMPT}"
                else:
                    prompt_text = PROMPT
            else:
                # Audio mode logic
                if item.get("audio_path"):
                    prompt_text = PROMPT
                elif transcript:
                    prompt_text = f"{transcript}\n{PROMPT}"
                else:
                    prompt_text = PROMPT
            
            # Load audio if needed
            audio = None
            if not self.add_text_only:
                audio_input = item.get("audio") or item.get("audio_path")
                if audio_input:
                    try:
                        # Use load_audio_input which handles string paths
                        target_sr = self.processor.feature_extractor.sampling_rate
                        audio = load_audio_input(audio_input, target_sr=target_sr)
                        
                        audio_ref = item.get("audio_ref")
                        if not audio_ref and isinstance(audio_input, str):
                            audio_ref = audio_input
                        if not audio_ref:
                            audio_ref = "audio"
                        # We do NOT add to user_content here per new instruction
                    except Exception as e:
                        logger.warning(f"Failed to load audio for {item}: {e}")
            
            # Build content for chat template
            # Do NOT create audio placeholder.
            user_content = [{"type": "text", "text": prompt_text}]
            messages = [{"role": "user", "content": user_content}]
            
            # Apply chat template
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_texts.append(text)
            
            if audio is not None:
                batch_audios.append(audio)
        
        # Process inputs using processor directly
        if batch_audios:
            inputs = self.processor(
                text=batch_texts,
                audio=batch_audios,
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
            )
            
        return inputs, batch_items, batch_texts


def run_analysis(args):
    """Main analysis function."""
    logger.info(f"Loading model from {args.model_path}...")
    
    # Output directory setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we can load cached features
    cached_features_path = output_dir / "cached_inference_data.pt"
    if args.use_cache and cached_features_path.exists():
        logger.info(f"Loading cached features from {cached_features_path}")
        cached_data = torch.load(cached_features_path)
        return analyze_cached_data(cached_data, args, output_dir)
    
    # Load model and processor
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": args.device,
        "trust_remote_code": True,
    }
    
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Check for PEFT adapter
    is_adapter = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
    
    if is_adapter:
        from peft import PeftConfig, PeftModel
        peft_config = PeftConfig.from_pretrained(args.model_path)
        base_model_path = peft_config.base_model_name_or_path
        logger.info(f"Detected LoRA adapter. Base model: {base_model_path}")
        
        try:
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, fix_mistral_regex=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True, fix_mistral_regex=True)
        
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            base_model_path, **model_kwargs
        )
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, fix_mistral_regex=True)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.model_path, **model_kwargs
        )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    
    # Load dataset
    logger.info(f"Loading dataset from {args.test_file}...")
    items = build_items(
        args.test_file,
        args.audio_dir,
        use_all_recordings=not args.add_text_only,
        add_text_only=args.add_text_only,
        train_text_only=args.add_text_only
    )
    
    if args.num_samples:
        random.seed(42)
        if len(items) > args.num_samples:
             items = random.sample(items, args.num_samples)
        logger.info(f"Randomly selected {len(items)} samples (requested {args.num_samples})")
    elif args.max_samples:
        items = items[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")
    
    dataset = SlurpDataset(items)
    logger.info(f"Dataset size: {len(dataset)}")
    
    collator = AnalysisCollator(processor, args.add_text_only)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator
    )
    
    model.eval()
    
    # Storage for extracted features
    all_hidden_states = []
    all_logits = []
    all_attentions = []
    all_results = []
    all_entropies = []
    all_predictions = []
    
    # For confusion matrix
    y_true = []
    y_pred = []
    
    logger.info("Starting inference with feature extraction...")
    
    for batch_idx, (inputs, batch_items, batch_texts) in enumerate(tqdm(dataloader)):
        # Move inputs to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Forward pass with hidden states and attentions
            forward_outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=args.save_attention,
                return_dict=True
            )
            
            # Generate for getting predictions
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # Extract hidden states (last layer, last token position)
        if forward_outputs.hidden_states:
            last_hidden = forward_outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            for i in range(last_hidden.size(0)):
                attn_mask = inputs.get("attention_mask")
                if attn_mask is not None:
                    last_pos = attn_mask[i].sum().item() - 1
                else:
                    last_pos = last_hidden.size(1) - 1
                hidden_vec = last_hidden[i, last_pos, :].cpu().float()
                all_hidden_states.append(hidden_vec)
        
        # Extract attention weights (optional)
        if args.save_attention and forward_outputs.attentions:
            for i in range(len(batch_items)):
                last_attn = forward_outputs.attentions[-1][i].cpu().float()
                all_attentions.append(last_attn)
        
        # Process generated outputs
        gen_sequences = gen_outputs.sequences[:, inputs["input_ids"].size(1):]
        responses = processor.batch_decode(gen_sequences, skip_special_tokens=True)
        
        # Extract logits/scores for entropy calculation
        if hasattr(gen_outputs, 'scores') and gen_outputs.scores:
            for i in range(len(batch_items)):
                if i < len(gen_outputs.scores[0]):
                    first_token_logits = gen_outputs.scores[0][i].cpu().float()
                    all_logits.append(first_token_logits)
                    
                    entropy = compute_entropy(first_token_logits.unsqueeze(0))
                    all_entropies.append(entropy.item())
        
        # Parse predictions and compute metrics
        for i, (item, response_text) in enumerate(zip(batch_items, responses)):
            parsed = extract_json(response_text)
            
            # Get ground truth from target
            try:
                target_data = json.loads(item.get("target", "{}"))
            except:
                target_data = {"scenario": "none", "action": "none"}
            
            gt_scenario = target_data.get("scenario", "none")
            gt_action = target_data.get("action", "none")
            pred_scenario = parsed.get("scenario", "none")
            pred_action = parsed.get("action", "none")
            
            # Combined label for confusion matrix
            gt_label = f"{gt_scenario}_{gt_action}"
            pred_label = f"{pred_scenario}_{pred_action}"
            
            y_true.append(gt_label)
            y_pred.append(pred_label)
            
            # Compute confidence
            entropy_val = all_entropies[len(all_results)] if len(all_entropies) > len(all_results) else 0.0
            max_entropy = 10.0
            confidence = max(0, 1 - entropy_val / max_entropy)
            
            is_correct = (gt_scenario == pred_scenario and gt_action == pred_action)
            
            classification = classify_sample(
                is_correct=is_correct,
                confidence=confidence,
                margin=confidence,
                high_conf_threshold=args.high_conf_threshold,
                low_conf_threshold=args.low_conf_threshold
            )
            
            result = {
                "index": len(all_results),
                "slurp_id": item.get("slurp_id"),
                "audio_path": item.get("audio_path"),
                "transcript": item.get("transcript", "")[:100],
                "gt_scenario": gt_scenario,
                "gt_action": gt_action,
                "pred_scenario": pred_scenario,
                "pred_action": pred_action,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "is_correct": is_correct,
                "entropy": entropy_val,
                "confidence": confidence,
                "classification": classification,
                "raw_output": response_text[:200]
            }
            all_results.append(result)
            all_predictions.append({
                "slurp_id": item.get("slurp_id"),
                "response": response_text,
                "parsed": parsed
            })
    
    # Save all extracted features to disk for later use
    logger.info("Saving extracted features to disk...")
    
    cached_data = {
        "hidden_states": torch.stack(all_hidden_states) if all_hidden_states else None,
        "logits": torch.stack(all_logits) if all_logits else None,
        "attentions": all_attentions if all_attentions else None,
        "results": all_results,
        "predictions": all_predictions,
        "y_true": y_true,
        "y_pred": y_pred,
        "model_path": args.model_path,
        "args": vars(args)
    }
    
    torch.save(cached_data, cached_features_path)
    logger.info(f"Cached inference data saved to {cached_features_path}")
    
    # Run analysis on the data
    return analyze_cached_data(cached_data, args, output_dir)


def analyze_cached_data(cached_data: Dict, args, output_dir: Path) -> Dict:
    """Analyze cached data and generate outputs."""
    logger.info("Analyzing cached data...")
    
    all_results = cached_data["results"]
    y_true = cached_data["y_true"]
    y_pred = cached_data["y_pred"]
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    unique_labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Find confusable pairs
    confusable_pairs = []
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:
                mutual_error = cm[i, j] + cm[j, i]
                if mutual_error > 0:
                    confusable_pairs.append({
                        "label_1": label_i,
                        "label_2": label_j,
                        "error_1_to_2": int(cm[i, j]),
                        "error_2_to_1": int(cm[j, i]),
                        "mutual_error": int(mutual_error)
                    })
    
    confusable_pairs.sort(key=lambda x: -x["mutual_error"])
    top_confusable = confusable_pairs[:args.top_k_confusable]
    
    logger.info(f"Top {len(top_confusable)} confusable pairs:")
    for pair in top_confusable:
        logger.info(f"  {pair['label_1']} <-> {pair['label_2']}: {pair['mutual_error']} errors")
    
    # Classification statistics
    classification_counts = defaultdict(int)
    for r in all_results:
        classification_counts[r["classification"]] += 1
    
    # Save results
    logger.info("Saving analysis results...")
    
    # Save hidden states
    if cached_data.get("hidden_states") is not None:
        torch.save(cached_data["hidden_states"], output_dir / "hidden_states.pt")
        logger.info(f"Saved hidden states: {cached_data['hidden_states'].shape}")
    
    # Save logits
    if cached_data.get("logits") is not None:
        torch.save(cached_data["logits"], output_dir / "logits.pt")
        logger.info(f"Saved logits: {cached_data['logits'].shape}")
    
    # Save attention weights
    if cached_data.get("attentions"):
        torch.save(cached_data["attentions"], output_dir / "attention_weights.pt")
        logger.info(f"Saved {len(cached_data['attentions'])} attention maps")
    
    # Save confusion matrix
    import numpy as np
    np.save(output_dir / "confusion_matrix.npy", cm)
    with open(output_dir / "confusion_labels.json", "w") as f:
        json.dump(unique_labels, f, indent=2)
    
    # Save analysis results
    analysis_summary = {
        "model_path": cached_data.get("model_path", "unknown"),
        "num_samples": len(all_results),
        "accuracy": sum(1 for r in all_results if r["is_correct"]) / len(all_results) if all_results else 0,
        "classification_counts": dict(classification_counts),
        "top_confusable_pairs": top_confusable,
        "avg_entropy": sum(r["entropy"] for r in all_results) / len(all_results) if all_results else 0,
    }
    
    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "sample_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    samples_by_class = defaultdict(list)
    for r in all_results:
        samples_by_class[r["classification"]].append(r["index"])
    
    with open(output_dir / "samples_by_classification.json", "w") as f:
        json.dump(dict(samples_by_class), f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Analysis Summary:")
    logger.info(f"  Total samples: {len(all_results)}")
    logger.info(f"  Accuracy: {analysis_summary['accuracy']:.4f}")
    logger.info(f"  Average entropy: {analysis_summary['avg_entropy']:.4f}")
    logger.info("  Classification breakdown:")
    for cls, count in classification_counts.items():
        logger.info(f"    {cls}: {count} ({count/len(all_results)*100:.1f}%)")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)
    
    return analysis_summary


def main():
    parser = argparse.ArgumentParser(description="Confusable Pairs Analysis for Speech LLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--test_file", type=str, default="slurp/dataset/slurp/test.jsonl", help="Path to test data")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio/slurp_real", help="Path to audio directory")
    parser.add_argument("--output_dir", type=str, default="Experiment_2/output", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (first N)")
    parser.add_argument("--num_samples", type=int, default=None, help="Randomly select N samples")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (must be 1 for proper handling)")
    parser.add_argument("--num_beams", type=int, default=3, help="Beam search size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--add_text_only", action="store_true", help="Use text-only mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use Flash Attention 2")
    parser.add_argument("--save_attention", action="store_true", help="Save attention weights (large!)")
    parser.add_argument("--top_k_confusable", type=int, default=10, help="Number of top confusable pairs")
    parser.add_argument("--high_conf_threshold", type=float, default=0.8, help="High confidence threshold")
    parser.add_argument("--low_conf_threshold", type=float, default=0.5, help="Low confidence threshold")
    parser.add_argument("--use_cache", action="store_true", help="Use cached inference data if available")
    
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
