#!/usr/bin/env python3
"""
Generate dummy cache data for Experiment 2 analysis verification.
This simulates the output of run_analysis.py without needing the actual model.
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict
import random

def generate_dummy_data(output_dir: str, num_samples: int = 100):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define scenarios
    scenarios = [
        {"scenario": "play", "action": "music"},
        {"scenario": "stop", "action": "music"},
        {"scenario": "weather", "action": "query"},
        {"scenario": "alarm", "action": "set"}
    ]
    
    labels = [f"{s['scenario']}_{s['action']}" for s in scenarios]
    label_to_id = {l: i for i, l in enumerate(labels)}
    
    # Centers for hidden states (to make t-SNE look structured)
    centers = {
        labels[0]: torch.tensor([0.0, 0.0] + [0.0]*126),  # play_music
        labels[1]: torch.tensor([0.5, 0.5] + [0.0]*126),  # stop_music (close)
        labels[2]: torch.tensor([5.0, 0.0] + [0.0]*126),  # weather_query (far)
        labels[3]: torch.tensor([0.0, 5.0] + [0.0]*126),  # alarm_set (far)
    }
    
    y_true = []
    y_pred = []
    all_results = []
    all_hidden_states = []
    all_logits = []
    all_attentions = []
    all_predictions = []
    
    # Generate samples
    for i in range(num_samples):
        # Pick a ground truth class
        gt_idx = random.choice(list(range(len(labels))))
        if i < num_samples * 0.4: # Make play_music common
            gt_idx = 0 
        elif i < num_samples * 0.7: # Make stop_music common
            gt_idx = 1
        
        gt_label = labels[gt_idx]
        gt_scenario = scenarios[gt_idx]["scenario"]
        gt_action = scenarios[gt_idx]["action"]
        
        # Decide prediction (simulate confusion between 0 and 1)
        if gt_idx == 0 and random.random() < 0.2:
            pred_idx = 1 # Confuse play with stop
        elif gt_idx == 1 and random.random() < 0.2:
            pred_idx = 0 # Confuse stop with play
        elif random.random() < 0.05:
            pred_idx = random.choice(list(range(len(labels)))) # Random noise
        else:
            pred_idx = gt_idx # Correct
            
        pred_label = labels[pred_idx]
        pred_scenario = scenarios[pred_idx]["scenario"]
        pred_action = scenarios[pred_idx]["action"]
        
        y_true.append(gt_label)
        y_pred.append(pred_label)
        
        is_correct = (gt_label == pred_label)
        
        # Generate hidden state: center + noise
        noise = torch.randn(128) * 0.5
        hidden_vec = centers[gt_label] + noise
        # If error, maybe shift towards predicted class
        if not is_correct:
            hidden_vec = (hidden_vec + centers[pred_label]) / 2 + noise
        
        all_hidden_states.append(hidden_vec)
        
        # Logits (dummy)
        logits = torch.randn(100) # vocab size 100 for dummy
        all_logits.append(logits)
        
        # Attention (dummy)
        all_attentions.append(torch.randn(4, 10, 10))
        
        # Classification category
        if is_correct:
            classification = "success" if random.random() > 0.3 else "ambiguous_success"
        else:
            classification = "ambiguous_failure" if gt_label in [labels[0], labels[1]] and pred_label in [labels[0], labels[1]] else "fatal_failure"

        # Entropy (dummy)
        entropy_val = random.uniform(0.1, 0.5) if is_correct else random.uniform(1.5, 3.0)
        confidence = 1.0 - (entropy_val / 4.0)
        
        result = {
            "index": i,
            "slurp_id": i + 1000,
            "audio_path": f"/dummy/audio/{i}.wav",
            "transcript": f"Dummy transcript for {gt_label}",
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
            "raw_output": f"{{\"scenario\": \"{pred_scenario}\", \"action\": \"{pred_action}\"}}"
        }
        all_results.append(result)
        
        all_predictions.append({
            "slurp_id": i+1000,
            "response": result["raw_output"],
            "parsed": {"scenario": pred_scenario, "action": pred_action}
        })

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    unique_labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Top confusable
    confusable_pairs = []
    for i, l1 in enumerate(unique_labels):
        for j, l2 in enumerate(unique_labels):
            if i < j:
                mutual_error = cm[i, j] + cm[j, i]
                if mutual_error > 0:
                    confusable_pairs.append({
                        "label_1": l1,
                        "label_2": l2,
                        "error_1_to_2": int(cm[i, j]),
                        "error_2_to_1": int(cm[j, i]),
                        "mutual_error": int(mutual_error)
                    })
    confusable_pairs.sort(key=lambda x: -x["mutual_error"])
    
    # Analysis Summary
    analysis_summary = {
        "model_path": "dummy_model",
        "num_samples": num_samples,
        "accuracy": sum(1 for r in all_results if r["is_correct"]) / num_samples,
        "classification_counts": dict(defaultdict(int, {r["classification"]: 0 for r in all_results})), # Initialize? No just generate from list
        "top_confusable_pairs": confusable_pairs,
        "avg_entropy": sum(r["entropy"] for r in all_results) / num_samples
    }
    # Fix counts
    counts = defaultdict(int)
    for r in all_results: counts[r["classification"]] += 1
    analysis_summary["classification_counts"] = dict(counts)
    
    # Save everything
    cached_data = {
        "hidden_states": torch.stack(all_hidden_states),
        "logits": torch.stack(all_logits),
        "attentions": all_attentions,
        "results": all_results,
        "predictions": all_predictions,
        "y_true": y_true,
        "y_pred": y_pred,
        "model_path": "dummy_model",
        "args": {}
    }
    
    print(f"Saving dummy data to {output_dir}...")
    torch.save(cached_data, output_dir / "cached_inference_data.pt")
    
    # Save individual files for visualize_features.py
    torch.save(cached_data["hidden_states"], output_dir / "hidden_states.pt")
    torch.save(cached_data["logits"], output_dir / "logits.pt")
    np.save(output_dir / "confusion_matrix.npy", cm)
    
    with open(output_dir / "confusion_labels.json", "w") as f:
        json.dump(unique_labels, f, indent=2)
        
    with open(output_dir / "sample_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(analysis_summary, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="Experiment_2/output")
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()
    
    generate_dummy_data(args.output_dir, args.num_samples)
