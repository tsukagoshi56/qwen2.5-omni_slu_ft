#!/usr/bin/env python3
"""
Entropy and Confidence Analysis for Speech LLM

Analyzes model uncertainty and "hesitation":
- Prediction entropy distribution
- Softmax margin between top predictions
- Correlation with ASR n-best disagreement
- Threshold-based sample categorization
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_entropy_data(input_dir: Path) -> Dict:
    """Load analysis results for entropy analysis."""
    data = {}
    
    # Load logits
    logits_path = input_dir / "logits.pt"
    if logits_path.exists():
        data["logits"] = torch.load(logits_path)
        logger.info(f"Loaded logits: {data['logits'].shape}")
    
    # Load sample results
    results_path = input_dir / "sample_results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            data["results"] = json.load(f)
        logger.info(f"Loaded {len(data['results'])} sample results")
    
    # Load summary
    summary_path = input_dir / "analysis_summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            data["summary"] = json.load(f)
    
    return data


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy from logits."""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy


def compute_margin(logits: torch.Tensor) -> torch.Tensor:
    """Compute margin between top-1 and top-2 probabilities."""
    probs = F.softmax(logits, dim=-1)
    top2_probs, _ = torch.topk(probs, 2, dim=-1)
    margin = top2_probs[..., 0] - top2_probs[..., 1]
    return margin


def plot_entropy_analysis(
    results: List[Dict],
    output_dir: Path
):
    """Comprehensive entropy analysis plots."""
    entropies = [r["entropy"] for r in results]
    confidences = [r["confidence"] for r in results]
    is_correct = [r["is_correct"] for r in results]
    classifications = [r["classification"] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Entropy distribution
    ax1 = axes[0, 0]
    ax1.hist(entropies, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(entropies), color='red', linestyle='--', label=f'Mean: {np.mean(entropies):.2f}')
    ax1.axvline(np.median(entropies), color='green', linestyle='--', label=f'Median: {np.median(entropies):.2f}')
    ax1.set_xlabel('Entropy')
    ax1.set_ylabel('Count')
    ax1.set_title('Entropy Distribution')
    ax1.legend()
    
    # 2. Entropy vs Correctness
    ax2 = axes[0, 1]
    correct_entropies = [e for e, c in zip(entropies, is_correct) if c]
    incorrect_entropies = [e for e, c in zip(entropies, is_correct) if not c]
    
    bp = ax2.boxplot([correct_entropies, incorrect_entropies], labels=['Correct', 'Incorrect'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Entropy: Correct vs Incorrect')
    
    # Statistical test
    if correct_entropies and incorrect_entropies:
        t_stat, p_value = stats.ttest_ind(correct_entropies, incorrect_entropies)
        ax2.text(0.05, 0.95, f'T-test p-value: {p_value:.4f}', transform=ax2.transAxes, verticalalignment='top')
    
    # 3. Confidence distribution by classification
    ax3 = axes[1, 0]
    class_colors = {
        "success": "#2ecc71",
        "ambiguous_success": "#f39c12",
        "ambiguous_failure": "#e74c3c",
        "fatal_failure": "#9b59b6"
    }
    
    for cls in class_colors.keys():
        cls_confidences = [c for c, cl in zip(confidences, classifications) if cl == cls]
        if cls_confidences:
            ax3.hist(cls_confidences, bins=30, alpha=0.5, label=cls.replace("_", " ").title(), 
                    color=class_colors[cls])
    
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Count')
    ax3.set_title('Confidence Distribution by Classification')
    ax3.legend()
    
    # 4. Entropy-Confidence scatter
    ax4 = axes[1, 1]
    colors = ['#2ecc71' if c else '#e74c3c' for c in is_correct]
    ax4.scatter(entropies, confidences, c=colors, alpha=0.5, s=20)
    ax4.set_xlabel('Entropy')
    ax4.set_ylabel('Confidence')
    ax4.set_title('Entropy vs Confidence (Green=Correct, Red=Incorrect)')
    
    # Add correlation
    if entropies and confidences:
        corr = np.corrcoef(entropies, confidences)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax4.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / "entropy_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved entropy analysis to {output_dir / 'entropy_analysis.png'}")


def plot_margin_analysis(
    logits: torch.Tensor,
    results: List[Dict],
    output_dir: Path
):
    """Analyze softmax margin between top predictions."""
    # Compute margins
    margins = compute_margin(logits).numpy()
    
    is_correct = [r["is_correct"] for r in results[:len(margins)]]
    classifications = [r["classification"] for r in results[:len(margins)]]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Margin distribution
    ax1 = axes[0]
    ax1.hist(margins, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(margins), color='red', linestyle='--', label=f'Mean: {np.mean(margins):.3f}')
    ax1.set_xlabel('Margin (Top-1 - Top-2 Probability)')
    ax1.set_ylabel('Count')
    ax1.set_title('Prediction Margin Distribution')
    ax1.legend()
    
    # 2. Margin by correctness
    ax2 = axes[1]
    correct_margins = [m for m, c in zip(margins, is_correct) if c]
    incorrect_margins = [m for m, c in zip(margins, is_correct) if not c]
    
    if correct_margins and incorrect_margins:
        bp = ax2.boxplot([correct_margins, incorrect_margins], labels=['Correct', 'Incorrect'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        ax2.set_ylabel('Margin')
        ax2.set_title('Prediction Margin: Correct vs Incorrect')
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(correct_margins, incorrect_margins)
        ax2.text(0.05, 0.95, f'T-test p-value: {p_value:.4f}', transform=ax2.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / "margin_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved margin analysis to {output_dir / 'margin_analysis.png'}")


def identify_hesitant_samples(
    results: List[Dict],
    entropy_threshold: float = 2.0,
    confidence_threshold: float = 0.5
) -> Dict[str, List[Dict]]:
    """
    Identify samples where the model is "hesitating":
    - High entropy (uncertain)
    - Low confidence
    - Close to decision boundary
    """
    hesitant_correct = []
    hesitant_incorrect = []
    confident_correct = []
    confident_incorrect = []
    
    for r in results:
        entropy = r.get("entropy", 0)
        confidence = r.get("confidence", 1)
        is_correct = r.get("is_correct", False)
        
        is_hesitant = entropy > entropy_threshold or confidence < confidence_threshold
        
        if is_hesitant:
            if is_correct:
                hesitant_correct.append(r)
            else:
                hesitant_incorrect.append(r)
        else:
            if is_correct:
                confident_correct.append(r)
            else:
                confident_incorrect.append(r)
    
    return {
        "hesitant_correct": hesitant_correct,
        "hesitant_incorrect": hesitant_incorrect,
        "confident_correct": confident_correct,
        "confident_incorrect": confident_incorrect
    }


def plot_hesitation_analysis(
    categorized: Dict[str, List],
    output_path: Path
):
    """Visualize the hesitation-based sample categorization."""
    categories = ["hesitant_correct", "hesitant_incorrect", "confident_correct", "confident_incorrect"]
    counts = [len(categorized[c]) for c in categories]
    total = sum(counts)
    
    colors = ['#f39c12', '#e74c3c', '#2ecc71', '#9b59b6']
    labels = ['Hesitant\nCorrect', 'Hesitant\nIncorrect', 'Confident\nCorrect', 'Confident\nIncorrect']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    ax1 = axes[0]
    bars = ax1.bar(labels, counts, color=colors, edgecolor='black')
    ax1.set_ylabel('Count')
    ax1.set_title('Sample Categorization by Hesitation')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}\n({count/total*100:.1f}%)', ha='center', va='bottom')
    
    # Pie chart
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, explode=[0.02]*4)
    ax2.set_title('Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved hesitation analysis to {output_path}")


def analyze_confusable_pair_entropy(
    results: List[Dict],
    confusable_pairs: List[Dict],
    output_dir: Path
):
    """
    Analyze entropy specifically for confusable pairs.
    Check if confusion correlates with high entropy.
    """
    if not confusable_pairs:
        logger.info("No confusable pairs to analyze")
        return
    
    pair_stats = []
    
    for pair in confusable_pairs[:5]:  # Top 5 pairs
        label_1 = pair["label_1"]
        label_2 = pair["label_2"]
        
        # Find samples belonging to this pair
        pair_samples = [r for r in results if r["gt_label"] in [label_1, label_2]]
        
        if not pair_samples:
            continue
        
        # Separate by correctness
        correct = [r for r in pair_samples if r["is_correct"]]
        confused = [r for r in pair_samples if not r["is_correct"] and r["pred_label"] in [label_1, label_2]]
        other_error = [r for r in pair_samples if not r["is_correct"] and r["pred_label"] not in [label_1, label_2]]
        
        pair_stats.append({
            "pair": f"{label_1} <-> {label_2}",
            "total": len(pair_samples),
            "correct": len(correct),
            "confused_within_pair": len(confused),
            "other_error": len(other_error),
            "avg_entropy_correct": np.mean([r["entropy"] for r in correct]) if correct else 0,
            "avg_entropy_confused": np.mean([r["entropy"] for r in confused]) if confused else 0,
        })
    
    # Plot
    if pair_stats:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pairs = [ps["pair"][:30] for ps in pair_stats]  # Truncate long labels
        correct_counts = [ps["correct"] for ps in pair_stats]
        confused_counts = [ps["confused_within_pair"] for ps in pair_stats]
        other_counts = [ps["other_error"] for ps in pair_stats]
        
        x = np.arange(len(pairs))
        width = 0.25
        
        ax.bar(x - width, correct_counts, width, label='Correct', color='#2ecc71')
        ax.bar(x, confused_counts, width, label='Confused within Pair', color='#e74c3c')
        ax.bar(x + width, other_counts, width, label='Other Error', color='#95a5a6')
        
        ax.set_ylabel('Count')
        ax.set_title('Confusable Pair Performance Breakdown')
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "confusable_pair_entropy.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save stats
        with open(output_dir / "confusable_pair_stats.json", "w") as f:
            json.dump(pair_stats, f, indent=2)
        
        logger.info(f"Saved confusable pair entropy analysis")


def main():
    parser = argparse.ArgumentParser(description="Entropy and confidence analysis")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with analysis results")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for figures")
    parser.add_argument("--entropy_threshold", type=float, default=2.0, help="Threshold for 'hesitant' classification")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Threshold for 'confident' classification")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "entropy_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_entropy_data(input_dir)
    
    results = data.get("results", [])
    summary = data.get("summary", {})
    logits = data.get("logits")
    
    if not results:
        logger.error("No sample results found. Run run_analysis.py first.")
        return
    
    # Plot entropy analysis
    plot_entropy_analysis(results, output_dir)
    
    # Plot margin analysis (if logits available)
    if logits is not None:
        plot_margin_analysis(logits, results, output_dir)
    
    # Identify and categorize hesitant samples
    categorized = identify_hesitant_samples(
        results,
        entropy_threshold=args.entropy_threshold,
        confidence_threshold=args.confidence_threshold
    )
    
    plot_hesitation_analysis(categorized, output_dir / "hesitation_categorization.png")
    
    # Save categorized sample indices
    categorized_indices = {k: [r["index"] for r in v] for k, v in categorized.items()}
    with open(output_dir / "hesitation_categories.json", "w") as f:
        json.dump(categorized_indices, f, indent=2)
    
    # Analyze confusable pair entropy
    confusable_pairs = summary.get("top_confusable_pairs", [])
    analyze_confusable_pair_entropy(results, confusable_pairs, output_dir)
    
    # Summary statistics
    logger.info("=" * 60)
    logger.info("Entropy Analysis Summary:")
    logger.info(f"  Total samples: {len(results)}")
    logger.info(f"  Hesitant + Correct: {len(categorized['hesitant_correct'])} ({len(categorized['hesitant_correct'])/len(results)*100:.1f}%)")
    logger.info(f"  Hesitant + Incorrect: {len(categorized['hesitant_incorrect'])} ({len(categorized['hesitant_incorrect'])/len(results)*100:.1f}%)")
    logger.info(f"  Confident + Correct: {len(categorized['confident_correct'])} ({len(categorized['confident_correct'])/len(results)*100:.1f}%)")
    logger.info(f"  Confident + Incorrect: {len(categorized['confident_incorrect'])} ({len(categorized['confident_incorrect'])/len(results)*100:.1f}%)")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
