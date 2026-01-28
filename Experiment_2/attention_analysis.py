#!/usr/bin/env python3
"""
Attention Analysis for Speech LLM

Analyzes and visualizes attention patterns:
- Audio-to-Token attention heatmaps
- Success vs Failure attention comparison
- Language bias detection (audio ignored patterns)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_attention_data(input_dir: Path) -> Dict:
    """Load attention weights and related data."""
    data = {}
    
    # Load attention weights
    attention_path = input_dir / "attention_weights.pt"
    if attention_path.exists():
        data["attentions"] = torch.load(attention_path)
        logger.info(f"Loaded {len(data['attentions'])} attention maps")
    
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


def plot_attention_heatmap(
    attention: torch.Tensor,
    title: str,
    output_path: Path,
    head_idx: Optional[int] = None,
    max_len: int = 100
):
    """Plot attention heatmap for a single sample."""
    # attention shape: (num_heads, seq_len, seq_len)
    if attention.dim() == 3:
        if head_idx is not None:
            attn = attention[head_idx].numpy()
        else:
            # Average over heads
            attn = attention.mean(dim=0).numpy()
    else:
        attn = attention.numpy()
    
    # Truncate if too long
    if attn.shape[0] > max_len:
        attn = attn[:max_len, :max_len]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        attn,
        cmap='viridis',
        ax=ax,
        xticklabels=False,
        yticklabels=False
    )
    
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved attention heatmap to {output_path}")


def plot_attention_comparison(
    success_attentions: List[torch.Tensor],
    failure_attentions: List[torch.Tensor],
    output_path: Path
):
    """Compare average attention patterns between success and failure cases."""
    if not success_attentions or not failure_attentions:
        logger.warning("Not enough data for attention comparison")
        return
    
    # Average attention patterns
    def average_attention(attentions: List[torch.Tensor]) -> np.ndarray:
        # Each attention is (num_heads, seq_len, seq_len)
        # Normalize to same size for averaging
        min_len = min(a.shape[1] for a in attentions[:20])  # Use first 20
        normalized = []
        for a in attentions[:20]:
            avg_head = a.mean(dim=0)[:min_len, :min_len]
            normalized.append(avg_head.numpy())
        return np.mean(normalized, axis=0)
    
    avg_success = average_attention(success_attentions)
    avg_failure = average_attention(failure_attentions)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Success attention
    sns.heatmap(
        avg_success,
        cmap='viridis',
        ax=axes[0],
        xticklabels=False,
        yticklabels=False
    )
    axes[0].set_title('Average Attention: Success')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    
    # Failure attention
    sns.heatmap(
        avg_failure,
        cmap='viridis',
        ax=axes[1],
        xticklabels=False,
        yticklabels=False
    )
    axes[1].set_title('Average Attention: Failure')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    
    # Difference
    diff = avg_success - avg_failure
    vmax = max(abs(diff.min()), abs(diff.max()))
    sns.heatmap(
        diff,
        cmap='RdBu_r',
        ax=axes[2],
        xticklabels=False,
        yticklabels=False,
        center=0,
        vmin=-vmax,
        vmax=vmax
    )
    axes[2].set_title('Difference (Success - Failure)')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved attention comparison to {output_path}")


def compute_audio_attention_ratio(attention: torch.Tensor, audio_end_pos: int) -> float:
    """
    Compute the ratio of attention to audio tokens vs text tokens.
    
    Higher ratio = model pays more attention to audio
    Lower ratio = model relies more on text (language bias)
    """
    if attention.dim() == 3:
        attn = attention.mean(dim=0)  # Average over heads
    else:
        attn = attention
    
    # Sum attention to audio tokens (positions 0 to audio_end_pos)
    audio_attn = attn[:, :audio_end_pos].sum().item()
    text_attn = attn[:, audio_end_pos:].sum().item()
    
    if text_attn == 0:
        return float('inf')
    
    return audio_attn / (audio_attn + text_attn)


def analyze_audio_attention(
    attentions: List[torch.Tensor],
    results: List[Dict],
    output_dir: Path,
    audio_token_ratio: float = 0.3  # Approximate ratio of audio tokens
):
    """
    Analyze whether model pays attention to audio or relies on text bias.
    """
    audio_ratios = []
    
    for i, (attn, result) in enumerate(zip(attentions, results)):
        # Estimate audio end position based on ratio
        seq_len = attn.shape[1]
        audio_end = int(seq_len * audio_token_ratio)
        
        ratio = compute_audio_attention_ratio(attn, audio_end)
        audio_ratios.append({
            "index": i,
            "audio_ratio": ratio,
            "classification": result.get("classification", "unknown"),
            "is_correct": result.get("is_correct", False)
        })
    
    # Plot audio attention ratio distribution by classification
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # By classification
    ax1 = axes[0]
    class_colors = {
        "success": "#2ecc71",
        "ambiguous_success": "#f39c12",
        "ambiguous_failure": "#e74c3c",
        "fatal_failure": "#9b59b6"
    }
    
    for cls in class_colors.keys():
        ratios = [r["audio_ratio"] for r in audio_ratios if r["classification"] == cls and r["audio_ratio"] < float('inf')]
        if ratios:
            ax1.hist(ratios, bins=20, alpha=0.5, label=cls.replace("_", " ").title(), color=class_colors[cls])
    
    ax1.set_xlabel('Audio Attention Ratio')
    ax1.set_ylabel('Count')
    ax1.set_title('Audio Attention Ratio by Classification')
    ax1.legend()
    ax1.axvline(x=0.5, color='black', linestyle='--', label='Equal attention')
    
    # By correctness
    ax2 = axes[1]
    correct_ratios = [r["audio_ratio"] for r in audio_ratios if r["is_correct"] and r["audio_ratio"] < float('inf')]
    incorrect_ratios = [r["audio_ratio"] for r in audio_ratios if not r["is_correct"] and r["audio_ratio"] < float('inf')]
    
    if correct_ratios and incorrect_ratios:
        ax2.boxplot([correct_ratios, incorrect_ratios], labels=['Correct', 'Incorrect'])
        ax2.set_ylabel('Audio Attention Ratio')
        ax2.set_title('Audio Attention: Correct vs Incorrect')
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = output_dir / "audio_attention_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved audio attention analysis to {output_path}")
    
    # Identify potential language bias cases (low audio attention + incorrect)
    bias_cases = [
        r for r in audio_ratios 
        if r["audio_ratio"] < 0.3 and not r["is_correct"] and r["audio_ratio"] < float('inf')
    ]
    
    if bias_cases:
        logger.info(f"Found {len(bias_cases)} potential language bias cases (low audio attention + incorrect)")
        with open(output_dir / "language_bias_cases.json", "w") as f:
            json.dump(bias_cases, f, indent=2)
    
    return audio_ratios


def plot_head_attention_diversity(
    attentions: List[torch.Tensor],
    output_path: Path,
    num_samples: int = 10
):
    """
    Visualize attention patterns across different heads.
    Some heads may specialize in audio while others focus on text.
    """
    if not attentions:
        return
    
    # Take first sample's attention
    attn = attentions[0]  # (num_heads, seq_len, seq_len)
    num_heads = attn.shape[0]
    
    # Create grid of head attention patterns
    n_cols = min(4, num_heads)
    n_rows = (num_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    max_len = min(50, attn.shape[1])  # Truncate for visibility
    
    for head_idx in range(num_heads):
        row = head_idx // n_cols
        col = head_idx % n_cols
        
        head_attn = attn[head_idx, :max_len, :max_len].numpy()
        
        axes[row, col].imshow(head_attn, cmap='viridis', aspect='auto')
        axes[row, col].set_title(f'Head {head_idx}', fontsize=10)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    
    # Hide empty subplots
    for idx in range(num_heads, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Attention Patterns Across Heads (Sample 0)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved head attention diversity to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze attention patterns")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with analysis results")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for figures")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of individual samples to visualize")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "attention_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_attention_data(input_dir)
    
    if "attentions" not in data:
        logger.error("No attention weights found. Run run_analysis.py with --save_attention first.")
        return
    
    attentions = data["attentions"]
    results = data.get("results", [])
    
    # Plot individual attention heatmaps
    for i in range(min(args.num_samples, len(attentions))):
        result = results[i] if i < len(results) else {}
        classification = result.get("classification", "unknown")
        is_correct = result.get("is_correct", False)
        
        plot_attention_heatmap(
            attentions[i],
            f"Sample {i}: {classification} ({'Correct' if is_correct else 'Incorrect'})",
            output_dir / f"attention_sample_{i}.png"
        )
    
    # Plot head diversity
    if attentions:
        plot_head_attention_diversity(
            attentions,
            output_dir / "head_attention_diversity.png"
        )
    
    # Compare success vs failure attention patterns
    if results:
        success_attentions = [
            attentions[i] for i, r in enumerate(results) 
            if i < len(attentions) and r.get("is_correct", False)
        ]
        failure_attentions = [
            attentions[i] for i, r in enumerate(results)
            if i < len(attentions) and not r.get("is_correct", False)
        ]
        
        plot_attention_comparison(
            success_attentions,
            failure_attentions,
            output_dir / "attention_comparison.png"
        )
    
    # Analyze audio attention ratio
    if attentions and results:
        analyze_audio_attention(attentions, results, output_dir)
    
    logger.info(f"All attention analysis saved to {output_dir}")


if __name__ == "__main__":
    main()
