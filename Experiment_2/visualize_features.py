#!/usr/bin/env python3
"""
Feature Space Visualization using t-SNE and UMAP

Visualizes the hidden state embeddings from the analysis results:
- t-SNE / UMAP dimensionality reduction
- Color by label, classification type
- Highlight confusable pairs
- Show misclassified samples at cluster boundaries
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_analysis_data(input_dir: Path) -> Dict:
    """Load all analysis data from the input directory."""
    data = {}
    
    # Load hidden states
    hidden_states_path = input_dir / "hidden_states.pt"
    if hidden_states_path.exists():
        data["hidden_states"] = torch.load(hidden_states_path)
        logger.info(f"Loaded hidden states: {data['hidden_states'].shape}")
    
    # Load sample results
    results_path = input_dir / "sample_results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            data["results"] = json.load(f)
        logger.info(f"Loaded {len(data['results'])} sample results")
    
    # Load analysis summary
    summary_path = input_dir / "analysis_summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            data["summary"] = json.load(f)
    
    # Load confusion matrix
    cm_path = input_dir / "confusion_matrix.npy"
    if cm_path.exists():
        data["confusion_matrix"] = np.load(cm_path)
        
    labels_path = input_dir / "confusion_labels.json"
    if labels_path.exists():
        with open(labels_path, "r") as f:
            data["labels"] = json.load(f)
    
    return data


def apply_tsne(hidden_states: np.ndarray, perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:
    """Apply t-SNE dimensionality reduction."""
    from sklearn.manifold import TSNE
    
    logger.info(f"Applying t-SNE with perplexity={perplexity}...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(hidden_states) - 1),
        n_iter=n_iter,
        random_state=42,
        verbose=1
    )
    embeddings = tsne.fit_transform(hidden_states)
    return embeddings


def apply_umap(hidden_states: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Apply UMAP dimensionality reduction."""
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed. Skipping UMAP. Install with: pip install umap-learn")
        return None
    
    logger.info(f"Applying UMAP with n_neighbors={n_neighbors}...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    embeddings = reducer.fit_transform(hidden_states)
    return embeddings


def plot_embeddings_by_label(
    embeddings: np.ndarray,
    labels: List[str],
    title: str,
    output_path: Path,
    highlight_labels: Optional[List[str]] = None,
    max_labels_in_legend: int = 20
):
    """Plot 2D embeddings colored by label."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    unique_labels = list(set(labels))
    
    if len(unique_labels) > max_labels_in_legend:
        # Too many labels, use colormap without legend
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        colors = [label_to_idx[l] for l in labels]
        
        scatter = ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=colors,
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        plt.colorbar(scatter, ax=ax, label='Label Index')
    else:
        # Create color palette
        palette = sns.color_palette("husl", len(unique_labels))
        label_to_color = {l: palette[i] for i, l in enumerate(unique_labels)}
        
        for label in unique_labels:
            mask = [l == label for l in labels]
            points = embeddings[mask]
            
            alpha = 0.8 if highlight_labels and label in highlight_labels else 0.4
            size = 50 if highlight_labels and label in highlight_labels else 20
            
            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=[label_to_color[label]],
                label=label,
                alpha=alpha,
                s=size
            )
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot to {output_path}")


def plot_embeddings_by_classification(
    embeddings: np.ndarray,
    classifications: List[str],
    title: str,
    output_path: Path
):
    """Plot 2D embeddings colored by classification type."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    class_colors = {
        "success": "#2ecc71",  # Green
        "ambiguous_success": "#f39c12",  # Orange
        "ambiguous_failure": "#e74c3c",  # Red
        "fatal_failure": "#9b59b6"  # Purple
    }
    
    class_markers = {
        "success": "o",
        "ambiguous_success": "s",
        "ambiguous_failure": "^",
        "fatal_failure": "X"
    }
    
    for cls in ["success", "ambiguous_success", "ambiguous_failure", "fatal_failure"]:
        mask = [c == cls for c in classifications]
        if any(mask):
            points = embeddings[mask]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=class_colors.get(cls, "#95a5a6"),
                marker=class_markers.get(cls, "o"),
                label=cls.replace("_", " ").title(),
                alpha=0.7,
                s=40
            )
    
    ax.legend(loc='upper right')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot to {output_path}")


def plot_confusable_pairs(
    embeddings: np.ndarray,
    results: List[Dict],
    confusable_pairs: List[Dict],
    output_dir: Path,
    method_name: str = "tsne"
):
    """Create focused plots for each confusable pair."""
    if not confusable_pairs:
        logger.info("No confusable pairs to plot")
        return
    
    for i, pair in enumerate(confusable_pairs[:5]):  # Top 5 pairs
        label_1 = pair["label_1"]
        label_2 = pair["label_2"]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Find samples for these labels
        mask_1 = [r["gt_label"] == label_1 for r in results]
        mask_2 = [r["gt_label"] == label_2 for r in results]
        other_mask = [not (m1 or m2) for m1, m2 in zip(mask_1, mask_2)]
        
        # Plot other samples faintly
        if any(other_mask):
            other_points = embeddings[other_mask]
            ax.scatter(
                other_points[:, 0],
                other_points[:, 1],
                c='#cccccc',
                alpha=0.2,
                s=10,
                label='Other'
            )
        
        # Plot the two confusable labels
        if any(mask_1):
            points_1 = embeddings[mask_1]
            correct_1 = [results[j]["is_correct"] for j, m in enumerate(mask_1) if m]
            colors_1 = ['#2ecc71' if c else '#e74c3c' for c in correct_1]
            ax.scatter(
                points_1[:, 0],
                points_1[:, 1],
                c=colors_1,
                marker='o',
                s=80,
                alpha=0.8,
                edgecolors='blue',
                linewidths=2,
                label=f'{label_1}'
            )
        
        if any(mask_2):
            points_2 = embeddings[mask_2]
            correct_2 = [results[j]["is_correct"] for j, m in enumerate(mask_2) if m]
            colors_2 = ['#2ecc71' if c else '#e74c3c' for c in correct_2]
            ax.scatter(
                points_2[:, 0],
                points_2[:, 1],
                c=colors_2,
                marker='s',
                s=80,
                alpha=0.8,
                edgecolors='orange',
                linewidths=2,
                label=f'{label_2}'
            )
        
        ax.set_title(f"Confusable Pair #{i+1}: {label_1} vs {label_2}\n"
                    f"Mutual errors: {pair['mutual_error']} (Green=Correct, Red=Incorrect)")
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend(loc='upper right')
        
        output_path = output_dir / f"confusable_pair_{i+1}_{method_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusable pair plot to {output_path}")


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], output_path: Path, top_n: int = 30):
    """Plot confusion matrix heatmap."""
    # If too many labels, show only top-N most common
    if len(labels) > top_n:
        # Sum row and column totals
        totals = cm.sum(axis=0) + cm.sum(axis=1)
        top_indices = np.argsort(totals)[-top_n:]
        cm = cm[top_indices][:, top_indices]
        labels = [labels[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Normalize by row
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
    
    sns.heatmap(
        cm_normalized,
        xticklabels=labels,
        yticklabels=labels,
        cmap='Blues',
        ax=ax,
        annot=False,
        fmt='.2f'
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Normalized Confusion Matrix (Row-normalized)')
    
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.yticks(fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_entropy_distribution(
    results: List[Dict],
    output_path: Path
):
    """Plot entropy distribution by classification type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Entropy histogram by classification
    ax1 = axes[0]
    class_colors = {
        "success": "#2ecc71",
        "ambiguous_success": "#f39c12",
        "ambiguous_failure": "#e74c3c",
        "fatal_failure": "#9b59b6"
    }
    
    for cls in class_colors.keys():
        entropies = [r["entropy"] for r in results if r["classification"] == cls]
        if entropies:
            ax1.hist(entropies, bins=30, alpha=0.5, label=cls.replace("_", " ").title(), color=class_colors[cls])
    
    ax1.set_xlabel('Entropy')
    ax1.set_ylabel('Count')
    ax1.set_title('Entropy Distribution by Classification')
    ax1.legend()
    
    # Plot 2: Entropy vs Correctness
    ax2 = axes[1]
    correct_entropies = [r["entropy"] for r in results if r["is_correct"]]
    incorrect_entropies = [r["entropy"] for r in results if not r["is_correct"]]
    
    ax2.boxplot([correct_entropies, incorrect_entropies], labels=['Correct', 'Incorrect'])
    ax2.set_ylabel('Entropy')
    ax2.set_title('Entropy: Correct vs Incorrect Predictions')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved entropy distribution to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize feature space from analysis results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with analysis results")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for figures")
    parser.add_argument("--method", type=str, choices=["tsne", "umap", "both"], default="both",
                       help="Dimensionality reduction method")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_analysis_data(input_dir)
    
    if "hidden_states" not in data:
        logger.error("No hidden states found. Run run_analysis.py first.")
        return
    
    hidden_states = data["hidden_states"].numpy()
    results = data.get("results", [])
    summary = data.get("summary", {})
    
    # Get labels and classifications
    gt_labels = [r["gt_label"] for r in results]
    classifications = [r["classification"] for r in results]
    confusable_pairs = summary.get("top_confusable_pairs", [])
    
    # Apply dimensionality reduction
    if args.method in ["tsne", "both"]:
        tsne_embeddings = apply_tsne(hidden_states, perplexity=args.perplexity)
        
        # Plot by label
        plot_embeddings_by_label(
            tsne_embeddings,
            gt_labels,
            "t-SNE: Feature Space by Label",
            output_dir / "tsne_by_label.png"
        )
        
        # Plot by classification
        plot_embeddings_by_classification(
            tsne_embeddings,
            classifications,
            "t-SNE: Feature Space by Classification",
            output_dir / "tsne_by_classification.png"
        )
        
        # Plot confusable pairs
        plot_confusable_pairs(
            tsne_embeddings,
            results,
            confusable_pairs,
            output_dir,
            method_name="tsne"
        )
    
    if args.method in ["umap", "both"]:
        umap_embeddings = apply_umap(hidden_states, n_neighbors=args.n_neighbors)
        
        if umap_embeddings is not None:
            # Plot by label
            plot_embeddings_by_label(
                umap_embeddings,
                gt_labels,
                "UMAP: Feature Space by Label",
                output_dir / "umap_by_label.png"
            )
            
            # Plot by classification
            plot_embeddings_by_classification(
                umap_embeddings,
                classifications,
                "UMAP: Feature Space by Classification",
                output_dir / "umap_by_classification.png"
            )
            
            # Plot confusable pairs
            plot_confusable_pairs(
                umap_embeddings,
                results,
                confusable_pairs,
                output_dir,
                method_name="umap"
            )
    
    # Plot confusion matrix
    if "confusion_matrix" in data and "labels" in data:
        plot_confusion_matrix(
            data["confusion_matrix"],
            data["labels"],
            output_dir / "confusion_matrix.png"
        )
    
    # Plot entropy distribution
    if results:
        plot_entropy_distribution(results, output_dir / "entropy_distribution.png")
    
    logger.info(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
