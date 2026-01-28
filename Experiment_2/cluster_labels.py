import argparse
import os
import sys
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_items(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def cluster_items(items: List[str], model: SentenceTransformer, group_size: int) -> List[List[str]]:
    if not items:
        return []
        
    n_items = len(items)
    if group_size <= 1:
        # Just return each item as a group
        return [[item] for item in items]
        
    embeddings = model.encode(items)
    
    # Cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # Mask self-similarity
    np.fill_diagonal(sim_matrix, -1.0)
    
    clusters = []
    processed = set()
    
    # Greedy clustering
    while len(processed) + group_size <= n_items:
        # Create a mask for valid (unprocessed) items
        valid_indices = [i for i in range(n_items) if i not in processed]
        
        # If we somehow don't have enough valid indices (should be covered by while condition)
        if len(valid_indices) < group_size:
            break
            
        # Work on submatrix to find best pair
        # We want to find (i, j) both in valid_indices maximizing sim_matrix[i, j]
        
        best_sim = -2.0
        best_pair = (-1, -1)
        
        # Optimization: Scanning only relevant rows/cols
        # For small N in this task (N < 100), full scan or strict loop is fine.
        # Let's do a masked approach on full matrix for simplicity of indexing.
        
        # Create a full mask where processed rows/cols are -2
        current_sims = sim_matrix.copy()
        
        mask_val = -2.0
        if processed:
             processed_list = list(processed)
             current_sims[processed_list, :] = mask_val
             current_sims[:, processed_list] = mask_val
             
        # Create linear index of max
        max_idx = np.argmax(current_sims)
        i, j = np.unravel_index(max_idx, current_sims.shape)
        
        # Start cluster
        current_indices = [i, j]
        
        # If group_size > 2, grow the cluster
        while len(current_indices) < group_size:
            # Centroid of current cluster
            cluster_emb = np.mean(embeddings[current_indices], axis=0).reshape(1, -1)
            
            # Distance to all
            dists = cosine_similarity(cluster_emb, embeddings)[0]
            
            # Mask processed and current items
            for idx in processed:
                dists[idx] = -2.0
            for idx in current_indices:
                dists[idx] = -2.0
                
            next_best = np.argmax(dists)
            current_indices.append(next_best)
            
        # Register cluster
        clusters.append([items[idx] for idx in current_indices])
        processed.update(current_indices)
        
    # Handle remainder
    remainder = [items[i] for i in range(n_items) if i not in processed]
    if remainder:
        clusters.append(remainder)
        
    return clusters

def main():
    parser = argparse.ArgumentParser(description="Cluster text lines into semantic groups of size n.")
    parser.add_argument("--input_files", nargs='+', required=True, help="List of files to process")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--group_size", "-n", type=int, default=2, help="Size of each cluster (min 2)")
    
    args = parser.parse_args()
    
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for filepath in args.input_files:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}. Skipping.")
            continue
            
        filename = os.path.basename(filepath)
        print(f"\nProcessing {filename}...")
        
        items = load_items(filepath)
        print(f"Loaded {len(items)} items.")
        
        if len(items) == 0:
            print("Empty file. Skipping.")
            continue
        
        clusters = cluster_items(items, model, args.group_size)
        
        # Output
        stem = os.path.splitext(filename)[0]
        out_name = f"{stem}_clustered_n{args.group_size}.txt"
        out_path = os.path.join(args.output_dir, out_name)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            for idx, cluster in enumerate(clusters, 1):
                f.write(f"Group {idx}:\n")
                for item in cluster:
                    f.write(f"  - {item}\n")
                f.write("\n")
                
        print(f"Saved clusters to {out_path}")

if __name__ == "__main__":
    main()
