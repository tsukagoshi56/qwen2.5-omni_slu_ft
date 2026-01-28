import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from collections import Counter
import os
import argparse  # 引数処理用

# ==========================================
# 0. Configuration
# ==========================================
# デフォルト値（引数が指定されなかった場合に使われます）
DEFAULT_PRED_FILE = "prediction.jsonl"
DEFAULT_TEST_FILE = "test.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"

# 可視化の設定
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'

# ==========================================
# 1. Data Loading
# ==========================================
def load_data(pred_path, test_path):
    print(f"Loading data from:\n  Pred: {pred_path}\n  Test: {test_path}")
    preds = {}
    
    # ファイル読み込み
    try:
        with open(pred_path, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                sid = d.get('slurp_id')
                if sid is not None:
                    preds[str(sid)] = d
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {pred_path}")
        return pd.DataFrame()

    data = []
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                sid = d.get('slurp_id')
                if sid is not None and str(sid) in preds:
                    pred_d = preds[str(sid)]
                    entry = {
                        'scenario_gt': d.get('scenario', 'unknown'),
                        'scenario_pred': pred_d.get('scenario', 'unknown'),
                        'action_gt': d.get('action', 'unknown'),
                        'action_pred': pred_d.get('action', 'unknown')
                    }
                    data.append(entry)
    except FileNotFoundError:
        print(f"Error: Test file not found at {test_path}")
        return pd.DataFrame()
                
    return pd.DataFrame(data)

# ==========================================
# 2. Gravity Field Visualization Logic
# ==========================================
def visualize_gravity_field(df, target_type='scenario', embedding_model_name=MODEL_NAME):
    gt_col = f'{target_type}_gt'
    pred_col = f'{target_type}_pred'
    
    print(f"\nProcessing {target_type.upper()} visualization...")
    
    error_df = df[df[gt_col] != df[pred_col]].copy()
    if len(error_df) == 0:
        print(f"No errors found for {target_type}. Skipping.")
        return

    print(f"Total Errors: {len(error_df)}")

    pred_counts = error_df[pred_col].value_counts().to_dict()
    pair_counts = Counter(zip(error_df[gt_col], error_df[pred_col]))
    all_labels = list(set(df[gt_col].unique()) | set(df[pred_col].unique()))
    
    print("Computing semantic embeddings (PCA)...")
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(all_labels)
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    label_pos = {l: c for l, c in zip(all_labels, coords)}
    
    # --- PLOTTING ---
    plt.figure(figsize=(14, 12), facecolor='white')
    ax = plt.gca()
    max_pred_count = max(pred_counts.values()) if pred_counts else 1
    
    # A. Arrows
    top_pairs = pair_counts.most_common(100)
    for (gt, pred), count in top_pairs:
        start = label_pos[gt]
        end = label_pos[pred]
        width = 0.8 + (count / max_pred_count) * 8.0
        alpha = 0.2 + (count / max_pred_count) * 0.8
        
        ax.annotate("", xy=end, xycoords='data', xytext=start, textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", 
                                    color="#c0392b", lw=width, alpha=alpha), zorder=4)

    # B. Nodes
    for label in all_labels:
        x, y = label_pos[label]
        absorb_score = pred_counts.get(label, 0)
        size = 150 + (absorb_score / max_pred_count) * 6000
        
        if absorb_score > max_pred_count * 0.15:
            color = '#e74c3c'; edgecolor = '#c0392b'; zorder = 10; fontweight = 'bold'; fontsize = 12
        else:
            color = '#3498db'; edgecolor = '#2980b9'; zorder = 5; fontweight = 'normal'; fontsize = 9
            
        plt.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors=edgecolor, linewidth=1.5, zorder=zorder)
        plt.text(x, y, label, fontsize=fontsize, fontweight=fontweight, 
                 ha='center', va='center', color='white', zorder=zorder+1)

    title_str = f"Semantic Gravity Field: {target_type.upper()}\n(Red = 'Black Hole' Labels absorbing errors)"
    plt.title(title_str, fontsize=18, pad=20)
    plt.xlabel("Semantic Dimension 1"); plt.ylabel("Semantic Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks([]); plt.yticks([])
    
    output_file = f"gravity_field_{target_type}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    # plt.show() # サーバー環境等でポップアップさせたくない場合はコメントアウト

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Visualize Semantic Gravity Field from prediction results.")
    parser.add_argument("--pred_file", type=str, default=DEFAULT_PRED_FILE, help="Path to prediction.jsonl")
    parser.add_argument("--test_file", type=str, default=DEFAULT_TEST_FILE, help="Path to test.jsonl")
    args = parser.parse_args()

    # 引数で受け取ったパスを使用
    df = load_data(args.pred_file, args.test_file)
    
    if len(df) == 0:
        print("Error: No data loaded. Please check file paths.")
        return
    
    visualize_gravity_field(df, target_type='scenario')
    visualize_gravity_field(df, target_type='action')

if __name__ == "__main__":
    main()