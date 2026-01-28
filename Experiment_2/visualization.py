import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 文字の縁取り効果用ライブラリ
import matplotlib.patheffects as pe
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from collections import Counter
import os
import argparse

# ==========================================
# 0. Configuration
# ==========================================
DEFAULT_PRED_FILE = "prediction.jsonl"
DEFAULT_TEST_FILE = "test.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_DIR = "Experiment_2"  # 指定の出力フォルダ

# 可視化の設定
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'

# ==========================================
# 1. Data Loading
# ==========================================
# (データロード部分は変更ありません)
def load_data(pred_path, test_path):
    print(f"Loading data from:\n  Pred: {pred_path}\n  Test: {test_path}")
    preds = {}
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
# 2. Gravity Field Visualization Logic (Enhanced)
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
    
    # A. Arrows (吸収方向の強調)
    top_pairs = pair_counts.most_common(150) # 表示数を少し増やす
    for (gt, pred), count in top_pairs:
        start = label_pos[gt]
        end = label_pos[pred]
        # 頻度に応じた太さと透明度
        width = 0.6 + (count / max_pred_count) * 8.0
        alpha = 0.2 + (count / max_pred_count) * 0.7
        # 矢印の頭を強調 (サイズを大きく)
        mutation_scale = 15 + (count / max_pred_count) * 35
        
        ax.annotate("",
                    xy=end, xycoords='data',
                    xytext=start, textcoords='data',
                    arrowprops=dict(arrowstyle="-|>", # 塗りつぶし矢印に変更して方向を明確化
                                    connectionstyle="arc3,rad=0.15",
                                    color="#c0392b",
                                    lw=width,
                                    alpha=alpha,
                                    mutation_scale=mutation_scale), # 矢印サイズ
                    zorder=4)

    # B. Nodes & Labels (視認性向上)
    # 文字の白い縁取り効果を定義
    path_effect = [pe.withStroke(linewidth=3, foreground="white")]

    for label in all_labels:
        x, y = label_pos[label]
        absorb_score = pred_counts.get(label, 0)
        size = 150 + (absorb_score / max_pred_count) * 6000
        
        if absorb_score > max_pred_count * 0.15:
            color = '#e74c3c'; edgecolor = '#c0392b'; zorder = 10
            fontweight = 'bold'
            fontsize = 14 # サイズアップ
        else:
            color = '#3498db'; edgecolor = '#2980b9'; zorder = 5
            fontweight = 'normal'
            fontsize = 11 # サイズアップ
            
        plt.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors=edgecolor, linewidth=1.5, zorder=zorder)
        
        # ラベル表示（黒文字 + 白縁取り）
        plt.text(x, y, label, fontsize=fontsize, fontweight=fontweight, 
                 ha='center', va='center', color='black', zorder=zorder+1,
                 path_effects=path_effect) # 縁取り適用

    # 装飾と余白調整
    title_str = f"Semantic Gravity Field: {target_type.upper()}\n(Large Arrows indicate absorption direction into 'Black Holes')"
    plt.title(title_str, fontsize=18, pad=15)
    plt.xlabel("Semantic Dimension 1"); plt.ylabel("Semantic Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks([]); plt.yticks([])
    
    # 余白を狭く設定
    plt.tight_layout(pad=1.2)
    
    # 出力ディレクトリ作成と保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"gravity_field_{target_type}.png")
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    plt.close()

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Visualize Semantic Gravity Field from prediction results.")
    parser.add_argument("--pred_file", type=str, default=DEFAULT_PRED_FILE, help="Path to prediction.jsonl")
    parser.add_argument("--test_file", type=str, default=DEFAULT_TEST_FILE, help="Path to test.jsonl")
    args = parser.parse_args()

    df = load_data(args.pred_file, args.test_file)
    
    if len(df) == 0:
        print("Error: No data loaded. Please check file paths.")
        return
    
    visualize_gravity_field(df, target_type='scenario')
    visualize_gravity_field(df, target_type='action')

if __name__ == "__main__":
    main()