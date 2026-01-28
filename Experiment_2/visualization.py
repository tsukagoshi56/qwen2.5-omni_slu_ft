import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from collections import Counter
import os

# ==========================================
# 0. Configuration
# ==========================================
PRED_FILE = "prediction.jsonl"
TEST_FILE = "test.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"

# 可視化の設定
plt.style.use('default') # デフォルトスタイル
plt.rcParams['font.family'] = 'DejaVu Sans' # 英語論文用フォント

# ==========================================
# 1. Data Loading
# ==========================================
def load_data(pred_path, test_path):
    print(f"Loading data from {pred_path} and {test_path}...")
    preds = {}
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            sid = d.get('slurp_id')
            if sid is not None:
                preds[str(sid)] = d
            
    data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            sid = d.get('slurp_id')
            if sid is not None and str(sid) in preds:
                pred_d = preds[str(sid)]
                
                # シナリオとアクションの両方を取得
                entry = {
                    'scenario_gt': d.get('scenario', 'unknown'),
                    'scenario_pred': pred_d.get('scenario', 'unknown'),
                    'action_gt': d.get('action', 'unknown'),
                    'action_pred': pred_d.get('action', 'unknown')
                }
                data.append(entry)
                
    return pd.DataFrame(data)

# ==========================================
# 2. Gravity Field Visualization Logic
# ==========================================
def visualize_gravity_field(df, target_type='scenario', embedding_model_name=MODEL_NAME):
    # カラム名の決定
    gt_col = f'{target_type}_gt'
    pred_col = f'{target_type}_pred'
    
    print(f"\nProcessing {target_type.upper()} visualization...")
    
    # 1. エラーデータの抽出
    error_df = df[df[gt_col] != df[pred_col]].copy()
    if len(error_df) == 0:
        print(f"No errors found for {target_type}. Skipping.")
        return

    print(f"Total Errors: {len(error_df)}")

    # 2. 集計
    # ノードサイズ用: そのラベルが「誤った予測先」として選ばれた回数 (Predator Score)
    pred_counts = error_df[pred_col].value_counts().to_dict()
    # 被害者（本来の正解）としても登場した回数（表示用）
    victim_counts = error_df[gt_col].value_counts().to_dict()
    
    # 全登場ラベル
    all_labels = list(set(df[gt_col].unique()) | set(df[pred_col].unique()))
    
    # 矢印用: (GT -> Pred) のペアの頻度
    pair_counts = Counter(zip(error_df[gt_col], error_df[pred_col]))
    
    # 3. 座標計算 (SentenceTransformer + PCA)
    print("Computing semantic embeddings...")
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(all_labels)
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    label_pos = {l: c for l, c in zip(all_labels, coords)}
    
    # --- PLOTTING ---
    plt.figure(figsize=(14, 12), facecolor='white')
    ax = plt.gca()
    
    # 最大値（正規化用）
    max_pred_count = max(pred_counts.values()) if pred_counts else 1
    
    # A. 矢印を描画 (Gravity Flow)
    # 上位のエラーフローのみ描画して視認性を確保 (Top 100ペア)
    top_pairs = pair_counts.most_common(100)
    
    print(f"Drawing top {len(top_pairs)} error flows...")
    for (gt, pred), count in top_pairs:
        start = label_pos[gt]
        end = label_pos[pred]
        
        # 頻度に応じて太さと透明度を変える
        width = 0.8 + (count / max_pred_count) * 8.0  # 最大でかなり太く
        alpha = 0.2 + (count / max_pred_count) * 0.8
        
        ax.annotate("",
                    xy=end, xycoords='data',
                    xytext=start, textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3,rad=0.15", # カーブ
                                    color="#c0392b", # Dark Red
                                    lw=width,
                                    alpha=alpha),
                    zorder=4)

    # B. ノード（円）を描画
    for label in all_labels:
        x, y = label_pos[label]
        
        # 吸い込み回数
        absorb_score = pred_counts.get(label, 0)
        
        # サイズ決定: ベースサイズ + 吸引力
        size = 150 + (absorb_score / max_pred_count) * 6000
        
        # 色決定
        # 吸い込みが多い(ブラックホール) -> 赤
        if absorb_score > max_pred_count * 0.15: # 上位15%クラスの吸引力
            color = '#e74c3c' # Red
            edgecolor = '#c0392b'
            zorder = 10
            fontweight = 'bold'
            fontsize = 12
        else:
            color = '#3498db' # Blue
            edgecolor = '#2980b9'
            zorder = 5
            fontweight = 'normal'
            fontsize = 9
            
        plt.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors=edgecolor, linewidth=1.5, zorder=zorder)
        
        # ラベル表示
        plt.text(x, y, label, fontsize=fontsize, fontweight=fontweight, 
                 ha='center', va='center', color='white', zorder=zorder+1)

    # 装飾
    title_str = f"Semantic Gravity Field: {target_type.upper()}\n(Red = 'Black Hole' Labels absorbing errors)"
    plt.title(title_str, fontsize=18, pad=20)
    plt.xlabel("Semantic Dimension 1 (PCA)", fontsize=12)
    plt.ylabel("Semantic Dimension 2 (PCA)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 軸の数値を消す（概念図として綺麗にするため）
    plt.xticks([])
    plt.yticks([])
    
    # ファイル保存
    output_file = f"gravity_field_{target_type}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    plt.show()

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    if not os.path.exists(PRED_FILE) or not os.path.exists(TEST_FILE):
        print("Error: Input files not found.")
        return

    # データロード
    df = load_data(PRED_FILE, TEST_FILE)
    if len(df) == 0:
        print("Error: No matched data found.")
        return
    
    # 可視化実行
    visualize_gravity_field(df, target_type='scenario')
    visualize_gravity_field(df, target_type='action')

if __name__ == "__main__":
    main()