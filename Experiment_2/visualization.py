import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # ネットワーク図描画用
import os
import argparse
from collections import Counter

# ==========================================
# 0. Configuration
# ==========================================
DEFAULT_PRED_FILE = "prediction.jsonl"
DEFAULT_TEST_FILE = "test.jsonl"
OUTPUT_DIR = "Experiment_2"

# デザイン設定
plt.rcParams['font.family'] = 'DejaVu Sans'

# ==========================================
# 1. Data Loading
# ==========================================
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
        print("Error: Prediction file not found.")
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
        print("Error: Test file not found.")
        return pd.DataFrame()
                
    return pd.DataFrame(data)

# ==========================================
# 2. Network Visualization Logic
# ==========================================
def visualize_error_network(df, target_type='scenario'):
    gt_col = f'{target_type}_gt'
    pred_col = f'{target_type}_pred'
    
    print(f"\nProcessing {target_type.upper()} Network...")
    
    # エラー抽出
    error_df = df[df[gt_col] != df[pred_col]].copy()
    if len(error_df) == 0: return

    # --- グラフデータの構築 ---
    G = nx.DiGraph() # 有向グラフ
    
    # 頻出ペアの集計
    pair_counts = Counter(zip(error_df[gt_col], error_df[pred_col]))
    
    # ノイズ除去: あまりに少ない間違い（1, 2回など）は描画しない方が綺麗
    # エラー総数に応じて足切りラインを調整
    min_edge_weight = 3 if len(error_df) > 500 else 1
    
    filtered_pairs = {k: v for k, v in pair_counts.items() if v >= min_edge_weight}
    
    if not filtered_pairs:
        print("No significant errors to plot.")
        return

    # エッジ（矢印）を追加
    for (src, dst), weight in filtered_pairs.items():
        G.add_edge(src, dst, weight=weight)
        
    # --- ノードの役割判定（色分け用） ---
    # In-Degree (入次数) = 何回間違えられたか = ブラックホール度
    in_degrees = dict(G.in_degree(weight='weight'))
    max_in = max(in_degrees.values()) if in_degrees else 1
    
    # ノードリスト作成
    node_colors = []
    node_sizes = []
    labels = {}
    
    # ブラックホール判定ライン
    hub_threshold = max_in * 0.2
    
    for node in G.nodes():
        score = in_degrees.get(node, 0)
        
        if score > hub_threshold:
            # ブラックホール (Hub) -> 赤くて巨大
            node_colors.append('#e74c3c') # Red
            node_sizes.append(3000 + (score/max_in)*5000) 
        else:
            # 被害者 (Leaf) -> 青くて小さい
            node_colors.append('#3498db') # Blue
            node_sizes.append(300)
            
        labels[node] = node

    # --- レイアウト計算 (これが「綺麗に配置」するキモ) ---
    # k: ノード間の反発力（大きいほど広がる）
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # --- 描画 ---
    plt.figure(figsize=(16, 14), facecolor='white')
    ax = plt.gca()
    
    # 1. ノード描画
    nx.draw_networkx_nodes(G, pos, 
                           node_color=node_colors, 
                           node_size=node_sizes, 
                           alpha=0.9, 
                           edgecolors='white', 
                           linewidths=2)
    
    # 2. ラベル描画
    nx.draw_networkx_labels(G, pos, labels, 
                            font_size=10, 
                            font_color='white', 
                            font_weight='bold')
    
    # 3. エッジ（矢印）描画
    edges = G.edges(data=True)
    weights = [d['weight'] for u, v, d in edges]
    
    # 太さの正規化
    max_weight = max(weights) if weights else 1
    widths = [1 + (w / max_weight) * 8 for w in weights]
    
    # カーブをつけて描画 (connectionstyle)
    # これで直線が重ならず、フローが見やすくなる
    nx.draw_networkx_edges(G, pos, 
                           width=widths, 
                           edge_color='#95a5a6', # グレーの矢印
                           arrowstyle='-|>', 
                           arrowsize=20,
                           connectionstyle="arc3,rad=0.15",
                           alpha=0.6)

    # タイトル
    plt.title(f"Error Concentration Network: {target_type.upper()}\n(Nodes naturally cluster around high-frequency error targets)", 
              fontsize=18, color='#2c3e50')
    
    # 枠線を消す
    plt.axis('off')
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"network_graph_{target_type}.png")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    plt.close()

# ==========================================
# 3. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default=DEFAULT_PRED_FILE)
    parser.add_argument("--test_file", type=str, default=DEFAULT_TEST_FILE)
    args = parser.parse_args()

    df = load_data(args.pred_file, args.test_file)
    if len(df) == 0: return
    
    visualize_error_network(df, target_type='scenario')
    visualize_error_network(df, target_type='action')

if __name__ == "__main__":
    main()