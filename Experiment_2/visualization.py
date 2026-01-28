import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patheffects as pe # 文字の縁取り用
import os
import argparse
from collections import Counter

# ==========================================
# 0. Configuration
# ==========================================
DEFAULT_PRED_FILE = "prediction.jsonl"
DEFAULT_TEST_FILE = "test.jsonl"
OUTPUT_DIR = "Experiment_2"

# 論文用のクリアなフォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

# ==========================================
# 1. Data Loading (変更なし)
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
# 2. Network Visualization Logic (Enhanced v2)
# ==========================================
def visualize_error_network(df, target_type='scenario'):
    gt_col = f'{target_type}_gt'
    pred_col = f'{target_type}_pred'
    
    print(f"\nProcessing {target_type.upper()} Network...")
    
    error_df = df[df[gt_col] != df[pred_col]].copy()
    if len(error_df) == 0: return
    total_errors = len(error_df)

    # --- グラフ構築とフィルタリング ---
    G = nx.DiGraph()
    pair_counts = Counter(zip(error_df[gt_col], error_df[pred_col]))
    
    # 【改善3: 矢印を減らす】
    # 閾値を動的に設定 (例: 全エラーの 0.5% 以上発生したペアのみ表示)
    # これでノイズが消え、主要なフローだけが残る
    # 少なくとも5回以上は発生していないと表示しない
    threshold = max(5, int(total_errors * 0.005))
    print(f"Edge weight threshold: {threshold} (Pairs with fewer counts are hidden)")
    
    filtered_pairs = {k: v for k, v in pair_counts.items() if v >= threshold}
    if not filtered_pairs:
        print("No significant errors to plot after filtering.")
        return

    for (src, dst), weight in filtered_pairs.items():
        G.add_edge(src, dst, weight=weight)
        
    # --- ノード設定 ---
    # 入次数（間違えられた回数）を取得
    in_degrees = dict(G.in_degree(weight='weight'))
    max_in = max(in_degrees.values()) if in_degrees else 1
    
    node_colors = []
    node_sizes = []
    labels = {}
    
    # ブラックホール認定ライン
    hub_threshold = max_in * 0.2
    
    for node in G.nodes():
        score = in_degrees.get(node, 0)
        labels[node] = node
        
        # 【改善1: サイズ差を強調】
        # 線形(score/max_in)ではなく、累乗(1.5乗)を使うことで、
        # スコアが大きいノードが加速度的に巨大になるようにする
        size_factor = (score / max_in)**1.5
        
        if score > hub_threshold:
            # Hub (捕食者) -> 赤、巨大
            node_colors.append('#e74c3c')
            node_sizes.append(5000 + size_factor * 10000)
        else:
            # Leaf (被害者) -> 青、中くらい
            node_colors.append('#3498db')
            node_sizes.append(800) 

    # --- レイアウト計算 ---
    # kの値が大きいほどノード間の反発力が強くなり、図が広がる
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42, weight='weight')
    
    # --- 描画 ---
    # 【改善4: 余白削減のため、描画領域を正方形に近くする】
    plt.figure(figsize=(12, 12), facecolor='white')
    ax = plt.gca()
    
    # 1. ノード描画
    nx.draw_networkx_nodes(G, pos, 
                           node_color=node_colors, 
                           node_size=node_sizes, 
                           alpha=0.9, 
                           edgecolors='#ecf0f1', # 薄い枠線
                           linewidths=2)
    
    # 2. エッジ（矢印）描画
    edges = G.edges(data=True)
    weights = [d['weight'] for u, v, d in edges]
    max_weight = max(weights) if weights else 1
    # 太さも累乗で強調
    widths = [2.0 + (w / max_weight)**1.5 * 12 for w in weights]
    
    nx.draw_networkx_edges(G, pos, 
                           width=widths, 
                           edge_color='#95a5a6', # 落ち着いたグレー
                           arrowstyle='-|>', # 塗りつぶし矢印
                           arrowsize=30, # 矢印の頭を大きく
                           connectionstyle="arc3,rad=0.15", # カーブ
                           alpha=0.6)

    # 3. ラベル描画 【改善2: 文字を大きく、黒く、縁取り】
    # 白い縁取り効果の定義
    path_effect = [pe.withStroke(linewidth=4, foreground="white")]
    
    for node, (x, y) in pos.items():
        score = in_degrees.get(node, 0)
        # Hubは特に大きく強調
        if score > hub_threshold:
            fontsize = 18
            fontweight = 'bold'
            zorder = 20
        else:
            fontsize = 14
            fontweight = 'normal'
            zorder = 15
            
        plt.text(x, y, node, 
                 fontsize=fontsize, 
                 fontweight=fontweight,
                 color='black', # 黒文字
                 ha='center', va='center',
                 path_effects=path_effect, # 縁取り適用
                 zorder=zorder)

    # タイトルと調整
    plt.title(f"Error Concentration Map: {target_type.upper()}\n(Nodes sized by mistake frequency. Major flows shown.)", 
              fontsize=16, color='#2c3e50')
    plt.axis('off')
    
    # 【改善4: 余白を極限まで減らす】
    plt.tight_layout(pad=0.1)
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"network_graph_v2_{target_type}.png")
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