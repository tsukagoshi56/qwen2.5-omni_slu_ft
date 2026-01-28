import json
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import argparse
import os
import random

# ==========================================
# 0. Configuration
# ==========================================
PRED_FILE = "prediction.jsonl"
TEST_FILE = "test.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"

# Pandas表示設定（表を綺麗に見せる）
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.colheader_justify', 'left') # 左寄せで見やすく

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
                str_sid = str(sid)
                if 'scenario' not in d: d['scenario'] = "unknown"
                if 'action' not in d: d['action'] = "unknown"
                preds[str_sid] = d
            
    gts = []
    pred_list = []
    
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            sid = d.get('slurp_id')
            if sid is not None:
                str_sid = str(sid)
                if str_sid in preds:
                    if 'scenario' not in d: d['scenario'] = "unknown"
                    if 'action' not in d: d['action'] = "unknown"
                    gts.append(d)
                    pred_list.append(preds[str_sid])
    
    return pd.DataFrame(gts), pd.DataFrame(pred_list)

class LabelClusterer:
    def __init__(self, labels, embedding_model_name=MODEL_NAME):
        self.labels = sorted(list(set(labels)))
        print(f"Embedding {len(self.labels)} unique labels...")
        if not self.labels:
            self.embeddings = []
            return
        self.model = SentenceTransformer(embedding_model_name)
        self.embeddings = self.model.encode(self.labels)
        self.id_to_label = {i: l for i, l in enumerate(self.labels)}
        
    def create_clusters_by_count(self, n_clusters):
        if not self.labels: return {}
        # クラスタ数がラベル数以上ならそのまま
        if n_clusters >= len(self.labels):
            return {l: i for i, l in enumerate(self.labels)}
        
        # 階層的クラスタリング
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
        labels = clustering.fit_predict(self.embeddings)
        return {self.id_to_label[i]: cluster_id for i, cluster_id in enumerate(labels)}

# ==========================================
# 2. Analysis Logic
# ==========================================

def calculate_in_group_rate(gt_errors, pred_errors, label_map):
    """
    エラーの中で、「正解ラベル」と「予測ラベル」が同じグループにいた割合
    """
    match_count = 0
    valid_comparisons = 0
    for gt, pred in zip(gt_errors, pred_errors):
        if gt in label_map and pred in label_map:
            valid_comparisons += 1
            if label_map[gt] == label_map[pred]:
                match_count += 1
    return (match_count / valid_comparisons) if valid_comparisons > 0 else 0.0

def get_random_baseline(all_labels, n_clusters, gt_errors, pred_errors, n_trials=10):
    rates = []
    unique_labels = list(all_labels)
    for _ in range(n_trials):
        random.shuffle(unique_labels)
        random_map = {label: i % n_clusters for i, label in enumerate(unique_labels)}
        rates.append(calculate_in_group_rate(gt_errors, pred_errors, random_map))
    return np.mean(rates)

def analyze_trend(df_gt, df_pred, target_col, clusterer, target_sizes):
    if len(df_gt) == 0: return

    error_mask = df_gt[target_col] != df_pred[target_col]
    total_errors = error_mask.sum()
    total_labels = len(clusterer.labels)
    
    if total_errors == 0:
        print(f"No errors found for {target_col}.")
        return

    gt_errors = df_gt.loc[error_mask, target_col].values
    pred_errors = df_pred.loc[error_mask, target_col].values

    print(f"\n========================================================")
    print(f" Analysis Target: {target_col.upper()} (Total Errors: {total_errors})")
    print(f"========================================================")
    
    # --- 1. グループサイズごとの「救済率」推移 ---
    print(f"\n[1] Recovery Rate by Group Size")
    print(f"    (Even if wrong, was the correct answer in the same group?)")
    
    results = []
    for size in target_sizes:
        if size >= total_labels: continue
        n_clusters = int(total_labels / size)
        if n_clusters < 2: continue
        
        # Semantic
        semantic_map = clusterer.create_clusters_by_count(n_clusters)
        sem_rate = calculate_in_group_rate(gt_errors, pred_errors, semantic_map)
        
        # Random
        rnd_rate = get_random_baseline(clusterer.labels, n_clusters, gt_errors, pred_errors)
        
        # Lift
        lift = sem_rate / rnd_rate if rnd_rate > 0 else 0.0
        
        results.append({
            "Avg_Grp_Size": f"~{size}", 
            "In_Group_Match": sem_rate,
            "Random_Chance": rnd_rate,
            "Lift (xBetter)": lift
        })
        
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False, formatters={
        'In_Group_Match': '{:.1%}'.format,
        'Random_Chance': '{:.1%}'.format,
        'Lift (xBetter)': '{:.2f}x'.format
    }))

    # --- 2. 具体的な間違いランキング (サイズ5くらいのグループを使用) ---
    ranking_size = 5
    print(f"\n[2] Top Confused Pairs (using approx group size ~{ranking_size})")
    print(f"    (Showing pairs that fell into the SAME semantic group)")
    
    n_clusters_rank = int(total_labels / ranking_size)
    if n_clusters_rank < 2: n_clusters_rank = 2
    
    label_map = clusterer.create_clusters_by_count(n_clusters_rank)
    
    # クラスターID -> メンバー一覧を作成
    cluster_members = {}
    for label, cid in label_map.items():
        if cid not in cluster_members: cluster_members[cid] = []
        cluster_members[cid].append(label)
        
    # 間違いの集計
    confusion_counts = {}
    for gt, pred in zip(gt_errors, pred_errors):
        if gt in label_map and pred in label_map:
            if label_map[gt] == label_map[pred]: # 同じグループ内での間違いのみ
                key = (gt, pred, label_map[gt]) # (正解, 予測, クラスタID)
                confusion_counts[key] = confusion_counts.get(key, 0) + 1
    
    # ソートして表示
    sorted_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_confusions:
        print("    No semantic confusions found at this granularity.")
    else:
        print("-" * 100)
        print(f"{'Count':<6} | {'Correct (GT) -> Predicted (Pred)':<40} | {'Context: Other labels in this group'}")
        print("-" * 100)
        
        for (gt, pred, cid), count in sorted_confusions[:20]: # Top 20を表示
            pair_str = f"{gt} -> {pred}"
            
            # コンテキスト（グループ内の仲間）を表示
            members = cluster_members[cid]
            # 自分たちは除外して表示したほうがわかりやすい
            others = [m for m in members if m != gt and m != pred]
            
            # 表示用に整形
            if len(others) > 0:
                context_str = ", ".join(others)
            else:
                context_str = "(Just these two)"
                
            # 長すぎたらカット
            if len(context_str) > 50: context_str = context_str[:47] + "..."
            
            print(f"{count:<6} | {pair_str:<40} | {context_str}")

# ==========================================
# 3. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default=PRED_FILE)
    parser.add_argument("--test_file", type=str, default=TEST_FILE)
    args = parser.parse_args()

    if not os.path.exists(args.pred_file) or not os.path.exists(args.test_file):
        print("Error: Files not found.")
        return

    df_gt, df_pred = load_data(args.pred_file, args.test_file)
    if len(df_gt) == 0: return

    # Embedding
    all_scenarios = list(set(df_gt['scenario'].unique().tolist() + df_pred['scenario'].unique().tolist()))
    all_actions = list(set(df_gt['action'].unique().tolist() + df_pred['action'].unique().tolist()))

    print("\nInitializing Clusterers (SentenceTransformers)...")
    sc_clusterer = LabelClusterer(all_scenarios)
    ac_clusterer = LabelClusterer(all_actions)
    
    # 分析実行 (サイズ指定: 平均して何個のグループにまとめるか)
    analyze_trend(df_gt, df_pred, 'scenario', sc_clusterer, [2, 3, 4, 6])
    analyze_trend(df_gt, df_pred, 'action', ac_clusterer, [2, 3, 5, 8, 10])

if __name__ == "__main__":
    main()