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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

# ==========================================
# 1. Data Loading & Clustering Logic
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
        
    def create_clusters(self, n_clusters):
        if not self.labels: return {}
        if n_clusters >= len(self.labels):
            return {l: i for i, l in enumerate(self.labels)}
        
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
        labels = clustering.fit_predict(self.embeddings)
        return {self.id_to_label[i]: cluster_id for i, cluster_id in enumerate(labels)}

# ==========================================
# 2. Analysis Logic
# ==========================================

def calculate_intra_rate(gt_errors, pred_errors, label_map):
    intra_count = 0
    valid_comparisons = 0
    for gt, pred in zip(gt_errors, pred_errors):
        if gt in label_map and pred in label_map:
            valid_comparisons += 1
            if label_map[gt] == label_map[pred]:
                intra_count += 1
    return (intra_count / valid_comparisons) if valid_comparisons > 0 else 0.0

def get_random_baseline(all_labels, n_clusters, gt_errors, pred_errors, n_trials=10):
    rates = []
    unique_labels = list(all_labels)
    for _ in range(n_trials):
        random.shuffle(unique_labels)
        random_map = {label: i % n_clusters for i, label in enumerate(unique_labels)}
        rate = calculate_intra_rate(gt_errors, pred_errors, random_map)
        rates.append(rate)
    return np.mean(rates)

def analyze_trend(df_gt, df_pred, target_col, clusterer, n_clusters_list):
    if len(df_gt) == 0: return

    error_mask = df_gt[target_col] != df_pred[target_col]
    total_errors = error_mask.sum()
    
    if total_errors == 0:
        print(f"No errors found for {target_col}.")
        return

    gt_errors = df_gt.loc[error_mask, target_col].values
    pred_errors = df_pred.loc[error_mask, target_col].values

    print(f"\n--- Analyzing Target: {target_col} (Total Errors: {total_errors}) ---")
    print(f"Total Unique Labels: {len(clusterer.labels)}")

    results = []
    for n_clusters in n_clusters_list:
        # 1. 意味的クラスタリング
        semantic_map = clusterer.create_clusters(n_clusters)
        semantic_rate = calculate_intra_rate(gt_errors, pred_errors, semantic_map)
        
        # 2. ランダムベースライン
        random_rate = get_random_baseline(clusterer.labels, n_clusters, gt_errors, pred_errors)
        
        # 3. 各種指標計算
        lift = semantic_rate / random_rate if random_rate > 0 else 0.0
        
        # 平均クラスタサイズ (全ラベル数 / クラスタ数)
        # 例: 18個のシナリオを3グループに分けたら、平均サイズは6
        avg_size = len(clusterer.labels) / n_clusters
        
        results.append({
            "N_Clust": n_clusters,
            "Avg_Sz": avg_size,     # <--- 追加: 1クラスあたりの平均ラベル数
            "Sem_Rate": semantic_rate,
            "Rnd_Rate": random_rate,
            "Lift": lift
        })
        
    df_res = pd.DataFrame(results)
    
    # 見やすいフォーマットで表示
    print(df_res.to_string(index=False, formatters={
        'Avg_Sz': '{:.1f}'.format,       # 小数点1桁
        'Sem_Rate': '{:.2%}'.format,
        'Rnd_Rate': '{:.2%}'.format,
        'Lift': '{:.2f}x'.format
    }))

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

    print("\nInitializing Clusterers...")
    sc_clusterer = LabelClusterer(all_scenarios)
    ac_clusterer = LabelClusterer(all_actions)
    
    # 分析実行
    # Scenario (ラベル種少ない) -> Avg_Sz が 2~4 くらいになるような分割を見る
    analyze_trend(df_gt, df_pred, 'scenario', sc_clusterer, [12, 8, 6, 4, 3])
    
    # Action (ラベル種多い) -> Avg_Sz が 2~5 くらいになるような分割を見る
    analyze_trend(df_gt, df_pred, 'action', ac_clusterer, [30, 20, 15, 10, 5])

if __name__ == "__main__":
    main()