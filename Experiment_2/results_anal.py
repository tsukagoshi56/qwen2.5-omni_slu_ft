import json
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import argparse
import os

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
    
    # --- 1. Load Predictions ---
    preds = {}
    pred_ids_sample = [] # デバッグ用
    
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            sid = d.get('slurp_id')
            if sid is not None:
                # 【重要】IDを文字列に統一してキーにする
                str_sid = str(sid)
                
                # キーの存在確認（なければunknown）
                if 'scenario' not in d: d['scenario'] = "unknown"
                if 'action' not in d: d['action'] = "unknown"
                
                preds[str_sid] = d
                if len(pred_ids_sample) < 3: pred_ids_sample.append(str_sid)
            
    # --- 2. Load Ground Truth (Test) ---
    gts = []
    pred_list = []
    test_ids_sample = [] # デバッグ用
    
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            sid = d.get('slurp_id')
            
            if sid is not None:
                # 【重要】こちらもIDを文字列に統一
                str_sid = str(sid)
                if len(test_ids_sample) < 3: test_ids_sample.append(str_sid)

                if str_sid in preds:
                    # 正解データのキー補完
                    if 'scenario' not in d: d['scenario'] = "unknown"
                    if 'action' not in d: d['action'] = "unknown"
                    
                    gts.append(d)
                    pred_list.append(preds[str_sid])

    # --- 3. Debug Info if No Match ---
    if len(gts) == 0:
        print("\n[WARNING] 0 matched samples found!")
        print(f"Top 3 Prediction IDs: {pred_ids_sample}")
        print(f"Top 3 Test IDs:       {test_ids_sample}")
        print("Please check if 'slurp_id' matches between files.\n")
    else:
        print(f"Successfully loaded {len(gts)} matched samples.")

    # DataFrame化
    df_gt = pd.DataFrame(gts)
    df_pred = pd.DataFrame(pred_list)
    
    return df_gt, df_pred

class LabelClusterer:
    def __init__(self, labels, embedding_model_name=MODEL_NAME):
        # 空リスト対策
        if not labels:
            self.labels = []
            self.embeddings = []
            print("Warning: No labels to embed.")
            return

        self.labels = sorted(list(set(labels)))
        print(f"Embedding {len(self.labels)} unique labels using {embedding_model_name}...")
        self.model = SentenceTransformer(embedding_model_name)
        self.embeddings = self.model.encode(self.labels)
        self.id_to_label = {i: l for i, l in enumerate(self.labels)}
        
    def create_clusters(self, n_clusters):
        if not self.labels:
            return {}
        if n_clusters >= len(self.labels):
            return {l: i for i, l in enumerate(self.labels)}
        
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
        labels = clustering.fit_predict(self.embeddings)
        return {self.id_to_label[i]: cluster_id for i, cluster_id in enumerate(labels)}

# ==========================================
# 2. Analysis Logic
# ==========================================
def analyze_trend(df_gt, df_pred, target_col, clusterer, n_clusters_list):
    if len(df_gt) == 0: return

    # エラー抽出
    error_mask = df_gt[target_col] != df_pred[target_col]
    total_errors = error_mask.sum()
    
    if total_errors == 0:
        print(f"No errors found for {target_col}.")
        return

    gt_errors = df_gt.loc[error_mask, target_col].values
    pred_errors = df_pred.loc[error_mask, target_col].values

    print(f"\n--- Analyzing Target: {target_col} (Total Errors: {total_errors}) ---")

    results = []
    for n_clusters in n_clusters_list:
        label_map = clusterer.create_clusters(n_clusters)
        if not label_map: continue

        intra_count = 0
        valid_comparisons = 0
        
        for gt, pred in zip(gt_errors, pred_errors):
            if gt in label_map and pred in label_map:
                valid_comparisons += 1
                if label_map[gt] == label_map[pred]:
                    intra_count += 1
        
        rate = intra_count / valid_comparisons if valid_comparisons > 0 else 0.0
        # ゼロ除算回避
        avg_cluster_size = len(clusterer.labels) / n_clusters if n_clusters > 0 else 0
        
        results.append({
            "N_Clusters": n_clusters,
            "Avg_Group_Size": avg_cluster_size,
            "Intra_Error_Count": intra_count,
            "Intra_Error_Rate": rate
        })
        
    print(pd.DataFrame(results).to_string(index=False))

def analyze_confusion_in_tightest_cluster(df_gt, df_pred, target_col, clusterer, tightest_n):
    if len(df_gt) == 0: return
    print(f"\n[Deep Dive] Top Confusions in Tightest Clusters (Target: {target_col}, N={tightest_n})")
    
    label_map = clusterer.create_clusters(tightest_n)
    if not label_map: return
    
    cluster_to_members = {}
    for label, cid in label_map.items():
        if cid not in cluster_to_members: cluster_to_members[cid] = []
        cluster_to_members[cid].append(label)

    error_mask = df_gt[target_col] != df_pred[target_col]
    gt_errors = df_gt.loc[error_mask, target_col].values
    pred_errors = df_pred.loc[error_mask, target_col].values
    
    confusion_counts = {}
    for gt, pred in zip(gt_errors, pred_errors):
        if gt in label_map and pred in label_map:
            cid_gt = label_map[gt]
            cid_pred = label_map[pred]
            if cid_gt == cid_pred:
                pair = f"{gt} -> {pred}"
                members = cluster_to_members[cid_gt]
                members_str = ", ".join(members)
                if len(members_str) > 50: members_str = members_str[:47] + "..."
                cluster_info = f"Cluster {cid_gt}: [{members_str}]"
                key = (pair, cluster_info)
                confusion_counts[key] = confusion_counts.get(key, 0) + 1
    
    sorted_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
    if not sorted_confusions:
        print("No intra-cluster confusions found.")
    else:
        print(f"{'Count':<6} | {'Confusion Pair (GT -> Pred)':<40} | {'Cluster Info'}")
        print("-" * 100)
        for (pair, info), count in sorted_confusions[:15]:
            print(f"{count:<6} | {pair:<40} | {info}")

# ==========================================
# 3. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default=PRED_FILE)
    parser.add_argument("--test_file", type=str, default=TEST_FILE)
    args = parser.parse_args()

    if not os.path.exists(args.pred_file) or not os.path.exists(args.test_file):
        print("Error: Input files not found.")
        return

    # Load Data
    df_gt, df_pred = load_data(args.pred_file, args.test_file)
    
    if len(df_gt) == 0:
        return # エラーメッセージはload_data内で表示済み

    # Init Clusterers
    all_scenarios = list(set(df_gt['scenario'].unique().tolist() + df_pred['scenario'].unique().tolist()))
    all_actions = list(set(df_gt['action'].unique().tolist() + df_pred['action'].unique().tolist()))

    print("\nInitializing Clusterers...")
    sc_clusterer = LabelClusterer(all_scenarios)
    ac_clusterer = LabelClusterer(all_actions)
    
    # Analysis
    analyze_trend(df_gt, df_pred, 'scenario', sc_clusterer, [18, 12, 8, 5, 3])
    analyze_trend(df_gt, df_pred, 'action', ac_clusterer, [40, 30, 20, 10, 5])
    
    analyze_confusion_in_tightest_cluster(df_gt, df_pred, 'action', ac_clusterer, tightest_n=10)

if __name__ == "__main__":
    main()