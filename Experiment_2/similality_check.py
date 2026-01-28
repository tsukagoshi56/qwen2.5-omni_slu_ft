import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import argparse
import os

# ==========================================
# 0. Configuration
# ==========================================
PRED_FILE = "prediction.jsonl"
TEST_FILE = "slurp/dataset/slurp/test.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"
N_CLUSTERS = 20  # 指定の20クラス

# ==========================================
# 1. Cluster Analyzer
# ==========================================
class ClusterAnalyzer:
    def __init__(self, labels, n_clusters=N_CLUSTERS, embedding_model_name=MODEL_NAME):
        self.labels = sorted(list(set(labels)))
        self.n_clusters = min(n_clusters, len(self.labels)) # ラベル数が20未満の場合のケア
        
        print(f"Embedding {len(self.labels)} labels and clustering into {self.n_clusters} groups...")
        self.model = SentenceTransformer(embedding_model_name)
        self.embeddings = self.model.encode(self.labels)
        
        # KMeans実行
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.embeddings)
        
        # マッピング作成
        self.label_to_cluster = {label: cluster_id for label, cluster_id in zip(self.labels, self.cluster_labels)}
        self.cluster_to_labels = {}
        for label, cluster_id in self.label_to_cluster.items():
            self.cluster_to_labels.setdefault(cluster_id, []).append(label)

    def get_cluster(self, label):
        return self.label_to_cluster.get(label, -1)

# ==========================================
# 2. Analysis Logic
# ==========================================

def analyze_cluster_errors(df_gt, df_pred, target_col, analyzer):
    error_mask = df_gt[target_col] != df_pred[target_col]
    total_errors = error_mask.sum()
    
    if total_errors == 0:
        print(f"No errors found for {target_col}.")
        return

    gt_errors = df_gt.loc[error_mask, target_col].values
    pred_errors = df_pred.loc[error_mask, target_col].values

    within_cluster_count = 0
    cluster_error_stats = {} # 各クラスタでのエラー発生頻度

    for gt, pred in zip(gt_errors, pred_errors):
        gt_cluster = analyzer.get_cluster(gt)
        pred_cluster = analyzer.get_cluster(pred)
        
        if gt_cluster == -1 or pred_cluster == -1: continue

        if gt_cluster == pred_cluster:
            within_cluster_count += 1
        
        # どのクラスタ(GT)でミスが起きているか集計
        if gt_cluster not in cluster_error_stats:
            cluster_error_stats[gt_cluster] = {"count": 0, "labels": analyzer.cluster_to_labels[gt_cluster]}
        cluster_error_stats[gt_cluster]["count"] += 1

    print(f"\n" + "="*60)
    print(f" CLUSTER ANALYSIS: {target_col.upper()}")
    print(f" Total Errors: {total_errors}")
    print(f" Total Clusters: {analyzer.n_clusters}")
    print(f"="*60)

    # --- 1. Overall Metrics ---
    within_rate = within_cluster_count / total_errors
    # ランダムに選んだ場合の期待値 (簡易計算: 1/N_CLUSTERS)
    random_expectation = 1.0 / analyzer.n_clusters

    print(f"\n[1] Semantic Consistency of Errors")
    print(f"    Errors within same cluster: {within_cluster_count} ({within_rate:.1%})")
    print(f"    Random baseline expectation: {random_expectation:.1%}")
    print(f"    Lift: {within_rate / random_expectation:.2f}x")

    if within_rate > 0.4:
        print(">>> RESULT: High semantic confusion. The model errors stay within the same topic group.")
    else:
        print(">>> RESULT: Low semantic consistency. Errors are jumping between different topic groups.")

    # --- 2. Top Error Clusters ---
    print(f"\n[2] Top 5 Clusters with Most Errors")
    sorted_clusters = sorted(cluster_error_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print("-" * 80)
    print(f"{'ClusterID':<10} | {'Error Count':<12} | {'Labels in Cluster'}")
    print("-" * 80)
    for cid, info in sorted_clusters[:5]:
        label_str = ", ".join(info['labels'][:6]) # 代表的なラベルをいくつか表示
        if len(info['labels']) > 6: label_str += "..."
        print(f"{cid:<10} | {info['count']:<12} | {label_str}")

# ==========================================
# 3. Data Loading & Main
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default=PRED_FILE)
    parser.add_argument("--test_file", type=str, default=TEST_FILE)
    args = parser.parse_args()

    if not os.path.exists(args.pred_file):
        print(f"Error: Prediction file not found at: {os.path.abspath(args.pred_file)}")
        print("Please specify the correct path using --pred_file <path>")
        return
    if not os.path.exists(args.test_file):
        print(f"Error: Test (GT) file not found at: {os.path.abspath(args.test_file)}")
        print("Please specify the correct path using --test_file <path>")
        return

    df_gt, df_pred = load_data(args.pred_file, args.test_file)

    # ユニークラベルの抽出
    all_scenarios = list(set(df_gt['scenario'].unique().tolist() + df_pred['scenario'].unique().tolist()))
    all_actions = list(set(df_gt['action'].unique().tolist() + df_pred['action'].unique().tolist()))

    # 分析実行
    print("\nRunning Scenario Clustering...")
    sc_analyzer = ClusterAnalyzer(all_scenarios, n_clusters=20)
    analyze_cluster_errors(df_gt, df_pred, 'scenario', sc_analyzer)

    print("\nRunning Action Clustering...")
    # Actionは数が多い場合があるので、ここも20またはそれ以上に調整可能
    ac_analyzer = ClusterAnalyzer(all_actions, n_clusters=20)
    analyze_cluster_errors(df_gt, df_pred, 'action', ac_analyzer)

if __name__ == "__main__":
    main()