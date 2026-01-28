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
PRED_FILE = "prediction.jsonl"  # 予測ファイルのパス
TEST_FILE = "test.jsonl"        # 正解ファイルのパス
MODEL_NAME = "all-MiniLM-L6-v2" # 埋め込みモデル

# 表示設定
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

# ==========================================
# 1. Data Loading & Clustering Logic
# ==========================================
def load_data(pred_path, test_path):
    """
    Load prediction and ground truth JSONL files.
    Directly extracts 'scenario' and 'action' from the JSON objects.
    """
    print(f"Loading data from {pred_path} and {test_path}...")
    
    # 1. Load Predictions
    preds = {}
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            sid = d.get('slurp_id')
            if sid is not None:
                # 予測ファイルにも scenario/action がある前提
                if 'scenario' not in d: d['scenario'] = "unknown"
                if 'action' not in d: d['action'] = "unknown"
                preds[sid] = d
            
    # 2. Load Ground Truth (Test)
    gts = []
    pred_list = []
    
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            sid = d.get('slurp_id')
            
            # 正解データから scenario と action を直接取得
            # test.jsonl にはすでにキーが含まれているため、推論ロジックは不要
            if 'scenario' not in d:
                d['scenario'] = "unknown"
            if 'action' not in d:
                d['action'] = "unknown"

            # IDマッチング
            if sid is not None and sid in preds:
                gts.append(d)
                pred_list.append(preds[sid])
                
    # DataFrame化
    df_gt = pd.DataFrame(gts)
    df_pred = pd.DataFrame(pred_list)
    
    print(f"Loaded {len(df_gt)} matched samples.")
    return df_gt, df_pred

class LabelClusterer:
    def __init__(self, labels, embedding_model_name=MODEL_NAME):
        # 重複を除去してソート
        self.labels = sorted(list(set(labels)))
        print(f"Embedding {len(self.labels)} unique labels using {embedding_model_name}...")
        
        self.model = SentenceTransformer(embedding_model_name)
        self.embeddings = self.model.encode(self.labels)
        self.id_to_label = {i: l for i, l in enumerate(self.labels)}
        
    def create_clusters(self, n_clusters):
        """
        指定されたクラスタ数でラベルをグループ化する
        """
        # ラベル数よりクラスタ数が多い場合はそのまま返す（クラスタリング不可）
        if n_clusters >= len(self.labels):
            return {l: i for i, l in enumerate(self.labels)}
        
        # Cosine distance based clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, 
            metric='cosine', 
            linkage='average'
        )
        labels = clustering.fit_predict(self.embeddings)
        return {self.id_to_label[i]: cluster_id for i, cluster_id in enumerate(labels)}

# ==========================================
# 2. Analysis Logic
# ==========================================
def analyze_trend(df_gt, df_pred, target_col, clusterer, n_clusters_list):
    """
    間違えたデータについて、それが「意味的に近いクラスタ内での間違い」かどうかを分析する
    """
    results = []
    
    # 正解と予測が異なるものだけを抽出
    error_mask = df_gt[target_col] != df_pred[target_col]
    total_errors = error_mask.sum()
    
    if total_errors == 0:
        print(f"No errors found for {target_col}.")
        return None

    gt_errors = df_gt.loc[error_mask, target_col].values
    pred_errors = df_pred.loc[error_mask, target_col].values

    print(f"\n--- Analyzing Target: {target_col} (Total Errors: {total_errors}) ---")

    for n_clusters in n_clusters_list:
        # 1. クラスタリング実行 (ラベル -> クラスタID の辞書を取得)
        label_map = clusterer.create_clusters(n_clusters)
        
        # 2. 間違いが「同じクラスタ内」かカウント
        intra_count = 0
        valid_comparisons = 0
        
        for gt, pred in zip(gt_errors, pred_errors):
            # 未知のラベル(unknown等)でEmbeddingに含まれないものはスキップ
            if gt in label_map and pred in label_map:
                valid_comparisons += 1
                if label_map[gt] == label_map[pred]:
                    intra_count += 1
        
        rate = intra_count / valid_comparisons if valid_comparisons > 0 else 0
        avg_cluster_size = len(clusterer.labels) / n_clusters
        
        results.append({
            "N_Clusters": n_clusters,
            "Avg_Group_Size": avg_cluster_size, # 平均して1グループに何個の候補があるか
            "Intra_Error_Count": intra_count,
            "Intra_Error_Rate": rate # これが高いほど「惜しい間違い」をしている
        })
        
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    return df_res

def analyze_confusion_in_tightest_cluster(df_gt, df_pred, target_col, clusterer, tightest_n):
    """
    最も狭い（細かい）クラスタリング設定での具体的な混同ペアを表示する
    """
    print(f"\n[Deep Dive] Top Confusions in Tightest Clusters (Target: {target_col}, N={tightest_n})")
    
    label_map = clusterer.create_clusters(tightest_n)
    
    # ID -> ラベルリストの逆引き辞書 (参考表示用)
    cluster_to_members = {}
    for label, cid in label_map.items():
        if cid not in cluster_to_members: cluster_to_members[cid] = []
        cluster_to_members[cid].append(label)

    # エラー抽出
    error_mask = df_gt[target_col] != df_pred[target_col]
    gt_errors = df_gt.loc[error_mask, target_col].values
    pred_errors = df_pred.loc[error_mask, target_col].values
    
    # 混同ペアのカウント
    confusion_counts = {}
    
    for gt, pred in zip(gt_errors, pred_errors):
        if gt in label_map and pred in label_map:
            cid_gt = label_map[gt]
            cid_pred = label_map[pred]
            
            # 同じクラスタID内で間違えている場合のみ分析対象
            if cid_gt == cid_pred:
                pair = f"{gt} -> {pred}"
                # クラスタ内の他のメンバーも参考表示（長すぎる場合は省略）
                members = cluster_to_members[cid_gt]
                members_str = ", ".join(members)
                if len(members_str) > 50: members_str = members_str[:47] + "..."
                
                cluster_info = f"Cluster {cid_gt}: [{members_str}]"
                key = (pair, cluster_info)
                confusion_counts[key] = confusion_counts.get(key, 0) + 1
    
    # カウント順にソートして上位を表示
    sorted_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_confusions:
        print("No intra-cluster confusions found at this granularity.")
    else:
        print(f"{'Count':<6} | {'Confusion Pair (GT -> Pred)':<40} | {'Cluster Info'}")
        print("-" * 100)
        for (pair, info), count in sorted_confusions[:15]:
            print(f"{count:<6} | {pair:<40} | {info}")

# ==========================================
# 3. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Analyze SLURP prediction results vs ground truth with clustering trend.")
    parser.add_argument("--pred_file", type=str, default=PRED_FILE, help="Path to the prediction.jsonl file")
    parser.add_argument("--test_file", type=str, default=TEST_FILE, help="Path to the test.jsonl (ground truth) file")
    args = parser.parse_args()

    # ファイル存在確認
    if not os.path.exists(args.pred_file) or not os.path.exists(args.test_file):
        print("Error: Input files not found. Please check paths.")
        return

    # Load
    df_gt, df_pred = load_data(args.pred_file, args.test_file)
    
    if len(df_gt) == 0:
        print("Error: No matched data found between prediction and test files.")
        return

    # Init Clusterers (ユニークなラベルを集めてEmbedding)
    all_scenarios = df_gt['scenario'].unique().tolist()
    all_actions = df_gt['action'].unique().tolist()
    
    # prediction側に未知のラベルがある場合も考慮してマージ
    all_scenarios = list(set(all_scenarios + df_pred['scenario'].unique().tolist()))
    all_actions = list(set(all_actions + df_pred['action'].unique().tolist()))

    print("\nInitializing Clusterers...")
    sc_clusterer = LabelClusterer(all_scenarios)
    ac_clusterer = LabelClusterer(all_actions)
    
    # Analysis Steps (分析するクラスタ数の粒度)
    # Scenarioは種類が少ないので少なめに設定
    sc_steps = [18, 12, 8, 5, 3] 
    # Actionは種類が多いので多めに設定
    ac_steps = [40, 30, 20, 10, 5] 
    
    # 1. Trend Analysis
    analyze_trend(df_gt, df_pred, 'scenario', sc_clusterer, sc_steps)
    analyze_trend(df_gt, df_pred, 'action', ac_clusterer, ac_steps)
    
    # 2. Deep Dive (Actionに関して、より細かい粒度での混同を見る)
    analyze_confusion_in_tightest_cluster(df_gt, df_pred, 'action', ac_clusterer, tightest_n=10)

if __name__ == "__main__":
    main()