import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
pd.set_option('display.colheader_justify', 'left')

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

# ==========================================
# 2. Rank Analyzer
# ==========================================
class RankAnalyzer:
    def __init__(self, labels, embedding_model_name=MODEL_NAME):
        self.labels = sorted(list(set(labels)))
        self.n_labels = len(self.labels)
        print(f"Embedding {self.n_labels} unique labels...")
        
        self.model = SentenceTransformer(embedding_model_name)
        self.embeddings = self.model.encode(self.labels)
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        
        # 全ペアの類似度行列 (N x N)
        self.sim_matrix = cosine_similarity(self.embeddings)
        
    def get_rank_of_pred(self, gt, pred):
        """
        GTに対して、Predが類似度ランキングで何位だったかを返す。
        (自分自身=GT は除外してランク付けする)
        """
        if gt not in self.label_to_idx or pred not in self.label_to_idx:
            return None, 0.0

        gt_idx = self.label_to_idx[gt]
        pred_idx = self.label_to_idx[pred]
        
        # GT行の類似度配列を取得
        sim_scores = self.sim_matrix[gt_idx]
        
        # 類似度が高い順にインデックスをソート (降順)
        # argsortは昇順なので [::-1]
        sorted_indices = np.argsort(sim_scores)[::-1]
        
        # sorted_indices の中には自分自身(GT)も含まれる（通常sim=1.0で1位）
        # なので、Predがリストの何番目にあるかを探す
        # np.where は該当するインデックスの位置を返す
        rank_position = np.where(sorted_indices == pred_idx)[0][0]
        
        # rank_position: 0なら1位(自分自身), 1なら2位(一番似てる他人)...
        # 「自分以外の中で何位か」を知りたいので、そのままの値が順位になる
        # (例: 0番目は自分なので無視、1番目がTop-1 Neighbor)
        return rank_position, sim_scores[pred_idx]

# ==========================================
# 3. Analysis Logic
# ==========================================

def analyze_neighbor_ranks(df_gt, df_pred, target_col, analyzer):
    if len(df_gt) == 0: return

    error_mask = df_gt[target_col] != df_pred[target_col]
    total_errors = error_mask.sum()
    
    if total_errors == 0:
        print(f"No errors found for {target_col}.")
        return

    gt_errors = df_gt.loc[error_mask, target_col].values
    pred_errors = df_pred.loc[error_mask, target_col].values

    print(f"\n========================================================")
    print(f" RANK ANALYSIS: {target_col.upper()} (Total Errors: {total_errors})")
    print(f" Candidates Pool Size: {analyzer.n_labels} labels")
    print(f"========================================================")
    
    ranks = []
    sims = []
    
    for gt, pred in zip(gt_errors, pred_errors):
        rank, sim = analyzer.get_rank_of_pred(gt, pred)
        if rank is not None:
            ranks.append(rank)
            sims.append(sim)
            
    ranks = np.array(ranks)
    
    # --- 1. Rank Distribution (Critical Proof) ---
    print(f"\n[1] How 'close' was the mistake? (Rank Distribution)")
    print(f"    If errors are random, this should be uniform (avg rank ~{analyzer.n_labels//2}).")
    print(f"    If errors are semantic, most should be Rank 1 or 2.")
    
    # 累積割合を計算
    top1_count = np.sum(ranks == 1)
    top3_count = np.sum(ranks <= 3)
    top5_count = np.sum(ranks <= 5)
    top10_count = np.sum(ranks <= 10)
    
    # Random Baseline Expectation
    # ランダムに選んだ場合、Top-Nに入る確率は N / (Total_Labels - 1)
    total_candidates = analyzer.n_labels - 1
    
    def get_random_prob(n):
        return min(1.0, n / total_candidates) if total_candidates > 0 else 0

    stats = [
        {"Range": "Top-1 (Closest Neighbor)", "Count": top1_count, "Rate": top1_count/total_errors, "Random_Base": get_random_prob(1)},
        {"Range": "Top-3",                    "Count": top3_count, "Rate": top3_count/total_errors, "Random_Base": get_random_prob(3)},
        {"Range": "Top-5",                    "Count": top5_count, "Rate": top5_count/total_errors, "Random_Base": get_random_prob(5)},
        {"Range": "Top-10",                   "Count": top10_count,"Rate": top10_count/total_errors,"Random_Base": get_random_prob(10)},
    ]
    
    df_stats = pd.DataFrame(stats)
    
    # Lift (Rate / Random_Base) を計算
    df_stats["Lift"] = df_stats.apply(lambda x: x["Rate"] / x["Random_Base"] if x["Random_Base"] > 0 else 0, axis=1)
    
    print(df_stats.to_string(index=False, formatters={
        'Rate': '{:.1%}'.format,
        'Random_Base': '{:.1%}'.format,
        'Lift': '{:.2f}x'.format
    }))
    
    # 結論判定
    top3_rate = top3_count / total_errors
    if top3_rate > 0.5:
        print(f"\n>>> PROOF POSITIVE: {top3_rate:.1%} of errors are within the Top-3 closest meanings.")
        print("    The model is definitively confused by semantic neighbors.")
    else:
        print(f"\n>>> RESULT WEAK: Only {top3_rate:.1%} of errors are Top-3 neighbors.")
    
    # --- 2. Top-1 Mistake Detail ---
    print(f"\n[2] The 'Twin' Confusions (Most Frequent Rank #1 Errors)")
    print("    These pairs are the absolute closest semantic neighbors.")
    
    top1_errors = {}
    for gt, pred, rank, sim in zip(gt_errors, pred_errors, ranks, sims):
        if rank == 1: # まさに一番似ているやつを選んだケース
            key = (gt, pred)
            top1_errors[key] = {'count': 0, 'sim': sim}
    
    # カウント集計
    for gt, pred, rank in zip(gt_errors, pred_errors, ranks):
        if rank == 1:
            top1_errors[(gt, pred)]['count'] += 1
            
    # ソート
    sorted_top1 = sorted(top1_errors.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print("-" * 80)
    print(f"{'Count':<6} | {'Correct -> Pred (The #1 Neighbor)':<40} | {'Similarity'}")
    print("-" * 80)
    
    for (gt, pred), info in sorted_top1[:15]:
        print(f"{info['count']:<6} | {gt:<18} -> {pred:<18} | {info['sim']:.4f}")

# ==========================================
# 4. Main
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

    # Init
    all_scenarios = list(set(df_gt['scenario'].unique().tolist() + df_pred['scenario'].unique().tolist()))
    all_actions = list(set(df_gt['action'].unique().tolist() + df_pred['action'].unique().tolist()))

    print("\nInitializing Rank Analyzer...")
    sc_analyzer = RankAnalyzer(all_scenarios)
    ac_analyzer = RankAnalyzer(all_actions)
    
    # Analyze
    analyze_neighbor_ranks(df_gt, df_pred, 'scenario', sc_analyzer)
    analyze_neighbor_ranks(df_gt, df_pred, 'action', ac_analyzer)

if __name__ == "__main__":
    main()