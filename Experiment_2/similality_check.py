import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish # 音響・文字列距離計算用
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
# 2. Multi-Modal Similarity Analyzer
# ==========================================
class MultiModalAnalyzer:
    def __init__(self, labels, embedding_model_name=MODEL_NAME):
        self.labels = sorted(list(set(labels)))
        print(f"Initializing Analyzer for {len(self.labels)} labels...")
        
        # 1. Semantic (Meaning) - Embedding
        self.model = SentenceTransformer(embedding_model_name)
        self.embeddings = self.model.encode(self.labels)
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        self.sem_matrix = cosine_similarity(self.embeddings) # Pre-compute N x N
        
    def get_semantic_sim(self, s1, s2):
        """意味的類似度 (Cosine Similarity: -1.0 ~ 1.0)"""
        if s1 not in self.label_to_idx or s2 not in self.label_to_idx: return 0.0
        return self.sem_matrix[self.label_to_idx[s1]][self.label_to_idx[s2]]

    def get_lexical_sim(self, s1, s2):
        """単語的類似度 (Jaro-Winkler: 0.0 ~ 1.0) - スペルの近さ"""
        # Jaro-Winklerは接頭辞の一致を重視するため、単語の類似度判定に向く
        return jellyfish.jaro_winkler_similarity(s1, s2)

    def get_phonetic_sim(self, s1, s2):
        """音響的類似度 (Metaphone Levenshtein) - 発音の近さ"""
        # Metaphoneアルゴリズムで発音コードに変換 (例: 'phone' -> 'FN')
        m1 = jellyfish.metaphone(s1)
        m2 = jellyfish.metaphone(s2)
        
        # 発音コード同士のレーベンシュタイン距離を計算し、0~1に正規化
        dist = jellyfish.levenshtein_distance(m1, m2)
        max_len = max(len(m1), len(m2))
        if max_len == 0: return 1.0 if m1 == m2 else 0.0
        
        # 距離なので、類似度に変換 (1 - 正規化距離)
        return 1.0 - (dist / max_len)

    def analyze_pair(self, gt, pred):
        return {
            "Semantic": self.get_semantic_sim(gt, pred),
            "Lexical":  self.get_lexical_sim(gt, pred),
            "Phonetic": self.get_phonetic_sim(gt, pred)
        }

# ==========================================
# 3. Analysis Logic
# ==========================================

def analyze_errors_3d(df_gt, df_pred, target_col, analyzer):
    if len(df_gt) == 0: return

    error_mask = df_gt[target_col] != df_pred[target_col]
    total_errors = error_mask.sum()
    
    if total_errors == 0:
        print(f"No errors found for {target_col}.")
        return

    gt_errors = df_gt.loc[error_mask, target_col].values
    pred_errors = df_pred.loc[error_mask, target_col].values

    print(f"\n========================================================")
    print(f" 3-AXIS ERROR ANALYSIS: {target_col.upper()} (Total Errors: {total_errors})")
    print(f"========================================================")
    
    # --- 1. Calculate Metrics for All Errors ---
    error_metrics = []
    for gt, pred in zip(gt_errors, pred_errors):
        metrics = analyzer.analyze_pair(gt, pred)
        metrics['GT'] = gt
        metrics['Pred'] = pred
        error_metrics.append(metrics)
        
    df_errors = pd.DataFrame(error_metrics)
    
    # --- 2. Calculate Random Baseline ---
    # ランダムなペアを大量に作って平均を取る
    random_metrics = []
    unique_labels = analyzer.labels
    for _ in range(1000): # 1000 trials
        l1, l2 = random.sample(unique_labels, 2)
        random_metrics.append(analyzer.analyze_pair(l1, l2))
    
    df_random = pd.DataFrame(random_metrics)
    
    # --- 3. Compare Averages (Hypothesis Verification) ---
    print(f"\n[1] Average Similarity Comparison (Error vs Random)")
    print(f"    (Which axis explains the errors best?)")
    
    comparison = []
    for metric in ["Semantic", "Lexical", "Phonetic"]:
        err_avg = df_errors[metric].mean()
        rnd_avg = df_random[metric].mean()
        lift = err_avg / rnd_avg if rnd_avg > 0 else 0
        comparison.append({
            "Metric": metric,
            "Error_Avg": err_avg,
            "Random_Avg": rnd_avg,
            "Lift (Bias)": lift
        })
        
    df_comp = pd.DataFrame(comparison)
    print(df_comp.to_string(index=False, formatters={
        'Error_Avg': '{:.4f}'.format,
        'Random_Avg': '{:.4f}'.format,
        'Lift (Bias)': '{:.2f}x'.format
    }))
    
    # 最もLiftが高いものが、間違いの主要因である可能性が高い
    best_metric = df_comp.sort_values("Lift (Bias)", ascending=False).iloc[0]
    print(f"\n>>> Main Driver: Errors are mostly biased by **{best_metric['Metric'].upper()}** similarity.")

    # --- 4. Deep Dive by Category ---
    # 各指標でトップランクの間違いを表示
    
    def show_top_k(df, metric_name, k=5):
        print(f"\n[Top {k} Errors by {metric_name} Similarity]")
        # その指標が高く、かつ他の指標と差があるものを見たいが、シンプルにその指標でソート
        top_df = df.sort_values(metric_name, ascending=False).head(k)
        
        print(f"{'Metric Val':<10} | {'Correct -> Pred':<30} | {'Other Scores (Sem/Lex/Phon)'}")
        print("-" * 80)
        for _, row in top_df.iterrows():
            scores = f"S:{row['Semantic']:.2f} L:{row['Lexical']:.2f} P:{row['Phonetic']:.2f}"
            print(f"{row[metric_name]:<10.4f} | {row['GT']:<12} -> {row['Pred']:<14} | {scores}")

    show_top_k(df_errors, "Semantic", k=8)
    show_top_k(df_errors, "Lexical", k=5)
    show_top_k(df_errors, "Phonetic", k=5)

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

    # Init Analyzer
    all_scenarios = list(set(df_gt['scenario'].unique().tolist() + df_pred['scenario'].unique().tolist()))
    all_actions = list(set(df_gt['action'].unique().tolist() + df_pred['action'].unique().tolist()))

    print("\nInitializing 3-Axis Analyzers...")
    sc_analyzer = MultiModalAnalyzer(all_scenarios)
    ac_analyzer = MultiModalAnalyzer(all_actions)
    
    # Analyze
    analyze_errors_3d(df_gt, df_pred, 'scenario', sc_analyzer)
    analyze_errors_3d(df_gt, df_pred, 'action', ac_analyzer)

if __name__ == "__main__":
    main()