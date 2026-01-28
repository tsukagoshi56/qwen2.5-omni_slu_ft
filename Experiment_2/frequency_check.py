import json
import pandas as pd
import argparse
import os
from collections import Counter

# ==========================================
# 0. Configuration
# ==========================================
PRED_FILE = "prediction.jsonl"
TEST_FILE = "test.jsonl"

pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
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
# 2. Frequency Bias Analysis
# ==========================================
def analyze_frequency_bias(df_gt, df_pred, target_col):
    if len(df_gt) == 0: return

    # 1. データ全体の頻度分布 (Ground Truth)
    # 「本来の正解データ」において、どのラベルがどれくらい多いか
    gt_counts = df_gt[target_col].value_counts(normalize=True)
    
    # 2. エラー時の予測分布 (Predictions in Error)
    # 「間違えたとき」に、モデルがどのラベルを選んでしまったか
    error_mask = df_gt[target_col] != df_pred[target_col]
    total_errors = error_mask.sum()
    
    if total_errors == 0:
        print(f"No errors found for {target_col}.")
        return

    pred_error_counts = df_pred.loc[error_mask, target_col].value_counts(normalize=True)
    
    print(f"\n========================================================")
    print(f" FREQUENCY BIAS ANALYSIS: {target_col.upper()} (Errors: {total_errors})")
    print(f"========================================================")
    
    # 比較用データフレーム作成
    # Index: Label
    # GT_Freq: 正解データ内でのシェア
    # Error_Pred_Freq: 間違い予測内でのシェア
    df_bias = pd.DataFrame({
        'GT_Freq': gt_counts,
        'Error_Pred_Freq': pred_error_counts
    }).fillna(0.0)
    
    # 「過剰予測率 (Over-Prediction Ratio)」
    # 1.0より大きければ、実力以上にそのラベルを乱発している（＝バイアス）
    df_bias['Bias_Ratio'] = df_bias['Error_Pred_Freq'] / df_bias['GT_Freq']
    
    # エラー予測のシェアが高い順にソート
    df_bias = df_bias.sort_values('Error_Pred_Freq', ascending=False)
    
    print(f"\n[1] Top Labels Predicted by Mistake")
    print(f"    (Is the model just spamming popular labels?)")
    print(f"    Bias_Ratio > 1.5 means the model is 'addicted' to this label.")
    
    print(df_bias.head(15).to_string(formatters={
        'GT_Freq': '{:.1%}'.format,
        'Error_Pred_Freq': '{:.1%}'.format,
        'Bias_Ratio': '{:.2f}x'.format
    }))
    
    # --- 3. Confusion Matrix (Top Pairs) ---
    print(f"\n[2] Top Specific Confusions")
    print(f"    (Specifically, what is being mistaken for what?)")
    
    confusion_counts = Counter()
    gt_errors = df_gt.loc[error_mask, target_col].values
    pred_errors = df_pred.loc[error_mask, target_col].values
    
    for gt, pred in zip(gt_errors, pred_errors):
        confusion_counts[(gt, pred)] += 1
        
    print(f"{'Count':<6} | {'Correct (GT) -> Predicted (Mistake)':<40}")
    print("-" * 60)
    for (gt, pred), count in confusion_counts.most_common(15):
        print(f"{count:<6} | {gt:<18} -> {pred:<18}")

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

    # Analyze
    analyze_frequency_bias(df_gt, df_pred, 'scenario')
    analyze_frequency_bias(df_gt, df_pred, 'action')

if __name__ == "__main__":
    main()