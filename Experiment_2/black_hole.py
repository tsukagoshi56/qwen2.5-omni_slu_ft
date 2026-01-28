import json
import pandas as pd
import argparse
import os

# ==========================================
# 0. Configuration
# ==========================================
PRED_FILE = "prediction.jsonl"
TEST_FILE = "slurp/dataset/slurp/test.jsonl"

# ユーザー指定の「怪しい」ラベルリスト (typo修正済み: qiorky -> quirky)
# ※ これらが Scenario なのか Action なのか自動判別して分析します
SUSPECT_LABELS = [
    "general", 
    "qa", 
    "calendar", 
    "quirky", 
    "query", 
    "factoid"
]

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
# 2. Targeted Absorption Analysis
# ==========================================
def analyze_specific_absorptions(df_gt, df_pred, target_col, suspects):
    # このカラムに存在するラベルだけをフィルタリング
    existing_labels = set(df_gt[target_col].unique()) | set(df_pred[target_col].unique())
    valid_suspects = [s for s in suspects if s in existing_labels]
    
    if not valid_suspects:
        return # このカラムには該当ラベルなし

    print(f"\n########################################################")
    print(f" ANALYSIS TARGET: {target_col.upper()} Column")
    print(f" Found Suspects: {valid_suspects}")
    print(f"########################################################")
    
    error_mask = df_gt[target_col] != df_pred[target_col]

    for suspect in valid_suspects:
        # モデルが「この容疑者(suspect)」だと予測したが、間違っていたケース
        # Pred == suspect  AND  GT != suspect
        absorption_mask = error_mask & (df_pred[target_col] == suspect)
        total_absorbed = absorption_mask.sum()
        
        if total_absorbed == 0:
            print(f"\n>>> Label '{suspect}' is clean. (No erroneous predictions predicted this label)")
            continue
            
        print(f"\n>>> PREDATOR: '{suspect}'")
        print(f"    (Predicted {total_absorbed} times incorrectly. What did it eat?)")
        
        # 被害者（本来の正解）を集計
        victims = df_gt.loc[absorption_mask, target_col].value_counts()
        
        # 上位を表示
        print(f"{'Count':<6} | {'Victim (Original Correct Label)':<35} | {'Share'}")
        print("-" * 60)
        
        for label, count in victims.head(10).items():
            share = count / total_absorbed
            print(f"{count:<6} | {label:<35} | {share:.1%}")

        # 簡易診断
        top_victim = victims.index[0]
        top_share = victims.iloc[0] / total_absorbed
        
        if top_share > 0.4:
            print(f"    [!] HIGH CONCENTRATION: Mostly eating '{top_victim}'.")
            print(f"        Likely a semantic hierarchy issue (Parent/Child confusion).")
        elif len(victims) > 5:
             print(f"    [!] TRASH BIN BEHAVIOR: Eating diverse labels.")
             print(f"        Used as a fallback for low confidence.")

# ==========================================
# 3. Main
# ==========================================
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
    if len(df_gt) == 0: return

    # Analyze specifically for the user's list
    # Scenario列とAction列の両方をチェックします
    analyze_specific_absorptions(df_gt, df_pred, 'scenario', SUSPECT_LABELS)
    analyze_specific_absorptions(df_gt, df_pred, 'action', SUSPECT_LABELS)

if __name__ == "__main__":
    main()