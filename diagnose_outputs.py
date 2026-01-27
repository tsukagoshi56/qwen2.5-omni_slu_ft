import json
import os
import glob

def check_yes_no(condition):
    return "YES" if condition else "NO"

def main():
    print("--- DIAGNOSTIC REPORT ---\n")
    
    # 1. Test Data Check
    test_file = "slurp/dataset/slurp/test.jsonl"
    print(f"[CHECK 1] Test Data ({test_file})")
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            try:
                data = json.loads(first_line)
                has_sentence = "sentence" in data or "transcript" in data
                print(f"  - File Exists: YES")
                print(f"  - Valid JSON Format: YES")
                print(f"  - Has Input Text (sentence/transcript): {check_yes_no(has_sentence)}")
            except:
                print(f"  - File Exists: YES")
                print(f"  - Valid JSON Format: NO")
    else:
        print(f"  - File Exists: NO")

    # 2. Predictions Check
    print(f"\n[CHECK 2] Predictions Analysis")
    # Search for predictions
    pred_files = glob.glob("inference_outputs/**/predictions.jsonl", recursive=True)
    if not pred_files:
        pred_files = glob.glob("outputs/**/predictions.jsonl", recursive=True)
    if not pred_files:
        pred_files = ["predictions.jsonl"] # Root check
    
    found_any = False
    for p in pred_files:
        if os.path.exists(p):
            found_any = True
            print(f"  - Found Prediction File: {p}")
            
            total = 0
            nulls = 0
            malformed = 0
            
            with open(p, 'r') as f:
                for line in f:
                    total += 1
                    try:
                        d = json.loads(line)
                        scenario = d.get("scenario")
                        action = d.get("action")
                        # Check strictly for "none" strings or actual None
                        is_null = (scenario in ["none", None] and action in ["none", None])
                        if is_null:
                            nulls += 1
                        
                        raw_out = d.get("raw_output", "")
                        if not raw_out or not raw_out.strip():
                             print("    [Warning] Found EMPTY raw_output!")
                        elif "none" in raw_out.lower():
                             # Just a sample check
                             pass
                    except:
                        malformed += 1
            
            if total > 0:
                print(f"  - Total Predictions: {total}")
                print(f"  - Null Outputs (scenario=none, action=none): {nulls} ({100*nulls/total:.1f}%)")
                print(f"  - Malformed/Unparseable: {malformed} ({100*malformed/total:.1f}%)")

                # Sample raw output
                with open(p, 'r') as f:
                    first_line = json.loads(f.readline())
                    print(f"  - Sample Raw Output (First line): {first_line.get('raw_output', 'N/A')[:100]}...")
                
                if nulls / total > 0.9:
                    print("\n  >> DIAGNOSIS: HIGH NULL RATE. Model is generating valid JSON but predicting 'none'.")
                    print("     Possible causes: Input format mismatch (e.g. prompt template), Audio/Text token confusion.")
                elif malformed / total > 0.9:
                     print("\n  >> DIAGNOSIS: HIGH MALFORMED RATE. JSON parsing is failing.")
                     print("     Possible causes: Model outputting raw text instead of JSON, or markdown blocks.")
            else:
                print("  - File is empty.")

    if not found_any:
        print("  - Prediction File Found: NO")

if __name__ == "__main__":
    main()
