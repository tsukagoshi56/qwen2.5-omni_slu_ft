import json
import os
import glob
import re

def inspect_jsonl(path, max_lines=5, check_field=None):
    print(f"\n--- Inspcting {path} ---")
    if not os.path.exists(path):
        print("File does not exist.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            try:
                data = json.loads(line)
                print(f"Line {i}: {str(data)[:200]}...") # Truncate for readability
                if check_field and check_field not in data:
                    print(f"  WARNING: Field '{check_field}' missing!")
            except json.JSONDecodeError:
                print(f"Line {i}: [Invalid JSON] {line[:100]}...")

def find_prediction_files():
    # Search for predictions.jsonl in likely output directories
    candidates = glob.glob("inference_outputs/**/predictions.jsonl", recursive=True)
    candidates += glob.glob("outputs/**/predictions.jsonl", recursive=True)
    candidates += ["predictions.jsonl"] # Root
    return [c for c in candidates if os.path.exists(c)]

def main():
    # 1. Check Test File (SLURP)
    test_file = "slurp/dataset/slurp/test.jsonl"
    inspect_jsonl(test_file, check_field="sentence")

    # 2. Check Predictions
    pred_files = find_prediction_files()
    if not pred_files:
        print("\nNo predictions.jsonl found in standard directories.")
    else:
        for p in pred_files:
            inspect_jsonl(p)
            
            # Analyze nulls
            print(f"  Analyzing content of {p}...")
            total = 0
            nulls = 0
            malformed = 0
            with open(p, 'r') as f:
                for line in f:
                    total += 1
                    try:
                        d = json.loads(line)
                        if d.get("scenario") in ["none", None] and d.get("action") in ["none", None]:
                            nulls += 1
                    except:
                        malformed += 1
            print(f"  Total: {total}, Nulls: {nulls} ({100*nulls/total if total else 0:.1f}%), Malformed: {malformed}")

if __name__ == "__main__":
    main()
