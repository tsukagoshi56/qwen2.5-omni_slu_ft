import json
import os
import argparse

# ==========================================
# Configuration
# ==========================================
# データセットがあるフォルダのパス (必要に応じて変更してください)
DEFAULT_DATA_DIR = "./slurp/dataset/slurp" 
FILES_TO_PROCESS = ["train.jsonl", "devel.jsonl", "test.jsonl"]

def extract_metadata(data_dir):
    unique_scenarios = set()
    unique_actions = set()
    unique_intents = set()
    unique_slots = set()
    
    # 統計用カウンター
    total_samples = 0
    
    print(f"Scanning files in '{data_dir}'...")

    for filename in FILES_TO_PROCESS:
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {filename} (Skipping)")
            continue
            
        print(f"  Processing {filename}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    total_samples += 1
                    
                    scenario = d.get('scenario')
                    action = d.get('action')
                    
                    # 1. Scenario Extraction
                    if scenario:
                        unique_scenarios.add(scenario)
                        
                    # 2. Action Extraction
                    if action:
                        unique_actions.add(action)

                    # 3. Intent (Scenario:Action) Extraction
                    if scenario and action:
                        unique_intents.add(f"{scenario}:{action}")
                        
                    # 4. Slot (Entity) Type Extraction
                    if 'entities' in d:
                        for entity in d['entities']:
                            if 'type' in entity:
                                unique_slots.add(entity['type'])
                                
                except json.JSONDecodeError:
                    continue

    # ソートしてリスト化
    sorted_scenarios = sorted(list(unique_scenarios))
    sorted_actions = sorted(list(unique_actions))
    sorted_intents = sorted(list(unique_intents))
    sorted_slots = sorted(list(unique_slots))
    
    return sorted_scenarios, sorted_actions, sorted_intents, sorted_slots, total_samples

def save_to_json(scenarios, actions, intents, slots, output_file="slurp_metadata.json"):
    data = {
        "scenarios": scenarios,
        "actions": actions,
        "intents": intents,
        "slot_types": slots,
        "counts": {
            "scenario_count": len(scenarios),
            "action_count": len(actions),
            "intent_count": len(intents),
            "slot_type_count": len(slots)
        }
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\nMetadata saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract all unique Scenario, Action, and Slot types from SLURP dataset.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing train.jsonl, devel.jsonl, test.jsonl")
    parser.add_argument("--output", type=str, default="slurp_metadata.json", help="Output JSON file name")
    args = parser.parse_args()

    # 抽出実行
    scenarios, actions, intents, slots, total = extract_metadata(args.data_dir)
    
    if total == 0:
        print("No data processed. Please check the data directory path.")
        return

    # --- 結果表示 ---
    print("\n" + "="*60)
    print(f" SLURP DATASET METADATA (Total Samples: {total})")
    print("="*60)

    print(f"\n[1] SCENARIOS ({len(scenarios)} types):")
    print("-" * 60)
    print(", ".join(scenarios))

    print(f"\n[2] ACTIONS ({len(actions)} types):")
    print("-" * 60)
    print(", ".join(actions))

    print(f"\n[3] INTENTS (SCENARIO:ACTION) ({len(intents)} types):")
    print("-" * 60)
    # 5つずつ表示
    for i in range(0, len(intents), 5):
        print(", ".join(intents[i:i+5]))

    print(f"\n[4] SLOT TYPES ({len(slots)} types):")
    print("-" * 60)
    print(", ".join(slots))
    
    # 保存
    save_to_json(scenarios, actions, intents, slots, args.output)

    # 意図リストをテキストファイルとしても保存
    intent_txt = args.output.replace(".json", "_intents.txt")
    with open(intent_txt, 'w', encoding='utf-8') as f:
        for it in intents:
            f.write(it + "\n")
    print(f"Intent list saved to {intent_txt}")

if __name__ == "__main__":
    main()