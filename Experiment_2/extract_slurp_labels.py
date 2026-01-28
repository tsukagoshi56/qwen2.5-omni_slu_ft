import json
import os
import argparse
from typing import Set, Dict, List
from collections import defaultdict

def load_jsonl(path: str) -> List[Dict]:
    data = []
    if not os.path.exists(path):
        print(f"Warning: File not found: {path} (Skipping)")
        return data
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding line in {path}: {e}")
    except Exception as e:
        print(f"Error reading {path}: {e}")
        
    return data

def extract_labels(data_dir: str):
    # Files to look for
    files = ["train.jsonl", "devel.jsonl", "test.jsonl"]
    
    scenarios: Set[str] = set()
    actions: Set[str] = set()
    intents: Set[str] = set()
    slots: Set[str] = set()
    scenario_to_actions = defaultdict(set)
    intent_to_slots = defaultdict(set)
    
    total_samples = 0
    found_files = 0
    
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        
        # Check if file exists before trying to load to give clear feedback
        if not os.path.exists(file_path):
            continue
            
        print(f"Reading {file_path}...")
        found_files += 1
        
        items = load_jsonl(file_path)
        total_samples += len(items)
        
        for item in items:
            # 1. Intent extraction (scenario + action)
            scenario = item.get("scenario")
            action = item.get("action")
            current_intent = None
            
            if scenario:
                scenarios.add(scenario)
            if action:
                actions.add(action)
            if scenario and action:
                current_intent = f"{scenario}_{action}"
                intents.add(current_intent)
                scenario_to_actions[scenario].add(action)
            
            # 2. Slot extraction (entities -> type)
            if "entities" in item and isinstance(item["entities"], list):
                for entity in item["entities"]:
                    e_type = entity.get("type")
                    if e_type:
                        slots.add(e_type)
                        # Add to intent mapping if we have a valid intent for this sample
                        if current_intent:
                            intent_to_slots[current_intent].add(e_type)
                        
    return {
        "scenarios": sorted(list(scenarios)),
        "actions": sorted(list(actions)),
        "intents": sorted(list(intents)),
        "slots": sorted(list(slots)),
        "scenario_to_actions": {k: sorted(list(v)) for k, v in scenario_to_actions.items()},
        "intent_to_slots": {k: sorted(list(v)) for k, v in intent_to_slots.items()},
        "total_samples": total_samples,
        "found_files": found_files
    }

def main():
    parser = argparse.ArgumentParser(description="Extract SLURP labels (Intent and Slot tags)")
    # Default matches the typical path structure seen in the repo scripts
    parser.add_argument("--data_dir", type=str, default="slurp/dataset/slurp", 
                        help="Directory containing train.jsonl, devel.jsonl, test.jsonl")
    parser.add_argument("--output_dir", type=str, default="Experiment_2", 
                        help="Directory to save output files")
    
    args = parser.parse_args()
    
    print(f"Looking for SLURP data in: {args.data_dir}")
    
    # Validation
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        print("Please specify the correct path using --data_dir")
        # Don't exit immediately, try to run extraction anyway. 
        # Sometimes paths are relative to where script is run.
    
    results = extract_labels(args.data_dir)
    
    if results["found_files"] == 0:
        print("No valid jsonl files (train/devel/test) found. Exiting.")
        return

    # Prepare outputs
    os.makedirs(args.output_dir, exist_ok=True)
    json_output = os.path.join(args.output_dir, "slurp_labels.json")
    txt_output = os.path.join(args.output_dir, "slurp_labels.txt")
    
    # New separate files
    scenarios_output = os.path.join(args.output_dir, "scenarios.txt")
    actions_output = os.path.join(args.output_dir, "actions.txt")
    slots_output = os.path.join(args.output_dir, "slots.txt")
    
    # 1. Save as JSON
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    # 2. Save Scenarios Only
    with open(scenarios_output, 'w', encoding='utf-8') as f:
        for s in results['scenarios']:
            f.write(f"{s}\n")

    # 3. Save Actions Only
    with open(actions_output, 'w', encoding='utf-8') as f:
        for a in results['actions']:
            f.write(f"{a}\n")
            
    # 4. Save Slots Only
    with open(slots_output, 'w', encoding='utf-8') as f:
        for s in results['slots']:
            f.write(f"{s}\n")

    # 5. Save as Readable Text (Summary)
    with open(txt_output, "w", encoding="utf-8") as f:
        f.write("=== SLURP LABEL STATISTICS ===\n")
        f.write(f"Total Samples Scanned: {results['total_samples']}\n")
        f.write(f"Total Unique Intents: {len(results['intents'])}\n")
        f.write(f"Total Unique Slots:   {len(results['slots'])}\n\n")
        
        f.write(f"--- INTENTS ({len(results['intents'])}) ---\n")
        for i in results['intents']:
            f.write(f"{i}\n")
        f.write("\n")
        
        f.write(f"--- SLOTS ({len(results['slots'])}) ---\n")
        for s in results['slots']:
            f.write(f"{s}\n")
            
        f.write(f"\n--- SCENARIOS ({len(results['scenarios'])}) ---\n")
        f.write(", ".join(results['scenarios']))
        f.write(f"\n\n--- ACTIONS ({len(results['actions'])}) ---\n")
        f.write(", ".join(results['actions']))
        
        f.write("\n\n--- SCENARIO -> ACTIONS MAPPING ---\n")
        for scenario, acts in results.get('scenario_to_actions', {}).items():
            f.write(f"{scenario}:\n")
            for act in acts:
                f.write(f"  - {act}\n")
        
        f.write("\n\n--- INTENT -> VALID SLOTS MAPPING ---\n")
        for intent, valid_slots in results.get('intent_to_slots', {}).items():
            f.write(f"{intent}:\n")
            if not valid_slots:
                f.write(f"  (No slots)\n")
            for slot in valid_slots:
                f.write(f"  - {slot}\n")
        f.write("\n")

    print(f"\nExtraction Complete!")
    print(f"Intents found: {len(results['intents'])}")
    print(f"Slots found:   {len(results['slots'])}")
    print(f"JSON saved to: {json_output}")
    print(f"Text summary:  {txt_output}")
    print(f"Scenarios:     {scenarios_output}")
    print(f"Actions:       {actions_output}")
    print(f"Slots:         {slots_output}")

if __name__ == "__main__":
    main()
