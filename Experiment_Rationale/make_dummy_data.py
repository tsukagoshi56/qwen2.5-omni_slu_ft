#!/usr/bin/env python3
import json
import os
import random
import argparse

# ASRエラーシミュレーション用の、簡単な同音異義語・類似音の辞書
# (単語: [置換候補])
HOMOPHONES = {
    "to": ["two", "too"],
    "for": ["four"],
    "see": ["sea"],
    "weather": ["whether"],
    "there": ["their", "they're"],
    "call": ["cold"],
    "set": ["sent"],
    "alarm": ["arm"],
    "play": ["pray"],
    "add": ["ad"],
    "song": ["sung"],
    "what": ["watt"],
    "time": ["thyme"],
}

def generate_hypotheses(sentence: str, num_hypotheses: int = 3) -> list:
    """
    一つの文から、ASRの揺れをシミュレートした複数の仮説を生成する。
    """
    hypotheses = []
    words = sentence.split()

    # 1. 正解の文を最も信頼度の高い仮説として追加
    hypotheses.append({"text": sentence, "confidence": 0.9 + random.uniform(0, 0.09)})

    # 2. その他の仮説を生成
    possible_errors = []

    # 2a. 同音異義語による置換
    for i, word in enumerate(words):
        if word.lower() in HOMOPHONES:
            new_words = words[:]
            new_words[i] = random.choice(HOMOPHONES[word.lower()])
            possible_errors.append(" ".join(new_words))

    # 2b. 単語の欠落
    if len(words) > 2:
        del_idx = random.randint(0, len(words) - 1)
        new_words = words[:del_idx] + words[del_idx+1:]
        possible_errors.append(" ".join(new_words))
        
    # 2c. 単語の繰り返し
    if len(words) > 1:
        rep_idx = random.randint(0, len(words) - 1)
        new_words = words[:rep_idx+1] + [words[rep_idx]] + words[rep_idx+1:]
        possible_errors.append(" ".join(new_words))

    # 生成されたエラー候補からランダムに選択して仮説リストを埋める
    random.shuffle(possible_errors)
    
    # 既存の仮説と重複しないように追加
    existing_texts = {h['text'] for h in hypotheses}
    for error_sentence in possible_errors:
        if len(hypotheses) >= num_hypotheses:
            break
        if error_sentence not in existing_texts:
            hypotheses.append({
                "text": error_sentence,
                "confidence": 0.1 + random.uniform(0, 0.4) # 低めの信頼度
            })
            existing_texts.add(error_sentence)

    # 信頼度順にソート
    hypotheses.sort(key=lambda x: x['confidence'], reverse=True)
    return hypotheses


def main():
    parser = argparse.ArgumentParser(description="Generate dummy ASR n-best data from SLURP test file.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="slurp/dataset/slurp/test.jsonl",
        help="Path to the input SLURP jsonl file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="Experiment_Rationale/dummy_test_data.jsonl",
        help="Path to save the output file with dummy hypotheses."
    )
    parser.add_argument(
        "--num_hypotheses",
        type=int,
        default=3,
        help="Number of ASR hypotheses to generate for each utterance."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of processed entries (for testing)."
    )
    args = parser.parse_args()

    # ベースディレクトリをプロジェクトルートに設定
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input_file)
    output_path = os.path.join(base_dir, args.output_file)

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")

    new_data = []
    with open(input_path, "r") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                print(f"Reached limit of {args.limit} entries.")
                break
            try:
                data = json.loads(line)
                sentence = data.get("sentence")
                if sentence:
                    # 元のデータにASR仮説を追加
                    data["asr_hypotheses"] = generate_hypotheses(sentence, args.num_hypotheses)
                    new_data.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON on line {i+1}")
                continue

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 新しいデータをファイルに書き込む
    with open(output_path, "w") as f:
        for item in new_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nSuccessfully generated {len(new_data)} entries with dummy ASR hypotheses.")
    print(f"Output saved to {output_path}")

    # 例として最初の3件の生成結果を表示
    if len(new_data) > 0:
        print("\n--- Example Output (first 3 entries) ---")
        for item in new_data[:3]:
            print(json.dumps(item, indent=2))
            print("-" * 20)


if __name__ == "__main__":
    main()
