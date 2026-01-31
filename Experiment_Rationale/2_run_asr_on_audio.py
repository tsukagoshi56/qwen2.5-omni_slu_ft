#!/usr/bin/env python3
import json
import os
import glob
import argparse
import random
import numpy as np
import torch
import librosa
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# --- ユーティリティ関数 ---

def set_reproducible(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_text(text: str) -> str:
    """テキストの正規化（空白除去、小文字化）"""
    return " ".join(text.lower().strip().split())

def levenshtein_distance(a: str, b: str) -> int:
    """レーベンシュタイン距離の計算"""
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    
    # メモリ効率化のため2行のみ保持
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]

def select_diverse_hypotheses(hypotheses: list, k: int) -> list:
    """
    編集距離に基づいて多様な仮説を選択する
    """
    if k <= 0 or not hypotheses:
        return []
    
    # 重複排除
    unique_map = {}
    for h in hypotheses:
        norm = normalize_text(h)
        if norm not in unique_map:
            unique_map[norm] = h
    unique_hyps = list(unique_map.values())
    
    if len(unique_hyps) <= k:
        return unique_hyps

    # 最初の1つを選択（他の候補との平均距離が最大のものをベースにする戦略）
    # シンプルに計算量を抑えるため、ここでは長さを基準に選ぶか、ランダムでも良いが
    # 元のロジックに沿って「平均距離が遠いもの」を選ぶ
    def get_avg_dist(target, candidates):
        total = sum(levenshtein_distance(normalize_text(target), normalize_text(c)) for c in candidates)
        return total / max(len(candidates), 1)

    # 最初の候補決定
    first_idx = max(range(len(unique_hyps)), key=lambda i: get_avg_dist(unique_hyps[i], unique_hyps))
    selected = [unique_hyps[first_idx]]
    remaining = [h for i, h in enumerate(unique_hyps) if i != first_idx]

    # Greedyに多様なものを追加
    while len(selected) < k and remaining:
        best_idx = -1
        max_min_dist = -1
        
        for idx, cand in enumerate(remaining):
            # 既存のselectedとの最小距離を計算
            min_dist = min(levenshtein_distance(normalize_text(cand), normalize_text(s)) for s in selected)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = idx
        
        selected.append(remaining.pop(best_idx))
            
    return selected

def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """librosaを使用して音声をロードし、リサンプリングする"""
    try:
        # librosaは自動的にfloat32のnumpy配列を返す
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return audio
    except Exception as e:
        print(f"Error loading audio {file_path}: {e}")
        return None

def load_slurp_metadata(slurp_file: str) -> dict:
    """SLURPのメタデータをロードし、slurp_idをキーにした辞書を返す"""
    metadata = {}
    if not os.path.exists(slurp_file):
        print(f"Warning: SLURP metadata file not found at {slurp_file}. Running in file-only mode.")
        return metadata
    
    print(f"Loading metadata from {slurp_file}...")
    with open(slurp_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "slurp_id" in entry:
                    metadata[str(entry["slurp_id"])] = entry
            except json.JSONDecodeError:
                continue
    return metadata

# --- メイン処理 ---

def main():
    parser = argparse.ArgumentParser(description="Run Whisper ASR to generate N-best hypotheses.")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-large-v3-turbo")
    parser.add_argument("--language", type=str, default="en")
    
    # Data arguments
    parser.add_argument("--audio_dir", type=str, default="slurp/slurp_real")
    parser.add_argument("--slurp_test_file", type=str, default="slurp/dataset/slurp/test.jsonl")
    parser.add_argument("--output_file", type=str, default="Experiment_Rationale/real_asr_test_data.jsonl")
    parser.add_argument("--recording_index", type=int, default=0, help="Index of recording to use from SLURP metadata")
    
    # Generation arguments
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_hypotheses", type=int, default=5, help="Number of hypotheses to return")
    parser.add_argument("--diversity_method", type=str, choices=["group_beam", "sampling", "beam"], default="beam")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--diversity_penalty", type=float, default=1.0, help="For group_beam")
    
    # System arguments
    parser.add_argument("--batch_size", type=int, default=1, help="Currently processing 1 file at a time is safer for diverse lengths")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files for testing")

    args = parser.parse_args()
    set_reproducible(args.seed)

    # パスの正規化
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_dir_path = os.path.join(base_dir, args.audio_dir) if not os.path.isabs(args.audio_dir) else args.audio_dir
    output_file_path = os.path.join(base_dir, args.output_file) if not os.path.isabs(args.output_file) else args.output_file
    slurp_test_path = os.path.join(base_dir, args.slurp_test_file) if not os.path.isabs(args.slurp_test_file) else args.slurp_test_file

    print(f"Model: {args.model_name_or_path}")
    print(f"Output: {output_file_path}")
    print(f"Device: {args.device}")

    # モデル読み込み
    try:
        processor = WhisperProcessor.from_pretrained(args.model_name_or_path)
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16 if "cuda" in args.device else torch.float32,
            low_cpu_mem_usage=True
        ).to(args.device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # データ準備
    slurp_metadata = load_slurp_metadata(slurp_test_path)
    
    # 音声ファイルのリスト作成
    tasks = [] # (slurp_id, audio_path, metadata_entry)
    
    # 1. メタデータがある場合、それに基づいてファイルを探す
    if slurp_metadata:
        for slurp_id, entry in slurp_metadata.items():
            recs = entry.get("recordings", [])
            if len(recs) > args.recording_index:
                filename = recs[args.recording_index].get("file")
                if filename:
                    path = os.path.join(audio_dir_path, filename)
                    if os.path.exists(path):
                        tasks.append((slurp_id, path, entry))
    
    # 2. メタデータで見つからなかった、またはメタデータが無い場合、ディレクトリをスキャン
    # （既存のtasksに含まれていないものだけ追加などのロジックが必要ならここに追加するが、
    #  今回はシンプルにメタデータ優先、なければglobフォールバックとする）
    if not tasks:
        print("No matching files found via metadata. Scanning directory...")
        audio_files = []
        for ext in ["*.flac", "*.wav", "*.mp3"]:
            audio_files.extend(glob.glob(os.path.join(audio_dir_path, ext)))
        
        for path in audio_files:
            # ファイル名をIDとする
            f_id = os.path.splitext(os.path.basename(path))[0]
            tasks.append((f_id, path, {"slurp_id": f_id}))

    if not tasks:
        print(f"Error: No audio files found in {audio_dir_path}")
        return

    if args.limit:
        tasks = tasks[:args.limit]
    
    print(f"Processing {len(tasks)} audio files...")

    # 結果保存用
    results = []

    # プログレスバー付きでループ
    for slurp_id, audio_path, meta_entry in tqdm(tasks):
        audio_data = load_audio(audio_path, target_sr=16000)
        if audio_data is None:
            continue

        # 入力作成
        inputs = processor(
            audio_data, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).to(args.device)
        
        if "cuda" in args.device:
            inputs["input_features"] = inputs["input_features"].half()

        # 生成設定
        gen_kwargs = {
            "max_new_tokens": 128,
            "language": args.language,
            "task": "transcribe",
            "return_dict_in_generate": True,
            "no_repeat_ngram_size": 3 
        }

        # Diversity Methodに応じた分岐
        try:
            with torch.no_grad():
                if args.diversity_method == "sampling":
                    # Sampling pool strategy
                    pool_size = max(args.num_hypotheses * 2, 20) # 多めに生成して選抜
                    outputs = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=0.95,
                        num_return_sequences=pool_size,
                        **gen_kwargs
                    )
                elif args.diversity_method == "group_beam":
                    # Group Beam Search
                    # num_beams must be divisible by num_beam_groups
                    # ここではシンプルにするため、beam数 = hypothesis数にする
                    n_beams = max(args.num_beams, args.num_hypotheses)
                    outputs = model.generate(
                        **inputs,
                        num_beams=n_beams,
                        num_beam_groups=n_beams, # 各グループ1ビームで多様性を強制
                        diversity_penalty=args.diversity_penalty,
                        num_return_sequences=args.num_hypotheses,
                        **gen_kwargs
                    )
                else:
                    # Standard Beam Search
                    n_beams = max(args.num_beams, args.num_hypotheses)
                    outputs = model.generate(
                        **inputs,
                        num_beams=n_beams,
                        num_return_sequences=args.num_hypotheses,
                        **gen_kwargs
                    )

            # テキストへデコード
            pred_texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
            
            # Samplingの場合は多様なものを選抜
            if args.diversity_method == "sampling":
                final_hyps_text = select_diverse_hypotheses(pred_texts, args.num_hypotheses)
            else:
                final_hyps_text = pred_texts[:args.num_hypotheses]

            # 整形
            hypotheses = [{"text": t.strip()} for t in final_hyps_text]
            
            # メタデータと結合
            output_item = meta_entry.copy()
            output_item["slurp_id"] = slurp_id # 念のため上書き
            output_item["asr_hypotheses"] = hypotheses
            
            results.append(output_item)

        except Exception as e:
            print(f"Error processing {slurp_id}: {e}")
            import traceback
            traceback.print_exc()

    # 保存
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Saved {len(results)} entries to {output_file_path}")

if __name__ == "__main__":
    main()