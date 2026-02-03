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

def select_diverse_hypotheses(hypotheses: list, k: int, dedup_mode: str = "normalize") -> list:
    """
    プールされた仮説の中から、互いに編集距離が遠いものを選抜して多様性を確保する
    """
    if k <= 0 or not hypotheses:
        return []
    
    # 1. 重複排除（完全に同じテキストはまとめる）
    if dedup_mode == "none":
        unique_hyps = list(hypotheses)
    else:
        unique_map = {}
        for h in hypotheses:
            # 空文字は除外しても良いが、ASR結果としてありうるので残す
            key = normalize_text(h) if dedup_mode == "normalize" else h
            if key not in unique_map:
                unique_map[key] = h
        unique_hyps = list(unique_map.values())
    
    # 候補が要求数以下ならそのまま返す
    if len(unique_hyps) <= k:
        return unique_hyps

    # 2. 最初の1つを選択
    # （他の全候補との平均編集距離が最も大きい＝最もユニークなもの、または中心的なものを選ぶ戦略などがあるが、
    #   ここでは「最も確率が高かったであろう最初に見つかったもの」ではなく、
    #   「平均的に他の候補と距離があるもの」をベースに選ぶ）
    def get_avg_dist(target, candidates):
        if len(candidates) <= 1: return 0
        total = sum(levenshtein_distance(normalize_text(target), normalize_text(c)) for c in candidates)
        return total / (len(candidates) - 1)

    # 平均距離が最大のもの（＝外れ値的）ではなく、あえてランダムに選ぶか、
    # あるいは単純にリストの先頭（モデルが最初に出した確率が高いもの）をベースにするのが安定的。
    # ここでは「リストの先頭（高確度）」を1つ目として採用し、そこから離れているものを探す。
    selected = [unique_hyps[0]]
    remaining = unique_hyps[1:]

    # 3. Greedyに「既存の選択済みリストとの最小距離」が最大になるものを追加していく
    while len(selected) < k and remaining:
        best_idx = -1
        max_min_dist = -1
        
        for idx, cand in enumerate(remaining):
            # selectedに含まれるすべてのテキストとの距離を測り、その最小値を「この候補のスコア」とする
            # つまり「すでに選んだどの文章とも似ていない」度合い
            min_dist = min(levenshtein_distance(normalize_text(cand), normalize_text(s)) for s in selected)
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = idx
        
        selected.append(remaining.pop(best_idx))
            
    return selected

def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    try:
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return audio
    except Exception as e:
        print(f"Error loading audio {file_path}: {e}")
        return None

def load_slurp_metadata(slurp_file: str) -> dict:
    metadata = {}
    if not os.path.exists(slurp_file):
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
    parser = argparse.ArgumentParser(description="Run Whisper ASR with Sampling for diverse hypotheses.")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-large-v3-turbo")
    parser.add_argument("--language", type=str, default="en")
    
    # Data arguments
    parser.add_argument("--audio_dir", type=str, default="slurp/slurp_real")
    parser.add_argument("--slurp_test_file", type=str, default="slurp/dataset/slurp/test.jsonl")
    parser.add_argument("--output_file", type=str, default="Experiment_Rationale/real_asr_sampling_data.jsonl")
    parser.add_argument("--recording_index", type=int, default=0)
    
    # Generation arguments (Sampling specific)
    parser.add_argument("--num_hypotheses", type=int, default=5, help="Number of final hypotheses to keep")
    parser.add_argument("--sampling_pool_size", type=int, default=20, help="Generate this many raw samples before filtering")
    parser.add_argument("--temperature", type=float, default=1.0, help="Higher = more diversity (try 0.8-1.2)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling probability")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--dedup_mode", type=str, default="normalize", choices=["normalize", "exact", "none"],
                        help="How to deduplicate before diversity selection")
    parser.add_argument("--debug_raw", action="store_true", help="Print raw sampling stats")
    parser.add_argument("--save_raw", action="store_true", help="Save raw sampled hypotheses to output JSONL")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run full ASR with a reduced number of samples for quick smoke checks.")
    parser.add_argument("--smoke_limit", type=int, default=100, help="Number of samples to process in --smoke mode.")

    args = parser.parse_args()
    set_reproducible(args.seed)
    if args.smoke_limit < 1:
        print(f"[ERROR] --smoke_limit must be >= 1, got {args.smoke_limit}")
        return

    # パス設定
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_dir_path = os.path.join(base_dir, args.audio_dir) if not os.path.isabs(args.audio_dir) else args.audio_dir
    output_file_path = os.path.join(base_dir, args.output_file) if not os.path.isabs(args.output_file) else args.output_file
    slurp_test_path = os.path.join(base_dir, args.slurp_test_file) if not os.path.isabs(args.slurp_test_file) else args.slurp_test_file

    print(f"=== Running Whisper with Sampling (do_sample=True) ===")
    print(f"Model: {args.model_name_or_path}")
    print(f"Temp: {args.temperature}, Top-p: {args.top_p}, Pool Size: {args.sampling_pool_size}")
    if args.smoke:
        print(f"[SMOKE] Enabled. Full Whisper inference runs with reduced samples (smoke_limit={args.smoke_limit}).")

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
    tasks = []
    
    if slurp_metadata:
        for slurp_id, entry in slurp_metadata.items():
            recs = entry.get("recordings", [])
            if len(recs) > args.recording_index:
                filename = recs[args.recording_index].get("file")
                if filename:
                    path = os.path.join(audio_dir_path, filename)
                    if os.path.exists(path):
                        tasks.append((slurp_id, path, entry))
    
    if not tasks:
        print("No metadata matches found. Scanning directory...")
        audio_files = []
        for ext in ["*.flac", "*.wav", "*.mp3"]:
            audio_files.extend(glob.glob(os.path.join(audio_dir_path, ext)))
        for path in audio_files:
            f_id = os.path.splitext(os.path.basename(path))[0]
            tasks.append((f_id, path, {"slurp_id": f_id}))

    if args.limit:
        tasks = tasks[:args.limit]
    if args.smoke:
        tasks = tasks[:args.smoke_limit]
    
    print(f"Processing {len(tasks)} audio files...")

    results = []

    for slurp_id, audio_path, meta_entry in tqdm(tasks):
        audio_data = load_audio(audio_path)
        if audio_data is None:
            continue

        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt").to(args.device)
        if "cuda" in args.device:
            inputs["input_features"] = inputs["input_features"].half()

        try:
            with torch.no_grad():
                # --- Sampling実行部分 ---
                # 多くの候補(pool_size)をサンプリングで生成する
                outputs = model.generate(
                    **inputs,
                    do_sample=True,          # これでSamplingを有効化
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    num_return_sequences=args.sampling_pool_size, # 多めに生成
                    max_new_tokens=128,
                    language=args.language,
                    task="transcribe",
                    no_repeat_ngram_size=3
                )

            # デコード
            raw_texts = processor.batch_decode(outputs, skip_special_tokens=True)
            if args.debug_raw:
                uniq = len(set(raw_texts))
                print(f"[{slurp_id}] raw_hypotheses={len(raw_texts)} unique={uniq}")
                print(f"[{slurp_id}] raw_sample={raw_texts[:5]}")
            
            # 多様性の選抜 (プールから多様なトップK個を選ぶ)
            final_texts = select_diverse_hypotheses(raw_texts, args.num_hypotheses, args.dedup_mode)

            hypotheses = [{"text": t.strip()} for t in final_texts]
            
            output_item = meta_entry.copy()
            output_item["slurp_id"] = slurp_id
            output_item["smoke_mode"] = bool(args.smoke)
            output_item["asr_hypotheses"] = hypotheses
            if args.save_raw:
                output_item["asr_raw_hypotheses"] = [{"text": t.strip()} for t in raw_texts]
            
            results.append(output_item)

        except Exception as e:
            print(f"Error processing {slurp_id}: {e}")

    # 保存
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Saved to {output_file_path}")

if __name__ == "__main__":
    main()
