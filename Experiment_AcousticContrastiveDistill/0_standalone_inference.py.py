#!/usr/bin/env python3
import argparse
import os
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
# audio_utilsなどは適宜実装が必要ですが、vLLMはパス指定でいける場合があります

def main():
    parser = argparse.ArgumentParser(description="vLLM Offline Inference for Qwen3-Omni")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-7B")
    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Transcribe this audio.")
    parser.add_argument("--num_beams", type=int, default=5)
    args = parser.parse_args()

    # 1. モデルのロード (ここでGPUメモリを確保します)
    #    サーバーを立てるのと同じ処理がここで行われます
    print(f"Loading model: {args.model}...")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"audio": 1} # 音声が含まれる設定
    )

    # 2. 推論パラメータの設定 (Beam Search)
    sampling_params = SamplingParams(
        n=args.num_beams,           # N-best の数
        best_of=args.num_beams,     # Beam Search の幅
        use_beam_search=True,       # Beam Search を有効化
        temperature=0.0,
        max_tokens=256,
    )

    # 3. 入力データの作成
    #    vLLMのオフライン推論では、プロンプトとマルチモーダルデータを辞書で渡せます
    #    ※Qwen3-Omniの仕様に合わせてプロンプトフォーマットは要調整
    
    # Qwen-Audio系は通常、特定のトークンが必要です
    # 例: <|audio_bos|><|AUDIO|><|audio_eos|> + User Prompt
    prompt_text = f"<|audio_bos|><|AUDIO|><|audio_eos|>\n{args.prompt}"

    inputs = [
        {
            "prompt": prompt_text,
            "multi_modal_data": {
                "audio": args.audio_file  # ファイルパスを直接渡せる場合が多い
                # エラーになる場合は、ここでlibrosa等で読み込んだnumpy arrayを渡します
            },
        }
    ]

    # 4. 高速推論実行
    print("Running inference...")
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    # 5. 結果の表示
    for output in outputs:
        print(f"--- Generated Output (Audio: {args.audio_file}) ---")
        for i, hyp in enumerate(output.outputs):
            print(f"Hypothesis {i+1}: {hyp.text}")

if __name__ == "__main__":
    main()