#!/usr/bin/env python3
import argparse
from typing import List

import librosa
import torch
from transformers import AutoProcessor

try:
    from transformers import Qwen2AudioForConditionalGeneration

    MODEL_CLS = Qwen2AudioForConditionalGeneration
except Exception:
    from transformers import AutoModelForCausalLM

    MODEL_CLS = AutoModelForCausalLM


def load_audio(path: str, sr: int = 16000) -> List[float]:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Qwen2-Audio inference script.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Transcribe this audio.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = MODEL_CLS.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    audio = load_audio(args.audio_file, sr=args.sampling_rate)
    user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": args.prompt}]
    text_input = processor.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(text=text_input, audio=[audio], sampling_rate=args.sampling_rate, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=max(args.num_beams, args.nbest),
            num_return_sequences=args.nbest,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    for idx, seq in enumerate(output_ids):
        text = processor.decode(seq[input_len:], skip_special_tokens=True).strip()
        print(f"Hypothesis {idx + 1}: {text}")


if __name__ == "__main__":
    main()

