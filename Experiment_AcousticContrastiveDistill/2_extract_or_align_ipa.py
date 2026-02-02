#!/usr/bin/env python3
import argparse
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common import write_json, write_jsonl


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().strip().split())


def normalize_ipa(text: str) -> str:
    t = str(text or "").strip()
    t = t.replace("/", " ").replace("[", " ").replace("]", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def levenshtein_tokens(a: List[str], b: List[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, start=1):
        cur = [i]
        for j, tb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            rep = prev[j - 1] + (ta != tb)
            cur.append(min(ins, dele, rep))
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein_tokens(ref, hyp) / max(1, len(ref))


def ipa_error_rate(reference: str, hypothesis: str) -> float:
    ref_norm = normalize_ipa(reference)
    hyp_norm = normalize_ipa(hypothesis)
    ref_toks = ref_norm.split() if " " in ref_norm else list(ref_norm)
    hyp_toks = hyp_norm.split() if " " in hyp_norm else list(hyp_norm)
    if not ref_toks:
        return 0.0 if not hyp_toks else 1.0
    return levenshtein_tokens(ref_toks, hyp_toks) / max(1, len(ref_toks))


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    s = raw.find("{")
    e = raw.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    chunk = raw[s : e + 1]
    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def array_to_b64_wav(audio_array: np.ndarray, sampling_rate: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio_array, sampling_rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_vllm_omni(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    audio_b64: str,
    nbest: int,
    num_beams: int,
    max_tokens: int,
    timeout_sec: int,
    audio_payload_mode: str,
) -> Dict[str, Any]:
    if audio_payload_mode == "input_audio":
        content = [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
        ]
    else:
        data_uri = f"data:audio/wav;base64,{audio_b64}"
        content = [
            {"type": "text", "text": prompt},
            {"type": "audio_url", "audio_url": {"url": data_uri}},
        ]

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "n": nbest,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "use_beam_search": True,
        "best_of": max(nbest, num_beams),
        "top_p": 1.0,
        "length_penalty": 1.0,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = api_base.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def decode_whisper_nbest(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_array: np.ndarray,
    sampling_rate: int,
    device: str,
    num_beams: int,
    nbest: int,
    language: str,
) -> List[str]:
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if device.startswith("cuda"):
        inputs["input_features"] = inputs["input_features"].half()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=max(num_beams, nbest),
            num_return_sequences=nbest,
            max_new_tokens=128,
            early_stopping=True,
            language=language,
            task="transcribe",
        )
    texts = processor.batch_decode(output_ids, skip_special_tokens=True)
    dedup: List[str] = []
    seen = set()
    for t in texts:
        norm = normalize_text(t)
        if norm in seen:
            continue
        seen.add(norm)
        dedup.append(str(t).strip())
    return dedup[:nbest]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step-2: install/load IPA corpus, run n-best beam search for Qwen3-Omni IPA and Whisper ASR, "
            "then compare accuracy gaps."
        )
    )
    parser.add_argument("--hf_dataset", type=str, required=True, help="Hugging Face dataset name")
    parser.add_argument("--hf_config", type=str, default=None)
    parser.add_argument("--hf_split", type=str, default="test")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--audio_column", type=str, default="audio")
    parser.add_argument("--text_column", type=str, default="sentence")
    parser.add_argument("--ipa_column", type=str, default="ipa")
    parser.add_argument("--id_column", type=str, default="")
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--omni_api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--omni_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--omni_model", type=str, default="Qwen/Qwen3-Omni-7B")
    parser.add_argument("--omni_audio_payload_mode", type=str, default="input_audio", choices=["input_audio", "audio_url"])
    parser.add_argument("--omni_num_beams", type=int, default=5)
    parser.add_argument("--omni_nbest", type=int, default=5)
    parser.add_argument("--omni_max_tokens", type=int, default=256)
    parser.add_argument("--omni_timeout_sec", type=int, default=120)
    parser.add_argument("--omni_retries", type=int, default=1)
    parser.add_argument("--omni_sleep_sec", type=float, default=0.2)

    parser.add_argument("--asr_model_name_or_path", type=str, default="openai/whisper-large-v3-turbo")
    parser.add_argument("--asr_language", type=str, default="en")
    parser.add_argument("--asr_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--asr_num_beams", type=int, default=5)
    parser.add_argument("--asr_nbest", type=int, default=5)

    parser.add_argument(
        "--output_samples",
        type=str,
        default="Experiment_AcousticContrastiveDistill/outputs/02_ipa_corpus_eval_samples.jsonl",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="Experiment_AcousticContrastiveDistill/outputs/02_ipa_corpus_eval_report.json",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    omni_api_key = os.environ.get(args.omni_api_key_env, "")

    print("[INFO] loading IPA corpus from Hugging Face (this step installs/downloads to local cache)")
    dataset = load_dataset(
        path=args.hf_dataset,
        name=args.hf_config,
        split=args.hf_split,
        cache_dir=args.cache_dir,
    )
    dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=args.sampling_rate))
    if args.limit and args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    print(f"[INFO] dataset={args.hf_dataset} split={args.hf_split} rows={len(dataset)}")
    print(f"[INFO] loading ASR model: {args.asr_model_name_or_path}")
    asr_processor = WhisperProcessor.from_pretrained(args.asr_model_name_or_path)
    asr_model = WhisperForConditionalGeneration.from_pretrained(
        args.asr_model_name_or_path,
        torch_dtype=torch.float16 if args.asr_device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
    ).to(args.asr_device)
    asr_model.eval()

    omni_prompt = (
        "You are an expert in phonetics.\n"
        "Return strict JSON only: {\"transcript\": \"...\", \"ipa\": \"...\"}\n"
        "No explanation, no markdown."
    )

    sample_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(tqdm(dataset, desc="Evaluating IPA corpus", unit="utt")):
        rid = ""
        if args.id_column:
            rid = str(row.get(args.id_column, "")).strip()
        if not rid:
            rid = str(idx)

        audio_item = row.get(args.audio_column) or {}
        audio_array = np.asarray(audio_item.get("array", []), dtype=np.float32)
        if audio_array.size == 0:
            sample_rows.append({"id": rid, "error": "empty_audio"})
            continue
        sr = int(audio_item.get("sampling_rate", args.sampling_rate))

        ref_text = str(row.get(args.text_column, "")).strip()
        ref_ipa = str(row.get(args.ipa_column, "")).strip() if args.ipa_column in row else ""

        asr_hyps = decode_whisper_nbest(
            model=asr_model,
            processor=asr_processor,
            audio_array=audio_array,
            sampling_rate=sr,
            device=args.asr_device,
            num_beams=args.asr_num_beams,
            nbest=args.asr_nbest,
            language=args.asr_language,
        )

        omni_hyps: List[Dict[str, str]] = []
        omni_error = ""
        audio_b64 = array_to_b64_wav(audio_array, sr)
        for _ in range(args.omni_retries + 1):
            try:
                omni_obj = call_vllm_omni(
                    api_base=args.omni_api_base,
                    api_key=omni_api_key,
                    model=args.omni_model,
                    prompt=omni_prompt,
                    audio_b64=audio_b64,
                    nbest=args.omni_nbest,
                    num_beams=args.omni_num_beams,
                    max_tokens=args.omni_max_tokens,
                    timeout_sec=args.omni_timeout_sec,
                    audio_payload_mode=args.omni_audio_payload_mode,
                )
                choices = omni_obj.get("choices", []) or []
                for ch in choices:
                    raw = str(ch.get("message", {}).get("content", "")).strip()
                    parsed = extract_json_object(raw) or {}
                    omni_hyps.append(
                        {
                            "raw": raw,
                            "transcript": str(parsed.get("transcript", "")).strip(),
                            "ipa": str(parsed.get("ipa", "")).strip(),
                        }
                    )
                omni_error = ""
                break
            except urllib.error.HTTPError as e:
                omni_error = f"http_error_{e.code}"
            except Exception as e:
                omni_error = str(e)
            time.sleep(args.omni_sleep_sec)

        asr_wer_top1 = None
        asr_wer_oracle = None
        omni_wer_top1 = None
        omni_wer_oracle = None
        omni_per_top1 = None
        omni_per_oracle = None

        if ref_text and asr_hyps:
            asr_wer_top1 = word_error_rate(ref_text, asr_hyps[0])
            asr_wer_oracle = min(word_error_rate(ref_text, h) for h in asr_hyps)
        if ref_text and omni_hyps:
            omni_texts = [h.get("transcript", "") for h in omni_hyps if h.get("transcript", "").strip()]
            if omni_texts:
                omni_wer_top1 = word_error_rate(ref_text, omni_texts[0])
                omni_wer_oracle = min(word_error_rate(ref_text, t) for t in omni_texts)
        if ref_ipa and omni_hyps:
            omni_ipas = [h.get("ipa", "") for h in omni_hyps if h.get("ipa", "").strip()]
            if omni_ipas:
                omni_per_top1 = ipa_error_rate(ref_ipa, omni_ipas[0])
                omni_per_oracle = min(ipa_error_rate(ref_ipa, p) for p in omni_ipas)

        sample_rows.append(
            {
                "id": rid,
                "reference_text": ref_text,
                "reference_ipa": ref_ipa,
                "asr_nbest": asr_hyps,
                "omni_nbest": omni_hyps,
                "metrics": {
                    "asr_wer_top1": asr_wer_top1,
                    "asr_wer_oracle": asr_wer_oracle,
                    "omni_wer_top1": omni_wer_top1,
                    "omni_wer_oracle": omni_wer_oracle,
                    "omni_per_top1": omni_per_top1,
                    "omni_per_oracle": omni_per_oracle,
                    "omni_vs_asr_top1_wer_gap": (
                        None
                        if omni_wer_top1 is None or asr_wer_top1 is None
                        else (omni_wer_top1 - asr_wer_top1)
                    ),
                },
                "error": omni_error,
            }
        )

    def collect(metric_key: str) -> List[float]:
        vals: List[float] = []
        for row in sample_rows:
            m = row.get("metrics", {}) or {}
            v = m.get(metric_key)
            if v is None:
                continue
            vals.append(float(v))
        return vals

    def mean_or_none(values: List[float]) -> Optional[float]:
        return None if not values else float(sum(values) / len(values))

    report = {
        "dataset": {
            "hf_dataset": args.hf_dataset,
            "hf_config": args.hf_config,
            "hf_split": args.hf_split,
            "num_rows_evaluated": len(sample_rows),
            "audio_column": args.audio_column,
            "text_column": args.text_column,
            "ipa_column": args.ipa_column,
        },
        "decoding": {
            "omni_num_beams": args.omni_num_beams,
            "omni_nbest": args.omni_nbest,
            "asr_num_beams": args.asr_num_beams,
            "asr_nbest": args.asr_nbest,
        },
        "metrics_mean": {
            "asr_wer_top1": mean_or_none(collect("asr_wer_top1")),
            "asr_wer_oracle": mean_or_none(collect("asr_wer_oracle")),
            "omni_wer_top1": mean_or_none(collect("omni_wer_top1")),
            "omni_wer_oracle": mean_or_none(collect("omni_wer_oracle")),
            "omni_per_top1": mean_or_none(collect("omni_per_top1")),
            "omni_per_oracle": mean_or_none(collect("omni_per_oracle")),
            "omni_vs_asr_top1_wer_gap": mean_or_none(collect("omni_vs_asr_top1_wer_gap")),
        },
        "counts": {
            "rows_with_text_ref": len([1 for r in sample_rows if str(r.get("reference_text", "")).strip()]),
            "rows_with_ipa_ref": len([1 for r in sample_rows if str(r.get("reference_ipa", "")).strip()]),
            "rows_with_omni_error": len([1 for r in sample_rows if str(r.get("error", "")).strip()]),
        },
    }

    write_jsonl(args.output_samples, sample_rows)
    write_json(args.output_report, report)
    print(f"[DONE] samples -> {args.output_samples}")
    print(f"[DONE] report  -> {args.output_report}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
