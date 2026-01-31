#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# ----------------------------
# Small helpers
# ----------------------------

STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","to","for","of","on","in","at","by","from",
    "is","are","was","were","be","been","being","do","does","did","have","has","had",
    "i","you","he","she","it","we","they","me","my","your","his","her","its","our","their",
    "this","that","these","those","there","here","now","please","can","could","would","should",
    "what","which","who","whom","when","where","why","how","with","as","about","up","down",
}

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path: str, items: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def resolve_audio_path(audio_root: str, filename: str) -> Optional[str]:
    if not filename:
        return None
    candidates = [
        os.path.join(audio_root, filename),
        os.path.join(audio_root, "slurp_real", filename),
        os.path.join("slurp", "audio", "slurp_real", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def load_audio(path: str, target_sr: int = 16000) -> Optional[List[float]]:
    try:
        audio, _ = librosa.load(path, sr=target_sr, mono=True)
        return audio
    except Exception as exc:
        print(f"[WARN] Failed to load audio: {path} ({exc})")
        return None

def build_entities(record: Dict[str, Any]) -> List[Dict[str, str]]:
    tokens = record.get("tokens", []) or []
    raw_entities = record.get("entities", []) or []
    results: List[Dict[str, str]] = []
    for ent in raw_entities:
        ent_type = ent.get("type", "")
        filler = ent.get("filler")
        if not filler:
            span = ent.get("span") or []
            if isinstance(span, list) and span:
                words = []
                for idx in span:
                    if isinstance(idx, int) and idx < len(tokens):
                        words.append(tokens[idx].get("surface", ""))
                filler = " ".join([w for w in words if w])
        if filler is None:
            filler = ""
        results.append({"type": ent_type, "filler": filler})
    return results

def extract_slot_types(entities: List[Dict[str, str]]) -> List[str]:
    seen = set()
    slot_types: List[str] = []
    for ent in entities:
        t = ent.get("type", "")
        if not t or t in seen:
            continue
        seen.add(t)
        slot_types.append(t)
    return slot_types

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    toks = re.findall(r"[a-z0-9']+", text.lower())
    return [t for t in toks if t and t not in STOPWORDS]

def summarize_nbest(hyp_texts: List[str], max_tokens: int = 6) -> Dict[str, List[str]]:
    if not hyp_texts:
        return {"stable": [], "unstable": []}
    hyp_tokens = [set(tokenize(t)) for t in hyp_texts if t]
    if not hyp_tokens:
        return {"stable": [], "unstable": []}
    counts = Counter()
    for tok_set in hyp_tokens:
        counts.update(tok_set)
    n = len(hyp_tokens)
    stable_thresh = max(1, math.ceil(n * 0.6))
    unstable_thresh = max(1, math.floor(n * 0.4))
    stable = [t for t, c in counts.items() if c >= stable_thresh]
    unstable = [t for t, c in counts.items() if c <= unstable_thresh]
    stable.sort(key=lambda t: (-counts[t], t))
    unstable.sort(key=lambda t: (counts[t], t))
    return {
        "stable": stable[:max_tokens],
        "unstable": unstable[:max_tokens],
    }

def load_metadata(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        return {"scenarios": [], "actions": [], "intents": [], "slot_types": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "scenarios": data.get("scenarios", []),
        "actions": data.get("actions", []),
        "intents": data.get("intents", []),
        "slot_types": data.get("slot_types", []),
    }

def load_clusters(path: str) -> Dict[str, Dict[str, List[str]]]:
    if not os.path.exists(path):
        return {"scenarios": {}, "actions": {}, "slot_types": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    def build_map(section: Dict[str, List[str]]) -> Dict[str, List[str]]:
        label_map = {}
        for _, members in section.items():
            for m in members:
                label_map[m] = members
        return label_map
    return {
        "scenarios": build_map(data.get("scenarios", {})),
        "actions": build_map(data.get("actions", {})),
        "slot_types": build_map(data.get("slot types", {})),
    }

def load_confusing_pairs(path: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("actions", []):
        pair = item.get("pair") or []
        sim = item.get("similarity", 0.0)
        if len(pair) != 2:
            continue
        a, b = pair
        mapping[a].append((b, sim))
        mapping[b].append((a, sim))
    result: Dict[str, List[str]] = {}
    for key, vals in mapping.items():
        vals.sort(key=lambda x: -x[1])
        result[key] = [v[0] for v in vals]
    return result

def pick_recording(recordings: List[Dict[str, Any]], index: int = 0) -> Optional[str]:
    if not recordings:
        return None
    if index < len(recordings):
        return recordings[index].get("file")
    return recordings[0].get("file")

def select_candidates_topk(
    gold: str,
    candidates: List[str],
    k: int,
    rng: random.Random,
) -> List[str]:
    k = max(1, k)
    ordered: List[str] = []
    if gold:
        ordered.append(gold)
    pool = [c for c in candidates if c != gold]
    rng.shuffle(pool)
    ordered.extend(pool[: max(0, k - len(ordered))])
    return ordered[:k]

def select_slot_types_topk(
    gold_types: List[str],
    slot_types: List[str],
    k: int,
    rng: random.Random,
) -> List[str]:
    k = max(1, k)
    ordered: List[str] = []
    for t in gold_types:
        if t not in ordered:
            ordered.append(t)
    pool = [t for t in slot_types if t not in ordered]
    rng.shuffle(pool)
    ordered.extend(pool[: max(0, k - len(ordered))])
    return ordered[:k]

def select_slot_candidates_topk(
    gold_types: List[str],
    slot_types: List[str],
    k: int,
    rng: random.Random,
) -> List[str]:
    return select_slot_types_topk(gold_types, slot_types, k, rng)

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        return None

def build_prompt_nbest(
    gold_scenario: str,
    gold_action: str,
    gold_slot_types: List[str],
    scenario_candidates: List[str],
    action_candidates: List[str],
    slot_candidates: List[str],
    nbest_texts: List[str],
    stable_tokens: List[str],
    unstable_tokens: List[str],
    use_fewshot: bool = True,
) -> str:
    scenario_note = f"scenario_candidates ({len(scenario_candidates)}):"
    action_note = f"action_candidates ({len(action_candidates)}):"
    slot_note = f"slot_candidates ({len(slot_candidates)}):"
    fewshot = ""
    if use_fewshot:
        fewshot = (
            "Example:\n"
            "gold_scenario: music\n"
            "gold_action: play\n"
            "gold_slot_types: [\"song_name\"]\n"
            "scenario_candidates (3): [\"music\",\"alarm\",\"weather\"]\n"
            "action_candidates (3): [\"play\",\"query\",\"set\"]\n"
            "slot_candidates (3): [\"song_name\",\"artist_name\",\"time\"]\n"
            "nbest: [\"play yesterday\",\"play yester day\",\"please play yesterday\"]\n"
            "stable_tokens: [\"play\",\"yesterday\"]\n"
            "unstable_tokens: []\n"
            "Output:\n"
            "{\n"
            "  \"evidence\": \"stable: play,yesterday\",\n"
            "  \"scenario_rejects\": [\n"
            "    {\"scenario\": \"alarm\", \"reason\": \"no alarm words\"},\n"
            "    {\"scenario\": \"weather\", \"reason\": \"no weather words\"}\n"
            "  ],\n"
            "  \"action_rejects\": [\n"
            "    {\"action\": \"query\", \"reason\": \"no question words\"},\n"
            "    {\"action\": \"set\", \"reason\": \"no setting words\"}\n"
            "  ],\n"
            "  \"slot_rejects\": [\n"
            "    {\"slot_type\": \"artist_name\", \"reason\": \"no artist mentioned\"},\n"
            "    {\"slot_type\": \"time\", \"reason\": \"no time mentioned\"}\n"
            "  ]\n"
            "}\n"
            "\n"
        )
    return (
        "You are an SLU rationale generator. Keep everything short.\n"
        "Steps:\n"
        "1) Check n-best variation and note reliable vs unreliable words.\n"
        "2) Reject non-gold scenario candidates with a few-word reason.\n"
        "3) Reject non-gold action candidates with a few-word reason.\n"
        "4) Reject non-gold slot candidates with a few-word reason.\n"
        "Output JSON only.\n"
        "Constraints: evidence <= 12 words; each reason <= 6 words.\n\n"
        f"{fewshot}"
        f"gold_scenario: {gold_scenario}\n"
        f"gold_action: {gold_action}\n"
        f"gold_slot_types: {json.dumps(gold_slot_types, ensure_ascii=False)}\n"
        f"{scenario_note} {json.dumps(scenario_candidates, ensure_ascii=False)}\n"
        f"{action_note} {json.dumps(action_candidates, ensure_ascii=False)}\n"
        f"{slot_note} {json.dumps(slot_candidates, ensure_ascii=False)}\n"
        f"nbest: {json.dumps(nbest_texts, ensure_ascii=False)}\n"
        f"stable_tokens: {json.dumps(stable_tokens, ensure_ascii=False)}\n"
        f"unstable_tokens: {json.dumps(unstable_tokens, ensure_ascii=False)}\n\n"
        "Output schema:\n"
        "{\n"
        '  "evidence": "<stable vs unstable words>",\n'
        '  "scenario_rejects": [{"scenario": "...", "reason": "..."}],\n'
        '  "action_rejects": [{"action": "...", "reason": "..."}],\n'
        '  "slot_rejects": [{"slot_type": "...", "reason": "..."}]\n'
        "}\n"
        "If gold_slot_types is empty, reject all slot candidates with 'not mentioned'."
    )

def build_prompt_audio(
    gold_scenario: str,
    gold_action: str,
    gold_slot_types: List[str],
    scenario_candidates: List[str],
    action_candidates: List[str],
    slot_candidates: List[str],
    use_fewshot: bool = True,
) -> str:
    scenario_note = f"scenario_candidates ({len(scenario_candidates)}):"
    action_note = f"action_candidates ({len(action_candidates)}):"
    slot_note = f"slot_candidates ({len(slot_candidates)}):"
    fewshot = ""
    if use_fewshot:
        fewshot = (
            "Example:\n"
            "gold_scenario: alarm\n"
            "gold_action: set\n"
            "gold_slot_types: [\"time\"]\n"
            "scenario_candidates (3): [\"alarm\",\"music\",\"weather\"]\n"
            "action_candidates (3): [\"set\",\"query\",\"remove\"]\n"
            "slot_candidates (3): [\"time\",\"date\",\"location\"]\n"
            "Output:\n"
            "{\n"
            "  \"evidence\": \"heard time phrase\",\n"
            "  \"scenario_rejects\": [\n"
            "    {\"scenario\": \"music\", \"reason\": \"no music words\"},\n"
            "    {\"scenario\": \"weather\", \"reason\": \"no weather words\"}\n"
            "  ],\n"
            "  \"action_rejects\": [\n"
            "    {\"action\": \"query\", \"reason\": \"not a question\"},\n"
            "    {\"action\": \"remove\", \"reason\": \"no removal words\"}\n"
            "  ],\n"
            "  \"slot_rejects\": [\n"
            "    {\"slot_type\": \"date\", \"reason\": \"no date mentioned\"},\n"
            "    {\"slot_type\": \"location\", \"reason\": \"no location mentioned\"}\n"
            "  ]\n"
            "}\n"
            "\n"
        )
    return (
        "You are an SLU rationale generator. Keep everything short.\n"
        "Steps:\n"
        "1) Note key audio evidence (very brief).\n"
        "2) Reject non-gold scenario candidates with a few-word reason.\n"
        "3) Reject non-gold action candidates with a few-word reason.\n"
        "4) Reject non-gold slot candidates with a few-word reason.\n"
        "Output JSON only.\n"
        "Constraints: evidence <= 12 words; each reason <= 6 words.\n\n"
        f"{fewshot}"
        f"gold_scenario: {gold_scenario}\n"
        f"gold_action: {gold_action}\n"
        f"gold_slot_types: {json.dumps(gold_slot_types, ensure_ascii=False)}\n"
        f"{scenario_note} {json.dumps(scenario_candidates, ensure_ascii=False)}\n"
        f"{action_note} {json.dumps(action_candidates, ensure_ascii=False)}\n"
        f"{slot_note} {json.dumps(slot_candidates, ensure_ascii=False)}\n\n"
        "Output schema:\n"
        "{\n"
        '  "evidence": "<audio keywords>",\n'
        '  "scenario_rejects": [{"scenario": "...", "reason": "..."}],\n'
        '  "action_rejects": [{"action": "...", "reason": "..."}],\n'
        '  "slot_rejects": [{"slot_type": "...", "reason": "..."}]\n'
        "}\n"
        "If gold_slot_types is empty, reject all slot candidates with 'not mentioned'."
    )

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate rationale with Qwen2-Audio (nbest or audio mode).")
    parser.add_argument("--mode", type=str, choices=["nbest", "audio"], default="nbest")
    parser.add_argument("--input_file", type=str, default="Experiment_Rationale/real_asr_sampling_data.jsonl")
    parser.add_argument("--slurp_file", type=str, default="slurp/dataset/slurp/test.jsonl")
    parser.add_argument("--audio_dir", type=str, default="slurp/slurp_real")
    parser.add_argument("--metadata_file", type=str, default="Experiment_3/slurp_metadata.json")
    parser.add_argument("--clusters_file", type=str, default="Experiment_3/slurp_clusters.json")
    parser.add_argument("--confusing_pairs_file", type=str, default="Experiment_3/slurp_confusing_pairs.json")
    parser.add_argument("--output_file", type=str, default="Experiment_Rationale/rationale_output.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recording_index", type=int, default=0)
    parser.add_argument("--num_hypotheses", type=int, default=5)
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidates for scenario/action/slot types.")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview", type=int, default=0, help="Print prompt and output for first N samples.")
    parser.add_argument("--limitmode", action="store_true", help="Print pretty JSON results to stdout.")
    parser.add_argument("--save_raw", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input_file) if not os.path.isabs(args.input_file) else args.input_file
    slurp_path = os.path.join(base_dir, args.slurp_file) if not os.path.isabs(args.slurp_file) else args.slurp_file
    audio_root = os.path.join(base_dir, args.audio_dir) if not os.path.isabs(args.audio_dir) else args.audio_dir
    metadata_path = os.path.join(base_dir, args.metadata_file) if not os.path.isabs(args.metadata_file) else args.metadata_file
    clusters_path = os.path.join(base_dir, args.clusters_file) if not os.path.isabs(args.clusters_file) else args.clusters_file
    confusing_path = os.path.join(base_dir, args.confusing_pairs_file) if not os.path.isabs(args.confusing_pairs_file) else args.confusing_pairs_file
    output_path = os.path.join(base_dir, args.output_file) if not os.path.isabs(args.output_file) else args.output_file

    rng = random.Random(args.seed)

    metadata = load_metadata(metadata_path)
    # clusters/confusing_pairs args are kept for compatibility (currently unused)

    slurp_map = {}
    if os.path.exists(slurp_path):
        slurp_map = {str(d.get("slurp_id")): d for d in read_jsonl(slurp_path)}

    items = read_jsonl(input_path)
    if args.limit:
        items = items[: args.limit]

    if not items:
        print(f"[ERROR] No input items found: {input_path}")
        return

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    torch_dtype = torch.bfloat16 if "cuda" in args.device else torch.float32
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(args.device)
    model.eval()

    results: List[Dict[str, Any]] = []
    sr = processor.feature_extractor.sampling_rate

    for idx, item in enumerate(items):
        slurp_id = str(item.get("slurp_id", ""))
        fallback = slurp_map.get(slurp_id)
        if fallback:
            for key in ["scenario", "action", "entities", "tokens", "recordings", "sentence"]:
                if item.get(key) in (None, "", [], {}):
                    item[key] = fallback.get(key)

        if args.mode == "nbest" and not item.get("asr_hypotheses"):
            continue

        scenario = item.get("scenario", "")
        action = item.get("action", "")
        gold_entities = build_entities(item)
        gold_slot_types = extract_slot_types(gold_entities)

        # n-best texts
        nbest_texts: List[str] = []
        if args.mode == "nbest":
            for h in item.get("asr_hypotheses", [])[: args.num_hypotheses]:
                if isinstance(h, dict):
                    txt = h.get("text", "")
                else:
                    txt = str(h)
                if txt:
                    nbest_texts.append(txt.strip())
        stable_unstable = summarize_nbest(nbest_texts)

        scenario_candidates = select_candidates_topk(
            gold=scenario,
            candidates=metadata["scenarios"],
            k=args.num_candidates,
            rng=rng,
        )
        action_candidates = select_candidates_topk(
            gold=action,
            candidates=metadata["actions"],
            k=args.num_candidates,
            rng=rng,
        )

        slot_candidates = select_slot_candidates_topk(
            gold_types=gold_slot_types,
            slot_types=metadata["slot_types"],
            k=args.num_candidates,
            rng=rng,
        )

        if args.mode == "nbest":
            prompt = build_prompt_nbest(
                gold_scenario=scenario,
                gold_action=action,
                gold_slot_types=gold_slot_types,
                scenario_candidates=scenario_candidates,
                action_candidates=action_candidates,
                slot_candidates=slot_candidates,
                nbest_texts=nbest_texts,
                stable_tokens=stable_unstable["stable"],
                unstable_tokens=stable_unstable["unstable"],
            )
            user_content = [{"type": "text", "text": prompt}]
            text_input = processor.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(text=text_input, return_tensors="pt")
        else:
            filename = pick_recording(item.get("recordings", []), args.recording_index)
            audio_path = resolve_audio_path(audio_root, filename) if filename else None
            audio = load_audio(audio_path, target_sr=sr) if audio_path else None
            if audio is None:
                continue
            prompt = build_prompt_audio(
                gold_scenario=scenario,
                gold_action=action,
                gold_slot_types=gold_slot_types,
                scenario_candidates=scenario_candidates,
                action_candidates=action_candidates,
                slot_candidates=slot_candidates,
            )
            user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": prompt}]
            text_input = processor.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")

        inputs = {k: v.to(args.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": processor.tokenizer.pad_token_id,
        }
        if args.do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            })

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        generated = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        parsed = extract_json(generated)

        result = {
            "slurp_id": slurp_id,
            "mode": args.mode,
            "gold_scenario": scenario,
            "gold_action": action,
            "gold_slot_types": gold_slot_types,
            "scenario_candidates": scenario_candidates,
            "action_candidates": action_candidates,
            "slot_candidates": slot_candidates,
            "nbest": nbest_texts if args.mode == "nbest" else [],
            "nbest_summary": stable_unstable if args.mode == "nbest" else {},
            "rationale": parsed,
        }
        if args.save_raw:
            result["rationale_raw"] = generated
        results.append(result)

        if args.preview and idx < args.preview:
            print("=" * 80)
            print(f"[PREVIEW] {idx+1} / {args.preview} | slurp_id={slurp_id} | mode={args.mode}")
            print("-" * 80)
            print("PROMPT:")
            print(prompt)
            print("-" * 80)
            print("OUTPUT:")
            print(generated)
        if args.limitmode:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("")

        if (idx + 1) % 10 == 0:
            print(f"[INFO] Processed {idx+1}/{len(items)}")

    write_jsonl(output_path, results)
    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    main()
