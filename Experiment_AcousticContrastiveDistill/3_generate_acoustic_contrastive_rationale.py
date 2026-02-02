#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from common import read_jsonl, write_jsonl


def build_prompt(row: Dict[str, Any]) -> str:
    slurp_id = row.get("slurp_id", "")
    sentence = row.get("sentence", "")
    gold = row.get("gold_intent", "")
    parent = row.get("parent_intent", "")
    competitors = row.get("competitor_intents", []) or []
    ipa = row.get("ipa_sequence", "")

    return (
        "Task: Explain why the detailed intent is correct and why broader/similar intents are wrong.\n"
        f"slurp_id: {slurp_id}\n"
        f"transcript: {sentence}\n"
        f"gold_intent: {gold}\n"
        f"parent_intent: {parent}\n"
        f"competitor_intents: {competitors}\n"
        f"ipa_evidence: {ipa}\n\n"
        "Constraints:\n"
        "- NEVER output IPA symbols or phonetic tokens directly.\n"
        "- Describe only acoustic facts in plain natural language.\n"
        "- Mention at least one reason to reject the parent_intent.\n"
        "- Return strict JSON with keys: acoustic_facts, rejection_reasons, final_rationale, final_intent.\n"
    )


def sanitize_ipa_like_tokens(text: str) -> str:
    # Remove typical uppercase phone sequences from fallback G2P.
    text = re.sub(r"\b(?:AA|AE|AH|AO|AW|AY|B|CH|D|DH|EH|ER|EY|F|G|HH|IH|IY|JH|K|L|M|N|NG|OW|OY|P|R|S|SH|T|TH|UH|UW|V|W|Y|Z|ZH)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def template_rationale(row: Dict[str, Any]) -> Dict[str, Any]:
    terms = row.get("focus_terms", []) or []
    gold = row.get("gold_intent", "")
    parent = row.get("parent_intent", "")
    competitors = row.get("competitor_intents", []) or []

    facts = []
    if terms:
        facts.append(f"The speech includes detail cues around: {', '.join(terms[:3])}.")
    else:
        facts.append("The utterance contains specific command-level lexical detail.")

    reject = []
    if parent:
        reject.append(
            {
                "intent": parent,
                "reason": "The utterance carries extra detail that is not explained by the broader parent intent.",
            }
        )
    for cand in competitors:
        if cand == parent:
            continue
        reject.append(
            {
                "intent": cand,
                "reason": "Domain/action focus mismatches the strongest acoustic-semantic cues in the utterance.",
            }
        )
        if len(reject) >= 4:
            break

    return {
        "acoustic_facts": facts,
        "rejection_reasons": reject,
        "final_rationale": "Specific spoken detail supports a fine-grained intent and rejects broader alternatives.",
        "final_intent": gold,
    }


def call_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout_sec: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": "You generate concise JSON rationale for contrastive intent reasoning from audio evidence.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        text = resp.read().decode("utf-8")
    obj = json.loads(text)
    return obj["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate acoustic-contrastive rationale from IPA-enriched rows.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="Experiment_AcousticContrastiveDistill/outputs/02_teacher_acoustic_view.jsonl",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="Experiment_AcousticContrastiveDistill/outputs/03_teacher_rationales.jsonl",
    )
    parser.add_argument("--backend", type=str, default="template", choices=["template", "openai_compatible"])
    parser.add_argument("--model", type=str, default="deepseek-r1")
    parser.add_argument("--api_base", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout_sec", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sleep_sec", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input_file)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No rows found: {args.input_file}")

    api_key = os.environ.get(args.api_key_env, "")
    if args.backend == "openai_compatible" and not api_key:
        raise SystemExit(f"{args.api_key_env} is empty. Set API key or use --backend template")

    outputs: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        prompt = build_prompt(row)
        parsed: Optional[Dict[str, Any]] = None
        raw_text = ""
        error_text = ""

        if args.backend == "template":
            parsed = template_rationale(row)
            raw_text = json.dumps(parsed, ensure_ascii=False)
        else:
            for attempt in range(1, args.max_retries + 1):
                try:
                    raw_text = call_openai_compatible(
                        base_url=args.api_base,
                        api_key=api_key,
                        model=args.model,
                        prompt=prompt,
                        temperature=args.temperature,
                        timeout_sec=args.timeout_sec,
                    )
                    parsed = extract_json_obj(raw_text)
                    if parsed:
                        break
                    error_text = "invalid_json"
                except urllib.error.HTTPError as e:
                    error_text = f"http_error_{e.code}"
                except Exception as e:
                    error_text = str(e)
                time.sleep(args.sleep_sec)

        if not parsed:
            parsed = template_rationale(row)
            parsed["fallback_reason"] = error_text or "parse_failed"

        final_rationale = sanitize_ipa_like_tokens(str(parsed.get("final_rationale", "")).strip())
        parsed["final_rationale"] = final_rationale
        parsed["final_intent"] = parsed.get("final_intent") or row.get("gold_intent", "")

        outputs.append(
            {
                **row,
                "teacher_backend": args.backend,
                "rationale": parsed,
                "rationale_raw": raw_text,
            }
        )
        if idx % 50 == 0:
            print(f"[INFO] processed {idx}/{len(rows)}")

    write_jsonl(args.output_file, outputs)
    print(f"[OK] wrote {len(outputs)} rows -> {args.output_file}")


if __name__ == "__main__":
    main()

