#!/usr/bin/env python3
import argparse
import json
import random
from typing import Any, Dict, List, Tuple

from common import read_jsonl, write_jsonl


SYSTEM_PROMPT = (
    "You are an SLU model. Listen to audio, describe key acoustic evidence in natural language, "
    "then output the final intent."
)

USER_PROMPT_TEMPLATE = (
    "Step 1: Explain why the utterance belongs to a fine-grained intent and why broader alternatives are insufficient.\n"
    "Step 2: Output JSON with keys `rationale` and `intent`."
)


def build_target(row: Dict[str, Any]) -> str:
    rationale_obj = row.get("rationale", {}) or {}
    rationale = str(rationale_obj.get("final_rationale", "")).strip()
    intent = str(rationale_obj.get("final_intent", "")).strip() or str(row.get("gold_intent", "")).strip()
    if not rationale:
        rationale = "Acoustic detail indicates a specific intent that is more precise than broad alternatives."
    return json.dumps({"rationale": rationale, "intent": intent}, ensure_ascii=False)


def split_train_eval(rows: List[Dict[str, Any]], eval_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_eval = int(len(shuffled) * eval_ratio)
    eval_rows = shuffled[:n_eval]
    train_rows = shuffled[n_eval:]
    return train_rows, eval_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build student distillation dataset from teacher rationale outputs.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="Experiment_AcousticContrastiveDistill/outputs/03_teacher_rationales.jsonl",
    )
    parser.add_argument(
        "--output_train",
        type=str,
        default="Experiment_AcousticContrastiveDistill/outputs/04_distill_train.jsonl",
    )
    parser.add_argument(
        "--output_eval",
        type=str,
        default="Experiment_AcousticContrastiveDistill/outputs/04_distill_eval.jsonl",
    )
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input_file)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No rows found: {args.input_file}")

    dataset_rows: List[Dict[str, Any]] = []
    dropped = 0
    for row in rows:
        audio_path = str(row.get("audio_path", "")).strip()
        if not audio_path:
            dropped += 1
            continue

        slurp_id = str(row.get("slurp_id", "")).strip()
        target_text = build_target(row)
        dataset_rows.append(
            {
                "id": slurp_id,
                "audio_path": audio_path,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": USER_PROMPT_TEMPLATE,
                "target_text": target_text,
                "gold_intent": row.get("gold_intent", ""),
                "parent_intent": row.get("parent_intent", ""),
                "sentence": row.get("sentence", ""),
            }
        )

    train_rows, eval_rows = split_train_eval(dataset_rows, eval_ratio=args.eval_ratio, seed=args.seed)
    write_jsonl(args.output_train, train_rows)
    write_jsonl(args.output_eval, eval_rows)
    print(f"[OK] train={len(train_rows)} eval={len(eval_rows)} dropped_no_audio={dropped}")
    print(f"[OK] wrote train -> {args.output_train}")
    print(f"[OK] wrote eval  -> {args.output_eval}")


if __name__ == "__main__":
    main()
