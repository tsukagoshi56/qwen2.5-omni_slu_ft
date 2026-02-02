#!/usr/bin/env python3
import argparse
import json
import math
import re
from statistics import mean
from typing import Any, Dict, List, Optional

from common import read_jsonl, tokenize_simple, write_json


def resolve_pred_intent(row: Dict[str, Any]) -> str:
    for key in ["pred_intent", "intent", "final_intent", "prediction_intent"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    text = str(row.get("prediction_text", ""))
    m = re.search(r'"intent"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1).strip()
    return ""


def resolve_score_map(row: Dict[str, Any]) -> Dict[str, float]:
    for key in ["intent_logprobs", "intent_scores", "score_map"]:
        value = row.get(key)
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue
            return out
    return {}


def resolve_rationale_text(row: Dict[str, Any]) -> str:
    for key in ["rationale", "final_rationale", "prediction_text"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def calc_overlap_ratio(a: str, b: str) -> float:
    ta = set(tokenize_simple(a))
    tb = set(tokenize_simple(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hierarchical recovery and boundary contrast metrics.")
    parser.add_argument(
        "--reference_file",
        type=str,
        default="Experiment_AcousticContrastiveDistill/outputs/01_contrastive_pairs.jsonl",
    )
    parser.add_argument(
        "--student_pred_file",
        type=str,
        default="Experiment_AcousticContrastiveDistill/outputs/05_student_predictions.jsonl",
    )
    parser.add_argument("--baseline_pred_file", type=str, default="")
    parser.add_argument(
        "--output_report",
        type=str,
        default="Experiment_AcousticContrastiveDistill/outputs/06_eval_report.json",
    )
    args = parser.parse_args()

    reference_rows = read_jsonl(args.reference_file)
    student_rows = read_jsonl(args.student_pred_file)
    baseline_rows = read_jsonl(args.baseline_pred_file) if args.baseline_pred_file else []

    ref_map = {str(r.get("slurp_id", "")).strip(): r for r in reference_rows}
    st_map = {str(r.get("id", r.get("slurp_id", ""))).strip(): r for r in student_rows}
    bl_map = {str(r.get("id", r.get("slurp_id", ""))).strip(): r for r in baseline_rows}

    matched_ids = [sid for sid in ref_map.keys() if sid in st_map]
    if not matched_ids:
        raise SystemExit("No overlapping slurp_id between reference and student predictions")

    n_total = len(matched_ids)
    n_correct = 0

    rr_den = 0
    rr_num = 0

    bcs_values: List[float] = []
    faithfulness_values: List[float] = []

    for sid in matched_ids:
        ref = ref_map[sid]
        st = st_map[sid]
        bl = bl_map.get(sid, {})

        gold = str(ref.get("gold_intent", "")).strip()
        parent = str(ref.get("parent_intent", "")).strip()
        st_pred = resolve_pred_intent(st)
        bl_pred = resolve_pred_intent(bl) if bl else ""

        if st_pred == gold:
            n_correct += 1

        if parent and bl:
            if bl_pred == parent and parent != gold:
                rr_den += 1
                if st_pred == gold:
                    rr_num += 1

        score_map = resolve_score_map(st)
        if score_map and gold in score_map and parent and parent in score_map:
            g = score_map[gold]
            p = score_map[parent]
            # If scores are probabilities, convert to log space.
            if 0.0 <= g <= 1.0 and 0.0 <= p <= 1.0:
                g = math.log(max(g, 1e-12))
                p = math.log(max(p, 1e-12))
            bcs_values.append(g - p)

        rationale_text = resolve_rationale_text(st)
        sentence = str(ref.get("sentence", "")).strip()
        faithfulness_values.append(calc_overlap_ratio(rationale_text, sentence))

    report: Dict[str, Any] = {
        "num_samples": n_total,
        "student_accuracy": n_correct / max(1, n_total),
        "hierarchical_recovery_rate": (rr_num / rr_den) if rr_den > 0 else None,
        "hierarchical_recovery_support": rr_den,
        "boundary_contrast_score_mean": mean(bcs_values) if bcs_values else None,
        "boundary_contrast_score_count": len(bcs_values),
        "rationale_faithfulness_overlap_mean": mean(faithfulness_values) if faithfulness_values else None,
    }

    write_json(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] report saved -> {args.output_report}")


if __name__ == "__main__":
    main()

