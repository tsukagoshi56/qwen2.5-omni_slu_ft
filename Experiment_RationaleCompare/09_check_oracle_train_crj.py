#!/usr/bin/env python3
import argparse
import json
import re
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common import normalize_intent_label, parse_j_from_output, read_jsonl


def _pick_text(row: Dict[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_prefixed_line(text: str, prefix: str) -> str:
    target = prefix.upper()
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(rf"^{re.escape(target)}\s*[:：]", line.upper()):
            body = line.split(":", 1)[1].strip() if ":" in line else ""
            if not body and "：" in line:
                body = line.split("：", 1)[1].strip()
            return body
    return ""


def _extract_intent_candidates(c_body: str) -> List[str]:
    text = str(c_body or "").strip()
    if not text:
        return []
    intent_part = text.split(";", 1)[0].strip()
    lowered = intent_part.lower()
    for prefix in ("intent candidates:", "intent candidate:", "intent:", "intents:"):
        if lowered.startswith(prefix):
            intent_part = intent_part[len(prefix):].strip()
            break
    values = [normalize_intent_label(x.strip()) for x in intent_part.split("|") if x.strip()]
    dedup: List[str] = []
    seen = set()
    for value in values:
        if not value or value in {"(none)", "none"} or value in seen:
            continue
        seen.add(value)
        dedup.append(value)
    return dedup


def _extract_slot_candidates(c_body: str) -> List[str]:
    text = str(c_body or "").strip()
    if not text:
        return []
    slot_part = text
    if ";" in text:
        maybe_slot = text.split(";", 1)[1].strip()
        if maybe_slot:
            slot_part = maybe_slot
    lowered = slot_part.lower()
    for prefix in ("slot candidates:", "slot candidate:", "slots:", "slot:"):
        if lowered.startswith(prefix):
            slot_part = slot_part[len(prefix):].strip()
            break
    if not slot_part:
        return []
    # artist(adele|sia) -> artist
    slot_part = re.sub(r"\([^)]*\)", "", slot_part)
    values = [x.strip().lower() for x in slot_part.split("|") if x.strip()]
    dedup: List[str] = []
    seen = set()
    for value in values:
        if not value or value in {"(none)", "none"} or value in seen:
            continue
        seen.add(value)
        dedup.append(value)
    return dedup


def _as_ratio_str(n: int, total: int) -> str:
    if total <= 0:
        return "0/0 (0.00%)"
    return f"{n}/{total} ({(100.0 * n / total):.2f}%)"


def _safe_mean(values: Sequence[int]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_median(values: Sequence[int]) -> float:
    return float(median(values)) if values else 0.0


def _summarize_counts(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0,
            "max": 0,
        }
    return {
        "n": len(values),
        "mean": _safe_mean(values),
        "median": _safe_median(values),
        "min": int(min(values)),
        "max": int(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check Oracle-train outputs for CRJ presence and candidate statistics "
            "(average number of intent/slot candidates)."
        )
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL (e.g., oracle_cot_train.jsonl).")
    parser.add_argument(
        "--text_fields",
        type=str,
        default="rationale_text,target,raw_output,output,response,assistant_text",
        help="Comma-separated fields to search CRJ text from, in priority order.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=20,
        help="How many failure sample IDs to print for each check.",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default="",
        help="Optional path to save full report JSON.",
    )
    args = parser.parse_args()

    fields = [x.strip() for x in str(args.text_fields).split(",") if x.strip()]
    rows = read_jsonl(args.input_file)
    total = len(rows)

    text_present = 0
    has_c = 0
    has_r = 0
    has_j_line = 0
    has_crj = 0
    valid_j = 0
    valid_crj = 0

    intent_counts_all: List[int] = []
    slot_counts_all: List[int] = []
    intent_counts_crj: List[int] = []
    slot_counts_crj: List[int] = []

    missing_crj_ids: List[str] = []
    invalid_j_ids: List[str] = []

    for i, row in enumerate(rows):
        rid = str(row.get("slurp_id") or row.get("id") or f"row_{i}")
        text = _pick_text(row, fields)
        if text:
            text_present += 1
        c_body = _extract_prefixed_line(text, "C")
        r_body = _extract_prefixed_line(text, "R")
        j_body = _extract_prefixed_line(text, "J")

        c_ok = bool(c_body)
        r_ok = bool(r_body)
        j_line_ok = bool(j_body)

        if c_ok:
            has_c += 1
        if r_ok:
            has_r += 1
        if j_line_ok:
            has_j_line += 1

        if c_ok and r_ok and j_line_ok:
            has_crj += 1
        else:
            if len(missing_crj_ids) < max(0, int(args.sample_limit)):
                missing_crj_ids.append(rid)

        parsed_j = parse_j_from_output(text)
        if isinstance(parsed_j, dict):
            scenario = str(parsed_j.get("scenario", "")).strip()
            action = str(parsed_j.get("action", "")).strip()
            if (scenario or action) and "entities" in parsed_j:
                valid_j += 1
            else:
                if len(invalid_j_ids) < max(0, int(args.sample_limit)):
                    invalid_j_ids.append(rid)
        else:
            if len(invalid_j_ids) < max(0, int(args.sample_limit)):
                invalid_j_ids.append(rid)

        intent_candidates = _extract_intent_candidates(c_body)
        slot_candidates = _extract_slot_candidates(c_body)
        intent_counts_all.append(len(intent_candidates))
        slot_counts_all.append(len(slot_candidates))
        if c_ok and r_ok and j_line_ok:
            intent_counts_crj.append(len(intent_candidates))
            slot_counts_crj.append(len(slot_candidates))

    valid_crj = min(has_crj, valid_j)

    report = {
        "input_file": args.input_file,
        "total_rows": total,
        "text_rows": text_present,
        "checks": {
            "has_c": {"count": has_c, "ratio": _as_ratio_str(has_c, total)},
            "has_r": {"count": has_r, "ratio": _as_ratio_str(has_r, total)},
            "has_j_line": {"count": has_j_line, "ratio": _as_ratio_str(has_j_line, total)},
            "has_crj_lines": {"count": has_crj, "ratio": _as_ratio_str(has_crj, total)},
            "j_json_valid": {"count": valid_j, "ratio": _as_ratio_str(valid_j, total)},
            "crj_and_j_valid": {"count": valid_crj, "ratio": _as_ratio_str(valid_crj, total)},
        },
        "candidate_stats": {
            "intent_per_row_all": _summarize_counts(intent_counts_all),
            "slot_per_row_all": _summarize_counts(slot_counts_all),
            "intent_per_row_crj_only": _summarize_counts(intent_counts_crj),
            "slot_per_row_crj_only": _summarize_counts(slot_counts_crj),
        },
        "samples": {
            "missing_crj_ids": missing_crj_ids,
            "invalid_j_ids": invalid_j_ids,
        },
    }

    print("=== Oracle Train CRJ Check ===")
    print(f"input_file: {args.input_file}")
    print(f"total_rows: {total}")
    print(f"text_rows:  {text_present}")
    print()
    print("[CRJ / J checks]")
    print(f"- has_c:           {report['checks']['has_c']['ratio']}")
    print(f"- has_r:           {report['checks']['has_r']['ratio']}")
    print(f"- has_j_line:      {report['checks']['has_j_line']['ratio']}")
    print(f"- has_crj_lines:   {report['checks']['has_crj_lines']['ratio']}")
    print(f"- j_json_valid:    {report['checks']['j_json_valid']['ratio']}")
    print(f"- crj_and_j_valid: {report['checks']['crj_and_j_valid']['ratio']}")
    print()
    print("[Candidate count stats]")
    i_all = report["candidate_stats"]["intent_per_row_all"]
    s_all = report["candidate_stats"]["slot_per_row_all"]
    i_crj = report["candidate_stats"]["intent_per_row_crj_only"]
    s_crj = report["candidate_stats"]["slot_per_row_crj_only"]
    print(
        f"- intent/all: mean={i_all['mean']:.3f}, median={i_all['median']:.3f}, "
        f"min={i_all['min']}, max={i_all['max']}"
    )
    print(
        f"- slot/all:   mean={s_all['mean']:.3f}, median={s_all['median']:.3f}, "
        f"min={s_all['min']}, max={s_all['max']}"
    )
    print(
        f"- intent/crj: mean={i_crj['mean']:.3f}, median={i_crj['median']:.3f}, "
        f"min={i_crj['min']}, max={i_crj['max']}"
    )
    print(
        f"- slot/crj:   mean={s_crj['mean']:.3f}, median={s_crj['median']:.3f}, "
        f"min={s_crj['min']}, max={s_crj['max']}"
    )

    if missing_crj_ids:
        print()
        print(f"[sample missing_crj_ids] {missing_crj_ids}")
    if invalid_j_ids:
        print()
        print(f"[sample invalid_j_ids] {invalid_j_ids}")

    if args.report_json.strip():
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print()
        print(f"[saved] report_json: {args.report_json}")


if __name__ == "__main__":
    main()
