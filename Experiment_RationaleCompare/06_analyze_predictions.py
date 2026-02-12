#!/usr/bin/env python3
"""
06_analyze_predictions.py
=========================
prediction.jsonl を分析するスクリプト。
evaluate.py の基準 (FMeasure / SpanFMeasure) に従って
  - Scenario / Action / Intent (scenario_action) の Accuracy & F1
  - Entity Span-F1 (exact match)
を算出し、さらに意味的類似・上位包含関係に着目したエラー分析を行う。

複数ファイルを指定すると、横並びで比較表示する。

Usage:
    # 単一モデル
    python 06_analyze_predictions.py pred.jsonl --gold test.jsonl

    # 複数モデル比較
    python 06_analyze_predictions.py modelA/prediction.jsonl modelB/prediction.jsonl --gold test.jsonl

    # フォルダ指定 (prediction.jsonl を自動探索)
    python 06_analyze_predictions.py output/modelA/ output/modelB/ --gold test.jsonl

    # モデル名を明示 (カンマ区切り、ファイルと同じ順)
    python 06_analyze_predictions.py a/pred.jsonl b/pred.jsonl --names "baseline,finetuned"
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# 1. Data loading  (evaluate.py 準拠)
# ============================================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _normalize_entity(entity: Dict[str, Any]) -> Tuple[str, str]:
    ent_type = str(entity.get("type", "")).strip().lower()
    filler = entity.get("filler")
    if filler is None:
        filler = entity.get("filter", "")
    if filler is None:
        filler = entity.get("value", "")
    filler = re.sub(r"\s+", " ", str(filler).strip().lower())
    return ent_type, filler


def _parse_entities_from_list(raw_entities: Any) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    if not isinstance(raw_entities, list):
        return results
    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue
        ent_type = str(ent.get("type", "")).strip()
        filler = ent.get("filler")
        if filler is None:
            filler = ent.get("filter", "")
        if filler is None:
            filler = ent.get("value", "")
        filler = str(filler or "")
        results.append({"type": ent_type, "filler": filler})
    return results


# ---------- Gold loading ----------

def load_gold_from_test_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    gold_map: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            tokens = example.get("tokens", [])
            entities = []
            for ent in example.get("entities", []):
                ent_type = ent.get("type", "unknown")
                span = ent.get("span", [])
                filler = " ".join(
                    tokens[i]["surface"].lower() for i in span if i < len(tokens)
                )
                entities.append({"type": ent_type, "filler": filler})
            base = {
                "scenario": example["scenario"],
                "action": example["action"],
                "entities": entities,
            }
            for rec in example.get("recordings", []):
                fname = rec.get("file", "")
                if fname:
                    res = dict(base)
                    res["wer"] = rec.get("wer")          # recording 固有
                    res["ent_wer"] = rec.get("ent_wer")
                    gold_map[fname] = res
    return gold_map


def load_gold_by_slurp_id(path: str) -> Dict[str, Dict[str, Any]]:
    gold_map: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            tokens = example.get("tokens", [])
            entities = []
            for ent in example.get("entities", []):
                ent_type = ent.get("type", "unknown")
                span = ent.get("span", [])
                filler = " ".join(
                    tokens[i]["surface"].lower() for i in span if i < len(tokens)
                )
                entities.append({"type": ent_type, "filler": filler})
            res = {
                "scenario": example["scenario"],
                "action": example["action"],
                "entities": entities,
            }
            gold_map[str(example["slurp_id"])] = res
    return gold_map


# ---------- Prediction / Gold extraction ----------

def _extract_gold_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    target_label = row.get("target_label")
    if isinstance(target_label, dict) and (
        target_label.get("scenario") or target_label.get("action") or target_label.get("entities")
    ):
        return {
            "scenario": str(target_label.get("scenario", "")).strip(),
            "action": str(target_label.get("action", "")).strip(),
            "entities": _parse_entities_from_list(target_label.get("entities", [])),
        }
    target_str = row.get("target", "")
    if isinstance(target_str, str) and target_str.strip():
        obj = _parse_j_line(target_str)
        if obj:
            return obj
    return None


def _parse_j_line(text: str) -> Optional[Dict[str, Any]]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("J:"):
            json_str = line[2:].strip()
            try:
                obj = json.loads(json_str)
                if isinstance(obj, dict):
                    s = str(obj.get("scenario", obj.get("Scenario", ""))).strip()
                    a = str(obj.get("action", obj.get("Action", ""))).strip()
                    if not s and not a:
                        intent = str(obj.get("intent", obj.get("Intent", ""))).strip()
                        if "_" in intent:
                            s, a = intent.split("_", 1)
                    ents = obj.get("entities", obj.get("Entities", obj.get("slots", [])))
                    return {"scenario": s, "action": a,
                            "entities": _parse_entities_from_list(ents or [])}
            except json.JSONDecodeError:
                pass
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            s = str(obj.get("scenario", "")).strip()
            a = str(obj.get("action", "")).strip()
            ents = obj.get("entities", [])
            if s or a:
                return {"scenario": s, "action": a,
                        "entities": _parse_entities_from_list(ents or [])}
    except json.JSONDecodeError:
        pass
    return None


def _extract_pred_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scenario": str(row.get("scenario", "")).strip(),
        "action": str(row.get("action", "")).strip(),
        "entities": _parse_entities_from_list(row.get("entities", [])),
    }


# ============================================================================
# 2. Metrics
# ============================================================================

def compute_prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class FMeasureAccumulator:
    def __init__(self):
        self.tp: Dict[str, float] = defaultdict(float)
        self.fp: Dict[str, float] = defaultdict(float)
        self.fn: Dict[str, float] = defaultdict(float)

    def add(self, gold: str, pred: str):
        if pred == gold:
            self.tp[pred] += 1
        else:
            self.fp[pred] += 1
            self.fn[gold] += 1

    def overall(self, average: str = "micro") -> Tuple[float, float, float]:
        if average == "macro":
            all_tags = set(self.tp) | set(self.fp) | set(self.fn)
            if not all_tags:
                return 0.0, 0.0, 0.0
            ps, rs, fs = [], [], []
            for t in all_tags:
                p, r, f = compute_prf(self.tp[t], self.fp[t], self.fn[t])
                ps.append(p); rs.append(r); fs.append(f)
            n = len(all_tags)
            return sum(ps)/n, sum(rs)/n, sum(fs)/n
        return compute_prf(sum(self.tp.values()), sum(self.fp.values()), sum(self.fn.values()))

    @property
    def accuracy(self) -> float:
        total_tp = sum(self.tp.values())
        total = total_tp + sum(self.fp.values())
        return total_tp / total if total > 0 else 0.0

    def per_label_f1(self) -> Dict[str, float]:
        all_tags = set(self.tp) | set(self.fp) | set(self.fn)
        result = {}
        for tag in all_tags:
            _, _, f = compute_prf(self.tp[tag], self.fp[tag], self.fn[tag])
            result[tag] = f
        return result


class SpanFMeasureAccumulator:
    def __init__(self):
        self.tp: Dict[str, float] = defaultdict(float)
        self.fp: Dict[str, float] = defaultdict(float)
        self.fn: Dict[str, float] = defaultdict(float)

    def add(self, gold_entities: List[Dict], pred_entities: List[Dict]):
        gold_normalized = [_normalize_entity(e) for e in (gold_entities or [])]
        pred_normalized = [_normalize_entity(e) for e in (pred_entities or [])]
        gold_copy = list(gold_normalized)
        for etype, filler in pred_normalized:
            if (etype, filler) in gold_copy:
                self.tp[etype] += 1
                gold_copy.remove((etype, filler))
            else:
                self.fp[etype] += 1
        for etype, _ in gold_copy:
            self.fn[etype] += 1

    def overall(self, average: str = "micro") -> Tuple[float, float, float]:
        if average == "macro":
            all_tags = set(self.tp) | set(self.fp) | set(self.fn)
            if not all_tags:
                return 0.0, 0.0, 0.0
            ps, rs, fs = [], [], []
            for t in all_tags:
                p, r, f = compute_prf(self.tp[t], self.fp[t], self.fn[t])
                ps.append(p); rs.append(r); fs.append(f)
            n = len(all_tags)
            return sum(ps)/n, sum(rs)/n, sum(fs)/n
        return compute_prf(sum(self.tp.values()), sum(self.fp.values()), sum(self.fn.values()))

    def per_label_f1(self) -> Dict[str, float]:
        all_tags = set(self.tp) | set(self.fp) | set(self.fn)
        result = {}
        for tag in all_tags:
            _, _, f = compute_prf(self.tp[tag], self.fp[tag], self.fn[tag])
            result[tag] = f
        return result


# ============================================================================
# 3. Error Analysis helpers
# ============================================================================

def top_confusions(pairs: List[Tuple[str, str]], top_n: int = 20) -> List[Tuple[Tuple[str, str], int]]:
    counts = Counter(p for p in pairs if p[0] != p[1])
    return counts.most_common(top_n)


def error_type_breakdown(intent_pairs: List[Tuple[str, str]]) -> Dict[str, int]:
    same_s_diff_a = 0; diff_s_same_a = 0; diff_both = 0
    for gold_intent, pred_intent in intent_pairs:
        if gold_intent == pred_intent:
            continue
        gp = gold_intent.split("_", 1); pp = pred_intent.split("_", 1)
        gs = gp[0]; ga = gp[1] if len(gp) > 1 else ""
        ps = pp[0]; pa = pp[1] if len(pp) > 1 else ""
        if gs == ps and ga != pa:
            same_s_diff_a += 1
        elif gs != ps and ga == pa:
            diff_s_same_a += 1
        else:
            diff_both += 1
    return {"same_scenario_diff_action": same_s_diff_a,
            "diff_scenario_same_action": diff_s_same_a,
            "diff_both": diff_both}


def entity_type_confusion(
    matched: List[Tuple[Dict, Dict]],
) -> Tuple[Counter, Counter, Counter]:
    type_fp = Counter(); type_fn = Counter(); type_swap = Counter()
    for pred, gold in matched:
        pred_set = set(_normalize_entity(e) for e in (pred["entities"] or []))
        gold_set = set(_normalize_entity(e) for e in (gold["entities"] or []))
        fp_ents = pred_set - gold_set
        fn_ents = gold_set - pred_set
        for etype, _ in fp_ents:
            type_fp[etype] += 1
        for etype, _ in fn_ents:
            type_fn[etype] += 1
        fp_by_f: Dict[str, List[str]] = defaultdict(list)
        fn_by_f: Dict[str, List[str]] = defaultdict(list)
        for t, f in fp_ents:
            if f: fp_by_f[f].append(t)
        for t, f in fn_ents:
            if f: fn_by_f[f].append(t)
        for filler in set(fp_by_f) & set(fn_by_f):
            for gt in fn_by_f[filler]:
                for pt in fp_by_f[filler]:
                    if gt != pt:
                        type_swap[(gt, pt)] += 1
    return type_fp, type_fn, type_swap


def find_confusion_clusters(
    intent_pairs: List[Tuple[str, str]], min_count: int = 2,
) -> List[Dict[str, Any]]:
    pair_count: Counter = Counter()
    for g, p in intent_pairs:
        if g != p:
            pair_count[tuple(sorted([g, p]))] += 1
    clusters = []
    for (a, b), cnt in pair_count.most_common():
        if cnt < min_count:
            break
        a2b = sum(1 for g, p in intent_pairs if g == a and p == b)
        b2a = sum(1 for g, p in intent_pairs if g == b and p == a)
        ap = a.split("_", 1); bp = b.split("_", 1)
        clusters.append({
            "label_a": a, "label_b": b, "total": cnt,
            "a_to_b": a2b, "b_to_a": b2a,
            "bidirectional": a2b > 0 and b2a > 0,
            "same_scenario": (ap[0] == bp[0]) if len(ap) > 1 and len(bp) > 1 else False,
        })
    return clusters


# ============================================================================
# 4. Per-model result container
# ============================================================================

@dataclass
class ModelResult:
    name: str
    path: str
    n_matched: int = 0
    scenario_acc: float = 0.0
    action_acc: float = 0.0
    intent_acc: float = 0.0
    scenario_f1: float = 0.0
    action_f1: float = 0.0
    intent_f1: float = 0.0
    entity_p: float = 0.0
    entity_r: float = 0.0
    entity_f1: float = 0.0
    # detailed accumulators
    scenario_metric: FMeasureAccumulator = field(default_factory=FMeasureAccumulator)
    action_metric: FMeasureAccumulator = field(default_factory=FMeasureAccumulator)
    intent_metric: FMeasureAccumulator = field(default_factory=FMeasureAccumulator)
    entity_metric: SpanFMeasureAccumulator = field(default_factory=SpanFMeasureAccumulator)
    intent_pairs: List[Tuple[str, str]] = field(default_factory=list)
    wer_list: List[Optional[float]] = field(default_factory=list)  # per-sample WER
    matched: List[Tuple[Dict, Dict]] = field(default_factory=list)
    error_breakdown: Dict[str, int] = field(default_factory=dict)


# ============================================================================
# 5. Gold matching
# ============================================================================

def _auto_detect_gold_path(prediction_path: str) -> Optional[str]:
    d = os.path.dirname(os.path.abspath(prediction_path))
    for _ in range(10):
        candidate = os.path.join(d, "slurp", "dataset", "slurp", "test.jsonl")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def match_predictions_with_gold(
    pred_rows: List[Dict[str, Any]],
    gold_map: Dict[str, Dict[str, Any]],
    key_field: str = "file",
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    matched = []; not_found = 0
    for row in pred_rows:
        key = row.get(key_field)
        if key is None:
            key = row.get("slurp_id") or row.get("id")
        key = str(key).strip() if key else ""
        if not key:
            not_found += 1; continue
        gold = gold_map.get(key)
        if gold is None:
            gold = gold_map.get(os.path.basename(key))
        if gold is None:
            not_found += 1; continue
        matched.append((_extract_pred_from_row(row), gold))
    if not_found > 0:
        print(f"  [WARN] {not_found}/{len(pred_rows)} predictions not matched", file=sys.stderr)
    return matched


def match_predictions_embedded_gold(
    pred_rows: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    matched = []; no_gold = 0
    for row in pred_rows:
        pred = _extract_pred_from_row(row)
        gold = _extract_gold_from_row(row)
        if gold is None or (not gold["scenario"] and not gold["action"]):
            no_gold += 1; continue
        matched.append((pred, gold))
    if no_gold > 0:
        print(f"  [WARN] {no_gold}/{len(pred_rows)} rows had no embedded gold label", file=sys.stderr)
    return matched


# ============================================================================
# 6. Process one model
# ============================================================================

def resolve_prediction_path(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "prediction.jsonl")
        if os.path.exists(candidate):
            return candidate
        print(f"ERROR: {path} is a directory but prediction.jsonl not found inside.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.", file=sys.stderr)
        sys.exit(1)
    return path


def derive_model_name(path: str) -> str:
    """パスから短いモデル名を推定。"""
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)
    name = os.path.basename(parent)
    if name in (".", ""):
        name = os.path.basename(abs_path).replace(".jsonl", "")
    return name


def process_model(
    prediction_path: str,
    model_name: str,
    gold_map: Optional[Dict[str, Dict[str, Any]]],
    key_field: str,
) -> ModelResult:
    pred_rows = read_jsonl(prediction_path)
    if not pred_rows:
        print(f"  [WARN] {model_name}: no data loaded", file=sys.stderr)
        return ModelResult(name=model_name, path=prediction_path)

    if gold_map is not None:
        matched = match_predictions_with_gold(pred_rows, gold_map, key_field=key_field)
    else:
        matched = match_predictions_embedded_gold(pred_rows)

    r = ModelResult(name=model_name, path=prediction_path, n_matched=len(matched), matched=matched)

    for pred, gold in matched:
        ps, pa = pred["scenario"], pred["action"]
        gs, ga = gold["scenario"], gold["action"]
        r.scenario_metric.add(gs, ps)
        r.action_metric.add(ga, pa)
        r.intent_metric.add(f"{gs}_{ga}", f"{ps}_{pa}")
        r.entity_metric.add(gold["entities"], pred["entities"])
        r.intent_pairs.append((f"{gs}_{ga}", f"{ps}_{pa}"))
        r.wer_list.append(gold.get("wer"))

    sp, sr, sf = r.scenario_metric.overall()
    ap, ar, af = r.action_metric.overall()
    ip, ir, if1 = r.intent_metric.overall()
    ep, er, ef = r.entity_metric.overall()
    r.scenario_acc = r.scenario_metric.accuracy
    r.action_acc = r.action_metric.accuracy
    r.intent_acc = r.intent_metric.accuracy
    r.scenario_f1 = sf; r.action_f1 = af; r.intent_f1 = if1
    r.entity_p = ep; r.entity_r = er; r.entity_f1 = ef
    r.error_breakdown = error_type_breakdown(r.intent_pairs)
    return r


# ============================================================================
# 7. Formatting helpers
# ============================================================================

def print_section(title: str, width: int = 80):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_table(headers: List[str], rows: List[List[Any]], max_rows: int = 60):
    col_widths = [len(str(h)) for h in headers]
    display_rows = rows[:max_rows]
    for row in display_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*[str(h) for h in headers]))
    print("-" * sum(col_widths + [2 * (len(col_widths) - 1)]))
    for row in display_rows:
        cells = [str(c) for c in row]
        while len(cells) < len(col_widths):
            cells.append("")
        print(fmt.format(*cells))
    if len(rows) > max_rows:
        print(f"  ... ({len(rows) - max_rows} more rows)")


def _f4(v: float) -> str:
    return f"{v:.4f}"


def _pct(n: int, total: int) -> str:
    return f"{100*n/total:.1f}%" if total > 0 else "-"


# ============================================================================
# 8. Comparison output
# ============================================================================

def print_overall_comparison(models: List[ModelResult]):
    print_section("A. Overall Metrics 比較")
    names = [m.name for m in models]
    headers = ["Metric"] + names
    rows = [
        ["Matched samples"] + [str(m.n_matched) for m in models],
        ["Scenario Acc"]    + [_f4(m.scenario_acc) for m in models],
        ["Scenario F1"]     + [_f4(m.scenario_f1) for m in models],
        ["Action Acc"]      + [_f4(m.action_acc) for m in models],
        ["Action F1"]       + [_f4(m.action_f1) for m in models],
        ["Intent Acc"]      + [_f4(m.intent_acc) for m in models],
        ["Intent F1"]       + [_f4(m.intent_f1) for m in models],
        ["Entity Prec"]     + [_f4(m.entity_p) for m in models],
        ["Entity Recall"]   + [_f4(m.entity_r) for m in models],
        ["Entity F1"]       + [_f4(m.entity_f1) for m in models],
    ]
    print_table(headers, rows)


def print_per_label_comparison(models: List[ModelResult], label_type: str, top_n: int):
    """Per-label F1 を横並び比較。label_type: 'scenario'|'action'|'intent'|'entity'"""
    title_map = {
        "scenario": "Scenario", "action": "Action",
        "intent": "Intent (scenario_action)", "entity": "Entity Type",
    }
    print(f"\n  --- {title_map[label_type]}: Per-label F1 比較 (worst-avg first, top {top_n}) ---")

    # Collect all labels
    all_labels: Set[str] = set()
    f1_maps: List[Dict[str, float]] = []
    for m in models:
        acc = getattr(m, f"{label_type}_metric")
        f1map = acc.per_label_f1()
        f1_maps.append(f1map)
        all_labels.update(f1map.keys())

    if not all_labels:
        print("  (no labels)")
        return

    # Sort by average F1 ascending (worst first)
    label_avg = {}
    for label in all_labels:
        vals = [fm.get(label, 0.0) for fm in f1_maps]
        label_avg[label] = sum(vals) / len(vals)
    sorted_labels = sorted(all_labels, key=lambda x: label_avg[x])

    names = [m.name for m in models]
    headers = ["Label"] + names + (["diff"] if len(models) == 2 else [])
    rows = []
    for label in sorted_labels[:top_n]:
        vals = [fm.get(label, 0.0) for fm in f1_maps]
        row: List[Any] = [label] + [_f4(v) for v in vals]
        if len(models) == 2:
            diff = vals[1] - vals[0]
            row.append(f"{diff:+.4f}")
        rows.append(row)
    print_table(headers, rows)


def print_error_breakdown_comparison(models: List[ModelResult]):
    print_section("C. Intent エラー分類 比較")
    names = [m.name for m in models]
    types = ["same_scenario_diff_action", "diff_scenario_same_action", "diff_both"]
    desc = {
        "same_scenario_diff_action": "同Scenario/異Action",
        "diff_scenario_same_action": "異Scenario/同Action",
        "diff_both":                 "両方異なる",
    }
    headers = ["Error Type"] + [f"{n} (n)" for n in names] + [f"{n} (%)" for n in names]
    rows = []
    totals = [sum(m.error_breakdown.values()) for m in models]
    for t in types:
        row: List[Any] = [desc[t]]
        for m in models:
            row.append(str(m.error_breakdown.get(t, 0)))
        for i, m in enumerate(models):
            row.append(_pct(m.error_breakdown.get(t, 0), m.n_matched))
        rows.append(row)
    # total row
    row_total: List[Any] = ["TOTAL errors"]
    for tot in totals:
        row_total.append(str(tot))
    for i, m in enumerate(models):
        row_total.append(_pct(totals[i], m.n_matched))
    rows.append(row_total)
    # correct row
    row_correct: List[Any] = ["CORRECT"]
    for i, m in enumerate(models):
        row_correct.append(str(m.n_matched - totals[i]))
    for i, m in enumerate(models):
        row_correct.append(_pct(m.n_matched - totals[i], m.n_matched))
    rows.append(row_correct)
    print_table(headers, rows)


def print_top_confusions_comparison(models: List[ModelResult], top_n: int):
    print_section("D. Intent 混同ペア 比較 (gold -> pred)")
    if len(models) == 1:
        _print_top_confusions_single(models[0], top_n)
        return

    # Collect all confusion pairs across models
    all_pairs: Set[Tuple[str, str]] = set()
    pair_counts: List[Counter] = []
    for m in models:
        cnt = Counter(p for p in m.intent_pairs if p[0] != p[1])
        pair_counts.append(cnt)
        all_pairs.update(cnt.keys())

    # Sort by max count across models
    sorted_pairs = sorted(all_pairs, key=lambda p: max(c[p] for c in pair_counts), reverse=True)

    names = [m.name for m in models]
    headers = ["Gold", "", "Pred"] + names + (["diff"] if len(models) == 2 else []) + ["Relation"]
    rows = []
    for gi, pi in sorted_pairs[:top_n]:
        gp = gi.split("_", 1); pp = pi.split("_", 1)
        relation = ""
        if gp[0] == pp[0]:
            relation = "同一Scenario"
        elif len(gp) > 1 and len(pp) > 1 and gp[1] == pp[1]:
            relation = "同一Action"
        vals = [pair_counts[i][(gi, pi)] for i in range(len(models))]
        row: List[Any] = [gi, "->", pi] + [str(v) for v in vals]
        if len(models) == 2:
            row.append(f"{vals[1]-vals[0]:+d}")
        row.append(relation)
        rows.append(row)
    print_table(headers, rows)


def _print_top_confusions_single(m: ModelResult, top_n: int):
    pairs = top_confusions(m.intent_pairs, top_n)
    rows = []
    for (gi, pi), cnt in pairs:
        gp = gi.split("_", 1); pp = pi.split("_", 1)
        relation = ""
        if gp[0] == pp[0]:
            relation = "同一Scenario"
        elif len(gp) > 1 and len(pp) > 1 and gp[1] == pp[1]:
            relation = "同一Action"
        rows.append([gi, "->", pi, cnt, relation])
    print_table(["Gold Intent", "", "Pred Intent", "Count", "Relation"], rows)


def print_scenario_comparison(models: List[ModelResult], top_n: int):
    print_section("E. Scenario 混同 比較")
    if len(models) == 1:
        pairs = [(g, p) for (gi, pi) in models[0].intent_pairs
                 for g, p in [(gi.split("_", 1)[0], pi.split("_", 1)[0])] if g != p]
        tc = top_confusions([(g, p) for g, p in pairs], top_n)
        if tc:
            print_table(["Gold", "", "Pred", "Count"],
                        [[g, "->", p, c] for (g, p), c in tc])
        else:
            print("  (なし)")
        return

    all_pairs: Set[Tuple[str, str]] = set()
    counters = []
    for m in models:
        sc_pairs = [(gi.split("_", 1)[0], pi.split("_", 1)[0])
                    for gi, pi in m.intent_pairs]
        cnt = Counter(p for p in sc_pairs if p[0] != p[1])
        counters.append(cnt)
        all_pairs.update(cnt.keys())
    sorted_p = sorted(all_pairs, key=lambda p: max(c[p] for c in counters), reverse=True)
    names = [m.name for m in models]
    headers = ["Gold", "", "Pred"] + names
    rows = []
    for g, p in sorted_p[:top_n]:
        rows.append([g, "->", p] + [str(c[(g, p)]) for c in counters])
    if rows:
        print_table(headers, rows)
    else:
        print("  (なし)")


def print_confusion_clusters_comparison(models: List[ModelResult], top_n: int):
    print_section("G. 意味的混同クラスタ 比較（双方向混同ペア）")
    print("  仮説: 意味的に類似・上位/下位関係のラベル間で双方向に混同が発生している")
    print()

    all_clusters_data: List[List[Dict]] = []
    all_pair_keys: Set[Tuple[str, str]] = set()
    for m in models:
        cl = find_confusion_clusters(m.intent_pairs, min_count=2)
        all_clusters_data.append(cl)
        for c in cl:
            all_pair_keys.add((c["label_a"], c["label_b"]))

    if not all_pair_keys:
        print("  (双方向混同なし)")
        return

    # Build lookup per model
    lookups = []
    for cl_list in all_clusters_data:
        d = {(c["label_a"], c["label_b"]): c for c in cl_list}
        lookups.append(d)

    # Sort by max total
    sorted_keys = sorted(all_pair_keys,
        key=lambda k: max(lu.get(k, {}).get("total", 0) for lu in lookups), reverse=True)

    names = [m.name for m in models]
    headers = ["Label A", "", "Label B"] + [f"{n}" for n in names] + ["Scope"]
    rows = []
    for a, b in sorted_keys[:top_n]:
        vals = []
        scope = ""
        for lu in lookups:
            c = lu.get((a, b))
            if c:
                direction = "<->" if c["bidirectional"] else " -> "
                vals.append(str(c["total"]))
                if c["same_scenario"]:
                    scope = "同一Scenario"
                else:
                    scope = "異Scenario"
            else:
                vals.append("0")
        rows.append([a, "<->", b] + vals + [scope])
    print_table(headers, rows)


def print_entity_type_confusion_comparison(models: List[ModelResult], top_n: int):
    print_section("I. Entity Type エラー 比較")

    # Collect all swap pairs
    all_swaps: Set[Tuple[str, str]] = set()
    swap_counters = []
    for m in models:
        _, _, sw = entity_type_confusion(m.matched)
        swap_counters.append(sw)
        all_swaps.update(sw.keys())

    if all_swaps:
        print("\n  --- Entity Type 入れ替え (同一 filler, 異なる type) ---")
        sorted_sw = sorted(all_swaps,
            key=lambda k: max(c[k] for c in swap_counters), reverse=True)
        names = [m.name for m in models]
        headers = ["Gold Type", "", "Pred Type"] + names
        rows = []
        for g, p in sorted_sw[:top_n]:
            rows.append([g, "->", p] + [str(c[(g, p)]) for c in swap_counters])
        print_table(headers, rows)
    else:
        print("\n  Entity Type 入れ替え: (なし)")


WER_BINS = [
    (0.0,  0.0,   "WER=0 (exact)"),
    (0.0,  0.20,  "0<WER<=0.20"),
    (0.20, 0.50,  "0.20<WER<=0.50"),
    (0.50, float("inf"), "WER>0.50"),
]


def _in_wer_bin(w: float, lo: float, hi: float) -> bool:
    if lo == 0.0 and hi == 0.0:
        return w == 0.0
    if hi == float("inf"):
        return w > lo
    return lo < w <= hi


def _compute_bin_metrics(
    intent_pairs: List[Tuple[str, str]],
    matched: List[Tuple[Dict, Dict]],
    wer_list: List[Optional[float]],
    lo: float, hi: float,
) -> Dict[str, Any]:
    """指定 WER 範囲のサンプルだけで Intent Acc / Entity F1 を算出。"""
    intent_m = FMeasureAccumulator()
    entity_m = SpanFMeasureAccumulator()
    n = 0
    for i, w in enumerate(wer_list):
        if w is None:
            continue
        if not _in_wer_bin(w, lo, hi):
            continue
        gi, pi = intent_pairs[i]
        intent_m.add(gi, pi)
        pred, gold = matched[i]
        entity_m.add(gold["entities"], pred["entities"])
        n += 1
    _, _, ef = entity_m.overall()
    return {
        "n": n,
        "intent_acc": intent_m.accuracy if n > 0 else 0.0,
        "entity_f1": ef,
    }


def print_wer_analysis_comparison(models: List[ModelResult]):
    print_section("K. WER 別 Intent Accuracy / Entity F1 比較")

    # Check if any model has wer data
    has_wer = any(
        any(w is not None for w in m.wer_list)
        for m in models
    )
    if not has_wer:
        print("  (WER データなし — --gold で test.jsonl を指定してください)")
        return

    names = [m.name for m in models]
    # Build header
    sub_cols = []
    for n in names:
        sub_cols.extend([f"{n} IntAcc", f"{n} EntF1"])
    headers = ["WER range", "N"] + sub_cols

    rows = []
    for lo, hi, label in WER_BINS:
        # N is based on first model (same gold)
        bins_data = [_compute_bin_metrics(m.intent_pairs, m.matched, m.wer_list, lo, hi)
                     for m in models]
        n_samples = max(d["n"] for d in bins_data) if bins_data else 0
        if n_samples == 0:
            continue
        row: List[Any] = [label, n_samples]
        for d in bins_data:
            row.append(_f4(d["intent_acc"]))
            row.append(_f4(d["entity_f1"]))
        rows.append(row)

    # ALL row
    row_all: List[Any] = ["ALL", max(m.n_matched for m in models)]
    for m in models:
        row_all.append(_f4(m.intent_acc))
        row_all.append(_f4(m.entity_f1))
    rows.append(row_all)

    print_table(headers, rows)

    # Error type breakdown per WER bin
    print(f"\n  --- WER 別 Intent エラー分類 (各モデル) ---")
    for m in models:
        has_m_wer = any(w is not None for w in m.wer_list)
        if not has_m_wer:
            continue
        print(f"\n  [{m.name}]")
        bin_headers = ["WER range", "N", "Err", "ErrRate", "同Scen/異Act", "異Scen/同Act", "両方異なる"]
        bin_rows = []
        for lo, hi, label in WER_BINS:
            # Filter indices
            idxs = [i for i, w in enumerate(m.wer_list)
                    if w is not None and _in_wer_bin(w, lo, hi)]
            if not idxs:
                continue
            sub_pairs = [m.intent_pairs[i] for i in idxs]
            bd = error_type_breakdown(sub_pairs)
            n_err = sum(bd.values())
            bin_rows.append([
                label, len(idxs), n_err,
                _pct(n_err, len(idxs)),
                str(bd["same_scenario_diff_action"]),
                str(bd["diff_scenario_same_action"]),
                str(bd["diff_both"]),
            ])
        print_table(bin_headers, bin_rows)


def print_scenario_acc_comparison(models: List[ModelResult]):
    print_section("J. Scenario 別 Intent Accuracy 比較")

    # Collect all gold scenarios
    all_scenarios: Set[str] = set()
    model_data = []
    for m in models:
        total_map: Dict[str, int] = defaultdict(int)
        err_map: Dict[str, int] = defaultdict(int)
        for gi, pi in m.intent_pairs:
            gs = gi.split("_", 1)[0]
            total_map[gs] += 1
            if gi != pi:
                err_map[gs] += 1
        all_scenarios.update(total_map.keys())
        model_data.append((total_map, err_map))

    names = [m.name for m in models]
    headers = ["Scenario", "Gold"] + [f"{n} Acc" for n in names]
    if len(models) == 2:
        headers.append("diff")
    rows = []
    for s in sorted(all_scenarios):
        gold_n = max(md[0][s] for md in model_data)
        accs = []
        for total_map, err_map in model_data:
            t = total_map[s]
            e = err_map[s]
            accs.append(1.0 - e / t if t > 0 else 0.0)
        row: List[Any] = [s, gold_n] + [_f4(a) for a in accs]
        if len(models) == 2:
            row.append(f"{accs[1]-accs[0]:+.4f}")
        rows.append(row)
    print_table(headers, rows)


# ============================================================================
# 9. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="prediction.jsonl のエラー分析・比較（evaluate.py 準拠）"
    )
    parser.add_argument("predictions", nargs="+", type=str,
                        help="prediction.jsonl のパス (複数指定で比較)")
    parser.add_argument("--gold", type=str, default=None,
                        help="gold test.jsonl のパス")
    parser.add_argument("--key", type=str, default="file",
                        help="マッチングキー (default: file)")
    parser.add_argument("--names", type=str, default=None,
                        help="モデル名 (カンマ区切り、ファイルと同じ順)")
    parser.add_argument("--top", type=int, default=20,
                        help="表示する上位件数 (default: 20)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="分析結果を JSON で保存するパス")
    args = parser.parse_args()

    # --- Resolve paths ---
    pred_paths = [resolve_prediction_path(p) for p in args.predictions]

    # --- Model names ---
    if args.names:
        names = [n.strip() for n in args.names.split(",")]
        if len(names) != len(pred_paths):
            print(f"ERROR: --names count ({len(names)}) != prediction files ({len(pred_paths)})",
                  file=sys.stderr)
            sys.exit(1)
    else:
        names = [derive_model_name(p) for p in pred_paths]
        # deduplicate
        seen: Dict[str, int] = {}
        for i, n in enumerate(names):
            if n in seen:
                seen[n] += 1
                names[i] = f"{n}_{seen[n]}"
            else:
                seen[n] = 0

    # --- Load gold ---
    gold_path = args.gold
    if gold_path is None:
        gold_path = _auto_detect_gold_path(pred_paths[0])
    gold_map: Optional[Dict[str, Dict[str, Any]]] = None

    if gold_path and os.path.exists(gold_path):
        print(f"Loading gold data: {gold_path}")
        if args.key == "slurp_id":
            gold_map = load_gold_by_slurp_id(gold_path)
        else:
            gold_map = load_gold_from_test_jsonl(gold_path)
        print(f"  Gold entries: {len(gold_map)}")
    else:
        if gold_path:
            print(f"  [WARN] --gold not found: {gold_path}", file=sys.stderr)
        print("  No --gold; using embedded target_label / target")

    # --- Process each model ---
    models: List[ModelResult] = []
    for path, name in zip(pred_paths, names):
        print(f"\nProcessing: {name}  ({path})")
        r = process_model(path, name, gold_map, args.key)
        print(f"  Matched: {r.n_matched}  IntentAcc={_f4(r.intent_acc)}  EntityF1={_f4(r.entity_f1)}")
        models.append(r)

    if not models or all(m.n_matched == 0 for m in models):
        print("\nERROR: No matched data.", file=sys.stderr)
        sys.exit(1)

    # ================================================================
    #  Output
    # ================================================================
    print_overall_comparison(models)

    print_section("B. Per-label F1 比較")
    for lt in ["scenario", "action", "intent", "entity"]:
        print_per_label_comparison(models, lt, args.top)

    print_error_breakdown_comparison(models)
    print_top_confusions_comparison(models, args.top)
    print_scenario_comparison(models, args.top)

    # F. Action confusion (single-style for brevity in multi)
    print_section("F. Action 混同 比較")
    all_act_pairs: Set[Tuple[str, str]] = set()
    act_counters = []
    for m in models:
        act_pairs = [(gi.split("_", 1)[1] if "_" in gi else gi,
                       pi.split("_", 1)[1] if "_" in pi else pi)
                      for gi, pi in m.intent_pairs]
        cnt = Counter(p for p in act_pairs if p[0] != p[1])
        act_counters.append(cnt)
        all_act_pairs.update(cnt.keys())
    if all_act_pairs:
        sorted_ap = sorted(all_act_pairs,
            key=lambda p: max(c[p] for c in act_counters), reverse=True)
        headers = ["Gold Action", "", "Pred Action"] + names
        rows_t = []
        for g, p in sorted_ap[:args.top]:
            rows_t.append([g, "->", p] + [str(c[(g, p)]) for c in act_counters])
        print_table(headers, rows_t)
    else:
        print("  (なし)")

    print_confusion_clusters_comparison(models, args.top)

    # H. Cross-scenario absorption
    print_section("H. Scenario横断の吸収分析 比較")
    all_intent_conf: Set[Tuple[str, str]] = set()
    intent_conf_counters = []
    for m in models:
        cnt = Counter(p for p in m.intent_pairs if p[0] != p[1])
        intent_conf_counters.append(cnt)
        all_intent_conf.update(cnt.keys())
    if all_intent_conf:
        sorted_ic = sorted(all_intent_conf,
            key=lambda p: max(c[p] for c in intent_conf_counters), reverse=True)
        headers = ["Gold", "", "Pred"] + names + ["Type"]
        rows_t = []
        for gi, pi in sorted_ic[:args.top]:
            gp = gi.split("_", 1); pp = pi.split("_", 1)
            gs, ga = (gp[0], gp[1]) if len(gp) > 1 else (gp[0], "")
            ps, pa = (pp[0], pp[1]) if len(pp) > 1 else (pp[0], "")
            tag = ""
            if ga == pa and gs != ps:
                tag = "Action共有->吸収"
            elif gs == ps and ga != pa:
                tag = "Scenario内混同"
            vals = [str(c[(gi, pi)]) for c in intent_conf_counters]
            rows_t.append([gi, "->", pi] + vals + [tag])
            if len(rows_t) >= args.top:
                break
        print_table(headers, rows_t)
    else:
        print("  (なし)")

    print_entity_type_confusion_comparison(models, args.top)
    print_scenario_acc_comparison(models)
    print_wer_analysis_comparison(models)

    # ================================================================
    # Summary JSON
    # ================================================================
    summary = {
        "models": [
            {
                "name": m.name, "path": m.path, "n_matched": m.n_matched,
                "scenario_acc": m.scenario_acc, "action_acc": m.action_acc,
                "intent_acc": m.intent_acc,
                "entity_f1": m.entity_f1, "entity_p": m.entity_p, "entity_r": m.entity_r,
                "error_breakdown": m.error_breakdown,
            }
            for m in models
        ],
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n  Summary JSON saved to: {args.output_json}")

    print_section("Summary")
    headers = ["Model", "IntentAcc", "EntityF1"]
    rows_t = [[m.name, _f4(m.intent_acc), _f4(m.entity_f1)] for m in models]
    print_table(headers, rows_t)
    print()


if __name__ == "__main__":
    main()
