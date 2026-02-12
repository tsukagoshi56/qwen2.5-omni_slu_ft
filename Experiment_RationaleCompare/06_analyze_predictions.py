#!/usr/bin/env python3
"""
06_analyze_predictions.py
=========================
prediction.jsonl を分析するスクリプト。
evaluate.py の基準 (FMeasure / SpanFMeasure) に従って
  - Scenario / Action / Intent (scenario_action) の Accuracy & F1
  - Entity Span-F1 (exact match)
を算出し、さらに意味的類似・上位包含関係に着目したエラー分析を行う。

Gold データの取得方式 (evaluate.py 準拠):
  1. --gold <test.jsonl> 指定時: test.jsonl を読み込み、recording の file 名で
     prediction とマッチングする (evaluate.py と同一方式)。
  2. --gold 未指定時: prediction.jsonl 内の target_label / target フィールドから
     gold ラベルを取得する。

Usage:
    # evaluate.py 方式 (推奨)
    python 06_analyze_predictions.py <prediction.jsonl> --gold slurp/dataset/slurp/test.jsonl

    # prediction.jsonl に gold が埋め込まれている場合
    python 06_analyze_predictions.py <prediction.jsonl>

    # フォルダ指定 (prediction.jsonl を自動探索)
    python 06_analyze_predictions.py output/modelname/ --gold slurp/dataset/slurp/test.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
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
    """Entity を (type, filler) の正規化タプルへ。"""
    ent_type = str(entity.get("type", "")).strip().lower()
    filler = entity.get("filler")
    if filler is None:
        filler = entity.get("filter", "")
    if filler is None:
        filler = entity.get("value", "")
    filler = re.sub(r"\s+", " ", str(filler).strip().lower())
    return ent_type, filler


def _parse_entities_from_list(raw_entities: Any) -> List[Dict[str, str]]:
    """Entity リストをパースして [{type, filler}] に統一。"""
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


# ---------- Gold loading (evaluate.py util.py release2prediction 準拠) ----------

def load_gold_from_test_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    """test.jsonl を読み込み、recording の file 名をキーとした辞書を返す。
    evaluate.py の load_gold_data / release2prediction と同一ロジック。
    """
    gold_map: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            # release2prediction 相当
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
            # 各 recording の file 名をキーに展開
            for rec in example.get("recordings", []):
                fname = rec.get("file", "")
                if fname:
                    gold_map[fname] = res
    return gold_map


def load_gold_by_slurp_id(path: str) -> Dict[str, Dict[str, Any]]:
    """test.jsonl を読み込み、slurp_id をキーとした辞書を返す。"""
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


# ---------- Prediction → Gold label extraction ----------

def _extract_gold_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """prediction.jsonl の行から gold label を取得 (target_label or target)。
    gold が取得できない場合は None を返す。"""
    target_label = row.get("target_label")
    if isinstance(target_label, dict) and (
        target_label.get("scenario") or target_label.get("action") or target_label.get("entities")
    ):
        return {
            "scenario": str(target_label.get("scenario", "")).strip(),
            "action": str(target_label.get("action", "")).strip(),
            "entities": _parse_entities_from_list(target_label.get("entities", [])),
        }
    # target フィールドから J: {...} をパースする
    target_str = row.get("target", "")
    if isinstance(target_str, str) and target_str.strip():
        obj = _parse_j_line(target_str)
        if obj:
            return obj
    return None


def _parse_j_line(text: str) -> Optional[Dict[str, Any]]:
    """target 文字列 ("C: ...\nR: ...\nJ: {...}") から J: の JSON をパース。"""
    # J: 行を探す
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("J:"):
            json_str = line[2:].strip()
            try:
                obj = json.loads(json_str)
                if isinstance(obj, dict):
                    s = str(obj.get("scenario", obj.get("Scenario", ""))).strip()
                    a = str(obj.get("action", obj.get("Action", ""))).strip()
                    # Intent フィールドからの fallback
                    if not s and not a:
                        intent = str(obj.get("intent", obj.get("Intent", ""))).strip()
                        if "_" in intent:
                            s, a = intent.split("_", 1)
                    ents = obj.get("entities", obj.get("Entities", obj.get("slots", [])))
                    return {
                        "scenario": s, "action": a,
                        "entities": _parse_entities_from_list(ents or []),
                    }
            except json.JSONDecodeError:
                pass
    # J: 行がなければ全体を JSON としてパース
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            s = str(obj.get("scenario", "")).strip()
            a = str(obj.get("action", "")).strip()
            ents = obj.get("entities", [])
            if s or a:
                return {
                    "scenario": s, "action": a,
                    "entities": _parse_entities_from_list(ents or []),
                }
    except json.JSONDecodeError:
        pass
    return None


def _extract_pred_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """prediction.jsonl の行から pred label を取得。"""
    return {
        "scenario": str(row.get("scenario", "")).strip(),
        "action": str(row.get("action", "")).strip(),
        "entities": _parse_entities_from_list(row.get("entities", [])),
    }


# ============================================================================
# 2. Metrics (evaluate.py 準拠)
# ============================================================================

def compute_prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class FMeasureAccumulator:
    """evaluate.py の FMeasure と同じ: 完全一致で TP/FP/FN を蓄積。"""
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


class SpanFMeasureAccumulator:
    """evaluate.py の SpanFMeasure と同じ: (type, filler) 完全一致。"""
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


# ============================================================================
# 3. Error Analysis helpers
# ============================================================================

def build_confusion_counter(pairs: List[Tuple[str, str]]) -> Counter:
    return Counter(pairs)


def top_confusions(confusion: Counter, top_n: int = 20) -> List[Tuple[Tuple[str, str], int]]:
    errors = {k: v for k, v in confusion.items() if k[0] != k[1]}
    return sorted(errors.items(), key=lambda x: -x[1])[:top_n]


def error_type_breakdown(intent_pairs: List[Tuple[str, str]]) -> Dict[str, int]:
    same_s_diff_a = 0
    diff_s_same_a = 0
    diff_both = 0
    for (gold_intent, pred_intent) in intent_pairs:
        if gold_intent == pred_intent:
            continue
        g_parts = gold_intent.split("_", 1)
        p_parts = pred_intent.split("_", 1)
        gs = g_parts[0] if len(g_parts) > 0 else ""
        ga = g_parts[1] if len(g_parts) > 1 else ""
        ps = p_parts[0] if len(p_parts) > 0 else ""
        pa = p_parts[1] if len(p_parts) > 1 else ""
        if gs == ps and ga != pa:
            same_s_diff_a += 1
        elif gs != ps and ga == pa:
            diff_s_same_a += 1
        else:
            diff_both += 1
    return {
        "same_scenario_diff_action": same_s_diff_a,
        "diff_scenario_same_action": diff_s_same_a,
        "diff_both": diff_both,
    }


def entity_type_confusion(
    matched: List[Tuple[Dict, Dict]],
) -> Tuple[Counter, Counter, Counter]:
    """Entity レベルのエラーを分析。
    matched: List of (pred, gold) dicts.
    """
    type_fp = Counter()
    type_fn = Counter()
    type_swap = Counter()

    for pred, gold in matched:
        pred_ents = [_normalize_entity(e) for e in (pred["entities"] or [])]
        gold_ents = [_normalize_entity(e) for e in (gold["entities"] or [])]

        pred_set = set(pred_ents)
        gold_set = set(gold_ents)

        fp_ents = pred_set - gold_set
        fn_ents = gold_set - pred_set

        for etype, _ in fp_ents:
            type_fp[etype] += 1
        for etype, _ in fn_ents:
            type_fn[etype] += 1

        # filler が同じだが type が異なるペアを検出
        fp_by_filler: Dict[str, List[str]] = defaultdict(list)
        fn_by_filler: Dict[str, List[str]] = defaultdict(list)
        for etype, filler in fp_ents:
            if filler:
                fp_by_filler[filler].append(etype)
        for etype, filler in fn_ents:
            if filler:
                fn_by_filler[filler].append(etype)
        for filler in set(fp_by_filler) & set(fn_by_filler):
            for gt in fn_by_filler[filler]:
                for pt in fp_by_filler[filler]:
                    if gt != pt:
                        type_swap[(gt, pt)] += 1

    return type_fp, type_fn, type_swap


def find_confusion_clusters(
    intent_pairs: List[Tuple[str, str]], min_count: int = 2,
) -> List[Dict[str, Any]]:
    """双方向混同ペアを検出。"""
    pair_count: Counter = Counter()
    for g, p in intent_pairs:
        if g != p:
            key = tuple(sorted([g, p]))
            pair_count[key] += 1

    clusters = []
    for (a, b), cnt in pair_count.most_common():
        if cnt < min_count:
            break
        a_to_b = sum(1 for g, p in intent_pairs if g == a and p == b)
        b_to_a = sum(1 for g, p in intent_pairs if g == b and p == a)
        bidirectional = a_to_b > 0 and b_to_a > 0
        a_parts = a.split("_", 1)
        b_parts = b.split("_", 1)
        same_scenario = (a_parts[0] == b_parts[0]) if len(a_parts) > 1 and len(b_parts) > 1 else False
        clusters.append({
            "label_a": a, "label_b": b, "total": cnt,
            "a_to_b": a_to_b, "b_to_a": b_to_a,
            "bidirectional": bidirectional, "same_scenario": same_scenario,
        })
    return clusters


# ============================================================================
# 4. Report formatting
# ============================================================================

def fmt_prf(p: float, r: float, f: float) -> str:
    return f"P={p:.4f}  R={r:.4f}  F1={f:.4f}"


def print_section(title: str, width: int = 72):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_table(headers: List[str], rows: List[List[Any]], max_rows: int = 50):
    col_widths = [len(h) for h in headers]
    display_rows = rows[:max_rows]
    for row in display_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * sum(col_widths + [2 * (len(col_widths) - 1)]))
    for row in display_rows:
        print(fmt.format(*[str(c) for c in row]))
    if len(rows) > max_rows:
        print(f"  ... ({len(rows) - max_rows} more rows)")


def per_label_f1(acc: FMeasureAccumulator, label_name: str, top_n: int = 20):
    all_tags = sorted(set(acc.tp) | set(acc.fp) | set(acc.fn))
    rows = []
    for tag in all_tags:
        p, r, f = compute_prf(acc.tp[tag], acc.fp[tag], acc.fn[tag])
        total = acc.tp[tag] + acc.fn[tag]
        rows.append([tag, f"{p:.4f}", f"{r:.4f}", f"{f:.4f}",
                      int(acc.tp[tag]), int(acc.fp[tag]), int(acc.fn[tag]), int(total)])
    rows.sort(key=lambda x: float(x[3]))
    print(f"\n  --- {label_name}: Per-label metrics (worst F1 first, top {top_n}) ---")
    print_table(["Label", "Prec", "Recall", "F1", "TP", "FP", "FN", "Gold"],
                rows[:top_n])


def per_entity_type_f1(acc: SpanFMeasureAccumulator, top_n: int = 20):
    all_tags = sorted(set(acc.tp) | set(acc.fp) | set(acc.fn))
    rows = []
    for tag in all_tags:
        p, r, f = compute_prf(acc.tp[tag], acc.fp[tag], acc.fn[tag])
        total = acc.tp[tag] + acc.fn[tag]
        rows.append([tag, f"{p:.4f}", f"{r:.4f}", f"{f:.4f}",
                      int(acc.tp[tag]), int(acc.fp[tag]), int(acc.fn[tag]), int(total)])
    rows.sort(key=lambda x: float(x[3]))
    print(f"\n  --- Entity type: Per-type metrics (worst F1 first, top {top_n}) ---")
    print_table(["Entity Type", "Prec", "Recall", "F1", "TP", "FP", "FN", "Gold"],
                rows[:top_n])


# ============================================================================
# 5. Gold matching  (evaluate.py と同一)
# ============================================================================

def _auto_detect_gold_path(prediction_path: str) -> Optional[str]:
    """prediction.jsonl のパスからプロジェクトルートを推定し test.jsonl を探す。"""
    # 上に辿って slurp/dataset/slurp/test.jsonl を探す
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
    """prediction 行と gold データをキーフィールドでマッチングする。
    Returns: List of (pred_label, gold_label) dicts.
    """
    matched = []
    not_found = 0
    for row in pred_rows:
        key = row.get(key_field)
        if key is None:
            key = row.get("slurp_id") or row.get("id")
        key = str(key).strip() if key else ""
        if not key:
            not_found += 1
            continue
        gold = gold_map.get(key)
        if gold is None:
            # basename fallback
            gold = gold_map.get(os.path.basename(key))
        if gold is None:
            not_found += 1
            continue
        pred = _extract_pred_from_row(row)
        matched.append((pred, gold))
    if not_found > 0:
        print(f"  [WARN] {not_found}/{len(pred_rows)} predictions had no matching gold entry",
              file=sys.stderr)
    return matched


def match_predictions_embedded_gold(
    pred_rows: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """prediction.jsonl 内の target_label / target フィールドから gold を取得。"""
    matched = []
    no_gold = 0
    for row in pred_rows:
        pred = _extract_pred_from_row(row)
        gold = _extract_gold_from_row(row)
        if gold is None or (not gold["scenario"] and not gold["action"]):
            no_gold += 1
            continue
        matched.append((pred, gold))
    if no_gold > 0:
        print(f"  [WARN] {no_gold}/{len(pred_rows)} rows had no embedded gold label",
              file=sys.stderr)
    return matched


# ============================================================================
# 6. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="prediction.jsonl のエラー分析（evaluate.py 準拠の指標 + 意味的混同分析）"
    )
    parser.add_argument("prediction", type=str,
                        help="prediction.jsonl のパス (フォルダ指定可)")
    parser.add_argument("--gold", type=str, default=None,
                        help="gold test.jsonl のパス (evaluate.py 方式のマッチング)")
    parser.add_argument("--key", type=str, default="file",
                        help="prediction と gold のマッチングキー (default: file)")
    parser.add_argument("--top", type=int, default=20,
                        help="表示する上位件数 (default: 20)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="分析結果を JSON で保存するパス")
    args = parser.parse_args()

    # --- Resolve prediction path ---
    prediction_path = args.prediction
    if os.path.isdir(prediction_path):
        candidate = os.path.join(prediction_path, "prediction.jsonl")
        if os.path.exists(candidate):
            prediction_path = candidate
        else:
            print(f"ERROR: {prediction_path} is a directory but prediction.jsonl not found inside.",
                  file=sys.stderr)
            sys.exit(1)
    if not os.path.exists(prediction_path):
        print(f"ERROR: {prediction_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading predictions: {prediction_path}")
    pred_rows = read_jsonl(prediction_path)
    print(f"  Loaded {len(pred_rows)} prediction rows")
    if not pred_rows:
        print("No data to analyze.", file=sys.stderr)
        sys.exit(1)

    # --- Diagnostic: show first row structure ---
    first = pred_rows[0]
    print(f"\n  [DEBUG] First row keys: {list(first.keys())}")
    print(f"  [DEBUG] First row 'file':         {first.get('file', '(MISSING)')!r}")
    print(f"  [DEBUG] First row 'slurp_id':     {first.get('slurp_id', '(MISSING)')!r}")
    print(f"  [DEBUG] First row 'scenario':     {first.get('scenario', '(MISSING)')!r}")
    print(f"  [DEBUG] First row 'action':       {first.get('action', '(MISSING)')!r}")
    tl = first.get("target_label")
    if isinstance(tl, dict):
        print(f"  [DEBUG] First row 'target_label': scenario={tl.get('scenario')!r}, "
              f"action={tl.get('action')!r}, entities={len(tl.get('entities', []))} items")
    else:
        print(f"  [DEBUG] First row 'target_label': {tl!r}")
    target_str = first.get("target", "")
    if isinstance(target_str, str) and len(target_str) > 100:
        print(f"  [DEBUG] First row 'target':       {target_str[:100]}...")
    else:
        print(f"  [DEBUG] First row 'target':       {target_str!r}")

    # --- Load gold & match ---
    gold_path = args.gold
    if gold_path is None:
        gold_path = _auto_detect_gold_path(prediction_path)
    gold_source = "none"

    if gold_path and os.path.exists(gold_path):
        print(f"\nLoading gold data: {gold_path}")
        if args.key == "slurp_id":
            gold_map = load_gold_by_slurp_id(gold_path)
        else:
            gold_map = load_gold_from_test_jsonl(gold_path)
        print(f"  Gold entries: {len(gold_map)}")

        matched = match_predictions_with_gold(pred_rows, gold_map, key_field=args.key)
        gold_source = f"test.jsonl (key={args.key})"
    else:
        if gold_path:
            print(f"  [WARN] --gold path not found: {gold_path}", file=sys.stderr)
        print("\n  No --gold specified. Using embedded target_label / target from prediction rows.")
        matched = match_predictions_embedded_gold(pred_rows)
        gold_source = "embedded (target_label/target)"

    print(f"\n  Matched (pred, gold) pairs: {len(matched)}")
    print(f"  Gold source: {gold_source}")

    if not matched:
        print("\nERROR: No matched pairs found. Possible causes:", file=sys.stderr)
        print("  - prediction.jsonl に 'file' キーがない → --key slurp_id を試す", file=sys.stderr)
        print("  - --gold の test.jsonl パスが間違っている", file=sys.stderr)
        print("  - prediction に target_label が埋め込まれていない", file=sys.stderr)
        sys.exit(1)

    # ---- Accumulate metrics ----
    scenario_metric = FMeasureAccumulator()
    action_metric = FMeasureAccumulator()
    intent_metric = FMeasureAccumulator()
    entity_metric = SpanFMeasureAccumulator()

    intent_pairs: List[Tuple[str, str]] = []
    scenario_pairs: List[Tuple[str, str]] = []
    action_pairs: List[Tuple[str, str]] = []

    for pred, gold in matched:
        ps, pa = pred["scenario"], pred["action"]
        gs, ga = gold["scenario"], gold["action"]
        pi = f"{ps}_{pa}"
        gi = f"{gs}_{ga}"

        scenario_metric.add(gs, ps)
        action_metric.add(ga, pa)
        intent_metric.add(gi, pi)
        entity_metric.add(gold["entities"], pred["entities"])

        intent_pairs.append((gi, pi))
        scenario_pairs.append((gs, ps))
        action_pairs.append((ga, pa))

    total = len(matched)

    # ================================================================
    # A. Overall Metrics
    # ================================================================
    print_section("A. Overall Metrics (evaluate.py 準拠)")

    sp, sr, sf = scenario_metric.overall()
    print(f"  Scenario   Acc={scenario_metric.accuracy:.4f}  {fmt_prf(sp, sr, sf)}")

    ap, ar, af = action_metric.overall()
    print(f"  Action     Acc={action_metric.accuracy:.4f}  {fmt_prf(ap, ar, af)}")

    ip, ir, if1 = intent_metric.overall()
    print(f"  Intent     Acc={intent_metric.accuracy:.4f}  {fmt_prf(ip, ir, if1)}")

    ep, er, ef = entity_metric.overall()
    print(f"  Entity (span-F1, exact match)   {fmt_prf(ep, er, ef)}")

    print(f"\n  Matched samples: {total}")

    # ================================================================
    # B. Per-label metrics
    # ================================================================
    print_section("B. Per-label Metrics (低精度ラベル)")
    per_label_f1(scenario_metric, "Scenario", args.top)
    per_label_f1(action_metric, "Action", args.top)
    per_label_f1(intent_metric, "Intent (scenario_action)", args.top)
    per_entity_type_f1(entity_metric, args.top)

    # ================================================================
    # C. Intent Error Type Breakdown
    # ================================================================
    print_section("C. Intent エラーの分類")
    breakdown = error_type_breakdown(intent_pairs)
    total_errors = sum(breakdown.values())
    print(f"  Total intent errors: {total_errors} / {total} ({100*total_errors/total:.1f}%)")
    print()
    for etype, cnt in sorted(breakdown.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / total_errors if total_errors > 0 else 0
        desc = {
            "same_scenario_diff_action": "同一 Scenario, 異なる Action（Scenario内の混同）",
            "diff_scenario_same_action": "異なる Scenario, 同一 Action（Scenario間の混同）",
            "diff_both": "Scenario も Action も異なる（完全に異なるラベル）",
        }
        print(f"  {etype:40s}: {cnt:5d} ({pct:5.1f}%)  -- {desc.get(etype, '')}")

    # ================================================================
    # D. Top Confused Intent Pairs
    # ================================================================
    print_section("D. 最も頻度の高い Intent 混同ペア (gold -> pred)")
    intent_confusion = build_confusion_counter(intent_pairs)
    top_pairs = top_confusions(intent_confusion, args.top)
    rows_table = []
    for (gi, pi), cnt in top_pairs:
        gp = gi.split("_", 1)
        pp = pi.split("_", 1)
        gs = gp[0] if len(gp) > 0 else ""
        ps = pp[0] if len(pp) > 0 else ""
        relation = ""
        if gs == ps:
            relation = "同一Scenario"
        elif len(gp) > 1 and len(pp) > 1 and gp[1] == pp[1]:
            relation = "同一Action"
        rows_table.append([gi, "->", pi, cnt, relation])
    print_table(["Gold Intent", "", "Pred Intent", "Count", "Relation"], rows_table)

    # ================================================================
    # E. Scenario Confusion
    # ================================================================
    print_section("E. Scenario 混同 (gold -> pred)")
    scenario_confusion = build_confusion_counter(scenario_pairs)
    top_sc = top_confusions(scenario_confusion, args.top)
    if top_sc:
        rows_table = [[g, "->", p, c] for (g, p), c in top_sc]
        print_table(["Gold Scenario", "", "Pred Scenario", "Count"], rows_table)
    else:
        print("  (Scenario 混同なし)")

    # ================================================================
    # F. Action Confusion
    # ================================================================
    print_section("F. Action 混同 (gold -> pred)")
    action_confusion = build_confusion_counter(action_pairs)
    top_ac = top_confusions(action_confusion, args.top)
    if top_ac:
        rows_table = [[g, "->", p, c] for (g, p), c in top_ac]
        print_table(["Gold Action", "", "Pred Action", "Count"], rows_table)
    else:
        print("  (Action 混同なし)")

    # ================================================================
    # G. Semantic Confusion Clusters
    # ================================================================
    print_section("G. 意味的混同クラスタ（双方向混同が起きているペア）")
    print("  仮説: 意味的に類似・上位/下位関係のラベル間で双方向に混同が発生している")
    print()
    clusters = find_confusion_clusters(intent_pairs, min_count=2)
    if clusters:
        rows_table = []
        for c in clusters[:args.top]:
            direction = "<->" if c["bidirectional"] else " -> "
            scope = "同一Scenario" if c["same_scenario"] else "異Scenario"
            rows_table.append([
                c["label_a"], direction, c["label_b"],
                c["total"], f"{c['a_to_b']}", f"{c['b_to_a']}", scope,
            ])
        print_table(["Label A", "", "Label B", "Total", "A->B", "B->A", "Scope"], rows_table)
    else:
        print("  (双方向混同なし)")

    # ================================================================
    # H. Cross-scenario absorption
    # ================================================================
    print_section("H. Scenario横断の吸収分析")
    print("  同じ Action が複数の Scenario に存在する場合、別 Scenario に吸収されている可能性")
    print()
    # Build scenario->action from gold
    scenario_actions: Dict[str, Set[str]] = defaultdict(set)
    for _, gold in matched:
        s, a = gold["scenario"], gold["action"]
        if s and a:
            scenario_actions[s].add(a)

    confusion = Counter()
    for gi, pi in intent_pairs:
        if gi != pi:
            confusion[(gi, pi)] += 1

    cross_rows = []
    for (gi, pi), cnt in confusion.most_common(args.top * 3):
        gp = gi.split("_", 1)
        pp = pi.split("_", 1)
        gs, ga = (gp[0], gp[1]) if len(gp) > 1 else (gp[0], "")
        ps, pa = (pp[0], pp[1]) if len(pp) > 1 else (pp[0], "")
        shared_action = (ga == pa and gs != ps)
        within_scenario = (gs == ps and ga != pa)
        tag = ""
        if shared_action:
            tag = "Action共有->吸収"
        elif within_scenario:
            tag = "Scenario内混同"
        cross_rows.append([gi, "->", pi, cnt, tag])
        if len(cross_rows) >= args.top:
            break
    if cross_rows:
        print_table(["Gold", "", "Pred", "Count", "Type"], cross_rows)
    else:
        print("  (該当なし)")

    # ================================================================
    # I. Entity Type Confusion
    # ================================================================
    print_section("I. Entity Type エラー分析")
    type_fp, type_fn, type_swap = entity_type_confusion(matched)

    print("\n  --- False Positive が多い Entity Type (過剰予測) ---")
    if type_fp:
        rows_table = [[t, c] for t, c in type_fp.most_common(args.top)]
        print_table(["Entity Type", "FP Count"], rows_table)
    else:
        print("  (なし)")

    print("\n  --- False Negative が多い Entity Type (見逃し) ---")
    if type_fn:
        rows_table = [[t, c] for t, c in type_fn.most_common(args.top)]
        print_table(["Entity Type", "FN Count"], rows_table)
    else:
        print("  (なし)")

    print("\n  --- Entity Type の入れ替え（同一 filler で type が異なる） ---")
    print("  仮説: 意味的に近い type ラベル間で混同が発生している")
    if type_swap:
        rows_table = [[g, "->", p, c] for (g, p), c in type_swap.most_common(args.top)]
        print_table(["Gold Type", "", "Pred Type", "Count"], rows_table)
    else:
        print("  (なし)")

    # ================================================================
    # J. Scenario 別 Action 混同率
    # ================================================================
    print_section("J. Scenario 別 Action 混同率")
    scenario_error_detail: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    scenario_total_detail: Dict[str, int] = defaultdict(int)
    for gi, pi in intent_pairs:
        gp = gi.split("_", 1)
        gs = gp[0] if len(gp) > 0 else ""
        scenario_total_detail[gs] += 1
        if gi != pi:
            scenario_error_detail[gs][pi] += 1

    rows_table = []
    for s in sorted(scenario_total_detail.keys()):
        total_s = scenario_total_detail[s]
        err_s = sum(scenario_error_detail[s].values())
        acc_s = 1.0 - err_s / total_s if total_s > 0 else 1.0
        top_mistake = ""
        if scenario_error_detail[s]:
            top_pred, top_cnt = max(scenario_error_detail[s].items(), key=lambda x: x[1])
            top_mistake = f"{top_pred} ({top_cnt})"
        rows_table.append([s, total_s, err_s, f"{acc_s:.4f}", top_mistake])
    print_table(["Scenario", "Total", "Errors", "Acc", "Most Common Mistake"], rows_table)

    # ================================================================
    # K. Summary JSON
    # ================================================================
    summary = {
        "prediction_file": prediction_path,
        "gold_source": gold_source,
        "total_matched": total,
        "metrics": {
            "scenario": {"accuracy": scenario_metric.accuracy, "precision": sp, "recall": sr, "f1": sf},
            "action": {"accuracy": action_metric.accuracy, "precision": ap, "recall": ar, "f1": af},
            "intent": {"accuracy": intent_metric.accuracy, "precision": ip, "recall": ir, "f1": if1},
            "entity_span_f1": {"precision": ep, "recall": er, "f1": ef},
        },
        "error_breakdown": breakdown,
        "top_intent_confusions": [
            {"gold": gi, "pred": pi, "count": c} for (gi, pi), c in top_pairs
        ],
        "confusion_clusters": clusters[:args.top],
    }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n  Summary JSON saved to: {args.output_json}")

    print_section("Done")
    print(f"  Intent Accuracy: {intent_metric.accuracy:.4f}")
    print(f"  Entity Span-F1:  {ef:.4f}")
    print()


if __name__ == "__main__":
    main()
