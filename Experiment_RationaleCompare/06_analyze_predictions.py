#!/usr/bin/env python3
"""
06_analyze_predictions.py
=========================
prediction.jsonl を分析するスクリプト。
evaluate.py の基準に従って Intent Accuracy / Entity Span-F1 を算出し、
さらに意味的類似・上位包含関係に着目したエラー分析を行う。

Usage:
    python 06_analyze_predictions.py <prediction.jsonl> [--gold <test.jsonl>] [--top N]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

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


def _extract_pred_gold(row: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """Extract (pred, gold) label dicts from a prediction row."""
    pred_scenario = str(row.get("scenario", "")).strip()
    pred_action = str(row.get("action", "")).strip()
    pred_entities = row.get("entities", []) or []

    target_label = row.get("target_label")
    if not isinstance(target_label, dict):
        try:
            target_label = json.loads(row.get("target", "{}"))
        except Exception:
            target_label = {}
    if not isinstance(target_label, dict):
        target_label = {}

    gold_scenario = str(target_label.get("scenario", "")).strip()
    gold_action = str(target_label.get("action", "")).strip()
    gold_entities = target_label.get("entities", []) or []

    pred = {"scenario": pred_scenario, "action": pred_action, "entities": pred_entities}
    gold = {"scenario": gold_scenario, "action": gold_action, "entities": gold_entities}
    return pred, gold


# ---------------------------------------------------------------------------
# 2. Metrics (evaluate.py 準拠)
# ---------------------------------------------------------------------------

def compute_prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class FMeasureAccumulator:
    """evaluate.py の FMeasure と同じ: 完全一致でTP/FP/FN を蓄積。"""
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


# ---------------------------------------------------------------------------
# 3. Error Analysis helpers
# ---------------------------------------------------------------------------

def build_confusion_counter(pairs: List[Tuple[str, str]]) -> Counter:
    """(gold, pred) ペアから混同カウンタを構築。"""
    return Counter(pairs)


def top_confusions(confusion: Counter, top_n: int = 20) -> List[Tuple[Tuple[str, str], int]]:
    """誤りのみ (gold != pred) のペアを頻度順に返す。"""
    errors = {k: v for k, v in confusion.items() if k[0] != k[1]}
    return sorted(errors.items(), key=lambda x: -x[1])[:top_n]


def error_type_breakdown(
    intent_pairs: List[Tuple[str, str]],
) -> Dict[str, int]:
    """Intent エラーを分類:
    - same_scenario_diff_action: scenario は一致、action が不一致
    - diff_scenario_same_action: scenario が不一致、action は一致
    - diff_both: scenario も action も不一致
    """
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
    rows: List[Dict[str, Any]],
) -> Tuple[Counter, Counter, Counter]:
    """Entity レベルのエラーを分析。
    Returns:
        type_fp_counter: 予測したが gold になかった entity type
        type_fn_counter: gold にあったが予測できなかった entity type
        type_swap_counter: (gold_type, pred_type) のペア (filler が一致するが type が異なるもの)
    """
    type_fp = Counter()
    type_fn = Counter()
    type_swap = Counter()

    for row in rows:
        pred, gold = _extract_pred_gold(row)
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


def scenario_action_cooccurrence(
    rows: List[Dict[str, Any]],
) -> Dict[str, Set[str]]:
    """gold データから scenario -> action の対応表を作成。"""
    mapping: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        _, gold = _extract_pred_gold(row)
        s = gold["scenario"]
        a = gold["action"]
        if s and a:
            mapping[s].add(a)
    return dict(mapping)


def cross_scenario_action_confusion(
    intent_pairs: List[Tuple[str, str]],
    scenario_actions: Dict[str, Set[str]],
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """scenario をまたいだ action 混同を分析。
    同じ action が複数 scenario に存在する場合、
    別 scenario の同 action に吸収されている可能性を検出。"""
    results = []
    confusion = Counter()
    for gold_intent, pred_intent in intent_pairs:
        if gold_intent == pred_intent:
            continue
        confusion[(gold_intent, pred_intent)] += 1

    for (gi, pi), cnt in confusion.most_common(top_n * 3):
        gp = gi.split("_", 1)
        pp = pi.split("_", 1)
        gs, ga = (gp[0], gp[1]) if len(gp) > 1 else (gp[0], "")
        ps, pa = (pp[0], pp[1]) if len(pp) > 1 else (pp[0], "")

        # 同じ action が別の scenario にも存在するか？
        shared_action = (ga == pa and gs != ps)
        # 同じ scenario 内で action を間違えているか？
        within_scenario = (gs == ps and ga != pa)

        results.append({
            "gold": gi, "pred": pi, "count": cnt,
            "shared_action_across_scenarios": shared_action,
            "within_scenario_confusion": within_scenario,
            "gold_scenario": gs, "gold_action": ga,
            "pred_scenario": ps, "pred_action": pa,
        })
        if len(results) >= top_n:
            break
    return results


# ---------------------------------------------------------------------------
# 4. Report formatting
# ---------------------------------------------------------------------------

def fmt_prf(p: float, r: float, f: float) -> str:
    return f"P={p:.4f}  R={r:.4f}  F1={f:.4f}"


def print_section(title: str, width: int = 72):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_table(headers: List[str], rows: List[List[Any]], max_rows: int = 50):
    """シンプルなテーブル表示。"""
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


# ---------------------------------------------------------------------------
# 5. Per-label F1 breakdown
# ---------------------------------------------------------------------------

def per_label_f1(acc: FMeasureAccumulator, label_name: str, top_n: int = 20):
    """各ラベルの P/R/F1 と TP/FP/FN を表示。"""
    all_tags = sorted(set(acc.tp) | set(acc.fp) | set(acc.fn))
    rows = []
    for tag in all_tags:
        p, r, f = compute_prf(acc.tp[tag], acc.fp[tag], acc.fn[tag])
        total = acc.tp[tag] + acc.fn[tag]  # gold count
        rows.append([tag, f"{p:.4f}", f"{r:.4f}", f"{f:.4f}",
                      int(acc.tp[tag]), int(acc.fp[tag]), int(acc.fn[tag]), int(total)])
    # sort by F1 ascending (worst first)
    rows.sort(key=lambda x: float(x[3]))
    print(f"\n  --- {label_name}: Per-label metrics (worst F1 first, top {top_n}) ---")
    print_table(["Label", "Prec", "Recall", "F1", "TP", "FP", "FN", "Gold"],
                rows[:top_n])


def per_entity_type_f1(acc: SpanFMeasureAccumulator, top_n: int = 20):
    """Entity type ごとの P/R/F1。"""
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


# ---------------------------------------------------------------------------
# 6. Semantic / hierarchical clustering of confused labels
# ---------------------------------------------------------------------------

def find_confusion_clusters(
    intent_pairs: List[Tuple[str, str]],
    min_count: int = 2,
) -> List[Dict[str, Any]]:
    """混同が双方向で起きているペアを検出（意味的に近い可能性が高い）。"""
    pair_count: Counter = Counter()
    for g, p in intent_pairs:
        if g != p:
            key = tuple(sorted([g, p]))
            pair_count[key] += 1

    clusters = []
    for (a, b), cnt in pair_count.most_common():
        if cnt < min_count:
            break
        # a → b と b → a をそれぞれ集計
        a_to_b = sum(1 for g, p in intent_pairs if g == a and p == b)
        b_to_a = sum(1 for g, p in intent_pairs if g == b and p == a)
        bidirectional = a_to_b > 0 and b_to_a > 0

        a_parts = a.split("_", 1)
        b_parts = b.split("_", 1)
        same_scenario = (a_parts[0] == b_parts[0]) if len(a_parts) > 1 and len(b_parts) > 1 else False

        clusters.append({
            "label_a": a,
            "label_b": b,
            "total": cnt,
            "a_to_b": a_to_b,
            "b_to_a": b_to_a,
            "bidirectional": bidirectional,
            "same_scenario": same_scenario,
        })
    return clusters


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="prediction.jsonl のエラー分析（evaluate.py 準拠の指標 + 意味的混同分析）"
    )
    parser.add_argument("prediction", type=str,
                        help="prediction.jsonl のパス")
    parser.add_argument("--gold", type=str, default=None,
                        help="gold test.jsonl のパス（指定時は evaluate.py と同一方式で評価）")
    parser.add_argument("--top", type=int, default=20,
                        help="表示する上位件数 (default: 20)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="分析結果を JSON で保存するパス")
    args = parser.parse_args()

    prediction_path = args.prediction
    if os.path.isdir(prediction_path):
        candidate = os.path.join(prediction_path, "prediction.jsonl")
        if os.path.exists(candidate):
            prediction_path = candidate
        else:
            print(f"ERROR: {prediction_path} is a directory but prediction.jsonl not found inside.", file=sys.stderr)
            sys.exit(1)

    if not os.path.exists(prediction_path):
        print(f"ERROR: {prediction_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading predictions: {prediction_path}")
    rows = read_jsonl(prediction_path)
    print(f"  Loaded {len(rows)} samples")

    if not rows:
        print("No data to analyze.", file=sys.stderr)
        sys.exit(1)

    # ---- Accumulate metrics ----
    scenario_metric = FMeasureAccumulator()
    action_metric = FMeasureAccumulator()
    intent_metric = FMeasureAccumulator()
    entity_metric = SpanFMeasureAccumulator()

    intent_pairs: List[Tuple[str, str]] = []  # (gold_intent, pred_intent)
    scenario_pairs: List[Tuple[str, str]] = []
    action_pairs: List[Tuple[str, str]] = []

    for row in rows:
        pred, gold = _extract_pred_gold(row)
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

    total = len(rows)

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

    print(f"\n  Total samples: {total}")

    # ================================================================
    # B. Per-label metrics (worst performers)
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
    print_section("D. 最も頻度の高い Intent 混同ペア (gold → pred)")
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
        rows_table.append([gi, "→", pi, cnt, relation])
    print_table(["Gold Intent", "", "Pred Intent", "Count", "Relation"], rows_table)

    # ================================================================
    # E. Scenario Confusion
    # ================================================================
    print_section("E. Scenario 混同 (gold → pred)")
    scenario_confusion = build_confusion_counter(scenario_pairs)
    top_sc = top_confusions(scenario_confusion, args.top)
    if top_sc:
        rows_table = [[g, "→", p, c] for (g, p), c in top_sc]
        print_table(["Gold Scenario", "", "Pred Scenario", "Count"], rows_table)
    else:
        print("  (Scenario 混同なし)")

    # ================================================================
    # F. Action Confusion
    # ================================================================
    print_section("F. Action 混同 (gold → pred)")
    action_confusion = build_confusion_counter(action_pairs)
    top_ac = top_confusions(action_confusion, args.top)
    if top_ac:
        rows_table = [[g, "→", p, c] for (g, p), c in top_ac]
        print_table(["Gold Action", "", "Pred Action", "Count"], rows_table)
    else:
        print("  (Action 混同なし)")

    # ================================================================
    # G. Semantic Confusion Clusters (双方向混同)
    # ================================================================
    print_section("G. 意味的混同クラスタ（双方向混同が起きているペア）")
    print("  仮説: 意味的に類似・上位/下位関係のラベル間で双方向に混同が発生している")
    print()
    clusters = find_confusion_clusters(intent_pairs, min_count=2)
    if clusters:
        rows_table = []
        for c in clusters[:args.top]:
            direction = "⇄" if c["bidirectional"] else "→"
            scope = "同一Scenario" if c["same_scenario"] else "異Scenario"
            rows_table.append([
                c["label_a"], direction, c["label_b"],
                c["total"], f"{c['a_to_b']}", f"{c['b_to_a']}",
                scope
            ])
        print_table(["Label A", "", "Label B", "Total", "A→B", "B→A", "Scope"], rows_table)
    else:
        print("  (双方向混同なし)")

    # ================================================================
    # H. Cross-scenario absorption analysis
    # ================================================================
    print_section("H. Scenario横断の吸収分析")
    print("  同じ Action が複数の Scenario に存在する場合、別 Scenario に吸収されている可能性")
    print()
    scenario_actions = scenario_action_cooccurrence(rows)
    cross = cross_scenario_action_confusion(intent_pairs, scenario_actions, args.top)
    if cross:
        rows_table = []
        for item in cross:
            tag = ""
            if item["shared_action_across_scenarios"]:
                tag = "Action共有→吸収"
            elif item["within_scenario_confusion"]:
                tag = "Scenario内混同"
            rows_table.append([
                item["gold"], "→", item["pred"], item["count"], tag
            ])
        print_table(["Gold", "", "Pred", "Count", "Type"], rows_table)
    else:
        print("  (該当なし)")

    # ================================================================
    # I. Entity Type Confusion
    # ================================================================
    print_section("I. Entity Type エラー分析")
    type_fp, type_fn, type_swap = entity_type_confusion(rows)

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
        rows_table = [[g, "→", p, c] for (g, p), c in type_swap.most_common(args.top)]
        print_table(["Gold Type", "", "Pred Type", "Count"], rows_table)
    else:
        print("  (なし)")

    # ================================================================
    # J. Scenario 内の Action 分布と混同率
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
    # K. Summary JSON output
    # ================================================================
    summary = {
        "prediction_file": prediction_path,
        "total_samples": total,
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
