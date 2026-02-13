#!/usr/bin/env python3
"""
07_plot_wer_metrics.py
=====================
WER 別の SLU 性能を折れ線グラフで可視化するスクリプト。
06_analyze_predictions.py の gold 読み込み・metric 計算ロジックを再利用。

Usage:
    python 07_plot_wer_metrics.py pred_a.jsonl pred_b.jsonl \\
        --gold test.jsonl --names "Model A,Model B"

Output:
    Experiment_RationaleCompare/figure/wer_metrics.pdf
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Import from 06 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.util import spec_from_file_location, module_from_spec as _mfs  # noqa: E402

_spec = spec_from_file_location(
    "_analyze", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "06_analyze_predictions.py"))
_mod = _mfs(_spec)
_spec.loader.exec_module(_mod)

# Re-export needed symbols
load_gold_from_test_jsonl = _mod.load_gold_from_test_jsonl
load_gold_by_slurp_id = _mod.load_gold_by_slurp_id
read_jsonl = _mod.read_jsonl
match_predictions_with_gold = _mod.match_predictions_with_gold
FMeasureAccumulator = _mod.FMeasureAccumulator
SpanFMeasureAccumulator = _mod.SpanFMeasureAccumulator
process_model = _mod.process_model
resolve_prediction_path = _mod.resolve_prediction_path
derive_model_name = _mod.derive_model_name
_auto_detect_gold_path = _mod._auto_detect_gold_path
ModelResult = _mod.ModelResult

# ============================================================================
# Plot style — publication quality
# ============================================================================

# Okabe-Ito colorblind-friendly palette
COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

LINE_STYLES = ["-", "--", "-.", ":"]


def setup_rcparams():
    """Set matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        # Axes
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        # Grid
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        # Ticks
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # Legend
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Math text
        "mathtext.fontset": "stix",
    })


# ============================================================================
# WER bins
# ============================================================================

WerBin = Tuple[float, float, str, bool]  # lo, hi, label, include_hi


def _format_pct_text(value_01: float) -> str:
    text = f"{value_01 * 100.0:.1f}"
    return text[:-2] if text.endswith(".0") else text


def build_wer_bins(
    bin_width_pct: int = 10,
    max_wer: float = 1.0,
    include_overflow: bool = False,
) -> List[WerBin]:
    """WER ビンを作成する。デフォルトは 0-10, ... ,90-100。"""
    if bin_width_pct <= 0:
        raise ValueError("--bin-width must be > 0")
    if max_wer <= 0:
        raise ValueError("--max-wer must be > 0")

    max_pct = int(round(max_wer * 100.0))
    if max_pct <= 0:
        raise ValueError("--max-wer is too small")

    bins: List[WerBin] = []
    lo_pct = 0
    while lo_pct < max_pct:
        hi_pct = min(lo_pct + bin_width_pct, max_pct)
        # Last finite bin includes hi (e.g., 90-100 includes 100%)
        include_hi = hi_pct >= max_pct
        bins.append((lo_pct / 100.0, hi_pct / 100.0, f"{lo_pct}-{hi_pct}", include_hi))
        lo_pct = hi_pct

    if include_overflow:
        bins.append((max_wer, float("inf"), f">{max_pct}", False))

    return bins


def _quantile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="linear"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="linear"))


def build_wer_bins_equal_count(
    models: List[ModelResult],
    n_bins: int = 10,
    max_wer: float = 1.0,
    include_overflow: bool = False,
) -> List[WerBin]:
    """WER の等頻度ビンを作成（各ビンの母数をほぼ均等化）。"""
    if n_bins <= 0:
        raise ValueError("--n-bins must be > 0 for equal_count binning")
    if max_wer <= 0:
        raise ValueError("--max-wer must be > 0")

    main_values: List[float] = []
    has_overflow = False
    for m in models:
        for w in m.wer_list:
            if w is None:
                continue
            wf = float(w)
            if wf <= max_wer:
                main_values.append(wf)
            elif include_overflow:
                has_overflow = True

    if not main_values:
        if include_overflow and has_overflow:
            return [(max_wer, float("inf"), f">{_format_pct_text(max_wer)}", False)]
        raise ValueError("No WER values available for equal_count binning.")

    values = np.asarray(main_values, dtype=np.float64)
    n_bins = min(n_bins, int(values.shape[0]))

    edges = [_quantile(values, i / n_bins) for i in range(n_bins + 1)]
    edges[0] = 0.0
    edges[-1] = max_wer
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]

    bins: List[WerBin] = []
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        include_hi = (i == n_bins - 1)
        bins.append((lo, hi, f"{_format_pct_text(lo)}-{_format_pct_text(hi)}", include_hi))

    if include_overflow and has_overflow:
        bins.append((max_wer, float("inf"), f">{_format_pct_text(max_wer)}", False))

    return bins


def build_wer_bins_log(
    models: List[ModelResult],
    n_bins: int = 10,
    max_wer: float = 1.0,
    include_overflow: bool = False,
    min_positive_wer: float = 1e-4,
) -> List[WerBin]:
    """WER の対数幅ビンを作成（低WERを細かく、高WERを粗く）。"""
    if n_bins <= 1:
        raise ValueError("--n-bins must be >= 2 for log binning")
    if max_wer <= 0:
        raise ValueError("--max-wer must be > 0")
    if min_positive_wer <= 0:
        raise ValueError("--min-positive-wer must be > 0")

    positive_values: List[float] = []
    has_overflow = False
    for m in models:
        for w in m.wer_list:
            if w is None:
                continue
            wf = float(w)
            if wf > max_wer:
                has_overflow = True
                continue
            if wf > 0.0:
                positive_values.append(wf)

    if not positive_values:
        # 全て 0 か overflow の場合
        bins: List[WerBin] = [(0.0, max_wer, f"0-{_format_pct_text(max_wer)}", True)]
        if include_overflow and has_overflow:
            bins.append((max_wer, float("inf"), f">{_format_pct_text(max_wer)}", False))
        return bins

    start = max(min_positive_wer, min(positive_values))
    if start >= max_wer:
        bins = [(0.0, max_wer, f"0-{_format_pct_text(max_wer)}", True)]
        if include_overflow and has_overflow:
            bins.append((max_wer, float("inf"), f">{_format_pct_text(max_wer)}", False))
        return bins

    # one linear bin for zero-heavy region + log-spaced bins for positive region
    n_log_bins = max(1, n_bins - 1)
    edges = np.geomspace(start, max_wer, num=n_log_bins + 1)

    bins = [(0.0, float(edges[0]), f"0-{_format_pct_text(float(edges[0]))}", False)]
    for i in range(n_log_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if hi <= lo:
            continue
        include_hi = (i == n_log_bins - 1)
        bins.append((lo, hi, f"{_format_pct_text(lo)}-{_format_pct_text(hi)}", include_hi))

    if include_overflow and has_overflow:
        bins.append((max_wer, float("inf"), f">{_format_pct_text(max_wer)}", False))
    return bins


def _in_wer_bin(w: float, lo: float, hi: float, include_hi: bool) -> bool:
    """WER ビン包含判定。overflow ビンは hi=inf で表現。"""
    if hi == float("inf"):
        return w > lo
    if include_hi:
        return lo <= w <= hi
    return lo <= w < hi


# ============================================================================
# Per-bin metric computation
# ============================================================================

def compute_bin_metrics(
    model: ModelResult,
    lo: float,
    hi: float,
    include_hi: bool,
) -> Dict[str, Any]:
    """指定 WER 範囲のサンプルで Scenario/Action/Intent Acc, SLU-F1 を算出。"""
    scen_m = FMeasureAccumulator()
    act_m = FMeasureAccumulator()
    int_m = FMeasureAccumulator()
    ent_m = SpanFMeasureAccumulator()
    n = 0
    for i, w in enumerate(model.wer_list):
        if w is None:
            continue
        if not _in_wer_bin(w, lo, hi, include_hi):
            continue
        gi, pi = model.intent_pairs[i]
        gs, ga = gi.split("_", 1)
        ps, pa = pi.split("_", 1)
        scen_m.add(gs, ps)
        act_m.add(ga, pa)
        int_m.add(gi, pi)
        pred, gold = model.matched[i]
        ent_m.add(gold["entities"], pred["entities"])
        n += 1
    _, _, ef = ent_m.overall()
    return {
        "n": n,
        "scenario_acc": scen_m.accuracy if n > 0 else None,
        "action_acc": act_m.accuracy if n > 0 else None,
        "intent_acc": int_m.accuracy if n > 0 else None,
        "slu_f1": ef if n > 0 else None,
    }


# ============================================================================
# Plotting
# ============================================================================

METRIC_KEYS = ["scenario_acc", "action_acc", "intent_acc", "slu_f1"]
METRIC_LABELS = ["Scenario Accuracy", "Action Accuracy", "Intent Accuracy", "SLU-F1"]

MIN_SAMPLES = 5  # ビン内サンプル数がこれ未満なら非表示


def _bin_center_percent(lo: float, hi: float, bin_width_pct: int) -> float:
    if hi == float("inf"):
        return lo * 100.0 + (bin_width_pct / 2.0)
    return (lo + hi) / 2.0 * 100.0


def estimate_bin_width_pct(wer_bins: List[WerBin], fallback: int = 10) -> int:
    widths = [
        (hi - lo) * 100.0
        for lo, hi, _, _ in wer_bins
        if hi != float("inf") and hi >= lo
    ]
    if not widths:
        return fallback
    mean_w = sum(widths) / len(widths)
    return max(1, int(round(mean_w)))


def _x_transform(value: float, mode: str) -> float:
    if mode == "log1p":
        return float(np.log1p(max(0.0, value)))
    return value


def _set_x_axis(
    ax: Any,
    x_scale: str,
    x_max_linear: float,
    x_tick_step_pct: Optional[int],
) -> None:
    if x_scale == "log1p":
        ax.set_xlim(_x_transform(0.0, x_scale) - 0.02, _x_transform(x_max_linear, x_scale) + 0.02)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{max(0.0, np.expm1(v)):.0f}")
        )
        return

    ax.set_xlim(-2, x_max_linear + 2.0)
    if x_tick_step_pct is not None and x_tick_step_pct > 0:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_step_pct))
    else:
        ax.xaxis.set_major_locator(mticker.AutoLocator())


def export_bin_metrics_csv(
    models: List[ModelResult],
    wer_bins: List[WerBin],
    model_data: List[List[Dict[str, Any]]],
    csv_path: str,
) -> None:
    """モデル×WERビンの集計を CSV で保存する。"""
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "model",
        "bin_label",
        "bin_lo",
        "bin_hi",
        "n",
        "scenario_acc",
        "action_acc",
        "intent_acc",
        "slu_f1",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m_idx, m in enumerate(models):
            for b_idx, (lo, hi, label, _) in enumerate(wer_bins):
                metrics = model_data[m_idx][b_idx]["metrics"]
                hi_str = "inf" if hi == float("inf") else f"{hi:.4f}"
                row = {
                    "model": m.name,
                    "bin_label": label,
                    "bin_lo": f"{lo:.4f}",
                    "bin_hi": hi_str,
                    "n": metrics["n"],
                    "scenario_acc": "" if metrics["scenario_acc"] is None else f"{metrics['scenario_acc']:.6f}",
                    "action_acc": "" if metrics["action_acc"] is None else f"{metrics['action_acc']:.6f}",
                    "intent_acc": "" if metrics["intent_acc"] is None else f"{metrics['intent_acc']:.6f}",
                    "slu_f1": "" if metrics["slu_f1"] is None else f"{metrics['slu_f1']:.6f}",
                }
                writer.writerow(row)
    print(f"Saved: {csv_path}")


def plot_wer_metrics(
    models: List[ModelResult],
    output_path: str,
    wer_bins: List[WerBin],
    bin_width_pct: int = 10,
    x_tick_step_pct: Optional[int] = 10,
    x_scale: str = "linear",
    min_samples: int = MIN_SAMPLES,
    csv_path: Optional[str] = None,
):
    """2x2 subplot の折れ線グラフを作成。"""
    setup_rcparams()

    # --- Compute all data ---
    # model_data[model_idx] = list of {"center","label","metrics"} per bin
    model_data: List[List[Dict[str, Any]]] = []
    for m in models:
        bins_metrics: List[Dict[str, Any]] = []
        for lo, hi, label, include_hi in wer_bins:
            d = compute_bin_metrics(m, lo, hi, include_hi)
            center = _bin_center_percent(lo, hi, bin_width_pct)
            bins_metrics.append({"center": center, "label": label, "metrics": d})
        model_data.append(bins_metrics)

    # --- Create figure ---
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), sharex=True)
    axes_flat = axes.flatten()

    for ax_idx, (metric_key, metric_label) in enumerate(
        zip(METRIC_KEYS, METRIC_LABELS)
    ):
        ax = axes_flat[ax_idx]

        for m_idx, m in enumerate(models):
            xs, ys = [], []
            for point in model_data[m_idx]:
                center = point["center"]
                d = point["metrics"]
                if d["n"] < min_samples:
                    continue
                val = d[metric_key]
                if val is None:
                    continue
                xs.append(_x_transform(center, x_scale))
                ys.append(val * 100)  # convert to percentage

            color = COLORS[m_idx % len(COLORS)]
            marker = MARKERS[m_idx % len(MARKERS)]
            ls = LINE_STYLES[m_idx % len(LINE_STYLES)]
            ax.plot(
                xs, ys,
                color=color, marker=marker, linestyle=ls,
                label=m.name,
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=3,
            )

        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))

        # x-axis
        max_center = max(
            point["center"] for per_model in model_data for point in per_model
        )
        x_max = max_center + (bin_width_pct / 2.0)
        _set_x_axis(ax, x_scale=x_scale, x_max_linear=x_max, x_tick_step_pct=x_tick_step_pct)

        # Only bottom row gets x-labels
        if ax_idx >= 2:
            if x_scale == "log1p":
                ax.set_xlabel("WER (%) [log1p scale]")
            else:
                ax.set_xlabel("WER (%)")

        # Subtle grid
        ax.grid(True, which="major", axis="y", linewidth=0.5, alpha=0.3)
        ax.grid(True, which="minor", axis="y", linewidth=0.3, alpha=0.15)
        ax.grid(True, which="major", axis="x", linewidth=0.3, alpha=0.15)

        # Subplot title
        ax.set_title(metric_label, fontsize=10, fontweight="bold", pad=4)

    # Shared legend at top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=min(len(models), 4),
            frameon=True,
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
        )

    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # --- Add sample count annotation below bottom-left subplot ---
    valid_bins = []
    for b_idx in range(len(wer_bins)):
        counts = [model_data[m_idx][b_idx]["metrics"]["n"] for m_idx in range(len(models))]
        n_max = max(counts)
        if n_max < min_samples:
            continue
        label = f"n={counts[0]}" if len(set(counts)) == 1 else f"nmax={n_max}"
        center = model_data[0][b_idx]["center"]
        valid_bins.append((_x_transform(center, x_scale), label))
    if valid_bins:
        ax_bottom = axes_flat[2]
        for center, n_label in valid_bins:
            ax_bottom.annotate(
                n_label, xy=(center, 0), xytext=(0, -28),
                textcoords="offset points", ha="center", va="top",
                fontsize=6.5, color="0.45",
            )

    # Save
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # Also save the other format (PDF <-> PNG)
    base, ext = os.path.splitext(output_path)
    other_ext = ".png" if ext.lower() == ".pdf" else ".pdf"
    other_path = base + other_ext
    fig.savefig(other_path, bbox_inches="tight")
    print(f"Saved: {other_path}")
    plt.close(fig)

    if csv_path:
        export_bin_metrics_csv(models, wer_bins, model_data, csv_path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WER 別 SLU 性能の折れ線グラフを作成"
    )
    parser.add_argument("predictions", nargs="+", type=str,
                        help="prediction.jsonl のパス (複数指定で比較)")
    parser.add_argument("--gold", type=str, default=None,
                        help="gold test.jsonl のパス")
    parser.add_argument("--key", type=str, default="file",
                        help="マッチングキー (default: file)")
    parser.add_argument("--names", type=str, default=None,
                        help="モデル名 (カンマ区切り)")
    parser.add_argument("--output", type=str, default=None,
                        help="出力先 (default: figure/wer_metrics.pdf)")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                        help=f"ビン内最小サンプル数 (default: {MIN_SAMPLES})")
    parser.add_argument("--bin-width", type=int, default=10,
                        help="WER ビン幅 (%%, default: 10)")
    parser.add_argument("--binning", type=str, default="fixed",
                        choices=["fixed", "equal_count", "log"],
                        help="ビン生成方式: fixed(固定幅), equal_count(等頻度), log(対数幅)")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="equal_count/log 時のビン数 (default: 10)")
    parser.add_argument("--min-positive-wer", type=float, default=1e-4,
                        help="log ビンの最小正WER (default: 1e-4)")
    parser.add_argument("--x-scale", type=str, default="linear",
                        choices=["linear", "log1p"],
                        help="横軸表示スケール (default: linear)")
    parser.add_argument("--max-wer", type=float, default=1.0,
                        help="有限ビンの最大 WER (default: 1.0)")
    parser.add_argument("--include-overflow", action="store_true",
                        help="max-wer より大きい WER 用の overflow ビンを追加")
    parser.add_argument("--csv", type=str, default=None,
                        help="ビン集計を CSV 保存")
    args = parser.parse_args()

    if args.min_samples <= 0:
        print("ERROR: --min-samples must be > 0", file=sys.stderr)
        sys.exit(1)

    # --- Resolve paths ---
    pred_paths = [resolve_prediction_path(p) for p in args.predictions]

    # --- Model names ---
    if args.names:
        names = [n.strip() for n in args.names.split(",")]
        if len(names) != len(pred_paths):
            print(f"ERROR: --names count ({len(names)}) != files ({len(pred_paths)})",
                  file=sys.stderr)
            sys.exit(1)
    else:
        names = [derive_model_name(p) for p in pred_paths]
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
        print(f"Loading gold: {gold_path}")
        if args.key == "slurp_id":
            gold_map = load_gold_by_slurp_id(gold_path)
        else:
            gold_map = load_gold_from_test_jsonl(gold_path)
        print(f"  Gold entries: {len(gold_map)}")
    else:
        print("  No --gold; using embedded target_label / target")

    # --- Process models ---
    models: List[ModelResult] = []
    for path, name in zip(pred_paths, names):
        print(f"Processing: {name}  ({path})")
        r = process_model(path, name, gold_map, args.key)
        has_wer = sum(1 for w in r.wer_list if w is not None)
        print(f"  Matched: {r.n_matched}  WER available: {has_wer}")
        models.append(r)

    if not models or all(m.n_matched == 0 for m in models):
        print("ERROR: No matched data.", file=sys.stderr)
        sys.exit(1)

    # Check WER availability
    total_wer = sum(1 for m in models for w in m.wer_list if w is not None)
    if total_wer == 0:
        print("ERROR: No WER data found. Provide --gold with test.jsonl.",
              file=sys.stderr)
        sys.exit(1)

    try:
        if args.binning == "equal_count":
            wer_bins = build_wer_bins_equal_count(
                models=models,
                n_bins=args.n_bins,
                max_wer=args.max_wer,
                include_overflow=args.include_overflow,
            )
            plot_bin_width = estimate_bin_width_pct(wer_bins, fallback=args.bin_width)
            x_tick_step = None
        elif args.binning == "log":
            wer_bins = build_wer_bins_log(
                models=models,
                n_bins=args.n_bins,
                max_wer=args.max_wer,
                include_overflow=args.include_overflow,
                min_positive_wer=args.min_positive_wer,
            )
            plot_bin_width = estimate_bin_width_pct(wer_bins, fallback=args.bin_width)
            x_tick_step = None
        else:
            wer_bins = build_wer_bins(
                bin_width_pct=args.bin_width,
                max_wer=args.max_wer,
                include_overflow=args.include_overflow,
            )
            plot_bin_width = args.bin_width
            x_tick_step = args.bin_width
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Output path ---
    if args.output:
        out = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(script_dir, "figure", "wer_metrics.pdf")

    plot_wer_metrics(
        models,
        out,
        wer_bins=wer_bins,
        bin_width_pct=plot_bin_width,
        x_tick_step_pct=x_tick_step,
        x_scale=args.x_scale,
        min_samples=args.min_samples,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()
