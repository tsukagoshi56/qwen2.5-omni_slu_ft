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
# WER bins (10% increments)
# ============================================================================

def build_wer_bins_10() -> List[Tuple[float, float, str]]:
    """0-10, 10-20, ..., 90-100 の WER ビンを作成。"""
    bins = []
    for lo in range(0, 100, 10):
        hi = lo + 10
        bins.append((lo / 100.0, hi / 100.0, f"{lo}-{hi}"))
    return bins


def _in_wer_bin_10(w: float, lo: float, hi: float) -> bool:
    """[lo, hi) で判定。最後のビン [0.9, 1.0] は hi=1.0 を含む。"""
    if hi >= 1.0:
        return lo <= w <= hi
    return lo <= w < hi


# ============================================================================
# Per-bin metric computation
# ============================================================================

def compute_bin_metrics(
    model: ModelResult, lo: float, hi: float,
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
        if not _in_wer_bin_10(w, lo, hi):
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


def plot_wer_metrics(
    models: List[ModelResult],
    output_path: str,
    min_samples: int = MIN_SAMPLES,
):
    """2x2 subplot の折れ線グラフを作成。"""
    setup_rcparams()

    wer_bins = build_wer_bins_10()

    # --- Compute all data ---
    # model_data[model_idx] = list of (bin_center, metrics_dict) per bin
    model_data = []
    for m in models:
        bins_metrics = []
        for lo, hi, label in wer_bins:
            d = compute_bin_metrics(m, lo, hi)
            center = (lo + hi) / 2.0 * 100  # percent
            bins_metrics.append((center, d))
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
            for center, d in model_data[m_idx]:
                if d["n"] < min_samples:
                    continue
                val = d[metric_key]
                if val is None:
                    continue
                xs.append(center)
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
        ax.set_xlim(-2, 102)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

        # Only bottom row gets x-labels
        if ax_idx >= 2:
            ax.set_xlabel("WER (%)")

        # Subtle grid
        ax.grid(True, which="major", axis="y", linewidth=0.5, alpha=0.3)
        ax.grid(True, which="minor", axis="y", linewidth=0.3, alpha=0.15)
        ax.grid(True, which="major", axis="x", linewidth=0.3, alpha=0.15)

        # Subplot title
        ax.set_title(metric_label, fontsize=10, fontweight="bold", pad=4)

    # Shared legend at top
    handles, labels = axes_flat[0].get_legend_handles_labels()
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
    valid_bins = [(c, d) for c, d in model_data[0] if d["n"] >= min_samples]
    if valid_bins:
        ax_bottom = axes_flat[2]
        for center, d in valid_bins:
            ax_bottom.annotate(
                f"n={d['n']}", xy=(center, 0), xytext=(0, -28),
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
    args = parser.parse_args()

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

    # --- Output path ---
    if args.output:
        out = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(script_dir, "figure", "wer_metrics.pdf")

    plot_wer_metrics(models, out, min_samples=args.min_samples)


if __name__ == "__main__":
    main()
