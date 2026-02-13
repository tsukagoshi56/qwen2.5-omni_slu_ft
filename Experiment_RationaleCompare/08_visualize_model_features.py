#!/usr/bin/env python3
"""
08_visualize_model_features.py
==============================
Extracts prompt-level model features directly from a model checkpoint path
(not from prediction.jsonl), then visualizes intent clusters and separation.

Design goals:
- Reuse inference-side preprocessing in:
  - audio_text_mix_e2e_re.py (SFT pipeline)
  - audio_text_mix_e2e_re_multitask.py (multitask pipeline)
- Save reusable feature artifacts under analysis/ for later reruns.
- Provide centroid distance statistics to inspect intent boundary separation.

Examples:
    # SFT-style prompt (C/R/J), audio-only test behavior
    python 08_visualize_model_features.py \
      --model_name_or_path outputs/qwen_rationale_label_ft \
      --pipeline sft \
      --task-mode cot

    # Multitask label-only prompt feature extraction
    python 08_visualize_model_features.py \
      --model_name_or_path outputs/qwen_rationale_label_ft_multitask \
      --pipeline multitask \
      --task-mode label

    # Reuse already extracted features only
    python 08_visualize_model_features.py \
      --reuse-dir Experiment_RationaleCompare/analysis/model_feats_run_x
"""

import argparse
import csv
import datetime as dt
import importlib.util
import inspect
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return iterable


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_PATH = os.path.join(SCRIPT_DIR, "audio_text_mix_e2e_re.py")
MULTITASK_PATH = os.path.join(SCRIPT_DIR, "audio_text_mix_e2e_re_multitask.py")


def _load_module_from_path(name: str, path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sanitize_name(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "run"


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
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


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_gold_intent_maps_from_test_jsonl(
    path: str,
) -> Tuple[Dict[str, Tuple[str, str, str]], Dict[str, Tuple[str, str, str]]]:
    """Load gold intent labels from SLURP-style test.jsonl keyed by slurp_id and file."""
    by_slurp_id: Dict[str, Tuple[str, str, str]] = {}
    by_file: Dict[str, Tuple[str, str, str]] = {}
    if not path or not os.path.exists(path):
        return by_slurp_id, by_file

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            scenario = str(obj.get("scenario", "") or "").strip().lower()
            action = str(obj.get("action", "") or "").strip().lower()
            if not scenario and not action:
                intent_raw = str(obj.get("intent", "") or "").strip().lower()
                if "_" in intent_raw:
                    scenario, action = intent_raw.split("_", 1)
            if not scenario or not action:
                continue
            intent = f"{scenario}_{action}"
            label = (scenario, action, intent)

            sid = str(obj.get("slurp_id", "") or "").strip()
            if sid:
                by_slurp_id.setdefault(sid, label)

            recs = obj.get("recordings")
            if isinstance(recs, list):
                for rec in recs:
                    if not isinstance(rec, dict):
                        continue
                    fname = str(rec.get("file", "") or "").strip()
                    if not fname:
                        continue
                    by_file.setdefault(fname, label)
                    by_file.setdefault(os.path.basename(fname), label)
    return by_slurp_id, by_file


def _extract_unused_model_kwargs_from_exception(exc: Exception) -> List[str]:
    text = str(exc or "")
    found: List[str] = []
    match = re.search(r"not used by the model:\s*\[(.*?)\]", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        body = match.group(1)
        for part in body.split(","):
            token = part.strip().strip("\"'`")
            if token:
                found.append(token)
    if not found:
        match = re.search(
            r"got an unexpected keyword argument\s+['\"]([^'\"]+)['\"]",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            found.append(match.group(1).strip())
    if not found:
        match = re.search(
            r"['\"]([^'\"]+)['\"]\s+is\s+(?:an\s+)?unsupported\s+keyword\s+argument",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            found.append(match.group(1).strip())

    normalized: List[str] = []
    seen = set()
    for token in found:
        variants = [token, token.replace(" ", "_"), token.replace("_", " ")]
        for variant in variants:
            key = variant.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            normalized.append(variant.strip())
    return normalized


def _key_matches_unused(key: str, unused: Sequence[str]) -> bool:
    key_norm = key.replace("_", " ").strip().lower()
    for cand in unused:
        cand_norm = str(cand).replace("_", " ").strip().lower()
        if key_norm == cand_norm:
            return True
    return False


def _model_dtype(model: Any) -> Optional[torch.dtype]:
    try:
        dtype = getattr(model, "dtype", None)
    except Exception:
        dtype = None
    if isinstance(dtype, torch.dtype) and torch.is_floating_point(torch.empty((), dtype=dtype)):
        return dtype
    return None


def _cast_floating_tensors_to_model_dtype(inputs: Dict[str, torch.Tensor], model: Any) -> Dict[str, torch.Tensor]:
    target_dtype = _model_dtype(model)
    if target_dtype is None:
        return dict(inputs)
    out: Dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            out[k] = v.to(dtype=target_dtype)
        else:
            out[k] = v
    return out


def _filter_inputs_by_forward_signature(model: Any, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    target = model.module if hasattr(model, "module") and getattr(model, "module") is not None else model
    forward = getattr(target, "forward", None)
    if not callable(forward):
        return dict(inputs)
    try:
        sig = inspect.signature(forward)
    except Exception:
        return dict(inputs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(inputs)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in inputs.items() if k in accepted}


def _forward_with_retry(model: Any, kwargs: Dict[str, Any], max_retry: int = 6) -> Any:
    working = dict(kwargs)
    for _ in range(max_retry):
        try:
            with torch.no_grad():
                return model(**working)
        except Exception as exc:
            unused = _extract_unused_model_kwargs_from_exception(exc)
            if not unused:
                raise
            removed = []
            for key in list(working.keys()):
                if _key_matches_unused(key, unused):
                    working.pop(key, None)
                    removed.append(key)
            if not removed:
                raise
    raise RuntimeError("forward() retry limit exceeded while dropping unsupported kwargs.")


def _extract_label_from_json_obj(obj: Dict[str, Any]) -> Tuple[str, str]:
    scenario = str(obj.get("scenario", obj.get("Scenario", "")) or "").strip().lower()
    action = str(obj.get("action", obj.get("Action", "")) or "").strip().lower()
    if not scenario and not action:
        intent = str(obj.get("intent", obj.get("Intent", "")) or "").strip().lower()
        if "_" in intent:
            scenario, action = intent.split("_", 1)
    return scenario, action


def _parse_label_from_target_text(target_text: str) -> Tuple[str, str]:
    text = str(target_text or "")
    for line in reversed(text.splitlines()):
        s = line.strip()
        if not s.startswith("J:"):
            continue
        payload = s[2:].strip()
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return _extract_label_from_json_obj(obj)
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return "", ""
    if isinstance(obj, dict):
        return _extract_label_from_json_obj(obj)
    return "", ""


def infer_intent_from_item(
    item: Dict[str, Any],
    gold_intent_by_slurp_id: Optional[Dict[str, Tuple[str, str, str]]] = None,
    gold_intent_by_file: Optional[Dict[str, Tuple[str, str, str]]] = None,
) -> Tuple[str, str, str]:
    target_obj = item.get("target_obj")
    scenario = ""
    action = ""
    if isinstance(target_obj, dict):
        scenario, action = _extract_label_from_json_obj(target_obj)
    if not scenario and not action:
        scenario, action = _parse_label_from_target_text(item.get("target", ""))
    if (not scenario or not action) and isinstance(item.get("pred_label"), dict):
        s2, a2 = _extract_label_from_json_obj(item.get("pred_label"))
        scenario = scenario or s2
        action = action or a2
    if not scenario and "scenario" in item:
        scenario = str(item.get("scenario", "") or "").strip().lower()
    if not action and "action" in item:
        action = str(item.get("action", "") or "").strip().lower()

    # Fallback to external gold map when item lacks resolved label.
    if (not scenario or not action) and gold_intent_by_slurp_id:
        sid_candidates = [
            item.get("slurp_id"),
            item.get("id"),
        ]
        for sid in sid_candidates:
            key = str(sid or "").strip()
            if not key:
                continue
            if key in gold_intent_by_slurp_id:
                scenario, action, _ = gold_intent_by_slurp_id[key]
                break

    if (not scenario or not action) and gold_intent_by_file:
        file_value = str(item.get("file", "") or "").strip()
        file_candidates = [file_value, os.path.basename(file_value) if file_value else ""]
        for fkey in file_candidates:
            if not fkey:
                continue
            if fkey in gold_intent_by_file:
                scenario, action, _ = gold_intent_by_file[fkey]
                break

    intent = f"{scenario}_{action}" if scenario and action else "__unknown__"
    return scenario, action, intent


def _get_hidden_from_outputs(outputs: Any, layer_index: int) -> torch.Tensor:
    hidden = None
    hs = getattr(outputs, "hidden_states", None)
    if hs is not None and isinstance(hs, (list, tuple)) and len(hs) > 0:
        idx = layer_index if layer_index >= 0 else len(hs) + layer_index
        idx = max(0, min(len(hs) - 1, idx))
        hidden = hs[idx]
    if hidden is None:
        hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None and isinstance(outputs, (tuple, list)):
        for part in outputs:
            if torch.is_tensor(part) and part.dim() >= 2:
                hidden = part
                break
    if hidden is None or not torch.is_tensor(hidden):
        raise RuntimeError("No hidden states were found in forward outputs.")
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)
    return hidden


def pool_feature(
    hidden: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pooling: str,
) -> np.ndarray:
    if hidden.dim() != 3:
        raise ValueError(f"Expected hidden shape [B,T,D], got {tuple(hidden.shape)}")
    if hidden.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {hidden.shape[0]}")

    h = hidden[0]
    if attention_mask is None:
        valid = torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
    else:
        m = attention_mask
        if m.dim() == 2:
            m = m[0]
        valid = m.to(dtype=torch.bool, device=h.device)
        if valid.shape[0] != h.shape[0]:
            n = min(valid.shape[0], h.shape[0])
            valid = valid[:n]
            h = h[:n]

    idx = torch.nonzero(valid, as_tuple=False).view(-1)
    if idx.numel() == 0:
        idx = torch.tensor([h.shape[0] - 1], device=h.device)

    if pooling == "last":
        vec = h[idx[-1]]
    else:
        vec = h[idx].mean(dim=0)
    return vec.detach().float().cpu().numpy()


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def pca_project_2d(x: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), (0.0, 0.0)
    if x.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32), (0.0, 0.0)
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    comp = vt[:2]
    if comp.shape[0] < 2:
        pad = np.zeros((2 - comp.shape[0], comp.shape[1]), dtype=comp.dtype)
        comp = np.vstack([comp, pad])
    proj = x_centered @ comp.T
    denom = float(max(x.shape[0] - 1, 1))
    var = (s ** 2) / denom
    total = float(var.sum()) if var.size > 0 else 0.0
    if total > 0:
        r0 = float(var[0] / total) if var.size > 0 else 0.0
        r1 = float(var[1] / total) if var.size > 1 else 0.0
    else:
        r0, r1 = 0.0, 0.0
    return proj.astype(np.float32), (r0, r1)


def _pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(diff, axis=2).astype(np.float32)


def compute_intent_distance_stats(
    features: np.ndarray,
    intents: Sequence[str],
    min_intent_samples: int,
) -> Dict[str, Any]:
    counts = Counter(intents)
    valid_intents = [
        intent for intent, c in counts.items()
        if intent != "__unknown__" and c >= min_intent_samples
    ]
    valid_intents.sort(key=lambda x: (-counts[x], x))

    if not valid_intents:
        return {
            "valid_intents": [],
            "centroids": np.zeros((0, features.shape[1]), dtype=np.float32),
            "distance_matrix": np.zeros((0, 0), dtype=np.float32),
            "intent_rows": [],
            "summary": {
                "num_samples": int(features.shape[0]),
                "num_intents_total": len(counts),
                "num_intents_used": 0,
                "mean_intra_distance": None,
                "mean_inter_centroid_distance": None,
                "mean_nearest_centroid_distance": None,
                "separation_ratio": None,
            },
            "counts": counts,
        }

    intents_arr = np.asarray(intents)
    centroids: List[np.ndarray] = []
    intra_mean: Dict[str, float] = {}
    for intent in valid_intents:
        idxs = np.where(intents_arr == intent)[0]
        vecs = features[idxs]
        c = vecs.mean(axis=0)
        centroids.append(c)
        intra = float(np.linalg.norm(vecs - c[None, :], axis=1).mean()) if len(vecs) > 0 else 0.0
        intra_mean[intent] = intra

    centroid_arr = np.vstack(centroids).astype(np.float32)
    dist = _pairwise_euclidean(centroid_arr)

    rows: List[Dict[str, Any]] = []
    for i, intent in enumerate(valid_intents):
        if len(valid_intents) > 1:
            near = float(np.min(np.delete(dist[i], i)))
        else:
            near = float("nan")
        intra = float(intra_mean[intent])
        ratio = (near / intra) if (np.isfinite(near) and intra > 0) else None
        rows.append({
            "intent": intent,
            "count": int(counts[intent]),
            "intra_mean_distance": intra,
            "nearest_centroid_distance": near if np.isfinite(near) else None,
            "nearest_over_intra": ratio,
        })

    mean_intra = float(np.mean([r["intra_mean_distance"] for r in rows])) if rows else None
    if len(valid_intents) > 1:
        upper = dist[np.triu_indices(len(valid_intents), k=1)]
        mean_inter = float(upper.mean()) if upper.size > 0 else None
        near_vals = [r["nearest_centroid_distance"] for r in rows if r["nearest_centroid_distance"] is not None]
        mean_near = float(np.mean(near_vals)) if near_vals else None
    else:
        mean_inter = None
        mean_near = None

    sep = (mean_near / mean_intra) if (mean_near is not None and mean_intra and mean_intra > 0) else None
    summary = {
        "num_samples": int(features.shape[0]),
        "num_intents_total": len(counts),
        "num_intents_used": len(valid_intents),
        "mean_intra_distance": mean_intra,
        "mean_inter_centroid_distance": mean_inter,
        "mean_nearest_centroid_distance": mean_near,
        "separation_ratio": sep,
    }

    return {
        "valid_intents": valid_intents,
        "centroids": centroid_arr,
        "distance_matrix": dist,
        "intent_rows": rows,
        "summary": summary,
        "counts": counts,
    }


def _pick_top_intents(counts: Counter, top_k: int) -> List[str]:
    return [name for name, _ in counts.most_common(max(top_k, 1))]


def plot_pca_scatter(
    projection: np.ndarray,
    intents: Sequence[str],
    counts: Counter,
    out_path: str,
    title: str,
    explained_ratio: Tuple[float, float],
    top_k: int,
) -> None:
    _ensure_dir(os.path.dirname(out_path) or ".")
    top_intents = set(_pick_top_intents(counts, top_k))
    plot_labels = [x if x in top_intents else "other" for x in intents]
    unique = sorted(set(plot_labels), key=lambda x: (x == "other", -counts.get(x, 0), x))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(unique))]
    color_map = dict(zip(unique, colors))

    fig, ax = plt.subplots(figsize=(10, 7))
    for label in unique:
        idx = [i for i, l in enumerate(plot_labels) if l == label]
        if not idx:
            continue
        label_text = f"{label} (n={len(idx)})"
        ax.scatter(
            projection[idx, 0],
            projection[idx, 1],
            s=14,
            alpha=0.8,
            c=[color_map[label]],
            label=label_text,
            edgecolors="none",
        )

    ax.set_title(
        f"{title}\nPCA-2D (var: PC1={explained_ratio[0]*100:.1f}%, PC2={explained_ratio[1]*100:.1f}%)",
        fontsize=11,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if len(unique) <= 25:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_centroid_heatmap(
    valid_intents: Sequence[str],
    distance_matrix: np.ndarray,
    counts: Counter,
    out_path: str,
    top_k: int,
) -> None:
    if len(valid_intents) == 0 or distance_matrix.size == 0:
        return

    top_order = sorted(valid_intents, key=lambda x: (-counts.get(x, 0), x))[: max(top_k, 1)]
    idxs = [valid_intents.index(name) for name in top_order]
    sub = distance_matrix[np.ix_(idxs, idxs)]

    fig, ax = plt.subplots(figsize=(max(6, len(top_order) * 0.4), max(5, len(top_order) * 0.35)))
    im = ax.imshow(sub, cmap="viridis", interpolation="nearest")
    ax.set_xticks(np.arange(len(top_order)))
    ax.set_yticks(np.arange(len(top_order)))
    ax.set_xticklabels(top_order, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(top_order, fontsize=7)
    ax.set_title("Centroid Distance Heatmap (Euclidean)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("Distance", rotation=90)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_intent_stats_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    fieldnames = [
        "intent",
        "count",
        "intra_mean_distance",
        "nearest_centroid_distance",
        "nearest_over_intra",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_centroid_distance_csv(
    path: str,
    valid_intents: Sequence[str],
    dist: np.ndarray,
) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["intent_a", "intent_b", "distance"])
        for i, a in enumerate(valid_intents):
            for j in range(i + 1, len(valid_intents)):
                b = valid_intents[j]
                writer.writerow([a, b, float(dist[i, j])])


def _unpack_batch(batch: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, torch.Tensor]]]:
    if not batch:
        return []
    items = batch.get("items") or []
    if "net_inputs_list" in batch and isinstance(batch["net_inputs_list"], list):
        out = []
        for item, net_inputs in zip(items, batch["net_inputs_list"]):
            if isinstance(net_inputs, dict):
                out.append((item, net_inputs))
        return out

    net_inputs = batch.get("net_inputs")
    if isinstance(net_inputs, dict) and "input_ids" in net_inputs and torch.is_tensor(net_inputs["input_ids"]):
        bs = int(net_inputs["input_ids"].shape[0])
        out = []
        for i in range(min(bs, len(items))):
            one = {}
            for k, v in net_inputs.items():
                if torch.is_tensor(v) and v.shape[0] == bs:
                    one[k] = v[i:i + 1]
            out.append((items[i], one))
        return out
    return []


def build_items(
    args: argparse.Namespace,
    sft_mod: Any,
    multitask_mod: Any,
) -> Tuple[Any, List[Dict[str, Any]]]:
    if args.pipeline == "sft":
        if args.task_mode not in {"cot", "candidates", "json_only"}:
            raise ValueError("--task-mode for pipeline=sft must be one of: cot, candidates, json_only")
        items = sft_mod.build_items_from_rationale_jsonl(
            args.test_file,
            args.audio_dir,
            add_text_only=False,
            text_only=args.text_only,
            max_samples=args.max_samples,
            allow_text_fallback_when_audio_missing=True if args.text_only else False,
            print_audio_search_paths=args.print_audio_search_paths,
            audio_search_print_limit=args.audio_search_print_limit,
            strict_audio_missing=args.strict_audio_missing,
            train_candidates_only=(args.task_mode == "candidates"),
            train_json_only=(args.task_mode == "json_only"),
        )
        return sft_mod, items

    if args.task_mode not in {"cot", "label", "candidates"}:
        raise ValueError("--task-mode for pipeline=multitask must be one of: cot, label, candidates")
    base_items = multitask_mod.build_items_from_rationale_jsonl(
        args.test_file,
        args.audio_dir,
        add_text_only=False,
        text_only=args.text_only,
        max_samples=args.max_samples,
        allow_text_fallback_when_audio_missing=True if args.text_only else False,
        print_audio_search_paths=args.print_audio_search_paths,
        audio_search_print_limit=args.audio_search_print_limit,
        strict_audio_missing=args.strict_audio_missing,
        multitask=False,
    )
    mode = multitask_mod.normalize_task_mode(args.task_mode)
    items = [multitask_mod.build_task_item(item, mode) for item in base_items]
    return multitask_mod, items


def extract_features_from_model(
    args: argparse.Namespace,
    pipeline_mod: Any,
    items: List[Dict[str, Any]],
    gold_intent_by_slurp_id: Optional[Dict[str, Tuple[str, str, str]]] = None,
    gold_intent_by_file: Optional[Dict[str, Tuple[str, str, str]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = pipeline_mod.AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    processor, tokenizer = pipeline_mod.ensure_processor_tokenizer_or_raise(processor, args.model_name_or_path)
    model = pipeline_mod.load_audio_model_from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    if hasattr(pipeline_mod, "attach_tokenizer_to_model_for_compat"):
        pipeline_mod.attach_tokenizer_to_model_for_compat(model, tokenizer)
    model.eval()

    if args.pipeline == "sft":
        audio_input_mode = pipeline_mod._infer_audio_input_mode(
            args.model_name_or_path,
            processor=processor,
            tokenizer=tokenizer,
        )
        collator = pipeline_mod.InferenceCollator(processor, audio_input_mode=audio_input_mode)
    else:
        collator = pipeline_mod.InferenceCollator(processor, per_sample=True)

    dataset = pipeline_mod.MixedDataset(items)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collator,
    )

    features: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []

    batch_iter = tqdm(
        loader,
        total=len(loader),
        desc=f"Extract [{args.pipeline}/{args.task_mode}]",
        unit="batch",
        dynamic_ncols=True,
    )
    for batch in batch_iter:
        unpacked = _unpack_batch(batch)
        if not unpacked:
            continue

        for item, net_inputs in unpacked:
            one = {k: v.to(device) for k, v in net_inputs.items() if torch.is_tensor(v)}
            if "input_ids" not in one:
                continue
            if "attention_mask" not in one:
                one["attention_mask"] = torch.ones_like(one["input_ids"], dtype=torch.long)
            if hasattr(pipeline_mod, "_ensure_feature_masks_for_generation"):
                one = pipeline_mod._ensure_feature_masks_for_generation(one)

            one = _cast_floating_tensors_to_model_dtype(one, model)
            one = _filter_inputs_by_forward_signature(model, one)

            kwargs: Dict[str, Any] = dict(one)
            kwargs["output_hidden_states"] = True
            kwargs["return_dict"] = True
            kwargs["use_cache"] = False

            outputs = _forward_with_retry(model, kwargs)
            hidden = _get_hidden_from_outputs(outputs, layer_index=args.layer_index)
            vec = pool_feature(hidden, one.get("attention_mask"), args.pooling)
            if args.l2_normalize:
                vec = _l2_normalize_rows(vec.reshape(1, -1))[0]

            scenario, action, intent = infer_intent_from_item(
                item,
                gold_intent_by_slurp_id=gold_intent_by_slurp_id,
                gold_intent_by_file=gold_intent_by_file,
            )
            rows.append({
                "id": str(item.get("id", "")),
                "slurp_id": str(item.get("slurp_id", "")),
                "file": str(item.get("file", "")),
                "scenario": scenario,
                "action": action,
                "intent": intent,
                "task_mode": str(item.get("task_mode", "")),
                "input_type": "audio" if item.get("audio_path") else "text",
                "audio_path": str(item.get("audio_path", "")) if item.get("audio_path") else "",
            })
            features.append(vec.astype(np.float32))
        if hasattr(batch_iter, "set_postfix"):
            batch_iter.set_postfix(extracted=len(features), refresh=False)

    if not features:
        raise RuntimeError("No features were extracted. Check test_file/audio_dir/model compatibility.")

    return np.vstack(features).astype(np.float32), rows


def save_feature_artifacts(
    out_dir: str,
    features: np.ndarray,
    projection: np.ndarray,
    explained_ratio: Tuple[float, float],
    rows: List[Dict[str, Any]],
) -> None:
    _ensure_dir(out_dir)
    intents = np.asarray([r.get("intent", "__unknown__") for r in rows], dtype=object)
    ids = np.asarray([r.get("id", "") for r in rows], dtype=object)
    files = np.asarray([r.get("file", "") for r in rows], dtype=object)
    np.savez_compressed(
        os.path.join(out_dir, "features.npz"),
        features=features.astype(np.float32),
        projection_2d=projection.astype(np.float32),
        explained_var_ratio=np.asarray(explained_ratio, dtype=np.float32),
        intents=intents,
        ids=ids,
        files=files,
    )
    _write_jsonl(os.path.join(out_dir, "metadata.jsonl"), rows)


def load_feature_artifacts(reuse_dir: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    npz_path = os.path.join(reuse_dir, "features.npz")
    meta_path = os.path.join(reuse_dir, "metadata.jsonl")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"features.npz not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    if "features" not in data:
        raise RuntimeError(f"'features' key not found in {npz_path}")
    features = np.asarray(data["features"], dtype=np.float32)
    rows = _read_jsonl(meta_path)
    if rows and len(rows) != int(features.shape[0]):
        n = min(len(rows), int(features.shape[0]))
        rows = rows[:n]
        features = features[:n]
    if not rows:
        intents = data["intents"] if "intents" in data else np.asarray(["__unknown__"] * features.shape[0], dtype=object)
        ids = data["ids"] if "ids" in data else np.asarray([""] * features.shape[0], dtype=object)
        files = data["files"] if "files" in data else np.asarray([""] * features.shape[0], dtype=object)
        rows = []
        for i in range(features.shape[0]):
            rows.append({
                "id": str(ids[i]),
                "slurp_id": "",
                "file": str(files[i]),
                "scenario": "",
                "action": "",
                "intent": str(intents[i]),
                "task_mode": "",
                "input_type": "",
                "audio_path": "",
            })
    return features, rows


def choose_device(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_output_dir(args: argparse.Namespace) -> str:
    analysis_root = os.path.abspath(args.analysis_dir)
    _ensure_dir(analysis_root)
    if args.reuse_dir:
        out_dir = os.path.abspath(args.reuse_dir)
        _ensure_dir(out_dir)
        return out_dir

    model_tag = _sanitize_name(os.path.basename(os.path.abspath(args.model_name_or_path)))
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_name = _sanitize_name(args.run_name)
    else:
        run_name = _sanitize_name(f"{model_tag}_{args.pipeline}_{args.task_mode}_{ts}")
    out_dir = os.path.join(analysis_root, run_name)
    _ensure_dir(out_dir)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract model features and visualize intent separation."
    )
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Model path/checkpoint.")
    parser.add_argument("--reuse-dir", type=str, default=None, help="Reuse saved features from this analysis dir.")
    parser.add_argument("--pipeline", type=str, default="sft", choices=["sft", "multitask"])
    parser.add_argument("--task-mode", type=str, default="cot", help="sft: cot|candidates|json_only, multitask: cot|label|candidates")
    parser.add_argument("--test_file", type=str, default="slurp/dataset/slurp/test.jsonl")
    parser.add_argument("--audio_dir", type=str, default="slurp/slurp_real")
    parser.add_argument("--text_only", action="store_true", help="Force text-only input path.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="e.g., cuda, cuda:0, cpu")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--layer-index", type=int, default=-1, help="Hidden layer index (-1 means last hidden layer).")
    parser.add_argument("--l2-normalize", action="store_true", help="L2 normalize each feature vector.")
    parser.add_argument("--analysis-dir", type=str, default=os.path.join(SCRIPT_DIR, "analysis"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--min-intent-samples", type=int, default=5)
    parser.add_argument("--top-intents", type=int, default=20)
    parser.add_argument("--print-audio-search-paths", action="store_true")
    parser.add_argument("--audio-search-print-limit", type=int, default=20)
    parser.add_argument("--strict-audio-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.reuse_dir and not args.model_name_or_path:
        raise SystemExit("ERROR: --model_name_or_path is required unless --reuse-dir is set.")

    args.device = choose_device(args.device)
    out_dir = build_output_dir(args)

    if args.reuse_dir:
        features, rows = load_feature_artifacts(out_dir)
        print(f"Loaded cached features: {features.shape} from {out_dir}")
    else:
        sft_mod = _load_module_from_path("_sft_mod", SFT_PATH)
        multitask_mod = _load_module_from_path("_mt_mod", MULTITASK_PATH)
        pipeline_mod, items = build_items(args, sft_mod, multitask_mod)
        if not items:
            raise SystemExit("ERROR: No items built from test data.")
        print(f"Items prepared: {len(items)}")
        gold_by_sid, gold_by_file = load_gold_intent_maps_from_test_jsonl(args.test_file)
        if gold_by_sid or gold_by_file:
            print(f"Gold map loaded: slurp_id={len(gold_by_sid)} file={len(gold_by_file)}")
        else:
            print("Gold map not available; falling back to item-derived intent only.")
        print(f"Loading model on device: {args.device}")
        features, rows = extract_features_from_model(
            args,
            pipeline_mod,
            items,
            gold_intent_by_slurp_id=gold_by_sid,
            gold_intent_by_file=gold_by_file,
        )
        print(f"Extracted features: {features.shape}")

    if len(rows) != int(features.shape[0]):
        n = min(len(rows), int(features.shape[0]))
        rows = rows[:n]
        features = features[:n]

    unknown_n = sum(1 for r in rows if r.get("intent", "__unknown__") == "__unknown__")
    resolved_n = len(rows) - unknown_n
    print(f"Intent labels resolved: {resolved_n}/{len(rows)} (unknown={unknown_n})")
    if unknown_n == len(rows) and len(rows) > 0:
        print("WARNING: All intents are unknown. Check --test_file / metadata parsing.", file=sys.stderr)

    projection, explained_ratio = pca_project_2d(features)
    intents = [r.get("intent", "__unknown__") for r in rows]
    counts = Counter(intents)

    stats = compute_intent_distance_stats(
        features=features,
        intents=intents,
        min_intent_samples=max(1, int(args.min_intent_samples)),
    )

    save_feature_artifacts(out_dir, features, projection, explained_ratio, rows)

    config = {
        "model_name_or_path": args.model_name_or_path,
        "pipeline": args.pipeline,
        "task_mode": args.task_mode,
        "test_file": args.test_file,
        "audio_dir": args.audio_dir,
        "text_only": bool(args.text_only),
        "pooling": args.pooling,
        "layer_index": int(args.layer_index),
        "l2_normalize": bool(args.l2_normalize),
        "min_intent_samples": int(args.min_intent_samples),
        "top_intents": int(args.top_intents),
        "device": args.device,
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "num_unique_intents": len(counts),
        "pca_explained_var_ratio": [float(explained_ratio[0]), float(explained_ratio[1])],
    }
    _write_json(os.path.join(out_dir, "config.json"), config)
    _write_json(os.path.join(out_dir, "summary.json"), stats["summary"])
    save_intent_stats_csv(os.path.join(out_dir, "intent_stats.csv"), stats["intent_rows"])
    save_centroid_distance_csv(
        os.path.join(out_dir, "centroid_distances.csv"),
        stats["valid_intents"],
        stats["distance_matrix"],
    )

    title = f"Intent Feature Map ({args.pipeline}/{args.task_mode})"
    plot_pca_scatter(
        projection=projection,
        intents=intents,
        counts=counts,
        out_path=os.path.join(out_dir, "pca_scatter_by_intent.png"),
        title=title,
        explained_ratio=explained_ratio,
        top_k=max(1, int(args.top_intents)),
    )
    plot_centroid_heatmap(
        valid_intents=stats["valid_intents"],
        distance_matrix=stats["distance_matrix"],
        counts=stats["counts"],
        out_path=os.path.join(out_dir, "centroid_distance_heatmap.png"),
        top_k=max(1, int(args.top_intents)),
    )

    print(f"Saved analysis dir: {out_dir}")
    print(f"- {os.path.join(out_dir, 'features.npz')}")
    print(f"- {os.path.join(out_dir, 'metadata.jsonl')}")
    print(f"- {os.path.join(out_dir, 'intent_stats.csv')}")
    print(f"- {os.path.join(out_dir, 'centroid_distances.csv')}")
    print(f"- {os.path.join(out_dir, 'pca_scatter_by_intent.png')}")
    print(f"- {os.path.join(out_dir, 'centroid_distance_heatmap.png')}")
    print("Summary:")
    print(json.dumps(stats["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
