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
- Visualize multiple 2D embeddings: PCA / t-SNE / UMAP.

Examples:
    # SFT-style prompt (C/R/J), audio-only test behavior
    python 08_visualize_model_features.py \
      --model_name_or_path outputs/qwen_rationale_label_ft \
      --pipeline sft \
      --task-mode cot \
      --embeddings pca,tsne,umap

    # SFT json_only prompt (shortcut flag)
    python 08_visualize_model_features.py \
      --model_name_or_path outputs/qwen_rationale_label_ft \
      --pipeline sft \
      --only-json

    # Multitask label-only prompt feature extraction
    python 08_visualize_model_features.py \
      --model_name_or_path outputs/qwen_rationale_label_ft_multitask \
      --pipeline multitask \
      --task-mode label

    # Reuse already extracted features only
    python 08_visualize_model_features.py \
      --reuse-dir Experiment_RationaleCompare/analysis/model_feats_run_x
    # (same as --resume)

    # All intents + mean-distance rank-step table (no re-inference)
    python 08_visualize_model_features.py \
      --reuse-dir Experiment_RationaleCompare/analysis/model_feats_run_x \
      --all-intents \
      --distance-rank-step 3

    # Extract vectors around intent-generation timing
    python 08_visualize_model_features.py \
      --model_name_or_path outputs/qwen_rationale_label_ft \
      --pipeline sft \
      --feature-source intent_generation \
      --intent-max-new-tokens 256

    # Debug: show which generated token region was used for intent feature pooling
    python 08_visualize_model_features.py \
      --model_name_or_path outputs/qwen_rationale_label_ft \
      --pipeline sft \
      --feature-source intent_generation \
      --debug-intent-focus \
      --debug-intent-focus-limit 10

    # Multi-GPU extraction with torchrun
    torchrun --nproc_per_node 2 08_visualize_model_features.py \
      --model_name_or_path outputs/qwen_rationale_label_ft \
      --pipeline sft \
      --task-mode cot
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
from matplotlib import colors as mcolors
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return iterable
try:
    from sklearn.manifold import TSNE
except Exception:  # pragma: no cover
    TSNE = None
try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_PATH = os.path.join(SCRIPT_DIR, "audio_text_mix_e2e_re.py")
MULTITASK_PATH = os.path.join(SCRIPT_DIR, "audio_text_mix_e2e_re_multitask.py")
DEFAULT_HEATMAP_VMIN = 0.0
DEFAULT_HEATMAP_VMAX = 100.0
DEFAULT_HEATMAP_VMAX_L2 = 2.0


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


def _generate_sequences_with_retry(model: Any, kwargs: Dict[str, Any], max_retry: int = 6) -> torch.Tensor:
    working = dict(kwargs)
    for _ in range(max_retry):
        try:
            with torch.no_grad():
                output = model.generate(**working)
            if torch.is_tensor(output):
                return output
            sequences = getattr(output, "sequences", None)
            if torch.is_tensor(sequences):
                return sequences
            if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
                return output[0]
            raise RuntimeError("generate() returned unexpected output type.")
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
    raise RuntimeError("generate() retry limit exceeded while dropping unsupported kwargs.")


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
    use_pred_label_fallback: bool = True,
) -> Tuple[str, str, str]:
    target_obj = item.get("target_obj")
    scenario = ""
    action = ""
    if isinstance(target_obj, dict):
        scenario, action = _extract_label_from_json_obj(target_obj)
    if not scenario and not action:
        scenario, action = _parse_label_from_target_text(item.get("target", ""))
    if use_pred_label_fallback and (not scenario or not action) and isinstance(item.get("pred_label"), dict):
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


def _decode_text_from_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    return str(
        tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        or ""
    )


def _decode_text_and_offsets(tokenizer: Any, token_ids: Sequence[int]) -> Tuple[str, List[Tuple[int, int]]]:
    text = ""
    offsets: List[Tuple[int, int]] = []
    for tid in token_ids:
        piece = _decode_text_from_token_ids(tokenizer, [int(tid)])
        s = len(text)
        text += piece
        e = len(text)
        offsets.append((s, e))
    return text, offsets


def _find_span_case_insensitive(text: str, needle: str) -> Optional[Tuple[int, int]]:
    t = str(text or "")
    n = str(needle or "")
    if not t or not n:
        return None
    i = t.lower().find(n.lower())
    if i < 0:
        return None
    return i, i + len(n)


def _find_j_line_span(text: str) -> Optional[Tuple[int, int]]:
    t = str(text or "")
    if not t:
        return None
    match = None
    for m in re.finditer(r"(?mi)^J\s*[:：]\s*(.+)$", t):
        match = m
    if match is None:
        return None
    return int(match.start(1)), int(match.end(1))


def _find_json_key_value_span(text: str, key: str, value: str) -> Optional[Tuple[int, int]]:
    t = str(text or "")
    k = str(key or "").strip()
    v = str(value or "").strip()
    if not t or not k or not v:
        return None
    pattern = re.compile(
        rf'(?is)["\']{re.escape(k)}["\']\s*[:：]\s*["\']({re.escape(v)})["\']'
    )
    match = None
    for m in pattern.finditer(t):
        match = m
    if match is None:
        return None
    return int(match.start(1)), int(match.end(1))


def _select_generated_intent_token_indices(
    tokenizer: Any,
    generated_ids: Sequence[int],
    intent_label: str,
    scenario: str = "",
    action: str = "",
) -> Tuple[List[int], str]:
    output_text, offsets = _decode_text_and_offsets(tokenizer, generated_ids)
    candidate_idxs = [i for i, (s, e) in enumerate(offsets) if e > s]

    spans: List[Tuple[int, int]] = []
    span_source = ""

    s_span = _find_json_key_value_span(output_text, "scenario", scenario)
    a_span = _find_json_key_value_span(output_text, "action", action)
    if s_span is not None:
        spans.append(s_span)
    if a_span is not None:
        spans.append(a_span)
    if spans:
        span_source = "scenario_action_keys"
    else:
        span = _find_span_case_insensitive(output_text, intent_label)
        if span is not None:
            spans = [span]
            span_source = "intent_label"
        else:
            span = _find_j_line_span(output_text)
            if span is not None:
                spans = [span]
                span_source = "j_line"

    if not spans:
        if candidate_idxs:
            return candidate_idxs, "full_generated"
        if offsets:
            return [max(0, len(offsets) - 1)], "fallback_last"
        return [], "empty_generated"

    selected: List[int] = []
    for a, b in spans:
        selected.extend([i for i, (s, e) in enumerate(offsets) if e > a and s < b and e > s])
    selected = sorted(set(selected))
    if selected:
        return selected, span_source
    if candidate_idxs:
        return candidate_idxs, "full_generated"
    if offsets:
        return [max(0, len(offsets) - 1)], "fallback_last"
    return [], "empty_generated"


def _merge_spans(spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    cleaned: List[Tuple[int, int]] = []
    for a, b in spans:
        x = int(min(a, b))
        y = int(max(a, b))
        if y <= x:
            continue
        cleaned.append((x, y))
    if not cleaned:
        return []
    cleaned.sort(key=lambda p: p[0])
    merged: List[Tuple[int, int]] = [cleaned[0]]
    for a, b in cleaned[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def _build_highlight_preview(
    text: str,
    spans: Sequence[Tuple[int, int]],
    context_chars: int = 80,
) -> str:
    t = str(text or "")
    if not t:
        return ""
    merged = _merge_spans(spans)
    if not merged:
        head = min(len(t), max(80, int(context_chars) * 2))
        out = t[:head]
        if head < len(t):
            out += "..."
        return out

    context = max(0, int(context_chars))
    left = max(0, merged[0][0] - context)
    right = min(len(t), merged[-1][1] + context)

    parts: List[str] = []
    pos = left
    for a, b in merged:
        s = max(left, a)
        e = min(right, b)
        if e <= s:
            continue
        if s > pos:
            parts.append(t[pos:s])
        parts.append("[[")
        parts.append(t[s:e])
        parts.append("]]")
        pos = e
    if pos < right:
        parts.append(t[pos:right])
    out = "".join(parts)
    if left > 0:
        out = "..." + out
    if right < len(t):
        out += "..."
    return out


def _build_intent_focus_debug_payload(
    tokenizer: Any,
    generated_ids: Sequence[int],
    selected_local: Sequence[int],
    selected_global: Sequence[int],
    context_chars: int = 80,
    max_token_dump: int = 64,
) -> Dict[str, Any]:
    output_text, offsets = _decode_text_and_offsets(tokenizer, generated_ids)
    sel = sorted(set(int(i) for i in selected_local if 0 <= int(i) < len(offsets)))
    ranges = [offsets[i] for i in sel]
    preview = _build_highlight_preview(output_text, ranges, context_chars=context_chars)

    token_dump: List[Dict[str, Any]] = []
    token_cap = max(0, int(max_token_dump))
    for j, i in enumerate(sel[:token_cap]):
        tid = int(generated_ids[i])
        piece = _decode_text_from_token_ids(tokenizer, [tid])
        gidx = int(selected_global[j]) if j < len(selected_global) else None
        token_dump.append({
            "local_index": int(i),
            "global_index": gidx,
            "token_id": tid,
            "piece": piece,
        })

    payload: Dict[str, Any] = {
        "generated_text": output_text,
        "focus_preview": preview,
        "selected_local_indices": [int(i) for i in sel],
        "selected_global_indices": [int(i) for i in selected_global],
        "selected_char_ranges": [[int(a), int(b)] for a, b in ranges],
        "selected_tokens": token_dump,
        "selected_token_count": int(len(sel)),
        "selected_token_dump_truncated": bool(len(sel) > token_cap),
    }
    return payload


def _pool_hidden_at_indices(hidden: torch.Tensor, token_indices: Sequence[int]) -> np.ndarray:
    if hidden.dim() != 3 or hidden.shape[0] != 1:
        raise ValueError(f"Expected hidden [1,T,D], got {tuple(hidden.shape)}")
    if not token_indices:
        idx = torch.tensor([hidden.shape[1] - 1], device=hidden.device, dtype=torch.long)
    else:
        idx = torch.tensor(sorted(set(int(i) for i in token_indices)), device=hidden.device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < hidden.shape[1])]
        if idx.numel() == 0:
            idx = torch.tensor([hidden.shape[1] - 1], device=hidden.device, dtype=torch.long)
    vec = hidden[0, idx].mean(dim=0)
    return vec.detach().float().cpu().numpy()


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


def _parse_csv_set(text: str) -> List[str]:
    values = [x.strip().lower() for x in str(text or "").split(",") if x.strip()]
    seen = set()
    out: List[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def tsne_project_2d(
    x: np.ndarray,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    init: str,
    random_state: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if TSNE is None:
        return None, "scikit-learn is not installed."
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), None
    if x.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32), None

    n = int(x.shape[0])
    max_perp = max(1.0, float(n - 1))
    perp = min(float(perplexity), max_perp)
    if perp >= n:
        perp = max(1.0, float(n - 1))

    kwargs: Dict[str, Any] = {
        "n_components": 2,
        "perplexity": perp,
        "learning_rate": float(learning_rate),
        "init": str(init),
        "random_state": int(random_state),
    }
    try:
        sig = inspect.signature(TSNE.__init__)
        if "n_iter" in sig.parameters:
            kwargs["n_iter"] = int(n_iter)
        elif "max_iter" in sig.parameters:
            kwargs["max_iter"] = int(n_iter)
    except Exception:
        kwargs["n_iter"] = int(n_iter)

    try:
        model = TSNE(**kwargs)
        proj = model.fit_transform(x)
        return np.asarray(proj, dtype=np.float32), None
    except Exception as exc:
        return None, str(exc)


def umap_project_2d(
    x: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if umap is None:
        return None, "umap-learn is not installed."
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), None
    if x.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32), None

    n = int(x.shape[0])
    nn = max(2, min(int(n_neighbors), n - 1))
    try:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=nn,
            min_dist=float(min_dist),
            metric=str(metric),
            random_state=int(random_state),
        )
        proj = reducer.fit_transform(x)
        return np.asarray(proj, dtype=np.float32), None
    except Exception as exc:
        return None, str(exc)


def _pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(diff, axis=2).astype(np.float32)


def compute_intent_distance_stats(
    features: np.ndarray,
    intents: Sequence[str],
    min_intent_samples: int,
    include_unknown: bool = False,
) -> Dict[str, Any]:
    counts = Counter(intents)
    valid_intents = [
        intent for intent, c in counts.items()
        if (include_unknown or intent != "__unknown__") and c >= min_intent_samples
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
    if int(top_k) <= 0:
        return sorted(counts.keys(), key=lambda x: (-counts.get(x, 0), x))
    return [name for name, _ in counts.most_common(max(int(top_k), 1))]


def plot_embedding_scatter(
    projection: np.ndarray,
    intents: Sequence[str],
    counts: Counter,
    out_path: str,
    title: str,
    x_label: str,
    y_label: str,
    subtitle: Optional[str],
    top_k: int,
    show_label_text: bool,
) -> None:
    _ensure_dir(os.path.dirname(out_path) or ".")
    if int(top_k) <= 0:
        plot_labels = list(intents)
    else:
        top_intents = set(_pick_top_intents(counts, top_k))
        plot_labels = [x if x in top_intents else "other" for x in intents]
    unique = sorted(set(plot_labels), key=lambda x: (x == "other", -counts.get(x, 0), x))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(unique))]
    color_map = dict(zip(unique, colors))
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "8", "p", "*", "d"]
    marker_map = {label: marker_cycle[i % len(marker_cycle)] for i, label in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(10, 7))
    for label in unique:
        idx = [i for i, l in enumerate(plot_labels) if l == label]
        if not idx:
            continue
        label_text = f"{label} (n={len(idx)})"
        marker = marker_map[label]
        ax.scatter(
            projection[idx, 0],
            projection[idx, 1],
            s=24,
            alpha=0.8,
            c=[color_map[label]],
            label=label_text,
            marker=marker,
            edgecolors="none",
        )
        if show_label_text and label != "other":
            cx = float(np.mean(projection[idx, 0]))
            cy = float(np.mean(projection[idx, 1]))
            ax.text(
                cx,
                cy,
                label,
                fontsize=7,
                color="black",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.6),
            )

    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    else:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        frameon=True,
        ncol=1,
        title="intent",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_pca_scatter(
    projection: np.ndarray,
    intents: Sequence[str],
    counts: Counter,
    out_path: str,
    title: str,
    explained_ratio: Tuple[float, float],
    top_k: int,
    show_label_text: bool,
) -> None:
    subtitle = f"PCA-2D (var: PC1={explained_ratio[0]*100:.1f}%, PC2={explained_ratio[1]*100:.1f}%)"
    plot_embedding_scatter(
        projection=projection,
        intents=intents,
        counts=counts,
        out_path=out_path,
        title=title,
        x_label="PC1",
        y_label="PC2",
        subtitle=subtitle,
        top_k=top_k,
        show_label_text=show_label_text,
    )


def plot_centroid_heatmap(
    valid_intents: Sequence[str],
    distance_matrix: np.ndarray,
    counts: Counter,
    out_path: str,
    top_k: int,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    heatmap_gamma: float = 1.0,
) -> None:
    if len(valid_intents) == 0 or distance_matrix.size == 0:
        return

    top_order = sorted(valid_intents, key=lambda x: (-counts.get(x, 0), x))
    if int(top_k) > 0:
        top_order = top_order[: int(top_k)]
    idxs = [valid_intents.index(name) for name in top_order]
    sub = distance_matrix[np.ix_(idxs, idxs)]

    fig, ax = plt.subplots(figsize=(max(6, len(top_order) * 0.4), max(5, len(top_order) * 0.35)))
    norm = None
    gamma = float(heatmap_gamma)
    if vmin is not None and vmax is not None and gamma > 0 and abs(gamma - 1.0) > 1e-6:
        norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    if norm is not None:
        im = ax.imshow(sub, cmap="viridis", interpolation="nearest", norm=norm)
    else:
        im = ax.imshow(sub, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
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


def build_intent_mean_distance_rows(
    valid_intents: Sequence[str],
    counts: Counter,
    dist: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = len(valid_intents)
    for i, intent in enumerate(valid_intents):
        if n > 1:
            other = np.delete(dist[i], i)
            mean_d = float(other.mean()) if other.size > 0 else None
            near_d = float(other.min()) if other.size > 0 else None
        else:
            mean_d = None
            near_d = None
        rows.append({
            "intent": intent,
            "count": int(counts.get(intent, 0)),
            "mean_centroid_distance": mean_d,
            "nearest_centroid_distance": near_d,
        })
    rows.sort(
        key=lambda r: (
            float("inf") if r["mean_centroid_distance"] is None else float(r["mean_centroid_distance"]),
            float("inf") if r["nearest_centroid_distance"] is None else float(r["nearest_centroid_distance"]),
            -int(r["count"]),
            str(r["intent"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def sample_intent_rows_by_rank_step(
    rows: Sequence[Dict[str, Any]],
    rank_step: int,
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    step = max(1, int(rank_step))
    if step <= 1:
        return list(rows)

    last_rank = int(rows[-1].get("rank", len(rows)))
    sampled: List[Dict[str, Any]] = []
    for row in rows:
        rank = int(row.get("rank", 0))
        if rank == 1 or rank == last_rank or (rank % step == 0):
            sampled.append(dict(row))
    return sampled


def save_intent_mean_distance_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    fieldnames = [
        "rank",
        "intent",
        "count",
        "mean_centroid_distance",
        "nearest_centroid_distance",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_distance_submatrix(
    valid_intents: Sequence[str],
    dist: np.ndarray,
    ordered_intents: Sequence[str],
) -> Tuple[List[str], np.ndarray]:
    idx_map = {name: i for i, name in enumerate(valid_intents)}
    order = [name for name in ordered_intents if name in idx_map]
    if not order:
        return [], np.zeros((0, 0), dtype=np.float32)
    idxs = [idx_map[name] for name in order]
    sub = dist[np.ix_(idxs, idxs)].astype(np.float32)
    return order, sub


def compute_heatmap_color_limits(distance_matrix: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if distance_matrix.size == 0:
        return None, None
    mat = np.asarray(distance_matrix, dtype=np.float32)
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return None, None
    vmax = float(np.max(finite))
    vmin = 0.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _load_heatmap_scale(path: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None, None
    if not isinstance(obj, dict):
        return None, None
    vmin = obj.get("vmin")
    vmax = obj.get("vmax")
    try:
        vmin_f = float(vmin) if vmin is not None else None
    except Exception:
        vmin_f = None
    try:
        vmax_f = float(vmax) if vmax is not None else None
    except Exception:
        vmax_f = None
    return vmin_f, vmax_f


def _save_heatmap_scale(path: str, vmin: Optional[float], vmax: Optional[float]) -> None:
    if vmin is None or vmax is None:
        return
    _ensure_dir(os.path.dirname(path) or ".")
    payload = {"vmin": float(vmin), "vmax": float(vmax)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_saved_run_l2_flag(reuse_dir: str) -> Optional[bool]:
    cfg_path = os.path.join(reuse_dir, "config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    value = obj.get("l2_normalize")
    if isinstance(value, bool):
        return value
    return None


def save_distance_table_csv(
    path: str,
    ordered_intents: Sequence[str],
    distance_matrix: np.ndarray,
) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["intent"] + list(ordered_intents))
        for i, intent in enumerate(ordered_intents):
            row = [intent] + [float(distance_matrix[i, j]) for j in range(len(ordered_intents))]
            writer.writerow(row)


def plot_distance_gradient_heatmap(
    ordered_intents: Sequence[str],
    distance_matrix: np.ndarray,
    out_path: str,
    title: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    heatmap_gamma: float = 1.0,
) -> None:
    if len(ordered_intents) == 0 or distance_matrix.size == 0:
        return

    _ensure_dir(os.path.dirname(out_path) or ".")
    sub = np.asarray(distance_matrix, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(max(6, len(ordered_intents) * 0.45), max(5, len(ordered_intents) * 0.38)))
    norm = None
    gamma = float(heatmap_gamma)
    if vmin is not None and vmax is not None and gamma > 0 and abs(gamma - 1.0) > 1e-6:
        norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    if norm is not None:
        im = ax.imshow(sub, cmap="viridis", interpolation="nearest", norm=norm)
    else:
        im = ax.imshow(sub, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(ordered_intents)))
    ax.set_yticks(np.arange(len(ordered_intents)))
    ax.set_xticklabels(ordered_intents, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(ordered_intents, fontsize=7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("Distance", rotation=90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


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
    show_progress: bool = True,
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
    debug_intent_focus_enabled = bool(args.debug_intent_focus) and str(args.feature_source) == "intent_generation"
    debug_intent_focus_limit = max(1, int(args.debug_intent_focus_limit))
    debug_intent_focus_context = max(0, int(args.debug_intent_focus_context_chars))
    debug_intent_focus_count = 0

    batch_iter = tqdm(
        loader,
        total=len(loader),
        desc=f"Extract [{args.pipeline}/{args.task_mode}]",
        unit="batch",
        dynamic_ncols=True,
        disable=not bool(show_progress),
    )
    for batch in batch_iter:
        unpacked = _unpack_batch(batch)
        if not unpacked:
            continue

        for item, net_inputs in unpacked:
            one = {k: v.to(device) for k, v in net_inputs.items() if torch.is_tensor(v)}
            if "input_ids" not in one:
                continue
            scenario, action, intent = infer_intent_from_item(
                item,
                gold_intent_by_slurp_id=gold_intent_by_slurp_id,
                gold_intent_by_file=gold_intent_by_file,
                use_pred_label_fallback=False,
            )
            if "attention_mask" not in one:
                one["attention_mask"] = torch.ones_like(one["input_ids"], dtype=torch.long)
            if hasattr(pipeline_mod, "_ensure_feature_masks_for_generation"):
                one = pipeline_mod._ensure_feature_masks_for_generation(one)

            one = _cast_floating_tensors_to_model_dtype(one, model)
            one = _filter_inputs_by_forward_signature(model, one)

            span_source = "prompt_pool"
            intent_token_count = 0
            intent_lookup_label_source = "n/a"
            intent_lookup_scenario = ""
            intent_lookup_action = ""
            intent_debug_payload: Optional[Dict[str, Any]] = None
            if args.feature_source == "intent_generation":
                gen_kwargs: Dict[str, Any] = dict(one)
                gen_kwargs["max_new_tokens"] = max(1, int(args.intent_max_new_tokens))
                gen_kwargs["do_sample"] = False
                if tokenizer.pad_token_id is not None:
                    gen_kwargs["pad_token_id"] = int(tokenizer.pad_token_id)
                elif tokenizer.eos_token_id is not None:
                    gen_kwargs["pad_token_id"] = int(tokenizer.eos_token_id)
                if tokenizer.eos_token_id is not None:
                    gen_kwargs["eos_token_id"] = int(tokenizer.eos_token_id)

                sequences = _generate_sequences_with_retry(model, gen_kwargs)
                if sequences.dim() == 1:
                    sequences = sequences.unsqueeze(0)
                if sequences.shape[0] != 1:
                    sequences = sequences[:1]

                prompt_len = int(one["input_ids"].shape[1])
                seq_len = int(sequences.shape[1])
                if seq_len > prompt_len:
                    generated_ids = sequences[0, prompt_len:].detach().cpu().tolist()
                    generated_text = _decode_text_from_token_ids(tokenizer, generated_ids)
                    lookup_scenario = ""
                    lookup_action = ""
                    parse_fn = getattr(pipeline_mod, "parse_prediction_label", None)
                    if callable(parse_fn):
                        try:
                            pred = parse_fn(generated_text)
                            if isinstance(pred, dict):
                                ps, pa = _extract_label_from_json_obj(pred)
                                if ps and pa:
                                    lookup_scenario, lookup_action = ps, pa
                                    intent_lookup_label_source = "predicted"
                        except Exception:
                            pass
                    if not lookup_scenario or not lookup_action:
                        ps, pa = _parse_label_from_target_text(generated_text)
                        if ps and pa:
                            lookup_scenario, lookup_action = ps, pa
                            intent_lookup_label_source = "predicted"
                    if (not lookup_scenario or not lookup_action) and scenario and action:
                        lookup_scenario, lookup_action = scenario, action
                        intent_lookup_label_source = "gold_fallback"
                    if not lookup_scenario or not lookup_action:
                        intent_lookup_label_source = "none"
                    intent_lookup_scenario = lookup_scenario
                    intent_lookup_action = lookup_action
                    lookup_intent_label = (
                        f"{lookup_scenario}_{lookup_action}" if lookup_scenario and lookup_action else ""
                    )

                    selected_local, span_source = _select_generated_intent_token_indices(
                        tokenizer=tokenizer,
                        generated_ids=generated_ids,
                        intent_label=lookup_intent_label,
                        scenario=lookup_scenario,
                        action=lookup_action,
                    )
                    intent_token_count = int(len(selected_local))
                    selected_global = [prompt_len + int(i) for i in selected_local]
                    if debug_intent_focus_enabled and debug_intent_focus_count < debug_intent_focus_limit:
                        intent_debug_payload = _build_intent_focus_debug_payload(
                            tokenizer=tokenizer,
                            generated_ids=generated_ids,
                            selected_local=selected_local,
                            selected_global=selected_global,
                            context_chars=debug_intent_focus_context,
                        )
                        intent_debug_payload.update({
                            "debug_status": "ok",
                            "span_source": str(span_source),
                            "lookup_label_source": str(intent_lookup_label_source),
                            "lookup_scenario": str(intent_lookup_scenario),
                            "lookup_action": str(intent_lookup_action),
                            "gold_scenario": str(scenario),
                            "gold_action": str(action),
                            "gold_intent": str(intent),
                        })
                        debug_intent_focus_count += 1
                        if show_progress:
                            preview_obj = {
                                "id": str(item.get("id", "")),
                                "file": str(item.get("file", "")),
                                "slurp_id": str(item.get("slurp_id", "")),
                                "gold_intent": str(intent),
                                "lookup": f"{intent_lookup_scenario}_{intent_lookup_action}" if intent_lookup_scenario and intent_lookup_action else "",
                                "span_source": str(span_source),
                                "token_count": int(intent_debug_payload.get("selected_token_count", 0)),
                                "focus_preview": str(intent_debug_payload.get("focus_preview", "")),
                            }
                            print("[intent-focus-debug] " + json.dumps(preview_obj, ensure_ascii=False))

                    full_inputs: Dict[str, Any] = dict(one)
                    full_inputs["input_ids"] = sequences.to(device)
                    full_inputs["attention_mask"] = torch.ones_like(full_inputs["input_ids"], dtype=torch.long)
                    if hasattr(pipeline_mod, "_ensure_feature_masks_for_generation"):
                        full_inputs = pipeline_mod._ensure_feature_masks_for_generation(full_inputs)
                    full_inputs = _cast_floating_tensors_to_model_dtype(full_inputs, model)
                    full_inputs = _filter_inputs_by_forward_signature(model, full_inputs)

                    kwargs: Dict[str, Any] = dict(full_inputs)
                    kwargs["output_hidden_states"] = True
                    kwargs["return_dict"] = True
                    kwargs["use_cache"] = False
                    outputs = _forward_with_retry(model, kwargs)
                    hidden = _get_hidden_from_outputs(outputs, layer_index=args.layer_index)
                    vec = _pool_hidden_at_indices(hidden, selected_global)
                else:
                    # If no new token is generated, fallback to prompt representation.
                    if debug_intent_focus_enabled and debug_intent_focus_count < debug_intent_focus_limit:
                        intent_debug_payload = {
                            "debug_status": "no_new_token_generated",
                            "span_source": "prompt_fallback_no_generation",
                            "lookup_label_source": str(intent_lookup_label_source),
                            "lookup_scenario": str(intent_lookup_scenario),
                            "lookup_action": str(intent_lookup_action),
                            "gold_scenario": str(scenario),
                            "gold_action": str(action),
                            "gold_intent": str(intent),
                            "generated_text": "",
                            "focus_preview": "",
                            "selected_local_indices": [],
                            "selected_global_indices": [],
                            "selected_char_ranges": [],
                            "selected_tokens": [],
                            "selected_token_count": 0,
                            "selected_token_dump_truncated": False,
                        }
                        debug_intent_focus_count += 1
                        if show_progress:
                            preview_obj = {
                                "id": str(item.get("id", "")),
                                "file": str(item.get("file", "")),
                                "slurp_id": str(item.get("slurp_id", "")),
                                "gold_intent": str(intent),
                                "span_source": "prompt_fallback_no_generation",
                                "note": "no token generated; used prompt pooling fallback",
                            }
                            print("[intent-focus-debug] " + json.dumps(preview_obj, ensure_ascii=False))
                    kwargs = dict(one)
                    kwargs["output_hidden_states"] = True
                    kwargs["return_dict"] = True
                    kwargs["use_cache"] = False
                    outputs = _forward_with_retry(model, kwargs)
                    hidden = _get_hidden_from_outputs(outputs, layer_index=args.layer_index)
                    vec = pool_feature(hidden, one.get("attention_mask"), args.pooling)
                    span_source = "prompt_fallback_no_generation"
            else:
                kwargs = dict(one)
                kwargs["output_hidden_states"] = True
                kwargs["return_dict"] = True
                kwargs["use_cache"] = False
                outputs = _forward_with_retry(model, kwargs)
                hidden = _get_hidden_from_outputs(outputs, layer_index=args.layer_index)
                vec = pool_feature(hidden, one.get("attention_mask"), args.pooling)
            if args.l2_normalize:
                vec = _l2_normalize_rows(vec.reshape(1, -1))[0]

            row_obj = {
                "_global_index": int(item.get("_global_index")) if item.get("_global_index") is not None else None,
                "id": str(item.get("id", "")),
                "slurp_id": str(item.get("slurp_id", "")),
                "file": str(item.get("file", "")),
                "scenario": scenario,
                "action": action,
                "intent": intent,
                "task_mode": str(item.get("task_mode", "")),
                "input_type": "audio" if item.get("audio_path") else "text",
                "audio_path": str(item.get("audio_path", "")) if item.get("audio_path") else "",
                "feature_source": str(args.feature_source),
                "label_source": "gold",
                "intent_lookup_label_source": intent_lookup_label_source,
                "intent_lookup_scenario": intent_lookup_scenario,
                "intent_lookup_action": intent_lookup_action,
                "intent_span_source": span_source,
                "intent_token_count": int(intent_token_count),
            }
            if intent_debug_payload is not None:
                row_obj["intent_debug"] = intent_debug_payload
            rows.append(row_obj)
            features.append(vec.astype(np.float32))
        if hasattr(batch_iter, "set_postfix"):
            batch_iter.set_postfix(extracted=len(features), refresh=False)

    if not features:
        return np.zeros((0, 0), dtype=np.float32), rows

    return np.vstack(features).astype(np.float32), rows


def save_feature_artifacts(
    out_dir: str,
    features: np.ndarray,
    rows: List[Dict[str, Any]],
    projections: Dict[str, np.ndarray],
    pca_explained_ratio: Tuple[float, float],
    projection_indices: Optional[np.ndarray] = None,
) -> None:
    _ensure_dir(out_dir)
    intents = np.asarray([r.get("intent", "__unknown__") for r in rows], dtype=object)
    ids = np.asarray([r.get("id", "") for r in rows], dtype=object)
    files = np.asarray([r.get("file", "") for r in rows], dtype=object)
    payload: Dict[str, Any] = {
        "features": features.astype(np.float32),
        "explained_var_ratio": np.asarray(pca_explained_ratio, dtype=np.float32),
        "intents": intents,
        "ids": ids,
        "files": files,
    }
    pca_proj = projections.get("pca")
    if pca_proj is not None:
        # backward compatible key
        payload["projection_2d"] = np.asarray(pca_proj, dtype=np.float32)
        payload["projection_pca_2d"] = np.asarray(pca_proj, dtype=np.float32)
    tsne_proj = projections.get("tsne")
    if tsne_proj is not None:
        payload["projection_tsne_2d"] = np.asarray(tsne_proj, dtype=np.float32)
    umap_proj = projections.get("umap")
    if umap_proj is not None:
        payload["projection_umap_2d"] = np.asarray(umap_proj, dtype=np.float32)
    if projection_indices is not None:
        payload["projection_indices"] = np.asarray(projection_indices, dtype=np.int64)

    np.savez_compressed(os.path.join(out_dir, "features.npz"), **payload)
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


def _dist_is_initialized() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def _dist_rank() -> int:
    if _dist_is_initialized():
        return int(dist.get_rank())
    return 0


def _dist_world_size() -> int:
    if _dist_is_initialized():
        return int(dist.get_world_size())
    return 1


def _is_main_process() -> bool:
    return _dist_rank() == 0


def setup_distributed(dist_backend: str = "auto") -> Tuple[bool, int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    rank_env = int(os.environ.get("RANK", "0"))
    if local_rank < 0 or world_size_env <= 1:
        return False, local_rank, rank_env, world_size_env
    if _dist_is_initialized():
        return True, local_rank, _dist_rank(), _dist_world_size()

    backend = str(dist_backend or "auto").strip().lower()
    if backend in {"", "auto"}:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return True, local_rank, _dist_rank(), _dist_world_size()


def finalize_distributed() -> None:
    if _dist_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def choose_device(device_arg: Optional[str], local_rank: int = -1) -> str:
    if local_rank >= 0:
        device_text = str(device_arg or "").strip().lower()
        if device_text.startswith("cpu"):
            return str(device_arg)
        if torch.cuda.is_available():
            return f"cuda:{local_rank}"
        return "cpu"
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
    parser.add_argument(
        "--reuse-dir",
        "--resume",
        dest="reuse_dir",
        type=str,
        default=None,
        help="Reuse saved features from this analysis dir.",
    )
    parser.add_argument("--dist-backend", type=str, default="auto", choices=["auto", "nccl", "gloo"],
                        help="torchrun時のdistributed backend")
    parser.add_argument("--pipeline", type=str, default="sft", choices=["sft", "multitask"])
    parser.add_argument(
        "--task-mode",
        "--task_model",
        "--task-model",
        dest="task_mode",
        type=str,
        default="cot",
        help="sft: cot|candidates|json_only, multitask: cot|label|candidates",
    )
    parser.add_argument("--only-json", "--json-only", action="store_true",
                        help="SFT時に task-mode を json_only にするショートカット")
    parser.add_argument("--test_file", type=str, default="slurp/dataset/slurp/test.jsonl")
    parser.add_argument("--audio_dir", type=str, default="slurp/slurp_real")
    parser.add_argument("--text_only", action="store_true", help="Force text-only input path.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="e.g., cuda, cuda:0, cpu")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--layer-index", type=int, default=-1, help="Hidden layer index (-1 means last hidden layer).")
    parser.add_argument(
        "--feature-source",
        type=str,
        default="prompt",
        choices=["prompt", "intent_generation"],
        help="prompt: 入力プロンプト hidden を集約 / intent_generation: 生成時にIntentが出るトークン近傍 hidden を集約",
    )
    parser.add_argument("--intent-max-new-tokens", type=int, default=256,
                        help="feature-source=intent_generation 時の生成長上限")
    parser.add_argument(
        "--debug-intent-focus",
        action="store_true",
        help="feature-source=intent_generation 時に、特徴量抽出で着目した生成トークン周辺を表示・保存する",
    )
    parser.add_argument(
        "--debug-intent-focus-limit",
        type=int,
        default=20,
        help="intent focus デバッグを何サンプル分まで出力するか (default: 20)",
    )
    parser.add_argument(
        "--debug-intent-focus-context-chars",
        type=int,
        default=80,
        help="intent focus プレビューの前後文脈文字数 (default: 80)",
    )
    parser.add_argument("--l2-normalize", action="store_true", help="L2 normalize each feature vector.")
    parser.add_argument("--embeddings", type=str, default="pca,tsne,umap",
                        help="2D embeddings to compute (comma): pca,tsne,umap")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-learning-rate", type=float, default=200.0)
    parser.add_argument("--tsne-n-iter", type=int, default=1000)
    parser.add_argument("--tsne-init", type=str, default="pca", choices=["pca", "random"])
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", type=str, default="euclidean")
    parser.add_argument("--analysis-dir", type=str, default=os.path.join(SCRIPT_DIR, "analysis"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--min-intent-samples", type=int, default=5)
    parser.add_argument("--top-intents", type=int, default=20,
                        help="Scatter/heatmapで表示するIntent数。0以下で全Intentを表示")
    parser.add_argument("--all-intents", action="store_true",
                        help="Intent解析を全Intent対象にする (min-intent-samples=1, top-intents=0)")
    parser.add_argument("--include-unknown-intent", action="store_true",
                        help="__unknown__ Intentも解析対象に含める")
    parser.add_argument("--distance-rank-step", type=int, default=3,
                        help="平均distance順位を何件おきに抽出するか (3なら rank=1,3,6,9,...,last)。2以上で可視化も抽出Intentのみで実行")
    parser.add_argument("--annotate-labels", action="store_true",
                        help="散布図上にIntentラベル文字を重ねて表示する（凡例は常時表示）")
    parser.add_argument("--heatmap-vmin", type=float, default=None,
                        help="ヒートマップ色スケール下限を固定（モデル間比較用）")
    parser.add_argument("--heatmap-vmax", type=float, default=None,
                        help="ヒートマップ色スケール上限を固定（モデル間比較用）")
    parser.add_argument("--heatmap-scale-file", type=str, default=None,
                        help="共通スケールJSON（存在すれば読込、無ければ今回値を書き出し）")
    parser.add_argument("--heatmap-gamma", type=float, default=0.6,
                        help="ヒートマップのグラデーション強度（PowerNorm）。1.0=線形、1未満で低中距離を強調")
    parser.add_argument("--print-audio-search-paths", action="store_true")
    parser.add_argument("--audio-search-print-limit", type=int, default=20)
    parser.add_argument("--strict-audio-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist_enabled = False
    local_rank = -1
    rank = 0
    world_size = 1
    try:
        dist_enabled, local_rank, rank, world_size = setup_distributed(args.dist_backend)
        if dist_enabled:
            print(
                f"[rank {rank}] Distributed enabled: world_size={world_size}, "
                f"local_rank={local_rank}, backend={dist.get_backend()}"
            )

        if not args.reuse_dir and not args.model_name_or_path:
            raise SystemExit("ERROR: --model_name_or_path is required unless --reuse-dir is set.")

        if args.only_json:
            if args.pipeline != "sft":
                raise SystemExit("ERROR: --only-json is available only when --pipeline sft.")
            args.task_mode = "json_only"
        if int(args.intent_max_new_tokens) <= 0:
            raise SystemExit("ERROR: --intent-max-new-tokens must be > 0.")
        if int(args.debug_intent_focus_limit) <= 0:
            raise SystemExit("ERROR: --debug-intent-focus-limit must be > 0.")
        if int(args.debug_intent_focus_context_chars) < 0:
            raise SystemExit("ERROR: --debug-intent-focus-context-chars must be >= 0.")
        if float(args.heatmap_gamma) <= 0:
            raise SystemExit("ERROR: --heatmap-gamma must be > 0.")
        if args.debug_intent_focus and args.feature_source != "intent_generation" and _is_main_process():
            print("WARNING: --debug-intent-focus is effective only with --feature-source intent_generation.", file=sys.stderr)
        if args.pipeline == "multitask" and _is_main_process():
            print(f"Multitask task_mode: {args.task_mode}")
        if args.all_intents:
            args.min_intent_samples = 1
            args.top_intents = 0
            if _is_main_process():
                print("All-intents mode: min_intent_samples=1, top_intents=0")

        args.device = choose_device(args.device, local_rank=local_rank)
        effective_l2_normalize = bool(args.l2_normalize)

        features: np.ndarray
        rows: List[Dict[str, Any]]
        out_dir: Optional[str] = None

        if args.reuse_dir:
            if not _is_main_process():
                return
            out_dir = build_output_dir(args)
            features, rows = load_feature_artifacts(out_dir)
            print(f"Loaded cached features: {features.shape} from {out_dir}")
            saved_l2_flag = _load_saved_run_l2_flag(out_dir)
            if saved_l2_flag is not None:
                effective_l2_normalize = bool(saved_l2_flag)
                print(f"Using saved l2_normalize from config: {effective_l2_normalize}")
        else:
            sft_mod = _load_module_from_path("_sft_mod", SFT_PATH)
            multitask_mod = _load_module_from_path("_mt_mod", MULTITASK_PATH)
            pipeline_mod, base_items = build_items(args, sft_mod, multitask_mod)
            if not base_items:
                raise SystemExit("ERROR: No items built from test data.")

            indexed_items: List[Dict[str, Any]] = []
            for i, item in enumerate(base_items):
                obj = dict(item)
                obj["_global_index"] = int(i)
                indexed_items.append(obj)

            if dist_enabled:
                my_items = indexed_items[rank::world_size]
            else:
                my_items = indexed_items
            print(f"[rank {rank}] Items assigned: {len(my_items)}/{len(indexed_items)}")

            gold_by_sid, gold_by_file = load_gold_intent_maps_from_test_jsonl(args.test_file)
            if _is_main_process():
                if gold_by_sid or gold_by_file:
                    print(f"Gold map loaded: slurp_id={len(gold_by_sid)} file={len(gold_by_file)}")
                else:
                    print("Gold map not available; falling back to item-derived intent only.")
                print(f"Loading model on device: {args.device}")

            local_features, local_rows = extract_features_from_model(
                args,
                pipeline_mod,
                my_items,
                gold_intent_by_slurp_id=gold_by_sid,
                gold_intent_by_file=gold_by_file,
                show_progress=_is_main_process(),
            )
            print(f"[rank {rank}] Local extracted features: {local_features.shape}")

            if dist_enabled:
                gathered: List[Any] = [None for _ in range(world_size)]
                dist.all_gather_object(
                    gathered,
                    {
                        "features": local_features,
                        "rows": local_rows,
                    },
                )
                if not _is_main_process():
                    return

                merged: List[Tuple[int, Dict[str, Any], np.ndarray]] = []
                merged_feature_dim = 0
                for payload in gathered:
                    if not isinstance(payload, dict):
                        continue
                    payload_rows = payload.get("rows") or []
                    payload_features = np.asarray(payload.get("features"), dtype=np.float32)
                    if payload_features.ndim == 1:
                        payload_features = payload_features.reshape(1, -1)
                    if payload_features.ndim != 2:
                        continue
                    if payload_features.shape[0] > 0 and payload_features.shape[1] > 0:
                        merged_feature_dim = max(merged_feature_dim, int(payload_features.shape[1]))
                    n = min(len(payload_rows), int(payload_features.shape[0]))
                    for i in range(n):
                        row_obj = payload_rows[i] if isinstance(payload_rows[i], dict) else {}
                        row = dict(row_obj)
                        try:
                            gidx = int(row.get("_global_index"))
                        except Exception:
                            gidx = 10**12 + len(merged)
                        merged.append((gidx, row, payload_features[i].astype(np.float32)))
                merged.sort(key=lambda x: x[0])
                if merged:
                    rows = [x[1] for x in merged]
                    features = np.vstack([x[2] for x in merged]).astype(np.float32)
                else:
                    rows = []
                    features = np.zeros((0, merged_feature_dim), dtype=np.float32)
                print(f"Gathered extracted features: {features.shape} from world_size={world_size}")
            else:
                features, rows = local_features, local_rows

            out_dir = build_output_dir(args)

        if not _is_main_process():
            return

        if out_dir is None:
            out_dir = build_output_dir(args)

        cleaned_rows: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            obj = dict(row)
            obj.pop("_global_index", None)
            cleaned_rows.append(obj)
        rows = cleaned_rows

        if int(features.shape[0]) == 0:
            raise RuntimeError("No features were extracted. Check test_file/audio_dir/model compatibility.")
        if len(rows) != int(features.shape[0]):
            n = min(len(rows), int(features.shape[0]))
            rows = rows[:n]
            features = features[:n]

        unknown_n = sum(1 for r in rows if r.get("intent", "__unknown__") == "__unknown__")
        resolved_n = len(rows) - unknown_n
        print(f"Intent labels resolved: {resolved_n}/{len(rows)} (unknown={unknown_n})")
        if unknown_n == len(rows) and len(rows) > 0:
            print("WARNING: All intents are unknown. Check --test_file / metadata parsing.", file=sys.stderr)

        intents = [r.get("intent", "__unknown__") for r in rows]
        counts = Counter(intents)
        stats = compute_intent_distance_stats(
            features=features,
            intents=intents,
            min_intent_samples=max(1, int(args.min_intent_samples)),
            include_unknown=bool(args.include_unknown_intent),
        )
        mean_rows = build_intent_mean_distance_rows(
            valid_intents=stats["valid_intents"],
            counts=stats["counts"],
            dist=stats["distance_matrix"],
        )
        sampled_rows = sample_intent_rows_by_rank_step(
            rows=mean_rows,
            rank_step=max(1, int(args.distance_rank_step)),
        )
        sampled_intents = [str(r.get("intent", "")) for r in sampled_rows if str(r.get("intent", ""))]
        sampled_order, sampled_dist = build_distance_submatrix(
            valid_intents=stats["valid_intents"],
            dist=stats["distance_matrix"],
            ordered_intents=sampled_intents,
        )
        auto_heatmap_vmin, auto_heatmap_vmax = compute_heatmap_color_limits(stats["distance_matrix"])
        heatmap_vmin, heatmap_vmax = auto_heatmap_vmin, auto_heatmap_vmax
        heatmap_scale_source = "auto"
        used_scale_file = bool(args.heatmap_scale_file and os.path.exists(args.heatmap_scale_file))

        if used_scale_file:
            file_vmin, file_vmax = _load_heatmap_scale(args.heatmap_scale_file)
            if file_vmin is not None and file_vmax is not None:
                heatmap_vmin, heatmap_vmax = file_vmin, file_vmax
                heatmap_scale_source = f"file:{args.heatmap_scale_file}"
                print(
                    f"Using heatmap scale from file: vmin={heatmap_vmin:.6f}, vmax={heatmap_vmax:.6f}"
                )

        cli_override = False
        if args.heatmap_vmin is not None:
            heatmap_vmin = float(args.heatmap_vmin)
            heatmap_scale_source = "cli"
            cli_override = True
        if args.heatmap_vmax is not None:
            heatmap_vmax = float(args.heatmap_vmax)
            heatmap_scale_source = "cli"
            cli_override = True

        # Hard-coded default scale for cross-model comparability when not explicitly overridden.
        if not used_scale_file and not cli_override:
            heatmap_vmin = float(DEFAULT_HEATMAP_VMIN)
            if effective_l2_normalize:
                heatmap_vmax = float(DEFAULT_HEATMAP_VMAX_L2)
                heatmap_scale_source = "fixed_default_l2"
            else:
                heatmap_vmax = float(DEFAULT_HEATMAP_VMAX)
                heatmap_scale_source = "fixed_default"

        if heatmap_vmin is not None and heatmap_vmax is not None and heatmap_vmax <= heatmap_vmin:
            raise SystemExit("ERROR: heatmap scale requires vmax > vmin.")
        if heatmap_vmin is not None and heatmap_vmax is not None:
            print(
                f"Heatmap scale: source={heatmap_scale_source} "
                f"vmin={heatmap_vmin:.6f} vmax={heatmap_vmax:.6f}"
            )

        if args.heatmap_scale_file and (not os.path.exists(args.heatmap_scale_file)):
            _save_heatmap_scale(args.heatmap_scale_file, heatmap_vmin, heatmap_vmax)
            if heatmap_vmin is not None and heatmap_vmax is not None:
                print(
                    f"Saved heatmap scale file: {args.heatmap_scale_file} "
                    f"(vmin={heatmap_vmin:.6f}, vmax={heatmap_vmax:.6f})"
                )

        use_rank_filtered_viz = int(args.distance_rank_step) > 1 and len(sampled_order) > 0
        sampled_set = set(sampled_order)
        if use_rank_filtered_viz:
            viz_indices = np.asarray(
                [i for i, intent in enumerate(intents) if intent in sampled_set],
                dtype=np.int64,
            )
        else:
            viz_indices = np.arange(len(intents), dtype=np.int64)
        if viz_indices.size == 0:
            viz_indices = np.arange(len(intents), dtype=np.int64)
            use_rank_filtered_viz = False

        viz_features = features[viz_indices]
        viz_intents = [intents[int(i)] for i in viz_indices.tolist()]
        viz_counts = Counter(viz_intents)
        viz_top_k = 0 if use_rank_filtered_viz else int(args.top_intents)
        if use_rank_filtered_viz:
            print(
                f"Visualization labels filtered by rank-step: intents={len(sampled_set)} samples={len(viz_intents)}"
            )

        requested_embeddings = _parse_csv_set(args.embeddings)
        if not requested_embeddings:
            requested_embeddings = ["pca"]
        valid_embedding_names = {"pca", "tsne", "umap"}
        unknown_embeddings = [x for x in requested_embeddings if x not in valid_embedding_names]
        if unknown_embeddings:
            print(f"WARNING: Unknown embeddings ignored: {unknown_embeddings}", file=sys.stderr)
        requested_embeddings = [x for x in requested_embeddings if x in valid_embedding_names]
        if not requested_embeddings:
            requested_embeddings = ["pca"]

        projections: Dict[str, np.ndarray] = {}
        embedding_notes: Dict[str, str] = {}
        explained_ratio = (0.0, 0.0)

        if "pca" in requested_embeddings:
            print("Computing PCA-2D...")
            pca_projection, explained_ratio = pca_project_2d(viz_features)
            projections["pca"] = pca_projection
            embedding_notes["pca"] = "ok"

        if "tsne" in requested_embeddings:
            print("Computing t-SNE-2D...")
            tsne_projection, tsne_err = tsne_project_2d(
                x=viz_features,
                perplexity=args.tsne_perplexity,
                learning_rate=args.tsne_learning_rate,
                n_iter=args.tsne_n_iter,
                init=args.tsne_init,
                random_state=args.random_state,
            )
            if tsne_projection is not None:
                projections["tsne"] = tsne_projection
                embedding_notes["tsne"] = "ok"
            else:
                embedding_notes["tsne"] = f"skipped: {tsne_err}"
                print(f"WARNING: t-SNE skipped: {tsne_err}", file=sys.stderr)

        if "umap" in requested_embeddings:
            print("Computing UMAP-2D...")
            umap_projection, umap_err = umap_project_2d(
                x=viz_features,
                n_neighbors=args.umap_n_neighbors,
                min_dist=args.umap_min_dist,
                metric=args.umap_metric,
                random_state=args.random_state,
            )
            if umap_projection is not None:
                projections["umap"] = umap_projection
                embedding_notes["umap"] = "ok"
            else:
                embedding_notes["umap"] = f"skipped: {umap_err}"
                print(f"WARNING: UMAP skipped: {umap_err}", file=sys.stderr)

        save_feature_artifacts(
            out_dir=out_dir,
            features=features,
            rows=rows,
            projections=projections,
            pca_explained_ratio=explained_ratio,
            projection_indices=viz_indices,
        )
        intent_focus_debug_path: Optional[str] = None
        intent_focus_debug_rows: List[Dict[str, Any]] = []
        if bool(args.debug_intent_focus):
            for row in rows:
                dbg = row.get("intent_debug")
                if not isinstance(dbg, dict):
                    continue
                intent_focus_debug_rows.append({
                    "id": str(row.get("id", "")),
                    "slurp_id": str(row.get("slurp_id", "")),
                    "file": str(row.get("file", "")),
                    "intent": str(row.get("intent", "")),
                    "intent_span_source": str(row.get("intent_span_source", "")),
                    "intent_token_count": int(row.get("intent_token_count", 0)),
                    "intent_lookup_label_source": str(row.get("intent_lookup_label_source", "")),
                    "intent_lookup_scenario": str(row.get("intent_lookup_scenario", "")),
                    "intent_lookup_action": str(row.get("intent_lookup_action", "")),
                    "intent_debug": dbg,
                })
            if intent_focus_debug_rows:
                intent_focus_debug_path = os.path.join(out_dir, "intent_focus_debug.jsonl")
                _write_jsonl(intent_focus_debug_path, intent_focus_debug_rows)
                print(
                    f"Saved intent focus debug: rows={len(intent_focus_debug_rows)} "
                    f"path={intent_focus_debug_path}"
                )

        config = {
            "model_name_or_path": args.model_name_or_path,
            "pipeline": args.pipeline,
            "task_mode": args.task_mode,
            "test_file": args.test_file,
            "audio_dir": args.audio_dir,
            "text_only": bool(args.text_only),
            "pooling": args.pooling,
            "layer_index": int(args.layer_index),
            "feature_source": str(args.feature_source),
            "intent_max_new_tokens": int(args.intent_max_new_tokens),
            "debug_intent_focus": bool(args.debug_intent_focus),
            "debug_intent_focus_limit": int(args.debug_intent_focus_limit),
            "debug_intent_focus_context_chars": int(args.debug_intent_focus_context_chars),
            "debug_intent_focus_rows": int(len(intent_focus_debug_rows)),
            "debug_intent_focus_file": intent_focus_debug_path,
            "label_source": "gold",
            "l2_normalize": bool(effective_l2_normalize),
            "min_intent_samples": int(args.min_intent_samples),
            "top_intents": int(args.top_intents),
            "all_intents": bool(args.all_intents),
            "include_unknown_intent": bool(args.include_unknown_intent),
            "distance_rank_step": int(args.distance_rank_step),
            "annotate_labels": bool(args.annotate_labels),
            "device": args.device,
            "distributed": bool(dist_enabled),
            "world_size": int(world_size),
            "rank": int(rank),
            "local_rank": int(local_rank),
            "num_samples": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "num_unique_intents": len(counts),
            "num_intents_sampled_by_rank_step": len(sampled_order),
            "visualization_filtered_by_rank_step": bool(use_rank_filtered_viz),
            "num_visualization_samples": int(viz_features.shape[0]),
            "num_visualization_intents": len(viz_counts),
            "visualization_top_intents": int(viz_top_k),
            "heatmap_vmin": float(heatmap_vmin) if heatmap_vmin is not None else None,
            "heatmap_vmax": float(heatmap_vmax) if heatmap_vmax is not None else None,
            "heatmap_gamma": float(args.heatmap_gamma),
            "heatmap_scale_source": heatmap_scale_source,
            "heatmap_scale_file": args.heatmap_scale_file,
            "embeddings_requested": requested_embeddings,
            "embeddings_computed": sorted(list(projections.keys())),
            "embedding_notes": embedding_notes,
            "random_state": int(args.random_state),
            "tsne_perplexity": float(args.tsne_perplexity),
            "tsne_learning_rate": float(args.tsne_learning_rate),
            "tsne_n_iter": int(args.tsne_n_iter),
            "tsne_init": str(args.tsne_init),
            "umap_n_neighbors": int(args.umap_n_neighbors),
            "umap_min_dist": float(args.umap_min_dist),
            "umap_metric": str(args.umap_metric),
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
        save_intent_mean_distance_csv(
            os.path.join(out_dir, "intent_mean_distance_ranking.csv"),
            mean_rows,
        )
        save_intent_mean_distance_csv(
            os.path.join(out_dir, "intent_mean_distance_rankstep.csv"),
            sampled_rows,
        )
        save_distance_table_csv(
            os.path.join(out_dir, "centroid_distance_rankstep_table.csv"),
            sampled_order,
            sampled_dist,
        )
        plot_distance_gradient_heatmap(
            ordered_intents=sampled_order,
            distance_matrix=sampled_dist,
            out_path=os.path.join(out_dir, "centroid_distance_rankstep_heatmap.png"),
            title=f"Centroid Distance Heatmap (rank-step={int(args.distance_rank_step)})",
            vmin=heatmap_vmin,
            vmax=heatmap_vmax,
            heatmap_gamma=float(args.heatmap_gamma),
        )

        title = f"Intent Feature Map ({args.pipeline}/{args.task_mode})"
        if "pca" in projections:
            plot_pca_scatter(
                projection=projections["pca"],
                intents=viz_intents,
                counts=viz_counts,
                out_path=os.path.join(out_dir, "pca_scatter_by_intent.png"),
                title=title,
                explained_ratio=explained_ratio,
                top_k=viz_top_k,
                show_label_text=bool(args.annotate_labels),
            )
        if "tsne" in projections:
            plot_embedding_scatter(
                projection=projections["tsne"],
                intents=viz_intents,
                counts=viz_counts,
                out_path=os.path.join(out_dir, "tsne_scatter_by_intent.png"),
                title=title,
                x_label="t-SNE-1",
                y_label="t-SNE-2",
                subtitle="t-SNE-2D",
                top_k=viz_top_k,
                show_label_text=bool(args.annotate_labels),
            )
        if "umap" in projections:
            plot_embedding_scatter(
                projection=projections["umap"],
                intents=viz_intents,
                counts=viz_counts,
                out_path=os.path.join(out_dir, "umap_scatter_by_intent.png"),
                title=title,
                x_label="UMAP-1",
                y_label="UMAP-2",
                subtitle="UMAP-2D",
                top_k=viz_top_k,
                show_label_text=bool(args.annotate_labels),
            )
        plot_centroid_heatmap(
            valid_intents=stats["valid_intents"],
            distance_matrix=stats["distance_matrix"],
            counts=stats["counts"],
            out_path=os.path.join(out_dir, "centroid_distance_heatmap.png"),
            top_k=int(args.top_intents),
            vmin=heatmap_vmin,
            vmax=heatmap_vmax,
            heatmap_gamma=float(args.heatmap_gamma),
        )

        print(f"Saved analysis dir: {out_dir}")
        print(f"- {os.path.join(out_dir, 'features.npz')}")
        print(f"- {os.path.join(out_dir, 'metadata.jsonl')}")
        if intent_focus_debug_path:
            print(f"- {intent_focus_debug_path}")
        print(f"- {os.path.join(out_dir, 'intent_stats.csv')}")
        print(f"- {os.path.join(out_dir, 'centroid_distances.csv')}")
        print(f"- {os.path.join(out_dir, 'intent_mean_distance_ranking.csv')}")
        print(f"- {os.path.join(out_dir, 'intent_mean_distance_rankstep.csv')}")
        print(f"- {os.path.join(out_dir, 'centroid_distance_rankstep_table.csv')}")
        print(f"- {os.path.join(out_dir, 'centroid_distance_rankstep_heatmap.png')}")
        if "pca" in projections:
            print(f"- {os.path.join(out_dir, 'pca_scatter_by_intent.png')}")
        if "tsne" in projections:
            print(f"- {os.path.join(out_dir, 'tsne_scatter_by_intent.png')}")
        if "umap" in projections:
            print(f"- {os.path.join(out_dir, 'umap_scatter_by_intent.png')}")
        print(f"- {os.path.join(out_dir, 'centroid_distance_heatmap.png')}")
        print("Summary:")
        print(json.dumps(stats["summary"], ensure_ascii=False, indent=2))
    finally:
        finalize_distributed()


if __name__ == "__main__":
    main()
