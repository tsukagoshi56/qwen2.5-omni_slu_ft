#!/usr/bin/env python3
"""
Audio/text mixed SLU training and distributed inference.

- Prompting follows Experiment_RationaleCompare/prompts.py templates.
- Default training target style is C/R/J:
  C: ...
  R: ...
  J: {"scenario": "...", "action": "...", "entities": [...]}
- Prediction parsing prioritizes the JSON after "J:".
"""

import argparse
import contextlib
import datetime
import glob
import importlib
import inspect
import json
import logging
import os
import random
import re
import signal
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import transformers as hf_transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
try:
    from transformers import Qwen2AudioForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen2AudioForConditionalGeneration = None
try:
    from transformers import AudioFlamingo3ForConditionalGeneration
except Exception:  # pragma: no cover
    AudioFlamingo3ForConditionalGeneration = None
from common import build_db_definitions, load_metadata

try:
    import librosa
except ImportError:
    librosa = None

try:
    import jiwer

    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

try:
    import numpy as np
except ImportError:
    np = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
AUDIO_ENCODER_MODULE_NAME_HINTS = (
    "audio_tower",
    "audio_encoder",
    "speech_encoder",
    "audio_backbone",
    "audio_model",
    "audio",
)
PROJECTOR_MODULE_NAME_HINTS = (
    "multi_modal_projector",
    "multimodal_projector",
    "audio_projector",
    "mm_projector",
)
_PROCESSOR_TOKENIZER_REGISTRY: Dict[int, Any] = {}
AUDIO_FLAMINGO2_DEFAULT_SPACE_REPO = "nvidia/audio-flamingo-2"
AUDIO_FLAMINGO2_DEFAULT_LM_PATH_BY_MODEL = {
    "nvidia/audio-flamingo-2": "Qwen/Qwen2.5-3B",
    "nvidia/audio-flamingo-2-0.5b": "Qwen/Qwen2.5-0.5B",
    "nvidia/audio-flamingo-2-1.5b": "Qwen/Qwen2.5-1.5B",
}
_TERMINATION_SIGNAL_COUNT = 0


class ProcessorWithTokenizerProxy:
    def __init__(self, base_processor: Any, tokenizer: Any):
        self._base_processor = base_processor
        self.tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_processor, name)

    def __call__(self, *args, **kwargs):
        return self._base_processor(*args, **kwargs)


def _is_tokenizer_like(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return False
    has_decode = hasattr(value, "decode") or hasattr(value, "batch_decode")
    has_tokenizer_api = callable(value) or has_decode
    return bool(has_tokenizer_api and has_decode)


def _coerce_tokenizer_candidate(value: Any) -> Optional[Any]:
    if _is_tokenizer_like(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            candidate = _coerce_tokenizer_candidate(item)
            if candidate is not None:
                return candidate
    if isinstance(value, dict):
        preferred_keys = (
            "tokenizer",
            "tokenizers",
            "text_tokenizer",
            "text_tokenizers",
            "lm_tokenizer",
            "language_tokenizer",
            "decoder_tokenizer",
        )
        for key in preferred_keys:
            if key in value:
                candidate = _coerce_tokenizer_candidate(value.get(key))
                if candidate is not None:
                    return candidate
        for item in value.values():
            candidate = _coerce_tokenizer_candidate(item)
            if candidate is not None:
                return candidate
    return None


def _extract_tokenizer_from_processor(processor: Any) -> Optional[Any]:
    preferred_attrs = (
        "tokenizer",
        "tokenizers",
        "text_tokenizer",
        "text_tokenizers",
        "lm_tokenizer",
        "language_tokenizer",
        "_tokenizer",
    )
    for attr in preferred_attrs:
        try:
            candidate = _coerce_tokenizer_candidate(getattr(processor, attr, None))
        except Exception:
            candidate = None
        if candidate is not None:
            return candidate

    try:
        mapping = vars(processor)
    except Exception:
        mapping = {}
    for key, value in mapping.items():
        if "tokenizer" not in str(key).lower():
            continue
        candidate = _coerce_tokenizer_candidate(value)
        if candidate is not None:
            return candidate
    return None


def _load_tokenizer_with_fallbacks(model_name_or_path: str) -> Tuple[Any, str]:
    attempts = (
        ("use_fast=False", {"trust_remote_code": True, "use_fast": False}),
        ("use_fast=True", {"trust_remote_code": True, "use_fast": True}),
        ("default", {"trust_remote_code": True}),
    )
    errors: List[str] = []
    for label, kwargs in attempts:
        try:
            return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs), label
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    detail = " | ".join(errors) if errors else "unknown"
    raise RuntimeError(
        f"AutoTokenizer loading failed for '{model_name_or_path}' across fallbacks. Details: {detail}"
    )


def load_audio_or_raise(audio_path: str, sr: int) -> Tuple[Any, int]:
    if librosa is None:
        raise ModuleNotFoundError(
            "librosa is required for audio processing. "
            "Install librosa for train/eval/test with audio, or run --recover_only."
        )
    return librosa.load(audio_path, sr=sr)


def configure_reproducibility(
    seed: int,
    *,
    strict_determinism: bool,
    deterministic_warn_only: bool,
) -> None:
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    hf_transformers.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(bool(strict_determinism), warn_only=bool(deterministic_warn_only))
    except TypeError:
        if strict_determinism:
            torch.use_deterministic_algorithms(True)

    pyhash_seed = os.environ.get("PYTHONHASHSEED", "")
    if pyhash_seed != str(seed):
        logger.warning(
            "PYTHONHASHSEED=%s differs from --seed=%d. "
            "For full reproducibility across Python hash-based ordering, launch with PYTHONHASHSEED=%d.",
            pyhash_seed,
            seed,
            seed,
        )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    if np is not None:
        np.random.seed(worker_seed)


def _infer_model_family(model_name_or_path: str) -> str:
    name_lc = str(model_name_or_path or "").lower()
    if "audio-flamingo-2" in name_lc:
        return "audio-flamingo-2"
    if "music-flamingo" in name_lc:
        return "music-flamingo"
    if "audio-flamingo-3" in name_lc or "flamingo" in name_lc:
        return "flamingo"
    if "voxtral" in name_lc:
        return "voxtral"
    if "qwen" in name_lc:
        return "qwen"
    return "other"


class AudioFlamingo2ProcessorProxy:
    def __init__(
        self,
        *,
        tokenizer: Any,
        clap_config: Dict[str, Any],
        sampling_rate: int,
        max_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.clap_config = dict(clap_config or {})
        self.sampling_rate = int(sampling_rate)
        self.max_tokens = int(max_tokens)
        self.audio_flamingo2 = True

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def save_pretrained(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(save_dir)


def _normalize_repo_like_name(value: str) -> str:
    return str(value or "").strip().lower()


def _repo_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "_", _normalize_repo_like_name(value))


def _resolve_audio_flamingo2_snapshot_or_local(
    *,
    repo_or_dir: str,
    cache_root: str,
    repo_type: Optional[str] = None,
    local_files_only: bool = False,
    allow_patterns: Optional[List[str]] = None,
) -> str:
    path = os.path.abspath(os.path.expanduser(str(repo_or_dir or "").strip()))
    if path and os.path.isdir(path):
        return path

    repo_id = str(repo_or_dir or "").strip()
    if not repo_id:
        raise ValueError("Audio-Flamingo-2 loader received an empty repo_or_dir.")

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to auto-download Audio-Flamingo-2 assets. "
            "Install with: pip install -U huggingface_hub"
        ) from exc

    local_dir = os.path.join(cache_root, _repo_slug(repo_id))
    cache_preexisting = os.path.isdir(local_dir) and any(True for _ in os.scandir(local_dir))
    logger.info(
        "AF2 snapshot resolve start: repo_id=%s repo_type=%s local_files_only=%s cache_dir=%s cache_preexisting=%s",
        repo_id,
        repo_type or "model",
        bool(local_files_only),
        local_dir,
        cache_preexisting,
    )
    download_kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "local_files_only": bool(local_files_only),
    }
    if repo_type:
        download_kwargs["repo_type"] = repo_type
    if allow_patterns:
        download_kwargs["allow_patterns"] = allow_patterns
    t0 = time.perf_counter()
    try:
        snapshot_path = snapshot_download(**download_kwargs)
    except TypeError:
        # Backward compatibility for older huggingface_hub that may reject allow_patterns/local_dir args.
        fallback_kwargs = {"repo_id": repo_id, "local_files_only": bool(local_files_only)}
        if repo_type:
            fallback_kwargs["repo_type"] = repo_type
        snapshot_path = snapshot_download(**fallback_kwargs)
    elapsed = time.perf_counter() - t0
    logger.info(
        "AF2 snapshot resolve done: repo_id=%s path=%s elapsed=%.1fs",
        repo_id,
        os.path.abspath(snapshot_path),
        elapsed,
    )
    return os.path.abspath(snapshot_path)


def _load_yaml_file_or_raise(yaml_path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required for Audio-Flamingo-2 configs. Install with: pip install -U pyyaml"
        ) from exc
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at '{yaml_path}' must be a mapping.")
    return data


def _audio_flamingo2_pick_safe_ckpt_dir_or_raise(model_dir: str) -> str:
    candidates = [
        os.path.join(model_dir, "safe_ckpt"),
        model_dir,
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "metadata.json")):
            return c
        if glob.glob(os.path.join(c, "*.safetensors")):
            return c
    raise FileNotFoundError(
        "Audio-Flamingo-2 safe checkpoint directory was not found. "
        f"Tried: {candidates}"
    )


def _audio_flamingo2_pick_clap_ckpt_or_raise(model_dir: str, checkpoint_name_hint: str) -> str:
    candidates: List[str] = []
    clap_dir = os.path.join(model_dir, "clap_ckpt")
    if checkpoint_name_hint:
        candidates.append(os.path.join(clap_dir, checkpoint_name_hint))
        candidates.append(os.path.join(model_dir, checkpoint_name_hint))
    candidates.extend(glob.glob(os.path.join(clap_dir, "*.pt")))
    candidates.extend(glob.glob(os.path.join(model_dir, "*.pt")))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Audio-Flamingo-2 CLAP checkpoint (.pt) was not found under model directory."
    )


def _audio_flamingo2_pick_lm_path(
    model_name_or_path: str,
    *,
    override_lm_path: str,
    fallback_lm_path: str,
) -> str:
    if str(override_lm_path or "").strip():
        return str(override_lm_path).strip()
    key = _normalize_repo_like_name(model_name_or_path)
    for repo_name, lm_path in AUDIO_FLAMINGO2_DEFAULT_LM_PATH_BY_MODEL.items():
        if _normalize_repo_like_name(repo_name) == key:
            return lm_path
    if "audio-flamingo-2-0.5" in key:
        return AUDIO_FLAMINGO2_DEFAULT_LM_PATH_BY_MODEL["nvidia/audio-flamingo-2-0.5b"]
    if "audio-flamingo-2-1.5" in key:
        return AUDIO_FLAMINGO2_DEFAULT_LM_PATH_BY_MODEL["nvidia/audio-flamingo-2-1.5b"]
    return str(fallback_lm_path or AUDIO_FLAMINGO2_DEFAULT_LM_PATH_BY_MODEL["nvidia/audio-flamingo-2"]).strip()


@contextlib.contextmanager
def _audio_flamingo2_disable_weights_only_torch_load() -> Iterator[None]:
    # PyTorch 2.6+ defaults torch.load(..., weights_only=True). AF2/CLAP legacy checkpoints
    # may require full unpickling, so force weights_only=False during AF2 factory construction.
    original_torch_load = torch.load
    try:
        supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        supports_weights_only = False
    if not supports_weights_only:
        yield
        return

    def _torch_load_no_weights_only(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _torch_load_no_weights_only
    try:
        yield
    finally:
        torch.load = original_torch_load


def _audio_flamingo2_load_state_dict_or_raise(safe_ckpt_dir: str) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError(
            "safetensors is required for Audio-Flamingo-2 checkpoints. "
            "Install with: pip install -U safetensors"
        ) from exc

    metadata_path = os.path.join(safe_ckpt_dir, "metadata.json")
    chunk_names: List[str] = []
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if isinstance(metadata, dict):
            chunk_names = [str(k) for k in metadata.keys()]
        elif isinstance(metadata, list):
            chunk_names = [str(x) for x in metadata]

    if not chunk_names:
        chunk_names = [os.path.basename(p) for p in glob.glob(os.path.join(safe_ckpt_dir, "*.safetensors"))]

    if not chunk_names:
        raise FileNotFoundError(
            f"No safetensors chunks found in '{safe_ckpt_dir}'."
        )

    checkpoint: Dict[str, torch.Tensor] = {}
    for chunk_name in chunk_names:
        chunk_name = str(chunk_name).strip()
        if not chunk_name:
            continue
        chunk_path = os.path.join(safe_ckpt_dir, chunk_name)
        if not os.path.isfile(chunk_path) and not chunk_name.endswith(".safetensors"):
            chunk_path = os.path.join(safe_ckpt_dir, f"{chunk_name}.safetensors")
        if not os.path.isfile(chunk_path):
            raise FileNotFoundError(f"Audio-Flamingo-2 checkpoint chunk not found: {chunk_path}")
        checkpoint.update(load_file(chunk_path))
    return checkpoint


def load_audio_flamingo2_bundle_or_raise(
    *,
    model_name_or_path: str,
    space_repo_or_dir: str,
    cache_dir: str,
    local_files_only: bool,
    override_lm_path: str,
    override_tokenizer_path: str,
) -> Tuple[Any, Any, Any]:
    cache_root = os.path.abspath(os.path.expanduser(str(cache_dir or "~/.cache/huggingface/audio_flamingo2")))
    os.makedirs(cache_root, exist_ok=True)
    logger.info(
        "AF2 bundle load start: model=%s space_repo=%s cache_root=%s local_files_only=%s",
        model_name_or_path,
        space_repo_or_dir,
        cache_root,
        bool(local_files_only),
    )

    space_dir = _resolve_audio_flamingo2_snapshot_or_local(
        repo_or_dir=space_repo_or_dir,
        cache_root=cache_root,
        repo_type="space",
        local_files_only=local_files_only,
        allow_patterns=["configs/**", "src/**", "*.py", "requirements*.txt"],
    )
    model_dir = _resolve_audio_flamingo2_snapshot_or_local(
        repo_or_dir=model_name_or_path,
        cache_root=cache_root,
        repo_type=None,
        local_files_only=local_files_only,
    )
    logger.info("AF2 resolved dirs: space_dir=%s model_dir=%s", space_dir, model_dir)

    infer_yaml = os.path.join(space_dir, "configs", "inference.yaml")
    if not os.path.isfile(infer_yaml):
        raise FileNotFoundError(f"Audio-Flamingo-2 inference config not found: {infer_yaml}")
    logger.info("AF2 loading inference config: %s", infer_yaml)
    cfg = _load_yaml_file_or_raise(infer_yaml)
    model_config = dict(cfg.get("model_config") or {})
    clap_config = dict(cfg.get("clap_config") or {})
    train_config = dict(cfg.get("train_config") or {})

    if not model_config:
        raise ValueError("Audio-Flamingo-2 inference.yaml has empty model_config.")

    lm_path = _audio_flamingo2_pick_lm_path(
        model_name_or_path,
        override_lm_path=override_lm_path,
        fallback_lm_path=str(model_config.get("lang_encoder_path") or ""),
    )
    tokenizer_path = str(override_tokenizer_path or "").strip() or lm_path
    model_config["lang_encoder_path"] = lm_path
    model_config["tokenizer_path"] = tokenizer_path
    model_config["cache_dir"] = cache_root

    clap_checkpoint_hint = str(clap_config.get("checkpoint") or "").strip()
    clap_checkpoint = _audio_flamingo2_pick_clap_ckpt_or_raise(model_dir, clap_checkpoint_hint)
    clap_config["checkpoint"] = clap_checkpoint
    logger.info("AF2 CLAP checkpoint: %s", clap_checkpoint)

    inserted_paths: List[str] = []
    for candidate in (space_dir, os.path.join(space_dir, "src")):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
            inserted_paths.append(candidate)
    try:
        try:
            factory_mod = importlib.import_module("src.factory")
        except ModuleNotFoundError as exc:
            if exc.name in {"my_laion_clap", "src.my_laion_clap"}:
                raise RuntimeError(
                    "Audio-Flamingo-2 source import failed: my_laion_clap is missing. "
                    "Delete partial cache and re-run so the full Space source is downloaded. "
                    f"cache_dir={cache_root}"
                ) from exc
            raise
        create_model_and_transforms = getattr(factory_mod, "create_model_and_transforms", None)
        if create_model_and_transforms is None:
            raise AttributeError("src.factory.create_model_and_transforms is missing.")
        logger.info("AF2 factory build start: create_model_and_transforms(...)")
        t0 = time.perf_counter()
        with _audio_flamingo2_disable_weights_only_torch_load():
            created = create_model_and_transforms(
                clap_config=clap_config,
                use_local_files=bool(local_files_only),
                **model_config,
            )
        logger.info("AF2 factory build done: elapsed=%.1fs", time.perf_counter() - t0)
    finally:
        for inserted in inserted_paths:
            try:
                sys.path.remove(inserted)
            except ValueError:
                pass

    if not isinstance(created, (tuple, list)) or len(created) < 2:
        raise RuntimeError("Audio-Flamingo-2 factory did not return (model, tokenizer).")
    model = created[0]
    tokenizer = _coerce_tokenizer_candidate(created[1]) or _coerce_tokenizer_candidate(created)
    if tokenizer is None:
        raise RuntimeError("Audio-Flamingo-2 tokenizer was not returned by factory.")

    safe_ckpt_dir = _audio_flamingo2_pick_safe_ckpt_dir_or_raise(model_dir)
    logger.info("AF2 loading safetensors from: %s", safe_ckpt_dir)
    t0 = time.perf_counter()
    state_dict = _audio_flamingo2_load_state_dict_or_raise(safe_ckpt_dir)
    logger.info("AF2 safetensors loaded: tensors=%d elapsed=%.1fs", len(state_dict), time.perf_counter() - t0)
    t0 = time.perf_counter()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("AF2 model.load_state_dict done: elapsed=%.1fs", time.perf_counter() - t0)
    if missing:
        logger.warning("Audio-Flamingo-2 load_state_dict missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Audio-Flamingo-2 load_state_dict unexpected keys: %d", len(unexpected))

    sampling_rate = int(clap_config.get("sampling_rate") or 16000)
    max_tokens = int(train_config.get("max_tokens") or model_config.get("max_tokens") or 4096)
    processor = AudioFlamingo2ProcessorProxy(
        tokenizer=tokenizer,
        clap_config=clap_config,
        sampling_rate=sampling_rate,
        max_tokens=max_tokens,
    )
    return model, processor, tokenizer


def _is_audio_flamingo2_processor(processor: Any) -> bool:
    return bool(getattr(processor, "audio_flamingo2", False))


def _audio_flamingo2_prompt_text(prompt_text: str) -> str:
    text = str(prompt_text or "").strip()
    if "<audio>" not in text.lower():
        text = f"<audio>{text}"
    return text


def _audio_flamingo2_window_params(clap_config: Dict[str, Any]) -> Tuple[int, int, int, float]:
    window_seconds = float(clap_config.get("window_length", 1.0) or 1.0)
    overlap_seconds = float(clap_config.get("window_overlap", 0.2) or 0.2)
    max_windows = int(clap_config.get("max_num_window", 16) or 16)
    sampling_rate = int(clap_config.get("sampling_rate", 16000) or 16000)
    return max_windows, sampling_rate, max(1, int(window_seconds * sampling_rate)), overlap_seconds


def _audio_flamingo2_make_audio_windows_or_silence(
    *,
    audio_path: Optional[str],
    clap_config: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_windows, sampling_rate, window_length, overlap_seconds = _audio_flamingo2_window_params(clap_config)
    overlap_length = int(max(0.0, overlap_seconds) * sampling_rate)
    stride = max(1, window_length - overlap_length)
    clip_duration = (window_length + (max_windows - 1) * stride) / float(sampling_rate)

    if audio_path:
        try:
            audio, _ = load_audio_or_raise(audio_path, sr=sampling_rate)
        except Exception:
            audio = None
    else:
        audio = None

    if audio is None:
        audio = torch.zeros((window_length,), dtype=torch.float32).numpy()

    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    if audio_tensor.dim() == 0:
        audio_tensor = audio_tensor.unsqueeze(0)
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.reshape(-1)
    max_samples = int(clip_duration * sampling_rate)
    if audio_tensor.numel() > max_samples:
        audio_tensor = audio_tensor[:max_samples]

    num_windows = int(max(1, min(max_windows, (audio_tensor.numel() - window_length) // stride + 1)))
    full_length = window_length + (num_windows - 1) * stride
    if audio_tensor.numel() < full_length:
        pad_len = full_length - audio_tensor.numel()
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_len), value=0.0)

    windows: List[torch.Tensor] = []
    for i in range(num_windows):
        start = i * stride
        end = start + window_length
        windows.append(audio_tensor[start:end].unsqueeze(0))
    audio_x = torch.cat(windows, dim=0)
    audio_mask = torch.ones((num_windows,), dtype=torch.float32)
    return audio_x, audio_mask


def _audio_flamingo2_pad_audio_windows(
    audio_windows_list: List[torch.Tensor],
    audio_masks_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not audio_windows_list:
        raise RuntimeError("Audio-Flamingo-2 collator received no audio windows to pad.")
    max_windows = max(int(x.shape[0]) for x in audio_windows_list)
    window_length = int(audio_windows_list[0].shape[1])

    padded_audio: List[torch.Tensor] = []
    padded_masks: List[torch.Tensor] = []
    for audio_x, mask_x in zip(audio_windows_list, audio_masks_list):
        cur_windows = int(audio_x.shape[0])
        if cur_windows < max_windows:
            pad_audio = torch.zeros((max_windows - cur_windows, window_length), dtype=audio_x.dtype)
            audio_x = torch.cat([audio_x, pad_audio], dim=0)
            pad_mask = torch.zeros((max_windows - cur_windows,), dtype=mask_x.dtype)
            mask_x = torch.cat([mask_x, pad_mask], dim=0)
        elif cur_windows > max_windows:
            audio_x = audio_x[:max_windows]
            mask_x = mask_x[:max_windows]
        padded_audio.append(audio_x)
        padded_masks.append(mask_x)

    audio_batch = torch.stack(padded_audio, dim=0)
    mask_batch = torch.stack(padded_masks, dim=0)
    return audio_batch, mask_batch


def _audio_flamingo2_labels_from_input_ids(
    input_ids: torch.Tensor,
    *,
    tokenizer: Any,
    ignore_index: int,
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = ignore_index
    media_token_id = tokenizer.convert_tokens_to_ids("<audio>")
    labels[labels == media_token_id] = ignore_index

    sep_token_id = tokenizer.sep_token_id
    eoc_token_id = tokenizer.convert_tokens_to_ids("<|endofchunk|>")
    eos_token_id = tokenizer.eos_token_id
    for i in range(labels.shape[0]):
        should_mask = True
        for j in range(labels.shape[1]):
            token_id = labels[i, j].item()
            if should_mask and token_id != eos_token_id:
                labels[i, j] = ignore_index
            if token_id == sep_token_id:
                should_mask = False
            elif token_id == eoc_token_id:
                should_mask = True

        tail = labels.shape[1] - 1
        while tail >= 0 and labels[i, tail].item() not in [ignore_index, eos_token_id, tokenizer.pad_token_id, eoc_token_id]:
            labels[i, tail] = ignore_index
            tail -= 1
    return labels


def load_audio_model_from_pretrained(
    model_name_or_path: str,
    *,
    torch_dtype: torch.dtype,
    trust_remote_code: bool = True,
):
    def _optional_transformers_class(*names: str) -> Optional[Any]:
        for name in names:
            cls = getattr(hf_transformers, name, None)
            if cls is not None:
                return cls
        return None

    def _append_attempt(
        attempts_list: List[Tuple[str, Any]],
        loader_name: str,
        loader_cls: Any,
        seen_loader_ids: set,
    ) -> None:
        if loader_cls is None:
            return
        key = id(loader_cls)
        if key in seen_loader_ids:
            return
        seen_loader_ids.add(key)
        attempts_list.append((loader_name, loader_cls))

    family = _infer_model_family(model_name_or_path)
    model_name_lc = str(model_name_or_path or "").lower()

    # Qwen path: mirror the stable script behavior and avoid AutoModel fallbacks.
    if family == "qwen":
        qwen_errors: List[str] = []
        if "qwen2.5-omni" in model_name_lc:
            qwen_omni_cls = _optional_transformers_class(
                "Qwen2_5OmniForConditionalGeneration",
                "Qwen2_5OmniForCausalLM",
                "Qwen2OmniForConditionalGeneration",
                "Qwen2OmniForCausalLM",
            )
            if qwen_omni_cls is not None:
                try:
                    return qwen_omni_cls.from_pretrained(
                        model_name_or_path,
                        torch_dtype=torch_dtype,
                        trust_remote_code=trust_remote_code,
                    )
                except Exception as exc:
                    qwen_errors.append(f"{qwen_omni_cls.__name__}: {exc}")

        if Qwen2AudioForConditionalGeneration is not None:
            try:
                return Qwen2AudioForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as exc:
                qwen_errors.append(f"Qwen2AudioForConditionalGeneration: {exc}")
        else:
            qwen_errors.append("Qwen2AudioForConditionalGeneration is unavailable in current transformers.")

        detail = " | ".join(qwen_errors) if qwen_errors else "no qwen loader available"
        raise RuntimeError(
            f"Failed to load Qwen model '{model_name_or_path}' with Qwen loaders only. Details: {detail}"
        )

    attempts: List[Tuple[str, Any]] = []
    seen_loader_ids = set()

    if family == "flamingo":
        _append_attempt(
            attempts,
            "AudioFlamingo3ForConditionalGeneration",
            AudioFlamingo3ForConditionalGeneration,
            seen_loader_ids,
        )
        music_flamingo_cls = _optional_transformers_class(
            "MusicFlamingoForConditionalGeneration",
            "MusicFlamingoForCausalLM",
        )
        _append_attempt(
            attempts,
            getattr(music_flamingo_cls, "__name__", "MusicFlamingo*"),
            music_flamingo_cls,
            seen_loader_ids,
        )
        _append_attempt(attempts, "AutoModelForCausalLM", AutoModelForCausalLM, seen_loader_ids)
        _append_attempt(attempts, "AutoModel", AutoModel, seen_loader_ids)

    elif family == "music-flamingo":
        music_flamingo_cls = _optional_transformers_class(
            "MusicFlamingoForConditionalGeneration",
            "MusicFlamingoForCausalLM",
        )
        _append_attempt(
            attempts,
            getattr(music_flamingo_cls, "__name__", "MusicFlamingo*"),
            music_flamingo_cls,
            seen_loader_ids,
        )
        _append_attempt(
            attempts,
            "AudioFlamingo3ForConditionalGeneration",
            AudioFlamingo3ForConditionalGeneration,
            seen_loader_ids,
        )
        _append_attempt(attempts, "AutoModelForCausalLM", AutoModelForCausalLM, seen_loader_ids)
        _append_attempt(attempts, "AutoModel", AutoModel, seen_loader_ids)

    elif family == "voxtral":
        voxtral_cls = _optional_transformers_class(
            "VoxtralForConditionalGeneration",
            "VoxtralForCausalLM",
        )
        _append_attempt(
            attempts,
            getattr(voxtral_cls, "__name__", "Voxtral*"),
            voxtral_cls,
            seen_loader_ids,
        )
        _append_attempt(attempts, "AutoModelForCausalLM", AutoModelForCausalLM, seen_loader_ids)
        _append_attempt(attempts, "AutoModel", AutoModel, seen_loader_ids)

    else:
        _append_attempt(attempts, "AutoModelForCausalLM", AutoModelForCausalLM, seen_loader_ids)
        _append_attempt(attempts, "AutoModel", AutoModel, seen_loader_ids)
        _append_attempt(
            attempts,
            "Qwen2AudioForConditionalGeneration",
            Qwen2AudioForConditionalGeneration,
            seen_loader_ids,
        )
        _append_attempt(
            attempts,
            "AudioFlamingo3ForConditionalGeneration",
            AudioFlamingo3ForConditionalGeneration,
            seen_loader_ids,
        )

    errors: List[str] = []
    for loader_name, loader_cls in attempts:
        try:
            return loader_cls.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            errors.append(f"{loader_name}: {exc}")

    detail = " | ".join(errors) if errors else "no loader available"
    raise RuntimeError(
        f"Failed to load model '{model_name_or_path}' with audio-capable loaders. Details: {detail}"
    )


def get_audio_sampling_rate_or_raise(processor: Any, model_name_or_path: str) -> int:
    sampling_rate = None
    candidates = [
        getattr(processor, "feature_extractor", None),
        getattr(processor, "audio_processor", None),
        processor,
    ]
    for obj in candidates:
        if obj is None:
            continue
        sr = getattr(obj, "sampling_rate", None)
        if sr is not None:
            sampling_rate = sr
            break
    if sampling_rate is None:
        raise ValueError(
            f"Model '{model_name_or_path}' is not audio-ready in this script. "
            "Audio input is mandatory, so use an audio-capable checkpoint."
        )
    try:
        return int(sampling_rate)
    except Exception as exc:
        raise ValueError(
            f"Invalid audio sampling rate '{sampling_rate}' for model '{model_name_or_path}'."
        ) from exc


def ensure_processor_tokenizer_or_raise(processor: Any, model_name_or_path: str) -> Tuple[Any, Any]:
    tokenizer = _extract_tokenizer_from_processor(processor)
    if tokenizer is None:
        try:
            tokenizer, load_mode = _load_tokenizer_with_fallbacks(model_name_or_path)
            logger.warning(
                "Processor %s has no tokenizer; loaded AutoTokenizer(%s) with %s.",
                type(processor).__name__,
                model_name_or_path,
                load_mode,
            )
        except Exception as exc:
            raise ValueError(
                f"Processor '{type(processor).__name__}' has no tokenizer and AutoTokenizer loading failed "
                f"for '{model_name_or_path}': {exc}"
            ) from exc

    if getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
        if eos_token_id is not None:
            tokenizer.pad_token_id = eos_token_id
        if getattr(tokenizer, "pad_token", None) is None:
            raise ValueError(
                f"Tokenizer for '{model_name_or_path}' has neither pad_token nor eos_token. "
                "Set tokenizer special tokens before training."
            )

    _PROCESSOR_TOKENIZER_REGISTRY[id(processor)] = tokenizer
    try:
        if getattr(processor, "tokenizer", None) is None:
            setattr(processor, "tokenizer", tokenizer)
    except Exception:
        processor = ProcessorWithTokenizerProxy(processor, tokenizer)
    _PROCESSOR_TOKENIZER_REGISTRY[id(processor)] = tokenizer
    return processor, tokenizer


def get_tokenizer_or_raise(processor: Any) -> Any:
    tokenizer = _extract_tokenizer_from_processor(processor)
    if tokenizer is not None:
        return tokenizer
    tokenizer = _coerce_tokenizer_candidate(_PROCESSOR_TOKENIZER_REGISTRY.get(id(processor)))
    if tokenizer is not None:
        return tokenizer
    base = getattr(processor, "_base_processor", None)
    if base is not None:
        tokenizer = _coerce_tokenizer_candidate(_PROCESSOR_TOKENIZER_REGISTRY.get(id(base)))
        if tokenizer is not None:
            return tokenizer
    raise AttributeError(
        f"No tokenizer is registered for processor type {type(processor).__name__}. "
        "Call ensure_processor_tokenizer_or_raise() before use."
    )


def attach_tokenizer_to_model_for_compat(model: Any, tokenizer: Any) -> None:
    # Some multimodal generation paths expect `model.tokenizer` to exist.
    try:
        if getattr(model, "tokenizer", None) is None:
            setattr(model, "tokenizer", tokenizer)
    except Exception:
        pass


def _chat_template_owner_or_raise(processor: Any) -> Any:
    if hasattr(processor, "apply_chat_template"):
        return processor
    tokenizer = get_tokenizer_or_raise(processor)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer
    raise RuntimeError(
        f"Neither processor ({type(processor).__name__}) nor tokenizer has apply_chat_template."
    )


def render_chat_template_as_text_or_raise(
    processor: Any,
    messages: List[Dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    owner = _chat_template_owner_or_raise(processor)
    attempts = [
        {"tokenize": False, "add_generation_prompt": add_generation_prompt},
        {"tokenize": False},
        {"add_generation_prompt": add_generation_prompt},
    ]
    errors: List[str] = []
    for kwargs in attempts:
        try:
            out = owner.apply_chat_template(messages, **kwargs)
            if isinstance(out, str):
                return out
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")
    detail = " | ".join(errors) if errors else "no chat-template variant accepted"
    raise RuntimeError(f"Failed to render chat template as text. Details: {detail}")


def _normalize_tokenized_chat_output(out: Any) -> Optional[Dict[str, torch.Tensor]]:
    if out is None:
        return None
    if torch.is_tensor(out):
        ids = out
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.long)}
    if isinstance(out, list):
        try:
            ids = torch.tensor(out, dtype=torch.long)
        except Exception:
            return None
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.long)}
    if not isinstance(out, dict):
        return None

    normalized: Dict[str, torch.Tensor] = {}
    for key, value in out.items():
        if torch.is_tensor(value):
            tensor = value
        else:
            try:
                tensor = torch.tensor(value)
            except Exception:
                continue
        if key in {"input_ids", "attention_mask"} and tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        normalized[key] = tensor

    if "input_ids" not in normalized:
        return None
    if "attention_mask" not in normalized:
        normalized["attention_mask"] = torch.ones_like(normalized["input_ids"], dtype=torch.long)
    return normalized


def _apply_chat_template_tokenized_or_raise(
    processor: Any,
    messages: List[Dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
) -> Dict[str, torch.Tensor]:
    owner = _chat_template_owner_or_raise(processor)
    attempts = [
        {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "return_dict": True,
            "return_tensors": "pt",
        },
        {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "return_tensors": "pt",
        },
        {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        },
    ]
    errors: List[str] = []
    for kwargs in attempts:
        try:
            out = owner.apply_chat_template(messages, **kwargs)
            normalized = _normalize_tokenized_chat_output(out)
            if normalized is not None:
                return normalized
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")
    detail = " | ".join(errors) if errors else "no tokenized chat-template variant accepted"
    raise RuntimeError(f"Failed to tokenize chat template. Details: {detail}")


def _audio_chat_content_variants(prompt_text: str, audio_ref: str) -> List[List[Dict[str, Any]]]:
    ref = str(audio_ref or "placeholder")
    return [
        [{"type": "text", "text": prompt_text}, {"type": "audio", "path": ref}],
        [{"type": "audio", "path": ref}, {"type": "text", "text": prompt_text}],
        [{"type": "audio", "audio_url": ref}, {"type": "text", "text": prompt_text}],
        [{"type": "audio", "url": ref}, {"type": "text", "text": prompt_text}],
        [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": prompt_text}],
    ]


def _infer_audio_input_mode(
    model_name_or_path: str,
    *,
    processor: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> str:
    probes: List[str] = [str(model_name_or_path or "")]
    for obj in (processor, tokenizer, getattr(processor, "tokenizer", None) if processor is not None else None):
        if obj is None:
            continue
        probes.append(type(obj).__name__.lower())
        for attr in ("name_or_path", "_name_or_path", "model_type"):
            try:
                value = getattr(obj, attr, None)
            except Exception:
                value = None
            if isinstance(value, str) and value.strip():
                probes.append(value.lower())
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            for attr in ("name_or_path", "_name_or_path", "model_type"):
                try:
                    value = getattr(cfg, attr, None)
                except Exception:
                    value = None
                if isinstance(value, str) and value.strip():
                    probes.append(value.lower())
    family = _infer_model_family(" ".join(probes))
    # For Qwen/Flamingo families, keep audio waveform path explicit via processor(..., audio=[...]).
    # Tokenized chat-template can succeed without real audio features on some model versions.
    if family in {"qwen", "flamingo", "music-flamingo", "voxtral", "audio-flamingo-2"}:
        return "processor_audio"
    return "tokenized_chat_template"


def decode_token_ids(processor: Any, token_ids: torch.Tensor) -> str:
    if hasattr(processor, "decode"):
        try:
            return processor.decode(token_ids, skip_special_tokens=True)
        except Exception:
            pass
    tokenizer = get_tokenizer_or_raise(processor)
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    raise RuntimeError("Neither processor.decode nor tokenizer.decode is available.")


def batch_decode_token_ids(processor: Any, token_ids: torch.Tensor) -> List[str]:
    if hasattr(processor, "batch_decode"):
        try:
            return processor.batch_decode(token_ids, skip_special_tokens=True)
        except Exception:
            pass
    tokenizer = get_tokenizer_or_raise(processor)
    if hasattr(tokenizer, "batch_decode"):
        return tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    return [decode_token_ids(processor, ids) for ids in token_ids]


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


def _drop_unsupported_feature_masks_for_generate(model: Any, inputs: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    candidate_keys = ("input_features_mask", "feature_attention_mask")
    present_keys = [k for k in candidate_keys if k in inputs]
    if not present_keys:
        return inputs, []

    try:
        target = model.module if hasattr(model, "module") and getattr(model, "module") is not None else model
        accepted: set = set()

        forward = getattr(target, "forward", None)
        if callable(forward):
            sig = inspect.signature(forward)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                return inputs, []
            accepted.update(sig.parameters.keys())

        prep = getattr(target, "prepare_inputs_for_generation", None)
        if callable(prep):
            sig = inspect.signature(prep)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                return inputs, []
            accepted.update(sig.parameters.keys())

        if not accepted:
            return inputs, []
    except Exception:
        return inputs, []

    filtered = dict(inputs)
    dropped: List[str] = []
    for key in present_keys:
        if key not in accepted:
            filtered.pop(key, None)
            dropped.append(key)
    return filtered, dropped


def _generate_with_retry_drop_unused_kwargs(
    model: Any,
    *,
    net_inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
    pad_token_id: Optional[int],
) -> Tuple[torch.Tensor, List[str]]:
    working = dict(net_inputs)
    working = _cast_floating_tensors_to_model_dtype(working, model)
    dropped: List[str] = []
    working, dropped_pre = _drop_unsupported_feature_masks_for_generate(model, working)
    dropped.extend(dropped_pre)
    for _ in range(6):
        try:
            with torch.no_grad():
                out = model.generate(
                    **working,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=pad_token_id,
                )
            return out, dropped
        except Exception as exc:
            unused = _extract_unused_model_kwargs_from_exception(exc)
            if not unused:
                raise
            removed_here: List[str] = []
            for key in list(working.keys()):
                key_norm = key.replace("_", " ").strip().lower()
                for candidate in unused:
                    cand_norm = candidate.replace("_", " ").strip().lower()
                    if key_norm == cand_norm:
                        working.pop(key, None)
                        removed_here.append(key)
                        break
            if not removed_here:
                raise
            dropped.extend(removed_here)
    raise RuntimeError(
        f"generate() retry limit exceeded while dropping unsupported kwargs. dropped={dropped}"
    )


def _model_floating_dtype(model: Any) -> Optional[torch.dtype]:
    try:
        dtype = getattr(model, "dtype", None)
    except Exception:
        dtype = None
    if isinstance(dtype, torch.dtype) and torch.is_floating_point(torch.empty((), dtype=dtype)):
        return dtype
    try:
        for p in model.parameters():
            if torch.is_tensor(p) and torch.is_floating_point(p):
                return p.dtype
    except Exception:
        pass
    return None


def _safe_model_device(model: Any) -> torch.device:
    if model is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device_attr = getattr(model, "device", None)
        if device_attr is not None:
            return torch.device(device_attr)
    except Exception:
        pass

    for iterator_name in ("parameters", "buffers"):
        iterator = getattr(model, iterator_name, None)
        if not callable(iterator):
            continue
        try:
            for tensor in iterator():
                if torch.is_tensor(tensor):
                    return tensor.device
        except Exception:
            continue

    wrapped = getattr(model, "module", None)
    if wrapped is not None and wrapped is not model:
        return _safe_model_device(wrapped)

    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _cast_floating_tensors_to_model_dtype(
    inputs: Dict[str, Any],
    model: Any,
) -> Dict[str, Any]:
    target_dtype = _model_floating_dtype(model)
    if target_dtype is None:
        return inputs
    casted: Dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value) and torch.is_floating_point(value) and value.dtype != target_dtype:
            casted[key] = value.to(dtype=target_dtype)
        else:
            casted[key] = value
    return casted


def _ensure_feature_masks_for_generation(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "input_features" not in inputs:
        return inputs

    f_mask = inputs.get("feature_attention_mask")
    i_mask = inputs.get("input_features_mask")
    if torch.is_tensor(f_mask) and not torch.is_tensor(i_mask):
        inputs["input_features_mask"] = f_mask
        return inputs
    if torch.is_tensor(i_mask) and not torch.is_tensor(f_mask):
        inputs["feature_attention_mask"] = i_mask
        return inputs
    if torch.is_tensor(f_mask) and torch.is_tensor(i_mask):
        return inputs

    feat = inputs["input_features"]
    if not torch.is_tensor(feat):
        return inputs
    if feat.dim() >= 2:
        bsz = int(feat.shape[0])
        tlen = int(max(feat.shape[1:]))
    else:
        bsz = 1
        tlen = int(feat.shape[0])
    mask = torch.ones((bsz, tlen), dtype=torch.long, device=feat.device)
    inputs["feature_attention_mask"] = mask
    inputs["input_features_mask"] = mask
    return inputs


def configure_audio_trainability(
    model: Any,
    *,
    train_audio_encoder: bool,
    freeze_projector: bool,
) -> Tuple[int, int]:
    def _resolve_attr_path(root: Any, attr_path: str) -> Optional[Any]:
        cur = root
        for part in attr_path.split("."):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur

    def _apply_requires_grad_from_modules(
        root: Any,
        module_paths: Tuple[str, ...],
        requires_grad: bool,
        seen_param_ids: Optional[set] = None,
    ) -> int:
        if seen_param_ids is None:
            seen_param_ids = set()
        before = len(seen_param_ids)
        for path in module_paths:
            module = _resolve_attr_path(root, path)
            if module is None or not hasattr(module, "parameters"):
                continue
            try:
                params = module.parameters()
            except Exception:
                continue
            for param in params:
                pid = id(param)
                if pid in seen_param_ids:
                    continue
                param.requires_grad_(requires_grad)
                seen_param_ids.add(pid)
        return len(seen_param_ids) - before

    audio_param_ids = set()
    projector_param_ids = set()
    for name, param in model.named_parameters():
        lname = str(name).lower()
        if any(hint in lname for hint in AUDIO_ENCODER_MODULE_NAME_HINTS):
            param.requires_grad_(bool(train_audio_encoder))
            audio_param_ids.add(id(param))
        if any(hint in lname for hint in PROJECTOR_MODULE_NAME_HINTS):
            param.requires_grad_(not bool(freeze_projector))
            projector_param_ids.add(id(param))

    _apply_requires_grad_from_modules(
        model,
        (
            "audio_tower",
            "model.audio_tower",
            "audio_encoder",
            "model.audio_encoder",
            "speech_encoder",
            "model.speech_encoder",
            "audio_model",
            "model.audio_model",
        ),
        bool(train_audio_encoder),
        seen_param_ids=audio_param_ids,
    )
    _apply_requires_grad_from_modules(
        model,
        (
            "multi_modal_projector",
            "model.multi_modal_projector",
            "multimodal_projector",
            "model.multimodal_projector",
            "audio_projector",
            "model.audio_projector",
            "mm_projector",
            "model.mm_projector",
        ),
        not bool(freeze_projector),
        seen_param_ids=projector_param_ids,
    )
    audio_matches = len(audio_param_ids)
    projector_matches = len(projector_param_ids)
    return audio_matches, projector_matches


SYSTEM_PROMPT_TEXT = (
    'System: Predict SLU labels from transcript.'
)
SYSTEM_PROMPT_AUDIO = (
    'System: Predict SLU labels from audio.'
)
OUTPUT_SCHEMA = (
    '{"Intent": "<scenario>_<action>", "entities": '
    '[{"type": "<entity_type>", "filler": "<entity_value>"}, ...]}'
)
PROMPT_OUTPUT_FORMAT = (
    "Output Format:\n"
    "C: Intent candidates: intent1 | intent2 | intent3; Slot candidates: slot_type1(value1|value2) | slot_type2\n"
    "R: label1!reason1; label2!reason2; ...\n"
    f"J: {OUTPUT_SCHEMA}"
)
PROMPT_OUTPUT_FORMAT_CANDIDATES_ONLY = (
    "Output Format:\n"
    "C: Intent candidates: intent1 | intent2 | intent3; Slot candidates: slot_type1(value1|value2) | slot_type2\n"
    f"J: {OUTPUT_SCHEMA}"
)
PROMPT_OUTPUT_FORMAT_JSON_ONLY = (
    "Output Format:\n"
    f"J: {OUTPUT_SCHEMA}"
)
PROMPT_DB_DEFINITIONS = "Intents: (none)\nSlot Types: (none)"


def set_prompt_db_definitions(db_definitions: str) -> None:
    global PROMPT_DB_DEFINITIONS
    text = str(db_definitions or "").strip()
    PROMPT_DB_DEFINITIONS = text if text else "Intents: (none)\nSlot Types: (none)"


def setup_file_logging(log_path: str) -> None:
    if not log_path:
        return
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    abs_log_path = os.path.abspath(log_path)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            existing = os.path.abspath(getattr(handler, "baseFilename", ""))
            if existing == abs_log_path:
                return
    file_handler = logging.FileHandler(abs_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger.addHandler(file_handler)


def format_nbest(candidates: List[str], max_items: int = 5) -> str:
    if not candidates:
        return "- (none)"
    lines = []
    for idx, text in enumerate(candidates[:max_items], start=1):
        lines.append(f"- {idx}. {text}")
    return "\n".join(lines)


def normalize_rationale_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        # Keep short and stable for prompting.
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def candidate_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "transcript", "hypothesis", "value"):
            text = value.get(key)
            if text is not None:
                return str(text).strip()
    return str(value).strip()


def _strip_prefix_case_insensitive(text: str, prefixes: List[str]) -> str:
    value = str(text or "").strip()
    lower = value.lower()
    for prefix in prefixes:
        p = prefix.lower()
        if lower.startswith(p):
            return value[len(prefix):].strip()
    return value


def _clean_intent_candidate(value: str) -> str:
    text = str(value or "").strip().strip("`'\" ")
    if not text:
        return ""
    text = re.sub(
        r"^\s*(?:intent\s*candidates?|intentcandidates?|intent\s*candidate|intentcandidate|intents?)\s*[:：\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace(":", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(
        r"^(?:intent_?candidates?|intent_?candidate|intents?)_+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"_+", "_", text).strip("_ ")
    if text.lower() in {"intent", "intents", "intentcandidate", "intentcandidates"}:
        return ""
    return text


def _sanitize_c_line(c_line: str) -> str:
    if not str(c_line or "").startswith("C:"):
        return str(c_line or "")
    body = c_line.split(":", 1)[1].strip() if ":" in c_line else ""
    if not body:
        return "C: (none)"

    parts = [p.strip() for p in body.split(";", 1)]
    intent_part = parts[0] if parts else ""
    slot_part = parts[1] if len(parts) > 1 else ""

    intent_part = _strip_prefix_case_insensitive(
        intent_part,
        ["Intent candidates:", "Intent candidate:", "Intent:", "Intents:"],
    )
    slot_part = _strip_prefix_case_insensitive(
        slot_part,
        ["Slot candidates:", "Slot candidate:", "Slot:", "Slots:"],
    )

    intents_raw = [x.strip() for x in intent_part.split("|") if x.strip()]
    intents_cleaned: List[str] = []
    seen = set()
    for cand in intents_raw:
        cleaned = _clean_intent_candidate(cand)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        intents_cleaned.append(cleaned)

    intent_part_new = " | ".join(intents_cleaned) if intents_cleaned else "(none)"
    if slot_part:
        return f"C: Intent candidates: {intent_part_new}; Slot candidates: {slot_part}"
    return f"C: Intent candidates: {intent_part_new}"


def build_prompt_text(item: Dict[str, Any], include_transcript: bool = False) -> str:
    transcript = str(item.get("transcript", "") or "").strip()
    task_mode = str(item.get("task_mode", "cot") or "cot").strip().lower()
    if task_mode == "json_only":
        output_format = PROMPT_OUTPUT_FORMAT_JSON_ONLY
    elif task_mode == "candidates":
        output_format = PROMPT_OUTPUT_FORMAT_CANDIDATES_ONLY
    else:
        output_format = PROMPT_OUTPUT_FORMAT

    if include_transcript and transcript:
        return (
            f"{SYSTEM_PROMPT_TEXT}\n\n"
            "[Input Data]\n"
            f"- Transcript: {transcript}\n\n"
            f"{output_format}"
        )
    return (
        f"{SYSTEM_PROMPT_AUDIO}\n\n"
        "[Input Data]\n"
        "- Audio: <AUDIO>\n\n"
        f"{output_format}"
    )


def build_training_target(
    rationale_text: str,
    final_json: str,
    use_rationale: bool = True,
    use_candidates: bool = True,
) -> str:
    if not use_rationale and not use_candidates:
        return f"J: {final_json}"
    rationale = (rationale_text or "").strip()
    if not rationale:
        return f"J: {final_json}"

    lines = [line.strip() for line in rationale.splitlines() if line.strip()]
    lines = [_sanitize_c_line(line) if line.startswith("C:") else line for line in lines]
    has_c = any(line.startswith("C:") for line in lines)
    has_j = any(line.startswith("J:") for line in lines)
    has_r = any(line.startswith("R:") for line in lines)

    if use_rationale and has_c and has_r and has_j:
        return "\n".join(lines)

    if not use_rationale:
        c_line = next((line for line in lines if line.startswith("C:")), "")
        if not c_line:
            c_line = "C: (none)"
        return "\n".join([c_line, f"J: {final_json}"])

    if has_j:
        return "\n".join(lines)
    return "\n".join(lines + [f"J: {final_json}"])


# ==============================================================================
# 1. Data Loading
# ==============================================================================


def resolve_audio_path(
    audio_root: str,
    filename: str,
    return_searched_paths: bool = False,
) -> Any:
    if not filename:
        if return_searched_paths:
            return None, []
        return None

    filename = str(filename).strip()
    searched_paths: List[str] = []
    if os.path.isabs(filename):
        searched_paths.append(filename)
        if os.path.exists(filename):
            if return_searched_paths:
                return filename, searched_paths
            return filename

    basename = os.path.basename(filename)
    candidates = [
        os.path.join(audio_root, filename),
        os.path.join(audio_root, basename),
        os.path.join(audio_root, "slurp_real", filename),
        os.path.join(audio_root, "slurp_real", basename),
        os.path.join("slurp", "audio", "slurp_real", filename),
        os.path.join("slurp", "audio", "slurp_real", basename),
    ]
    for path in candidates:
        searched_paths.append(path)
        if os.path.exists(path):
            if return_searched_paths:
                return path, searched_paths
            return path

    # Last-resort fallback: build one-time basename index under audio_root.
    if audio_root and os.path.isdir(audio_root):
        if not hasattr(resolve_audio_path, "_audio_index"):
            index: Dict[str, str] = {}
            for root, _, files in os.walk(audio_root):
                for fn in files:
                    if fn not in index:
                        index[fn] = os.path.join(root, fn)
            resolve_audio_path._audio_index = index
        fallback = resolve_audio_path._audio_index.get(basename)
        if fallback and os.path.exists(fallback):
            searched_paths.append(f"[indexed] {fallback}")
            if return_searched_paths:
                return fallback, searched_paths
            return fallback

    if not hasattr(resolve_audio_path, "_debug_count"):
        resolve_audio_path._debug_count = 0
    if resolve_audio_path._debug_count < 10:
        logger.warning("Could not find %s. Checked: %s", filename, candidates)
        resolve_audio_path._debug_count += 1
    if return_searched_paths:
        return None, searched_paths
    return None


def parse_entities(raw_entities: Any) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    if not isinstance(raw_entities, list):
        return results

    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue
        ent_type = str(ent.get("type", "")).strip()
        filler = ent.get("filler")
        if filler is None:
            filler = ent.get("filter")
        if filler is None:
            filler = ent.get("value")
        if filler is None:
            filler = ""
        results.append({"type": ent_type, "filler": str(filler)})
    return results


def get_dict_value_ci(obj: Dict[str, Any], *names: str) -> Any:
    if not isinstance(obj, dict):
        return None
    for name in names:
        if name in obj:
            return obj[name]
    lowered: Dict[str, Any] = {}
    for k, v in obj.items():
        lowered[str(k).strip().lower()] = v
    for name in names:
        key = str(name).strip().lower()
        if key in lowered:
            return lowered[key]
    return None


def intent_to_scenario_action(intent: str) -> Tuple[str, str]:
    intent = (intent or "").strip()
    if "_" in intent:
        scenario, action = intent.split("_", 1)
        return scenario, action
    return "", ""


def extract_target_obj(record: Dict[str, Any]) -> Dict[str, Any]:
    final_obj = record.get("final")
    if not isinstance(final_obj, dict):
        final_obj = {}

    scenario = str(get_dict_value_ci(final_obj, "scenario") or "").strip()
    action = str(get_dict_value_ci(final_obj, "action") or "").strip()

    intent = str(get_dict_value_ci(final_obj, "intent") or "").strip()
    if (not scenario or not action) and intent:
        inferred_scenario, inferred_action = intent_to_scenario_action(intent)
        scenario = scenario or inferred_scenario
        action = action or inferred_action

    entities = parse_entities(get_dict_value_ci(final_obj, "entities") or [])

    return {
        "scenario": scenario,
        "action": action,
        "entities": entities,
    }


def parse_json_like(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    for _ in range(2):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, str):
            text = obj.strip()
            continue
        return obj
    return None


def pick_first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", ""))
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)
    return str(content)


def extract_messages_texts(record: Dict[str, Any]) -> Tuple[str, str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return "", ""
    user_text = ""
    assistant_text = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        text = content_to_text(msg.get("content"))
        if role == "user" and not user_text:
            user_text = text
        elif role == "assistant" and not assistant_text:
            assistant_text = text
    return user_text, assistant_text


def extract_candidates_from_user_text(user_text: str) -> List[str]:
    if not user_text:
        return []
    results: List[str] = []
    for raw_line in user_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^(?:[-*]|\d+[.)])\s*(.+)$", line)
        if m:
            cand = m.group(1).strip()
            if cand and len(cand.split()) >= 2:
                results.append(cand)
    return results[:10]


def extract_filename_from_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text.strip():
        return ""

    patterns = [
        r'"filename"\s*:\s*"([^"]+)"',
        r'\\"filename\\"\s*:\s*\\"([^\\"]+)\\"',
        r"'filename'\s*:\s*'([^']+)'",
        r'filename\s*[:=]\s*["\']([^"\']+)["\']',
        r"(audio-[A-Za-z0-9_-]+\.(?:flac|wav|mp3|m4a|ogg))",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return str(m.group(1)).strip()
    return ""


def extract_sample_id(record: Dict[str, Any], fallback_index: int) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    rationale_obj = parse_json_like(record.get("rationale_text"))
    rationale_meta = (
        rationale_obj.get("meta", {})
        if isinstance(rationale_obj, dict) and isinstance(rationale_obj.get("meta"), dict)
        else {}
    )

    sample_id = pick_first_nonempty(
        record.get("id"),
        record.get("slurp_id"),
        record.get("uid"),
        record.get("uuid"),
        meta.get("id"),
        meta.get("slurp_id"),
        rationale_obj.get("id") if isinstance(rationale_obj, dict) else None,
        rationale_meta.get("id"),
        rationale_meta.get("slurp_id"),
    )
    if sample_id:
        return sample_id
    return f"row_{fallback_index}"


def extract_filename(record: Dict[str, Any]) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    if not meta:
        meta_obj = parse_json_like(record.get("meta"))
        if isinstance(meta_obj, dict):
            meta = meta_obj
    rationale_obj = parse_json_like(record.get("rationale_text"))
    rationale_meta = (
        rationale_obj.get("meta", {})
        if isinstance(rationale_obj, dict) and isinstance(rationale_obj.get("meta"), dict)
        else {}
    )

    recordings = record.get("recordings")
    rec_file = None
    if isinstance(recordings, list) and recordings and isinstance(recordings[0], dict):
        rec_file = recordings[0].get("file")
    audios = record.get("audios")
    audio0 = None
    if isinstance(audios, list) and audios:
        audio0 = audios[0]

    user_text, _ = extract_messages_texts(record)

    filename = pick_first_nonempty(
        record.get("filename"),
        record.get("file"),
        record.get("audio_filename"),
        record.get("audio_file"),
        meta.get("filename"),
        meta.get("file"),
        rationale_obj.get("filename") if isinstance(rationale_obj, dict) else None,
        rationale_obj.get("file") if isinstance(rationale_obj, dict) else None,
        rationale_meta.get("filename"),
        rationale_meta.get("file"),
        rec_file,
        audio0,
    )
    if filename:
        return filename

    # Last fallback for non-JSON rationale text containing filename snippets.
    return pick_first_nonempty(
        extract_filename_from_text(record.get("rationale_text")),
        extract_filename_from_text(record.get("rationale")),
        extract_filename_from_text(record.get("meta")),
        extract_filename_from_text(user_text),
        extract_filename_from_text(json.dumps(record, ensure_ascii=False)),
    )


def _normalize_loaded_obj_to_records(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, str):
        try:
            return _normalize_loaded_obj_to_records(json.loads(obj))
        except json.JSONDecodeError:
            return []
    if isinstance(obj, dict):
        for key in ("data", "items", "records", "examples"):
            maybe_list = obj.get(key)
            if isinstance(maybe_list, list):
                return [x for x in maybe_list if isinstance(x, dict)]

        # If this already looks like a single record, do not flatten dict values.
        record_like_keys = {
            "id",
            "slurp_id",
            "filename",
            "file",
            "audio_filename",
            "audio_file",
            "candidates",
            "final",
            "rationale_text",
            "meta",
            "recordings",
            "messages",
            "audios",
        }
        if any(k in obj for k in record_like_keys):
            return [obj]

        # Handle map-style datasets: {"1234": {...}, "1235": {...}, ...}
        # Keep this strict to avoid breaking one-sample dict records.
        keys = list(obj.keys())
        values = list(obj.values())
        if (
            len(obj) >= 2
            and all(isinstance(v, dict) for v in values)
            and all(re.fullmatch(r"[0-9]{1,8}", str(k)) for k in keys)
        ):
            return [v for v in values if isinstance(v, dict)]
        return [obj]
    if isinstance(obj, list):
        results: List[Dict[str, Any]] = []
        for x in obj:
            if isinstance(x, dict):
                results.append(x)
            elif isinstance(x, str):
                try:
                    parsed = json.loads(x)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    results.append(parsed)
        return results
    return []


def is_record_like_dict(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    record_like_keys = {
        "id",
        "slurp_id",
        "filename",
        "file",
        "audio_filename",
        "audio_file",
        "candidates",
        "final",
        "rationale_text",
        "meta",
        "recordings",
        "messages",
        "audios",
    }
    if any(k in record for k in record_like_keys):
        return True

    # Accept cases where only rationale text contains recoverable filename.
    if extract_filename(record):
        return True
    return False


def extract_target_obj_from_assistant(record: Dict[str, Any]) -> Dict[str, Any]:
    _, assistant_text = extract_messages_texts(record)
    parsed = parse_json_like(assistant_text)
    if isinstance(parsed, dict):
        if "scenario" in parsed or "action" in parsed or "entities" in parsed:
            return {
                "scenario": str(get_dict_value_ci(parsed, "scenario") or "").strip(),
                "action": str(get_dict_value_ci(parsed, "action") or "").strip(),
                "entities": parse_entities(get_dict_value_ci(parsed, "entities") or []),
            }
        if "final" in parsed:
            return extract_target_obj(parsed)
        intent = get_dict_value_ci(parsed, "intent")
        if intent is not None:
            scenario, action = intent_to_scenario_action(str(intent))
            return {
                "scenario": scenario,
                "action": action,
                "entities": parse_entities(get_dict_value_ci(parsed, "entities") or []),
            }
    return {"scenario": "", "action": "", "entities": []}


def load_rationale_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if not raw.strip():
        return []

    # 1) Try full-file JSON (supports JSON array/object files).
    try:
        obj = json.loads(raw)
        records = _normalize_loaded_obj_to_records(obj)
        if records:
            logger.info("Loaded %s as full JSON (%d records).", path, len(records))
            return records
    except json.JSONDecodeError:
        pass

    # 2) Fallback: parse as JSONL.
    records: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        for rec in _normalize_loaded_obj_to_records(obj):
            if is_record_like_dict(rec):
                records.append(rec)
    if records:
        logger.info("Loaded %s as JSONL (%d records).", path, len(records))
        return records

    # 3) Last fallback: parse as a stream of concatenated JSON objects.
    decoder = json.JSONDecoder()
    i = 0
    n = len(raw)
    while i < n:
        while i < n and raw[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, j = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            i += 1
            continue
        for rec in _normalize_loaded_obj_to_records(obj):
            if is_record_like_dict(rec):
                records.append(rec)
        i = j
    logger.info("Loaded %s as JSON stream fallback (%d records).", path, len(records))
    return records


def build_items_from_rationale_jsonl(
    jsonl_path: str,
    audio_dir: str,
    add_text_only: bool = False,
    text_only: bool = False,
    max_samples: Optional[int] = None,
    allow_text_fallback_when_audio_missing: bool = True,
    print_audio_search_paths: bool = False,
    audio_search_print_limit: int = 100,
    strict_audio_missing: bool = False,
    train_candidates_only: bool = False,
    train_json_only: bool = False,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    fallback_text_items: List[Dict[str, Any]] = []
    if not os.path.exists(jsonl_path):
        logger.warning("JSONL file not found: %s", jsonl_path)
        return items

    records = load_rationale_records(jsonl_path)

    parsed_rows = 0
    rows_with_filename = 0
    rows_with_audio = 0
    rows_missing_audio = 0
    row_parse_errors = 0

    for idx, data in enumerate(records):
        if max_samples is not None and len(items) >= max_samples:
            break
        try:
            if not isinstance(data, dict):
                continue
            if not is_record_like_dict(data):
                continue
            parsed_rows += 1

            # Extract messages and audios first
            user_text, assistant_text = extract_messages_texts(data)

            raw_audios = data.get("audios")
            
            # Check if this is a pre-formatted record (has messages and audios)
            is_preformatted = (
                isinstance(raw_audios, list) 
                and len(raw_audios) > 0 
                and assistant_text 
                and "candidates" not in data  # Assuming pre-formatted data might not have top-level candidates/final
            )

            # If pre-formatted, trust the messages/audios directly
            if is_preformatted or (raw_audios and assistant_text):
                # Use the first audio path
                filename = raw_audios[0]

                # Try to resolve generic info for logging/eval (optional)
                sample_id = extract_sample_id(data, fallback_index=parsed_rows)
                candidates = []  # Not needed for training if target is pre-built

                target_obj = extract_target_obj(data)
                if (
                    not target_obj.get("scenario")
                    and not target_obj.get("action")
                    and not target_obj.get("entities")
                ):
                    target_obj = extract_target_obj_from_assistant(data)
                final_json = json.dumps(target_obj, ensure_ascii=False)

                rationale_text = normalize_rationale_text(data.get("rationale_text"))
                if not rationale_text:
                    rationale_text = assistant_text.strip()
                target_str = build_training_target(
                    rationale_text,
                    final_json,
                    use_rationale=(not train_candidates_only and not train_json_only),
                    use_candidates=(not train_json_only),
                )

                # Prioritize explicit transcript fields
                transcript = pick_first_nonempty(
                    data.get("transcript"),
                    data.get("text"),
                    data.get("sentence"),
                )
            else:
                # --- Original Logic for Raw/Component Data ---
                sample_id = extract_sample_id(data, fallback_index=parsed_rows)
                filename = extract_filename(data)
                
                candidates = data.get("candidates", [])
                if not isinstance(candidates, list):
                    candidates = []
                candidates = [candidate_to_text(c) for c in candidates]
                candidates = [c for c in candidates if c]

                if not candidates:
                    candidates = extract_candidates_from_user_text(user_text)

                # Prioritize explicit transcript fields, fallback to candidates[0]
                transcript = pick_first_nonempty(
                    data.get("transcript"),
                    data.get("text"),
                    data.get("sentence"),
                    candidates[0] if candidates else ""
                )
                rationale_text = normalize_rationale_text(data.get("rationale_text"))

                target_obj = extract_target_obj(data)
                if not target_obj.get("scenario") and not target_obj.get("action") and not target_obj.get("entities"):
                    target_obj = extract_target_obj_from_assistant(data)

                final_json = json.dumps(target_obj, ensure_ascii=False)
                target_str = build_training_target(
                    rationale_text,
                    final_json,
                    use_rationale=(not train_candidates_only and not train_json_only),
                    use_candidates=(not train_json_only),
                )
            
            # Common file resolution
            if filename:
                audio_path, searched_paths = resolve_audio_path(
                    audio_dir,
                    filename,
                    return_searched_paths=True,
                )
            else:
                audio_path, searched_paths = None, []

            base_item = {
                "id": sample_id,
                "slurp_id": sample_id,
                "file": filename,
                "audio_path": audio_path,
                "transcript": transcript,
                "candidates": candidates,
                "rationale_text": rationale_text,
                "target": target_str,
                "target_obj": target_obj,
                "prompt_text": user_text.strip() if user_text else "",
                "task_mode": (
                    "json_only"
                    if train_json_only
                    else ("candidates" if train_candidates_only else "cot")
                ),
            }
            text_only_item = {**base_item, "audio_path": None}
            fallback_text_items.append(text_only_item)

            if text_only:
                items.append(text_only_item)
                continue

            if add_text_only:
                items.append(text_only_item)

            if audio_path:
                items.append(base_item)
                rows_with_audio += 1
                if print_audio_search_paths and rows_with_audio <= audio_search_print_limit:
                    print(f"[AUDIO_OK] id={sample_id} file={filename}")
                    print(f"  resolved={audio_path}")
            else:
                rows_missing_audio += 1
                if rows_missing_audio <= audio_search_print_limit:
                    if not filename:
                        print(f"[AUDIO_NG] id={sample_id} file=<empty> (filename parse failed)")
                    else:
                        print(f"[AUDIO_NG] id={sample_id} file={filename} (not found)")
                    for p in searched_paths:
                        print(f"  searched: {p}")
                if strict_audio_missing:
                    raise RuntimeError(
                        f"Audio not found for id={sample_id}, file={filename}. "
                        f"Searched paths: {searched_paths}"
                    )
        except Exception as exc:
            row_parse_errors += 1
            if row_parse_errors <= 20:
                head = str(data)
                if len(head) > 300:
                    head = head[:300] + "...(truncated)"
                logger.error(
                    "Row parse error at index=%d type=%s error=%s record_head=%s",
                    idx,
                    type(data).__name__,
                    exc,
                    head,
                )
            continue

    if (not add_text_only) and len(items) == 0 and parsed_rows > 0 and allow_text_fallback_when_audio_missing:
        logger.warning(
            "No audio could be resolved from %s. Falling back to text-only items (%d rows).",
            jsonl_path,
            len(fallback_text_items),
        )
        items.extend(fallback_text_items)

    logger.info(
        (
            "Loaded %s -> %d items "
            "(parsed_rows=%d, rows_with_filename=%d, rows_with_audio=%d, rows_missing_audio=%d, row_parse_errors=%d)"
        ),
        jsonl_path,
        len(items),
        parsed_rows,
        rows_with_filename,
        rows_with_audio,
        rows_missing_audio,
        row_parse_errors,
    )
    return items


class MixedDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {**self.items[idx], "original_idx": idx}


# ==============================================================================
# 2. Sampler
# ==============================================================================


class DistributedHomogeneousBatchSampler(Sampler):
    def __init__(
        self,
        dataset: MixedDataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = False,
        seed: int = 0,
        shuffle: bool = True,
        total_epochs: int = 1,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.shuffle = shuffle
        self.total_epochs = max(1, total_epochs)

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package")
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

        all_audio = [i for i, item in enumerate(dataset.items) if item.get("audio_path") is not None]
        all_text = [i for i, item in enumerate(dataset.items) if item.get("audio_path") is None]

        self.local_audio_indices = all_audio[self.rank :: self.num_replicas]
        self.local_text_indices = all_text[self.rank :: self.num_replicas]

    def __iter__(self) -> Iterator[List[int]]:
        g_static = torch.Generator()
        g_static.manual_seed(self.seed)

        audio_indices_tensor = torch.tensor(self.local_audio_indices)
        if self.shuffle and len(audio_indices_tensor) > 0:
            perm_static = torch.randperm(len(audio_indices_tensor), generator=g_static)
            shuffled_audio = audio_indices_tensor[perm_static]
        else:
            shuffled_audio = audio_indices_tensor

        total_audio_count = len(shuffled_audio)
        chunk_size = max(1, total_audio_count // self.total_epochs) if total_audio_count else 0
        current_chunk_idx = self.epoch % self.total_epochs

        start_idx = current_chunk_idx * chunk_size
        if current_chunk_idx == self.total_epochs - 1:
            end_idx = total_audio_count
        else:
            end_idx = start_idx + chunk_size

        active_audio_indices = shuffled_audio[start_idx:end_idx]
        active_text_indices = torch.tensor(self.local_text_indices)

        g_dynamic = torch.Generator()
        g_dynamic.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            if len(active_audio_indices) > 0:
                audio_perm = torch.randperm(len(active_audio_indices), generator=g_dynamic)
                audio_idxs = active_audio_indices[audio_perm].tolist()
            else:
                audio_idxs = []
            if len(active_text_indices) > 0:
                text_perm = torch.randperm(len(active_text_indices), generator=g_dynamic)
                text_idxs = active_text_indices[text_perm].tolist()
            else:
                text_idxs = []
        else:
            audio_idxs = active_audio_indices.tolist()
            text_idxs = active_text_indices.tolist()

        batches: List[List[int]] = []
        for i in range(0, len(audio_idxs), self.batch_size):
            batch = audio_idxs[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        for i in range(0, len(text_idxs), self.batch_size):
            batch = text_idxs[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        total_audio = len(self.local_audio_indices)
        chunk_size = total_audio // self.total_epochs if self.total_epochs > 0 else total_audio
        current_audio_len = chunk_size
        current_text_len = len(self.local_text_indices)
        if self.drop_last:
            audio_batches = current_audio_len // self.batch_size
            text_batches = current_text_len // self.batch_size
        else:
            audio_batches = (current_audio_len + self.batch_size - 1) // self.batch_size
            text_batches = (current_text_len + self.batch_size - 1) // self.batch_size
        return audio_batches + text_batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch


# ==============================================================================
# 3. Collator
# ==============================================================================


@dataclass
class AudioFlamingo2TrainCollator:
    processor: Any
    max_length: int = 4096
    ignore_index: int = -100
    debug: bool = False
    _print_count: int = 0

    def __post_init__(self):
        self._print_count = 0

    def _build_target_text(self, item: Dict[str, Any], is_audio_batch: bool) -> str:
        prompt_text = build_prompt_text(item, include_transcript=not is_audio_batch)
        prompt_text = _audio_flamingo2_prompt_text(prompt_text)
        tokenizer = get_tokenizer_or_raise(self.processor)
        sep = str(getattr(tokenizer, "sep_token", "") or "")
        if not sep:
            raise ValueError("Audio-Flamingo-2 tokenizer has no sep_token.")
        eos = str(getattr(tokenizer, "eos_token", "") or "")
        if not eos:
            raise ValueError("Audio-Flamingo-2 tokenizer has no eos_token.")
        return f"{prompt_text}{sep}{str(item.get('target', '')).strip()}<|endofchunk|>{eos}"

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise RuntimeError("AudioFlamingo2TrainCollator received an empty batch.")
        tokenizer = get_tokenizer_or_raise(self.processor)
        clap_config = dict(getattr(self.processor, "clap_config", {}) or {})
        if not clap_config:
            raise ValueError("Audio-Flamingo-2 processor is missing clap_config.")

        is_audio_batch = batch[0].get("audio_path") is not None
        text_inputs: List[str] = []
        audio_windows_list: List[torch.Tensor] = []
        audio_masks_list: List[torch.Tensor] = []

        for item in batch:
            text_inputs.append(self._build_target_text(item, is_audio_batch=is_audio_batch))
            audio_x, audio_mask = _audio_flamingo2_make_audio_windows_or_silence(
                audio_path=item.get("audio_path"),
                clap_config=clap_config,
            )
            audio_windows_list.append(audio_x)
            audio_masks_list.append(audio_mask)

            if self.debug and self._print_count < 3:
                logger.info(
                    "[AF2 DEBUG] sample=%s windows=%d text_len=%d",
                    item.get("id"),
                    int(audio_x.shape[0]),
                    len(text_inputs[-1]),
                )
                self._print_count += 1

        tokenized = tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=max(16, int(self.max_length)),
            return_tensors="pt",
        )
        lang_x = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(lang_x, dtype=torch.long)
        labels = _audio_flamingo2_labels_from_input_ids(
            lang_x,
            tokenizer=tokenizer,
            ignore_index=self.ignore_index,
        )

        audio_x, audio_x_mask = _audio_flamingo2_pad_audio_windows(audio_windows_list, audio_masks_list)
        return {
            "lang_x": lang_x,
            "attention_mask": attention_mask,
            "labels": labels,
            "audio_x": audio_x,
            "audio_x_mask": audio_x_mask,
        }


@dataclass
class AudioFlamingo2InferenceCollator:
    processor: Any
    max_length: int = 4096

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        if not batch:
            return {}
        tokenizer = get_tokenizer_or_raise(self.processor)
        clap_config = dict(getattr(self.processor, "clap_config", {}) or {})
        if not clap_config:
            raise ValueError("Audio-Flamingo-2 processor is missing clap_config.")

        is_audio_batch = batch[0].get("audio_path") is not None
        valid_items: List[Dict[str, Any]] = []
        net_inputs_list: List[Dict[str, torch.Tensor]] = []
        prompt_lens: List[int] = []

        for item in batch:
            try:
                prompt_text = build_prompt_text(item, include_transcript=not is_audio_batch)
                prompt_text = _audio_flamingo2_prompt_text(prompt_text)
                sep = str(getattr(tokenizer, "sep_token", "") or "")
                if not sep:
                    raise ValueError("Audio-Flamingo-2 tokenizer has no sep_token.")
                prompt = f"{prompt_text}{sep}"
                tok = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max(16, int(self.max_length)),
                )
                lang_x = tok["input_ids"]
                attention_mask = tok.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(lang_x, dtype=torch.long)

                audio_x, audio_mask = _audio_flamingo2_make_audio_windows_or_silence(
                    audio_path=item.get("audio_path"),
                    clap_config=clap_config,
                )
                net_inputs_list.append(
                    {
                        "lang_x": lang_x,
                        "attention_mask": attention_mask,
                        "audio_x": audio_x.unsqueeze(0),
                        "audio_x_mask": audio_mask.unsqueeze(0),
                    }
                )
                prompt_lens.append(int(lang_x.shape[1]))
                valid_items.append(item)
            except Exception as exc:
                logger.warning("Failed to build AF2 inference input for %s: %s", item.get("id"), exc)
                continue

        if not net_inputs_list:
            return {}
        return {"net_inputs_list": net_inputs_list, "items": valid_items, "prompt_lens": prompt_lens}


@dataclass
class SmartCollator:
    processor: Any
    max_length: int = 512
    ignore_index: int = -100
    debug: bool = False
    _print_count: int = 0
    _audio_fallback_warn_count: int = 0

    def __post_init__(self):
        self._print_count = 0
        self._audio_fallback_warn_count = 0

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        if len(batch) == 0:
            raise RuntimeError("SmartCollator received an empty batch.")
        is_audio_batch = batch[0].get("audio_path") is not None
        if is_audio_batch:
            out = self._collate_audio(batch)
        else:
            out = self._collate_text(batch)
        if not out or ("input_ids" not in out) or ("labels" not in out):
            raise RuntimeError("SmartCollator produced an invalid batch without input_ids/labels.")
        return out

    def _build_audio_chat(self, item: Dict[str, Any]) -> str:
        prompt_text = build_prompt_text(item)
        user_content = [
            {"type": "audio", "audio_url": "placeholder"},
            {"type": "text", "text": prompt_text},
        ]
        return render_chat_template_as_text_or_raise(
            self.processor,
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
        )

    def _build_text_chat(self, item: Dict[str, Any]) -> str:
        prompt_text = build_prompt_text(item, include_transcript=True)
        user_content = [{"type": "text", "text": prompt_text}]
        return render_chat_template_as_text_or_raise(
            self.processor,
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
        )

    def _collate_audio(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list, labels_list = [], []
        input_features_list, feature_mask_list = [], []

        tokenizer = get_tokenizer_or_raise(self.processor)
        sr = get_audio_sampling_rate_or_raise(self.processor, type(self.processor).__name__)
        eos_token = tokenizer.eos_token or "<|endoftext|>"

        for item in batch:
            if item.get("audio_path") is None:
                continue
            try:
                audio, _ = load_audio_or_raise(item["audio_path"], sr=sr)
            except Exception:
                continue

            text_input = self._build_audio_chat(item)
            full_text = text_input + item["target"] + eos_token

            if self.debug and self._print_count < 5:
                print(f"\n[DEBUG Visualizer] Audio Sample ID: {item.get('id')}")
                print(f"[DEBUG Visualizer] Input Prompt:\n{text_input}")
                print(f"[DEBUG Visualizer] Target:\n{item['target']}")
                self._print_count += 1

            inputs = self.processor(
                text=full_text,
                audio=[audio],
                sampling_rate=sr,
                return_tensors="pt",
            )
            prompt_inputs = self.processor(
                text=text_input,
                audio=[audio],
                sampling_rate=sr,
                return_tensors="pt",
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]

            ids = inputs["input_ids"][0]
            lbs = ids.clone()
            lbs[:prompt_len] = self.ignore_index

            input_ids_list.append(ids)
            labels_list.append(lbs)

            feat = inputs["input_features"]
            while feat.dim() > 2:
                feat = feat.squeeze(0)
            input_features_list.append(feat)

            f_mask = None
            if "input_features_mask" in inputs:
                f_mask = inputs["input_features_mask"]
            elif "feature_attention_mask" in inputs:
                f_mask = inputs["feature_attention_mask"]
            if f_mask is not None:
                while f_mask.dim() > 1:
                    f_mask = f_mask.squeeze(0)
                feature_mask_list.append(f_mask)

        if not input_ids_list:
            # Keep training alive when every audio sample in this batch fails decoding/processor parsing.
            # We degrade to text-only for this step instead of returning an empty dict.
            if self._audio_fallback_warn_count < 20:
                batch_ids = [str(x.get("id", "")) for x in batch[:8]]
                logger.warning(
                    "All audio samples in a batch were skipped; fallback to text-only. ids(head)=%s",
                    batch_ids,
                )
                self._audio_fallback_warn_count += 1
            text_fallback_batch = [{**item, "audio_path": None} for item in batch]
            return self._collate_text(text_fallback_batch)

        if feature_mask_list:
            feature_attention_mask = pad_sequence(feature_mask_list, batch_first=True, padding_value=0)
        else:
            feature_attention_mask = pad_sequence(
                [torch.ones(feat.shape[0], dtype=torch.long) for feat in input_features_list],
                batch_first=True,
                padding_value=0,
            )

        return {
            "input_ids": pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ),
            "labels": pad_sequence(
                labels_list,
                batch_first=True,
                padding_value=self.ignore_index,
            ),
            "attention_mask": pad_sequence(
                [torch.ones_like(ids) for ids in input_ids_list],
                batch_first=True,
                padding_value=0,
            ),
            "input_features": pad_sequence(
                input_features_list,
                batch_first=True,
                padding_value=0.0,
            ),
            "feature_attention_mask": feature_attention_mask,
            "input_features_mask": feature_attention_mask,
        }

    def _collate_text(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list, labels_list = [], []
        tokenizer = get_tokenizer_or_raise(self.processor)

        eos_token = tokenizer.eos_token or "<|endoftext|>"
        for item in batch:
            if item.get("audio_path") is not None:
                continue
            text_input = self._build_text_chat(item)
            full_text = text_input + item["target"] + eos_token

            if self.debug and self._print_count < 5:
                print(f"\n[DEBUG Visualizer] Text Sample ID: {item.get('id')}")
                print(f"[DEBUG Visualizer] Input Prompt:\n{text_input}")
                print(f"[DEBUG Visualizer] Target:\n{item['target']}")
                self._print_count += 1

            inputs = tokenizer(full_text, return_tensors="pt")
            prompt_inputs = tokenizer(text_input, return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]

            ids = inputs["input_ids"][0]
            lbs = ids.clone()
            lbs[:prompt_len] = self.ignore_index
            input_ids_list.append(ids)
            labels_list.append(lbs)

        if not input_ids_list:
            raise RuntimeError("Text collator produced an empty batch after filtering.")

        return {
            "input_ids": pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ),
            "labels": pad_sequence(
                labels_list,
                batch_first=True,
                padding_value=self.ignore_index,
            ),
            "attention_mask": pad_sequence(
                [torch.ones_like(ids) for ids in input_ids_list],
                batch_first=True,
                padding_value=0,
            ),
        }


# ==============================================================================
# 4. Trainer
# ==============================================================================


class CustomTrainer(Trainer):
    @staticmethod
    def _sanitize_model_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in inputs.items():
            if value is None:
                continue
            sanitized[key] = value

        if "input_ids" in sanitized and "attention_mask" not in sanitized:
            input_ids = sanitized["input_ids"]
            if torch.is_tensor(input_ids):
                sanitized["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)

        if (
            "input_features" in sanitized
            and "feature_attention_mask" not in sanitized
            and "input_features_mask" not in sanitized
        ):
            feat = sanitized["input_features"]
            if torch.is_tensor(feat):
                if feat.dim() >= 2:
                    bsz = int(feat.shape[0])
                    tlen = int(feat.shape[1])
                else:
                    bsz = 1
                    tlen = int(feat.shape[0])
                sanitized["feature_attention_mask"] = torch.ones(
                    (bsz, tlen),
                    dtype=torch.long,
                    device=feat.device,
                )
                sanitized["input_features_mask"] = sanitized["feature_attention_mask"]
        elif "feature_attention_mask" in sanitized and "input_features_mask" not in sanitized:
            sanitized["input_features_mask"] = sanitized["feature_attention_mask"]
        elif "input_features_mask" in sanitized and "feature_attention_mask" not in sanitized:
            sanitized["feature_attention_mask"] = sanitized["input_features_mask"]
        return sanitized

    @staticmethod
    def _drop_unsupported_feature_masks(model: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            target = model
            module = getattr(model, "module", None)
            if module is not None:
                target = module
            forward = getattr(target, "forward", None)
            if forward is None:
                return inputs
            sig = inspect.signature(forward)
            params = sig.parameters
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                return inputs
            supported = set(params.keys())
        except Exception:
            return inputs

        filtered = dict(inputs)
        for key in ("input_features_mask", "feature_attention_mask"):
            if key in filtered and key not in supported:
                filtered.pop(key, None)
        return filtered

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = self._sanitize_model_inputs(inputs)
        inputs = self._drop_unsupported_feature_masks(model, inputs)
        inputs = _cast_floating_tensors_to_model_dtype(inputs, model)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        seed = int(getattr(self.args, "seed", 0))
        batch_sampler = DistributedHomogeneousBatchSampler(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            drop_last=self.args.dataloader_drop_last,
            seed=seed,
            shuffle=True,
            total_epochs=int(self.args.num_train_epochs),
        )
        worker_gen = torch.Generator()
        worker_gen.manual_seed(seed)
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=_seed_worker,
            generator=worker_gen,
        )


# ==============================================================================
# 5. Callback
# ==============================================================================


class SampleGenerationCallback(TrainerCallback):
    def __init__(self, eval_items, processor, model, num_samples: int = 3, max_new_tokens: int = 4096):
        self.eval_items = eval_items
        self.processor = processor
        self.model = model
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return

        logger.info("\n\n*** Validation Sample Generation (Audio) ***")
        audio_items = [item for item in self.eval_items if item.get("audio_path") is not None]
        if not audio_items:
            logger.info("No audio items found in validation set.")
            return

        samples = random.sample(audio_items, min(self.num_samples, len(audio_items)))
        device = _safe_model_device(self.model)
        self.model.eval()
        sr = get_audio_sampling_rate_or_raise(self.processor, type(self.processor).__name__)

        for item in samples:
            try:
                audio, _ = load_audio_or_raise(item["audio_path"], sr=sr)
                prompt_text = build_prompt_text(item)
                user_content = [
                    {"type": "audio", "audio_url": "placeholder"},
                    {"type": "text", "text": prompt_text},
                ]
                text_input = render_chat_template_as_text_or_raise(
                    self.processor,
                    [{"role": "user", "content": user_content}],
                    add_generation_prompt=True,
                )

                inputs = self.processor(
                    text=text_input,
                    audio=[audio],
                    sampling_rate=sr,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
                if "input_ids" not in inputs:
                    tok = get_tokenizer_or_raise(self.processor)(text_input, return_tensors="pt")
                    if "input_ids" in tok:
                        inputs["input_ids"] = tok["input_ids"].to(device)
                        if "attention_mask" in tok:
                            inputs["attention_mask"] = tok["attention_mask"].to(device)
                if "attention_mask" not in inputs and "input_ids" in inputs:
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)
                inputs = _ensure_feature_masks_for_generation(inputs)

                output_ids, dropped = _generate_with_retry_drop_unused_kwargs(
                    self.model,
                    net_inputs=inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=getattr(get_tokenizer_or_raise(self.processor), "pad_token_id", None),
                )

                input_len = inputs["input_ids"].shape[1]
                generated_text = decode_token_ids(self.processor, output_ids[0][input_len:])
                clean_pred = clean_json_text(generated_text)

                logger.info("-" * 60)
                logger.info("File:       %s", item.get("file"))
                logger.info("Transcript: %s", item.get("transcript"))
                logger.info("Prediction: %s", clean_pred)
            except Exception as exc:
                logger.error("Failed to generate sample for %s: %s", item.get("file"), exc)

        logger.info("-" * 60 + "\n")
        self.model.train()


# ==============================================================================
# 6. Inference + Evaluation helpers
# ==============================================================================


def clean_json_text(text: str) -> str:
    text = text.strip()
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def _extract_labeled_tail(text: str, labels: List[str]) -> str:
    if not isinstance(text, str):
        return ""
    for label in labels:
        # Accept `J:`, `j:`, `J ：` and similar variants.
        pattern = rf"(?is)(?:^|\n)\s*{re.escape(label)}\s*[:：]\s*(.+)$"
        m = re.search(pattern, text)
        if m:
            return str(m.group(1)).strip()
    return ""


def _parse_first_json_dict(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    if not candidate:
        return None

    decoder = json.JSONDecoder()
    for probe in (clean_json_text(candidate), candidate):
        probe = probe.strip()
        if not probe:
            continue
        # Some model outputs include doubled braces from prompt examples.
        normalized_probe = probe.replace("{{", "{").replace("}}", "}")
        for target in (probe, normalized_probe):
            try:
                obj = json.loads(target)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

            # Find the first decodable JSON object anywhere in the text.
            for m in re.finditer(r"\{", target):
                start = m.start()
                try:
                    obj, _ = decoder.raw_decode(target[start:])
                except Exception:
                    continue
                if isinstance(obj, dict):
                    return obj
    return None


def _is_error_label(label: Dict[str, Any]) -> bool:
    if not isinstance(label, dict):
        return True
    scenario = str(label.get("scenario", "") or "").strip().lower()
    action = str(label.get("action", "") or "").strip().lower()
    entities = parse_entities(label.get("entities", []))
    return scenario == "error" and action == "error" and len(entities) == 0


def _label_info_score(label: Dict[str, Any]) -> int:
    if not isinstance(label, dict):
        return -1
    scenario = str(label.get("scenario", "") or "").strip()
    action = str(label.get("action", "") or "").strip()
    entities = parse_entities(label.get("entities", []))
    return int(bool(scenario)) + int(bool(action)) + int(bool(entities))


def parse_prediction_label(raw_output: str) -> Dict[str, Any]:
    default_obj = {"scenario": "error", "action": "error", "entities": []}

    text = str(raw_output or "")
    probes: List[str] = []

    j_tail = _extract_labeled_tail(text, ["J", "SLU", "FINAL", "Final", "Output"])
    if j_tail:
        probes.append(j_tail)
    probes.append(text)

    parsed = None
    for probe in probes:
        parsed = _parse_first_json_dict(probe)
        if isinstance(parsed, dict):
            break

    if not isinstance(parsed, dict):
        return default_obj

    # Extraction Logic
    # 1. Check for "final" wrapper FIRST (To handle: "final": {"intent": "..."})
    for wrapper_key in ("final", "Final", "j", "J", "output", "prediction", "result"):
        wrapped = parsed.get(wrapper_key)
        if isinstance(wrapped, dict):
            parsed = wrapped

    scenario = get_dict_value_ci(parsed, "scenario")
    action = get_dict_value_ci(parsed, "action")
    entities = get_dict_value_ci(parsed, "entities", "slots") or []

    # 2. If scenario/action are missing, try to parse "intent"
    if not scenario and not action:
        intent = get_dict_value_ci(parsed, "intent")
        if isinstance(intent, str):
            intent = intent.strip()
            if "_" in intent:
                # Split by FIRST underscore
                scenario, action = intent.split("_", 1)
            else:
                # Fallback: keep scenario empty, use intent as action
                scenario = ""
                action = intent

    return {
        "scenario": str(scenario or "").strip(),
        "action": str(action or "").strip(),
        "entities": parse_entities(entities),
    }


def recover_prediction_file(input_path: str, output_path: str) -> Dict[str, int]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"recover input not found: {input_path}")

    total = 0
    changed = 0
    intent_key_recovered = 0
    rows: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue

            old_s = str(row.get("scenario", "") or "").strip()
            old_a = str(row.get("action", "") or "").strip()
            old_e = parse_entities(row.get("entities", []))

            candidate_outputs: List[str] = []
            for key in (
                "raw_output",
                "prediction",
                "pred_text",
                "model_output",
                "output",
                "response",
                "assistant",
                "assistant_text",
                "text",
            ):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    candidate_outputs.append(value)

            parsed_candidates: List[Dict[str, Any]] = []
            for text in candidate_outputs:
                parsed = parse_prediction_label(text)
                if not _is_error_label(parsed):
                    parsed_candidates.append(parsed)

            parsed_from_row = parse_prediction_label(json.dumps(row, ensure_ascii=False))
            if not _is_error_label(parsed_from_row):
                parsed_candidates.append(parsed_from_row)

            chosen = {}
            if parsed_candidates:
                chosen = sorted(parsed_candidates, key=_label_info_score, reverse=True)[0]

            new_s = str(chosen.get("scenario", "") or old_s).strip()
            new_a = str(chosen.get("action", "") or old_a).strip()
            new_e = chosen.get("entities") or old_e
            new_e = parse_entities(new_e)

            if (not old_s and not old_a) and (new_s or new_a):
                intent_key_recovered += 1

            row["scenario"] = new_s
            row["action"] = new_a
            row["entities"] = new_e
            row["pred_label"] = {"scenario": new_s, "action": new_a, "entities": new_e}

            if (old_s != new_s) or (old_a != new_a) or (old_e != new_e):
                changed += 1

            rows.append(row)

    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "num_rows": total,
        "num_changed": changed,
        "num_intent_key_recovered": intent_key_recovered,
    }


def calculate_wer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0
    if HAS_JIWER:
        return float(jiwer.wer(reference, hypothesis))
    return 0.0 if reference.strip() == hypothesis.strip() else 1.0



@dataclass
class InferenceCollator:
    processor: Any
    audio_input_mode: str = "auto"

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        if not batch:
            return {}
        tokenizer = get_tokenizer_or_raise(self.processor)
        tokenizer.padding_side = "left"
        mode = str(self.audio_input_mode or "").strip().lower()
        if mode not in {"processor_audio", "tokenized_chat_template"}:
            mode = _infer_audio_input_mode("", processor=self.processor, tokenizer=tokenizer)
        use_processor_audio_flow = mode == "processor_audio"
        sr = get_audio_sampling_rate_or_raise(self.processor, type(self.processor).__name__)
        is_audio_batch = batch[0].get("audio_path") is not None

        net_inputs_list: List[Dict[str, torch.Tensor]] = []
        valid_items: List[Dict[str, Any]] = []

        for item in batch:
            try:
                if is_audio_batch:
                    audio_path = item.get("audio_path")
                    if not audio_path:
                        continue
                    prompt_text = build_prompt_text(item)
                    if use_processor_audio_flow:
                        audio, _ = load_audio_or_raise(audio_path, sr=sr)
                        user_content = [
                            {"type": "audio", "audio_url": "placeholder"},
                            {"type": "text", "text": prompt_text},
                        ]
                        text_input = render_chat_template_as_text_or_raise(
                            self.processor,
                            [{"role": "user", "content": user_content}],
                            add_generation_prompt=True,
                        )
                        net_inputs = self.processor(
                            text=text_input,
                            audio=[audio],
                            sampling_rate=sr,
                            return_tensors="pt",
                        )
                        if "input_ids" not in net_inputs:
                            tok = tokenizer(text_input, return_tensors="pt")
                            if "input_ids" in tok:
                                net_inputs["input_ids"] = tok["input_ids"]
                            if "attention_mask" in tok:
                                net_inputs["attention_mask"] = tok["attention_mask"]
                    else:
                        net_inputs = None
                        for content in _audio_chat_content_variants(prompt_text, audio_path):
                            messages = [{"role": "user", "content": content}]
                            try:
                                net_inputs = _apply_chat_template_tokenized_or_raise(
                                    self.processor,
                                    messages,
                                    add_generation_prompt=True,
                                )
                                break
                            except Exception:
                                continue

                        if net_inputs is None:
                            audio, _ = load_audio_or_raise(audio_path, sr=sr)
                            user_content = [
                                {"type": "audio", "audio_url": "placeholder"},
                                {"type": "text", "text": prompt_text},
                            ]
                            text_input = render_chat_template_as_text_or_raise(
                                self.processor,
                                [{"role": "user", "content": user_content}],
                                add_generation_prompt=True,
                            )
                            net_inputs = self.processor(
                                text=text_input,
                                audio=[audio],
                                sampling_rate=sr,
                                return_tensors="pt",
                            )
                            if "input_ids" not in net_inputs:
                                tok = tokenizer(text_input, return_tensors="pt")
                                if "input_ids" in tok:
                                    net_inputs["input_ids"] = tok["input_ids"]
                                if "attention_mask" in tok:
                                    net_inputs["attention_mask"] = tok["attention_mask"]
                else:
                    prompt_text = build_prompt_text(item, include_transcript=True)
                    text_input = render_chat_template_as_text_or_raise(
                        self.processor,
                        [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
                        add_generation_prompt=True,
                    )
                    net_inputs = tokenizer(text_input, return_tensors="pt")

                net_inputs = {k: v for k, v in net_inputs.items() if torch.is_tensor(v)}
                if "input_ids" not in net_inputs:
                    logger.warning("Skip item %s because input_ids is missing after encoding.", item.get("id"))
                    continue
                if "attention_mask" not in net_inputs:
                    net_inputs["attention_mask"] = torch.ones_like(net_inputs["input_ids"], dtype=torch.long)
                net_inputs = _ensure_feature_masks_for_generation(net_inputs)
                net_inputs_list.append(net_inputs)
                valid_items.append(item)
            except Exception as e:
                logger.warning("Failed to build inference input for %s: %s", item.get("id"), e)
                continue

        if not net_inputs_list:
            return {}
        return {"net_inputs_list": net_inputs_list, "items": valid_items}


def _generate_batch(
    model,
    processor,
    batch_data: Dict[str, Any],
    device,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    if not batch_data:
        return []

    net_inputs_list = batch_data.get("net_inputs_list")
    if net_inputs_list is None:
        net_inputs = batch_data.get("net_inputs")
        net_inputs_list = [net_inputs] if net_inputs is not None else []
    items = batch_data["items"]
    tokenizer = get_tokenizer_or_raise(processor)

    results: List[Dict[str, Any]] = []
    for item, net_inputs in zip(items, net_inputs_list):
        if not isinstance(net_inputs, dict):
            continue
        net_inputs = {k: v.to(device) for k, v in net_inputs.items() if torch.is_tensor(v)}
        if "input_ids" not in net_inputs:
            logger.warning("Skip generation item %s because input_ids is missing.", item.get("id"))
            continue
        if "attention_mask" not in net_inputs:
            net_inputs["attention_mask"] = torch.ones_like(net_inputs["input_ids"], dtype=torch.long)
        net_inputs = _ensure_feature_masks_for_generation(net_inputs)

        output_ids, dropped = _generate_with_retry_drop_unused_kwargs(
            model,
            net_inputs=net_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

        input_len = int(net_inputs["input_ids"].shape[1])
        raw_output = decode_token_ids(processor, output_ids[0][input_len:])
        pred_label = parse_prediction_label(raw_output)
        wer_score = calculate_wer(item.get("transcript", ""), raw_output)

        result_entry = {
            "scenario": pred_label["scenario"],
            "action": pred_label["action"],
            "entities": pred_label["entities"],
            "pred_label": pred_label,
            "file": item.get("file"),
            "slurp_id": item.get("slurp_id"),
            "id": item.get("id"),
            "wer": wer_score,
            "transcript": item.get("transcript", ""),
            "candidates": item.get("candidates", []),
            "rationale_text": item.get("rationale_text", ""),
            "raw_output": raw_output,
            "target": item.get("target", ""),
            "target_label": item.get("target_obj", {}),
            "type": "audio" if item.get("audio_path") else "text",
        }
        results.append(result_entry)

    return results


def _generate_batch_audio_flamingo2(
    model,
    processor,
    batch_data: Dict[str, Any],
    device,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    if not batch_data:
        return []

    net_inputs_list = batch_data.get("net_inputs_list") or []
    items = batch_data.get("items") or []
    prompt_lens = batch_data.get("prompt_lens") or [0] * len(net_inputs_list)
    tokenizer = get_tokenizer_or_raise(processor)

    results: List[Dict[str, Any]] = []
    for item, net_inputs, prompt_len in zip(items, net_inputs_list, prompt_lens):
        if not isinstance(net_inputs, dict):
            continue
        net_inputs = {k: v.to(device) for k, v in net_inputs.items() if torch.is_tensor(v)}
        if "lang_x" not in net_inputs:
            logger.warning("Skip AF2 generation item %s because lang_x is missing.", item.get("id"))
            continue
        if "attention_mask" not in net_inputs:
            net_inputs["attention_mask"] = torch.ones_like(net_inputs["lang_x"], dtype=torch.long)

        output_ids, dropped = _generate_with_retry_drop_unused_kwargs(
            model,
            net_inputs=net_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        if dropped:
            logger.debug("AF2 generate dropped unsupported kwargs for item %s: %s", item.get("id"), dropped)

        input_len = int(prompt_len) if int(prompt_len) > 0 else int(net_inputs["lang_x"].shape[1])
        raw_output = decode_token_ids(processor, output_ids[0][input_len:])
        pred_label = parse_prediction_label(raw_output)
        wer_score = calculate_wer(item.get("transcript", ""), raw_output)

        results.append(
            {
                "scenario": pred_label["scenario"],
                "action": pred_label["action"],
                "entities": pred_label["entities"],
                "pred_label": pred_label,
                "file": item.get("file"),
                "slurp_id": item.get("slurp_id"),
                "id": item.get("id"),
                "wer": wer_score,
                "transcript": item.get("transcript", ""),
                "candidates": item.get("candidates", []),
                "rationale_text": item.get("rationale_text", ""),
                "raw_output": raw_output,
                "target": item.get("target", ""),
                "target_label": item.get("target_obj", {}),
                "type": "audio" if item.get("audio_path") else "text",
            }
        )

    return results


def _distributed_barrier_or_raise(*, rank: int, stage: str) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if rank == 0:
        logger.info("Distributed barrier start: %s", stage)
    t0 = time.perf_counter()
    try:
        dist.barrier()
    except Exception as exc:
        raise RuntimeError(f"Distributed barrier failed at stage='{stage}': {exc}") from exc
    if rank == 0:
        logger.info("Distributed barrier done: %s (elapsed=%.1fs)", stage, time.perf_counter() - t0)


def _install_termination_signal_handlers() -> Dict[int, Any]:
    previous_handlers: Dict[int, Any] = {}

    def _handle_signal(signum: int, _frame: Any) -> None:
        global _TERMINATION_SIGNAL_COUNT
        _TERMINATION_SIGNAL_COUNT += 1
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            sig_name = str(signum)
        logger.warning(
            "Received %s (%d), interrupt_count=%d",
            sig_name,
            signum,
            _TERMINATION_SIGNAL_COUNT,
        )
        if _TERMINATION_SIGNAL_COUNT >= 2:
            logger.warning("Second interrupt received. Forcing immediate exit.")
            os._exit(128 + int(signum))
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[int(sig)] = signal.getsignal(sig)
            signal.signal(sig, _handle_signal)
        except Exception:
            continue
    return previous_handlers


def _restore_termination_signal_handlers(previous_handlers: Dict[int, Any]) -> None:
    for signum, handler in previous_handlers.items():
        try:
            signal.signal(signum, handler)
        except Exception:
            continue


def run_distributed_inference(
    model,
    processor,
    items,
    output_path,
    model_name_or_path: str,
    device,
    rank,
    world_size,
    batch_size=1,
    max_new_tokens: int = 2048,
    num_workers: int = 0,
    seed: int = 0,
):
    model.eval()

    my_items = items[rank::world_size]
    # Split audio/text to keep clean batches
    my_audio_items = [x for x in my_items if x.get("audio_path") is not None]
    my_text_items = [x for x in my_items if x.get("audio_path") is None]

    local_results: List[Dict[str, Any]] = []
    tokenizer = get_tokenizer_or_raise(processor)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    family = _infer_model_family(model_name_or_path)
    af2_mode = family == "audio-flamingo-2" or _is_audio_flamingo2_processor(processor)
    audio_input_mode = ""
    if not af2_mode:
        audio_input_mode = _infer_audio_input_mode(
            model_name_or_path,
            processor=processor,
            tokenizer=tokenizer,
        )
    inference_max_len = int(getattr(processor, "max_tokens", 4096) or 4096)

    if rank == 0:
        logger.info("Starting Inference. Items: %d (Audio: %d, Text: %d), Batch size: %d",
                    len(my_items), len(my_audio_items), len(my_text_items), batch_size)
        if af2_mode:
            logger.info("Inference mode: Audio-Flamingo-2 dedicated route (family=%s)", family)
        else:
            logger.info("Inference audio input mode: %s", audio_input_mode)

    # Audio Loader
    if my_audio_items:
        audio_loader_gen = torch.Generator()
        audio_loader_gen.manual_seed(int(seed) + int(rank) * 10007 + 1)
        collator = (
            AudioFlamingo2InferenceCollator(processor, max_length=inference_max_len)
            if af2_mode
            else InferenceCollator(processor, audio_input_mode=audio_input_mode)
        )
        audio_loader = DataLoader(
            MixedDataset(my_audio_items),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collator,
            drop_last=False,
            shuffle=False,
            worker_init_fn=_seed_worker,
            generator=audio_loader_gen,
        )
        for i, batch_data in enumerate(audio_loader):
            if rank == 0 and i % 10 == 0:
                logger.info("Audio batch %d/%d", i + 1, len(audio_loader))
            try:
                generator = _generate_batch_audio_flamingo2 if af2_mode else _generate_batch
                local_results.extend(
                    generator(
                        model=model,
                        processor=processor,
                        batch_data=batch_data,
                        device=device,
                        max_new_tokens=max_new_tokens,
                    )
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                logger.error("Rank %d failed on audio batch %d: %s", rank, i, exc)

    # Text Loader
    if my_text_items:
        text_loader_gen = torch.Generator()
        text_loader_gen.manual_seed(int(seed) + int(rank) * 10007 + 2)
        collator = (
            AudioFlamingo2InferenceCollator(processor, max_length=inference_max_len)
            if af2_mode
            else InferenceCollator(processor, audio_input_mode=audio_input_mode)
        )
        text_loader = DataLoader(
            MixedDataset(my_text_items),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collator,
            drop_last=False,
            shuffle=False,
            worker_init_fn=_seed_worker,
            generator=text_loader_gen,
        )
        for i, batch_data in enumerate(text_loader):
            if rank == 0 and i % 10 == 0:
                logger.info("Text batch %d/%d", i + 1, len(text_loader))
            try:
                generator = _generate_batch_audio_flamingo2 if af2_mode else _generate_batch
                local_results.extend(
                    generator(
                        model=model,
                        processor=processor,
                        batch_data=batch_data,
                        device=device,
                        max_new_tokens=max_new_tokens,
                    )
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                logger.error("Rank %d failed on text batch %d: %s", rank, i, exc)

    temp_output_path = f"{output_path}.rank{rank}"
    try:
        with open(temp_output_path, "w", encoding="utf-8") as f:
            for res in local_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as exc:
        logger.error("Rank %d failed to save temp file: %s", rank, exc)

    if world_size > 1:
        _distributed_barrier_or_raise(rank=rank, stage="inference_temp_file_flush")

    if rank == 0:
        logger.info("Merging results to %s", output_path)
        pattern = f"{output_path}.rank*"
        temp_files = sorted(glob.glob(pattern))
        with open(output_path, "w", encoding="utf-8") as outfile:
            for fname in temp_files:
                try:
                    with open(fname, "r", encoding="utf-8") as infile:
                        shutil.copyfileobj(infile, outfile)
                    os.remove(fname)
                except Exception as exc:
                    logger.error("Merge error %s: %s", fname, exc)
        merged_count = 0
        with open(output_path, "r", encoding="utf-8") as infile:
            for line in infile:
                if line.strip():
                    merged_count += 1
        logger.info("Merged predictions: %d / %d items", merged_count, len(items))
        if len(items) > 0 and merged_count == 0:
            raise RuntimeError(
                "Inference produced 0 predictions although test items were loaded. "
                "Check warnings/errors above."
            )


def save_label_only_predictions(full_prediction_path: str, label_only_path: str):
    rows = []
    with open(full_prediction_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "id": row.get("id"),
                    "file": row.get("file"),
                    "slurp_id": row.get("slurp_id"),
                    "scenario": row.get("scenario", ""),
                    "action": row.get("action", ""),
                    "entities": parse_entities(row.get("entities", [])),
                }
            )

    with open(label_only_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_entity(entity: Dict[str, Any]) -> Tuple[str, str]:
    if not isinstance(entity, dict):
        return "", ""
    ent_type = str(entity.get("type", "")).strip().lower()
    filler = str(entity.get("filler", "")).strip().lower()
    filler = re.sub(r"\s+", " ", filler)
    return ent_type, filler


def evaluate_prediction_file(prediction_path: str) -> Dict[str, float]:
    total = 0
    scenario_correct = 0
    action_correct = 0
    intent_correct = 0

    tp = 0
    fp = 0
    fn = 0

    with open(prediction_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue

            target_label = row.get("target_label")
            if not isinstance(target_label, dict):
                try:
                    target_label = json.loads(row.get("target", "{}"))
                except Exception:
                    target_label = {}

            pred_scenario = str(row.get("scenario", "")).strip()
            pred_action = str(row.get("action", "")).strip()
            gold_scenario = str(target_label.get("scenario", "")).strip()
            gold_action = str(target_label.get("action", "")).strip()

            total += 1
            scenario_correct += int(pred_scenario == gold_scenario)
            action_correct += int(pred_action == gold_action)
            intent_correct += int(
                (pred_scenario + "_" + pred_action) == (gold_scenario + "_" + gold_action)
            )

            pred_entities = {
                _normalize_entity(e) for e in parse_entities(row.get("entities", []))
            }
            gold_entities = {
                _normalize_entity(e) for e in parse_entities(target_label.get("entities", []))
            }

            tp += len(pred_entities & gold_entities)
            fp += len(pred_entities - gold_entities)
            fn += len(gold_entities - pred_entities)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    entity_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if total == 0:
        return {
            "num_samples": 0,
            "scenario_acc": 0.0,
            "action_acc": 0.0,
            "intent_acc": 0.0,
            "entity_precision": 0.0,
            "entity_recall": 0.0,
            "entity_f1": 0.0,
        }

    return {
        "num_samples": total,
        "scenario_acc": scenario_correct / total,
        "action_acc": action_correct / total,
        "intent_acc": intent_correct / total,
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": entity_f1,
    }


def sample_train_items_by_slurp_id(
    items: List[Dict[str, Any]],
    ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not items:
        return items, {"unique_ids": 0, "selected_ids": 0, "items_before": 0, "items_after": 0}

    item_ids: List[str] = []
    for idx, item in enumerate(items):
        sample_id = pick_first_nonempty(item.get("slurp_id"), item.get("id"))
        if not sample_id:
            sample_id = f"__missing_id_{idx}"
        item_ids.append(sample_id)

    unique_ids = sorted(set(item_ids))
    unique_count = len(unique_ids)

    if ratio >= 1.0:
        return (
            items,
            {
                "unique_ids": unique_count,
                "selected_ids": unique_count,
                "items_before": len(items),
                "items_after": len(items),
            },
        )
    if ratio <= 0.0:
        return (
            [],
            {
                "unique_ids": unique_count,
                "selected_ids": 0,
                "items_before": len(items),
                "items_after": 0,
            },
        )

    selected_count = int(unique_count * ratio)
    if selected_count == 0:
        selected_count = 1
    selected_count = min(selected_count, unique_count)

    rng = random.Random(seed)
    selected_ids = set(rng.sample(unique_ids, selected_count))
    sampled_items = [item for item, sample_id in zip(items, item_ids) if sample_id in selected_ids]

    return (
        sampled_items,
        {
            "unique_ids": unique_count,
            "selected_ids": selected_count,
            "items_before": len(items),
            "items_after": len(sampled_items),
        },
    )


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        type=str,
        default="Experiment_RationaleCompare/sft_success_train.jsonl",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="Experiment_RationaleCompare/sft_success_train.jsonl",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="slurp/dataset/slurp/test.jsonl",
        help="Path to test jsonl.",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/lustre/home/71200138/INTERSPEECH/experiment1/slurp/audio/slurp_real",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="Experiment_3/slurp_metadata.json",
        help="Metadata JSON used to build DB Definitions for prompts.",
    )

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for Python/NumPy/PyTorch/Trainer/Sampler.",
    )
    parser.add_argument(
        "--allow_nondeterministic",
        action="store_true",
        help="Allow non-deterministic CUDA kernels (faster but weaker reproducibility).",
    )
    parser.add_argument(
        "--deterministic_warn_only",
        action="store_true",
        help="Warn (instead of error) when non-deterministic ops are encountered.",
    )
    parser.add_argument(
        "--train_id_sample_ratio",
        type=float,
        default=0.1,
        help="Randomly keep only this ratio of unique train SLURP IDs (0.0-1.0). Default: 0.1",
    )
    parser.add_argument(
        "--train_id_sample_seed",
        type=int,
        default=None,
        help="Random seed for train SLURP ID subsampling (default: --seed).",
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_rationale_label_ft")
    parser.add_argument(
        "--output_file",
        "--output-file",
        "--outpt_file",
        "--outpt-file",
        dest="output_file",
        type=str,
        default="",
        help="Prediction output JSONL path (default: <output_dir>/prediction.jsonl).",
    )
    parser.add_argument(
        "--log_file",
        "--log-file",
        dest="log_file",
        type=str,
        default="",
        help="Training log file path (default: <output_dir>/train.log).",
    )
    parser.add_argument(
        "--recover_prediction_file",
        "--recover-prediction-file",
        dest="recover_prediction_file",
        type=str,
        default="",
        help="Existing prediction JSONL to recover/fix parser outputs from.",
    )
    parser.add_argument(
        "--recover_output_file",
        "--recover-output-file",
        dest="recover_output_file",
        type=str,
        default="",
        help="Recovered prediction JSONL path (default: <recover_input>.recovered.jsonl).",
    )
    parser.add_argument(
        "--recover_inplace",
        "--recover-inplace",
        dest="recover_inplace",
        action="store_true",
        help="Overwrite recover input file instead of writing a new output file.",
    )
    parser.add_argument(
        "--recover_only",
        "--recover-only",
        dest="recover_only",
        action="store_true",
        help="Run recovery mode only and exit (no train/inference).",
    )
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Trainer logging interval in optimizer steps.",
    )
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=None,
        help="Cap eval set size to speed up validation (None means no extra cap).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument(
        "--inference_num_workers",
        type=int,
        default=0,
        help="DataLoader workers for test inference (0 is safer to avoid deadlocks).",
    )
    parser.add_argument(
        "--ddp_timeout_seconds",
        type=int,
        default=900,
        help="Timeout for torch.distributed collective ops in seconds.",
    )
    parser.add_argument(
        "--audio_flamingo2_space_repo",
        type=str,
        default=AUDIO_FLAMINGO2_DEFAULT_SPACE_REPO,
        help="HF Space repo id (or local dir) that contains Audio-Flamingo-2 src/configs.",
    )
    parser.add_argument(
        "--audio_flamingo2_cache_dir",
        type=str,
        default="~/.cache/huggingface/audio_flamingo2",
        help="Cache directory used to download/load Audio-Flamingo-2 assets.",
    )
    parser.add_argument(
        "--audio_flamingo2_local_files_only",
        action="store_true",
        help="Do not use network for Audio-Flamingo-2 asset loading (use local cache/files only).",
    )
    parser.add_argument(
        "--audio_flamingo2_lang_encoder_path",
        type=str,
        default="",
        help="Override base LLM path for Audio-Flamingo-2 (default inferred from model id).",
    )
    parser.add_argument(
        "--audio_flamingo2_tokenizer_path",
        type=str,
        default="",
        help="Override tokenizer path for Audio-Flamingo-2 (default: same as lang encoder).",
    )
    parser.add_argument(
        "--train_audio_encoder",
        action="store_true",
        help="Enable training of audio-related encoder parameters.",
    )
    parser.add_argument(
        "--export_label_eval",
        action="store_true",
        help="Also export label-only predictions and metrics after inference.",
    )
    parser.add_argument(
        "--train_candidates_only",
        "--train-candidates-only",
        "--no_r_train",
        "--no-r-train",
        dest="train_candidates_only",
        action="store_true",
        default=False,
        help="Use C+J targets/prompts (no R) for train/eval splits. Default: disabled.",
    )
    parser.add_argument(
        "--train_with_rationale",
        "--train-with-rationale",
        dest="train_candidates_only",
        action="store_false",
        help="Use full C/R/J targets/prompts for train/eval splits.",
    )
    parser.add_argument(
        "--train_json_only",
        "--train-json-only",
        "--no_c_train",
        "--no-c-train",
        dest="train_json_only",
        action="store_true",
        default=False,
        help="Use J-only targets/prompts (no C/R) for train/eval splits.",
    )
    parser.add_argument("--add_text_only", action="store_true", help="Also add text-only samples.")
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Use text-only samples for all splits (audio paths are ignored).",
    )
    parser.add_argument(
        "--no_text_fallback_when_audio_missing",
        action="store_true",
        help="Disable automatic text-only fallback when audio files cannot be resolved.",
    )
    parser.add_argument(
        "--print_audio_search_paths",
        action="store_true",
        help="Print searched audio paths to stdout.",
    )
    parser.add_argument(
        "--audio_search_print_limit",
        type=int,
        default=100,
        help="Maximum number of audio path debug prints per split.",
    )
    parser.add_argument(
        "--strict_audio_missing",
        action="store_true",
        help="Raise an error immediately when an audio file cannot be resolved.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run tiny smoke test.")

    args = parser.parse_args()
    if args.train_id_sample_seed is None:
        args.train_id_sample_seed = int(args.seed)
    configure_reproducibility(
        args.seed,
        strict_determinism=not bool(args.allow_nondeterministic),
        deterministic_warn_only=bool(args.deterministic_warn_only),
    )
    model_family = _infer_model_family(args.model_name_or_path)

    if not (0.0 <= args.train_id_sample_ratio <= 1.0):
        raise ValueError("--train_id_sample_ratio must be between 0.0 and 1.0")

    if args.recover_only:
        recover_input = args.recover_prediction_file.strip()
        if not recover_input:
            raise ValueError("--recover_only requires --recover_prediction_file")
        if args.recover_inplace:
            recover_output = recover_input
        else:
            recover_output = args.recover_output_file.strip()
            if not recover_output:
                base, ext = os.path.splitext(recover_input)
                recover_output = f"{base}.recovered{ext}" if ext else f"{recover_input}.recovered.jsonl"
        stats = recover_prediction_file(recover_input, recover_output)
        logger.info(
            "Recover done: input=%s output=%s rows=%d changed=%d intent_key_recovered=%d",
            recover_input,
            recover_output,
            stats["num_rows"],
            stats["num_changed"],
            stats["num_intent_key_recovered"],
        )
        return


    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        timeout_seconds = max(60, int(args.ddp_timeout_seconds))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}" if backend == "nccl" else "cpu")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        log_path = args.log_file.strip() if args.log_file.strip() else os.path.join(args.output_dir, "train.log")
        setup_file_logging(log_path)
        logger.info("File logging enabled: %s", os.path.abspath(log_path))
        logger.info(
            "Reproducibility config | seed=%d train_id_sample_seed=%d strict_determinism=%s warn_only=%s",
            int(args.seed),
            int(args.train_id_sample_seed),
            not bool(args.allow_nondeterministic),
            bool(args.deterministic_warn_only),
        )
        if world_size > 1:
            logger.info(
                "Distributed init done | backend=%s world_size=%d rank=%d local_rank=%d timeout=%ds",
                dist.get_backend(),
                world_size,
                rank,
                local_rank,
                max(60, int(args.ddp_timeout_seconds)),
            )

    metadata = load_metadata(args.metadata_file)
    db_definitions = build_db_definitions(metadata)
    set_prompt_db_definitions(db_definitions)
    if rank == 0:
        logger.info("Using prompts with DB Definitions from: %s", args.metadata_file)
        if not os.path.exists(args.metadata_file):
            logger.warning("metadata_file not found: %s (using empty DB Definitions)", args.metadata_file)
        if args.text_only:
            logger.info("text_only=True: all splits will use text-only items.")
        logger.info(
            "Train/eval target mode: %s",
            (
                "J only (no C/R)"
                if args.train_json_only
                else ("C+J (no R)" if args.train_candidates_only else "C/R/J")
            ),
        )

    if rank == 0:
        logger.info("Using test_file: %s", args.test_file)
        logger.info("Detected model family: %s", model_family)

    train_max_samples = args.max_samples
    eval_max_samples = args.eval_max_samples
    if eval_max_samples is None:
        eval_max_samples = (args.max_samples // 2) if args.max_samples else None

    if args.smoke:
        if rank == 0:
            logger.info("SMOKE MODE ON")
        # Increase smoke learn data to 2000, keep eval small
        train_max_samples = 2000
        eval_max_samples = 200
        args.num_train_epochs = 1

    train_items = build_items_from_rationale_jsonl(
        args.train_file,
        args.audio_dir,
        add_text_only=args.add_text_only,
        text_only=args.text_only,
        max_samples=train_max_samples,
        allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
        print_audio_search_paths=args.print_audio_search_paths,
        audio_search_print_limit=args.audio_search_print_limit,
        strict_audio_missing=args.strict_audio_missing,
        train_candidates_only=args.train_candidates_only,
        train_json_only=args.train_json_only,
    )
    train_items, train_sample_stats = sample_train_items_by_slurp_id(
        train_items,
        ratio=args.train_id_sample_ratio,
        seed=args.train_id_sample_seed,
    )
    if rank == 0:
        logger.info(
            (
                "Train SLURP-ID subsampling: ratio=%.4f seed=%d "
                "(unique_ids=%d -> selected_ids=%d, items=%d -> %d)"
            ),
            args.train_id_sample_ratio,
            args.train_id_sample_seed,
            train_sample_stats["unique_ids"],
            train_sample_stats["selected_ids"],
            train_sample_stats["items_before"],
            train_sample_stats["items_after"],
        )

    eval_items = build_items_from_rationale_jsonl(
        args.eval_file,
        args.audio_dir,
        add_text_only=args.add_text_only,
        text_only=args.text_only,
        max_samples=eval_max_samples,
        allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
        print_audio_search_paths=args.print_audio_search_paths,
        audio_search_print_limit=args.audio_search_print_limit,
        strict_audio_missing=args.strict_audio_missing,
        train_candidates_only=args.train_candidates_only,
        train_json_only=args.train_json_only,
    )

    if rank == 0:
        logger.info("Train items: %d | Eval items: %d", len(train_items), len(eval_items))

    if len(train_items) == 0:
        raise RuntimeError("No train items loaded. Check train_file/audio_dir paths.")

    if model_family == "audio-flamingo-2":
        model, processor, tokenizer = load_audio_flamingo2_bundle_or_raise(
            model_name_or_path=args.model_name_or_path,
            space_repo_or_dir=args.audio_flamingo2_space_repo,
            cache_dir=args.audio_flamingo2_cache_dir,
            local_files_only=args.audio_flamingo2_local_files_only,
            override_lm_path=args.audio_flamingo2_lang_encoder_path,
            override_tokenizer_path=args.audio_flamingo2_tokenizer_path,
        )
        processor, tokenizer = ensure_processor_tokenizer_or_raise(processor, args.model_name_or_path)
    else:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        processor, tokenizer = ensure_processor_tokenizer_or_raise(processor, args.model_name_or_path)
        model = load_audio_model_from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    _ = get_audio_sampling_rate_or_raise(processor, args.model_name_or_path)
    if rank == 0:
        logger.info("Moving model to device: %s", device)
    t0 = time.perf_counter()
    model = model.to(device)
    if rank == 0:
        logger.info("Model move to device done: elapsed=%.1fs", time.perf_counter() - t0)
    attach_tokenizer_to_model_for_compat(model, tokenizer)

    # Keep audio modules trainable by default and keep projector trainable, as in prior behavior.
    audio_matches, projector_matches = configure_audio_trainability(
        model,
        train_audio_encoder=True,
        freeze_projector=False,
    )
    if rank == 0:
        logger.info(
            "Trainability | audio_params=%d (enabled=%s), projector_params=%d (enabled=%s)",
            audio_matches,
            True,
            projector_matches,
            True,
        )
        if audio_matches == 0:
            logger.warning(
                "No audio-related parameters were detected by name hints. "
                "Model loading succeeded, but verify fine-tuning targets for this architecture."
            )

    training_kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "bf16": True,
        "logging_steps": max(1, int(args.logging_steps)),
        "logging_first_step": True,
        "eval_strategy": "steps" if len(eval_items) > 0 else "no",
        "eval_steps": 2 if args.smoke else 50,
        "save_strategy": "no",
        "save_total_limit": None,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": True,
        "report_to": "none",
        "disable_tqdm": True,
        "seed": int(args.seed),
        "data_seed": int(args.seed),
    }
    training_sig = inspect.signature(TrainingArguments.__init__).parameters
    if "full_determinism" in training_sig:
        training_kwargs["full_determinism"] = not bool(args.allow_nondeterministic)
    if "tf32" in training_sig:
        training_kwargs["tf32"] = False
    training_args = TrainingArguments(**training_kwargs)

    use_af2_route = model_family == "audio-flamingo-2" or _is_audio_flamingo2_processor(processor)
    train_collator = (
        AudioFlamingo2TrainCollator(
            processor,
            max_length=int(getattr(processor, "max_tokens", 4096) or 4096),
            debug=args.smoke,
        )
        if use_af2_route
        else SmartCollator(processor, debug=args.smoke)
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": MixedDataset(train_items),
        "eval_dataset": MixedDataset(eval_items) if len(eval_items) > 0 else None,
        "data_collator": train_collator,
    }
    trainer_init_params = inspect.signature(CustomTrainer.__init__).parameters
    if "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        logger.warning(
            "Trainer init has neither 'tokenizer' nor 'processing_class'. "
            "Proceeding without explicitly passing tokenizer."
        )
    if rank == 0:
        logger.info("Initializing CustomTrainer...")
    init_start = time.perf_counter()
    trainer = CustomTrainer(**trainer_kwargs)
    if rank == 0:
        logger.info("CustomTrainer initialized: elapsed=%.1fs", time.perf_counter() - init_start)

    if rank == 0:
        logger.info(
            "Starting trainer.train() | epochs=%s batch_size=%s grad_acc=%s logging_steps=%s",
            args.num_train_epochs,
            args.batch_size,
            args.gradient_accumulation_steps,
            max(1, int(args.logging_steps)),
        )
    train_start = time.perf_counter()
    trainer.train()
    if rank == 0:
        logger.info("trainer.train() finished: elapsed=%.1fs", time.perf_counter() - train_start)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    if world_size > 1:
        _distributed_barrier_or_raise(rank=rank, stage="post_train_pre_inference")

    test_max_samples = 10 if args.smoke else None
    if rank == 0 and args.smoke:
        logger.info("Loading only %d test items (smoke).", test_max_samples)

    test_items = build_items_from_rationale_jsonl(
        args.test_file,
        args.audio_dir,
        add_text_only=False,
        text_only=args.text_only,
        max_samples=test_max_samples,
        # Follow original script behavior for test: audio-only (no text fallback).
        allow_text_fallback_when_audio_missing=False if not args.text_only else True,
        print_audio_search_paths=args.print_audio_search_paths,
        audio_search_print_limit=args.audio_search_print_limit,
        strict_audio_missing=args.strict_audio_missing,
        train_candidates_only=False,
        train_json_only=False,
    )

    output_jsonl = args.output_file.strip() if args.output_file.strip() else os.path.join(args.output_dir, "prediction.jsonl")
    output_parent = os.path.dirname(output_jsonl)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    if rank == 0:
        logger.info("Test inference DataLoader workers: %d", args.inference_num_workers)
        logger.info("Prediction output file: %s", output_jsonl)
    run_distributed_inference(
        model=model,
        processor=processor,
        items=test_items,
        output_path=output_jsonl,
        model_name_or_path=args.model_name_or_path,
        device=device,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_workers=args.inference_num_workers,
        seed=args.seed,
    )

    if world_size > 1:
        _distributed_barrier_or_raise(rank=rank, stage="post_inference_pre_export")

    if rank == 0 and args.export_label_eval:
        eval_output_dir = os.path.dirname(output_jsonl) or "."
        label_only_path = os.path.join(eval_output_dir, "prediction_labels_only.jsonl")
        save_label_only_predictions(output_jsonl, label_only_path)

        metrics = evaluate_prediction_file(output_jsonl)
        metrics_path = os.path.join(eval_output_dir, "metrics_label_only.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        logger.info("Label-only evaluation metrics: %s", json.dumps(metrics, ensure_ascii=False))
        logger.info("Saved full predictions: %s", output_jsonl)
        logger.info("Saved label-only predictions: %s", label_only_path)
        logger.info("Saved metrics: %s", metrics_path)
    elif rank == 0:
        logger.info("Saved predictions: %s", output_jsonl)

    if world_size > 1:
        _distributed_barrier_or_raise(rank=rank, stage="finalize")


if __name__ == "__main__":
    _previous_handlers = _install_termination_signal_handlers()
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (KeyboardInterrupt).")
        raise
    finally:
        _restore_termination_signal_handlers(_previous_handlers)
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as exc:
                logger.warning("Failed to destroy process group cleanly: %s", exc)
