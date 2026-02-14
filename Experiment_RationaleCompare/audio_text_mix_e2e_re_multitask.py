#!/usr/bin/env python3
"""
Audio/text mixed SLU training and distributed inference for multitask outputs.

- Task A (CoT): C/R/J rationale output.
- Task B (Label): J-only output.
- Multitask training loss: 0.5 * L_cot + 0.5 * L_label.
"""

import argparse
import glob
import inspect
import json
import logging
import os
import random
import re
import shutil
import warnings
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
try:
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover
    AutoModelForImageTextToText = None
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


class _SuppressSystemPromptModifiedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage().lower()
        except Exception:
            return True
        if "system prompt modified" in msg or "system prompts modified" in msg:
            return False
        return True


if os.environ.get("SHOW_SYSTEM_PROMPT_WARNING", "0") != "1":
    warnings.filterwarnings(
        "ignore",
        message=r".*system prompts? modified.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*system prompt.*modified.*",
        category=UserWarning,
    )
    logging.getLogger().addFilter(_SuppressSystemPromptModifiedFilter())
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
    if "music-flamingo" in name_lc:
        return "music-flamingo"
    if "audio-flamingo-3" in name_lc or "flamingo" in name_lc:
        return "flamingo"
    if "voxtral" in name_lc:
        return "voxtral"
    if "qwen" in name_lc:
        return "qwen"
    return "other"


def load_audio_model_from_pretrained(
    model_name_or_path: str,
    *,
    torch_dtype: torch.dtype,
    trust_remote_code: bool = True,
):
    def _has_unimplemented_forward(model: Any) -> bool:
        try:
            cls_forward = getattr(type(model), "forward", None)
        except Exception:
            return False
        if cls_forward is None:
            return True
        base_forward = getattr(torch.nn.Module, "forward", None)
        if cls_forward is base_forward:
            return True
        fwd_name = getattr(cls_forward, "__name__", "")
        fwd_qualname = getattr(cls_forward, "__qualname__", "")
        if fwd_name == "_forward_unimplemented" or "_forward_unimplemented" in fwd_qualname:
            return True
        return False

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

    # Qwen path: prioritize dedicated loaders, then fallback to AutoModel loaders.
    if family == "qwen":
        qwen_attempts: List[Tuple[str, Any]] = []
        seen_loader_ids = set()
        is_qwen_omni = "omni" in model_name_lc
        if is_qwen_omni:
            qwen_omni_cls = _optional_transformers_class(
                "Qwen2_5OmniThinkerForConditionalGeneration",
                "Qwen2_5OmniThinkerForCausalLM",
                "Qwen2_5OmniForConditionalGeneration",
                "Qwen2_5OmniForCausalLM",
                "Qwen2OmniForConditionalGeneration",
                "Qwen2OmniForCausalLM",
            )
            _append_attempt(
                qwen_attempts,
                getattr(qwen_omni_cls, "__name__", "Qwen2_5Omni*"),
                qwen_omni_cls,
                seen_loader_ids,
            )
        if not is_qwen_omni:
            _append_attempt(
                qwen_attempts,
                "Qwen2AudioForConditionalGeneration",
                Qwen2AudioForConditionalGeneration,
                seen_loader_ids,
            )
        _append_attempt(
            qwen_attempts,
            "AutoModelForImageTextToText",
            AutoModelForImageTextToText,
            seen_loader_ids,
        )
        _append_attempt(qwen_attempts, "AutoModelForCausalLM", AutoModelForCausalLM, seen_loader_ids)
        _append_attempt(qwen_attempts, "AutoModel", AutoModel, seen_loader_ids)

        qwen_errors: List[str] = []
        for loader_name, loader_cls in qwen_attempts:
            try:
                model = loader_cls.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                )
                if _has_unimplemented_forward(model):
                    raise RuntimeError(
                        f"{loader_name} loaded but forward() is unimplemented in this environment."
                    )
                return model
            except Exception as exc:
                qwen_errors.append(f"{loader_name}: {exc}")

        detail = " | ".join(qwen_errors) if qwen_errors else "no qwen loader available"
        raise RuntimeError(
            f"Failed to load Qwen model '{model_name_or_path}'. Details: {detail}"
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
            model = loader_cls.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            if _has_unimplemented_forward(model):
                raise RuntimeError(
                    f"{loader_name} loaded but forward() is unimplemented in this environment."
                )
            return model
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


def ensure_model_vocab_size_for_compat(model: Any, tokenizer: Optional[Any] = None) -> Optional[int]:
    target = getattr(model, "module", None) if getattr(model, "module", None) is not None else model

    def _to_valid_int(value: Any) -> Optional[int]:
        try:
            ivalue = int(value)
        except Exception:
            return None
        return ivalue if ivalue > 0 else None

    nested_candidates = (
        "text_config",
        "language_config",
        "llm_config",
        "decoder_config",
        "model_config",
        "thinker_config",
        "thinking_config",
    )
    seen_cfg_ids = set()
    configs: List[Any] = []

    def _add_config(cfg: Any) -> None:
        if cfg is None:
            return
        cid = id(cfg)
        if cid in seen_cfg_ids:
            return
        seen_cfg_ids.add(cid)
        configs.append(cfg)

    _add_config(getattr(target, "config", None))
    module_attrs = (
        "model",
        "base_model",
        "language_model",
        "thinker",
        "decoder",
        "transformer",
    )
    for attr in module_attrs:
        module = getattr(target, attr, None)
        if module is None:
            continue
        _add_config(getattr(module, "config", None))
        _add_config(getattr(getattr(module, "model", None), "config", None))

    idx = 0
    while idx < len(configs):
        cfg = configs[idx]
        idx += 1
        for attr in nested_candidates:
            _add_config(getattr(cfg, attr, None))

    if not configs:
        return None

    candidate: Optional[int] = None
    for cfg in configs:
        found = _to_valid_int(getattr(cfg, "vocab_size", None))
        if found is not None:
            candidate = found
            break

    if candidate is None:
        getter = getattr(target, "get_input_embeddings", None)
        if callable(getter):
            try:
                emb = getter()
            except Exception:
                emb = None
            if emb is not None and hasattr(emb, "weight"):
                try:
                    candidate = _to_valid_int(emb.weight.shape[0])
                except Exception:
                    candidate = None

    if candidate is None and tokenizer is not None:
        try:
            candidate = _to_valid_int(len(tokenizer))
        except Exception:
            candidate = None

    if candidate is None:
        return None

    updated_count = 0
    for cfg in configs:
        current = _to_valid_int(getattr(cfg, "vocab_size", None))
        if current == int(candidate):
            continue
        try:
            setattr(cfg, "vocab_size", int(candidate))
            updated_count += 1
        except Exception:
            continue
    if updated_count > 0:
        logger.info(
            "Synchronized vocab_size=%d across %d config object(s) for compatibility.",
            int(candidate),
            updated_count,
        )
    return int(candidate)


def _chat_template_owner_or_raise(processor: Any) -> Any:
    if hasattr(processor, "apply_chat_template"):
        return processor
    tokenizer = get_tokenizer_or_raise(processor)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer
    raise RuntimeError(
        f"Neither processor ({type(processor).__name__}) nor tokenizer has apply_chat_template."
    )


def _decode_token_ids_maybe(processor: Any, token_ids: Any) -> Optional[str]:
    try:
        if torch.is_tensor(token_ids):
            ids = token_ids.detach().cpu()
        else:
            ids = torch.tensor(token_ids)
        if ids.dim() > 1:
            ids = ids[0]
        tokenizer = get_tokenizer_or_raise(processor)
        return tokenizer.decode(ids, skip_special_tokens=False)
    except Exception:
        return None


def _coerce_chat_template_text_output(processor: Any, out: Any) -> Optional[str]:
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)):
        if len(out) == 1:
            return _coerce_chat_template_text_output(processor, out[0])
        if out and all(isinstance(x, str) for x in out):
            return "\n".join([x for x in out if x is not None])
        decoded = _decode_token_ids_maybe(processor, out)
        if decoded is not None:
            return decoded
        return None
    if isinstance(out, dict):
        for key in ("text", "prompt", "formatted_text", "chat"):
            if key in out:
                text = _coerce_chat_template_text_output(processor, out.get(key))
                if text is not None:
                    return text
        for key in ("input_ids", "ids", "tokens"):
            if key in out:
                decoded = _decode_token_ids_maybe(processor, out.get(key))
                if decoded is not None:
                    return decoded
        return None
    decoded = _decode_token_ids_maybe(processor, out)
    if decoded is not None:
        return decoded
    return None


def render_chat_template_as_text_or_raise(
    processor: Any,
    messages: List[Dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    owner = None
    errors: List[str] = []
    try:
        owner = _chat_template_owner_or_raise(processor)
    except Exception as exc:
        errors.append(f"owner: {exc}")

    attempts = [
        {"tokenize": False, "add_generation_prompt": add_generation_prompt},
        {"tokenize": False, "add_generation_prompt": add_generation_prompt, "return_dict": False},
        {"tokenize": False},
        {"add_generation_prompt": add_generation_prompt},
    ]
    if owner is not None:
        for kwargs in attempts:
            try:
                out = owner.apply_chat_template(messages, **kwargs)
                rendered = _coerce_chat_template_text_output(processor, out)
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
                errors.append(f"{kwargs}: returned {type(out).__name__}")
            except Exception as exc:
                errors.append(f"{kwargs}: {exc}")

    detail = " | ".join(errors) if errors else "no chat-template variant accepted"
    raise RuntimeError(f"Failed to render chat template as text. Details: {detail}")


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
PROMPT_OUTPUT_FORMAT_LABEL_ONLY = (
    "Output Format:\n"
    f"J: {OUTPUT_SCHEMA}"
)
PROMPT_DB_DEFINITIONS = "Intents: (none)\nSlot Types: (none)"


def normalize_target_components_or_raise(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        raise ValueError("target components cannot be empty")

    aliases = {
        "CANDIDATES": "C",
        "CANDIDATE": "C",
        "RATIONALE": "R",
        "REASONING": "R",
        "JSON": "J",
    }
    for key, token in aliases.items():
        raw = raw.replace(key, token)
    raw = raw.replace(",", "").replace("/", "").replace("|", "").replace(" ", "")

    picked = set(ch for ch in raw if ch in {"C", "R", "J"})
    normalized = "".join(ch for ch in "CRJ" if ch in picked)
    if not normalized:
        raise ValueError(
            f"Invalid target components '{value}'. Use any combination of C, R, J (e.g., CRJ, CJ, J)."
        )
    return normalized


def default_target_components_from_legacy_flags(*, train_candidates_only: bool) -> str:
    return "CJ" if train_candidates_only else "CRJ"


def describe_target_components(components: str) -> str:
    labels = {"C": "Candidates", "R": "Rationale", "J": "JSON"}
    return "+".join(labels[ch] for ch in components if ch in labels)


def prompt_output_format_for_components(components: str) -> str:
    comps = normalize_target_components_or_raise(components)
    if comps == "CRJ":
        return PROMPT_OUTPUT_FORMAT
    if comps == "CJ":
        return PROMPT_OUTPUT_FORMAT_CANDIDATES_ONLY
    if comps == "J":
        return PROMPT_OUTPUT_FORMAT_LABEL_ONLY

    lines = ["Output Format:"]
    if "C" in comps:
        lines.append(
            "C: Intent candidates: intent1 | intent2 | intent3; Slot candidates: slot_type1(value1|value2) | slot_type2"
        )
    if "R" in comps:
        lines.append("R: label1!reason1; label2!reason2; ...")
    if "J" in comps:
        lines.append(f"J: {OUTPUT_SCHEMA}")
    return "\n".join(lines)


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


def normalize_task_mode(task_mode: Any) -> str:
    mode = str(task_mode or "cot").strip().lower()
    if mode in ("candidates", "cand"):
        return "candidates"
    if mode == "label":
        return "label"
    return "cot"


def task_id_from_mode(task_mode: Any) -> int:
    return 1 if normalize_task_mode(task_mode) == "label" else 0


def build_prompt_text(
    item: Dict[str, Any],
    include_transcript: bool = False,
    include_user_prompt: bool = False,
) -> str:
    transcript = str(item.get("transcript", "") or "").strip()
    task_mode = normalize_task_mode(item.get("task_mode", "cot"))
    if task_mode == "label":
        output_format = PROMPT_OUTPUT_FORMAT_LABEL_ONLY
    else:
        components = str(item.get("target_components", "") or "").strip()
        if components:
            output_format = prompt_output_format_for_components(components)
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
    target_components: Optional[str] = None,
) -> str:
    if target_components is not None:
        components = normalize_target_components_or_raise(target_components)
        rationale = (rationale_text or "").strip()
        lines = [line.strip() for line in rationale.splitlines() if line.strip()]
        lines = [_sanitize_c_line(line) if line.startswith("C:") else line for line in lines]

        c_line = next((line for line in lines if line.startswith("C:")), "")
        r_line = next((line for line in lines if line.startswith("R:")), "")

        out_lines: List[str] = []
        if "C" in components:
            out_lines.append(c_line if c_line else "C: (none)")
        if "R" in components:
            out_lines.append(r_line if r_line else "R: (none)")
        if "J" in components:
            out_lines.append(f"J: {final_json}")
        if out_lines:
            return "\n".join(out_lines)
        return f"J: {final_json}"

    rationale = (rationale_text or "").strip()
    if not rationale:
        return f"J: {final_json}"

    lines = [line.strip() for line in rationale.splitlines() if line.strip()]
    lines = [_sanitize_c_line(line) if line.startswith("C:") else line for line in lines]
    has_c = any(line.startswith("C:") for line in lines)
    has_r = any(line.startswith("R:") for line in lines)
    has_j = any(line.startswith("J:") for line in lines)

    if has_c and has_r and has_j:
        return "\n".join(lines)
    if has_j:
        return "\n".join(lines)
    return "\n".join(lines + [f"J: {final_json}"])


def build_label_only_target(final_json: str) -> str:
    return f"J: {final_json}"


def build_candidates_only_target(
    rationale_text: str,
    final_json: str,
    fallback_text: str = "",
) -> str:
    c_line = ""
    for source_text in (rationale_text or "", fallback_text or ""):
        if not source_text:
            continue
        for line in source_text.splitlines():
            s = line.strip()
            if s.startswith("C:"):
                c_line = _sanitize_c_line(s)
                break
        if c_line:
            break
    if not c_line:
        c_line = "C: (none)"
    return "\n".join([c_line, f"J: {final_json}"])


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


def expand_multitask_items(
    base_item: Dict[str, Any],
    cot_task_mode: str = "cot",
    cot_target_components: Optional[str] = None,
) -> List[Dict[str, Any]]:
    target_obj = base_item.get("target_obj", {})
    final_json = json.dumps(target_obj, ensure_ascii=False)
    cot_components = (
        normalize_target_components_or_raise(cot_target_components)
        if cot_target_components is not None and str(cot_target_components).strip()
        else None
    )
    if cot_components is not None:
        cot_target = build_training_target(
            base_item.get("rationale_text", ""),
            final_json,
            target_components=cot_components,
        )
        cot_mode = "candidates" if cot_components == "CJ" else "cot"
    else:
        cot_mode = "candidates" if normalize_task_mode(cot_task_mode) == "candidates" else "cot"
        if cot_mode == "candidates":
            cot_target = build_candidates_only_target(
                base_item.get("rationale_text", ""),
                final_json,
                fallback_text=base_item.get("target", ""),
            )
        else:
            cot_target = base_item.get("target", build_label_only_target(final_json))
    cot_item = {
        **base_item,
        "task_mode": "candidates" if cot_mode == "candidates" else "cot",
        "task_id": 0,
        "target": cot_target,
    }
    if cot_components is not None:
        cot_item["target_components"] = cot_components
    label_item = {
        **base_item,
        "task_mode": "label",
        "task_id": 1,
        "target": build_label_only_target(final_json),
    }
    return [cot_item, label_item]


def build_task_item(
    base_item: Dict[str, Any],
    task_mode: str,
    cot_target_components: Optional[str] = None,
) -> Dict[str, Any]:
    mode = normalize_task_mode(task_mode)
    target_obj = base_item.get("target_obj", {})
    final_json = json.dumps(target_obj, ensure_ascii=False)
    if mode == "label":
        return {
            **base_item,
            "task_mode": "label",
            "task_id": 1,
            "target": build_label_only_target(final_json),
        }
    cot_components = (
        normalize_target_components_or_raise(cot_target_components)
        if cot_target_components is not None and str(cot_target_components).strip()
        else None
    )
    if cot_components is not None:
        item = {
            **base_item,
            "task_mode": "candidates" if cot_components == "CJ" else "cot",
            "task_id": 0,
            "target": build_training_target(
                base_item.get("rationale_text", ""),
                final_json,
                target_components=cot_components,
            ),
            "target_components": cot_components,
        }
        return item

    if mode == "candidates":
        return {
            **base_item,
            "task_mode": "candidates",
            "task_id": 0,
            "target": build_candidates_only_target(
                base_item.get("rationale_text", ""),
                final_json,
                fallback_text=base_item.get("target", ""),
            ),
        }
    return {
        **base_item,
        "task_mode": "cot",
        "task_id": 0,
        "target": base_item.get("target", build_label_only_target(final_json)),
    }


def build_multisource_multitask_items(
    label_items: List[Dict[str, Any]],
    cot_items: List[Dict[str, Any]],
    cot_task_mode: str = "cot",
    cot_target_components: Optional[str] = None,
) -> List[Dict[str, Any]]:
    mixed: List[Dict[str, Any]] = []
    mixed.extend(build_task_item(item, "label") for item in label_items)
    mixed.extend(
        build_task_item(item, cot_task_mode, cot_target_components=cot_target_components)
        for item in cot_items
    )
    return mixed


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
    multitask: bool = True,
    cot_task_mode: str = "cot",
    train_target_components: Optional[str] = None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    fallback_text_items: List[Dict[str, Any]] = []
    explicit_components: Optional[str] = None
    if train_target_components is not None and str(train_target_components).strip():
        explicit_components = normalize_target_components_or_raise(train_target_components)
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
                    target_components=explicit_components,
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
                    target_components=explicit_components,
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
            }
            if explicit_components is not None:
                base_item["target_components"] = explicit_components
            text_only_item = {**base_item, "audio_path": None}
            text_only_items = (
                expand_multitask_items(
                    text_only_item,
                    cot_task_mode=cot_task_mode,
                    cot_target_components=explicit_components,
                )
                if multitask
                else [text_only_item]
            )
            fallback_text_items.extend(text_only_items)

            if text_only:
                items.extend(text_only_items)
                continue

            if add_text_only:
                items.extend(text_only_items)

            if audio_path:
                audio_items = (
                    expand_multitask_items(
                        base_item,
                        cot_task_mode=cot_task_mode,
                        cot_target_components=explicit_components,
                    )
                    if multitask
                    else [base_item]
                )
                items.extend(audio_items)
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
class SmartCollator:
    processor: Any
    max_length: int = 512
    ignore_index: int = -100
    debug: bool = False
    _print_count: int = 0
    _feature_mask_synth_warn_count: int = 0
    _audio_fallback_warn_count: int = 0

    def __post_init__(self):
        self._print_count = 0
        self._feature_mask_synth_warn_count = 0
        self._audio_fallback_warn_count = 0

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        if len(batch) == 0:
            return {}
        is_audio_batch = batch[0].get("audio_path") is not None
        if is_audio_batch:
            return self._collate_audio(batch)
        return self._collate_text(batch)

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
        task_ids: List[int] = []

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
                print(f"[DEBUG Visualizer] Task: {item.get('task_mode', 'cot')}")
                print(f"[DEBUG Visualizer] Input Prompt:\n{text_input}")
                print(f"[DEBUG Visualizer] Target:\n{item['target']}")
                self._print_count += 1

            try:
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
            except Exception:
                continue
            prompt_ids = prompt_inputs.get("input_ids")
            if prompt_ids is None:
                tok_prompt = tokenizer(text_input, return_tensors="pt")
                prompt_ids = tok_prompt.get("input_ids")
            if prompt_ids is None:
                continue
            prompt_len = prompt_ids.shape[1]

            full_ids = inputs.get("input_ids")
            if full_ids is None:
                tok_full = tokenizer(full_text, return_tensors="pt")
                full_ids = tok_full.get("input_ids")
            if full_ids is None:
                continue
            ids = full_ids[0]
            lbs = ids.clone()
            lbs[:prompt_len] = self.ignore_index

            input_ids_list.append(ids)
            labels_list.append(lbs)
            task_ids.append(int(item.get("task_id", 0)))

            feat = inputs["input_features"]
            while feat.dim() > 2:
                feat = feat.squeeze(0)
            input_features_list.append(feat)

            f_mask = inputs.get("feature_attention_mask")
            if f_mask is None:
                f_mask = inputs.get("input_features_mask")
            if f_mask is not None:
                while f_mask.dim() > 1:
                    f_mask = f_mask.squeeze(0)
            inferred_tlen = int(max(feat.shape)) if feat.dim() >= 2 else int(feat.shape[0])
            if f_mask is None or int(f_mask.shape[0]) != inferred_tlen:
                if self._feature_mask_synth_warn_count < 20:
                    logger.warning(
                        "Synthesizing feature mask for sample id=%s (mask_missing_or_mismatch).",
                        item.get("id"),
                    )
                    self._feature_mask_synth_warn_count += 1
                f_mask = torch.ones(inferred_tlen, dtype=torch.long, device=feat.device)
            else:
                f_mask = f_mask.to(device=feat.device, dtype=torch.long)
            feature_mask_list.append(f_mask)

        if not input_ids_list:
            if self._audio_fallback_warn_count < 20:
                batch_ids = [str(x.get("id", "")) for x in batch[:8]]
                logger.warning(
                    "All audio samples in a batch were skipped; fallback to text-only. ids(head)=%s",
                    batch_ids,
                )
                self._audio_fallback_warn_count += 1
            text_fallback_batch = [{**item, "audio_path": None} for item in batch]
            return self._collate_text(text_fallback_batch)

        batch_out = {
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
            "task_ids": torch.tensor(task_ids, dtype=torch.long),
        }
        fmask = pad_sequence(feature_mask_list, batch_first=True, padding_value=0)
        batch_out["feature_attention_mask"] = fmask
        batch_out["input_features_mask"] = fmask
        return batch_out

    def _collate_text(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list, labels_list = [], []
        task_ids: List[int] = []

        tokenizer = get_tokenizer_or_raise(self.processor)
        eos_token = tokenizer.eos_token or "<|endoftext|>"
        for item in batch:
            if item.get("audio_path") is not None:
                continue
            text_input = self._build_text_chat(item)
            full_text = text_input + item["target"] + eos_token

            if self.debug and self._print_count < 5:
                print(f"\n[DEBUG Visualizer] Text Sample ID: {item.get('id')}")
                print(f"[DEBUG Visualizer] Task: {item.get('task_mode', 'cot')}")
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
            task_ids.append(int(item.get("task_id", 0)))

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
            "task_ids": torch.tensor(task_ids, dtype=torch.long),
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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not getattr(self, "_vocab_size_compat_checked", False):
            trainer_tokenizer = getattr(self, "tokenizer", None)
            ensure_model_vocab_size_for_compat(model, trainer_tokenizer)
            self._vocab_size_compat_checked = True
        task_ids = inputs.pop("task_ids", None)
        inputs = self._sanitize_model_inputs(inputs)
        inputs = self._drop_unsupported_feature_masks(model, inputs)
        inputs = _cast_floating_tensors_to_model_dtype(inputs, model)
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if task_ids is None or labels is None or not hasattr(outputs, "logits"):
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size(0), shift_labels.size(1))

        valid_mask = (shift_labels != -100).float()
        per_sample_loss = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)

        task_ids = task_ids.to(per_sample_loss.device)
        cot_mask = task_ids == 0
        label_mask = task_ids == 1

        if cot_mask.any() and label_mask.any():
            cot_loss = per_sample_loss[cot_mask].mean()
            label_loss = per_sample_loss[label_mask].mean()
            loss = 0.5 * cot_loss + 0.5 * label_loss
        elif cot_mask.any():
            loss = per_sample_loss[cot_mask].mean()
        elif label_mask.any():
            loss = per_sample_loss[label_mask].mean()
        else:
            loss = per_sample_loss.mean()

        return (loss, outputs) if return_outputs else loss


# ==============================================================================
# 5. Callback
# ==============================================================================


class SampleGenerationCallback(TrainerCallback):
    def __init__(self, eval_items, processor, model, num_samples: int = 3, max_new_tokens: int = 128):
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
        tokenizer = get_tokenizer_or_raise(self.processor)

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
                    tok = tokenizer(text_input, return_tensors="pt")
                    if "input_ids" in tok:
                        inputs["input_ids"] = tok["input_ids"].to(device)
                    if "attention_mask" in tok:
                        inputs["attention_mask"] = tok["attention_mask"].to(device)
                if "attention_mask" not in inputs and "input_ids" in inputs:
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)
                inputs = _ensure_feature_masks_for_generation(inputs)

                output_ids, _ = _generate_with_retry_drop_unused_kwargs(
                    self.model,
                    net_inputs=inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
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
    per_sample: bool = True

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        if self.per_sample:
            return self._collate_per_sample(batch)
        return self._collate_batched(batch)

    def _collate_per_sample(self, batch: List[Dict]) -> Dict[str, Any]:
        if not batch:
            return {}
        tokenizer = get_tokenizer_or_raise(self.processor)
        tokenizer.padding_side = "left"
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
                    audio, _ = load_audio_or_raise(audio_path, sr=sr)
                    prompt_text = build_prompt_text(item, include_user_prompt=True)
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
                    prompt_text = build_prompt_text(
                        item,
                        include_transcript=True,
                        include_user_prompt=True,
                    )
                    text_input = render_chat_template_as_text_or_raise(
                        self.processor,
                        [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
                        add_generation_prompt=True,
                    )
                    net_inputs = tokenizer(text_input, return_tensors="pt")

                net_inputs = {k: v for k, v in net_inputs.items() if torch.is_tensor(v)}
                if "input_ids" not in net_inputs:
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

    def _collate_batched(self, batch: List[Dict]) -> Dict[str, Any]:
        if not batch:
            return {}
        tokenizer = get_tokenizer_or_raise(self.processor)
        tokenizer.padding_side = "left"
        sr = get_audio_sampling_rate_or_raise(self.processor, type(self.processor).__name__)
        is_audio_batch = batch[0].get("audio_path") is not None

        texts: List[str] = []
        audios: List[Any] = []
        valid_items: List[Dict[str, Any]] = []

        for item in batch:
            try:
                if is_audio_batch:
                    audio_path = item.get("audio_path")
                    if not audio_path:
                        continue
                    audio, _ = load_audio_or_raise(audio_path, sr=sr)
                    audios.append(audio)
                    prompt_text = build_prompt_text(item, include_user_prompt=True)
                    user_content = [
                        {"type": "audio", "audio_url": "placeholder"},
                        {"type": "text", "text": prompt_text},
                    ]
                else:
                    prompt_text = build_prompt_text(
                        item,
                        include_transcript=True,
                        include_user_prompt=True,
                    )
                    user_content = [{"type": "text", "text": prompt_text}]

                text_input = render_chat_template_as_text_or_raise(
                    self.processor,
                    [{"role": "user", "content": user_content}],
                    add_generation_prompt=True,
                )
                texts.append(text_input)
                valid_items.append(item)
            except Exception as e:
                logger.warning("Failed to build batched inference input for %s: %s", item.get("id"), e)
                continue

        if not texts:
            return {}

        processor_kwargs: Dict[str, Any] = {
            "text": texts,
            "sampling_rate": sr,
            "padding": True,
            "return_tensors": "pt",
        }
        if is_audio_batch:
            processor_kwargs["audio"] = audios
        net_inputs = self.processor(**processor_kwargs)
        net_inputs = {k: v for k, v in net_inputs.items() if torch.is_tensor(v)}
        if "input_ids" not in net_inputs:
            tok = tokenizer(texts, padding=True, return_tensors="pt")
            if "input_ids" in tok:
                net_inputs["input_ids"] = tok["input_ids"]
            if "attention_mask" in tok:
                net_inputs["attention_mask"] = tok["attention_mask"]
        if "input_ids" not in net_inputs:
            return {}
        if "attention_mask" not in net_inputs:
            net_inputs["attention_mask"] = torch.ones_like(net_inputs["input_ids"], dtype=torch.long)
        net_inputs = _ensure_feature_masks_for_generation(net_inputs)
        return {"net_inputs": net_inputs, "items": valid_items}


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
    items = batch_data["items"]
    tokenizer = get_tokenizer_or_raise(processor)

    if net_inputs_list is None:
        net_inputs = batch_data.get("net_inputs")
        if not isinstance(net_inputs, dict):
            return []
        net_inputs = {k: v.to(device) for k, v in net_inputs.items() if torch.is_tensor(v)}
        if "input_ids" not in net_inputs:
            return []
        if "attention_mask" not in net_inputs:
            net_inputs["attention_mask"] = torch.ones_like(net_inputs["input_ids"], dtype=torch.long)
        net_inputs = _ensure_feature_masks_for_generation(net_inputs)

        output_ids, _ = _generate_with_retry_drop_unused_kwargs(
            model,
            net_inputs=net_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        input_len = int(net_inputs["input_ids"].shape[1])
        raw_outputs = batch_decode_token_ids(processor, output_ids[:, input_len:])

        results: List[Dict[str, Any]] = []
        for item, raw_output in zip(items, raw_outputs):
            pred_label = parse_prediction_label(raw_output)
            wer_score = calculate_wer(item.get("transcript", ""), raw_output)
            task_mode = normalize_task_mode(item.get("task_mode", "cot"))
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
                "task_mode": task_mode,
                "task_id": int(item.get("task_id", task_id_from_mode(task_mode))),
            }
            results.append(result_entry)
        return results

    results: List[Dict[str, Any]] = []
    for item, net_inputs in zip(items, net_inputs_list):
        if not isinstance(net_inputs, dict):
            continue
        net_inputs = {k: v.to(device) for k, v in net_inputs.items() if torch.is_tensor(v)}
        if "input_ids" not in net_inputs:
            continue
        if "attention_mask" not in net_inputs:
            net_inputs["attention_mask"] = torch.ones_like(net_inputs["input_ids"], dtype=torch.long)
        net_inputs = _ensure_feature_masks_for_generation(net_inputs)

        output_ids, _ = _generate_with_retry_drop_unused_kwargs(
            model,
            net_inputs=net_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

        input_len = int(net_inputs["input_ids"].shape[1])
        raw_output = decode_token_ids(processor, output_ids[0][input_len:])
        pred_label = parse_prediction_label(raw_output)
        wer_score = calculate_wer(item.get("transcript", ""), raw_output)
        task_mode = normalize_task_mode(item.get("task_mode", "cot"))

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
            "task_mode": task_mode,
            "task_id": int(item.get("task_id", task_id_from_mode(task_mode))),
        }
        results.append(result_entry)

    return results


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
    max_new_tokens: int = 128,
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
    tokenizer.padding_side = "left"
    family = _infer_model_family(model_name_or_path)
    per_sample_inference = family in {"flamingo", "music-flamingo"}

    if rank == 0:
        logger.info("Starting Inference. Items: %d (Audio: %d, Text: %d), Batch size: %d",
                    len(my_items), len(my_audio_items), len(my_text_items), batch_size)
        logger.info("Inference mode: %s (family=%s)", "per-sample" if per_sample_inference else "batched", family)

    # Audio Loader
    if my_audio_items:
        audio_loader_gen = torch.Generator()
        audio_loader_gen.manual_seed(int(seed) + int(rank) * 10007 + 1)
        audio_loader = DataLoader(
            MixedDataset(my_audio_items),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=InferenceCollator(processor, per_sample=per_sample_inference),
            drop_last=False,
            shuffle=False,
            worker_init_fn=_seed_worker,
            generator=audio_loader_gen,
        )
        for i, batch_data in enumerate(audio_loader):
            if rank == 0 and i % 10 == 0:
                logger.info("Audio batch %d/%d", i + 1, len(audio_loader))
            try:
                local_results.extend(
                    _generate_batch(
                        model=model,
                        processor=processor,
                        batch_data=batch_data,
                        device=device,
                        max_new_tokens=max_new_tokens,
                    )
                )
            except Exception as exc:
                logger.error("Rank %d failed on audio batch %d: %s", rank, i, exc)

    # Text Loader
    if my_text_items:
        text_loader_gen = torch.Generator()
        text_loader_gen.manual_seed(int(seed) + int(rank) * 10007 + 2)
        text_loader = DataLoader(
            MixedDataset(my_text_items),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=InferenceCollator(processor, per_sample=per_sample_inference),
            drop_last=False,
            shuffle=False,
            worker_init_fn=_seed_worker,
            generator=text_loader_gen,
        )
        for i, batch_data in enumerate(text_loader):
            if rank == 0 and i % 10 == 0:
                logger.info("Text batch %d/%d", i + 1, len(text_loader))
            try:
                local_results.extend(
                    _generate_batch(
                        model=model,
                        processor=processor,
                        batch_data=batch_data,
                        device=device,
                        max_new_tokens=max_new_tokens,
                    )
                )
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
        dist.barrier()

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


def _build_derangement(ids: List[str], rng: random.Random) -> List[str]:
    if len(ids) <= 1:
        return ids[:]
    shuffled = ids[:]
    rng.shuffle(shuffled)
    for shift in range(len(shuffled)):
        candidate = shuffled[shift:] + shuffled[:shift]
        if all(src != dst for src, dst in zip(ids, candidate)):
            return candidate
    # Deterministic fallback.
    return ids[1:] + ids[:1]


def _is_cot_branch_item(item: Dict[str, Any]) -> bool:
    task_id = item.get("task_id")
    try:
        task_id_int = int(task_id)
    except Exception:
        task_id_int = None
    if task_id_int == 0:
        return True
    if task_id_int == 1:
        return False
    task_mode = str(item.get("task_mode", "") or "").strip().lower()
    return task_mode in {"cot", "candidates", "cand"}


def apply_random_cot_target_ablation(
    items: List[Dict[str, Any]],
    seed: int = 42,
) -> int:
    cot_groups: Dict[str, List[int]] = {}
    cot_targets: Dict[str, str] = {}

    for idx, item in enumerate(items):
        if not _is_cot_branch_item(item):
            continue
        sample_id = pick_first_nonempty(item.get("slurp_id"), item.get("id"))
        if not sample_id:
            sample_id = f"__missing_id_{idx}"
        cot_groups.setdefault(sample_id, []).append(idx)
        if sample_id not in cot_targets:
            cot_targets[sample_id] = str(item.get("target", "") or "")

    cot_ids = sorted(cot_groups.keys())
    if len(cot_ids) <= 1:
        logger.warning("random_cot enabled but CoT items are insufficient: %d", len(cot_ids))
        return 0

    rng = random.Random(seed)
    donor_ids = _build_derangement(cot_ids, rng)

    for receiver_id, donor_id in zip(cot_ids, donor_ids):
        donor_target = cot_targets.get(donor_id, "")
        for idx in cot_groups[receiver_id]:
            items[idx]["target"] = donor_target
            items[idx]["random_cot_source_id"] = donor_id

    return len(cot_ids)


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
        "--cot_train_file",
        "--cot-train-file",
        dest="cot_train_file",
        type=str,
        default="",
        help=(
            "Optional CoT-only train file for multitask training. "
            "When set, --train_file is used for label task and this file is used for CoT task."
        ),
    )
    parser.add_argument(
        "--cot_eval_file",
        "--cot-eval-file",
        dest="cot_eval_file",
        type=str,
        default="",
        help=(
            "Optional CoT-only eval file for multitask validation. "
            "When set, --eval_file is used for label task and this file is used for CoT task."
        ),
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
        default=1.0,
        help="Randomly keep only this ratio of unique train SLURP IDs (0.0-1.0). Default: 1.0",
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
        help=(
            "Prediction output JSONL path. "
            "Default: <output_dir>/prediction_<mode>.jsonl. "
            "When --test_task_mode both is used, '_cot' and '_label' are appended."
        ),
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
        "--logging_enable",
        "--logging-enable",
        dest="logging_enable",
        action="store_true",
        default=True,
        help="Enable file logging (default: enabled).",
    )
    parser.add_argument(
        "--no_logging_enable",
        "--no-logging-enable",
        dest="logging_enable",
        action="store_false",
        help="Disable file logging.",
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
    parser.add_argument(
        "--inference_only",
        "--inference-only",
        dest="inference_only",
        action="store_true",
        help="Skip train/eval and run test inference only using --model_name_or_path.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=None,
        help="Cap eval set size to speed up validation (None means no extra cap).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--inference_num_workers",
        type=int,
        default=0,
        help="DataLoader workers for test inference (0 is safer to avoid deadlocks).",
    )
    parser.add_argument(
        "--train_audio_encoder",
        action="store_true",
        help="Enable training of audio-related encoder parameters.",
    )
    parser.add_argument(
        "--random_cot",
        "--random-cot",
        "--rondom_cot",
        "--rondom-cot",
        dest="random_cot",
        action="store_true",
        help=(
            "Ablation mode: randomize CoT-branch teacher targets across samples "
            "while keeping label-branch J targets unchanged."
        ),
    )
    parser.add_argument(
        "--random_cot_seed",
        "--random-cot-seed",
        type=int,
        default=None,
        help="Random seed for --random_cot target shuffling (default: --seed).",
    )
    parser.add_argument(
        "--export_label_eval",
        action="store_true",
        help="Also export label-only predictions and metrics after inference.",
    )
    parser.add_argument(
        "--train_target_components",
        "--train-target-components",
        type=str,
        default="",
        help=(
            "CoT branch target components as combination of C,R,J "
            "(examples: CRJ, CJ, J, RJ). "
            "When set, this overrides --train_candidates_only for train/eval."
        ),
    )
    parser.add_argument(
        "--train_candidates_only",
        "--train-candidates-only",
        "--no_r_train",
        "--no-r-train",
        dest="train_candidates_only",
        action="store_true",
        help="Legacy option: CoT branch uses C+J targets/prompts (no R) instead of C/R/J.",
    )
    parser.add_argument(
        "--no_train_candidates_only",
        "--no-train-candidates-only",
        dest="train_candidates_only",
        action="store_false",
        help="Legacy option: Disable C+J train/eval mode and use standard C/R/J for CoT branch.",
    )
    parser.add_argument(
        "--test_task_mode",
        type=str,
        choices=["cot", "candidates", "label", "both"],
        default="cot",
        help="Prompt/output mode used only for test inference (default: cot). Use 'both' to run cot and label.",
    )
    parser.add_argument(
        "--candidates_only",
        "--candidates-only",
        dest="test_task_mode",
        action="store_const",
        const="candidates",
        help="Alias for --test_task_mode candidates (C+J generation at test time).",
    )
    parser.add_argument(
        "--no_cot",
        "--no-cot",
        dest="test_task_mode",
        action="store_const",
        const="label",
        help="Alias for --test_task_mode label (J-only generation at test time).",
    )
    parser.add_argument(
        "--with_cot",
        "--with-cot",
        dest="test_task_mode",
        action="store_const",
        const="cot",
        help="Alias for --test_task_mode cot (C/R/J generation at test time).",
    )
    parser.add_argument(
        "--test_both_modes",
        "--test-both-modes",
        "--both_test_modes",
        "--both-test-modes",
        dest="test_task_mode",
        action="store_const",
        const="both",
        help="Run both test modes in one execution: cot and label.",
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
    if args.random_cot_seed is None:
        args.random_cot_seed = int(args.seed)
    strict_determinism = not bool(args.allow_nondeterministic)
    deterministic_warn_only = bool(args.deterministic_warn_only)
    if args.smoke and strict_determinism:
        # Smoke is for quick sanity checks; strict deterministic kernels can stall on some stacks.
        logger.warning(
            "Smoke mode: strict deterministic algorithms are disabled to avoid startup stalls. "
            "Seed-based reproducibility remains enabled."
        )
        strict_determinism = False
        deterministic_warn_only = True
    configure_reproducibility(
        args.seed,
        strict_determinism=strict_determinism,
        deterministic_warn_only=deterministic_warn_only,
    )
    explicit_components = str(args.train_target_components or "").strip()
    if explicit_components:
        selected_train_components = normalize_target_components_or_raise(explicit_components)
    else:
        selected_train_components = default_target_components_from_legacy_flags(
            train_candidates_only=bool(args.train_candidates_only),
        )
    cot_train_task_mode = "candidates" if (not explicit_components and args.train_candidates_only) else "cot"
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
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        if args.logging_enable:
            log_path = args.log_file.strip() if args.log_file.strip() else os.path.join(args.output_dir, "train.log")
            setup_file_logging(log_path)
            logger.info("File logging enabled: %s", os.path.abspath(log_path))
        else:
            logger.info("File logging disabled.")
        logger.info(
            "Reproducibility config | seed=%d train_id_sample_seed=%d random_cot_seed=%d strict_determinism=%s warn_only=%s",
            int(args.seed),
            int(args.train_id_sample_seed),
            int(args.random_cot_seed),
            bool(strict_determinism),
            bool(deterministic_warn_only),
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
            "Train/eval target mode: label=J only, cot=%s (%s)",
            selected_train_components,
            describe_target_components(selected_train_components),
        )
        if explicit_components:
            logger.info(
                "Using explicit --train_target_components=%s for CoT branch (legacy mode flags are ignored).",
                selected_train_components,
            )

    if rank == 0:
        logger.info("Using test_file: %s", args.test_file)

    train_items: List[Dict[str, Any]] = []
    eval_items: List[Dict[str, Any]] = []
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

    cot_train_file = args.cot_train_file.strip()
    cot_eval_file = args.cot_eval_file.strip()

    if not args.inference_only:
        if cot_train_file:
            if rank == 0:
                logger.info(
                    "Using split multitask train files | label=%s | cot=%s",
                    args.train_file,
                    cot_train_file,
                )
            label_train_items = build_items_from_rationale_jsonl(
                args.train_file,
                args.audio_dir,
                add_text_only=args.add_text_only,
                text_only=args.text_only,
                max_samples=train_max_samples,
                allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
                print_audio_search_paths=args.print_audio_search_paths,
                audio_search_print_limit=args.audio_search_print_limit,
                strict_audio_missing=args.strict_audio_missing,
                multitask=False,
                train_target_components=None,
            )
            cot_train_items = build_items_from_rationale_jsonl(
                cot_train_file,
                args.audio_dir,
                add_text_only=args.add_text_only,
                text_only=args.text_only,
                max_samples=train_max_samples,
                allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
                print_audio_search_paths=args.print_audio_search_paths,
                audio_search_print_limit=args.audio_search_print_limit,
                strict_audio_missing=args.strict_audio_missing,
                multitask=False,
                train_target_components=(selected_train_components if explicit_components else None),
            )
            train_items = build_multisource_multitask_items(
                label_train_items,
                cot_train_items,
                cot_task_mode=cot_train_task_mode,
                cot_target_components=(selected_train_components if explicit_components else None),
            )
        else:
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
                multitask=True,
                cot_task_mode=cot_train_task_mode,
                train_target_components=(selected_train_components if explicit_components else None),
            )
        train_items, train_sample_stats = sample_train_items_by_slurp_id(
            train_items,
            ratio=args.train_id_sample_ratio,
            seed=args.train_id_sample_seed,
        )
        if args.random_cot:
            randomized = apply_random_cot_target_ablation(
                train_items,
                seed=args.random_cot_seed,
            )
            if rank == 0:
                logger.info(
                    "random_cot=True: randomized CoT targets for %d sample ids (seed=%d).",
                    randomized,
                    args.random_cot_seed,
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

        if cot_eval_file:
            if rank == 0:
                logger.info(
                    "Using split multitask eval files | label=%s | cot=%s",
                    args.eval_file,
                    cot_eval_file,
                )
            label_eval_items = build_items_from_rationale_jsonl(
                args.eval_file,
                args.audio_dir,
                add_text_only=args.add_text_only,
                text_only=args.text_only,
                max_samples=eval_max_samples,
                allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
                print_audio_search_paths=args.print_audio_search_paths,
                audio_search_print_limit=args.audio_search_print_limit,
                strict_audio_missing=args.strict_audio_missing,
                multitask=False,
                train_target_components=None,
            )
            cot_eval_items = build_items_from_rationale_jsonl(
                cot_eval_file,
                args.audio_dir,
                add_text_only=args.add_text_only,
                text_only=args.text_only,
                max_samples=eval_max_samples,
                allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
                print_audio_search_paths=args.print_audio_search_paths,
                audio_search_print_limit=args.audio_search_print_limit,
                strict_audio_missing=args.strict_audio_missing,
                multitask=False,
                train_target_components=(selected_train_components if explicit_components else None),
            )
            eval_items = build_multisource_multitask_items(
                label_eval_items,
                cot_eval_items,
                cot_task_mode=cot_train_task_mode,
                cot_target_components=(selected_train_components if explicit_components else None),
            )
        else:
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
                multitask=True,
                cot_task_mode=cot_train_task_mode,
                train_target_components=(selected_train_components if explicit_components else None),
            )

        if rank == 0:
            logger.info("Train items: %d | Eval items: %d", len(train_items), len(eval_items))
            train_cot = sum(1 for x in train_items if int(x.get("task_id", -1)) == 0)
            train_label = sum(1 for x in train_items if int(x.get("task_id", -1)) == 1)
            eval_cot = sum(1 for x in eval_items if int(x.get("task_id", -1)) == 0)
            eval_label = sum(1 for x in eval_items if int(x.get("task_id", -1)) == 1)
            logger.info("CoT branch mode for train/eval: %s", cot_train_task_mode)
            logger.info(
                "CoT branch target components for train/eval: %s (%s)",
                selected_train_components,
                describe_target_components(selected_train_components),
            )
            logger.info(
                "Multitask split | train: cot=%d label=%d | eval: cot=%d label=%d",
                train_cot,
                train_label,
                eval_cot,
                eval_label,
            )

        if len(train_items) == 0:
            raise RuntimeError("No train items loaded. Check train_file/audio_dir paths.")
    elif rank == 0:
        logger.info("Inference-only mode enabled. Skipping train/eval data build and training.")

    model_path = str(args.model_name_or_path).strip()
    if not model_path:
        raise ValueError("--model_name_or_path must be non-empty.")
    if rank == 0:
        logger.info("Loading processor/model from: %s", model_path)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor, tokenizer = ensure_processor_tokenizer_or_raise(processor, model_path)
    _ = get_audio_sampling_rate_or_raise(processor, model_path)

    model = load_audio_model_from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    attach_tokenizer_to_model_for_compat(model, tokenizer)
    vocab_size_for_compat = ensure_model_vocab_size_for_compat(model, tokenizer)
    if rank == 0 and vocab_size_for_compat is None:
        logger.warning(
            "Could not infer model.config.vocab_size for compatibility. "
            "If a downstream component expects vocab_size, model-specific code may still fail."
        )

    # Match prior behavior: train audio encoder/projector for training runs; freeze both in inference-only mode.
    audio_matches, projector_matches = configure_audio_trainability(
        model,
        train_audio_encoder=not args.inference_only,
        freeze_projector=args.inference_only,
    )
    if rank == 0:
        logger.info(
            "Trainability | audio_params=%d (enabled=%s), projector_params=%d (enabled=%s)",
            audio_matches,
            not args.inference_only,
            projector_matches,
            not args.inference_only,
        )
        if audio_matches == 0:
            logger.warning(
                "No audio-related parameters were detected by name hints. "
                "Model loading succeeded, but verify fine-tuning targets for this architecture."
            )

    if not args.inference_only:
        training_kwargs: Dict[str, Any] = {
            "output_dir": args.output_dir,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "bf16": True,
            "logging_steps": 1 if args.smoke else 10,
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
            training_kwargs["full_determinism"] = bool(strict_determinism)
        if "tf32" in training_sig:
            training_kwargs["tf32"] = False
        training_args = TrainingArguments(**training_kwargs)

        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": MixedDataset(train_items),
            "eval_dataset": MixedDataset(eval_items) if len(eval_items) > 0 else None,
            "data_collator": SmartCollator(processor, debug=args.smoke),
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
        trainer = CustomTrainer(**trainer_kwargs)

        trainer.train()

        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            trainer.save_model(args.output_dir)
            processor.save_pretrained(args.output_dir)

    if world_size > 1:
        dist.barrier()

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
        multitask=False,
    )
    requested_test_mode = str(args.test_task_mode or "cot").strip().lower()
    test_modes = ["cot", "label"] if requested_test_mode == "both" else [requested_test_mode]
    raw_output_file = args.output_file.strip()

    if rank == 0:
        logger.info("Test inference DataLoader workers: %d", args.inference_num_workers)
        logger.info("Test task modes: %s", ", ".join(test_modes))

    for mode in test_modes:
        normalized_mode = normalize_task_mode(mode)
        cot_components_for_mode = selected_train_components if normalized_mode == "cot" else None
        mode_items = [
            build_task_item(item, normalized_mode, cot_target_components=cot_components_for_mode)
            for item in test_items
        ]
        if raw_output_file:
            base, ext = os.path.splitext(raw_output_file)
            if not ext:
                ext = ".jsonl"
            output_jsonl = f"{base}_{mode}{ext}" if len(test_modes) > 1 else raw_output_file
        else:
            output_jsonl = os.path.join(args.output_dir, f"prediction_{mode}.jsonl")

        output_parent = os.path.dirname(output_jsonl)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)

        if rank == 0:
            logger.info("Test task mode: %s", mode)
            logger.info("Prediction output file: %s", output_jsonl)

        run_distributed_inference(
            model=model,
            processor=processor,
            items=mode_items,
            output_path=output_jsonl,
            model_name_or_path=model_path,
            device=device,
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_workers=args.inference_num_workers,
            seed=args.seed,
        )

        if rank == 0 and args.export_label_eval:
            eval_output_dir = os.path.dirname(output_jsonl) or "."
            label_only_path = os.path.join(eval_output_dir, f"prediction_labels_only_{mode}.jsonl")
            save_label_only_predictions(output_jsonl, label_only_path)

            metrics = evaluate_prediction_file(output_jsonl)
            metrics_path = os.path.join(eval_output_dir, f"metrics_label_only_{mode}.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            logger.info("Label-only evaluation metrics (%s): %s", mode, json.dumps(metrics, ensure_ascii=False))
            logger.info("Saved full predictions (%s): %s", mode, output_jsonl)
            logger.info("Saved label-only predictions (%s): %s", mode, label_only_path)
            logger.info("Saved metrics (%s): %s", mode, metrics_path)
        elif rank == 0:
            logger.info("Saved predictions (%s): %s", mode, output_jsonl)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
