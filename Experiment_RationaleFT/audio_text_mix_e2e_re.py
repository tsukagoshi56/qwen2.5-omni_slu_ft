#!/usr/bin/env python3
"""
Audio+N-best+Rationale training and distributed inference for SLU labels.

- Train target format is kept as:
  {"scenario": "...", "action": "...", "entities": [{"type": "...", "filler": "..."}]}
- Input prompt (audio mode) uses ASR n-best and rationale text.
- Test output JSONL keeps full context (n-best/rationale/target/raw output).
- Evaluation uses label-only extraction from prediction JSONL.
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
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import librosa
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
    import jiwer

    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

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

class ProcessorWithTokenizerProxy:
    def __init__(self, base_processor: Any, tokenizer: Any):
        self._base_processor = base_processor
        self.tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_processor, name)

    def __call__(self, *args, **kwargs):
        return self._base_processor(*args, **kwargs)


def _optional_transformers_class(*names: str) -> Optional[Any]:
    for name in names:
        cls = getattr(hf_transformers, name, None)
        if cls is not None:
            return cls
    return None


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


PROMPT_HEADER = (
    "Predict the final SLU label from ASR n-best and rationale.\n"
    "Output JSON only with keys: scenario, action, entities."
)


def load_audio_model_from_pretrained(
    model_name_or_path: str,
    *,
    torch_dtype: torch.dtype,
    trust_remote_code: bool = True,
):
    model_name_lc = str(model_name_or_path).lower()
    attempts: List[Tuple[str, Any]] = []
    if "qwen2.5-omni" in model_name_lc:
        qwen_omni_cls = _optional_transformers_class(
            "Qwen2_5OmniForConditionalGeneration",
            "Qwen2_5OmniForCausalLM",
            "Qwen2OmniForConditionalGeneration",
            "Qwen2OmniForCausalLM",
        )
        if qwen_omni_cls is not None:
            attempts.append((qwen_omni_cls.__name__, qwen_omni_cls))
    if "voxtral" in model_name_lc:
        voxtral_cls = _optional_transformers_class(
            "VoxtralForConditionalGeneration",
            "VoxtralForCausalLM",
        )
        if voxtral_cls is not None:
            attempts.append((voxtral_cls.__name__, voxtral_cls))
    if "music-flamingo" in model_name_lc:
        music_flamingo_cls = _optional_transformers_class(
            "MusicFlamingoForConditionalGeneration",
            "MusicFlamingoForCausalLM",
        )
        if music_flamingo_cls is not None:
            attempts.append((music_flamingo_cls.__name__, music_flamingo_cls))
    if "audio-flamingo-3" in model_name_lc and AudioFlamingo3ForConditionalGeneration is not None:
        attempts.append(("AudioFlamingo3ForConditionalGeneration", AudioFlamingo3ForConditionalGeneration))
    attempts.extend(
        [
            ("AutoModelForCausalLM", AutoModelForCausalLM),
            ("AutoModel", AutoModel),
        ]
    )
    if Qwen2AudioForConditionalGeneration is not None:
        attempts.append(("Qwen2AudioForConditionalGeneration", Qwen2AudioForConditionalGeneration))
    if "audio-flamingo-3" not in model_name_lc and AudioFlamingo3ForConditionalGeneration is not None:
        attempts.append(("AudioFlamingo3ForConditionalGeneration", AudioFlamingo3ForConditionalGeneration))

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
    # Some multimodal model implementations (including certain AudioFlamingo stacks)
    # access `self.tokenizer` inside generation utilities.
    try:
        if getattr(model, "tokenizer", None) is None:
            setattr(model, "tokenizer", tokenizer)
    except Exception:
        pass


def _chat_template_owner_or_raise(processor: Any) -> Any:
    if hasattr(processor, "apply_chat_template"):
        return processor
    tokenizer = None
    try:
        tokenizer = get_tokenizer_or_raise(processor)
    except Exception:
        tokenizer = None
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer
    raise RuntimeError(
        f"Neither processor ({type(processor).__name__}) nor tokenizer has apply_chat_template."
    )


def configure_audio_trainability(
    model: Any,
    *,
    train_audio_encoder: bool,
    freeze_projector: bool,
) -> Tuple[int, int]:
    audio_matches = 0
    projector_matches = 0
    for name, param in model.named_parameters():
        lname = str(name).lower()
        if any(hint in lname for hint in AUDIO_ENCODER_MODULE_NAME_HINTS):
            param.requires_grad_(bool(train_audio_encoder))
            audio_matches += 1
        if freeze_projector and any(hint in lname for hint in PROJECTOR_MODULE_NAME_HINTS):
            param.requires_grad_(False)
            projector_matches += 1
    return audio_matches, projector_matches


def _audio_chat_content_variants(prompt_text: str, audio_ref: str) -> List[List[Dict[str, str]]]:
    ref = str(audio_ref or "placeholder")
    return [
        [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": prompt_text}],
        [{"type": "audio", "audio_url": ref}, {"type": "text", "text": prompt_text}],
        [{"type": "audio", "url": ref}, {"type": "text", "text": prompt_text}],
        [{"type": "audio", "path": ref}, {"type": "text", "text": prompt_text}],
        [{"type": "text", "text": prompt_text}, {"type": "audio", "url": ref}],
        [{"type": "text", "text": prompt_text}, {"type": "audio", "path": ref}],
    ]


def _dedupe_preserve_order(values: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _apply_chat_template_text_or_raise(processor: Any, messages: List[Dict[str, Any]]) -> str:
    owner = _chat_template_owner_or_raise(processor)
    attempts = [
        {"tokenize": False, "add_generation_prompt": True},
        {"tokenize": False},
        {"tokenize": False, "add_generation_prompt": False},
        {"add_generation_prompt": True},
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
    add_generation_prompt: bool,
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


def build_audio_chat_text_or_raise(processor: Any, prompt_text: str, audio_path: Optional[str]) -> str:
    cached = getattr(processor, "_audio_chat_variant_idx", None)
    contents = _audio_chat_content_variants(prompt_text, audio_path or "placeholder")
    order: List[int] = []
    if isinstance(cached, int) and 0 <= cached < len(contents):
        order.append(cached)
    order.extend([idx for idx in range(len(contents)) if idx not in order])

    last_exc: Optional[Exception] = None
    for idx in order:
        messages = [{"role": "user", "content": contents[idx]}]
        try:
            text = _apply_chat_template_text_or_raise(processor, messages)
            setattr(processor, "_audio_chat_variant_idx", idx)
            return text
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to build audio chat text: {last_exc}")


def build_text_chat_text_or_raise(processor: Any, prompt_text: str) -> str:
    variants: List[List[Dict[str, Any]]] = [
        [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
        [{"role": "user", "content": prompt_text}],
    ]
    last_exc: Optional[Exception] = None
    for messages in variants:
        try:
            return _apply_chat_template_text_or_raise(processor, messages)
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to build text chat prompt: {last_exc}")


def _call_processor_with_audio_or_raise(
    processor: Any,
    *,
    text: str,
    audio: Any,
    sampling_rate: int,
    padding: bool = False,
) -> Dict[str, torch.Tensor]:
    cached = getattr(processor, "_audio_processor_mode", None)
    modes = _dedupe_preserve_order(
        [
            cached,
            ("audio", True, True),
            ("audios", True, True),
            ("audio", False, True),
            ("audios", False, True),
            ("audio", True, False),
            ("audios", True, False),
            ("input_audio", True, True),
            ("input_audios", True, True),
        ]
    )

    errors: List[str] = []
    for mode in modes:
        if not mode:
            continue
        key, wrap_list, use_sr = mode
        for include_padding in (True, False):
            kwargs: Dict[str, Any] = {"text": text, "return_tensors": "pt"}
            if include_padding:
                kwargs["padding"] = padding
            kwargs[key] = [audio] if wrap_list else audio
            if use_sr:
                kwargs["sampling_rate"] = sampling_rate
            try:
                out = processor(**kwargs)
                setattr(processor, "_audio_processor_mode", mode)
                return out
            except Exception as exc:
                errors.append(f"{mode}/{include_padding}: {exc}")
                continue
    detail = " | ".join(errors) if errors else "no audio processor signature accepted"
    raise RuntimeError(f"Failed to encode audio/text inputs. Details: {detail}")


def build_audio_train_inputs_or_raise(
    processor: Any,
    *,
    prompt_text: str,
    target_text: str,
    eos_token: str,
    audio: Any,
    audio_path: Optional[str],
    sampling_rate: int,
) -> Tuple[Dict[str, torch.Tensor], int]:
    # Primary route for Qwen2-Audio style processors.
    prompt_chat_text = None
    try:
        prompt_chat_text = build_audio_chat_text_or_raise(processor, prompt_text, audio_path)
        full_text = prompt_chat_text + target_text + eos_token
        inputs = _call_processor_with_audio_or_raise(
            processor,
            text=full_text,
            audio=audio,
            sampling_rate=sampling_rate,
            padding=False,
        )
        prompt_inputs = _call_processor_with_audio_or_raise(
            processor,
            text=prompt_chat_text,
            audio=audio,
            sampling_rate=sampling_rate,
            padding=False,
        )
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        return inputs, prompt_len
    except Exception:
        pass

    if not audio_path:
        raise RuntimeError("Tokenized chat-template fallback requires a valid audio_path.")

    last_exc: Optional[Exception] = None
    for content in _audio_chat_content_variants(prompt_text, audio_path):
        user_messages = [{"role": "user", "content": content}]
        try:
            prompt_inputs = _apply_chat_template_tokenized_or_raise(
                processor,
                user_messages,
                add_generation_prompt=True,
            )
            assistant_variants = [
                {"role": "assistant", "content": [{"type": "text", "text": target_text + eos_token}]},
                {"role": "assistant", "content": target_text + eos_token},
            ]
            for assistant_message in assistant_variants:
                full_messages = user_messages + [assistant_message]
                try:
                    inputs = _apply_chat_template_tokenized_or_raise(
                        processor,
                        full_messages,
                        add_generation_prompt=False,
                    )
                    prompt_len = int(prompt_inputs["input_ids"].shape[1])
                    if int(inputs["input_ids"].shape[1]) <= prompt_len:
                        continue
                    return inputs, prompt_len
                except Exception as exc:
                    last_exc = exc
                    continue
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to build audio train inputs for processor/model: {last_exc}")


def build_audio_generation_inputs_or_raise(
    processor: Any,
    *,
    prompt_text: str,
    audio: Any,
    audio_path: Optional[str],
    sampling_rate: int,
) -> Dict[str, torch.Tensor]:
    try:
        prompt_chat_text = build_audio_chat_text_or_raise(processor, prompt_text, audio_path)
        return _call_processor_with_audio_or_raise(
            processor,
            text=prompt_chat_text,
            audio=audio,
            sampling_rate=sampling_rate,
            padding=False,
        )
    except Exception:
        pass

    if not audio_path:
        raise RuntimeError("Tokenized chat-template fallback requires a valid audio_path.")

    last_exc: Optional[Exception] = None
    for content in _audio_chat_content_variants(prompt_text, audio_path):
        messages = [{"role": "user", "content": content}]
        try:
            return _apply_chat_template_tokenized_or_raise(
                processor,
                messages,
                add_generation_prompt=True,
            )
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to build audio generation inputs for processor/model: {last_exc}")


def _pad_tensor_list(tensors: List[torch.Tensor], padding_value: float = 0.0) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot pad empty tensor list.")
    if len(tensors) == 1:
        return tensors[0].unsqueeze(0)

    dims = {t.dim() for t in tensors}
    if dims == {1}:
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    if dims == {2} and len({t.shape[1] for t in tensors}) == 1:
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    try:
        return torch.stack(tensors, dim=0)
    except Exception:
        pass

    if len(dims) != 1:
        shapes = [tuple(t.shape) for t in tensors]
        raise RuntimeError(f"Cannot pad tensors with mixed ranks: {shapes}")

    ndims = next(iter(dims))
    max_shape = [max(t.shape[d] for t in tensors) for d in range(ndims)]
    out_shape = (len(tensors),) + tuple(max_shape)
    out = tensors[0].new_full(out_shape, padding_value)
    for idx, tensor in enumerate(tensors):
        slices = (idx,) + tuple(slice(0, size) for size in tensor.shape)
        out[slices] = tensor
    return out


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


def build_prompt_text(item: Dict[str, Any], include_transcript: bool = False) -> str:
    if item.get("prompt_text"):
        return str(item["prompt_text"])
    
    return "Predict the final SLU label."


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

    scenario = str(final_obj.get("scenario", "")).strip()
    action = str(final_obj.get("action", "")).strip()

    intent = str(final_obj.get("intent", "")).strip()
    if (not scenario or not action) and intent:
        inferred_scenario, inferred_action = intent_to_scenario_action(intent)
        scenario = scenario or inferred_scenario
        action = action or inferred_action

    entities = parse_entities(final_obj.get("entities", []))

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
                "scenario": str(parsed.get("scenario", "")).strip(),
                "action": str(parsed.get("action", "")).strip(),
                "entities": parse_entities(parsed.get("entities", [])),
            }
        if "final" in parsed:
            return extract_target_obj(parsed)
        if "intent" in parsed:
            scenario, action = intent_to_scenario_action(str(parsed.get("intent", "")))
            return {
                "scenario": scenario,
                "action": action,
                "entities": parse_entities(parsed.get("entities", [])),
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
    max_samples: Optional[int] = None,
    allow_text_fallback_when_audio_missing: bool = True,
    print_audio_search_paths: bool = False,
    audio_search_print_limit: int = 100,
    strict_audio_missing: bool = False,
    input_format: str = "asr",
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

            # Prepend explicit task instruction based on format
            if input_format == "ipa":
                context_desc = "IPA (International Phonetic Alphabet) n-best context"
            elif input_format == "arp":
                context_desc = "ARPAbet n-best context"
            else:
                context_desc = "ASR n-best context"

            TASK_INSTRUCTION = (
                f"Analyze the provided audio and {context_desc} to predict the SLU label (scenario, action, entities) with a rationale. "
                "Output the result in JSON format."
            )
            if user_text:
                user_text = f"{TASK_INSTRUCTION}\n\n{user_text}"
            else:
                user_text = TASK_INSTRUCTION

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
                 # Use the assistant text EXACTLY as the target
                target_str = assistant_text
                # Use the first audio path
                filename = raw_audios[0]
                
                # Try to resolve generic info for logging/eval (optional)
                sample_id = extract_sample_id(data, fallback_index=parsed_rows)
                candidates = [] # Not needed for training if target is pre-built
                rationale_text = "" 
                target_obj = extract_target_obj_from_assistant(data) # Attempt to parse back for eval metrics
                
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
                if not rationale_text and user_text:
                    rationale_text = user_text

                target_obj = extract_target_obj(data)
                if not target_obj.get("scenario") and not target_obj.get("action") and not target_obj.get("entities"):
                    target_obj = extract_target_obj_from_assistant(data)
                
                final_json = assistant_text.strip() if assistant_text else json.dumps(target_obj, ensure_ascii=False)
                
                # Construct Chain-of-Thought Target: Candidates -> Rationale -> SLU
                nbest_block = "ASR n-best hypotheses:\n" + format_nbest(candidates)
                rationale_block = "Rationale:\n" + (rationale_text if rationale_text else "(none)")
                target_str = f"{nbest_block}\n{rationale_block}\nSLU:{final_json}"
            
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
            fallback_text_items.append({**base_item, "audio_path": None})

            if add_text_only:
                items.append({**base_item, "audio_path": None})

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
        return build_audio_chat_text_or_raise(
            self.processor,
            prompt_text=prompt_text,
            audio_path=item.get("audio_path"),
        )

    def _build_text_chat(self, item: Dict[str, Any]) -> str:
        prompt_text = build_prompt_text(item, include_transcript=True)
        return build_text_chat_text_or_raise(self.processor, prompt_text=prompt_text)

    def _collate_audio(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        aux_tensors: Dict[str, List[torch.Tensor]] = {}

        sr = get_audio_sampling_rate_or_raise(self.processor, "runtime_processor")
        tokenizer = get_tokenizer_or_raise(self.processor)
        eos_token = tokenizer.eos_token or "<|endoftext|>"

        for item in batch:
            if item.get("audio_path") is None:
                continue
            try:
                audio, _ = librosa.load(item["audio_path"], sr=sr)
            except Exception:
                continue

            prompt_text = build_prompt_text(item)

            if self.debug and self._print_count < 5:
                print(f"\n[DEBUG Visualizer] Audio Sample ID: {item.get('id')}")
                print(f"[DEBUG Visualizer] Prompt:\n{prompt_text}")
                print(f"[DEBUG Visualizer] Target:\n{item['target']}")
                self._print_count += 1

            try:
                inputs, prompt_len = build_audio_train_inputs_or_raise(
                    self.processor,
                    prompt_text=prompt_text,
                    target_text=item["target"],
                    eos_token=eos_token,
                    audio=audio,
                    audio_path=item.get("audio_path"),
                    sampling_rate=sr,
                )
            except Exception as exc:
                logger.warning("Skip audio sample %s due to processor mismatch: %s", item.get("id"), exc)
                continue

            ids = inputs["input_ids"][0] if inputs["input_ids"].dim() > 1 else inputs["input_ids"]
            lbs = ids.clone()
            lbs[:prompt_len] = self.ignore_index

            input_ids_list.append(ids)
            labels_list.append(lbs)

            for key, value in inputs.items():
                if key == "input_ids" or not torch.is_tensor(value):
                    continue
                tensor = value
                if tensor.dim() > 0 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                aux_tensors.setdefault(key, []).append(tensor)

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

        batch_dict: Dict[str, torch.Tensor] = {
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
        }

        for key, tensors in aux_tensors.items():
            if not tensors:
                continue
            pad_value = 0.0
            if "mask" in key:
                pad_value = 0
            batch_dict[key] = _pad_tensor_list(tensors, padding_value=pad_value)

        if "attention_mask" not in batch_dict:
            batch_dict["attention_mask"] = (
                batch_dict["input_ids"] != tokenizer.pad_token_id
            ).long()

        return batch_dict

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
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        batch_sampler = DistributedHomogeneousBatchSampler(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            drop_last=self.args.dataloader_drop_last,
            shuffle=True,
            total_epochs=int(self.args.num_train_epochs),
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
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
        device = self.model.device
        self.model.eval()
        sr = get_audio_sampling_rate_or_raise(self.processor, "runtime_processor")

        for item in samples:
            try:
                audio, _ = librosa.load(item["audio_path"], sr=sr)
                prompt_text = build_prompt_text(item)
                inputs = build_audio_generation_inputs_or_raise(
                    self.processor,
                    prompt_text=prompt_text,
                    audio=audio,
                    audio_path=item.get("audio_path"),
                    sampling_rate=sr,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

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


def parse_prediction_label(raw_output: str) -> Dict[str, Any]:
    default_obj = {"scenario": "error", "action": "error", "entities": []}

    # If "SLU:" marker exists, focus on content after it.
    if "SLU:" in raw_output:
        raw_output = raw_output.rsplit("SLU:", 1)[-1]  # Use rsplit to be safe
    
    json_str = clean_json_text(raw_output)
    parsed = None
    try:
        parsed = json.loads(json_str)
    except Exception:
        # Last fallback: extract first {...} block.
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                pass
    
    if not isinstance(parsed, dict):
        return default_obj

    # Extraction Logic
    # 1. Check for "final" wrapper FIRST (To handle: "final": {"intent": "..."})
    final_obj = parsed.get("final")
    if isinstance(final_obj, dict):
        parsed = final_obj

    scenario = parsed.get("scenario")
    action = parsed.get("action")
    entities = parsed.get("entities") or []

    # 2. If scenario/action are missing, try to parse "intent"
    if not scenario and not action:
        intent = parsed.get("intent")
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


def calculate_wer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0
    if HAS_JIWER:
        return float(jiwer.wer(reference, hypothesis))
    return 0.0 if reference.strip() == hypothesis.strip() else 1.0



@dataclass
class InferenceCollator:
    processor: Any

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        tokenizer = get_tokenizer_or_raise(self.processor)
        tokenizer.padding_side = "left"
        sr = get_audio_sampling_rate_or_raise(self.processor, "runtime_processor")

        net_inputs_list: List[Dict[str, torch.Tensor]] = []
        valid_items: List[Dict[str, Any]] = []

        for item in batch:
            if item.get("audio_path"):
                try:
                    audio, _ = librosa.load(item["audio_path"], sr=sr)
                    prompt_text = build_prompt_text(item)
                    net_inputs = build_audio_generation_inputs_or_raise(
                        self.processor,
                        prompt_text=prompt_text,
                        audio=audio,
                        audio_path=item.get("audio_path"),
                        sampling_rate=sr,
                    )
                except Exception as e:
                    logger.warning(f"Failed to load audio for {item.get('id')}: {e}")
                    continue
            else:
                prompt_text = build_prompt_text(item, include_transcript=True)
                text_input = build_text_chat_text_or_raise(self.processor, prompt_text)
                net_inputs = self.processor(
                    text=text_input,
                    padding=False,
                    return_tensors="pt",
                )

            net_inputs_list.append(net_inputs)
            valid_items.append(item)

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

    results: List[Dict[str, Any]] = []
    tokenizer = get_tokenizer_or_raise(processor)
    for item, net_inputs in zip(items, net_inputs_list):
        net_inputs = {k: v.to(device) for k, v in net_inputs.items() if torch.is_tensor(v)}
        with torch.no_grad():
            output_ids = model.generate(
                **net_inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
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


def run_distributed_inference(
    model,
    processor,
    items,
    output_path,
    device,
    rank,
    world_size,
    batch_size=1,
    max_new_tokens: int = 2048,
    num_workers: int = 0,
):
    model.eval()

    my_items = items[rank::world_size]
    # Split audio/text to keep clean batches
    my_audio_items = [x for x in my_items if x.get("audio_path") is not None]
    my_text_items = [x for x in my_items if x.get("audio_path") is None]

    local_results: List[Dict[str, Any]] = []
    tokenizer = get_tokenizer_or_raise(processor)
    tokenizer.padding_side = "left"

    if rank == 0:
        logger.info("Starting Inference. Items: %d (Audio: %d, Text: %d), Batch size: %d",
                    len(my_items), len(my_audio_items), len(my_text_items), batch_size)

    # Audio Loader
    if my_audio_items:
        audio_loader = DataLoader(
            MixedDataset(my_audio_items),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=InferenceCollator(processor),
            drop_last=False,
            shuffle=False,
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
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                logger.error("Rank %d failed on audio batch %d: %s", rank, i, exc)

    # Text Loader
    if my_text_items:
        text_loader = DataLoader(
            MixedDataset(my_text_items),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=InferenceCollator(processor),
            drop_last=False,
            shuffle=False,
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


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        type=str,
        default="/experiments/training_file_make/ASR_cot_data_train.jsonl",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="/lustre/home/71200138/qwen_test/experiments/CoT_maker/ASR_cot_devel.jsonl",
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

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_rationale_label_ft")
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
    parser.add_argument("--max_new_tokens", type=int, default=2048)
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
        "--export_label_eval",
        action="store_true",
        help="Also export label-only predictions and metrics after inference.",
    )
    parser.add_argument("--add_text_only", action="store_true", help="Also add text-only samples.")
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

    parser.add_argument(
        "--input_format",
        type=str,
        default="asr",
        choices=["asr", "ipa", "arp"],
        help="Input text format for the prompt instruction (asr/ipa/arp).",
    )

    args = parser.parse_args()

    # Handle legacy flags if user tries to use them (optional convenience)
    # But since we use argparse choices, we rely on --input_format argument.


    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        logger.info("Using test_file: %s", args.test_file)

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
        max_samples=train_max_samples,
        allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
        print_audio_search_paths=args.print_audio_search_paths,
        audio_search_print_limit=args.audio_search_print_limit,
        strict_audio_missing=args.strict_audio_missing,
        input_format=args.input_format,
    )
    eval_items = build_items_from_rationale_jsonl(
        args.eval_file,
        args.audio_dir,
        add_text_only=args.add_text_only,
        max_samples=eval_max_samples,
        allow_text_fallback_when_audio_missing=not args.no_text_fallback_when_audio_missing,
        print_audio_search_paths=args.print_audio_search_paths,
        audio_search_print_limit=args.audio_search_print_limit,
        strict_audio_missing=args.strict_audio_missing,
        input_format=args.input_format,
    )

    if rank == 0:
        logger.info("Train items: %d | Eval items: %d", len(train_items), len(eval_items))

    if len(train_items) == 0:
        raise RuntimeError("No train items loaded. Check train_file/audio_dir paths.")

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    processor, tokenizer = ensure_processor_tokenizer_or_raise(processor, args.model_name_or_path)
    _ = get_audio_sampling_rate_or_raise(processor, args.model_name_or_path)

    model = load_audio_model_from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    attach_tokenizer_to_model_for_compat(model, tokenizer)

    # Keep the original lightweight FT setup by default.
    audio_matches, projector_matches = configure_audio_trainability(
        model,
        train_audio_encoder=args.train_audio_encoder,
        freeze_projector=True,
    )
    if rank == 0:
        logger.info(
            "Trainability | audio_params=%d (enabled=%s), projector_params=%d (enabled=%s)",
            audio_matches,
            args.train_audio_encoder,
            projector_matches,
            False,
        )
        if args.train_audio_encoder and audio_matches == 0:
            logger.warning(
                "No audio-related parameters were detected by name hints. "
                "Model loading succeeded, but verify fine-tuning targets for this architecture."
            )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=1 if args.smoke else 10,
        eval_strategy="steps" if len(eval_items) > 0 else "no",
        eval_steps=2 if args.smoke else 50,
        save_strategy="no",
        save_total_limit=None,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
        report_to="none",
        disable_tqdm=True,
    )

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
        max_samples=test_max_samples,
        # Follow original script behavior for test: audio-only (no text fallback).
        allow_text_fallback_when_audio_missing=False,
        print_audio_search_paths=args.print_audio_search_paths,
        audio_search_print_limit=args.audio_search_print_limit,
        strict_audio_missing=args.strict_audio_missing,
        input_format=args.input_format,
    )

    output_jsonl = os.path.join(args.output_dir, "prediction.jsonl")
    if rank == 0:
        logger.info("Test inference DataLoader workers: %d", args.inference_num_workers)
    run_distributed_inference(
        model=model,
        processor=processor,
        items=test_items,
        output_path=output_jsonl,
        device=device,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_workers=args.inference_num_workers,
    )

    if world_size > 1:
        dist.barrier()

    if rank == 0 and args.export_label_eval:
        label_only_path = os.path.join(args.output_dir, "prediction_labels_only.jsonl")
        save_label_only_predictions(output_jsonl, label_only_path)

        metrics = evaluate_prediction_file(output_jsonl)
        metrics_path = os.path.join(args.output_dir, "metrics_label_only.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        logger.info("Label-only evaluation metrics: %s", json.dumps(metrics, ensure_ascii=False))
        logger.info("Saved full predictions: %s", output_jsonl)
        logger.info("Saved label-only predictions: %s", label_only_path)
        logger.info("Saved metrics: %s", metrics_path)
    elif rank == 0:
        logger.info("Saved predictions: %s", output_jsonl)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
