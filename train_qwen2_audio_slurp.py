import argparse
import inspect
import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback

try:
    from transformers import Qwen2AudioForConditionalGeneration

    MODEL_CLS = Qwen2AudioForConditionalGeneration
except Exception:
    from transformers import AutoModelForCausalLM

    MODEL_CLS = AutoModelForCausalLM

PROMPT = 'Extract scenario, action, and entities (empty list if none) and return a single-line JSON: {"scenario": "<string>", "action": "<string>", "entities": [{"<entity_type>": "<entity_value>"}, ...]}'


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_audio(path: str, target_sr: Optional[int]) -> torch.Tensor:
    try:
        import soundfile as sf

        audio, sr = sf.read(path)
        if hasattr(audio, "ndim") and audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = torch.tensor(audio, dtype=torch.float32)
    except Exception:
        import torchaudio

        audio, sr = torchaudio.load(path)
        if audio.dim() > 1:
            audio = audio.mean(dim=0)
    if target_sr is not None and sr != target_sr:
        try:
            import torchaudio

            audio = torchaudio.functional.resample(audio, sr, target_sr)
            sr = target_sr
        except Exception as exc:
            try:
                import librosa

                audio = torch.tensor(
                    librosa.resample(audio.numpy(), orig_sr=sr, target_sr=target_sr),
                    dtype=torch.float32,
                )
                sr = target_sr
            except Exception:
                raise RuntimeError(
                    "Resample required but torchaudio/librosa not available."
                ) from exc
    return audio


def load_audio_input(
    audio_input: Any, target_sr: Optional[int]
) -> Optional[torch.Tensor]:
    if audio_input is None:
        return None
    if isinstance(audio_input, str):
        return load_audio(audio_input, target_sr)

    if isinstance(audio_input, (tuple, list)) and len(audio_input) == 2:
        audio, sr = audio_input
        audio = torch.tensor(audio, dtype=torch.float32)
        return resample_audio(audio, sr, target_sr)

    if isinstance(audio_input, dict):
        if "array" in audio_input:
            audio = torch.tensor(audio_input["array"], dtype=torch.float32)
            sr = audio_input.get("sampling_rate")
            return resample_audio(audio, sr, target_sr)
        if "bytes" in audio_input:
            try:
                import soundfile as sf

                audio, sr = sf.read(BytesIO(audio_input["bytes"]))
                if hasattr(audio, "ndim") and audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = torch.tensor(audio, dtype=torch.float32)
                return resample_audio(audio, sr, target_sr)
            except Exception:
                import torchaudio

                audio, sr = torchaudio.load(BytesIO(audio_input["bytes"]))
                if audio.dim() > 1:
                    audio = audio.mean(dim=0)
                return resample_audio(audio, sr, target_sr)
        if "path" in audio_input and isinstance(audio_input["path"], str):
            return load_audio(audio_input["path"], target_sr)

    raise ValueError("Unsupported audio input type.")


def resample_audio(
    audio: torch.Tensor, sr: Optional[int], target_sr: Optional[int]
) -> torch.Tensor:
    if target_sr is None or sr is None or sr == target_sr:
        return audio
    try:
        import torchaudio

        return torchaudio.functional.resample(audio, sr, target_sr)
    except Exception as exc:
        try:
            import librosa

            return torch.tensor(
                librosa.resample(audio.numpy(), orig_sr=sr, target_sr=target_sr),
                dtype=torch.float32,
            )
        except Exception:
            raise RuntimeError(
                "Resample required but torchaudio/librosa not available."
            ) from exc


def resolve_slurp_root(
    data_dir: str, audio_dir: str, slurp_root: Optional[str]
) -> str:
    if slurp_root:
        return str(Path(slurp_root).resolve())

    data_path = Path(data_dir).resolve()
    if data_path.name == "slurp" and data_path.parent.name == "dataset":
        return str(data_path.parent.parent)

    audio_path = Path(audio_dir).resolve()
    if audio_path.name == "audio":
        return str(audio_path.parent)

    return str(Path.cwd() / "slurp")


def ensure_slurp_repo(slurp_root: str, repo_url: str, download_slurp: bool) -> None:
    data_dir = os.path.join(slurp_root, "dataset", "slurp")
    if os.path.exists(data_dir):
        return
    if not download_slurp:
        raise FileNotFoundError(f"Missing SLURP data at {data_dir}.")
    if os.path.exists(slurp_root) and os.listdir(slurp_root):
        raise FileExistsError(
            f"{slurp_root} exists but SLURP data is missing. "
            "Set --slurp_root to an empty path or disable --download_slurp."
        )
    if shutil.which("git") is None:
        raise RuntimeError("git is required to download the SLURP repository.")
    subprocess.run(["git", "clone", repo_url, slurp_root], check=True)


def has_audio(audio_dir: str) -> bool:
    # Check in audio_dir itself
    if os.path.isdir(os.path.join(audio_dir, "slurp_real")):
        return True
    
    # Check in parent directory (slurp/slurp_real instead of slurp/audio/slurp_real)
    parent_dir = os.path.dirname(audio_dir)
    if os.path.isdir(os.path.join(parent_dir, "slurp_real")):
        return True
    
    # Check for .flac files directly in audio_dir
    if os.path.isdir(audio_dir):
        for name in os.listdir(audio_dir):
            if name.endswith(".flac"):
                return True
    return False


def ensure_audio(slurp_root: str, audio_dir: str, download_audio: bool) -> str:
    if has_audio(audio_dir):
        return audio_dir
    if not download_audio:
        raise FileNotFoundError(
            f"Missing audio directory: {audio_dir}. "
            "Re-run with --download_audio or run slurp/scripts/download_audio.sh."
        )
    script_path = os.path.join(slurp_root, "scripts", "download_audio.sh")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Missing download script: {script_path}")
    subprocess.run(["bash", script_path], cwd=slurp_root, check=True)
    default_audio_dir = os.path.join(slurp_root, "audio")
    if audio_dir != default_audio_dir and has_audio(default_audio_dir):
        return default_audio_dir
    if not has_audio(audio_dir):
        raise FileNotFoundError(
            f"Audio download did not populate {audio_dir}. "
            "Try setting --audio_dir to slurp/audio."
        )
    return audio_dir


def resolve_data_dir(slurp_root: str, data_dir: str) -> str:
    if os.path.exists(data_dir):
        return data_dir
    default_data = os.path.join(slurp_root, "dataset", "slurp")
    if os.path.exists(default_data):
        return default_data
    return data_dir


def resolve_audio_path(audio_root: str, filename: str) -> Optional[str]:
    # Check multiple possible locations for audio files
    parent_dir = os.path.dirname(audio_root)
    candidates = [
        os.path.join(audio_root, filename),
        os.path.join(audio_root, "slurp_real", filename),
        # Also check if slurp_real is sibling to audio_root
        os.path.join(parent_dir, "slurp_real", filename),
        # Explicitly check slurp/slurp_real relative to CWD
        os.path.join("slurp", "slurp_real", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def build_massive_entities(
    tokens: Sequence[str], labels: Sequence[str], outside_label: str
) -> List[Dict[str, str]]:
    entities: List[Dict[str, str]] = []
    current_label: Optional[str] = None
    current_tokens: List[str] = []

    def flush() -> None:
        nonlocal current_label, current_tokens
        if current_label and current_tokens:
            entities.append({current_label: " ".join(current_tokens)})
        current_label = None
        current_tokens = []

    for token, label in zip(tokens, labels):
        if label in {outside_label, "O", "Other"}:
            flush()
            continue
        if current_label and label != current_label:
            flush()
        current_label = label
        current_tokens.append(token)

    flush()
    return entities


def build_massive_target(
    record: Dict[str, Any], outside_label: str
) -> str:
    scenario = record.get("scenario_str") or str(record.get("scenario", ""))
    action = record.get("intent_str") or str(record.get("intent", ""))
    tokens = record.get("tokens") or []
    labels = record.get("labels") or []
    entities = build_massive_entities(tokens, labels, outside_label)
    payload = {
        "scenario": scenario,
        "action": action,
        "entities": entities,
    }
    return json.dumps(payload, ensure_ascii=False)


def load_speech_massive_split(
    dataset_name: str,
    dataset_config: str,
    split: str,
    cache_dir: Optional[str],
) -> Any:
    try:
        from datasets import Audio, load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for Speech-MASSIVE.") from exc

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        cache_dir=cache_dir,
    )
    if "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def combine_datasets(datasets_list: List[Any]) -> Any:
    if len(datasets_list) == 1:
        return datasets_list[0]
    try:
        from datasets import concatenate_datasets
    except Exception as exc:
        raise RuntimeError("datasets is required for Speech-MASSIVE.") from exc
    return concatenate_datasets(datasets_list)


def build_entities(tokens: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for ent in entities:
        span = ent.get("span") or []
        words = [tokens[i]["surface"] for i in span if i < len(tokens)]
        value = " ".join(words)
        results.append({ent.get("type", "unknown"): value})
    return results


def build_target(record: Dict[str, Any]) -> str:
    entities = build_entities(record.get("tokens", []), record.get("entities", []))
    payload = {
        "scenario": record.get("scenario", ""),
        "action": record.get("action", ""),
        "entities": entities,
    }
    return json.dumps(payload, ensure_ascii=False)


def select_recordings(
    recordings: List[Dict[str, Any]], use_all: bool
) -> List[Dict[str, Any]]:
    if not recordings:
        return []
    correct = [r for r in recordings if r.get("status") == "correct"]
    if use_all:
        return correct or recordings
    pool = correct or recordings

    def wer_key(item: Dict[str, Any]) -> float:
        wer = item.get("wer")
        return wer if isinstance(wer, (int, float)) else 1e9

    return [min(pool, key=wer_key)]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def build_items(
    jsonl_path: str,
    audio_root: str,
    use_all_recordings: bool,
    add_text_only: bool,
) -> List[Dict[str, Any]]:
    records = load_jsonl(jsonl_path)
    items: List[Dict[str, Any]] = []
    missing_audio = 0
    for record in records:
        target = build_target(record)
        transcript = record.get("sentence", "")
        recordings = select_recordings(record.get("recordings", []), use_all_recordings)
        for rec in recordings:
            audio_path = resolve_audio_path(audio_root, rec["file"])
            if audio_path is None:
                missing_audio += 1
                continue
            items.append(
                {
                    "slurp_id": record.get("slurp_id"),
                    "audio_path": audio_path,
                    "transcript": transcript,
                    "target": target,
                }
            )
        if add_text_only:
            items.append(
                {
                    "slurp_id": record.get("slurp_id"),
                    "audio_path": None,
                    "transcript": transcript,
                    "target": target,
                }
            )
    if missing_audio:
        print(f"Warning: {missing_audio} recordings missing from {audio_root}.")
    return items


class SlurpDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


class SpeechMassiveDataset(Dataset):
    def __init__(
        self,
        dataset: Any,
        transcript_field: str,
        outside_label: str,
        add_text_only: bool,
    ):
        self.dataset = dataset
        self.transcript_field = transcript_field
        self.outside_label = outside_label
        self.add_text_only = add_text_only

    def __len__(self) -> int:
        if self.add_text_only:
            return len(self.dataset) * 2
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.add_text_only:
            base_idx = idx // 2
            text_only = idx % 2 == 1
        else:
            base_idx = idx
            text_only = False

        record = self.dataset[base_idx]
        transcript = (
            record.get(self.transcript_field)
            or record.get("utt")
            or record.get("text")
            or ""
        )
        target = build_massive_target(record, self.outside_label)
        audio = None if text_only else record.get("audio")
        audio_ref = None
        if not text_only:
            audio_ref = record.get("path")
            if not audio_ref and isinstance(record.get("audio"), dict):
                audio_ref = record["audio"].get("path")
        return {
            "audio": audio,
            "audio_ref": audio_ref,
            "transcript": transcript,
            "target": target,
        }


@dataclass
class CollatorConfig:
    include_transcript: bool
    max_length: int
    audio_sampling_rate: Optional[int]


class SampleGenerationCallback(TrainerCallback):
    """Callback to generate samples during training to monitor progress."""
    def __init__(self, processor, items, debug_steps=100, num_samples=3):
        self.processor = processor
        # Randomly sample items for debugging
        if len(items) > num_samples:
            self.items = random.sample(items, num_samples)
        else:
            self.items = items
        self.debug_steps = debug_steps
        self.tokenizer = processor.tokenizer
        print(f"SampleGenerationCallback initialized with {len(self.items)} random samples. Debug steps: {self.debug_steps}", flush=True)
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.debug_steps == 0:
            print(f"\n[Debug Generation] Step {state.global_step}", flush=True)
            model.eval()
            with torch.no_grad():
                for i, item in enumerate(self.items):
                    transcript = item.get("transcript", "")
                    prompt_text = f"{PROMPT}\nTranscript: {transcript}"
                    target = item.get("target")
                    
                    # Prepare input
                    user_content = []
                    audio_input = None
                    
                    if item.get("audio_path"):
                        audio_path = item["audio_path"]
                        # Load audio using the script's global helper if available or local logic
                        # Reusing load_audio helper from global scope
                        try: 
                            target_sr = self.processor.feature_extractor.sampling_rate
                            audio_input = load_audio(audio_path, target_sr=target_sr)
                            user_content.append({"type": "audio", "audio": audio_path}) # Placeholder for template, actual audio passed to processor
                        except Exception as e:
                            print(f"Failed to load audio for debug sample {i}: {e}")
                            
                    user_content.append({"type": "text", "text": prompt_text})
                    
                    messages = [{"role": "user", "content": user_content}]
                    
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    # Prepare arguments for processor
                    process_kwargs = {
                        "text": [text],
                        "return_tensors": "pt",
                        "padding": True
                    }
                    if audio_input is not None:
                        process_kwargs["audios"] = [audio_input]

                    inputs = self.processor(**process_kwargs).to(model.device)
                    
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False
                    )
                    
                    # Decode only the new tokens
                    generated_ids = [
                        output_ids[len(input_ids):] 
                        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]
                    
                    print(f"--- Sample {i+1} ---", flush=True)
                    print(f"Input: {transcript[:50]}...", flush=True)
                    print(f"Target: {target}", flush=True)
                    print(f"Pred:   {response}", flush=True)
            model.train()



class Qwen2AudioCollator:
    def __init__(self, processor: AutoProcessor, config: CollatorConfig):
        self.processor = processor
        self.config = config

    def build_prompt(self, transcript: str) -> str:
        if self.config.include_transcript:
            return f"{PROMPT}\nTranscript: {transcript}"
        return PROMPT

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        features: List[Dict[str, torch.Tensor]] = []
        labels: List[torch.Tensor] = []
        for item in batch:
            audio = None
            audio_input = item.get("audio")
            if audio_input is None:
                audio_input = item.get("audio_path")
            if audio_input:
                audio = load_audio_input(audio_input, self.config.audio_sampling_rate)

            prompt_text = self.build_prompt(item["transcript"])
            user_content = []
            if audio is not None:
                audio_ref = item.get("audio_ref")
                if not audio_ref and isinstance(audio_input, str):
                    audio_ref = audio_input
                if not audio_ref:
                    audio_ref = "audio"
                user_content.append({"type": "audio", "audio": audio_ref})
            user_content.append({"type": "text", "text": prompt_text})

            messages = [{"role": "user", "content": user_content}]
            full_messages = messages + [
                {"role": "assistant", "content": [{"type": "text", "text": item["target"]}]}
            ]

            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )

            call_kwargs = {
                "return_tensors": "pt",
                "truncation": True,
                "max_length": self.config.max_length,
            }
            if audio is not None:
                prompt_inputs = self.processor(text=prompt_text, audios=[audio], **call_kwargs)
                full_inputs = self.processor(text=full_text, audios=[audio], **call_kwargs)
            else:
                prompt_inputs = self.processor(text=prompt_text, **call_kwargs)
                full_inputs = self.processor(text=full_text, **call_kwargs)

            prompt_len = prompt_inputs["input_ids"].shape[1]
            full_ids = full_inputs["input_ids"].squeeze(0)
            if prompt_len >= full_ids.shape[0]:
                raise ValueError(
                    "Prompt length exceeds or equals full sequence length. "
                    "Increase max_length."
                )

            label_ids = full_ids.clone()
            label_ids[:prompt_len] = -100

            feature = {k: v.squeeze(0) for k, v in full_inputs.items()}
            features.append(feature)
            labels.append(label_ids)

        # Separate text and audio
        text_features = [
            {k: v for k, v in f.items() if k in ["input_ids", "attention_mask"]}
            for f in features
        ]
        batch_out = self.processor.tokenizer.pad(text_features, padding=True, return_tensors="pt")

        if "input_features" in features[0]:
            audio_features = [{"input_features": f["input_features"]} for f in features]
            if hasattr(self.processor.feature_extractor, "pad"):
                audio_out = self.processor.feature_extractor.pad(
                    audio_features, padding=True, return_tensors="pt"
                )
                batch_out["input_features"] = audio_out["input_features"]
                if "feature_attention_mask" in audio_out:
                    batch_out["feature_attention_mask"] = audio_out["feature_attention_mask"]
            else:
                batch_out["input_features"] = torch.stack([f["input_features"] for f in features])

        max_len = batch_out["input_ids"].shape[1]
        label_batch = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, label_ids in enumerate(labels):
            label_batch[i, : label_ids.shape[0]] = label_ids
        batch_out["labels"] = label_batch
        return batch_out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument(
        "--dataset",
        choices=["slurp", "speech_massive"],
        default="slurp",
        help="Select dataset pipeline.",
    )
    parser.add_argument("--data_dir", default="slurp/dataset/slurp")
    parser.add_argument("--audio_dir", default="slurp/audio")
    parser.add_argument("--slurp_root", default=None)
    parser.add_argument("--slurp_repo_url", default="https://github.com/pswietojanski/slurp")
    parser.add_argument(
        "--download_slurp",
        action="store_true",
        default=False,
        help="Force download/clone of SLURP repository. If not specified and dataset missing, will try to download.",
    )
    parser.add_argument(
        "--no_download_slurp",
        action="store_false",
        dest="download_slurp",
        help="Do not download SLURP repository (deprecated).",
    )
    parser.add_argument("--download_audio", action="store_true", default=False)
    parser.add_argument("--massive_dataset_name", default="FBK-MT/Speech-MASSIVE")
    parser.add_argument("--massive_dataset_config", default="fr-FR")
    parser.add_argument("--massive_train_split", default="train_115")
    parser.add_argument("--massive_eval_split", default="validation")
    parser.add_argument("--massive_cache_dir", default=None)
    parser.add_argument("--massive_transcript_field", default="utt")
    parser.add_argument("--massive_outside_label", default="Other")
    parser.add_argument("--train_file", default="train.jsonl")
    parser.add_argument("--eval_file", default="devel.jsonl")
    parser.add_argument("--output_dir", default="outputs/qwen2-audio-slurp")
    parser.add_argument("--include_transcript", action="store_true", default=True)
    parser.add_argument("--no_include_transcript", action="store_false", dest="include_transcript")
    parser.add_argument("--add_text_only", action="store_true")
    parser.add_argument("--use_all_recordings", action="store_true")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--warmup_ratio", type=float, default=0.04)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_lora", action="store_true", default=False)
    # parser.add_argument("--no_lora", action="store_false", dest="use_lora") # Deprecated since False is default
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Push the trained model to the Hugging Face Hub.")
    parser.add_argument("--debug_generation", action="store_true", help="Enable debug generation during training.")
    parser.add_argument("--debug_generation_steps", type=int, default=5, help="Steps between debug generations.")
    return parser


def make_training_arguments(
    args: argparse.Namespace, eval_strategy: str
) -> TrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps if args.max_steps > 0 else -1,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 2,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": 2,
        "push_to_hub": args.push_to_hub,
    }

    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_strategy

    # Default to cosine decay per paper
    if kwargs.get("lr_scheduler_type") is None:
        kwargs["lr_scheduler_type"] = "cosine"

    return TrainingArguments(**kwargs)


def maybe_apply_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if not args.use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise RuntimeError("peft is required for --use_lora.") from exc

    target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    eval_items: List[Dict[str, Any]] = []
    if args.dataset == "slurp":
        slurp_root = resolve_slurp_root(args.data_dir, args.audio_dir, args.slurp_root)
        
        # Check if dataset exists, if not, try to download (clone)
        dataset_path = os.path.join(slurp_root, "dataset", "slurp")
        should_download = args.download_slurp or not os.path.exists(dataset_path)
        
        if should_download and not os.path.exists(dataset_path):
            print(f"SLURP dataset not found at {dataset_path}. Downloading...")
        
        ensure_slurp_repo(slurp_root, args.slurp_repo_url, download_slurp=should_download)

        data_dir = resolve_data_dir(slurp_root, os.path.abspath(args.data_dir))
        audio_dir = os.path.abspath(args.audio_dir)
        
        # Check if audio exists, if not and download is requested (or force check)
        if not args.add_text_only:
            audio_dir = ensure_audio(slurp_root, audio_dir, args.download_audio)

        train_path = os.path.join(data_dir, args.train_file)
        eval_path = os.path.join(data_dir, args.eval_file)
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Missing training file: {train_path}")
        if not os.path.exists(audio_dir) and not args.add_text_only:
            raise FileNotFoundError(
                f"Missing audio directory: {audio_dir} (run scripts/download_audio.sh)"
            )

        train_items = build_items(
            train_path, audio_dir, args.use_all_recordings, args.add_text_only
        )
        if args.max_train_samples:
            train_items = train_items[: args.max_train_samples]
        
        print(f"Num train items: {len(train_items)}")
        
        if os.path.exists(eval_path):
            eval_items = build_items(
                eval_path, audio_dir, args.use_all_recordings, args.add_text_only
            )
            if args.max_eval_samples:
                eval_items = eval_items[: args.max_eval_samples]
            print(f"Num eval items: {len(eval_items)}")
        
        train_dataset = SlurpDataset(train_items)
        eval_dataset = SlurpDataset(eval_items) if eval_items else None
    else:
        configs = [
            cfg.strip()
            for cfg in args.massive_dataset_config.split(",")
            if cfg.strip()
        ]
        train_datasets = [
            load_speech_massive_split(
                args.massive_dataset_name,
                cfg,
                args.massive_train_split,
                args.massive_cache_dir,
            )
            for cfg in configs
        ]
        train_hf = combine_datasets(train_datasets)
        if args.max_train_samples:
            train_hf = train_hf.select(range(args.max_train_samples))

        eval_dataset = None
        if args.massive_eval_split:
            eval_datasets = [
                load_speech_massive_split(
                    args.massive_dataset_name,
                    cfg,
                    args.massive_eval_split,
                    args.massive_cache_dir,
                )
                for cfg in configs
            ]
            eval_hf = combine_datasets(eval_datasets)
            if args.max_eval_samples:
                eval_hf = eval_hf.select(range(args.max_eval_samples))
            eval_dataset = SpeechMassiveDataset(
                eval_hf,
                args.massive_transcript_field,
                args.massive_outside_label,
                args.add_text_only,
            )

        train_dataset = SpeechMassiveDataset(
            train_hf,
            args.massive_transcript_field,
            args.massive_outside_label,
            args.add_text_only,
        )

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    sampling_rate = None
    if hasattr(processor, "feature_extractor") and hasattr(
        processor.feature_extractor, "sampling_rate"
    ):
        sampling_rate = processor.feature_extractor.sampling_rate
    if hasattr(processor, "sampling_rate"):
        sampling_rate = processor.sampling_rate

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }


    model = MODEL_CLS.from_pretrained(
        args.model_name_or_path, **model_kwargs
    )
    model.config.use_cache = False
    # Freeze audio components as per paper
    if hasattr(model, "audio_tower"):
        for param in model.audio_tower.parameters():
            param.requires_grad = False
    if hasattr(model, "multi_modal_projector"):
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False

    model = maybe_apply_lora(model, args)

    # Add custom config flag heavily relying on duck-typing model config
    if hasattr(model, "config"):
        model.config.slurp_fmt = "paper_v2"
    elif hasattr(model, "module") and hasattr(model.module, "config"):
        model.module.config.slurp_fmt = "paper_v2"

    collator = Qwen2AudioCollator(
        processor,
        CollatorConfig(
            include_transcript=args.include_transcript,
            max_length=args.max_length,
            audio_sampling_rate=sampling_rate,
        ),
    )

    eval_strategy = "steps" if eval_dataset else "no"
    training_args = make_training_arguments(args, eval_strategy)

    trainer_callbacks = []
    if args.debug_generation:
        # Use eval items if available, else train items
        debug_items = eval_items if eval_items else train_items
        trainer_callbacks.append(
            SampleGenerationCallback(
                processor, 
                debug_items, 
                debug_steps=args.debug_generation_steps
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=trainer_callbacks,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    
    # Explicitly save config and processor to ensure checkpoint is complete
    # When using LoRA/PEFT, model is wrapped in PeftModel, so we need to get base model's config
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            base_config = model.get_base_model().config
        else:
            base_config = model.config
    except ImportError:
        base_config = model.config
    
    base_config.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
