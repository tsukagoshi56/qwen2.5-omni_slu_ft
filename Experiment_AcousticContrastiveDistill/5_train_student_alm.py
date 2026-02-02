#!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from common import read_jsonl, write_jsonl

try:
    from transformers import Qwen2AudioForConditionalGeneration

    MODEL_CLS = Qwen2AudioForConditionalGeneration
except Exception:
    MODEL_CLS = AutoModelForCausalLM


def build_prompt(row: Dict[str, Any]) -> str:
    system = str(row.get("system_prompt", "")).strip()
    user = str(row.get("user_prompt", "")).strip()
    return f"System: {system}\nUser: {user}\nAssistant:"


def safe_load_audio(path: str, sampling_rate: int) -> List[float]:
    wav, _ = librosa.load(path, sr=sampling_rate, mono=True)
    return wav.tolist()


def parse_pred_intent(text: str) -> str:
    m = re.search(r'"intent"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1).strip()
    return ""


class DistillDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


@dataclass
class TrainCollator:
    processor: Any
    sampling_rate: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [build_prompt(row) for row in batch]
        targets = [str(row.get("target_text", "")).strip() for row in batch]
        texts = [f"{p} {t}" for p, t in zip(prompts, targets)]
        audios = [safe_load_audio(str(row["audio_path"]), self.sampling_rate) for row in batch]

        model_inputs = self.processor(
            text=texts,
            audios=audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        labels = model_inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        model_inputs["labels"] = labels
        return model_inputs


def run_train(args: argparse.Namespace) -> None:
    train_rows = read_jsonl(args.train_file)
    eval_rows = read_jsonl(args.eval_file) if args.eval_file else []
    if args.limit:
        train_rows = train_rows[: args.limit]
        eval_rows = eval_rows[: max(1, args.limit // 10)] if eval_rows else []
    if not train_rows:
        raise SystemExit(f"No train rows found: {args.train_file}")

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = MODEL_CLS.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    train_dataset = DistillDataset(train_rows)
    eval_dataset = DistillDataset(eval_rows) if eval_rows else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=TrainCollator(processor=processor, sampling_rate=args.sampling_rate),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[OK] model saved -> {args.output_dir}")


def run_predict(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input_file)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No rows found: {args.input_file}")

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = MODEL_CLS.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    outputs: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        prompt = build_prompt(row)
        audio_path = str(row.get("audio_path", "")).strip()
        audio = safe_load_audio(audio_path, args.sampling_rate)

        inputs = processor(
            text=[prompt],
            audios=[audio],
            sampling_rate=args.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        out_ids = gen_ids[0][prompt_len:]
        text = processor.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        pred_intent = parse_pred_intent(text)

        outputs.append(
            {
                **row,
                "prediction_text": text,
                "pred_intent": pred_intent,
            }
        )
        if idx % 20 == 0:
            print(f"[INFO] predicted {idx}/{len(rows)}")

    write_jsonl(args.output_file, outputs)
    print(f"[OK] wrote predictions -> {args.output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/predict student ALM for acoustic-rationale distillation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train")
    p_train.add_argument("--train_file", type=str, default="Experiment_AcousticContrastiveDistill/outputs/04_distill_train.jsonl")
    p_train.add_argument("--eval_file", type=str, default="Experiment_AcousticContrastiveDistill/outputs/04_distill_eval.jsonl")
    p_train.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    p_train.add_argument("--output_dir", type=str, default="outputs/acoustic_contrastive_student")
    p_train.add_argument("--num_train_epochs", type=float, default=1.0)
    p_train.add_argument("--per_device_train_batch_size", type=int, default=1)
    p_train.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p_train.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p_train.add_argument("--learning_rate", type=float, default=1e-5)
    p_train.add_argument("--warmup_ratio", type=float, default=0.03)
    p_train.add_argument("--logging_steps", type=int, default=10)
    p_train.add_argument("--save_steps", type=int, default=200)
    p_train.add_argument("--eval_steps", type=int, default=200)
    p_train.add_argument("--sampling_rate", type=int, default=16000)
    p_train.add_argument("--dataloader_num_workers", type=int, default=2)
    p_train.add_argument("--limit", type=int, default=None)

    p_pred = subparsers.add_parser("predict")
    p_pred.add_argument("--input_file", type=str, default="Experiment_AcousticContrastiveDistill/outputs/04_distill_eval.jsonl")
    p_pred.add_argument("--output_file", type=str, default="Experiment_AcousticContrastiveDistill/outputs/05_student_predictions.jsonl")
    p_pred.add_argument("--model_name_or_path", type=str, default="outputs/acoustic_contrastive_student")
    p_pred.add_argument("--sampling_rate", type=int, default=16000)
    p_pred.add_argument("--max_new_tokens", type=int, default=128)
    p_pred.add_argument("--device", type=str, default="auto")
    p_pred.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
