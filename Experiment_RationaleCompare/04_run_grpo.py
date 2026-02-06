#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from common import (
    build_db_definitions,
    compute_reward,
    label_from_record,
    load_metadata,
    parse_j_from_output,
    read_jsonl,
    resolve_audio_path,
)
from prompts import render_infer_audio_prompt, render_infer_text_prompt


@dataclass
class GrpoItem:
    slurp_id: Any
    sentence: str
    audio_path: Optional[str]
    gold_label: Dict[str, Any]
    mode: str


class GrpoDataset(Dataset):
    def __init__(self, items: List[GrpoItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> GrpoItem:
        return self.items[idx]


def collate_grpo_items(batch: List[GrpoItem]) -> List[GrpoItem]:
    # Keep dataclass items as-is; default_collate cannot stack custom classes.
    return batch


def build_items(input_file: str, audio_dir: str, include_text: bool) -> List[GrpoItem]:
    records = read_jsonl(input_file)
    items: List[GrpoItem] = []
    for record in records:
        gold_label = label_from_record(record)
        sentence = str(record.get("sentence") or record.get("text") or "").strip()
        recordings = record.get("recordings", []) if isinstance(record.get("recordings"), list) else []

        audio_path = None
        if recordings:
            rec = recordings[0] if isinstance(recordings[0], dict) else None
            filename = rec.get("file") if rec else None
            audio_path = resolve_audio_path(audio_dir, filename) if filename else None
            if audio_path:
                items.append(
                    GrpoItem(
                        slurp_id=record.get("slurp_id"),
                        sentence=sentence,
                        audio_path=audio_path,
                        gold_label=gold_label,
                        mode="audio",
                    )
                )

        if include_text and sentence:
            items.append(
                GrpoItem(
                    slurp_id=record.get("slurp_id"),
                    sentence=sentence,
                    audio_path=None,
                    gold_label=gold_label,
                    mode="text",
                )
            )

    return items


def build_chat_input(processor: AutoProcessor, prompt: str, audio: bool) -> str:
    if audio:
        user_content = [
            {"type": "audio", "audio_url": "placeholder"},
            {"type": "text", "text": prompt},
        ]
    else:
        user_content = [{"type": "text", "text": prompt}]
    return processor.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_inputs(
    processor: AutoProcessor,
    prompt_text: str,
    full_text: str,
    audio: Optional[torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if audio is None:
        inputs = processor(text=full_text, return_tensors="pt")
        prompt_inputs = processor(text=prompt_text, return_tensors="pt")
    else:
        sr = processor.feature_extractor.sampling_rate
        inputs = processor(text=full_text, audio=[audio], sampling_rate=sr, return_tensors="pt")
        prompt_inputs = processor(text=prompt_text, audio=[audio], sampling_rate=sr, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
    inputs["_prompt_len"] = prompt_inputs["input_ids"].shape[1]
    return inputs


def model_forward(model: Qwen2AudioForConditionalGeneration, inputs: Dict[str, torch.Tensor]):
    kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
    }
    if "input_features" in inputs:
        kwargs["input_features"] = inputs["input_features"]
    if "feature_attention_mask" in inputs:
        kwargs["feature_attention_mask"] = inputs["feature_attention_mask"]
    return model(**kwargs)


def compute_logprob_sum(
    model: Qwen2AudioForConditionalGeneration,
    inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    prompt_len = int(inputs.get("_prompt_len", 0))
    outputs = model_forward(model, inputs)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    input_ids = inputs["input_ids"]

    target_ids = input_ids[:, 1:]
    log_probs = log_probs[:, :-1, :]
    start = max(prompt_len - 1, 0)
    if start >= log_probs.shape[1]:
        return torch.tensor(0.0, device=input_ids.device)
    token_logprobs = log_probs[0, start:, :].gather(1, target_ids[0, start:].unsqueeze(-1)).squeeze(-1)
    return token_logprobs.sum()


def generate_samples(
    model: Qwen2AudioForConditionalGeneration,
    processor: AutoProcessor,
    prompt_text: str,
    audio: Optional[torch.Tensor],
    device: torch.device,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> List[str]:
    text_input = build_chat_input(processor, prompt_text, audio is not None)
    if audio is None:
        inputs = processor(text=text_input, return_tensors="pt")
    else:
        sr = processor.feature_extractor.sampling_rate
        inputs = processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs: List[str] = []
    for _ in range(group_size):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        input_len = inputs["input_ids"].shape[1]
        generated_text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        outputs.append(generated_text)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO fine-tuning after SF-CoT.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, default="Experiment_3/slurp_metadata.json")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio/slurp_real")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--ref_model_name_or_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/grpo")
    parser.add_argument("--include_text", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--kl_beta", type=float, default=0.01)
    parser.add_argument("--advantage_normalize", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reward_w_scenario", type=float, default=1.0)
    parser.add_argument("--reward_w_action", type=float, default=1.0)
    parser.add_argument("--reward_w_intent", type=float, default=0.5)
    parser.add_argument("--reward_w_entity", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, args.train_file) if not os.path.isabs(args.train_file) else args.train_file
    metadata_path = os.path.join(base_dir, args.metadata_file) if not os.path.isabs(args.metadata_file) else args.metadata_file
    audio_dir = os.path.join(base_dir, args.audio_dir) if not os.path.isabs(args.audio_dir) else args.audio_dir
    output_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    metadata = load_metadata(metadata_path)
    db_definitions = build_db_definitions(metadata)

    items = build_items(train_path, audio_dir, include_text=args.include_text)
    dataset = GrpoDataset(items)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_grpo_items,
    )

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.train()

    ref_path = args.ref_model_name_or_path or args.model_name_or_path
    ref_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        ref_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    global_step = 0
    for epoch in range(args.num_train_epochs):
        for batch in dataloader:
            batch_loss = torch.tensor(0.0, device=device)
            sample_count = 0

            for item in batch:
                audio = None
                if item.audio_path:
                    sr = processor.feature_extractor.sampling_rate
                    audio, _ = librosa.load(item.audio_path, sr=sr)

                if item.mode == "audio":
                    prompt = render_infer_audio_prompt(db_definitions)
                else:
                    prompt = render_infer_text_prompt(db_definitions, item.sentence)

                samples = generate_samples(
                    model=model,
                    processor=processor,
                    prompt_text=prompt,
                    audio=audio,
                    device=device,
                    group_size=args.group_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                )

                rewards: List[float] = []
                pred_labels: List[Dict[str, Any]] = []
                for text in samples:
                    pred_obj = parse_j_from_output(text) or {}
                    pred_label = {
                        "scenario": str(pred_obj.get("scenario", "")).strip(),
                        "action": str(pred_obj.get("action", "")).strip(),
                        "entities": pred_obj.get("entities", []) if isinstance(pred_obj.get("entities", []), list) else [],
                    }
                    reward, _ = compute_reward(
                        pred_label,
                        item.gold_label,
                        w_scenario=args.reward_w_scenario,
                        w_action=args.reward_w_action,
                        w_intent=args.reward_w_intent,
                        w_entity=args.reward_w_entity,
                    )
                    rewards.append(reward)
                    pred_labels.append(pred_label)

                mean_reward = sum(rewards) / max(len(rewards), 1)
                if args.advantage_normalize:
                    variance = sum((r - mean_reward) ** 2 for r in rewards) / max(len(rewards), 1)
                    std = math.sqrt(variance) if variance > 0 else 1.0
                else:
                    std = 1.0

                for sample_text, reward in zip(samples, rewards):
                    advantage = (reward - mean_reward) / (std + 1e-6)
                    prompt_text = build_chat_input(processor, prompt, audio is not None)
                    full_text = prompt_text + sample_text

                    inputs = prepare_inputs(
                        processor=processor,
                        prompt_text=prompt_text,
                        full_text=full_text,
                        audio=audio,
                        device=device,
                    )
                    logprob = compute_logprob_sum(model, inputs)

                    with torch.no_grad():
                        ref_inputs = prepare_inputs(
                            processor=processor,
                            prompt_text=prompt_text,
                            full_text=full_text,
                            audio=audio,
                            device=device,
                        )
                        ref_logprob = compute_logprob_sum(ref_model, ref_inputs)

                    kl = logprob - ref_logprob
                    loss = -(advantage * logprob) + args.kl_beta * kl
                    batch_loss += loss
                    sample_count += 1

            if sample_count == 0:
                continue
            batch_loss = batch_loss / sample_count
            batch_loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (global_step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if args.log_every and global_step % args.log_every == 0:
                print(f"[GRPO] step={global_step} loss={batch_loss.item():.4f} samples={sample_count}")

            if args.save_every and global_step > 0 and global_step % args.save_every == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)

            global_step += 1

    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
