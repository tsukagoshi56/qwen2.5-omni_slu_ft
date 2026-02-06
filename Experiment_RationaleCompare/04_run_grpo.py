#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import librosa
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from common import (
    compare_labels,
    compute_reward,
    label_from_record,
    parse_j_from_output,
    read_jsonl,
    resolve_audio_path,
)


SYSTEM_PROMPT_TEXT = (
    'System: SLU Logic Analyst. Infer the intent and slots using "Transcript".'
)
SYSTEM_PROMPT_AUDIO = (
    'System: SLU Logic Analyst. Infer the intent and slots using "Audio".'
)
PROMPT_OUTPUT_FORMAT = (
    "Output Format:\n"
    "C: Intent candidates: intent1 | intent2 | intent3; Slot candidates: slot_type1(value1|value2) | slot_type2\n"
    "R: label1!reason1; label2!reason2; ...\n"
    "J: [Final JSON]"
)
DEFAULT_ONLY_GRPO_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"


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


def build_grpo_prompt(mode: str, sentence: str) -> str:
    if mode == "audio":
        return (
            f"{SYSTEM_PROMPT_AUDIO}\n\n"
            f"{PROMPT_OUTPUT_FORMAT}\n\n"
            "[Input Data]\n"
            "- Audio: <AUDIO>"
        )
    text = str(sentence or "").strip()
    return (
        f"{SYSTEM_PROMPT_TEXT}\n\n"
        f"{PROMPT_OUTPUT_FORMAT}\n\n"
        "[Input Data]\n"
        f"- Transcript: {text}"
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


def _shorten(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def _debug_write_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _debug_print_dataset(items: List[GrpoItem], preview_items: int) -> None:
    audio_count = sum(1 for x in items if x.mode == "audio")
    text_count = sum(1 for x in items if x.mode == "text")
    print(f"[DEBUG] dataset_size={len(items)} audio_items={audio_count} text_items={text_count}")
    for idx, item in enumerate(items[:preview_items]):
        print(
            "[DEBUG] item_preview "
            f"idx={idx} slurp_id={item.slurp_id} mode={item.mode} "
            f"audio_path={item.audio_path} "
            f"sentence_len={len(item.sentence)} "
            f"gold={json.dumps(item.gold_label, ensure_ascii=False)}"
        )


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank_world() -> Tuple[int, int]:
    if _is_distributed():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def evaluate_model(
    model,
    processor: AutoProcessor,
    items: List[GrpoItem],
    device: torch.device,
    max_new_tokens: int,
    rank: int,
    world_size: int,
    debug: bool = False,
    debug_max_chars: int = 1200,
    collect_predictions: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    eval_model = _unwrap_model(model)
    was_training = eval_model.training
    eval_model.eval()

    local_items = items[rank::world_size] if world_size > 1 else items
    local_count = 0.0
    reward_sum = 0.0
    scenario_sum = 0.0
    action_sum = 0.0
    intent_sum = 0.0
    entity_f1_sum = 0.0
    local_prediction_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for idx, item in enumerate(local_items):
            audio = None
            if item.audio_path:
                try:
                    sr = processor.feature_extractor.sampling_rate
                    audio, _ = librosa.load(item.audio_path, sr=sr)
                except Exception:
                    continue

            prompt = build_grpo_prompt(item.mode, item.sentence)
            text_input = build_chat_input(processor, prompt, audio is not None)
            if audio is None:
                inputs = processor(text=text_input, return_tensors="pt")
            else:
                sr = processor.feature_extractor.sampling_rate
                inputs = processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            output_ids = eval_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            input_len = inputs["input_ids"].shape[1]
            generated_text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

            pred_obj = parse_j_from_output(generated_text) or {}
            pred_label = {
                "scenario": str(pred_obj.get("scenario", "")).strip(),
                "action": str(pred_obj.get("action", "")).strip(),
                "entities": pred_obj.get("entities", []) if isinstance(pred_obj.get("entities", []), list) else [],
            }
            reward, _ = compute_reward(pred_label, item.gold_label)
            stats = compare_labels(pred_label, item.gold_label)

            local_count += 1.0
            reward_sum += float(reward)
            scenario_sum += 1.0 if stats["scenario_ok"] else 0.0
            action_sum += 1.0 if stats["action_ok"] else 0.0
            intent_sum += 1.0 if stats["intent_ok"] else 0.0
            entity_f1_sum += float(stats["entity_f1"])

            if collect_predictions:
                local_prediction_rows.append(
                    {
                        "scenario": pred_label["scenario"],
                        "action": pred_label["action"],
                        "entities": pred_label["entities"],
                        "pred_label": pred_label,
                        "file": os.path.basename(item.audio_path) if item.audio_path else None,
                        "slurp_id": item.slurp_id,
                        "id": item.slurp_id,
                        "wer": 0.0,
                        "transcript": item.sentence,
                        "candidates": [],
                        "rationale_text": "",
                        "raw_output": generated_text,
                        "target": "",
                        "target_label": item.gold_label,
                        "type": "audio" if item.audio_path else "text",
                    }
                )

            if debug and rank == 0 and idx < 2:
                print(f"[EVAL-DEBUG] mode={item.mode} slurp_id={item.slurp_id}")
                print(f"[EVAL-DEBUG] gold={json.dumps(item.gold_label, ensure_ascii=False)}")
                print(f"[EVAL-DEBUG] pred={json.dumps(pred_label, ensure_ascii=False)} reward={reward:.4f}")
                print("[EVAL-DEBUG] raw:")
                print(_shorten(generated_text, debug_max_chars))

    metrics_tensor = torch.tensor(
        [local_count, reward_sum, scenario_sum, action_sum, intent_sum, entity_f1_sum],
        device=device,
        dtype=torch.float32,
    )
    if world_size > 1 and _is_distributed():
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

    total = float(metrics_tensor[0].item())
    if total <= 0:
        result = {
            "num_samples": 0.0,
            "reward_mean": 0.0,
            "scenario_acc": 0.0,
            "action_acc": 0.0,
            "intent_acc": 0.0,
            "entity_f1_mean": 0.0,
        }
    else:
        result = {
            "num_samples": total,
            "reward_mean": float(metrics_tensor[1].item() / total),
            "scenario_acc": float(metrics_tensor[2].item() / total),
            "action_acc": float(metrics_tensor[3].item() / total),
            "intent_acc": float(metrics_tensor[4].item() / total),
            "entity_f1_mean": float(metrics_tensor[5].item() / total),
        }

    prediction_rows: List[Dict[str, Any]] = []
    if collect_predictions:
        if world_size > 1 and _is_distributed():
            gathered: List[List[Dict[str, Any]]] = [None] * world_size
            dist.all_gather_object(gathered, local_prediction_rows)
            if rank == 0:
                for chunk in gathered:
                    if chunk:
                        prediction_rows.extend(chunk)
        else:
            prediction_rows = local_prediction_rows

    if was_training:
        eval_model.train()
    return result, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO fine-tuning after SF-CoT.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default="", help="Optional eval jsonl path.")
    parser.add_argument("--test_file", type=str, default="", help="Optional test jsonl path.")
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="Experiment_3/slurp_metadata.json",
        help="Unused in current minimal-prompt mode (kept for backward compatibility).",
    )
    parser.add_argument("--audio_dir", type=str, default="slurp/slurp_real")
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--ref_model_name_or_path", type=str, default="")
    parser.add_argument(
        "--only_grpo",
        "--only-grpo",
        dest="only_grpo",
        action="store_true",
        help="Run GRPO directly from the specified base model (no SFT prerequisite in this script).",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/grpo")
    parser.add_argument(
        "--include_text",
        dest="include_text",
        action="store_true",
        help="Include text-mode samples in training (default: enabled).",
    )
    parser.add_argument(
        "--no_include_text",
        "--no-include-text",
        dest="include_text",
        action="store_false",
        help="Disable text-mode samples and train with audio-mode samples only.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=0, help="Run eval every N global steps (0 disables).")
    parser.add_argument("--eval_max_samples", type=int, default=None, help="Cap eval items for faster validation.")
    parser.add_argument("--test_every", type=int, default=0, help="Run test every N global steps (0 disables).")
    parser.add_argument("--test_max_samples", type=int, default=None, help="Cap test items for faster validation.")
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
    parser.add_argument("--debug", action="store_true", help="Print rich debug information.")
    parser.add_argument("--debug_preview_items", type=int, default=5, help="Dataset preview rows in debug.")
    parser.add_argument("--debug_preview_steps", type=int, default=3, help="Training steps to trace in debug.")
    parser.add_argument("--debug_preview_samples", type=int, default=3, help="Generated samples to show per item.")
    parser.add_argument("--debug_max_chars", type=int, default=1200, help="Max chars per debug text field.")
    parser.add_argument(
        "--debug_output_file",
        type=str,
        default="",
        help="Optional JSONL path for debug traces (default: <output_dir>/grpo_debug_trace.jsonl).",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a tiny sanity-check setting.")
    parser.add_argument("--smoke_train_samples", type=int, default=64)
    parser.add_argument("--smoke_eval_samples", type=int, default=32)
    parser.add_argument("--smoke_test_samples", type=int, default=32)
    parser.set_defaults(include_text=False)
    args = parser.parse_args()
    if args.smoke:
        args.num_train_epochs = 1
        args.group_size = min(args.group_size, 2)
        args.max_new_tokens = min(args.max_new_tokens, 96)
        args.log_every = 1
        if args.eval_file and args.eval_every <= 0:
            args.eval_every = 10
        if args.test_file and args.test_every <= 0:
            args.test_every = 10

    auto_model_from_only_grpo = args.only_grpo and not str(args.model_name_or_path).strip()
    if auto_model_from_only_grpo:
        args.model_name_or_path = DEFAULT_ONLY_GRPO_MODEL
    if not str(args.model_name_or_path).strip():
        raise ValueError(
            "--model_name_or_path is required unless --only-grpo is set "
            f"(then it defaults to {DEFAULT_ONLY_GRPO_MODEL})."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank != -1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        rank, world_size = _get_rank_world()
    else:
        rank, world_size = 0, 1

    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, args.train_file) if not os.path.isabs(args.train_file) else args.train_file
    eval_path = (
        os.path.join(base_dir, args.eval_file)
        if (args.eval_file and not os.path.isabs(args.eval_file))
        else args.eval_file
    )
    test_path = (
        os.path.join(base_dir, args.test_file)
        if (args.test_file and not os.path.isabs(args.test_file))
        else args.test_file
    )
    audio_dir = os.path.join(base_dir, args.audio_dir) if not os.path.isabs(args.audio_dir) else args.audio_dir
    output_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    if distributed:
        dist.barrier()

    if args.debug and rank == 0:
        if args.debug_output_file:
            debug_output_path = (
                args.debug_output_file
                if os.path.isabs(args.debug_output_file)
                else os.path.join(base_dir, args.debug_output_file)
            )
        else:
            debug_output_path = os.path.join(output_dir, "grpo_debug_trace.jsonl")
        os.makedirs(os.path.dirname(debug_output_path), exist_ok=True)
        if os.path.exists(debug_output_path):
            os.remove(debug_output_path)
    else:
        debug_output_path = ""

    items = build_items(train_path, audio_dir, include_text=args.include_text)
    if args.smoke:
        items = items[: max(1, args.smoke_train_samples)]

    eval_items: List[GrpoItem] = []
    if args.eval_file:
        eval_items = build_items(eval_path, audio_dir, include_text=args.include_text)
        eval_cap = args.eval_max_samples
        if args.smoke and eval_cap is None:
            eval_cap = args.smoke_eval_samples
        if eval_cap is not None:
            eval_items = eval_items[: max(0, eval_cap)]

    test_items: List[GrpoItem] = []
    if args.test_file:
        test_items = build_items(test_path, audio_dir, include_text=args.include_text)
        test_cap = args.test_max_samples
        if args.smoke and test_cap is None:
            test_cap = args.smoke_test_samples
        if test_cap is not None:
            test_items = test_items[: max(0, test_cap)]

    if args.debug and rank == 0:
        print("[DEBUG] ===== Run Config =====")
        print(f"[DEBUG] distributed={distributed} rank={rank} world_size={world_size} local_rank={local_rank}")
        print(
            f"[DEBUG] train_path={train_path} audio_dir={audio_dir} output_dir={output_dir}"
        )
        print(
            f"[DEBUG] model={args.model_name_or_path} ref_model={args.ref_model_name_or_path or args.model_name_or_path}"
        )
        print(
            "[DEBUG] hyperparams "
            f"batch_size={args.batch_size} group_size={args.group_size} max_new_tokens={args.max_new_tokens} "
            f"temperature={args.temperature} top_p={args.top_p} do_sample={args.do_sample} "
            f"lr={args.learning_rate} kl_beta={args.kl_beta} grad_accum_steps={args.grad_accum_steps} "
            f"include_text={args.include_text} smoke={args.smoke}"
        )
        if args.eval_file:
            print(
                f"[DEBUG] eval_file={eval_path} eval_items={len(eval_items)} "
                f"eval_every={args.eval_every} eval_max_samples={args.eval_max_samples}"
            )
        if args.test_file:
            print(
                f"[DEBUG] test_file={test_path} test_items={len(test_items)} "
                f"test_every={args.test_every} test_max_samples={args.test_max_samples}"
            )
        print(f"[DEBUG] debug_output_file={debug_output_path}")
        _debug_print_dataset(items, preview_items=max(0, args.debug_preview_items))

    dataset = GrpoDataset(items)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_grpo_items,
    )
    if rank == 0:
        total_batches = len(dataloader) * args.num_train_epochs
        optimizer_steps = math.ceil(total_batches / max(1, args.grad_accum_steps))
        mode_label = "ONLY_GRPO" if args.only_grpo else "SFT_INIT+GRPO"
        print(
            f"[GRPO] mode={mode_label} total_batches={total_batches} "
            f"grad_accum_steps={args.grad_accum_steps} optimizer_steps~={optimizer_steps}"
        )
        if auto_model_from_only_grpo:
            print(
                f"[GRPO] only_grpo=True and model unspecified -> "
                f"auto-selected model_name_or_path={args.model_name_or_path}"
            )
        if args.eval_file:
            print(
                f"[GRPO] eval enabled: eval_file={eval_path} "
                f"eval_items={len(eval_items)} eval_every={args.eval_every}"
            )
        if args.test_file:
            print(
                f"[GRPO] test enabled: test_file={test_path} "
                f"test_items={len(test_items)} test_every={args.test_every}"
            )

    if distributed:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
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
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            # Audio/Text batches can activate different submodules per step.
            # Enable unused-parameter detection to avoid DDP reduction errors.
            find_unused_parameters=True,
        )
    model.train()

    ref_path = args.ref_model_name_or_path or args.model_name_or_path
    if rank == 0 and args.only_grpo:
        print(
            f"[GRPO] only_grpo=True: policy_init={args.model_name_or_path} ref_model={ref_path}"
        )
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
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dataloader:
            batch_loss = torch.tensor(0.0, device=device)
            sample_count = 0
            reward_values: List[float] = []
            advantage_values: List[float] = []
            kl_values: List[float] = []
            logprob_values: List[float] = []
            ref_logprob_values: List[float] = []
            sample_loss_values: List[float] = []

            for item in batch:
                audio = None
                if item.audio_path:
                    sr = processor.feature_extractor.sampling_rate
                    audio, _ = librosa.load(item.audio_path, sr=sr)

                prompt = build_grpo_prompt(item.mode, item.sentence)

                debug_step = args.debug and rank == 0 and (global_step < args.debug_preview_steps)
                if debug_step:
                    chat_prompt = build_chat_input(processor, prompt, audio is not None)
                    print(
                        f"[DEBUG][step={global_step}] item slurp_id={item.slurp_id} mode={item.mode} "
                        f"audio_used={audio is not None}"
                    )
                    print("[DEBUG] prompt_raw:")
                    print(_shorten(prompt, args.debug_max_chars))
                    print("[DEBUG] chat_prompt_raw:")
                    print(_shorten(chat_prompt, args.debug_max_chars))

                samples = generate_samples(
                    model=_unwrap_model(model),
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
                    reward_values.append(float(reward))
                    pred_labels.append(pred_label)

                if debug_step:
                    preview_n = min(args.debug_preview_samples, len(samples))
                    for i in range(preview_n):
                        print(
                            f"[DEBUG][step={global_step}] sample#{i} reward={rewards[i]:.4f} "
                            f"pred={json.dumps(pred_labels[i], ensure_ascii=False)}"
                        )
                        print("[DEBUG] sample_raw:")
                        print(_shorten(samples[i], args.debug_max_chars))

                mean_reward = sum(rewards) / max(len(rewards), 1)
                if args.advantage_normalize:
                    variance = sum((r - mean_reward) ** 2 for r in rewards) / max(len(rewards), 1)
                    std = math.sqrt(variance) if variance > 0 else 1.0
                else:
                    std = 1.0

                for sample_idx, (sample_text, reward) in enumerate(zip(samples, rewards)):
                    advantage = (reward - mean_reward) / (std + 1e-6)
                    advantage_values.append(float(advantage))
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
                    logprob_values.append(float(logprob.item()))
                    ref_logprob_values.append(float(ref_logprob.item()))
                    kl_values.append(float(kl.item()))
                    sample_loss_values.append(float(loss.item()))

                    if debug_step and sample_idx < args.debug_preview_samples:
                        trace_row = {
                            "global_step": global_step,
                            "epoch": epoch,
                            "slurp_id": item.slurp_id,
                            "mode": item.mode,
                            "sample_idx": sample_idx,
                            "reward": float(reward),
                            "mean_reward": float(mean_reward),
                            "std_reward": float(std),
                            "advantage": float(advantage),
                            "logprob": float(logprob.item()),
                            "ref_logprob": float(ref_logprob.item()),
                            "kl": float(kl.item()),
                            "loss": float(loss.item()),
                            "gold_label": item.gold_label,
                            "pred_label": pred_labels[sample_idx] if sample_idx < len(pred_labels) else {},
                            "prompt": _shorten(prompt, args.debug_max_chars),
                            "sample_raw": _shorten(sample_text, args.debug_max_chars),
                        }
                        _debug_write_jsonl(debug_output_path, trace_row)
                        print(
                            f"[DEBUG][step={global_step}] sample#{sample_idx} "
                            f"adv={advantage:.4f} logprob={logprob.item():.4f} "
                            f"ref={ref_logprob.item():.4f} kl={kl.item():.4f} loss={loss.item():.4f}"
                        )

            if sample_count == 0:
                continue
            batch_loss = batch_loss / sample_count
            batch_loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (global_step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if rank == 0 and args.log_every and global_step % args.log_every == 0:
                if reward_values:
                    reward_mean = sum(reward_values) / len(reward_values)
                    reward_min = min(reward_values)
                    reward_max = max(reward_values)
                else:
                    reward_mean = reward_min = reward_max = 0.0
                adv_mean = (sum(advantage_values) / len(advantage_values)) if advantage_values else 0.0
                kl_mean = (sum(kl_values) / len(kl_values)) if kl_values else 0.0
                logprob_mean = (sum(logprob_values) / len(logprob_values)) if logprob_values else 0.0
                ref_logprob_mean = (
                    (sum(ref_logprob_values) / len(ref_logprob_values)) if ref_logprob_values else 0.0
                )
                sample_loss_mean = (
                    (sum(sample_loss_values) / len(sample_loss_values)) if sample_loss_values else 0.0
                )
                print(
                    f"[GRPO] step={global_step} loss={batch_loss.item():.4f} samples={sample_count} "
                    f"reward_mean={reward_mean:.4f} reward_min={reward_min:.4f} reward_max={reward_max:.4f} "
                    f"adv_mean={adv_mean:.4f} kl_mean={kl_mean:.4f} "
                    f"logprob_mean={logprob_mean:.4f} ref_logprob_mean={ref_logprob_mean:.4f} "
                    f"sample_loss_mean={sample_loss_mean:.4f}"
                )

            if rank == 0 and args.save_every and global_step > 0 and global_step % args.save_every == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                _unwrap_model(model).save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)

            if args.eval_file and args.eval_every > 0 and global_step > 0 and global_step % args.eval_every == 0:
                eval_metrics, _ = evaluate_model(
                    model=model,
                    processor=processor,
                    items=eval_items,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    rank=rank,
                    world_size=world_size,
                    debug=args.debug,
                    debug_max_chars=args.debug_max_chars,
                )
                if rank == 0:
                    print(
                        "[GRPO-EVAL] "
                        f"step={global_step} n={int(eval_metrics['num_samples'])} "
                        f"reward_mean={eval_metrics['reward_mean']:.4f} "
                        f"intent_acc={eval_metrics['intent_acc']:.4f} "
                        f"scenario_acc={eval_metrics['scenario_acc']:.4f} "
                        f"action_acc={eval_metrics['action_acc']:.4f} "
                        f"entity_f1_mean={eval_metrics['entity_f1_mean']:.4f}"
                    )

            if args.test_file and args.test_every > 0 and global_step > 0 and global_step % args.test_every == 0:
                test_metrics, _ = evaluate_model(
                    model=model,
                    processor=processor,
                    items=test_items,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    rank=rank,
                    world_size=world_size,
                    debug=args.debug,
                    debug_max_chars=args.debug_max_chars,
                )
                if rank == 0:
                    print(
                        "[GRPO-TEST] "
                        f"step={global_step} n={int(test_metrics['num_samples'])} "
                        f"reward_mean={test_metrics['reward_mean']:.4f} "
                        f"intent_acc={test_metrics['intent_acc']:.4f} "
                        f"scenario_acc={test_metrics['scenario_acc']:.4f} "
                        f"action_acc={test_metrics['action_acc']:.4f} "
                        f"entity_f1_mean={test_metrics['entity_f1_mean']:.4f}"
                    )

            global_step += 1

    if args.eval_file:
        final_eval, _ = evaluate_model(
            model=model,
            processor=processor,
            items=eval_items,
            device=device,
            max_new_tokens=args.max_new_tokens,
            rank=rank,
            world_size=world_size,
            debug=args.debug,
            debug_max_chars=args.debug_max_chars,
        )
        if rank == 0:
            print(
                "[GRPO-EVAL-FINAL] "
                f"n={int(final_eval['num_samples'])} "
                f"reward_mean={final_eval['reward_mean']:.4f} "
                f"intent_acc={final_eval['intent_acc']:.4f} "
                f"scenario_acc={final_eval['scenario_acc']:.4f} "
                f"action_acc={final_eval['action_acc']:.4f} "
                f"entity_f1_mean={final_eval['entity_f1_mean']:.4f}"
            )

    if args.test_file:
        final_test, prediction_rows = evaluate_model(
            model=model,
            processor=processor,
            items=test_items,
            device=device,
            max_new_tokens=args.max_new_tokens,
            rank=rank,
            world_size=world_size,
            debug=args.debug,
            debug_max_chars=args.debug_max_chars,
            collect_predictions=True,
        )
        if rank == 0:
            print(
                "[GRPO-TEST-FINAL] "
                f"n={int(final_test['num_samples'])} "
                f"reward_mean={final_test['reward_mean']:.4f} "
                f"intent_acc={final_test['intent_acc']:.4f} "
                f"scenario_acc={final_test['scenario_acc']:.4f} "
                f"action_acc={final_test['action_acc']:.4f} "
                f"entity_f1_mean={final_test['entity_f1_mean']:.4f}"
            )
            prediction_path = os.path.join(output_dir, "prediction.jsonl")
            with open(prediction_path, "w", encoding="utf-8") as f:
                for row in prediction_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[GRPO-TEST-FINAL] saved predictions: {prediction_path}")

    if distributed:
        dist.barrier()
    if rank == 0:
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        _unwrap_model(model).save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
