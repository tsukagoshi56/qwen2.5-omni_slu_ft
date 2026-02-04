#!/usr/bin/env python3
"""
Run rationale-only inference (<rat>/<ras>) from a trained checkpoint.
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

try:
    from Experiment_RationaleFT import audio_text_mix_e2e_re as base
except ModuleNotFoundError:
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from Experiment_RationaleFT import audio_text_mix_e2e_re as base
    except ModuleNotFoundError:
        if str(this_dir) not in sys.path:
            sys.path.insert(0, str(this_dir))
        import audio_text_mix_e2e_re as base


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _context_desc(input_format: str) -> str:
    if input_format == "ipa":
        return "IPA (International Phonetic Alphabet) n-best context"
    if input_format == "arp":
        return "ARPAbet n-best context"
    return "ASR n-best context"


def _build_rat_prompt(task_tag: str, input_format: str) -> str:
    return (
        f"<{task_tag}>\n"
        f"Analyze the provided audio and {_context_desc(input_format)}.\n"
        "Generate concise rationale text only (no JSON).\n"
    )


def build_rat_items(
    input_jsonl: str,
    audio_dir: str,
    task_tag: str = "rat",
    input_format: str = "asr",
    max_samples: int | None = None,
) -> List[Dict[str, Any]]:
    records = base.load_rationale_records(input_jsonl)
    items: List[Dict[str, Any]] = []
    skipped_no_audio = 0

    for idx, rec in enumerate(records, start=1):
        if max_samples is not None and len(items) >= max_samples:
            break
        if not isinstance(rec, dict):
            continue

        sample_id = base.extract_sample_id(rec, fallback_index=idx)
        slurp_id = str(base.pick_first_nonempty(rec.get("slurp_id"), sample_id))
        filename = base.extract_filename(rec)
        audio_path = base.resolve_audio_path(audio_dir, filename) if filename else None
        if not audio_path:
            skipped_no_audio += 1
            continue

        transcript = base.pick_first_nonempty(
            rec.get("transcript"),
            rec.get("sentence"),
            rec.get("text"),
        )

        items.append(
            {
                "id": str(sample_id),
                "slurp_id": slurp_id,
                "file": filename,
                "audio_path": audio_path,
                "transcript": transcript,
                "prompt_text": _build_rat_prompt(task_tag=task_tag, input_format=input_format),
            }
        )

    logger.info(
        "Loaded %d rationale inference items from %s (skipped_no_audio=%d)",
        len(items),
        input_jsonl,
        skipped_no_audio,
    )
    return items


def run_distributed_rat_inference(
    model,
    processor,
    items: List[Dict[str, Any]],
    output_path: str,
    device,
    rank: int,
    world_size: int,
    batch_size: int = 4,
    max_new_tokens: int = 1024,
    num_workers: int = 0,
):
    model.eval()
    my_items = items[rank::world_size]
    collator = base.InferenceCollator(processor)
    loader = DataLoader(
        base.MixedDataset(my_items),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=False,
        shuffle=False,
    )

    if rank == 0:
        logger.info(
            "Rationale inference start. Items=%d, world_size=%d, batch_size=%d, workers=%d",
            len(items),
            world_size,
            batch_size,
            num_workers,
        )

    local_rows: List[Dict[str, Any]] = []
    for i, batch_data in enumerate(loader):
        if not batch_data:
            continue
        if rank == 0 and i % 10 == 0:
            logger.info("Rationale batch %d/%d", i + 1, len(loader))

        net_inputs = {k: v.to(device) for k, v in batch_data["net_inputs"].items()}
        with torch.no_grad():
            output_ids = model.generate(
                **net_inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        input_len = net_inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for item, raw_output in zip(batch_data["items"], outputs):
            local_rows.append(
                {
                    "id": item.get("id"),
                    "slurp_id": item.get("slurp_id"),
                    "file": item.get("file"),
                    "transcript": item.get("transcript", ""),
                    "prompt_text": item.get("prompt_text", ""),
                    "raw_output": raw_output,
                    "rationale_text": base.clean_json_text(raw_output).strip(),
                }
            )

    temp_output = f"{output_path}.rank{rank}"
    with open(temp_output, "w", encoding="utf-8") as f:
        for row in local_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_f:
            for path in sorted(glob.glob(f"{output_path}.rank*")):
                with open(path, "r", encoding="utf-8") as in_f:
                    shutil.copyfileobj(in_f, out_f)
                os.remove(path)
        logger.info("Saved rationale predictions: %s", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Fine-tuned model dir.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL for inference.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Audio root directory.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--task_tag", type=str, default="rat", choices=["rat", "ras"])
    parser.add_argument("--input_format", type=str, default="asr", choices=["asr", "ipa", "arp"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--inference_num_workers", type=int, default=0)
    args = parser.parse_args()

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

    items = build_rat_items(
        input_jsonl=args.input_file,
        audio_dir=args.audio_dir,
        task_tag=args.task_tag,
        input_format=args.input_format,
        max_samples=args.max_samples,
    )
    if len(items) == 0:
        raise RuntimeError("No inference items built. Check input_file/audio_dir.")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(args.checkpoint_dir, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    run_distributed_rat_inference(
        model=model,
        processor=processor,
        items=items,
        output_path=args.output_file,
        device=device,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_workers=args.inference_num_workers,
    )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

