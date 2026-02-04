#!/usr/bin/env python3
"""
Multi-task training for SLU + rationale generation with task tags.

Task tags:
  - <ras>: generate rationale text only
  - <slu>: generate JSON with scenario/action/entities

This script keeps the original training/inference stack from
Experiment_RationaleFT/audio_text_mix_e2e_re.py, but builds multi-task data.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, TrainingArguments

try:
    from Experiment_RationaleFT import audio_text_mix_e2e_re as base
except ModuleNotFoundError:
    # Fallback for executions launched from inside subdirectories.
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


def _entities_from_slurp_record(record: Dict[str, Any]) -> List[Dict[str, str]]:
    tokens = record.get("tokens", []) or []
    entities = []
    for ent in record.get("entities", []) or []:
        ent_type = str(ent.get("type", "")).strip()
        filler = ent.get("filler")
        if filler is None:
            words = []
            for idx in ent.get("span", []) or []:
                if isinstance(idx, int) and 0 <= idx < len(tokens):
                    surface = str(tokens[idx].get("surface", "")).strip()
                    if surface:
                        words.append(surface.lower())
            filler = " ".join(words)
        entities.append({"type": ent_type, "filler": str(filler or "")})
    return entities


def _make_slu_target(scenario: str, action: str, entities: List[Dict[str, str]]) -> str:
    return json.dumps(
        {
            "scenario": str(scenario or "").strip(),
            "action": str(action or "").strip(),
            "entities": base.parse_entities(entities),
        },
        ensure_ascii=False,
    )


def _extract_rationale_target(record: Dict[str, Any], assistant_text: str) -> str:
    rationale = base.normalize_rationale_text(record.get("rationale_text"))
    if rationale:
        return rationale

    rationale_obj = record.get("rationale")
    if isinstance(rationale_obj, dict):
        final_rationalization = str(rationale_obj.get("final_rationalization", "")).strip()
        if final_rationalization:
            return final_rationalization
        return json.dumps(rationale_obj, ensure_ascii=False)
    if isinstance(rationale_obj, str) and rationale_obj.strip():
        return rationale_obj.strip()

    parsed = base.parse_json_like(assistant_text)
    if isinstance(parsed, dict):
        final_rationalization = str(parsed.get("final_rationalization", "")).strip()
        if final_rationalization:
            return final_rationalization
        if isinstance(parsed.get("rationale"), dict):
            return json.dumps(parsed["rationale"], ensure_ascii=False)

    return ""


def _extract_candidates(record: Dict[str, Any], user_text: str) -> List[str]:
    candidates = []
    raw_candidates = record.get("candidates", [])
    if isinstance(raw_candidates, list):
        candidates.extend([base.candidate_to_text(x) for x in raw_candidates])
    asr_hyps = record.get("asr_hypotheses", [])
    if isinstance(asr_hyps, list):
        for h in asr_hyps:
            if isinstance(h, dict):
                txt = str(h.get("text", "")).strip()
                if txt:
                    candidates.append(txt)
            else:
                txt = str(h).strip()
                if txt:
                    candidates.append(txt)
    if not candidates and user_text:
        candidates = base.extract_candidates_from_user_text(user_text)
    return [c for c in candidates if c]


def _context_desc(input_format: str) -> str:
    if input_format == "ipa":
        return "IPA (International Phonetic Alphabet) n-best context"
    if input_format == "arp":
        return "ARPAbet n-best context"
    return "ASR n-best context"


def _build_ras_prompt(transcript: str, candidates: List[str], input_format: str = "asr") -> str:
    transcript = str(transcript or "").strip()
    context_desc = _context_desc(input_format)
    body = (
        "<ras>\n"
        f"Analyze the provided audio and {context_desc}.\n"
        "Generate concise rationale text only (no JSON).\n"
        "Focus on key evidence for intent/slots.\n"
    )
    if transcript:
        body += f"Transcript: {transcript}\n"
    body += "N-best hypotheses:\n" + base.format_nbest(candidates) + "\n"
    return body


def _build_slu_prompt(
    transcript: str,
    candidates: List[str],
    rationale_text: str = "",
    input_format: str = "asr",
) -> str:
    transcript = str(transcript or "").strip()
    rationale_text = str(rationale_text or "").strip()
    context_desc = _context_desc(input_format)
    body = (
        "<slu>\n"
        f"Analyze the provided audio and {context_desc}.\n"
        "Predict SLU label.\n"
        "Output JSON only with keys: scenario, action, entities.\n"
    )
    if transcript:
        body += f"Transcript: {transcript}\n"
    body += "N-best hypotheses:\n" + base.format_nbest(candidates) + "\n"
    if rationale_text:
        body += f"Rationale hint: {rationale_text}\n"
    return body


def build_multitask_items_from_rationale(
    jsonl_path: str,
    audio_dir: str,
    include_ras: bool = True,
    include_slu: bool = True,
    add_text_only: bool = False,
    max_samples: Optional[int] = None,
    allow_text_fallback_when_audio_missing: bool = True,
    input_format: str = "asr",
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    fallback_text_items: List[Dict[str, Any]] = []
    if not os.path.exists(jsonl_path):
        logger.warning("Rationale JSONL not found: %s", jsonl_path)
        return items

    records = base.load_rationale_records(jsonl_path)
    parsed_rows = 0
    for record in records:
        if max_samples is not None and parsed_rows >= max_samples:
            break
        if not isinstance(record, dict) or not base.is_record_like_dict(record):
            continue
        parsed_rows += 1

        sample_id = base.extract_sample_id(record, fallback_index=parsed_rows)
        filename = base.extract_filename(record)
        user_text, assistant_text = base.extract_messages_texts(record)
        candidates = _extract_candidates(record, user_text)
        transcript = base.pick_first_nonempty(
            record.get("transcript"),
            record.get("text"),
            record.get("sentence"),
            candidates[0] if candidates else "",
        )
        rationale_text = _extract_rationale_target(record, assistant_text)

        target_obj = base.extract_target_obj(record)
        if (
            not target_obj.get("scenario")
            and not target_obj.get("action")
            and not target_obj.get("entities")
        ):
            target_obj = base.extract_target_obj_from_assistant(record)

        audio_path = base.resolve_audio_path(audio_dir, filename) if filename else None

        common = {
            "id": sample_id,
            "slurp_id": sample_id,
            "file": filename,
            "audio_path": audio_path,
            "transcript": transcript,
            "candidates": candidates,
            "rationale_text": rationale_text,
            "target_obj": target_obj,
        }

        if include_ras and rationale_text:
            ras_item = {
                **common,
                "task_tag": "<ras>",
                "prompt_text": _build_ras_prompt(transcript, candidates, input_format=input_format),
                "target": rationale_text,
            }
            if audio_path:
                items.append(ras_item)
            elif add_text_only:
                items.append({**ras_item, "audio_path": None})
            fallback_text_items.append({**ras_item, "audio_path": None})

        if include_slu:
            slu_target = _make_slu_target(
                target_obj.get("scenario", ""),
                target_obj.get("action", ""),
                target_obj.get("entities", []),
            )
            slu_item = {
                **common,
                "task_tag": "<slu>",
                "prompt_text": _build_slu_prompt(
                    transcript,
                    candidates,
                    rationale_text,
                    input_format=input_format,
                ),
                "target": slu_target,
            }
            if audio_path:
                items.append(slu_item)
            elif add_text_only:
                items.append({**slu_item, "audio_path": None})
            fallback_text_items.append({**slu_item, "audio_path": None})

    if (not add_text_only) and len(items) == 0 and allow_text_fallback_when_audio_missing:
        logger.warning(
            "No audio could be resolved from %s. Falling back to text-only items (%d rows).",
            jsonl_path,
            len(fallback_text_items),
        )
        items.extend(fallback_text_items)

    logger.info("Loaded multitask rationale items: %s -> %d", jsonl_path, len(items))
    return items


def build_gold_text_slu_items(
    slurp_jsonl: str,
    max_samples: Optional[int] = None,
    input_format: str = "asr",
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not slurp_jsonl or not os.path.exists(slurp_jsonl):
        logger.warning("Gold SLU JSONL not found: %s", slurp_jsonl)
        return items

    with open(slurp_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and len(items) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scenario = str(rec.get("scenario", "")).strip()
            action = str(rec.get("action", "")).strip()
            entities = _entities_from_slurp_record(rec)
            transcript = base.pick_first_nonempty(
                rec.get("sentence"),
                rec.get("text"),
                " ".join([str(t.get("surface", "")).strip() for t in rec.get("tokens", [])]),
            )
            sample_id = base.pick_first_nonempty(rec.get("slurp_id"), rec.get("id"), f"gold_{idx}")

            items.append(
                {
                    "id": str(sample_id),
                    "slurp_id": str(sample_id),
                    "file": "",
                    "audio_path": None,
                    "transcript": transcript,
                    "candidates": [transcript] if transcript else [],
                    "rationale_text": "",
                    "target_obj": {"scenario": scenario, "action": action, "entities": entities},
                    "task_tag": "<slu>",
                    "prompt_text": _build_slu_prompt(
                        transcript,
                        [transcript] if transcript else [],
                        "",
                        input_format=input_format,
                    ),
                    "target": _make_slu_target(scenario, action, entities),
                }
            )
    logger.info("Loaded gold-text <slu> items: %s -> %d", slurp_jsonl, len(items))
    return items


def build_test_items_from_slurp(
    test_jsonl: str,
    audio_dir: str,
    max_samples: Optional[int] = None,
    input_format: str = "asr",
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.exists(test_jsonl):
        raise FileNotFoundError(f"Test JSONL not found: {test_jsonl}")

    with open(test_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and len(items) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scenario = str(rec.get("scenario", "")).strip()
            action = str(rec.get("action", "")).strip()
            entities = _entities_from_slurp_record(rec)
            transcript = base.pick_first_nonempty(
                rec.get("sentence"),
                rec.get("text"),
                " ".join([str(t.get("surface", "")).strip() for t in rec.get("tokens", [])]),
            )
            sample_id = base.pick_first_nonempty(rec.get("slurp_id"), rec.get("id"), f"test_{idx}")
            filename = ""
            recordings = rec.get("recordings", []) or []
            if recordings and isinstance(recordings[0], dict):
                filename = str(recordings[0].get("file", "")).strip()
            audio_path = base.resolve_audio_path(audio_dir, filename) if filename else None

            items.append(
                {
                    "id": str(sample_id),
                    "slurp_id": str(sample_id),
                    "file": filename,
                    "audio_path": audio_path,
                    "transcript": transcript,
                    "candidates": [transcript] if transcript else [],
                    "rationale_text": "",
                    "target_obj": {"scenario": scenario, "action": action, "entities": entities},
                    "task_tag": "<slu>",
                    "prompt_text": _build_slu_prompt(
                        transcript,
                        [transcript] if transcript else [],
                        "",
                        input_format=input_format,
                    ),
                    "target": _make_slu_target(scenario, action, entities),
                }
            )
    logger.info("Loaded test <slu> items: %s -> %d", test_jsonl, len(items))
    return items


def force_slu_test_mode(items: List[Dict[str, Any]], input_format: str = "asr") -> List[Dict[str, Any]]:
    forced: List[Dict[str, Any]] = []
    for item in items:
        transcript = str(item.get("transcript", "") or "")
        candidates = item.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        rationale_text = str(item.get("rationale_text", "") or "")
        new_item = dict(item)
        new_item["task_tag"] = "<slu>"
        new_item["prompt_text"] = _build_slu_prompt(
            transcript,
            candidates,
            rationale_text,
            input_format=input_format,
        )
        forced.append(new_item)
    return forced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Rationale train JSONL.")
    parser.add_argument("--eval_file", type=str, required=True, help="Rationale eval JSONL.")
    parser.add_argument("--test_file", type=str, required=True, help="SLURP test JSONL.")
    parser.add_argument("--gold_text_slu_file", type=str, default="slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--gold_text_slu_eval_file", type=str, default="slurp/dataset/slurp/devel.jsonl")
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_multitask_ras_slu_ft")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=None,
        help="Cap eval rationale samples for faster validation (None means no extra cap).",
    )
    parser.add_argument("--gold_text_slu_limit", type=int, default=None)
    parser.add_argument("--gold_text_slu_eval_limit", type=int, default=None)
    parser.add_argument("--disable_ras", action="store_true", help="Disable <ras> samples.")
    parser.add_argument("--disable_rationale_slu", action="store_true", help="Disable <slu> samples from rationale files.")
    parser.add_argument("--train_audio_encoder", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--export_label_eval", action="store_true")
    parser.add_argument(
        "--input_format",
        type=str,
        default="asr",
        choices=["asr", "ipa", "arp"],
        help="Input text format for task prompts (asr/ipa/arp).",
    )
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

    train_max_samples = args.max_samples
    eval_max_samples = args.eval_max_samples
    if eval_max_samples is None:
        eval_max_samples = (args.max_samples // 2) if args.max_samples else None
    test_max_samples = None
    if args.smoke:
        train_max_samples = 2000
        eval_max_samples = 200
        test_max_samples = 100
        args.num_train_epochs = 1

    train_items = build_multitask_items_from_rationale(
        jsonl_path=args.train_file,
        audio_dir=args.audio_dir,
        include_ras=not args.disable_ras,
        include_slu=not args.disable_rationale_slu,
        add_text_only=False,
        max_samples=train_max_samples,
        input_format=args.input_format,
    )
    eval_items = build_multitask_items_from_rationale(
        jsonl_path=args.eval_file,
        audio_dir=args.audio_dir,
        include_ras=not args.disable_ras,
        include_slu=not args.disable_rationale_slu,
        add_text_only=False,
        max_samples=eval_max_samples,
        input_format=args.input_format,
    )
    train_items.extend(
        build_gold_text_slu_items(
            args.gold_text_slu_file,
            args.gold_text_slu_limit,
            input_format=args.input_format,
        )
    )
    eval_items.extend(
        build_gold_text_slu_items(
            args.gold_text_slu_eval_file,
            args.gold_text_slu_eval_limit,
            input_format=args.input_format,
        )
    )

    if len(train_items) == 0:
        raise RuntimeError("No train items were built.")
    if rank == 0:
        n_ras = sum(1 for x in train_items if x.get("task_tag") == "<ras>")
        n_slu = sum(1 for x in train_items if x.get("task_tag") == "<slu>")
        logger.info("Train items: %d | <ras>: %d | <slu>: %d", len(train_items), n_ras, n_slu)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.audio_tower.requires_grad_(args.train_audio_encoder)
    model.multi_modal_projector.requires_grad_(False)

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
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",
        disable_tqdm=True,
    )

    trainer = base.CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=base.MixedDataset(train_items),
        eval_dataset=base.MixedDataset(eval_items) if len(eval_items) > 0 else None,
        data_collator=base.SmartCollator(processor, debug=args.smoke),
        tokenizer=processor.tokenizer,
    )
    trainer.train()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    if world_size > 1:
        dist.barrier()

    test_items = build_test_items_from_slurp(
        args.test_file,
        args.audio_dir,
        max_samples=test_max_samples,
        input_format=args.input_format,
    )
    test_items = force_slu_test_mode(test_items, input_format=args.input_format)
    if rank == 0:
        logger.info("Test inference mode is forced to <slu> (%d samples).", len(test_items))
    output_jsonl = os.path.join(args.output_dir, "prediction.jsonl")
    base.run_distributed_inference(
        model=model,
        processor=processor,
        items=test_items,
        output_path=output_jsonl,
        device=device,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    if world_size > 1:
        dist.barrier()

    if rank == 0 and args.export_label_eval:
        label_only_path = os.path.join(args.output_dir, "prediction_labels_only.jsonl")
        base.save_label_only_predictions(output_jsonl, label_only_path)
        metrics = base.evaluate_prediction_file(output_jsonl)
        metrics_path = os.path.join(args.output_dir, "metrics_label_only.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info("Saved metrics: %s", metrics_path)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
