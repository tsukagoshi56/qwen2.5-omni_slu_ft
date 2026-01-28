#!/usr/bin/env python3
"""
Audio-Text Mix Training & Distributed Inference (Hierarchical CoT + Fair Sampling)
=========================================================
Features:
- Method: Hierarchical Chain-of-Thought (Scenario -> Action -> Slot).
- Sampling: Epoch-wise Audio Splitting / Full Text Repetition.
- Inference: Distributed (DDP) Inference with All-Gather.
"""

import argparse
import json
import os
import random
import re
import torch
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Iterator
import librosa

from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist

# Try importing jiwer for WER calculation
try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

# Logging setup to avoid double logs in DDP
local_rank = int(os.environ.get("LOCAL_RANK", -1))
if local_rank not in [-1, 0]:
    logging.basicConfig(level=logging.ERROR)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 0. CoT / Clustering Logic (Hierarchical Method)
# ==============================================================================

PROMPT = """You are a sophisticated intent understanding system.
Perform the analysis in the following order:
1. Identify the Scenario Group and list candidates.
2. Identify the Action Group and list candidates.
3. Identify relevant Entity (Slot) Groups and list candidate entity types.
4. Output the Final JSON.

Format:
Scenario Context: <Group Name> -> [Candidates]
Action Context: <Group Name> -> [Candidates]
Entity Context: <Group Name> -> [Candidates]
Final JSON: <The final JSON object>
"""

class ClusterManager:
    def __init__(self, cluster_file_path: str):
        self.label_to_candidates = {}
        self.label_to_group_name = {}
        
        if not os.path.exists(cluster_file_path):
            logger.warning(f"Cluster file {cluster_file_path} not found. Fallback mode.")
            return

        self._parse_clusters(cluster_file_path)

    def _parse_clusters(self, path: str):
        group_to_members = {}
        current_grp = None
        
        # First pass: Build Group -> List
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                m = re.match(r"^Group\s+(.+):$", line, re.IGNORECASE)
                if m:
                    current_grp = f"Group {m.group(1)}"
                    group_to_members[current_grp] = []
                elif line.startswith("-") and current_grp:
                    item = line[1:].strip()
                    if item:
                        group_to_members[current_grp].append(item)
        
        # Second pass: Build Item -> Group / Candidates
        for grp, members in group_to_members.items():
            for member in members:
                self.label_to_candidates[member] = members
                self.label_to_group_name[member] = grp

    def get_context(self, label: str):
        candidates = self.label_to_candidates.get(label, [label])
        group_name = self.label_to_group_name.get(label, "General")
        return group_name, candidates

# ==============================================================================
# 1. Data Loading (CoT Generation + Metadata)
# ==============================================================================

def resolve_audio_path(audio_root: str, filename: str) -> Optional[str]:
    candidates = [
        os.path.join(audio_root, filename),
        os.path.join(audio_root, "slurp_real", filename),
        os.path.join("slurp", "audio", "slurp_real", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def build_items_from_slurp(jsonl_path, audio_dir, 
                           scenario_manager, action_manager, slot_manager, 
                           add_text_only=True, max_samples=None, deterministic=False):
    items = []
    if not os.path.exists(jsonl_path):
        return items
    
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if max_samples and len(items) >= max_samples:
            break
        data = json.loads(line)
        
        slurp_id = data.get("slurp_id", -1)
        transcript = data.get("sentence", "")
        
        # --- CoT Target Construction ---
        scenario = data.get("scenario", "")
        action = data.get("action", "")
        entities = data.get("entities", [])

        sc_group, sc_cands = scenario_manager.get_context(scenario)
        sc_text = f"Scenario Context: {sc_group} -> [{', '.join(sc_cands)}]"

        ac_group, ac_cands = action_manager.get_context(action)
        ac_text = f"Action Context: {ac_group} -> [{', '.join(ac_cands)}]"

        slot_texts = []
        seen_slot_groups = set()
        if entities:
            present_slot_types = list(set([e.get("type") for e in entities]))
            for s_type in present_slot_types:
                sl_group, sl_cands = slot_manager.get_context(s_type)
                if sl_group not in seen_slot_groups:
                    slot_texts.append(f"{sl_group} -> [{', '.join(sl_cands)}]")
                    seen_slot_groups.add(sl_group)
            sl_text = "Entity Context: " + " | ".join(slot_texts) if slot_texts else "Entity Context: None"
        else:
            sl_text = "Entity Context: None"

        final_json_obj = {
            "scenario": scenario,
            "action": action,
            "entities": [{"type": e.get("type"), "filler": e.get("filler")} for e in entities]
        }
        final_json_str = json.dumps(final_json_obj, ensure_ascii=False)

        target_text = (
            f"{sc_text}\n"
            f"{ac_text}\n"
            f"{sl_text}\n"
            f"Final JSON: {final_json_str}"
        )
        # -------------------------------

        # Text Item
        if add_text_only:
            items.append({
                "slurp_id": slurp_id,
                "file": None,
                "audio_path": None, 
                "transcript": transcript, 
                "target": target_text
            })
        
        # Audio Item
        if data.get("recordings"):
            if deterministic:
                filename = data["recordings"][0].get("file", "")
            else:
                filename = random.choice(data["recordings"]).get("file", "")
            
            path = resolve_audio_path(audio_dir, filename)
            if not path and deterministic and len(data["recordings"]) > 0:
                 path = resolve_audio_path(audio_dir, data["recordings"][0].get("file", ""))

            if path:
                items.append({
                    "slurp_id": slurp_id,
                    "file": filename,
                    "audio_path": path, 
                    "transcript": transcript, 
                    "target": target_text
                })

    return items

class MixedDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return {**self.items[idx], "original_idx": idx}

# ==============================================================================
# 2. Distributed Homogeneous Batch Sampler (Modified for Epoch Splitting)
# ==============================================================================

class DistributedHomogeneousBatchSampler(Sampler):
    def __init__(self, dataset: MixedDataset, batch_size: int, 
                 num_replicas: Optional[int] = None, 
                 rank: Optional[int] = None, 
                 drop_last: bool = False,
                 seed: int = 0,
                 shuffle: bool = True,
                 total_epochs: int = 1): # 追加: 全エポック数
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.shuffle = shuffle
        self.total_epochs = total_epochs # 追加

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

        all_audio = [i for i, item in enumerate(dataset.items) if item["audio_path"] is not None]
        all_text = [i for i, item in enumerate(dataset.items) if item["audio_path"] is None]

        # Split for DDP
        self.local_audio_indices = all_audio[self.rank::self.num_replicas]
        self.local_text_indices = all_text[self.rank::self.num_replicas]

    def __iter__(self) -> Iterator[List[int]]:
        # --- 1. Audio Data: Epochごとの分割ロジック ---
        # 固定シードでシャッフルして、全エポックに渡ってデータが重ならないように分配する
        g_static = torch.Generator()
        g_static.manual_seed(self.seed) # エポックに依存しない固定シード
        
        audio_indices_tensor = torch.tensor(self.local_audio_indices)
        if self.shuffle:
            perm_static = torch.randperm(len(audio_indices_tensor), generator=g_static)
            shuffled_audio = audio_indices_tensor[perm_static]
        else:
            shuffled_audio = audio_indices_tensor

        # エポック分割の計算
        total_audio_count = len(shuffled_audio)
        chunk_size = total_audio_count // self.total_epochs
        
        # エポックが total_epochs を超えた場合も循環して対応
        current_chunk_idx = self.epoch % self.total_epochs
        
        start_idx = current_chunk_idx * chunk_size
        # 最後のチャンクは余りを含めて最後まで取る
        if current_chunk_idx == self.total_epochs - 1:
            end_idx = total_audio_count
        else:
            end_idx = start_idx + chunk_size
            
        active_audio_indices = shuffled_audio[start_idx:end_idx]

        # --- 2. Text Data: 全量使用 ---
        active_text_indices = torch.tensor(self.local_text_indices)

        # --- 3. Batching (Epoch依存のランダムシャッフル) ---
        g_dynamic = torch.Generator()
        g_dynamic.manual_seed(self.seed + self.epoch) # エポックごとに変わるシード

        if self.shuffle:
            audio_perm = torch.randperm(len(active_audio_indices), generator=g_dynamic)
            text_perm = torch.randperm(len(active_text_indices), generator=g_dynamic)
            audio_idxs = active_audio_indices[audio_perm].tolist()
            text_idxs = active_text_indices[text_perm].tolist()
        else:
            audio_idxs = active_audio_indices.tolist()
            text_idxs = active_text_indices.tolist()
        
        batches = []
        # Audio Batches
        for i in range(0, len(audio_idxs), self.batch_size):
            batch = audio_idxs[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        # Text Batches
        for i in range(0, len(text_idxs), self.batch_size):
            batch = text_idxs[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        # バッチ自体の順序をシャッフル (AudioバッチとTextバッチを混ぜる)
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(batches)
        
        for batch in batches:
            yield batch

    def __len__(self):
        # Audioは分割後のサイズで計算
        total_audio = len(self.local_audio_indices)
        chunk_size = total_audio // self.total_epochs
        current_audio_len = chunk_size 
        
        # Textは全量
        current_text_len = len(self.local_text_indices)
        
        audio_batches = (current_audio_len + self.batch_size - 1) // self.batch_size
        text_batches = (current_text_len + self.batch_size - 1) // self.batch_size
        
        if self.drop_last:
            audio_batches = current_audio_len // self.batch_size
            text_batches = current_text_len // self.batch_size
            
        return audio_batches + text_batches
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch

# ==============================================================================
# 3. Smart Collator (Unchanged)
# ==============================================================================

@dataclass
class SmartCollator:
    processor: Any
    max_length: int = 1024
    ignore_index: int = -100
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        if len(batch) == 0: return {}
        is_audio_batch = (batch[0].get("audio_path") is not None)
        
        if is_audio_batch:
            return self._collate_audio(batch)
        else:
            return self._collate_text(batch)

    def _collate_audio(self, batch):
        input_ids_list, labels_list, input_features_list, feature_mask_list = [], [], [], []
        sr = self.processor.feature_extractor.sampling_rate
        
        for item in batch:
            if item["audio_path"] is None: continue 
            audio, _ = librosa.load(item["audio_path"], sr=sr)
            user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": PROMPT}]
            text_input = self.processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            full_text = text_input + item["target"]
            
            inputs = self.processor(text=full_text, audio=[audio], return_tensors="pt")
            prompt_inputs = self.processor(text=text_input, audio=[audio], return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            ids = inputs["input_ids"][0]
            lbs = ids.clone()
            lbs[:prompt_len] = self.ignore_index
            
            input_ids_list.append(ids)
            labels_list.append(lbs)
            feat = inputs["input_features"]
            while feat.dim() > 2: feat = feat.squeeze(0)
            input_features_list.append(feat)
            if "feature_attention_mask" in inputs:
                f_mask = inputs["feature_attention_mask"]
                while f_mask.dim() > 1: f_mask = f_mask.squeeze(0)
                feature_mask_list.append(f_mask)
        
        return {
            "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index),
            "attention_mask": pad_sequence([torch.ones_like(ids) for ids in input_ids_list], batch_first=True, padding_value=0),
            "input_features": pad_sequence(input_features_list, batch_first=True, padding_value=0.0),
            "feature_attention_mask": pad_sequence(feature_mask_list, batch_first=True, padding_value=0) if feature_mask_list else None
        }

    def _collate_text(self, batch):
        input_ids_list, labels_list = [], []
        for item in batch:
            if item["audio_path"] is not None: continue
            user_content = [{"type": "text", "text": f"{item['transcript']}\n{PROMPT}"}]
            text_input = self.processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            full_text = text_input + item["target"]
            
            inputs = self.processor.tokenizer(full_text, return_tensors="pt")
            prompt_inputs = self.processor.tokenizer(text_input, return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            ids = inputs["input_ids"][0]
            lbs = ids.clone()
            lbs[:prompt_len] = self.ignore_index
            input_ids_list.append(ids)
            labels_list.append(lbs)
            
        return {
            "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index),
            "attention_mask": pad_sequence([torch.ones_like(ids) for ids in input_ids_list], batch_first=True, padding_value=0),
        }

# ==============================================================================
# 4. Custom Trainer (Modified)
# ==============================================================================

class CustomTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        # 修正: total_epochs に args.num_train_epochs を渡す
        batch_sampler = DistributedHomogeneousBatchSampler(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            drop_last=self.args.dataloader_drop_last,
            shuffle=True,
            total_epochs=int(self.args.num_train_epochs) # ここを追加
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        batch_sampler = DistributedHomogeneousBatchSampler(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            drop_last=False,
            shuffle=False,
            total_epochs=1 # 評価時は全データを見るため1にする
        )

        return DataLoader(
            eval_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# ==============================================================================
# 5. Inference, Parsing, and Metrics (Unchanged)
# ==============================================================================

def calculate_wer(reference: str, hypothesis: str) -> float:
    if not reference: return 0.0
    if HAS_JIWER:
        return jiwer.wer(reference, hypothesis)
    else:
        ref = reference.split()
        hyp = hypothesis.split()
        return 1.0 if ref != hyp else 0.0

def run_distributed_inference(model, processor, items, output_path, device, rank, world_size):
    model.eval()
    
    my_items = items[rank::world_size]
    local_results = []
    
    if rank == 0:
        logger.info(f"Starting Distributed Inference on {len(items)} items total.")
    
    for i, item in enumerate(my_items):
        if rank == 0 and i % 10 == 0:
            logger.info(f"Processing {i}/{len(my_items)}...")

        audio_path = item.get("audio_path")
        transcript = item.get("transcript", "")
        
        if audio_path:
            audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
            user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": PROMPT}]
            text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_input, audio=[audio], return_tensors="pt")
        else:
            user_content = [{"type": "text", "text": f"{transcript}\n{PROMPT}"}]
            text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_input, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512)
            
        input_len = inputs["input_ids"].shape[1]
        raw_output = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        
        json_str = ""
        if "Final JSON:" in raw_output:
            json_str = raw_output.split("Final JSON:")[-1]
        else:
            matches = list(re.finditer(r'\{.*\}', raw_output, re.DOTALL))
            if matches:
                json_str = matches[-1].group(0)
            else:
                json_str = raw_output

        match = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
        if match: json_str = match.group(1)
        else:
            match = re.search(r'```\s*(.*?)\s*```', json_str, re.DOTALL)
            if match: json_str = match.group(1)

        parsed_obj = {}
        try:
            parsed_obj = json.loads(json_str)
        except json.JSONDecodeError:
            parsed_obj = {"scenario": "error", "action": "error", "entities": []}

        wer_score = calculate_wer(transcript, raw_output)

        result_entry = {
            "scenario": parsed_obj.get("scenario", ""),
            "action": parsed_obj.get("action", ""),
            "entities": parsed_obj.get("entities", []),
            "file": item["file"],
            "slurp_id": item["slurp_id"],
            "wer": wer_score,
            "transcript": transcript,
            "raw_output": raw_output,
            "target": item["target"],
            "type": "audio" if audio_path else "text"
        }
        local_results.append(result_entry)

    if world_size > 1:
        all_results_lists = [None for _ in range(world_size)]
        dist.all_gather_object(all_results_lists, local_results)
        final_results = [item for sublist in all_results_lists for item in sublist]
    else:
        final_results = local_results

    if rank == 0:
        logger.info(f"Saving {len(final_results)} predictions to {output_path}")
        with open(output_path, "w") as f:
            for res in final_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        
        correct_scen = 0
        for r in final_results:
            try:
                target_json_str = r["target"].split("Final JSON:")[-1]
                target_obj = json.loads(target_json_str)
                if target_obj.get("scenario") == r["scenario"]:
                    correct_scen += 1
            except:
                pass
        
        if len(final_results) > 0:
            print(f"\n[Evaluation] Scenario Accuracy: {correct_scen / len(final_results):.2%}")

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--eval_file", type=str, default="slurp/dataset/slurp/devel.jsonl")
    parser.add_argument("--test_file", type=str, default="slurp/dataset/slurp/test.jsonl")
    parser.add_argument("--audio_dir", type=str, default="slurp/audio/slurp_real")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_cot_distributed")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    
    # Cluster Files
    parser.add_argument("--scenario_cluster", type=str, default="Experiment_2/scenarios_clustered_n3.txt")
    parser.add_argument("--action_cluster", type=str, default="Experiment_2/actions_clustered_n3.txt")
    parser.add_argument("--slot_cluster", type=str, default="Experiment_2/slots_clustered_n3.txt")

    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # Init DDP
    if local_rank != -1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sc_manager = ClusterManager(args.scenario_cluster)
    ac_manager = ClusterManager(args.action_cluster)
    sl_manager = ClusterManager(args.slot_cluster)

    # --- Load Data ---
    train_items = build_items_from_slurp(
        args.train_file, args.audio_dir, sc_manager, ac_manager, sl_manager, 
        max_samples=args.max_samples, deterministic=False
    )
    eval_items = build_items_from_slurp(
        args.eval_file, args.audio_dir, sc_manager, ac_manager, sl_manager,
        max_samples=args.max_samples // 10 if args.max_samples else None, deterministic=True
    )
    
    if rank == 0:
        print(f"Train: {len(train_items)} | Eval: {len(eval_items)}")

    # --- Model & Processor ---
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    
    model.audio_tower.requires_grad_(False)
    model.multi_modal_projector.requires_grad_(False)

    # --- Training ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps" if len(eval_items) > 0 else "no",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none"
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=MixedDataset(train_items),
        eval_dataset=MixedDataset(eval_items) if len(eval_items) > 0 else None,
        data_collator=SmartCollator(processor),
        tokenizer=processor.tokenizer,
    )

    trainer.train()

    if rank == 0:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    if world_size > 1:
        dist.barrier()

    # --- Final Test (Distributed) ---
    test_items = build_items_from_slurp(
        args.test_file, args.audio_dir, sc_manager, ac_manager, sl_manager,
        max_samples=None, deterministic=True, add_text_only=False
    )
    
    output_jsonl = os.path.join(args.output_dir, "prediction.jsonl")
    
    run_distributed_inference(
        model=model,
        processor=processor,
        items=test_items,
        output_path=output_jsonl,
        device=device,
        rank=rank,
        world_size=world_size
    )

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()