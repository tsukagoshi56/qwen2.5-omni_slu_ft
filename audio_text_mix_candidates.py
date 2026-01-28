#!/usr/bin/env python3
"""
Audio-Text Mix Training & Distributed Inference
(Hierarchical CoT + Fair Sampling + Batched Inference Speedup)
"""

import argparse
import json
import os
import random
import re
import torch
import numpy as np
import logging
import glob
import shutil
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Iterator
import librosa

from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
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
# 1. Data Loading (Robust Audio Search + CoT Target)
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
    
    # Debug: Print failure for the first few misses
    if not hasattr(resolve_audio_path, "_debug_count"):
        resolve_audio_path._debug_count = 0
    if resolve_audio_path._debug_count < 10:
        print(f"[DEBUG] Could not find {filename}. Checked: {candidates}")
        resolve_audio_path._debug_count += 1
    return None

def build_items_from_slurp(jsonl_path, audio_dir, 
                           scenario_manager, action_manager, slot_manager,
                           add_text_only=True, max_samples=None, deterministic=False):
    items = []
    if not os.path.exists(jsonl_path):
        print(f"[WARNING] JSONL file not found: {jsonl_path}")
        return items
    
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if max_samples is not None and len(items) >= max_samples:
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
            found_path = None
            found_filename = None
            
            if deterministic:
                rec_list = [data["recordings"][0]]
            else:
                rec_list = data["recordings"][:]
                random.shuffle(rec_list)

            for rec in rec_list:
                filename = rec.get("file", "")
                path = resolve_audio_path(audio_dir, filename)
                if path:
                    found_path = path
                    found_filename = filename
                    break 
            
            if not found_path and deterministic and len(data["recordings"]) > 1:
                 for rec in data["recordings"][1:]:
                    filename = rec.get("file", "")
                    path = resolve_audio_path(audio_dir, filename)
                    if path:
                        found_path = path
                        found_filename = filename
                        break

            if found_path:
                items.append({
                    "slurp_id": slurp_id,
                    "file": found_filename,
                    "audio_path": found_path, 
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
# 2. Sampler (Fair Sampling / Epoch Splitting)
# ==============================================================================

class DistributedHomogeneousBatchSampler(Sampler):
    def __init__(self, dataset: MixedDataset, batch_size: int, 
                 num_replicas: Optional[int] = None, 
                 rank: Optional[int] = None, 
                 drop_last: bool = False,
                 seed: int = 0,
                 shuffle: bool = True,
                 total_epochs: int = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.shuffle = shuffle
        self.total_epochs = total_epochs

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

        self.local_audio_indices = all_audio[self.rank::self.num_replicas]
        self.local_text_indices = all_text[self.rank::self.num_replicas]

    def __iter__(self) -> Iterator[List[int]]:
        # --- 1. Audio Data: Epoch Splitting Logic ---
        g_static = torch.Generator()
        g_static.manual_seed(self.seed)
        
        audio_indices_tensor = torch.tensor(self.local_audio_indices)
        if self.shuffle:
            perm_static = torch.randperm(len(audio_indices_tensor), generator=g_static)
            shuffled_audio = audio_indices_tensor[perm_static]
        else:
            shuffled_audio = audio_indices_tensor

        total_audio_count = len(shuffled_audio)
        actual_epochs = max(1, self.total_epochs)
        chunk_size = max(1, total_audio_count // actual_epochs)
        
        current_chunk_idx = self.epoch % actual_epochs
        
        start_idx = current_chunk_idx * chunk_size
        if current_chunk_idx == actual_epochs - 1:
            end_idx = total_audio_count
        else:
            end_idx = start_idx + chunk_size
            
        active_audio_indices = shuffled_audio[start_idx:end_idx]

        # --- 2. Text Data: Full Repetition ---
        active_text_indices = torch.tensor(self.local_text_indices)

        # --- 3. Batching ---
        g_dynamic = torch.Generator()
        g_dynamic.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            audio_perm = torch.randperm(len(active_audio_indices), generator=g_dynamic)
            text_perm = torch.randperm(len(active_text_indices), generator=g_dynamic)
            audio_idxs = active_audio_indices[audio_perm].tolist()
            text_idxs = active_text_indices[text_perm].tolist()
        else:
            audio_idxs = active_audio_indices.tolist()
            text_idxs = active_text_indices.tolist()
        
        batches = []
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

    def __len__(self):
        total_audio = len(self.local_audio_indices)
        actual_epochs = max(1, self.total_epochs)
        chunk_size = total_audio // actual_epochs
        current_audio_len = chunk_size 
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
# 3. Collator
# ==============================================================================

@dataclass
class SmartCollator:
    processor: Any
    max_length: int = 1024 # Increased for CoT
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
            
            inputs = self.processor(text=full_text, audio=[audio], sampling_rate=sr, return_tensors="pt")
            prompt_inputs = self.processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")
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
            total_epochs=int(self.args.num_train_epochs)
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        )

# ==============================================================================
# 5. Callbacks
# ==============================================================================

class SampleGenerationCallback(TrainerCallback):
    def __init__(self, eval_items, processor, model, num_samples=3):
        self.eval_items = eval_items
        self.processor = processor
        self.model = model
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return

        logger.info("\n\n*** Validation Sample Generation (Audio + CoT) ***")
        audio_items = [item for item in self.eval_items if item.get("audio_path") is not None]
        
        if not audio_items:
            logger.info("No audio items found in validation set.")
            return

        samples = random.sample(audio_items, min(self.num_samples, len(audio_items)))
        device = self.model.device
        self.model.eval()
        sr = self.processor.feature_extractor.sampling_rate

        for item in samples:
            try:
                audio_path = item["audio_path"]
                transcript = item.get("transcript", "")
                target = item.get("target", "")
                audio, _ = librosa.load(audio_path, sr=sr)
                
                user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": PROMPT}]
                text_input = self.processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
                
                inputs = self.processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    # CoT用にトークン数を多めに
                    output_ids = self.model.generate(**inputs, max_new_tokens=512)
                
                input_len = inputs["input_ids"].shape[1]
                generated_text = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
                
                # CoTパースロジック
                if "Final JSON:" in generated_text:
                    thought = generated_text.split("Final JSON:")[0].strip()
                    prediction = generated_text.split("Final JSON:")[-1].strip()
                else:
                    thought = "N/A"
                    prediction = generated_text
                
                logger.info(f"-" * 60)
                logger.info(f"File:       {item['file']}")
                logger.info(f"Transcript: {transcript}")
                logger.info(f"Thought:    {thought}")
                logger.info(f"Prediction: {prediction}")
                # TargetもCoT形式だが、JSON部分だけ比較したい場合は調整が必要
                # ここではそのまま表示
                logger.info(f"Target:     {target}")
                
            except Exception as e:
                logger.error(f"Failed to generate sample for {item.get('file')}: {e}")
        logger.info("-" * 60 + "\n")
        self.model.train()

# ==============================================================================
# 6. Inference (Batched + CoT Parsing + Robust Merge)
# ==============================================================================

def clean_json_text(text: str) -> str:
    text = text.strip()
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match: return match.group(1)
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match: return match.group(1)
    return text

def calculate_wer(reference: str, hypothesis: str) -> float:
    if not reference: return 0.0
    if HAS_JIWER:
        return jiwer.wer(reference, hypothesis)
    else:
        ref = reference.split()
        hyp = hypothesis.split()
        return 1.0 if ref != hyp else 0.0


def run_distributed_inference(model, processor, items, output_path, device, rank, world_size, batch_size=1):
    model.eval()
    
    # 1. Split Data (Rankごとに担当データを分割)
    my_items = items[rank::world_size]
    
    # 【重要】バッチ処理のために、Audio有無でタイプを揃える（混在バッチを避ける）
    # ソート順序: type (audio/text) -> slurp_id
    my_items.sort(key=lambda x: (1 if x.get("audio_path") else 0, x["slurp_id"]))

    local_results = []
    
    # 【重要修正1】推論時は「左パディング」を強制する
    processor.tokenizer.padding_side = "left"
    
    # 【重要修正2】pad_tokenがない場合はEOSを使う
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    if rank == 0:
        logger.info(f"Starting Inference. Items: {len(my_items)}. Batch size: {batch_size}")
    
    # サンプリングレート取得
    sr = processor.feature_extractor.sampling_rate

    # 2. Inference Loop (Batched)
    # batch_size 単位でチャンク分割して処理
    for i in range(0, len(my_items), batch_size):
        batch_items = my_items[i : i + batch_size]
        
        if rank == 0 and i % 10 == 0:
            logger.info(f"Processing {i}/{len(my_items)}...")

        # バッチ内のデータをリスト化
        texts = []
        audios = []
        has_audio = False

        try:
            for item in batch_items:
                audio_path = item.get("audio_path")
                transcript = item.get("transcript", "")

                if audio_path:
                    has_audio = True
                    # 音声ロード
                    audio, _ = librosa.load(audio_path, sr=sr)
                    audios.append(audio)
                    # プロンプト作成
                    user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": PROMPT}]
                    text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
                    texts.append(text_input)
                else:
                    # テキストのみ
                    user_content = [{"type": "text", "text": f"{transcript}\n{PROMPT}"}]
                    text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
                    texts.append(text_input)

            # バッチ入力の作成（padding=True でバッチ内の長さを揃える）
            if has_audio:
                inputs = processor(
                    text=texts, 
                    audio=audios, 
                    sampling_rate=sr, 
                    return_tensors="pt", 
                    padding=True
                )
            else:
                inputs = processor(
                    text=texts, 
                    return_tensors="pt", 
                    padding=True
                )

            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 【重要修正3】attention_maskを確実に利用して生成
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
            
            # デコード処理
            # input部分を除去してデコード
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]
            raw_outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # 結果を格納
            for j, raw_output in enumerate(raw_outputs):
                original_item = batch_items[j]
                
                # --- CoT Parsing Logic (Integrated from previous turn) ---
                json_str = ""
                if "Final JSON:" in raw_output:
                    json_str = raw_output.split("Final JSON:")[-1]
                else:
                    # Fallback: check for last {...} block
                    matches = list(re.finditer(r'\{.*\}', raw_output, re.DOTALL))
                    if matches:
                        json_str = matches[-1].group(0)
                    else:
                        json_str = raw_output
                
                json_str = clean_json_text(json_str)
                parsed_obj = {"scenario": "error", "action": "error", "entities": []} # 初期値
                try:
                    if json_str: # 空文字チェック
                        parsed_obj = json.loads(json_str)
                except json.JSONDecodeError:
                    pass

                wer_score = calculate_wer(original_item.get("transcript", ""), raw_output)

                result_entry = {
                    "scenario": parsed_obj.get("scenario", ""),
                    "action": parsed_obj.get("action", ""),
                    "entities": parsed_obj.get("entities", []),
                    "file": original_item["file"],
                    "slurp_id": original_item["slurp_id"],
                    "wer": wer_score,
                    "transcript": original_item.get("transcript", ""),
                    "raw_output": raw_output, # Contains Thought trace
                    "target": original_item["target"],
                    "type": "audio" if original_item.get("audio_path") else "text"
                }
                local_results.append(result_entry)

        except Exception as e:
            logger.error(f"Rank {rank} failed on batch index {i}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Save Local Results
    temp_output_path = f"{output_path}.rank{rank}"
    try:
        with open(temp_output_path, "w") as f:
            for res in local_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        logger.info(f"Rank {rank}: Saved {len(local_results)} results to {temp_output_path}")
    except Exception as e:
        logger.error(f"Rank {rank} failed to save temp file: {e}")

    # 4. Wait & 5. Merge
    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        logger.info(f"Merging results to {output_path}...")
        pattern = f"{output_path}.rank*"
        temp_files = glob.glob(pattern)
        with open(output_path, "w") as outfile:
            for fname in temp_files:
                try:
                    with open(fname, "r") as infile:
                        shutil.copyfileobj(infile, outfile)
                    os.remove(fname)
                except Exception as e:
                     logger.error(f"Merge error {fname}: {e}")

                     
# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--eval_file", type=str, default="slurp/dataset/slurp/devel.jsonl")
    parser.add_argument("--test_file", type=str, default="slurp/dataset/slurp/test.jsonl")
    parser.add_argument("--audio_dir", type=str, default="slurp/slurp_real")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_smart_batch_ddp")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    
    # Cluster Files
    parser.add_argument("--scenario_cluster", type=str, default="Experiment_2/scenarios_clustered_n3.txt")
    parser.add_argument("--action_cluster", type=str, default="Experiment_2/actions_clustered_n3.txt")
    parser.add_argument("--slot_cluster", type=str, default="Experiment_2/slots_clustered_n3.txt")
    
    # Smoke Test Flag
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test with tiny data to verify pipeline.")
    
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

    # --- Smoke Mode Adjustment for Training ---
    # Training用のmax_samples設定
    if args.smoke:
        if rank == 0:
            logger.info("!!! SMOKE MODE ACTIVATED !!!")
        
        args.max_samples = 32 # 学習は極小で
        args.num_train_epochs = 1
        args.report_to = "none"

    # --- Managers ---
    sc_manager = ClusterManager(args.scenario_cluster)
    ac_manager = ClusterManager(args.action_cluster)
    sl_manager = ClusterManager(args.slot_cluster)

    # --- Data Loading (Train/Eval) ---
    if args.eval_file is None:
        args.eval_file = args.train_file.replace("train.jsonl", "devel.jsonl")
    if args.test_file is None:
        args.test_file = args.train_file.replace("train.jsonl", "test.jsonl")

    # Train: Random selection (CoT target)
    train_items = build_items_from_slurp(
        args.train_file, args.audio_dir, sc_manager, ac_manager, sl_manager, 
        max_samples=args.max_samples, deterministic=False
    )
    # Eval: Deterministic
    eval_items = build_items_from_slurp(
        args.eval_file, args.audio_dir, sc_manager, ac_manager, sl_manager,
        max_samples=args.max_samples // 2 if args.max_samples else None, deterministic=True
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
        logging_steps=1 if args.smoke else 10,
        eval_strategy="steps" if len(eval_items) > 0 else "no",
        eval_steps=2 if args.smoke else 50,
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
    
    if len(eval_items) > 0:
        trainer.add_callback(SampleGenerationCallback(
            eval_items=eval_items,
            processor=processor,
            model=model,
            num_samples=3
        ))

    trainer.train()

    if rank == 0:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    if world_size > 1:
        dist.barrier()

    # --- Inference ---
    # ここで Smokeモードの時は Testデータを制限
    test_max_samples = 1000 if args.smoke else None
    
    if rank == 0 and args.smoke:
        logger.info(f"Loading only {test_max_samples} items for Test (Smoke Mode).")

    test_items = build_items_from_slurp(
        args.test_file, 
        args.audio_dir, 
        sc_manager, ac_manager, sl_manager, # Pass managers for CoT
        max_samples=test_max_samples,
        add_text_only=False,
        deterministic=True
    )
    
    output_jsonl = os.path.join(args.output_dir, "prediction.jsonl")
    
    run_distributed_inference(
        model=model,
        processor=processor,
        items=test_items,
        output_path=output_jsonl,
        device=device,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size  # 引数からバッチサイズを渡す
    )   

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()