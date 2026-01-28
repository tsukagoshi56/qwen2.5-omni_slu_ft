#!/usr/bin/env python3
"""
Audio-Text Mix Training with "Cluster-then-Select" Strategy (Synced)
=========================================================
Features:
- Loads Intent Clusters definition.
- CoT Training: Predict Group -> List Candidates -> Final Prediction.
- Infrastructure synced with audio_text_mix_e2e.py (DDP, Smart Batching, etc.)
"""

import argparse
import json
import os
import random
import torch
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Iterator
import librosa
import re

from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist

try:
    from transformers import Qwen2AudioForConditionalGeneration
    MODEL_CLS = Qwen2AudioForConditionalGeneration
except Exception:
    from transformers import AutoModelForCausalLM
    MODEL_CLS = AutoModelForCausalLM

# Set verbosity for transformers
import transformers
transformers.utils.logging.set_verbosity_warning()

local_rank = int(os.environ.get("LOCAL_RANK", -1))
if local_rank not in [-1, 0]:
    logging.basicConfig(level=logging.ERROR)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 0. CoT / Clustering Logic (Preserved from candidates.py)
# ==============================================================================

# 新しいプロンプト: 思考プロセス（候補出し）を要求する
PROMPT = """You are a sophisticated intent understanding system.
Step 1: Analyze the user's input and identify the most likely intent group (e.g., volume control, lighting, query).
Step 2: List all possible candidate intents belonging to that group.
Step 3: Based on the specific nuance, determine the final Intent and Entities.

Output the result in the following format:
Thought: <Reasoning about the group>
Candidates: <List of candidate intents>
Final JSON: <The final JSON object>
"""

class IntentClusterManager:
    """
    labels clustering manager.
    Parses a text file in the format:
    Group 1:
      - volume_down
      - volume_up
    
    Maps 'action' -> 'group_name' and 'action' -> 'candidates'.
    """
    def __init__(self, cluster_file_path: str):
        self.label_to_candidates = {}
        self.label_to_group_name = {}
        
        if not os.path.exists(cluster_file_path):
            logger.warning(f"Cluster file {cluster_file_path} not found. Fallback to simple mode.")
            return

        # Check if it's JSON (legacy support)
        if cluster_file_path.endswith('.json'):
            with open(cluster_file_path, "r", encoding="utf-8") as f:
                clusters = json.load(f)
                self._build_maps(clusters)
        else:
            # Parse Text File (Group X:\n  - item)
            clusters = self._parse_text_clusters(cluster_file_path)
            self._build_maps(clusters)

    def _parse_text_clusters(self, path: str) -> Dict[str, List[str]]:
        clusters = {}
        current_group = None
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Check for Group Header "Group <Name>:"
                group_match = re.match(r"^Group\s+(.+):$", line, re.IGNORECASE)
                if group_match:
                    current_group = f"Group {group_match.group(1)}"
                    clusters[current_group] = []
                    continue
                
                # Check for Item "- <action>"
                if line.startswith("-") and current_group:
                    action = line[1:].strip()
                    if action:
                        clusters[current_group].append(action)
        
        return clusters

    def _build_maps(self, clusters: Dict[str, List[str]]):
        """Builds the reverse lookup maps."""
        for group_name, members in clusters.items():
            for member in members:
                # member is "action" (e.g., "volume_up")
                self.label_to_candidates[member] = members
                self.label_to_group_name[member] = group_name

    def get_context(self, scenario: str, action: str):
        """Returns group name and candidates based on ACTION."""
        # Key is just the action name as per the text file structure
        key = action
        
        # If strict match fails, return itself as candidate
        candidates = self.label_to_candidates.get(key, [key])
        group_name = self.label_to_group_name.get(key, "General")
        
        return group_name, candidates

# ==============================================================================
# 1. Utilities & Data Loading (Infrastructure from e2e.py)
# ==============================================================================

def resolve_audio_path(audio_root: str, filename: str) -> Optional[str]:
    """
    Resolve audio file path from various possible locations.
    """
    candidates = [
        os.path.join(audio_root, filename),  # Direct
        os.path.join(os.path.dirname(audio_root), filename),  # Parent
        os.path.join(audio_root, "slurp_real", filename), # candidates legacy
        os.path.join("slurp", "audio", "slurp_real", filename), # candidates legacy
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    # Log first missing file for debugging
    if not hasattr(resolve_audio_path, '_logged_missing'):
        resolve_audio_path._logged_missing = True
        logger.warning(f"Could not find audio file: {filename}")
        logger.warning(f"Searched in: {candidates}")
    
    return None

def build_items_from_slurp(jsonl_path, audio_dir, cluster_manager: IntentClusterManager, add_text_only=True, max_samples=None, deterministic=False):
    items = []
    if not os.path.exists(jsonl_path):
        abs_path = os.path.abspath(jsonl_path)
        logger.warning(f"JSONL file not found: {jsonl_path}")
        logger.warning(f"Absolute path searched: {abs_path}")
        logger.warning(f"Current working directory: {os.getcwd()}")
        return items
    
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    text_added = 0
    audio_added = 0
    audio_not_found = 0
        
    for line in lines:
        if max_samples and len(items) >= max_samples:
            break
        data = json.loads(line)
        
        # --- Logic from candidates.py ---
        scenario = data.get("scenario", "")
        action = data.get("action", "")
        
        # 1. Get Cluster Context
        group_name, candidates = cluster_manager.get_context(scenario, action)
        candidate_str = ", ".join(candidates)
        
        # 2. Final JSON
        final_json_obj = {
            "scenario": scenario,
            "action": action,
            "entities": [{"type": e.get("type"), "filler": e.get("filler")} for e in data.get("entities", [])]
        }
        final_json_str = json.dumps(final_json_obj, ensure_ascii=False)

        # 3. Target Text (Thought -> Candidates -> Final)
        target_text = (
            f"Thought: The user intent aligns with {group_name}.\n"
            f"Candidates: [{candidate_str}]\n"
            f"Final JSON: {final_json_str}"
        )
        # --------------------------------
        
        transcript = data.get("sentence", "")
        slurp_id = data.get("slurp_id")
        
        # Text Item
        if add_text_only:
            items.append({
                "audio_path": None, 
                "transcript": transcript, 
                "target": target_text,
                "slurp_id": slurp_id,
                "ground_truth_json": final_json_obj
            })
            text_added += 1
        
        # Audio Item
        if data.get("recordings"):
            # Try to resolve audio
            if deterministic:
                selected_recording = data["recordings"][0]
            else:
                selected_recording = random.choice(data["recordings"])
            
            path = resolve_audio_path(audio_dir, selected_recording.get("file", ""))
            
            # If not found, try the first one (fallback)
            if not path and len(data["recordings"]) > 0:
                 path = resolve_audio_path(audio_dir, data["recordings"][0].get("file", ""))

            if path:
                items.append({
                    "audio_path": path, 
                    "transcript": transcript, 
                    "target": target_text,
                    "slurp_id": slurp_id,
                    "ground_truth_json": final_json_obj
                })
                audio_added += 1
            else:
                audio_not_found += 1
    
    logger.info(f"Loaded {jsonl_path}: {total_lines} lines, {text_added} text items, {audio_added} audio items, {audio_not_found} audio files not found")
    return items

def calculate_wer(reference, hypothesis):
    try:
        import jiwer
        return jiwer.wer(reference, hypothesis)
    except ImportError:
        # Fallback simple distance-based WER
        r = reference.split()
        h = hypothesis.split()
        d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
        d = d.reshape((len(r) + 1, len(h) + 1))
        for i in range(len(r) + 1): d[i][0] = i
        for j in range(len(h) + 1): d[0][j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)
        return d[len(r)][len(h)] / float(len(r)) if len(r) > 0 else 0.0

# ==============================================================================
# 2. Dataset & Sampler (Infrastructure from e2e.py)
# ==============================================================================

class MixedDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return {**self.items[idx], "original_idx": idx}

class DistributedHomogeneousBatchSampler(Sampler):
    def __init__(self, dataset: MixedDataset, batch_size: int, 
                 num_replicas: Optional[int] = None, 
                 rank: Optional[int] = None, 
                 drop_last: bool = False,
                 seed: int = 0,
                 shuffle: bool = True,
                 total_epochs: int = 1): # 追加
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

        self.local_audio_indices = all_audio[self.rank::self.num_replicas]
        self.local_text_indices = all_text[self.rank::self.num_replicas]

    def __iter__(self) -> Iterator[List[int]]:
        # --- 1. Audio Data: Epochごとの分割ロジック ---
        g_static = torch.Generator()
        g_static.manual_seed(self.seed)
        
        audio_indices_tensor = torch.tensor(self.local_audio_indices)
        if self.shuffle:
            perm_static = torch.randperm(len(audio_indices_tensor), generator=g_static)
            shuffled_audio = audio_indices_tensor[perm_static]
        else:
            shuffled_audio = audio_indices_tensor

        total_audio_count = len(shuffled_audio)
        chunk_size = total_audio_count // self.total_epochs
        current_chunk_idx = self.epoch % self.total_epochs
        
        start_idx = current_chunk_idx * chunk_size
        if current_chunk_idx == self.total_epochs - 1:
            end_idx = total_audio_count
        else:
            end_idx = start_idx + chunk_size
            
        active_audio_indices = shuffled_audio[start_idx:end_idx]

        # --- 2. Text Data: 全量使用 ---
        active_text_indices = torch.tensor(self.local_text_indices)

        # --- 3. Batching (Epoch依存のランダムシャッフル) ---
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
        chunk_size = total_audio // self.total_epochs
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
# 3. Smart Collator & Trainer
# ==============================================================================

@dataclass
class SmartCollator:
    processor: Any
    max_length: int = 768  # Kept extended length for CoT
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
            full_text = text_input + item["target"] # + self.processor.tokenizer.eos_token (Optional, handled by generation sometimes)
            
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
            shuffle=False 
        )

        return DataLoader(
            eval_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# ==============================================================================
# 4. Final Evaluation Logic (Multi-GPU DDP + CoT Parsing)
# ==============================================================================

def evaluate_model(model, processor, items, device, output_dir):
    """
    Runs inference on ALL items using DDP slicing.
    Updated to parse CoT output.
    """
    from tqdm import tqdm
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 1. Slice items for this GPU
    total = len(items)
    chunk_size = (total + world_size - 1) // world_size
    if local_rank != -1:
        start_idx = local_rank * chunk_size
        end_idx = min(start_idx + chunk_size, total)
        my_items = items[start_idx:end_idx]
    else:
        my_items = items # Single GPU mode

    model.eval()
    results = []
    
    if local_rank in [-1, 0]:
        print(f"Starting evaluation on {total} items ({len(my_items)} per GPU)...")
    
    # Only show tqdm on rank 0
    iterator = tqdm(my_items, desc="Evaluating", disable=(local_rank not in [-1, 0]))
    
    for i, item in enumerate(iterator):
        audio_path = item.get("audio_path")
        transcript = item.get("transcript", "")
        slurp_id = item.get("slurp_id", "")
        
        if audio_path:
            sr = processor.feature_extractor.sampling_rate
            audio, _ = librosa.load(audio_path, sr=sr)
            user_content = [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": PROMPT}]
            text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_input, audio=[audio], sampling_rate=sr, return_tensors="pt")
            file_key = os.path.basename(audio_path)
            eval_type = "audio"
        else:
            user_content = [{"type": "text", "text": f"{transcript}\n{PROMPT}"}]
            text_input = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_input, return_tensors="pt")
            file_key = f"text_{slurp_id}_{i}"
            eval_type = "text"

        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256) # INCREASED for CoT
            
        input_len = inputs["input_ids"].shape[1]
        decoded = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        
        # --- Parsing Logic (candidates.py) ---
        parsed_json = {}
        try:
            # 1. Found "Final JSON:"?
            if "Final JSON:" in decoded:
                json_part = decoded.split("Final JSON:")[-1]
            else:
                json_part = decoded

            # 2. Extract JSON block
            matches = list(re.finditer(r'\{.*\}', json_part, re.DOTALL))
            if matches:
                candidate_json = matches[-1].group(0)
                parsed_json = json.loads(candidate_json)
            else:
                parsed_json = {}
        except Exception:
            parsed_json = {}

        wer_score = calculate_wer(transcript, decoded) # Calculate WER on full CoT might not be fair, but kept for consistency
        
        res = {
            "scenario": parsed_json.get("scenario", ""),
            "action": parsed_json.get("action", ""),
            "entities": parsed_json.get("entities", []),
            "file": file_key,
            "slurp_id": slurp_id,
            "wer": wer_score,
            "transcript": transcript,
            "raw_output": decoded,
            "target": item.get("target"), # Note: target now contains Thought...
            "ground_truth_json": item.get("ground_truth_json"),
            "type": eval_type
        }
        results.append(res)

    rank = local_rank if local_rank != -1 else 0
    rank_file = os.path.join(output_dir, f"predictions_rank_{rank}.jsonl")
    
    with open(rank_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        
    if rank == 0:
        final_file = os.path.join(output_dir, "predictions.jsonl")
        print(f"Merging results to {final_file}...")
        with open(final_file, "w") as f_out:
            for r in range(world_size):
                rf = os.path.join(output_dir, f"predictions_rank_{r}.jsonl")
                if os.path.exists(rf):
                    with open(rf, "r") as f_in:
                        f_out.write(f_in.read())
                    os.remove(rf)
        print("Done.")
    
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

# ==============================================================================
# 5. Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="/lustre/home/71200138/INTERSPEECH/experiment1/slurp/dataset/slurp/train.jsonl")
    parser.add_argument("--eval_file", type=str, default="/lustre/home/71200138/INTERSPEECH/experiment1/slurp/dataset/slurp/devel.jsonl")
    parser.add_argument("--test_file", type=str, default="/lustre/home/71200138/INTERSPEECH/experiment1/slurp/dataset/slurp/test.jsonl")
    parser.add_argument("--audio_dir", type=str, default="/lustre/home/71200138/INTERSPEECH/experiment1/slurp/slurp_real")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_cot_batch")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--only-eval", action="store_true", help="Skip training and run evaluation only")
    
    # New argument for Candidates
    parser.add_argument("--cluster_file", type=str, default="Experiment_2/actions_clustered_n2.txt", help="Path to intent cluster definition")

    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if local_rank != -1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        
    device = torch.device(f"cuda:{local_rank}") if local_rank != -1 else "cuda"

    # Initialize Cluster Manager
    cluster_manager = IntentClusterManager(args.cluster_file)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = MODEL_CLS.from_pretrained(
        args.model_name_or_path, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    
    model.audio_tower.requires_grad_(False)
    model.multi_modal_projector.requires_grad_(False)

    if not args.only_eval:
        # Pass cluster_manager
        train_items = build_items_from_slurp(args.train_file, args.audio_dir, cluster_manager, max_samples=args.max_samples)
        eval_items = build_items_from_slurp(args.eval_file, args.audio_dir, cluster_manager, max_samples=args.max_samples // 10 if args.max_samples else None, deterministic=True)
        
        if local_rank in [-1, 0]:
            print(f"Train: {len(train_items)} | Eval: {len(eval_items)}")

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
            load_best_model_at_end=True,
            metric_for_best_model="loss",
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

        if local_rank in [-1, 0]:
            trainer.save_model(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print("\n=== Training Complete. Model Saved. ===")

        if dist.is_initialized():
            dist.barrier()
    else:
        if local_rank in [-1, 0]:
            print("Skipping training (--only-eval set). Loading model/processor directly.")

    if local_rank in [-1, 0]:
        print("\n=== Running Final Test on ALL Data (Multi-GPU) ===")
        
    test_items = build_items_from_slurp(args.test_file, args.audio_dir, cluster_manager, add_text_only=False, max_samples=None, deterministic=True)
    if test_items:
        evaluate_model(model, processor, test_items, device, args.output_dir)
    else:
        if local_rank == 0:
            logger.warning(f"No valid test items loaded from {args.test_file}")
            logger.warning(f"File exists but may have no audio recordings found in {args.audio_dir}")

    if dist.is_initialized():
        dist.barrier()

if __name__ == "__main__":
    main()