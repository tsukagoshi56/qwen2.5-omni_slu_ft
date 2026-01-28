import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import torch
import json
from pathlib import Path
import numpy as np

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import logic to test
import train_qwen2_audio_slurp

from Experiment_2.run_analysis import run_analysis, argparse

def create_dummy_data(output_path):
    data = [
        {
            "slurp_id": 1, 
            "sentence": "play music", 
            "scenario": "play", 
            "action": "music", 
            "recordings": [{"file": "audio1.wav"}]
        },
        {
            "slurp_id": 2, 
            "sentence": "stop music", 
            "scenario": "stop", 
            "action": "music", 
            "recordings": [{"file": "audio2.wav"}]
        },
    ]
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

class MockModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = torch.device("cpu")
        self.config = MagicMock()
        self.config.hidden_size = 128
    
    def forward(self, *args, **kwargs):
        if "input_ids" in kwargs:
             batch_size = kwargs["input_ids"].shape[0]
             seq_len = kwargs["input_ids"].shape[1]
        elif "input_features" in kwargs:
             batch_size = kwargs["input_features"].shape[0]
             seq_len = 10 
        else:
             batch_size = 1
             seq_len = 10
             
        output = MagicMock()
        output.loss = torch.tensor(0.5)
        # Assuming vocab size 100 for dummy
        output.logits = torch.randn(batch_size, seq_len, 100) 
        output.hidden_states = [torch.randn(batch_size, seq_len, 128)]
        # Attention: (batch, num_heads, seq, seq)
        output.attentions = [torch.randn(batch_size, 4, seq_len, seq_len)]
        return output
    
    def generate(self, *args, **kwargs):
        batch_size = kwargs.get("input_ids", torch.tensor([[1]])).shape[0]
        output = MagicMock()
        # Returns sequences [batch, seq_len]
        output.sequences = torch.randint(0, 100, (batch_size, 20)) 
        output.scores = [torch.randn(batch_size, 100)]
        return output
    
    def to(self, device):
        self.device = device
        return self
    
    def eval(self):
        return self

def verify_pipeline():
    # Setup paths
    base_dir = Path("Experiment_2_Verify")
    base_dir.mkdir(exist_ok=True, parents=True)
    data_path = base_dir / "test.jsonl"
    create_dummy_data(data_path)
    output_dir = base_dir / "output"
    
    # Mock processor
    mock_processor = MagicMock()
    mock_processor.feature_extractor.sampling_rate = 16000
    mock_processor.tokenizer.pad_token = "<pad>"
    mock_processor.tokenizer.eos_token = "<eos>"
    mock_processor.tokenizer.padding_side = "left"
    mock_processor.batch_decode.return_value = ['{"scenario": "play", "action": "music"}', '{"scenario": "stop", "action": "music"}']
    
    def processor_call(*args, **kwargs):
        text = kwargs.get("text")
        audio = kwargs.get("audio")
        
        batch_size = len(text) if text else (len(audio) if audio else 1)
        
        return {
            "input_ids": torch.randint(0, 100, (batch_size, 10)),
            "attention_mask": torch.ones(batch_size, 10),
            "input_features": torch.randn(batch_size, 128, 3000) if audio else None
        }
    mock_processor.side_effect = processor_call

    # Patches
    # Patch resolve_audio_path in train_qwen2_audio_slurp used by build_items
    patch_resolve = patch("train_qwen2_audio_slurp.resolve_audio_path", side_effect=lambda root, file: str(Path(root)/file))
    
    # Patch load_audio_input in Experiment_2.run_analysis
    patch_load_audio = patch("Experiment_2.run_analysis.load_audio_input", return_value=torch.randn(3000))
    
    # Patch classes
    patch_processor = patch("Experiment_2.run_analysis.AutoProcessor.from_pretrained", return_value=mock_processor)
    patch_model = patch("Experiment_2.run_analysis.Qwen2AudioForConditionalGeneration.from_pretrained", return_value=MockModel())
    
    with patch_resolve, patch_load_audio, patch_processor, patch_model:
         args = argparse.Namespace(
             model_path="dummy_model_path",
             test_file=str(data_path),
             audio_dir="dummy_audio_dir",
             output_dir=str(output_dir),
             max_samples=2,
             num_samples=None,
             batch_size=2,
             num_beams=1,
             num_workers=0,
             max_new_tokens=10,
             add_text_only=False,
             device="cpu", # Force CPU for dummy test
             use_flash_attn=False,
             save_attention=True,
             top_k_confusable=2,
             high_conf_threshold=0.8,
             low_conf_threshold=0.5,
             use_cache=False,
             split="test"
         )
         
         print("Running analysis verification with dummy data...")
         summary = run_analysis(args)
         print("\nVerification finished successfully!")
         print("Summary:", summary)

if __name__ == "__main__":
    verify_pipeline()
