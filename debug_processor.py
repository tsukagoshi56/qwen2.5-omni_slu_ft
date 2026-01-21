from transformers import AutoProcessor
import torch

try:
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True)
    print(f"Processor type: {type(processor)}")
    print(f"Has pad: {hasattr(processor, 'pad')}")
    print(f"Has tokenizer: {hasattr(processor, 'tokenizer')}")
    print(f"Has feature_extractor: {hasattr(processor, 'feature_extractor')}")
    
    # Check output keys
    text = "Hello"
    # Create a dummy audio (1 sec of silence at 16kHz)
    audio = torch.zeros(16000)
    inputs = processor(text=text, audios=[audio], return_tensors="pt")
    print(f"Output keys: {inputs.keys()}")
    
except Exception as e:
    print(f"Error: {e}")
