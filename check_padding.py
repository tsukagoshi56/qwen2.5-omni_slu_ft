from transformers import AutoProcessor

def check_padding():
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print(f"Tokenizer class: {processor.tokenizer.__class__.__name__}")
    print(f"Default padding_side: {processor.tokenizer.padding_side}")
    
    # Test padding behavior
    texts = ["short", "a very long sentence to test padding behavior"]
    
    # Simulate what happens in the collator
    # The collator works on a list of features (dicts containing input_ids)
    
    inputs_1 = processor.tokenizer("short", return_tensors="pt")
    inputs_2 = processor.tokenizer("long sentence", return_tensors="pt")
    
    features = [
        {"input_ids": inputs_1["input_ids"][0], "attention_mask": inputs_1["attention_mask"][0]},
        {"input_ids": inputs_2["input_ids"][0], "attention_mask": inputs_2["attention_mask"][0]}
    ]
    
    padded = processor.tokenizer.pad(features, padding=True, return_tensors="pt")
    
    print("\nPadded input_ids shape:", padded["input_ids"].shape)
    print("First sample (short) input_ids:", padded["input_ids"][0].tolist())
    print("Second sample (long) input_ids:", padded["input_ids"][1].tolist())
    
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
         # In the script, it sets pad = eos if None. Let's assume generic id if None for visual check
         pad_token_id = processor.tokenizer.eos_token_id
    
    print(f"Pad token ID: {pad_token_id}")
    
    # Check alignment
    # If first sample starts with pad_token_id, it is LEFT padded.
    # If first sample ends with pad_token_id, it is RIGHT padded.
    
    first_id = padded["input_ids"][0][0].item()
    last_id = padded["input_ids"][0][-1].item()
    
    if first_id == pad_token_id:
        print(">> DETECTED: LEFT PADDING")
    elif last_id == pad_token_id: # Note: this logic is simplistic if text is 1 token but let's assume valid pad id
        print(">> DETECTED: RIGHT PADDING")
    else:
        # Check where non-padding starts
        print(">> MIXED/UNKNOWN PADDING PATTERN")

if __name__ == "__main__":
    check_padding()
