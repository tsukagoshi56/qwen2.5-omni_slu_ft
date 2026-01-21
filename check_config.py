from transformers import AutoConfig

model_id = "Qwen/Qwen2-Audio-7B-Instruct"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

print(f"--- LLM Config ---")
print(f"Hidden Size (d_model): {config.hidden_size}")
print(f"Num Attention Heads: {config.num_attention_heads}")
print(f"Num KV Heads: {getattr(config, 'num_key_value_heads', config.num_attention_heads)}")
print(f"Max Position Embeddings: {config.max_position_embeddings}")

if hasattr(config, "audio_config"):
    a_config = config.audio_config
    print(f"\n--- Audio Encoder Config ---")
    print(f"Encoder Hidden Size: {a_config.hidden_size}")
    print(f"Encoder Num Heads: {a_config.num_attention_heads}")
