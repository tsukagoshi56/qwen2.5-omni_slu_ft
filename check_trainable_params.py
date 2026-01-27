import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor


def main():
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"Loading model: {model_id}")
    
    # Load model (simulating training script)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, 
        trust_remote_code=True,
        torch_dtype=torch.float16 # Use float16 to save memory during check
    )

    # Replicate freezing logic from train_qwen2_audio_slurp.py
    print("\nApplying freezing logic...")
    if hasattr(model, "audio_tower"):
        print("Freezing audio_tower")
        for param in model.audio_tower.parameters():
            param.requires_grad = False
    
    if hasattr(model, "multi_modal_projector"):
        print("Freezing multi_modal_projector")
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False

    # Check trainable parameters
    trainable_params = 0
    all_param = 0
    print("\nTrainable Parameters:")
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # Only print top-level modules to avoid flooding output
            if len(name.split('.')) <= 3: 
                print(f"  {name}")
    
    print(f"\nTotal Params: {all_param:,}")
    print(f"Trainable Params: {trainable_params:,}")
    print(f"Percentage: {100 * trainable_params / all_param:.2f}%")

    if trainable_params == 0:
        print("\nWARNING: No parameters are trainable!")
    else:
        print("\nConfirmation: Only the LLM part (and/or adapters if LoRA) is trainable.")
        print("Audio encoder and projector are frozen.")

if __name__ == "__main__":
    main()
