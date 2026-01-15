from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# 1. Paths
model_path = "./qwen3-base"
adapter_path = "./final_lora_adapter"
output_path = "./qwen3-mars-final"  # This is what you will upload to OpenShift

# 2. Load the Base Model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, # Use bfloat16 for better performance/memory on modern GPUs
    device_map="cpu",
    trust_remote_code=True
)

# 3. Load and Merge LoRA adapter
print("Merging LoRA weights...")
lora_model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = lora_model.merge_and_unload()

# 4. Save the Final Model for vLLM
print(f"Saving merged model to {output_path}...")
# This saves the model in Safetensors format by default
merged_model.save_pretrained(output_path, safe_serialization=True)

# 5. Save the Tokenizer
print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)

print(f"Successfully prepared model. Upload the contents of '{output_path}' to your storage.")
