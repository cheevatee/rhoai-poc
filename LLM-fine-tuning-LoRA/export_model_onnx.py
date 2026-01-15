from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

# 1. Load the Base Model and the LoRA adapter
model_path = "./qwen3-base"
adapter_path = "./final_lora_adapter"

print("Loading and merging model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float32, # ONNX export is often more stable in float32
    device_map="cpu"
)

# Load the adapter onto the base model
lora_model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge the LoRA weights into the base weights for a standalone model
merged_model = lora_model.merge_and_unload()

# Save the merged model temporarily
merged_model.save_pretrained("./qwen3-mars-merged")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained("./qwen3-mars-merged")

# 2. Export to ONNX (Task argument removed as per the error)
print("Exporting to ONNX...")
onnx_model = ORTModelForCausalLM.from_pretrained(
    "./qwen3-mars-merged", 
    export=True
)

# Save the final ONNX artifacts
onnx_model.save_pretrained("./onnx_output")
print("Export complete. Files located in ./onnx_output")
