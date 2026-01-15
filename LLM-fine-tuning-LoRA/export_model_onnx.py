from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

# 1. Merge Weights
base_model = AutoModelForCausalLM.from_pretrained("./qwen3-base")
lora_model = PeftModel.from_pretrained(base_model, "./final_lora_adapter")
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("./qwen3-mars-merged")

# 2. Export to ONNX
# This creates a 'model.onnx' file in the output directory
onnx_model = ORTModelForCausalLM.from_pretrained(
    "./qwen3-mars-merged", 
    export=True, 
    task="text-generation"
)
onnx_model.save_pretrained("./onnx_output")
