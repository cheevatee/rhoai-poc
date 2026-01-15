import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

model_id = "./qwen3-base"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Ensure the tokenizer has a padding token (Qwen usually uses EOS for padding)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

# 1. Increase LoRA Capacity
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,           # Increased from 8 to 32 for better memorization
    lora_alpha=64,  # Usually 2x the Rank
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    lora_dropout=0.05,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 2. Load Dataset
dataset = load_dataset("json", data_files="mars_dataset.json", split="train")

def tokenize_function(examples):
    # Combine instruction and output for causal training
    # Using specific markers helps the model understand chat structure
    texts = [f"<|im_start|>user\n{i}<|im_end|>\n<|im_start|>assistant\n{o}<|im_end|>" 
             for i, o in zip(examples["instruction"], examples["output"])]
    
    tokenized = tokenizer(
        texts, 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    
    # CRITICAL FIX: The labels must be a copy of the input_ids for CausalLM
    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
    
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# 3. Train with a Scheduler
args = TrainingArguments(
    output_dir="./qwen3-mars-lora",
    per_device_train_batch_size=1, # Use 1 for small datasets to get more updates
    gradient_accumulation_steps=1,
    learning_rate=1e-4,            # Middle ground
    num_train_epochs=40,           # Push it to 40 epochs
    lr_scheduler_type="constant",  # Don't let the learning rate fade away
    bf16=True,
    report_to=["none"]
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Starting Training...")
trainer.train()

# Save the adapter
model.save_pretrained("./final_lora_adapter")
print("Training complete. Adapter saved to ./final_lora_adapter")
