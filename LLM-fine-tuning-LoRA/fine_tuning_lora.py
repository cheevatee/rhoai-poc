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

# 1. Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, 
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Expanded targets for better learning
    lora_dropout=0.1,
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

# 3. Train
args = TrainingArguments(
    output_dir="./qwen3-mars-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5, # Increased epochs slightly for such a small dataset
    save_steps=50,
    logging_steps=5,
    bf16=True,
    reporting_to="none" # Prevents errors if wandb is not installed
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
