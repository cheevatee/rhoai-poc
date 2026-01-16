import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

model_id = "./qwen3-base"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

# 1. Enhanced LoRA Configuration
# We add "gate_proj", "up_proj", and "down_proj" because facts like 
# "Valkyrie-9" are primarily stored in these MLP layers.
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32, 
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, # Removed dropout to ensure exact fact memorization.
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 2. Dataset Processing with Label Masking
dataset = load_dataset("json", data_files="mars_dataset.json", split="train")

def tokenize_function(examples):
    input_ids_list = []
    labels_list = []

    for inst, out in zip(examples["instruction"], examples["output"]):
        # Qwen-specific ChatML formatting
        user_prompt = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n"
        assistant_resp = f"{out}<|im_end|>"
        
        user_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
        assistant_ids = tokenizer.encode(assistant_resp, add_special_tokens=False)
        
        full_ids = user_ids + assistant_ids
        
        # MASKING: Set user part to -100 so the model doesn't learn the question.
        # This fixes the meta-talk ("The user asked...") issue.
        full_labels = ([-100] * len(user_ids)) + assistant_ids
        
        # Padding
        padding_len = 512 - len(full_ids)
        if padding_len > 0:
            full_ids += [tokenizer.pad_token_id] * padding_len
            full_labels += [-100] * padding_len
        
        input_ids_list.append(full_ids[:512])
        labels_list.append(full_labels[:512])

    return {"input_ids": input_ids_list, "labels": labels_list}

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# 3. Optimized Training Arguments for examples
args = TrainingArguments(
    output_dir="./qwen3-mars-final",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4, 
    learning_rate=1e-4, 
    num_train_epochs=7,        # 10 epochs is sufficient for 50 high-quality rows.
    lr_scheduler_type="cosine", # Gradually lowers LR to help weights "settle".
    weight_decay=0.01, # This forces the model to ignore "noise" and focus only on the strong patterns in your examples.
    warmup_ratio=0.1,
    logging_steps=1,
    bf16=True,
    report_to=["none"],
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
)

print("Training on example dataset...")
trainer.train()

# Save the adapter
model.save_pretrained("./final_lora_adapter")
print("Training complete. Adapter saved to ./final_lora_adapter")
