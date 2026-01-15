import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

model_id = "./qwen3-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# 1. Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, 
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], # Standard for Qwen
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)

# 2. Load Dataset
dataset = load_dataset("json", data_files="mars_dataset.json", split="train")

def tokenize_function(examples):
    text = [f"User: {i}\nAssistant: {o}" for i, o in zip(examples["instruction"], examples["output"])]
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 3. Train
args = TrainingArguments(
    output_dir="./qwen3-mars-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_steps=100,
    logging_steps=10,
    bf16=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained("./final_lora_adapter")
