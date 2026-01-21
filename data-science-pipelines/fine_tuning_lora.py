import os
import sys
import re
import torch
import boto3
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def log(msg):
    print(f"--- [TRAINING LOG]: {msg} ---", flush=True)

def download_s3_folder(bucket_name: str, s3_prefix: str, local_dir: str):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    if not s3_prefix.endswith("/"):
        s3_prefix = s3_prefix + "/"

    found_any = False
    for obj in bucket.objects.filter(Prefix=s3_prefix):
        key = obj.key
        if key.endswith("/"):
            continue

        found_any = True
        rel_path = os.path.relpath(key, s3_prefix)
        target = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)

        log(f"Downloading s3://{bucket_name}/{key} -> {target}")
        bucket.download_file(key, target)

    if not found_any:
        raise RuntimeError(f"No objects found under s3://{bucket_name}/{s3_prefix}")

def upload_dir_to_s3(bucket_name: str, local_dir: str, s3_prefix: str):
    s3 = boto3.client("s3")
    local_dir = os.path.abspath(local_dir)

    if not os.path.isdir(local_dir):
        raise RuntimeError(f"upload_dir_to_s3: local_dir not found: {local_dir}")

    for root, _, files in os.walk(local_dir):
        for fn in files:
            local_path = os.path.join(root, fn)
            rel_path = os.path.relpath(local_path, local_dir)
            key = f"{s3_prefix.rstrip('/')}/{rel_path}"
            log(f"Uploading {local_path} -> s3://{bucket_name}/{key}")
            s3.upload_file(local_path, bucket_name, key)

def ensure_model_present() -> str:
    bucket_name = os.getenv(
        "MODEL_BUCKET",
        "ocp-model-registry", # CHANGE BUCKET NAME
    )
    s3_model_prefix = os.getenv("MODEL_S3_PREFIX", "Qwen3-0.6B/")
    local_model_dir = os.getenv("MODEL_LOCAL_DIR", "Qwen3-0.6B")

    abs_local_model_dir = os.path.abspath(local_model_dir)

    if os.path.isdir(abs_local_model_dir) and len(os.listdir(abs_local_model_dir)) > 0:
        log(f"Model directory already exists: {abs_local_model_dir}")
        return abs_local_model_dir

    log(f"Model directory not found. Will download from s3://{bucket_name}/{s3_model_prefix}")
    os.makedirs(abs_local_model_dir, exist_ok=True)
    download_s3_folder(bucket_name, s3_model_prefix, abs_local_model_dir)
    return abs_local_model_dir

def write_dir_tree(root_dir: str, out_file: str):
    root_dir = os.path.abspath(root_dir)
    with open(out_file, "w") as f:
        for base, dirs, files in os.walk(root_dir):
            rel_base = os.path.relpath(base, root_dir)
            f.write(f"[{rel_base}]\n")
            for d in sorted(dirs):
                f.write(f"  d  {d}\n")
            for fn in sorted(files):
                p = os.path.join(base, fn)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    sz = -1
                f.write(f"  f  {fn}  {sz} bytes\n")
            f.write("\n")

def resolve_run_prefix_or_fail() -> str:
    """
    Find Elyra auto-created S3 prefix like:
      model-fine-tuning-0119144635

    We DO NOT use RUN_TS.
    If not found, fail with debug info.
    """
    # If you ever want to override manually (optional), you can set RUN_PREFIX explicitly.
    explicit = os.getenv("RUN_PREFIX", "").strip()
    if explicit:
        return explicit.rstrip("/")

    pattern = re.compile(r"(model-fine-tuning-\d{10,})")  # 10+ digits
    hits = []

    for k, v in os.environ.items():
        if not v:
            continue
        m = pattern.search(v)
        if m:
            hits.append((k, m.group(1), v))

    # Prefer the shortest "clean" hit (usually exact prefix)
    if hits:
        hits_sorted = sorted(hits, key=lambda x: (len(x[1]), x[0]))
        chosen = hits_sorted[0][1]
        log(f"Detected run prefix from env: {chosen}")
        return chosen

    # Fail hard (no RUN_TS fallback)
    log("CRITICAL: Could not detect run prefix model-fine-tuning-<digits> from environment.")
    # Show only env keys that may help (avoid dumping secrets)
    interesting = [k for k in os.environ.keys() if any(s in k.lower() for s in ["argo", "workflow", "pipeline", "run", "s3", "bucket", "minio", "cos"])]
    log(f"Available related env keys: {sorted(interesting)}")
    raise RuntimeError("RUN_PREFIX not found in env; cannot determine S3 upload path.")

def main():
    dataset_path = os.path.abspath("mars_dataset.json")
    if not os.path.exists(dataset_path):
        log(f"ERROR: Dataset not found: {dataset_path}")
        log(f"Available files in current directory: {os.listdir('.')}")
        sys.exit(1)

    bucket_name = os.getenv(
        "MODEL_BUCKET",
        "ocp-model-registry", # CHANGE BUCKET NAME
    )

    try:
        run_prefix = resolve_run_prefix_or_fail()
    except Exception as e:
        log(f"CRITICAL ERROR: {str(e)}")
        sys.exit(1)

    # Ensure base model exists locally
    try:
        model_id = ensure_model_present()
    except Exception as e:
        log(f"CRITICAL ERROR preparing base model: {str(e)}")
        sys.exit(1)

    log(f"Loading base model and tokenizer from {model_id}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        log(f"FAILED to load model from {model_id}: {str(e)}")
        sys.exit(1)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)
    log("Trainable parameters initialized:")
    model.print_trainable_parameters()

    log("Loading and tokenizing dataset...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    max_len = int(os.getenv("MAX_LEN", "512"))

    def tokenize_function(examples):
        input_ids_list = []
        labels_list = []
        for inst, out in zip(examples["instruction"], examples["output"]):
            user_prompt = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n"
            assistant_resp = f"{out}<|im_end|>"

            user_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
            assistant_ids = tokenizer.encode(assistant_resp, add_special_tokens=False)

            full_ids = user_ids + assistant_ids
            full_labels = ([-100] * len(user_ids)) + assistant_ids

            padding_len = max_len - len(full_ids)
            if padding_len > 0:
                full_ids += [tokenizer.pad_token_id] * padding_len
                full_labels += [-100] * padding_len

            input_ids_list.append(full_ids[:max_len])
            labels_list.append(full_labels[:max_len])

        return {"input_ids": input_ids_list, "labels": labels_list}

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir="./Qwen3-0.6B-mars-final",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=7,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=1,
        bf16=True,
        report_to=["none"],
        save_strategy="no",
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized_dataset)

    log("Starting Trainer...")
    trainer.train()

    # Save adapter (folder)
    output_adapter_dir = "final_lora_adapter"
    model.save_pretrained(output_adapter_dir)

    # Marker + file list
    with open(os.path.join(output_adapter_dir, "_SUCCESS"), "w") as f:
        f.write("ok\n")
    write_dir_tree(output_adapter_dir, os.path.join(output_adapter_dir, "_FILES.txt"))

    log(f"Training complete. Adapter saved to {output_adapter_dir}")
    log(f"Adapter dir contents: {os.listdir(output_adapter_dir)}")

    # Upload to:
    # model-fine-tuning-0119144635/final_lora_adapter/
    adapter_s3_prefix = f"{run_prefix}/final_lora_adapter"
    log(f"Uploading adapter folder to s3://{bucket_name}/{adapter_s3_prefix}/")
    upload_dir_to_s3(bucket_name, output_adapter_dir, adapter_s3_prefix)
    log(f"Upload complete: s3://{bucket_name}/{adapter_s3_prefix}/")

if __name__ == "__main__":
    main()
