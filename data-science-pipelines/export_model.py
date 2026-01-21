import os
import sys
import re
import torch
import boto3
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def log(msg):
    print(f"--- [EXPORT MODEL LOG]: {msg} ---", flush=True)

def download_s3_folder(bucket_name: str, s3_prefix: str, local_dir: str):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    if not s3_prefix.endswith("/"):
        s3_prefix += "/"

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
    """
    Upload directory recursively to S3 prefix.
    """
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

def ensure_dir_nonempty(local_dir: str, desc: str):
    p = os.path.abspath(local_dir)
    if not os.path.isdir(p):
        raise RuntimeError(f"{desc} dir not found: {p}")
    if len(os.listdir(p)) == 0:
        raise RuntimeError(f"{desc} dir is empty: {p}")

def validate_adapter_dir(adapter_dir: str):
    """
    Ensure adapter looks like PEFT LoRA adapter.
    """
    p = os.path.abspath(adapter_dir)
    files = set(os.listdir(p))

    required_any = ["adapter_model.safetensors", "adapter_model.bin"]
    required_all = ["adapter_config.json"]

    if not any(x in files for x in required_any):
        raise RuntimeError(
            f"Adapter dir {p} missing adapter weights. Expected one of {required_any}. Found: {sorted(files)}"
        )
    for x in required_all:
        if x not in files:
            raise RuntimeError(
                f"Adapter dir {p} missing {x}. Found: {sorted(files)}"
            )

def ensure_base_model_present() -> str:
    bucket_name = os.getenv(
        "MODEL_BUCKET",
        "ocp-model-registry", # CHANGE BUCKET NAME
    )
    s3_model_prefix = os.getenv("MODEL_S3_PREFIX", "Qwen3-0.6B/")
    local_model_dir = os.getenv("MODEL_LOCAL_DIR", "Qwen3-0.6B")

    abs_local_model_dir = os.path.abspath(local_model_dir)

    if os.path.isdir(abs_local_model_dir) and len(os.listdir(abs_local_model_dir)) > 0:
        log(f"Base model dir already exists: {abs_local_model_dir}")
        return abs_local_model_dir

    log(f"Base model dir not found. Downloading from s3://{bucket_name}/{s3_model_prefix}")
    os.makedirs(abs_local_model_dir, exist_ok=True)
    download_s3_folder(bucket_name, s3_model_prefix, abs_local_model_dir)
    ensure_dir_nonempty(abs_local_model_dir, "Base model")
    return abs_local_model_dir

def resolve_run_prefix_or_fail() -> str:
    """
    Detect Elyra run prefix like: model-fine-tuning-0119144635
    Optional override: RUN_PREFIX
    """
    explicit = os.getenv("RUN_PREFIX", "").strip()
    if explicit:
        log(f"Using explicit RUN_PREFIX={explicit}")
        return explicit.rstrip("/")

    pattern = re.compile(r"(model-fine-tuning-\d{10,})")
    hits = []
    for k, v in os.environ.items():
        if not v:
            continue
        m = pattern.search(v)
        if m:
            hits.append((k, m.group(1), v))

    if hits:
        hits_sorted = sorted(hits, key=lambda x: (len(x[1]), x[0]))
        chosen = hits_sorted[0][1]
        log(f"Detected run prefix from env: {chosen}")
        return chosen

    log("CRITICAL: Cannot detect run prefix model-fine-tuning-<digits> from env.")
    interesting = [k for k in os.environ.keys()
                   if any(s in k.lower() for s in ["argo", "workflow", "pipeline", "run", "s3", "bucket", "minio", "cos"])]
    log(f"Available related env keys: {sorted(interesting)}")
    raise RuntimeError("RUN_PREFIX not found in env; cannot determine S3 upload path.")

def resolve_adapter_s3_prefix() -> str:
    """
    Prefer explicit ADAPTER_S3_PREFIX
    Else: <run_prefix>/final_lora_adapter
    """
    adapter_s3_prefix = os.getenv("ADAPTER_S3_PREFIX", "").strip()
    if adapter_s3_prefix:
        return adapter_s3_prefix.rstrip("/")

    run_prefix = resolve_run_prefix_or_fail()
    return f"{run_prefix}/final_lora_adapter"

def ensure_adapter_present() -> str:
    adapter_local_dir = os.getenv("ADAPTER_LOCAL_DIR", "final_lora_adapter")
    abs_adapter_dir = os.path.abspath(adapter_local_dir)

    if os.path.isdir(abs_adapter_dir) and len(os.listdir(abs_adapter_dir)) > 0:
        log(f"Adapter dir already exists: {abs_adapter_dir}")
        validate_adapter_dir(abs_adapter_dir)
        return abs_adapter_dir

    bucket_name = os.getenv(
        "MODEL_BUCKET",
        "ocp-model-registry", # CHANGE BUCKET NAME
    )
    adapter_s3_prefix = resolve_adapter_s3_prefix()

    log(f"Downloading adapter from s3://{bucket_name}/{adapter_s3_prefix}/")
    os.makedirs(abs_adapter_dir, exist_ok=True)
    download_s3_folder(bucket_name, adapter_s3_prefix, abs_adapter_dir)

    ensure_dir_nonempty(abs_adapter_dir, "Adapter")
    validate_adapter_dir(abs_adapter_dir)
    return abs_adapter_dir

def write_marker(output_dir: str):
    with open(os.path.join(output_dir, "_SUCCESS"), "w") as f:
        f.write("ok\n")

def main():
    bucket_name = os.getenv(
        "MODEL_BUCKET",
        "ocp-model-registry", # CHANGE BUCKET NAME
    )

    # Local export dir (temporary)
    local_output_dir = os.getenv("LOCAL_OUTPUT_DIR", "Qwen3-0.6B-mars")
    abs_output_dir = os.path.abspath(local_output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    # S3 destination folder name (under run_prefix)
    export_s3_folder = os.getenv("EXPORT_S3_FOLDER", "Qwen3-0.6B-mars").strip()

    try:
        run_prefix = resolve_run_prefix_or_fail()
        model_path = ensure_base_model_present()
        adapter_path = ensure_adapter_present()
    except Exception as e:
        log(f"CRITICAL ERROR preparing inputs: {str(e)}")
        log(f"PWD: {os.getcwd()}")
        log(f"Files here: {os.listdir('.')}")
        sys.exit(1)

    export_s3_prefix = f"{run_prefix}/{export_s3_folder}"

    log(f"Run prefix:      {run_prefix}")
    log(f"Base model:      {model_path}")
    log(f"Adapter:         {adapter_path}")
    log(f"Local output:    {abs_output_dir}")
    log(f"S3 export path:  s3://{bucket_name}/{export_s3_prefix}/")

    # 1) Load base model on CPU
    log("Loading base model (CPU)...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
    except Exception as e:
        log(f"FAILED to load base model: {str(e)}")
        sys.exit(1)

    # 2) Merge LoRA
    log("Merging LoRA adapter...")
    try:
        lora_model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = lora_model.merge_and_unload()
    except Exception as e:
        log(f"FAILED to merge adapter: {str(e)}")
        sys.exit(1)

    # 3) Save merged model
    log("Saving merged model locally...")
    try:
        merged_model.save_pretrained(abs_output_dir, safe_serialization=True)
    except Exception as e:
        log(f"FAILED to save merged model: {str(e)}")
        sys.exit(1)

    # 4) Save tokenizer
    log("Saving tokenizer locally...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(abs_output_dir)
    except Exception as e:
        log(f"FAILED to save tokenizer: {str(e)}")
        sys.exit(1)

    write_marker(abs_output_dir)

    # 5) Upload to S3
    log("Uploading merged model folder to S3...")
    try:
        upload_dir_to_s3(bucket_name, abs_output_dir, export_s3_prefix)
    except Exception as e:
        log(f"FAILED to upload merged model to S3: {str(e)}")
        sys.exit(1)

    log("Export + Upload complete.")
    log(f"Uploaded to: s3://{bucket_name}/{export_s3_prefix}/")
    log(f"Local output contents: {os.listdir(abs_output_dir)}")

if __name__ == "__main__":
    main()
