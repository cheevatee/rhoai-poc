#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import re

def log(msg):
    print(f"[deploy_model] {msg}", flush=True)

def fatal(msg, code=1):
    print(f"[deploy_model][ERROR] {msg}", flush=True)
    sys.exit(code)

def env(name, default=None, required=False):
    v = os.getenv(name, default)
    if required and (v is None or str(v).strip() == ""):
        fatal(f"Missing required env var: {name}")
    return v

def ensure_pkg(import_name: str, pip_spec: str):
    """
    Elyra python script image often doesn't install requirements-elyra.txt.
    Install deps at runtime if missing.

    Note: In your environment, '--user' is not visible in the virtualenv.
    So we try normal install first, then fall back to --user.
    """
    try:
        __import__(import_name)
        return
    except Exception:
        log(f"Installing missing package: {pip_spec}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_spec])
            return
        except Exception as e1:
            log(f"pip install (system/venv) failed, trying --user. Details: {e1}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pip_spec])

def prune_none(d):
    if isinstance(d, dict):
        return {k: prune_none(v) for k, v in d.items() if v is not None}
    if isinstance(d, list):
        return [prune_none(x) for x in d]
    return d

def resolve_run_prefix_or_fail() -> str:
    """
    Detect Elyra run prefix like: model-fine-tuning-0120155410
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
        # pick deterministic one
        hits_sorted = sorted(hits, key=lambda x: (x[1], x[0]))
        chosen = hits_sorted[0][1]
        log(f"Detected run prefix from env: {chosen}")
        return chosen

    log("CRITICAL: Cannot detect run prefix model-fine-tuning-<digits> from env.")
    interesting = [k for k in os.environ.keys()
                   if any(s in k.lower() for s in ["argo", "workflow", "pipeline", "run", "s3", "bucket", "minio", "cos"])]
    log(f"Available related env keys: {sorted(interesting)}")
    raise RuntimeError("RUN_PREFIX not found in env; cannot determine S3 path.")

def resolve_s3_path(storage_path_override: str | None) -> str:
    """
    Priority:
      1) S3_PATH env (explicit)
      2) Derived from export_model.py logic: <run_prefix>/<EXPORT_S3_FOLDER>
    """
    if storage_path_override and storage_path_override.strip():
        return storage_path_override.strip().rstrip("/")

    export_s3_folder = os.getenv("EXPORT_S3_FOLDER", "Qwen3-0.6B-mars").strip().strip("/")
    run_prefix = resolve_run_prefix_or_fail()

    s3_path = f"{run_prefix}/{export_s3_folder}"
    return s3_path

def main():
    # ---- Ensure deps ----
    ensure_pkg("kubernetes", "kubernetes>=28.1.0")
    ensure_pkg("yaml", "pyyaml>=6.0.1")

    # ---- Import kubernetes client ----
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except Exception as e:
        fatal(f"Failed to import kubernetes python client even after install. Details: {e}")

    # ---- Parameters (allow override via env vars) ----
    namespace   = env("NAMESPACE", "rhoai-model")
    model_name  = env("MODEL_NAME", "qwen3-06b-mars")
    storage_key = env("STORAGE_KEY", "aws-s3-models")

    # If S3_PATH is not set, derive from RUN_PREFIX + EXPORT_S3_FOLDER (same as export_model.py)
    s3_path     = resolve_s3_path(env("S3_PATH", ""))

    cpu         = env("CPU", "1")
    memory      = env("MEMORY", "12Gi")
    gpu         = env("GPU", "1")

    runtime_image = env(
        "RUNTIME_IMAGE",
        "registry.redhat.io/rhaiis/vllm-cuda-rhel9@sha256:ad756c01ec99a99cc7d93401c41b8d92ca96fb1ab7c5262919d818f2be4f3768",
    )

    # This is the SA KServe will use to access the S3 connection
    sa_name = env("SERVICE_ACCOUNT_NAME", "aws-s3-models-sa")

    deployment_mode = env("DEPLOYMENT_MODE", "RawDeployment")

    # Optional: wait for resources READY
    wait_ready = env("WAIT_READY", "true").lower() in ("1", "true", "yes", "y")
    wait_timeout_sec = int(env("WAIT_TIMEOUT_SEC", "600"))
    wait_interval_sec = int(env("WAIT_INTERVAL_SEC", "10"))

    log("Starting...")
    log(f"namespace={namespace}")
    log(f"model_name={model_name}")
    log(f"storage_key={storage_key}")
    log(f"s3_path={s3_path}")
    log(f"runtime_image={runtime_image}")
    log(f"resources: cpu={cpu}, memory={memory}, gpu={gpu}")
    log(f"serviceAccountName={sa_name}")
    log(f"deployment_mode={deployment_mode}")
    log(f"wait_ready={wait_ready} timeout={wait_timeout_sec}s interval={wait_interval_sec}s")

    # In-cluster config (DSP component pod)
    try:
        config.load_incluster_config()
    except Exception as e:
        fatal(f"Failed to load in-cluster kube config. Are you running inside the cluster? Details: {e}")

    co_api = client.CustomObjectsApi()

    def create_or_patch(group, version, plural, body):
        body = prune_none(body)
        name = body["metadata"]["name"]
        ns = body["metadata"].get("namespace", namespace)
        try:
            co_api.get_namespaced_custom_object(group, version, ns, plural, name)
            co_api.patch_namespaced_custom_object(group, version, ns, plural, name, body)
            log(f"Patched {plural}/{name} in ns={ns}")
        except ApiException as e:
            if e.status == 404:
                co_api.create_namespaced_custom_object(group, version, ns, plural, body)
                log(f"Created {plural}/{name} in ns={ns}")
            else:
                fatal(f"Failed applying {plural}/{name}: {e.status} {e.reason} {e.body}")

    def get_condition(obj, cond_type):
        for c in (obj.get("status", {}).get("conditions", []) or []):
            if c.get("type") == cond_type:
                return c
        return None

    def wait_inferenceservice_ready():
        deadline = time.time() + wait_timeout_sec
        while time.time() < deadline:
            try:
                isvc = co_api.get_namespaced_custom_object(
                    "serving.kserve.io", "v1beta1", namespace, "inferenceservices", model_name
                )
            except ApiException as e:
                log(f"Waiting: cannot get InferenceService yet ({e.status} {e.reason})")
                time.sleep(wait_interval_sec)
                continue

            ready = get_condition(isvc, "Ready")
            predictor_ready = get_condition(isvc, "PredictorReady")
            ingress_ready = get_condition(isvc, "IngressReady")

            r = ready.get("status") if ready else "Unknown"
            pr = predictor_ready.get("status") if predictor_ready else "Unknown"
            ir = ingress_ready.get("status") if ingress_ready else "Unknown"
            url = isvc.get("status", {}).get("url")

            log(f"Ready={r} PredictorReady={pr} IngressReady={ir} url={url}")

            if ready and ready.get("status") == "True":
                log("InferenceService is Ready=True")
                return

            time.sleep(wait_interval_sec)

        fatal(f"Timeout waiting for InferenceService {model_name} to become Ready")

    # ---- ServingRuntime ----
    servingruntime = {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": model_name,
            "namespace": namespace,
            "labels": {"opendatahub.io/dashboard": "true"},
            "annotations": {
                "opendatahub.io/apiProtocol": "REST",
                "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
                "opendatahub.io/runtime-version": "v0.9.1.0",
                "opendatahub.io/serving-runtime-scope": "global",
                "opendatahub.io/template-display-name": "vLLM NVIDIA GPU ServingRuntime for KServe",
                "opendatahub.io/template-name": "vllm-cuda-runtime-template",
                "openshift.io/display-name": "vLLM NVIDIA GPU ServingRuntime for KServe",
            },
        },
        "spec": {
            "annotations": {
                "opendatahub.io/kserve-runtime": "vllm",
                "prometheus.io/path": "/metrics",
                "prometheus.io/port": "8080",
            },
            "containers": [
                {
                    "name": "kserve-container",
                    "image": runtime_image,
                    "command": ["python", "-m", "vllm.entrypoints.openai.api_server"],
                    "args": [
                        "--port=8080",
                        "--model=/mnt/models",
                        "--served-model-name={{.Name}}",
                    ],
                    "env": [{"name": "HF_HOME", "value": "/tmp/hf_home"}],
                    "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                }
            ],
            "multiModel": False,
            "supportedModelFormats": [{"name": "vLLM", "autoSelect": True}],
        },
    }

    # ---- InferenceService ----
    inferenceservice = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace,
            "labels": {
                "networking.kserve.io/visibility": "exposed",
                "opendatahub.io/dashboard": "true",
                "opendatahub.io/genai-asset": "true",
            },
            "annotations": {
                "opendatahub.io/connection-path": s3_path,
                "opendatahub.io/connections": storage_key,
                "opendatahub.io/genai-use-case": "chat",
                "opendatahub.io/hardware-profile-name": "gpu-profile",
                "opendatahub.io/hardware-profile-namespace": "redhat-ods-applications",
                "opendatahub.io/model-type": "generative",
                "openshift.io/display-name": model_name,
                "security.opendatahub.io/enable-auth": "false",
                "serving.kserve.io/deploymentMode": deployment_mode,
            },
        },
        "spec": {
            "predictor": {
                "automountServiceAccountToken": False,
                "minReplicas": 1,
                "maxReplicas": 1,
                "serviceAccountName": sa_name,
                "model": {
                    "modelFormat": {"name": "vLLM"},
                    "runtime": model_name,
                    "resources": {
                        "requests": {"cpu": cpu, "memory": memory, "nvidia.com/gpu": gpu},
                        "limits": {"cpu": cpu, "memory": memory, "nvidia.com/gpu": gpu},
                    },
                    "storage": {"key": storage_key, "path": s3_path},
                },
            }
        },
    }

    # Apply resources (RBAC required for pipeline-runner-dspa)
    create_or_patch("serving.kserve.io", "v1alpha1", "servingruntimes", servingruntime)
    create_or_patch("serving.kserve.io", "v1beta1", "inferenceservices", inferenceservice)

    log("All resources applied successfully.")

    if wait_ready:
        wait_inferenceservice_ready()

    log("Done.")

if __name__ == "__main__":
    main()
