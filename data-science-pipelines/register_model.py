#!/usr/bin/env python3
import os, sys, re, subprocess, base64, json

def log(m): print(f"[register_model] {m}", flush=True)
def fatal(m, c=1): print(f"[register_model][ERROR] {m}", flush=True); sys.exit(c)

# model registry route
REGISTRY_URL = "https://model-registry-rest.apps.ocp.example.com"

def env(name, default=None):
    v = os.getenv(name, default)
    return v if v is not None else default

def ensure_pkg(import_name: str, pip_spec: str):
    try:
        __import__(import_name); return
    except Exception:
        log(f"Installing missing package: {pip_spec}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_spec])

def resolve_run_prefix_or_fail() -> str:
    explicit = os.getenv("RUN_PREFIX", "").strip()
    if explicit:
        log(f"Using explicit RUN_PREFIX={explicit}")
        return explicit.rstrip("/")
    pattern = re.compile(r"(model-fine-tuning-\d{10,})")
    for v in os.environ.values():
        if not v:
            continue
        m = pattern.search(v)
        if m:
            log(f"Detected run prefix from env: {m.group(1)}")
            return m.group(1)
    raise RuntimeError("RUN_PREFIX not found")

def resolve_s3_path() -> str:
    s3_path = os.getenv("S3_PATH", "").strip().rstrip("/")
    if s3_path:
        return s3_path
    run_prefix = resolve_run_prefix_or_fail()
    export_s3_folder = os.getenv("EXPORT_S3_FOLDER", "Qwen3-0.6B-mars").strip().strip("/")
    return f"{run_prefix}/{export_s3_folder}"

def read_registry_token_from_secret(namespace: str, secret_name: str, key: str) -> str:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    try:
        sec = v1.read_namespaced_secret(secret_name, namespace)
    except ApiException as e:
        raise RuntimeError(f"Cannot read secret {namespace}/{secret_name}: {e.status} {e.reason} {e.body}")
    if not sec.data or key not in sec.data:
        raise RuntimeError(f"Secret {namespace}/{secret_name} missing key '{key}'")
    return base64.b64decode(sec.data[key]).decode("utf-8").strip()

def pick_items(data):
    if data is None:
        return []
    if isinstance(data, dict):
        it = data.get("items", [])
        return it if isinstance(it, list) else []
    if isinstance(data, list):
        return data
    return []

def main():
    if "<your-model-registry-route>" in REGISTRY_URL:
        fatal("Please set REGISTRY_URL at top of script to your real Model Registry Route (https://...)")

    ensure_pkg("kubernetes", "kubernetes>=28.1.0")
    ensure_pkg("requests", "requests>=2.31.0")
    import requests

    namespace  = env("NAMESPACE", "rhoai-model")
    model_name = env("MODEL_NAME", "qwen3-06b-mars")
    version    = env("MODEL_VERSION", resolve_run_prefix_or_fail())
    bucket     = env("MODEL_BUCKET", "ocp-model-registry") # CHANGE BUCKET NAME
    s3_path    = resolve_s3_path()
    model_uri  = env("MODEL_URI", f"s3://{bucket}/{s3_path}/")

    token = read_registry_token_from_secret(
        namespace,
        env("MODEL_REGISTRY_TOKEN_SECRET", "model-registry-token"),
        env("MODEL_REGISTRY_TOKEN_KEY", "token"),
    )

    registry = REGISTRY_URL.rstrip("/")
    base = f"{registry}/api/model_registry/v1alpha3"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}

    def req(method, url, payload=None):
        r = requests.request(method, url, headers=headers, json=payload, timeout=60, verify=False)
        data = None
        if r.headers.get("content-type","").startswith("application/json"):
            try:
                data = r.json()
            except Exception:
                data = None
        return r.status_code, r.text, data

    log(f"MODEL_REGISTRY_URL(HARDCODED)={registry}")
    log(f"MODEL_NAME={model_name}")
    log(f"MODEL_VERSION={version}")
    log(f"MODEL_URI={model_uri}")
    log(f"API_BASE={base}")

    # 1) Get-or-create RegisteredModel
    sc, txt, data = req("GET", f"{base}/registered_models")
    if sc != 200:
        fatal(f"Failed to list registered_models: {sc} {txt}")

    rm_id = None
    for it in pick_items(data):
        if it.get("name") == model_name:
            rm_id = it.get("id")
            break

    if rm_id:
        log(f"RegisteredModel exists id={rm_id}")
    else:
        sc, txt, data = req("POST", f"{base}/registered_models", {"name": model_name, "description": "Registered by pipeline"})
        if sc not in (200, 201):
            fatal(f"Failed to create RegisteredModel: {sc} {txt}")
        rm_id = (data or {}).get("id")
        if not rm_id:
            fatal(f"RegisteredModel created but id missing: {data}")
        log(f"RegisteredModel created id={rm_id}")

    rm_id_str = str(rm_id)  # <-- IMPORTANT: registry expects registeredModelId as string

    # 2) Create Artifact
    sc, txt, data = req("POST", f"{base}/model_artifacts", {
        "name": f"{model_name}:{version}",
        "description": "Model storage reference",
        "uri": model_uri
    })
    if sc not in (200, 201):
        fatal(f"Failed to create Artifact at /model_artifacts with uri: {sc} {txt}")

    artifact_id = (data or {}).get("id")
    if not artifact_id:
        fatal(f"Artifact created but id missing: {data}")
    log(f"✅ Artifact created id={artifact_id}")

    # 3) Create ModelVersion
    #
    # FIX: registeredModelId MUST be a STRING, not a number
    #
    version_payload = {
        "name": version,
        "description": f"Exported from run {version}",
        "registeredModelId": rm_id_str,  # <-- FIX
    }

    candidates = [
        f"{base}/model_versions",
        f"{base}/registered_models/{rm_id_str}/versions",
    ]

    created_version = None
    last_err = None

    for url in candidates:
        sc, txt, data = req("POST", url, version_payload)
        log(f"TRY POST {url} payload={{name,description,registeredModelId}} -> {sc} BODY: {txt[:300]}")
        if sc in (200, 201, 409):
            created_version = data
            log(f"✅ ModelVersion created/exists via {url}")
            break
        last_err = f"{url} -> {sc}: {txt}"

    if not created_version and "-> 409" not in (last_err or ""):
        fatal(
            "Failed to create ModelVersion even after fixing registeredModelId type.\n"
            f"Last error: {last_err}"
        )

    # Try to get version id from response; if missing, search it
    mv_id = (created_version or {}).get("id")
    if not mv_id:
        sc, txt, data = req("GET", f"{base}/model_versions")
        if sc == 200:
            for it in pick_items(data):
                if it.get("name") == version and str(it.get("registeredModelId")) == rm_id_str:
                    mv_id = it.get("id")
                    break

    # 4) OPTIONAL: attempt to link artifact to model version (best-effort)
    linked = False
    if mv_id:
        link_payloads = [
            {"modelVersionId": int(mv_id), "modelArtifactId": int(artifact_id)},
            {"model_version_id": int(mv_id), "model_artifact_id": int(artifact_id)},
            {"modelVersionId": int(mv_id), "artifactId": int(artifact_id)},
        ]
        link_endpoints = [
            f"{base}/model_version_artifacts",
            f"{base}/modelversion_artifacts",
            f"{base}/model_versions/{mv_id}/artifacts",
            f"{base}/model_versions/{mv_id}/model_artifacts",
        ]

        for ep in link_endpoints:
            for lp in link_payloads:
                sc, txt, data = req("POST", ep, lp)
                if sc in (200, 201, 409):
                    log(f"✅ Linked artifact to version via {ep} payload_keys={list(lp.keys())}")
                    linked = True
                    break
            if linked:
                break

    result = {
        "registered_model_id": rm_id,
        "artifact_id": artifact_id,
        "model_version_id": mv_id,
        "version": version,
        "model_uri": model_uri,
        "artifact_linked": linked
    }
    print(json.dumps(result, indent=2))

    if not linked:
        log("NOTE: Artifact linking not confirmed. Version is created, but registry may link artifacts differently (or auto-link via UI).")

if __name__ == "__main__":
    main()
