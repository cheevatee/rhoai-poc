#!/usr/bin/env python3
import os, sys, re, subprocess, base64, json
from typing import Any, Dict, Optional, Tuple, List

def log(m): print(f"[register_model] {m}", flush=True)
def fatal(m, c=1): print(f"[register_model][ERROR] {m}", flush=True); sys.exit(c)

REGISTRY_URL = os.getenv("MODEL_REGISTRY_URL", "https://model-registry-rest.apps.ocp.tl4lg.sandbox782.opentlc.com").rstrip("/")

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
    s3_path = os.getenv("S3_PATH", "").strip().strip("/")
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
    ensure_pkg("kubernetes", "kubernetes>=28.1.0")
    ensure_pkg("requests", "requests>=2.31.0")
    import requests

    namespace  = env("NAMESPACE", "rhoai-model")
    model_name = env("MODEL_NAME", "qwen3-06b-mars")
    version    = env("MODEL_VERSION", resolve_run_prefix_or_fail())

    bucket      = env("MODEL_BUCKET", "ocp-qnn9k-model-registry-us-east-2-werohpwyueeilfniqmlecbgxqyb")
    s3_path     = resolve_s3_path().strip().strip("/")
    s3_region   = env("MODEL_S3_REGION", env("AWS_REGION", "us-east-2"))
    s3_endpoint = env("MODEL_S3_ENDPOINT", f"https://s3.{s3_region}.amazonaws.com")

    model_uri = env("MODEL_URI", f"s3://{bucket}/{s3_path}")

    model_format_name = env("MODEL_FORMAT_NAME", "hf-transformers")
    model_format_ver  = env("MODEL_FORMAT_VERSION", "1")

    token = read_registry_token_from_secret(
        namespace,
        env("MODEL_REGISTRY_TOKEN_SECRET", "model-registry-token"),
        env("MODEL_REGISTRY_TOKEN_KEY", "token"),
    )

    base = f"{REGISTRY_URL}/api/model_registry/v1alpha3"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}

    def req(method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Tuple[int, str, Optional[Dict[str, Any]]]:
        r = requests.request(method, url, headers=headers, json=payload, timeout=60, verify=False)
        data = None
        if r.headers.get("content-type","").startswith("application/json"):
            try:
                data = r.json()
            except Exception:
                data = None
        return r.status_code, r.text, data

    log(f"MODEL_REGISTRY_URL={REGISTRY_URL}")
    log(f"API_BASE={base}")
    log(f"MODEL_NAME={model_name}")
    log(f"MODEL_VERSION={version}")
    log(f"MODEL_URI={model_uri}")
    log(f"S3 endpoint={s3_endpoint} region={s3_region} bucket={bucket} path={s3_path}")
    log(f"MODEL_FORMAT name={model_format_name} version={model_format_ver}")

    # 1) RegisteredModel
    sc, txt, data = req("GET", f"{base}/registered_models")
    if sc != 200:
        fatal(f"Failed to list registered_models: {sc} {txt}")

    rm_id = None
    for it in pick_items(data):
        if it.get("name") == model_name:
            rm_id = it.get("id")
            break

    if not rm_id:
        sc, txt, data = req("POST", f"{base}/registered_models", {"name": model_name, "description": "Registered by pipeline"})
        if sc not in (200, 201):
            fatal(f"Failed to create RegisteredModel: {sc} {txt}")
        rm_id = (data or {}).get("id")
        if not rm_id:
            fatal(f"RegisteredModel created but id missing: {data}")
        log(f"RegisteredModel created id={rm_id}")
    else:
        log(f"RegisteredModel exists id={rm_id}")

    rm_id = str(rm_id)

    # 2) ModelVersion (minimal)
    mv_payload = {"name": version, "description": f"Exported from run {version}", "registeredModelId": rm_id}
    sc, txt, data = req("POST", f"{base}/model_versions", mv_payload)
    if sc in (200, 201):
        mv_id = str((data or {}).get("id"))
    else:
        # find existing
        sc2, txt2, data2 = req("GET", f"{base}/model_versions")
        if sc2 != 200:
            fatal(f"Failed to create/find ModelVersion: {sc} {txt}")
        mv_id = None
        for it in pick_items(data2):
            if it.get("name") == version and str(it.get("registeredModelId")) == rm_id:
                mv_id = str(it.get("id"))
                break
        if not mv_id:
            fatal(f"Failed to create/find ModelVersion: {sc} {txt}")
    log(f"✅ ModelVersion id={mv_id}")

    # 3) Create ModelArtifact (minimal create, then PATCH format if needed)
    art_payload = {"name": f"{model_name}:{version}", "description": "Model storage reference", "uri": model_uri}
    sc, txt, data = req("POST", f"{base}/model_artifacts", art_payload)
    if sc not in (200, 201):
        fatal(f"Failed to create ModelArtifact: {sc} {txt}")
    created_aid = str((data or {}).get("id"))
    log(f"✅ ModelArtifact created id={created_aid}")

    # PATCH format (your API accepts patching these even when POST rejected extra fields)
    sc, txt, _ = req("PATCH", f"{base}/model_artifacts/{created_aid}", {"modelFormatName": model_format_name, "modelFormatVersion": model_format_ver})
    if sc in (200, 201):
        log(f"✅ Patched artifact format on {created_aid}")
    else:
        log(f"[WARN] Could not patch artifact format on {created_aid}: {sc} {txt[:200]}")

    # 4) Link ModelVersion -> Artifact
    link_payload = {"artifactType": "ModelArtifact", "artifactId": created_aid}
    sc, txt, data = req("POST", f"{base}/model_versions/{mv_id}/artifacts", link_payload)
    if sc not in (200, 201, 409):
        fatal(f"Failed linking artifact: {sc} {txt}")
    log("✅ Linked version -> artifact (request accepted)")

    # 5) IMPORTANT: Find the *actual* artifact id the version returns (your API may show a different id)
    sc, txt, data = req("GET", f"{base}/model_versions/{mv_id}/artifacts")
    if sc != 200:
        fatal(f"Failed to list version artifacts: {sc} {txt}")

    linked_items = pick_items(data)
    if not linked_items:
        fatal("No artifacts returned from /model_versions/{id}/artifacts")

    # Prefer artifact with matching uri; else take first
    linked_aid = None
    for it in linked_items:
        if it.get("uri") == model_uri:
            linked_aid = str(it.get("id"))
            break
    if not linked_aid:
        linked_aid = str(linked_items[0].get("id"))

    log(f"Linked artifact id seen by version = {linked_aid} (created was {created_aid})")

    # Ensure the linked artifact has uri + format
    req("PATCH", f"{base}/model_artifacts/{linked_aid}", {"uri": model_uri})
    req("PATCH", f"{base}/model_artifacts/{linked_aid}", {"modelFormatName": model_format_name, "modelFormatVersion": model_format_ver})

    # Optional: try making it visible by state
    sc, txt, _ = req("PATCH", f"{base}/model_artifacts/{linked_aid}", {"state": "LIVE"})
    if sc not in (200, 201):
        log(f"[INFO] Artifact state patch not supported (ok): {sc} {txt[:120]}")

    result = {
        "registered_model_id": rm_id,
        "model_version_id": mv_id,
        "created_artifact_id": created_aid,
        "linked_artifact_id": linked_aid,
        "model_uri": model_uri,
        "object_storage": {"endpoint": s3_endpoint, "region": s3_region, "bucket": bucket, "path": s3_path},
        "format": {"name": model_format_name, "version": model_format_ver},
        "note": "UI Model location may still require typed customProperties / connection payload (capture from UI Network tab)."
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
