export TOKEN="$(oc whoami -t)"
oc -n rhoai-model create secret generic model-registry-token \
  --from-literal=token="$TOKEN"
