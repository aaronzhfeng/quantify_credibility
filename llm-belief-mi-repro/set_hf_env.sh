#!/usr/bin/env bash
# Convenience wrapper: choose one of the specialized env scripts.
# Usage examples:
#   source ./set_hf_env.sh endpoint   # default: HF Inference Endpoint
#   source ./set_hf_env.sh router     # HF Router + Fireworks
#   source ./set_hf_env.sh local      # Local LM Studio

MODE="$1"
if [ -z "$MODE" ]; then
  MODE="endpoint"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$MODE" in
  endpoint)
    source "$SCRIPT_DIR/set_env_endpoint.sh"
    ;;
  router)
    source "$SCRIPT_DIR/set_env_fireworks_router.sh"
    ;;
  local)
    source "$SCRIPT_DIR/set_env_local_lmstudio.sh"
    ;;
  *)
    echo "Unknown mode '$MODE'. Use: endpoint | router | local" >&2
    return 1
    ;;
esac

