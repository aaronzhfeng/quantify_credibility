#!/usr/bin/env bash
# Hugging Face Inference Endpoint (default)

export HF_TOKEN=${HF_TOKEN:-"REDACTED"}
export HF_ENDPOINT_URL=${HF_ENDPOINT_URL:-"https://itjufncm4jaicprw.us-east-1.aws.endpoints.huggingface.cloud"}

export LLM_API_BASE="${HF_ENDPOINT_URL%/}"
export LLM_MODEL=""          # Not required when using a direct endpoint URL
export LLM_API_KEY="$HF_TOKEN"
export PROVIDER="hf"

echo "[endpoint] LLM_API_BASE=$LLM_API_BASE"
echo "[endpoint] PROVIDER=$PROVIDER"
if [ -n "$LLM_API_KEY" ]; then
  echo "[endpoint] LLM_API_KEY=*** (hidden)"
else
  echo "[endpoint] LLM_API_KEY is empty"
fi


