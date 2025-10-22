#!/usr/bin/env bash
# Hugging Face Router + Fireworks backend (OpenAI-compatible)

export HF_TOKEN=${HF_TOKEN:-"REDACTED"}
export HF_ROUTER_BASE=${HF_ROUTER_BASE:-"https://router.huggingface.co"}
export LLM_MODEL=${LLM_MODEL:-"meta-llama/Llama-3.1-8B-Instruct:fireworks-ai"}

export LLM_API_BASE="${HF_ROUTER_BASE%/}/v1"
export LLM_API_KEY="$HF_TOKEN"
export PROVIDER="openai"

echo "[router] LLM_API_BASE=$LLM_API_BASE"
echo "[router] LLM_MODEL=$LLM_MODEL"
echo "[router] PROVIDER=$PROVIDER"
if [ -n "$LLM_API_KEY" ]; then
  echo "[router] LLM_API_KEY=*** (hidden)"
else
  echo "[router] LLM_API_KEY is empty"
fi


