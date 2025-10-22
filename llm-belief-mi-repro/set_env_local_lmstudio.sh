#!/usr/bin/env bash
# Local LM Studio (OpenAI-compatible on http://127.0.0.1:1234/v1)

export LLM_API_BASE=${LLM_API_BASE:-"http://127.0.0.1:1234/v1"}
export LLM_API_KEY=${LLM_API_KEY:-"lm-studio"}
export LLM_MODEL=${LLM_MODEL:-"llama31_8b"}
export PROVIDER="openai"

echo "[local] LLM_API_BASE=$LLM_API_BASE"
echo "[local] LLM_MODEL=$LLM_MODEL"
echo "[local] PROVIDER=$PROVIDER"


