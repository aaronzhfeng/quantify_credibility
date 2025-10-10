from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx


@dataclass
class OpenAICompatibleLLMClient:
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = ""
    request_timeout_s: float = 60.0

    def _endpoint(self, path: str) -> str:
        if self.base_url.rstrip("/").endswith("/v1"):
            return f"{self.base_url.rstrip('/')}{path}"
        return f"{self.base_url.rstrip('/')}/v1{path}"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> str:
        url = self._endpoint("/chat/completions")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": max(0.0, float(temperature)),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        with httpx.Client(timeout=self.request_timeout_s) as client:
            resp = client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unexpected response: {json.dumps(data)[:500]}") from exc
        return content.strip()


    # Placeholder for backends that support logprobs in the future
    def supports_logprobs(self) -> bool:
        return False

    def chat_completion_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 128,
        top_logprobs: int = 1,
    ) -> tuple[str, Optional[List[float]]]:
        """Attempt to request token logprobs.

        Returns (text, token_logprobs or None if unavailable).
        """
        url = self._endpoint("/chat/completions")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": max(0.0, float(temperature)),
            "max_tokens": int(max_tokens),
            "stream": False,
            "logprobs": True,
            "top_logprobs": int(top_logprobs),
        }
        with httpx.Client(timeout=self.request_timeout_s) as client:
            resp = client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unexpected response: {json.dumps(data)[:500]}") from exc

        # OpenAI-style: choices[0].logprobs.content -> list of {token, logprob}
        token_logprobs: Optional[List[float]] = None
        try:
            lp_items = data["choices"][0]["logprobs"]["content"]
            token_logprobs = [float(item.get("logprob", 0.0)) for item in lp_items]
        except Exception:
            token_logprobs = None
        return content, token_logprobs


