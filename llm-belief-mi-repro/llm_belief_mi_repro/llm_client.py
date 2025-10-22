from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import asyncio
import time

import httpx
from .cache import SQLiteCache, canonical_key


@dataclass
class OpenAICompatibleLLMClient:
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = ""
    request_timeout_s: float = 60.0
    cache: Optional[SQLiteCache] = None

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
        on_raw: Optional[callable] = None,
    ) -> str:
        url = self._endpoint("/chat/completions")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": max(0.0, float(temperature)),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        # Cache lookup
        sampling_mode = float(temperature) > 0.0
        if self.cache is not None and not sampling_mode:
            key = canonical_key({"url": url, **payload})
            hit = self.cache.get(key)
            if hit.hit and hit.content is not None:
                return hit.content
        with httpx.Client(timeout=self.request_timeout_s) as client:
            t0 = time.time()
            resp = client.post(url, headers=self._headers(), json=payload)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"HF endpoint error {e.response.status_code}: {e.response.text[:400]}") from e
            data = resp.json()
        # on_raw may not be provided by older call sites
        cb = locals().get('on_raw')
        if cb is not None:
            try:
                cb(data)
            except Exception:
                pass
        # Cache store
        if self.cache is not None:
            try:
                content_try = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage") or {}
                self.cache.put(
                    key,
                    request=payload,
                    response=data,
                    content=content_try,
                    token_logprobs=None,
                    usage_prompt_tokens=usage.get("prompt_tokens"),
                    usage_completion_tokens=usage.get("completion_tokens"),
                    duration_ms=int((time.time() - t0) * 1000),
                )
            except Exception:
                pass
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
        on_raw: Optional[callable] = None,
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
        # Cache lookup
        key = None
        if self.cache is not None:
            key = canonical_key({"url": url, **payload})
            # Allow cache for logprobs requests regardless of temperature
            hit = self.cache.get(key)
            if hit.hit and hit.content is not None:
                return hit.content, hit.token_logprobs
        with httpx.Client(timeout=self.request_timeout_s) as client:
            t0 = time.time()
            resp = client.post(url, headers=self._headers(), json=payload)
            if resp.status_code == 422:
                # Retry without extras
                payload_no_details = {
                    "inputs": _join_messages_as_prompt(messages),
                    "parameters": {
                        "temperature": max(1e-6, float(temperature)),
                        "max_new_tokens": int(max_tokens),
                        "return_full_text": False,
                    },
                }
                resp = client.post(url, headers=self._headers(), json=payload_no_details)
                if resp.status_code == 422:
                    # Final fallback: only inputs
                    resp = client.post(url, headers=self._headers(), json={"inputs": _join_messages_as_prompt(messages)})
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"HF endpoint error {e.response.status_code}: {e.response.text[:400]}") from e
            data = resp.json()
        # on_raw may be absent in some legacy call sites
        try:
            cb2 = on_raw
        except Exception:
            cb2 = None
        if cb2 is not None:
            try:
                cb2(data)
            except Exception:
                pass
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
        # Cache store
        if self.cache is not None and key is not None:
            usage = data.get("usage") or {}
            try:
                self.cache.put(
                    key,
                    request=payload,
                    response=data,
                    content=content,
                    token_logprobs=token_logprobs,
                    usage_prompt_tokens=usage.get("prompt_tokens"),
                    usage_completion_tokens=usage.get("completion_tokens"),
                    duration_ms=int((time.time() - t0) * 1000),
                )
            except Exception:
                pass
        return content, token_logprobs


@dataclass
class AsyncOpenAICompatibleLLMClient:
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = ""
    request_timeout_s: float = 60.0
    semaphore: Optional[asyncio.Semaphore] = None
    cache: Optional[SQLiteCache] = None

    def _endpoint(self, path: str) -> str:
        if self.base_url.rstrip("/").endswith("/v1"):
            return f"{self.base_url.rstrip('/')}" + path
        return f"{self.base_url.rstrip('/')}" + "/v1" + path

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 128,
        logprobs: bool = False,
        top_logprobs: int = 1,
        on_raw: Optional[callable] = None,
    ) -> tuple[str, Optional[List[float]]]:
        url = self._endpoint("/chat/completions")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": max(0.0, float(temperature)),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if logprobs:
            payload.update({"logprobs": True, "top_logprobs": int(top_logprobs)})
        sampling_mode = (float(temperature) > 0.0) and (not logprobs)
        if self.cache is not None and not sampling_mode:
            key = canonical_key({"url": url, **payload})
            hit = self.cache.get(key)
            if hit.hit and hit.content is not None:
                return hit.content, hit.token_logprobs
        # Use semaphore if provided; otherwise proceed without a context manager
        if self.semaphore is not None:
            async with self.semaphore:
                async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
                    resp = await client.post(url, headers=self._headers(), json=payload)
                    if resp.status_code == 422:
                        payload_no_details = {
                            "inputs": "test",
                            "parameters": {
                                "temperature": max(0.0, float(temperature)),
                                "max_new_tokens": int(max_tokens),
                                "return_full_text": False,
                            },
                        }
                        resp = await client.post(url, headers=self._headers(), json=payload_no_details)
                    try:
                        resp.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        raise RuntimeError(f"HF endpoint error {e.response.status_code}: {e.response.text[:400]}") from e
                    data = resp.json()
        else:
            async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
                resp = await client.post(url, headers=self._headers(), json=payload)
                if resp.status_code == 422:
                    text_inp = _join_messages_as_prompt(messages)
                    payload_no_details = {
                        "inputs": text_inp,
                        "parameters": {
                            "temperature": max(1e-6, float(temperature)),
                            "max_new_tokens": int(max_tokens),
                            "return_full_text": False,
                        },
                    }
                    resp = await client.post(url, headers=self._headers(), json=payload_no_details)
                    if resp.status_code == 422:
                        resp = await client.post(url, headers=self._headers(), json={"inputs": text_inp})
                if resp.status_code == 422:
                    text_inp = _join_messages_as_prompt(messages)
                    payload_no_details = {
                        "inputs": text_inp,
                        "parameters": {
                            "temperature": max(1e-6, float(temperature)),
                            "max_new_tokens": int(max_tokens),
                            "return_full_text": False,
                        },
                    }
                    resp = await client.post(url, headers=self._headers(), json=payload_no_details)
                    if resp.status_code == 422:
                        resp = await client.post(url, headers=self._headers(), json={"inputs": text_inp})
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise RuntimeError(f"HF endpoint error {e.response.status_code}: {e.response.text[:400]}") from e
                data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unexpected response: {json.dumps(data)[:500]}") from exc
        token_logprobs: Optional[List[float]] = None
        if logprobs:
            try:
                lp_items = data["choices"][0]["logprobs"]["content"]
                token_logprobs = [float(item.get("logprob", 0.0)) for item in lp_items]
            except Exception:
                token_logprobs = None
        if self.cache is not None:
            usage = data.get("usage") or {}
            try:
                self.cache.put(
                    canonical_key({"url": url, **payload}),
                    request=payload,
                    response=data,
                    content=content,
                    token_logprobs=token_logprobs,
                    usage_prompt_tokens=usage.get("prompt_tokens"),
                    usage_completion_tokens=usage.get("completion_tokens"),
                    duration_ms=0,
                )
            except Exception:
                pass
        return content, token_logprobs



# ------------------------- Hugging Face Inference API -------------------------

def _join_messages_as_prompt(messages: List[Dict[str, str]]) -> str:
    system_parts: List[str] = []
    user_parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = str(m.get("content", ""))
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        else:
            user_parts.append(content)
    sys_text = ("\n\n".join(system_parts) + "\n\n") if system_parts else ""
    return f"{sys_text}{user_parts[-1] if user_parts else ''}"


@dataclass
class HuggingFaceInferenceClient:
    base_url: str = "https://api-inference.huggingface.co/models"
    api_key: str = ""
    model: str = ""
    request_timeout_s: float = 60.0
    cache: Optional[SQLiteCache] = None

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        # If caller provided a full endpoint URL (Inference Endpoint), it may already point to the model
        if "/models/" in base or base.endswith(".cloud") or base.endswith(".hf.space"):
            return base
        if self.model:
            return f"{base}/{self.model}"
        return base

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 128) -> str:
        url = self._endpoint()
        prompt = _join_messages_as_prompt(messages)
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": max(1e-6, float(temperature)),
                "max_new_tokens": int(max_tokens),
                "return_full_text": False,
                "do_sample": bool(float(temperature) > 0.0),
            },
        }
        sampling_mode = float(temperature) > 0.0
        if self.cache is not None and not sampling_mode:
            key = canonical_key({"url": url, **payload})
            hit = self.cache.get(key)
            if hit.hit and hit.content is not None:
                return hit.content
        with httpx.Client(timeout=self.request_timeout_s) as client:
            t0 = time.time()
            resp = client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
        # TGI returns list of {generated_text, details}
        try:
            if isinstance(data, list) and data:
                text = str(data[0].get("generated_text", "")).strip()
            elif isinstance(data, dict):  # Some deployments return a dict
                text = str(data.get("generated_text", "")).strip()
            else:
                text = ""
        except Exception:
            text = ""
        if self.cache is not None:
            try:
                self.cache.put(
                    key,
                    request=payload,
                    response=data if isinstance(data, dict) else {"data": data},
                    content=text,
                    token_logprobs=None,
                    usage_prompt_tokens=None,
                    usage_completion_tokens=None,
                    duration_ms=int((time.time() - t0) * 1000),
                )
            except Exception:
                pass
        return text

    def chat_completion_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 128,
        top_logprobs: int = 1,
        on_raw: Optional[callable] = None,
    ) -> tuple[str, Optional[List[float]]]:
        url = self._endpoint()
        prompt = _join_messages_as_prompt(messages)
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": max(1e-6, float(temperature)),
                "max_new_tokens": int(max_tokens),
                "return_full_text": False,
                # Per HF Inference Endpoint docs: details+logprobs under parameters
                "details": True,
                # Some TGI versions use top_n_tokens instead of logprobs
                "logprobs": int(max(1, int(top_logprobs))),
                "top_n_tokens": int(max(1, int(top_logprobs))),
                "do_sample": bool(float(temperature) > 0.0),
            },
        }
        key = None
        if self.cache is not None:
            key = canonical_key({"url": url, **payload})
            hit = self.cache.get(key)
            if hit.hit and hit.content is not None:
                return hit.content, hit.token_logprobs
        with httpx.Client(timeout=self.request_timeout_s) as client:
            t0 = time.time()
            resp = client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
        # raw callback if provided
        if on_raw is not None:
            try:
                on_raw(data)
            except Exception:
                pass
        text = ""
        token_logprobs: Optional[List[float]] = None
        try:
            node = data[0] if isinstance(data, list) and data else data
            text = str(node.get("generated_text", "")).strip()
            details = node.get("details") if isinstance(node, dict) else None
            if isinstance(details, dict):
                toks = details.get("tokens") or []
                if toks:
                    token_logprobs = [float(t.get("logprob", 0.0)) for t in toks if isinstance(t, dict)] or None
                if not token_logprobs:
                    trans = details.get("transition_scores")
                    if isinstance(trans, list) and trans:
                        try:
                            token_logprobs = [float(x) for x in trans]
                        except Exception:
                            token_logprobs = None
        except Exception:
            token_logprobs = None
        if self.cache is not None and key is not None:
            try:
                self.cache.put(
                    key,
                    request=payload,
                    response=data if isinstance(data, dict) else {"data": data},
                    content=text,
                    token_logprobs=token_logprobs,
                    usage_prompt_tokens=None,
                    usage_completion_tokens=None,
                    duration_ms=int((time.time() - t0) * 1000),
                )
            except Exception:
                pass
        return text, token_logprobs


@dataclass
class AsyncHuggingFaceInferenceClient:
    base_url: str = "https://api-inference.huggingface.co/models"
    api_key: str = ""
    model: str = ""
    request_timeout_s: float = 60.0
    semaphore: Optional[asyncio.Semaphore] = None
    cache: Optional[SQLiteCache] = None

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if "/models/" in base or base.endswith(".cloud") or base.endswith(".hf.space"):
            return base
        if self.model:
            return f"{base}/{self.model}"
        return base

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 128,
        logprobs: bool = False,
        top_logprobs: int = 1,
        on_raw: Optional[callable] = None,
    ) -> tuple[str, Optional[List[float]]]:
        url = self._endpoint()
        prompt = _join_messages_as_prompt(messages)
        params = {
            "temperature": max(1e-6, float(temperature)),
            "max_new_tokens": int(max_tokens),
            "return_full_text": False,
            "do_sample": bool(float(temperature) > 0.0),
        }
        if logprobs:
            params.update({
                "details": True,
                "logprobs": int(max(1, int(top_logprobs))),
                "top_n_tokens": int(max(1, int(top_logprobs))),
            })
        payload = {
            "inputs": prompt,
            "parameters": params,
        }
        sampling_mode = (float(temperature) > 0.0) and (not logprobs)
        if self.cache is not None and not sampling_mode:
            key = canonical_key({"url": url, **payload})
            hit = self.cache.get(key)
            if hit.hit and hit.content is not None:
                return hit.content, hit.token_logprobs
        if self.semaphore is not None:
            async with self.semaphore:
                async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
                    resp = await client.post(url, headers=self._headers(), json=payload)
                    resp.raise_for_status()
                    data = resp.json()
        else:
            async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
                resp = await client.post(url, headers=self._headers(), json=payload)
                if resp.status_code == 422:
                    payload_no_details = {
                        "inputs": prompt,
                        "parameters": {
                            "temperature": max(0.0, float(temperature)),
                            "max_new_tokens": int(max_tokens),
                            "return_full_text": False,
                        },
                    }
                    resp = await client.post(url, headers=self._headers(), json=payload_no_details)
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise RuntimeError(f"HF endpoint error {e.response.status_code}: {e.response.text[:400]}") from e
                data = resp.json()
        # raw callback if provided
        if on_raw is not None:
            try:
                on_raw(data)
            except Exception:
                pass
        try:
            node = data[0] if isinstance(data, list) and data else data
            text = str(node.get("generated_text", "")).strip()
        except Exception:
            text = ""
        token_logprobs: Optional[List[float]] = None
        if logprobs:
            try:
                node = data[0] if isinstance(data, list) and data else data
                details = node.get("details") if isinstance(node, dict) else None
                if isinstance(details, dict):
                    toks = details.get("tokens") or []
                    if toks:
                        token_logprobs = [float(t.get("logprob", 0.0)) for t in toks if isinstance(t, dict)] or None
                    if not token_logprobs:
                        trans = details.get("transition_scores")
                        if isinstance(trans, list) and trans:
                            try:
                                token_logprobs = [float(x) for x in trans]
                            except Exception:
                                token_logprobs = None
            except Exception:
                token_logprobs = None
        if self.cache is not None:
            try:
                self.cache.put(
                    canonical_key({"url": url, **payload}),
                    request=payload,
                    response=data if isinstance(data, dict) else {"data": data},
                    content=text,
                    token_logprobs=token_logprobs,
                    usage_prompt_tokens=None,
                    usage_completion_tokens=None,
                    duration_ms=0,
                )
            except Exception:
                pass
        return text, token_logprobs

