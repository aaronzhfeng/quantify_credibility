from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def ensure_db(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS completions (
              key TEXT PRIMARY KEY,
              request_json TEXT NOT NULL,
              response_json TEXT NOT NULL,
              content TEXT NOT NULL,
              token_logprobs_json TEXT,
              usage_prompt_tokens INTEGER,
              usage_completion_tokens INTEGER,
              duration_ms INTEGER,
              created_at TEXT
            )
            """
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
    finally:
        conn.close()


def canonical_key(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass
class CacheResult:
    hit: bool
    content: Optional[str]
    token_logprobs: Optional[list]
    usage_prompt_tokens: Optional[int]
    usage_completion_tokens: Optional[int]
    raw_response: Optional[Dict[str, Any]]


class SQLiteCache:
    def __init__(self, path: str, mode: str = "readwrite") -> None:
        self.path = path
        self.mode = mode  # readwrite|read|write|off
        ensure_db(path)

    def get(self, key: str) -> CacheResult:
        if self.mode in ("write", "off"):
            return CacheResult(False, None, None, None, None, None)
        conn = sqlite3.connect(self.path)
        try:
            cur = conn.execute("SELECT content, token_logprobs_json, response_json, usage_prompt_tokens, usage_completion_tokens FROM completions WHERE key=?", (key,))
            row = cur.fetchone()
            if not row:
                return CacheResult(False, None, None, None, None, None)
            content = row[0]
            token_lps = json.loads(row[1]) if row[1] else None
            resp = json.loads(row[2]) if row[2] else None
            return CacheResult(True, content, token_lps, row[3], row[4], resp)
        finally:
            conn.close()

    def put(
        self,
        key: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        content: str,
        token_logprobs: Optional[list],
        usage_prompt_tokens: Optional[int],
        usage_completion_tokens: Optional[int],
        duration_ms: int,
    ) -> None:
        if self.mode in ("read", "off"):
            return
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO completions
                (key, request_json, response_json, content, token_logprobs_json, usage_prompt_tokens, usage_completion_tokens, duration_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    key,
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(response, ensure_ascii=False),
                    content,
                    json.dumps(token_logprobs) if token_logprobs is not None else None,
                    usage_prompt_tokens,
                    usage_completion_tokens,
                    duration_ms,
                ),
            )
            conn.commit()
        finally:
            conn.close()


