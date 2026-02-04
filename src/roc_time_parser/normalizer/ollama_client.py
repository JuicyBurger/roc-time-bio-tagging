from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class OllamaResult:
    ok: bool
    response: str
    raw: dict[str, Any] | None = None
    error: str | None = None


def query_ollama(
    prompt: str,
    *,
    model_name: str,
    ollama_url: str,
    timeout_s: int = 60,
    temperature: float = 0.0,
) -> OllamaResult:
    """
    Query Ollama /api/generate (non-streaming) and return the 'response' text.

    Compatible with user-provided .env:
      OLLAMA_URL=http://host:11434/api/generate
    """
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(ollama_url, json=payload, timeout=timeout_s)
        if resp.status_code != 200:
            return OllamaResult(ok=False, response="", raw=None, error=f"HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        return OllamaResult(ok=True, response=str(data.get("response", "")), raw=data, error=None)
    except Exception as e:  # noqa: BLE001
        return OllamaResult(ok=False, response="", raw=None, error=str(e))

