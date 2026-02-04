from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import os


@dataclass(frozen=True)
class Settings:
    model_name: str
    ollama_host: str
    ollama_url: str


_ENV_MODEL_NAME: Final[str] = "MODEL_NAME"
_ENV_OLLAMA_HOST: Final[str] = "OLLAMA_HOST"
_ENV_OLLAMA_URL: Final[str] = "OLLAMA_URL"


def _parse_dotenv(path: Path) -> dict[str, str]:
    """
    Minimal .env parser:
    - KEY=VALUE pairs
    - ignores blanks and lines starting with '#'
    - strips surrounding quotes on VALUE
    """
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k:
            data[k] = v
    return data


def _default_dotenv_path() -> Path:
    # Prefer current working directory (repo root when invoked from there).
    return Path.cwd() / ".env"


def load_settings(dotenv_path: Path | None = None) -> Settings:
    """
    Load settings from environment variables, falling back to `.env`.

    Security: this function never prints the dotenv content.
    """
    dotenv_path = dotenv_path or _default_dotenv_path()
    dotenv = _parse_dotenv(dotenv_path)

    def get(name: str) -> str | None:
        return os.environ.get(name) or dotenv.get(name)

    model_name = get(_ENV_MODEL_NAME) or ""
    ollama_url = get(_ENV_OLLAMA_URL) or ""
    ollama_host = get(_ENV_OLLAMA_HOST) or ""

    if not ollama_host and ollama_url:
        p = urlparse(ollama_url)
        if p.scheme and p.netloc:
            ollama_host = f"{p.scheme}://{p.netloc}"

    missing = [k for k, v in [(_ENV_MODEL_NAME, model_name), (_ENV_OLLAMA_URL, ollama_url)] if not v]
    if missing:
        raise RuntimeError(
            "Missing required settings: "
            + ", ".join(missing)
            + ". Provide them via environment variables or a .env file."
        )

    return Settings(model_name=model_name, ollama_host=ollama_host, ollama_url=ollama_url)

