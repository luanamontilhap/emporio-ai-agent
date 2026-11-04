"""Runtime configuration read from environment variables (optionally from a .env file).

The LLM layer speaks the OpenAI Chat Completions protocol. Pointing ``LLM_BASE_URL``
at a different server (a hosted API or a local one such as Ollama) is the only
change needed to switch providers.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import urlparse

DEFAULT_BASE_URL = "https://api.openai.com/v1"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}
LOCAL_API_KEY_PLACEHOLDER = "not-needed"  # the SDK requires a value; local servers ignore it
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when the environment configuration is missing or invalid."""


def _text(env: Mapping[str, str], name: str) -> str | None:
    value = (env.get(name) or "").strip()
    return value or None


def _number(env: Mapping[str, str], name: str, default, cast: Callable, minimum, maximum):
    raw = _text(env, name)
    if raw is None:
        return default
    try:
        value = cast(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a valid {cast.__name__}, got {raw!r}.") from None
    if not minimum <= value <= maximum:  # also rejects NaN
        raise ConfigError(f"{name} must be between {minimum} and {maximum}, got {raw}.")
    return value


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    temperature: float | None  # None -> parameter omitted (some models reject it)
    timeout_seconds: float
    max_retries: int
    max_tool_rounds: int
    max_validation_retries: int
    max_history_messages: int
    rag_top_k: int
    rag_min_score: float

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        load_dotenv_file: bool = True,
        dotenv_path: str | Path | None = None,
    ) -> "Settings":
        """Build settings from ``env`` (defaults to ``os.environ`` plus the project's .env).

        Variables already present in the real environment win over the .env file.
        """
        if env is None:
            if load_dotenv_file:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path or PROJECT_ROOT / ".env", override=False)
            env = os.environ

        base_url = _text(env, "LLM_BASE_URL") or DEFAULT_BASE_URL
        host = (urlparse(base_url).hostname or "").casefold()
        if not host:
            raise ConfigError(f"LLM_BASE_URL is not a valid URL: {base_url!r}.")

        model = _text(env, "LLM_MODEL")
        if model is None:
            raise ConfigError(
                "LLM_MODEL is not set. Set it to the model name served by LLM_BASE_URL "
                "(see .env.example)."
            )

        api_key = _text(env, "LLM_API_KEY")
        if api_key is None:
            if host not in LOCAL_HOSTS:
                raise ConfigError(f"LLM_API_KEY is required for {host}.")
            api_key = LOCAL_API_KEY_PLACEHOLDER

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=_number(env, "LLM_TEMPERATURE", None, float, 0.0, 2.0),
            timeout_seconds=_number(env, "LLM_TIMEOUT_SECONDS", 60.0, float, 1.0, 600.0),
            max_retries=_number(env, "LLM_MAX_RETRIES", 2, int, 0, 5),
            max_tool_rounds=_number(env, "AGENT_MAX_TOOL_ROUNDS", 6, int, 1, 12),
            max_validation_retries=_number(env, "AGENT_MAX_VALIDATION_RETRIES", 1, int, 0, 3),
            max_history_messages=_number(env, "AGENT_MAX_HISTORY_MESSAGES", 12, int, 0, 100),
            rag_top_k=_number(env, "RAG_TOP_K", 4, int, 1, 10),
            rag_min_score=_number(env, "RAG_MIN_SCORE", 0.20, float, 0.0, 1.0),
        )
