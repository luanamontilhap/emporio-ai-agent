"""Thin adapter over any OpenAI-compatible Chat Completions endpoint.

The rest of the agent only depends on ``ChatClient`` and the small dataclasses below,
so the provider (OpenAI, Ollama, vLLM, ...) can change without touching the pipeline,
and tests can inject a scripted fake instead of a network call.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from .config import Settings

# Some local reasoning models inline their chain of thought; it must never reach the
# customer or the validator.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    """The model endpoint failed or returned an unusable response."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text exactly as produced by the model (may be malformed)


@dataclass(frozen=True)
class AssistantMessage:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def to_message(self) -> dict:
        """Serialize in the Chat Completions message format for the next request."""
        message: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": call.id, "type": "function",
                 "function": {"name": call.name, "arguments": call.arguments}}
                for call in self.tool_calls
            ]
        return message


class ChatClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantMessage: ...


def strip_reasoning(text: str | None) -> str | None:
    if text is None:
        return None
    return _THINK_RE.sub("", text).strip()


class OpenAICompatibleClient:
    """Chat Completions client with native tool calling."""

    def __init__(self, settings: Settings, client=None):
        self._settings = settings
        if client is None:
            try:
                import openai
            except ImportError as exc:
                raise RuntimeError("The 'openai' package is missing. Install requirements.txt first.") from exc
            client = openai.OpenAI(
                base_url=settings.base_url,
                api_key=settings.api_key,
                timeout=settings.timeout_seconds,
                max_retries=settings.max_retries,
            )
        self._client = client

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantMessage:
        request: dict = {
            "model": self._settings.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        if self._settings.temperature is not None:
            request["temperature"] = self._settings.temperature
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:  # adapter boundary: SDK/network errors become LLMError
            raise LLMError(f"LLM request failed ({type(exc).__name__}): {exc}") from exc
        return self._parse(response)

    @staticmethod
    def _parse(response) -> AssistantMessage:
        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMError("The LLM response contained no choices.")
        message = choices[0].message
        calls = []
        for index, raw in enumerate(getattr(message, "tool_calls", None) or []):
            function = getattr(raw, "function", None)  # other tool-call kinds are ignored
            if function is None or not getattr(function, "name", None):
                continue
            calls.append(ToolCall(
                id=getattr(raw, "id", None) or f"call_{index}",
                name=function.name,
                arguments=function.arguments or "{}",
            ))
        return AssistantMessage(content=strip_reasoning(message.content), tool_calls=tuple(calls))
