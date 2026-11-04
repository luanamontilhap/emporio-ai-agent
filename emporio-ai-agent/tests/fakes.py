"""Test doubles: no network, no model download."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emporio_agent.config import Settings
from emporio_agent.data_loader import StoreData
from emporio_agent.llm_client import AssistantMessage, ToolCall
from emporio_agent.policy_rag import PolicyRAG
from emporio_agent.tools import StoreTools

_ids = itertools.count(1)


def settings(**overrides) -> Settings:
    env = {"LLM_BASE_URL": "http://localhost:11434/v1", "LLM_MODEL": "test-model"}
    env.update({key: str(value) for key, value in overrides.items()})
    return Settings.from_env(env)


def tool_call(name: str, **arguments) -> AssistantMessage:
    return AssistantMessage(tool_calls=(ToolCall(f"call_{next(_ids)}", name, json.dumps(arguments)),))


def raw_tool_call(name: str, raw_arguments: str) -> AssistantMessage:
    return AssistantMessage(tool_calls=(ToolCall(f"call_{next(_ids)}", name, raw_arguments),))


def say(text: str) -> AssistantMessage:
    return AssistantMessage(content=text)


def last_tool_result(messages: list[dict]) -> dict:
    tool_messages = [m for m in messages if m["role"] == "tool"]
    return json.loads(tool_messages[-1]["content"])


class ScriptedLLM:
    """Replays a fixed list of replies. A step may be a callable(messages) -> AssistantMessage,
    which lets a test build its answer from the real tool output it just received."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests: list[list[dict]] = []

    def complete(self, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        if not self.steps:
            raise AssertionError("The agent called the LLM more times than the test scripted.")
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step


class FakeEncoder:
    """Tiny deterministic encoder (same idea as tests/test_policy_rag.py)."""

    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            text = sentence.casefold()
            vectors.append([
                float("pix" in text or "pagamento" in text),
                float("devolução" in text or "arrependimento" in text),
                float("garantia" in text or "defeito" in text),
                float("cobre" in text),
            ])
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


def make_policy_rag() -> PolicyRAG:
    return PolicyRAG(PROJECT_ROOT / "documents" / "politicas_da_loja.pdf",
                     encoder=FakeEncoder(), chunk_size=800, chunk_overlap=100)


class SpyTools(StoreTools):
    """Real tools that record which functions the agent executed (top-level calls only)."""

    def __init__(self, data):
        super().__init__(data)
        self.executed: list[str] = []
        self._depth = 0

    def __getattribute__(self, name):
        attribute = super().__getattribute__(name)
        if name.startswith(("search_", "get_")) and callable(attribute):
            def recorded(*args, **kwargs):
                if self._depth == 0:
                    self.executed.append(name)
                self._depth += 1
                try:
                    return attribute(*args, **kwargs)
                finally:
                    self._depth -= 1
            return recorded
        return attribute


def make_spy_tools() -> SpyTools:
    return SpyTools(StoreData(PROJECT_ROOT / "data"))
