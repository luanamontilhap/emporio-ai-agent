import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from emporio_agent.config import ConfigError, Settings


def load(**env):
    return Settings.from_env(env, load_dotenv_file=False)


def test_local_server_needs_only_url_and_model():
    s = load(LLM_BASE_URL="http://localhost:11434/v1", LLM_MODEL="llama3.1")
    assert (s.base_url, s.model, s.api_key) == ("http://localhost:11434/v1", "llama3.1", "not-needed")
    assert s.temperature is None and s.max_tool_rounds == 6 and s.rag_top_k == 4 and s.rag_min_score == 0.20


def test_hosted_api_requires_a_key_and_defaults_to_openai_url():
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        load(LLM_MODEL="some-model")
    s = load(LLM_MODEL="some-model", LLM_API_KEY="sk-test")
    assert s.base_url == "https://api.openai.com/v1" and s.api_key == "sk-test"


def test_model_has_no_default_so_it_cannot_go_stale():
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        load(LLM_BASE_URL="http://localhost:11434/v1")


@pytest.mark.parametrize("name, value", [
    ("LLM_TEMPERATURE", "hot"), ("LLM_TEMPERATURE", "3"), ("LLM_TIMEOUT_SECONDS", "0"),
    ("AGENT_MAX_TOOL_ROUNDS", "0"), ("AGENT_MAX_TOOL_ROUNDS", "2.5"), ("RAG_TOP_K", "11"),
    ("RAG_MIN_SCORE", "nan"), ("RAG_MIN_SCORE", "1.5"),
])
def test_invalid_numbers_name_the_offending_variable(name, value):
    with pytest.raises(ConfigError, match=name):
        load(LLM_BASE_URL="http://localhost:11434/v1", LLM_MODEL="m", **{name: value})


def test_overrides_are_parsed():
    s = load(LLM_BASE_URL="http://localhost:1/v1", LLM_MODEL="m", LLM_TEMPERATURE="0.1",
             AGENT_MAX_TOOL_ROUNDS="3", RAG_MIN_SCORE="0.35", AGENT_MAX_VALIDATION_RETRIES="0")
    assert (s.temperature, s.max_tool_rounds, s.rag_min_score, s.max_validation_retries) == (0.1, 3, 0.35, 0)


def test_blank_values_fall_back_to_defaults():
    s = load(LLM_BASE_URL="http://localhost:1/v1", LLM_MODEL="m", LLM_TEMPERATURE="  ", RAG_TOP_K="")
    assert s.temperature is None and s.rag_top_k == 4


def test_invalid_url_is_rejected():
    with pytest.raises(ConfigError, match="LLM_BASE_URL"):
        load(LLM_BASE_URL="not a url", LLM_MODEL="m", LLM_API_KEY="k")


def test_dotenv_file_is_read_but_real_environment_wins(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_BASE_URL=http://localhost:11434/v1\nLLM_MODEL=from-file\nRAG_TOP_K=7\n")
    monkeypatch.setattr(os, "environ", {"RAG_TOP_K": "2"})  # isolated, so the real env is not polluted

    s = Settings.from_env(dotenv_path=env_file)

    assert s.model == "from-file" and s.rag_top_k == 2
