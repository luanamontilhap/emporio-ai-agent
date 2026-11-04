"""Smoke tests against a REAL model. Skipped unless RUN_LIVE_LLM_TESTS=1.

    RUN_LIVE_LLM_TESTS=1 PYTHONPATH=src pytest tests/test_live_llm.py -q -s

Uses the real embedding model and whatever LLM_* variables are in the environment/.env
(hosted API or Ollama). Assertions check behavior (which tool ran, hand-off, grounding),
not exact wording, because model output varies.
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE_LLM_TESTS") != "1", reason="set RUN_LIVE_LLM_TESTS=1 to run")


@pytest.fixture(scope="module")
def agent():
    from emporio_agent.agent import build_agent
    from emporio_agent.config import Settings
    return build_agent(Settings.from_env())


def tools_used(response):
    return {call.name for call in response.tool_trace}


def test_price_question_calls_a_product_tool(agent):
    response = agent.ask("Quanto custa o ukulele Kalani KAL-700T se eu pagar no PIX?")
    assert response.status == "answered"
    assert tools_used(response) & {"get_product_price", "search_products", "get_product_details"}


def test_policy_question_uses_rag_and_is_grounded(agent):
    response = agent.ask("Qual o prazo para devolução por arrependimento numa compra online?")
    assert response.status == "answered" and "search_store_policy" in tools_used(response)
    assert "7" in response.text


def test_out_of_scope_is_forwarded(agent):
    response = agent.ask("Me explica como funciona a bolsa de valores?")
    assert response.status == "handoff" and response.handoff_reason == "out_of_scope"


def test_unanswerable_policy_question_does_not_invent(agent):
    response = agent.ask("Vocês aceitam pagamento em bitcoin?")
    assert response.status in {"abstained", "handoff"} or "bitcoin" not in response.text.casefold()
