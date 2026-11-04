import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from fakes import make_policy_rag, make_spy_tools, tool_call
from emporio_agent.session import SessionState
from emporio_agent.tool_registry import (HANDOFF_TOOL, TOOL_DEFINITIONS, TOOL_SCHEMAS, ToolDispatcher, TurnContext,
                                         has_positive_evidence, validate_arguments)


@pytest.fixture(scope="module")
def dispatcher():
    return ToolDispatcher(make_spy_tools(), make_policy_rag())


def run(dispatcher, message, *, user_messages=("pergunta",)):
    call = message.tool_calls[0]
    return dispatcher.execute(call, TurnContext(SessionState(), list(user_messages)))


def test_every_schema_is_well_formed_and_has_a_handler(dispatcher):
    names = [item["function"]["name"] for item in TOOL_DEFINITIONS]
    assert len(names) == len(set(names))
    for item in TOOL_DEFINITIONS:
        params = item["function"]["parameters"]
        assert item["type"] == "function" and params["type"] == "object" and params["additionalProperties"] is False
        assert set(params["required"]) <= set(params["properties"])
    assert set(names) - {HANDOFF_TOOL} == set(dispatcher._handlers)


@pytest.mark.parametrize("tool, args, ok", [
    ("search_products", {"query": "violão", "limit": 5}, True),
    ("search_products", {"limit": 0}, False),
    ("search_products", {"limit": 21}, False),
    ("search_products", {"max_price_brl": -1}, False),
    ("search_products", {"available_only": "false"}, True),
    ("get_product_price", {"product_id": 127, "payment_method": "PIX"}, True),
    ("get_product_price", {"product_id": 127, "payment_method": "bitcoin"}, False),
    ("get_product_price", {"product_id": True}, False),
    ("get_product_price", {}, False),
    ("get_store_information", {"x": 1}, False),
    ("search_store_policy", {"question": "ab"}, False),
])
def test_argument_validation_rules(tool, args, ok):
    cleaned, error = validate_arguments(TOOL_SCHEMAS[tool], args)
    assert (error is None) is ok
    assert (cleaned is not None) is ok


def test_validation_normalizes_values():
    cleaned, _ = validate_arguments(TOOL_SCHEMAS["get_product_price"], {"product_id": "127", "payment_method": " Pix "})
    assert cleaned == {"product_id": 127, "payment_method": "pix"}


def test_tool_results_are_strictly_json_serializable(dispatcher):
    for message in (tool_call("get_product_price", product_id=127), tool_call("get_product_details", product_id=127),
                    tool_call("search_products", query="Kalani"), tool_call("get_store_information")):
        outcome = run(dispatcher, message)
        assert outcome.result["ok"]
        json.loads(outcome.content)  # raises if numpy scalars leaked


def test_policy_tool_returns_passages_with_page_and_section(dispatcher):
    outcome = run(dispatcher, tool_call("search_store_policy", question="garantia contra defeitos de fabricação"))
    result = outcome.result
    assert result["ok"] and result["found_evidence"]
    assert all({"page", "section", "text", "relevance"} <= set(p) for p in result["passages"])


def test_contact_note_only_triggers_on_the_question_not_on_incidental_evidence(dispatcher):
    delivery = run(dispatcher, tool_call("search_store_policy", question="prazo de entrega em Campo Grande")).result
    assert "notes" not in delivery  # the delivery passage mentions 'telefone', which must not trigger the note
    contact = run(dispatcher, tool_call("search_store_policy", question="qual o whatsapp da loja?")).result
    assert any("3321-4500" in note for note in contact["notes"])


def test_handoff_tool_normalizes_reason_and_hides_nothing_unsafe(dispatcher):
    outcome = run(dispatcher, tool_call(HANDOFF_TOOL, reason="out_of_scope", note="x" * 1000))
    assert outcome.handoff_reason == "out_of_scope" and len(outcome.trace.arguments["note"]) == 300


def test_phone_guard_requires_the_digits_to_come_from_the_customer(dispatcher):
    guessed = run(dispatcher, tool_call("get_customer_orders", customer_phone="(67) 98432-1098"),
                  user_messages=("quero ver meus pedidos",))
    assert not guessed.result["ok"] and "not provided" in guessed.result["error"]
    incomplete = run(dispatcher, tool_call("get_customer_orders", customer_phone="98432"),
                     user_messages=("meu telefone é 98432",))
    assert "incomplete" in incomplete.result["error"]
    typed = run(dispatcher, tool_call("get_customer_orders", customer_phone="(67) 98432-1098"),
                user_messages=("meu telefone é +55 (67) 98432-1098",))
    assert typed.result["ok"]


@pytest.mark.parametrize("tool, result, expected", [
    ("search_products", {"ok": True, "total_matches": 0}, False),
    ("search_products", {"ok": True, "total_matches": 2}, True),
    ("search_store_policy", {"ok": True, "found_evidence": False}, False),
    ("search_store_policy", {"ok": True, "found_evidence": True}, True),
    ("get_product_details", {"ok": False, "error": "not found"}, False),
    ("get_store_information", {"ok": True}, True),
    (HANDOFF_TOOL, {"ok": True}, False),
])
def test_positive_evidence_rules(tool, result, expected):
    assert has_positive_evidence(tool, result) is expected
