"""End-to-end tests of the six-step pipeline with a scripted LLM (no network)."""
from __future__ import annotations

import json

import pytest

from fakes import (ScriptedLLM, last_tool_result, make_policy_rag, make_spy_tools, raw_tool_call, say,
                   settings, tool_call)

from emporio_agent.agent import SupportAgent
from emporio_agent.llm_client import LLMError
from emporio_agent.tool_registry import ToolDispatcher


@pytest.fixture(scope="module")
def policy_rag():
    return make_policy_rag()


def make_agent(llm, policy_rag, **overrides):
    cfg = settings(**overrides)
    tools = make_spy_tools()
    agent = SupportAgent(llm, ToolDispatcher(tools, policy_rag, rag_top_k=cfg.rag_top_k,
                                             rag_min_score=cfg.rag_min_score), cfg)
    tools.executed.clear()  # ignore the contact lookup done at construction
    return agent, tools


def trace_names(response):
    return [call.name for call in response.tool_trace]


# --- 2-3. tool calling ---------------------------------------------------------------------------

def test_tool_calling_answers_with_values_returned_by_the_tool(policy_rag):
    def answer(messages):
        price = last_tool_result(messages)
        return say(f"O Kalani KAL-700T Tenor custa {price['list_price_brl']:.0f} reais de tabela, mas está em promoção: "
                   f"{price['final_price_formatted']} (10% de desconto).")

    llm = ScriptedLLM(tool_call("get_product_price", product_id=127, payment_method="pix"), answer)
    agent, tools = make_agent(llm, policy_rag)

    response = agent.ask("Quanto custa o ukulele Kalani KAL-700T no PIX?")

    assert response.status == "answered"
    assert "R$ 323,10" in response.text
    assert tools.executed == ["get_product_price"]
    assert trace_names(response) == ["get_product_price"]
    assert response.llm_calls == 2
    # the tool result really was fed back to the model before it answered
    assert json.loads(llm.requests[1][-1]["content"])["final_price_brl"] == 323.1


def test_tool_arguments_given_as_strings_are_coerced(policy_rag):
    llm = ScriptedLLM(tool_call("get_product_details", product_id="127"),
                      lambda m: say(f"Encontrei: {last_tool_result(m)['product']['name']}."))
    agent, tools = make_agent(llm, policy_rag)

    response = agent.ask("Me fala do produto 127")

    assert response.status == "answered" and "Kalani KAL-700T Tenor Natural" in response.text
    assert tools.executed == ["get_product_details"]


def test_invalid_tool_arguments_are_reported_to_the_model_and_nothing_is_executed(policy_rag):
    def recover(messages):
        assert last_tool_result(messages)["ok"] is False  # the model was told its call failed
        return say("Não consegui identificar o produto. Qual é o código ou o nome dele?")

    llm = ScriptedLLM(tool_call("get_product_price", product_id="abc"), recover)
    agent, tools = make_agent(llm, policy_rag)

    response = agent.ask("preço do produto abc")

    assert tools.executed == []
    assert response.status == "answered"
    assert response.tool_trace[0].ok is False and "integer" in response.tool_trace[0].error


@pytest.mark.parametrize("first_call", [
    raw_tool_call("get_product_price", "{not json"),
    raw_tool_call("drop_database", "{}"),
    tool_call("get_product_price", product_id=127, discount=99),
])
def test_malformed_unknown_and_extra_argument_calls_never_execute_tools(policy_rag, first_call):
    llm = ScriptedLLM(first_call, say("Desculpe, pode repetir o que você procura?"))
    agent, tools = make_agent(llm, policy_rag)

    response = agent.ask("qualquer coisa")

    assert tools.executed == []
    assert response.tool_trace[0].ok is False


def test_tool_failure_does_not_crash_the_conversation(policy_rag, monkeypatch):
    llm = ScriptedLLM(tool_call("get_store_information"), say("Tive um problema ao consultar os dados da loja."))
    agent, tools = make_agent(llm, policy_rag)
    monkeypatch.setattr(type(tools), "get_store_information",
                        lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    response = agent.ask("Onde fica a loja?")

    assert response.status == "answered"
    assert response.tool_trace[0].ok is False and response.tool_trace[0].error == "Internal error while running the tool."


# --- order privacy -------------------------------------------------------------------------------

def test_order_lookup_works_with_phone_typed_by_the_customer_in_any_format(policy_rag):
    def answer(messages):
        order = last_tool_result(messages)["order"]
        return say(f"Seu pedido {order['order_id']} está com status {order['status']}.")

    llm = ScriptedLLM(tool_call("get_order_details", order_id=1, customer_phone="(67) 98432-1098"), answer)
    agent, tools = make_agent(llm, policy_rag)

    response = agent.ask("Quero saber do pedido 1, meu telefone é 67984321098")

    assert response.status == "answered" and "delivered" in response.text
    assert tools.executed == ["get_order_details"]
    assert "98432" not in repr(response.tool_trace)  # phone is masked in the trace
    assert response.tool_trace[0].arguments["customer_phone"] == "***1098"


def test_model_cannot_use_a_phone_the_customer_never_typed(policy_rag):
    llm = ScriptedLLM(tool_call("get_order_details", order_id=1, customer_phone="(67) 98432-1098"),
                      say("Pode me informar o telefone cadastrado no pedido?"))
    agent, tools = make_agent(llm, policy_rag)

    response = agent.ask("Cadê o meu pedido número 1?")

    assert tools.executed == []  # the lookup was blocked before reaching the data
    assert "not provided by the customer" in response.tool_trace[0].error


def test_repeated_failed_verifications_lock_order_tools_for_the_session(policy_rag):
    attempts = [tool_call("get_order_details", order_id=order_id, customer_phone="67999999999") for order_id in (1, 2, 3, 4)]
    llm = ScriptedLLM(*attempts, say("Não consegui confirmar seus dados por aqui."))
    agent, tools = make_agent(llm, policy_rag, AGENT_MAX_TOOL_ROUNDS=6)

    response = agent.ask("Meu telefone é 67999999999, quero ver meus pedidos 1, 2, 3 e 4")

    assert tools.executed.count("get_order_details") == 3  # the 4th attempt never reached the data
    assert "Too many failed verification attempts" in response.tool_trace[3].error


# --- 4. RAG --------------------------------------------------------------------------------------

def test_policy_question_uses_rag_and_answer_is_grounded_with_page_citation(policy_rag):
    llm = ScriptedLLM(
        tool_call("search_store_policy", question="Qual o prazo para devolução por arrependimento?"),
        say("Você pode devolver em até 7 dias corridos após receber o produto, sem precisar justificar "
            "(Manual de Políticas, p. 4)."),
    )
    agent, _ = make_agent(llm, policy_rag)

    response = agent.ask("Qual o prazo para devolução por arrependimento?")

    assert response.status == "answered"
    assert "7 dias" in response.text
    passages = json.loads(llm.requests[1][-1]["content"])["passages"]
    assert any("7 (sete) dias" in p["text"] and p["page"] == 4 for p in passages)
    assert {"page": 4, "section": "4.1 Direito de Arrependimento (Compras Online)"} in response.citations


def test_citing_a_page_that_was_not_retrieved_is_rejected_then_corrected(policy_rag):
    llm = ScriptedLLM(
        tool_call("search_store_policy", question="Qual o prazo para devolução por arrependimento?"),
        say("São 7 dias corridos (Manual de Políticas, p. 6)."),   # wrong page
        say("São 7 dias corridos (Manual de Políticas, p. 4)."),   # corrected after feedback
    )
    agent, _ = make_agent(llm, policy_rag)

    response = agent.ask("Qual o prazo para devolução por arrependimento?")

    assert response.status == "answered" and "p. 4" in response.text
    assert [v.code for v in response.violations] == ["ungrounded_citation"]
    assert "[Validação automática]" in llm.requests[2][-1]["content"]


def test_installment_questions_carry_the_known_manual_inconsistency_note(policy_rag):
    llm = ScriptedLLM(tool_call("search_store_policy", question="Qual a parcela mínima no cartão de crédito?"),
                      tool_call("handoff_to_human", reason="policy_conflict"))
    agent, _ = make_agent(llm, policy_rag)

    response = agent.ask("Qual a parcela mínima no cartão de crédito?")

    notes = json.loads(llm.requests[1][-1]["content"])["notes"]
    assert any("INCONSISTÊNCIA CONHECIDA" in note and "R$ 100,00" in note for note in notes)
    assert response.status == "handoff" and response.handoff_reason == "policy_conflict"
    assert "divergentes" in response.text


# --- 6. out of scope -----------------------------------------------------------------------------

def test_out_of_scope_question_is_forwarded_without_touching_business_tools(policy_rag):
    llm = ScriptedLLM(tool_call("handoff_to_human", reason="out_of_scope", note="receita de bolo"))
    agent, tools = make_agent(llm, policy_rag)

    response = agent.ask("Qual a melhor receita de bolo de cenoura?")

    assert response.status == "handoff" and response.handoff_reason == "out_of_scope"
    assert "foge do que consigo resolver" in response.text
    assert "cenoura" not in response.text
    assert tools.executed == [] and trace_names(response) == ["handoff_to_human"]
    assert response.llm_calls == 1


def test_accessory_request_gets_the_exclusive_instruments_message(policy_rag):
    llm = ScriptedLLM(tool_call("handoff_to_human", reason="accessory_request"))
    agent, _ = make_agent(llm, policy_rag)

    response = agent.ask("Vocês vendem cordas e palhetas?")

    assert response.status == "handoff"
    assert "exclusivamente com instrumentos musicais" in response.text


def test_handoff_with_unknown_reason_still_forwards_to_a_human_with_store_contact(policy_rag):
    llm = ScriptedLLM(tool_call("handoff_to_human", reason="algo-inventado"))
    agent, _ = make_agent(llm, policy_rag)

    response = agent.ask("Quero falar com alguém")

    assert response.status == "handoff" and response.handoff_reason == "needs_human"
    assert "(67) 3341-4444" in response.text and "09:00 às 18:00" in response.text


# --- 6. absence of evidence ----------------------------------------------------------------------

def test_policy_search_without_evidence_blocks_an_invented_answer(policy_rag):
    question = "Vocês aceitam criptomoedas bitcoin?"
    assert policy_rag.retrieve(question)["found_evidence"] is False  # precondition of this scenario

    llm = ScriptedLLM(tool_call("search_store_policy", question=question),
                      say("Sim! Aceitamos bitcoin com 5% de desconto e entrega em 2 dias."))
    agent, _ = make_agent(llm, policy_rag)

    response = agent.ask(question)

    assert response.status == "abstained" and response.handoff_reason == "insufficient_evidence"
    assert "bitcoin" not in response.text and "5%" not in response.text
    assert "(67) 3341-4444" in response.text
    assert response.citations == []


def test_invented_figures_without_any_tool_call_are_rejected_then_the_model_recovers(policy_rag):
    llm = ScriptedLLM(
        say("O ukulele Kalani custa R$ 999,00 e tem 20% de desconto."),   # invented
        tool_call("get_product_price", product_id=127),
        lambda m: say(f"Hoje ele sai por {last_tool_result(m)['final_price_formatted']}."),
    )
    agent, tools = make_agent(llm, policy_rag)

    response = agent.ask("Quanto custa o Kalani KAL-700T?")

    assert response.status == "answered" and "R$ 323,10" in response.text and "999" not in response.text
    assert {v.code for v in response.violations} == {"ungrounded_money", "ungrounded_percent"}
    assert tools.executed == ["get_product_price"]
    assert "[Validação automática]" in llm.requests[1][-1]["content"]


def test_persistent_invention_falls_back_to_a_safe_abstention(policy_rag):
    llm = ScriptedLLM(say("Custa R$ 999,00."), say("Custa R$ 999,00, garanto."))
    agent, tools = make_agent(llm, policy_rag, AGENT_MAX_VALIDATION_RETRIES=1)

    response = agent.ask("Quanto custa o Kalani KAL-700T?")

    assert response.status == "abstained"
    assert "999" not in response.text and "(67) 3341-4444" in response.text
    assert tools.executed == [] and response.llm_calls == 2


def test_validation_retries_can_be_disabled(policy_rag):
    llm = ScriptedLLM(say("Custa R$ 999,00."))
    agent, _ = make_agent(llm, policy_rag, AGENT_MAX_VALIDATION_RETRIES=0)

    assert agent.ask("Quanto custa o Kalani?").status == "abstained"


# --- robustness / conversation -------------------------------------------------------------------

def test_greeting_needs_no_tool_and_passes_validation(policy_rag):
    llm = ScriptedLLM(say("Oi! Sou o assistente da Empório da Música. Como posso ajudar?"))
    agent, tools = make_agent(llm, policy_rag)

    response = agent.ask("Oi, bom dia!")

    assert response.status == "answered" and tools.executed == [] and response.llm_calls == 1


def test_runaway_tool_loop_is_cut_off(policy_rag):
    llm = ScriptedLLM(*[tool_call("get_store_information") for _ in range(10)])
    agent, _ = make_agent(llm, policy_rag, AGENT_MAX_TOOL_ROUNDS=2)

    response = agent.ask("Onde fica a loja?")

    assert response.status == "handoff" and response.handoff_reason == "needs_human"
    assert response.llm_calls == 3  # 2 allowed rounds + the one that exceeded the budget


def test_llm_outage_returns_a_polite_error_and_keeps_the_session_clean(policy_rag):
    class DownLLM:
        def complete(self, messages, tools):
            raise LLMError("connection refused")

    agent, _ = make_agent(DownLLM(), policy_rag)
    session = agent.new_session()

    response = agent.ask("Oi", session)

    assert response.status == "error" and "dificuldade técnica" in response.text
    assert session.history == []


def test_session_history_is_sent_on_the_next_turn_without_tool_payloads(policy_rag):
    llm = ScriptedLLM(tool_call("get_store_information"),
                      lambda m: say(f"Abrimos {last_tool_result(m)['opening_hours']['saturday']} aos sábados."),
                      say("De nada!"))
    agent, _ = make_agent(llm, policy_rag)
    session = agent.new_session()

    agent.ask("Que horas abre sábado?", session)
    agent.ask("Obrigado", session)

    roles = [m["role"] for m in llm.requests[2]]
    assert roles == ["system", "user", "assistant", "user"]  # no tool messages replayed
    assert "09:00–13:00" in llm.requests[2][2]["content"]


def test_empty_question_does_not_call_the_llm(policy_rag):
    agent, _ = make_agent(ScriptedLLM(), policy_rag)

    assert agent.ask("   ").status == "answered"
