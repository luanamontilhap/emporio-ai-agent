from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from emporio_agent.validation import collect_evidence, parse_number, validate_answer


def codes(text, numbers=(), pages=()):
    return [v.code for v in validate_answer(text, set(numbers), set(pages))]


@pytest.mark.parametrize("token, value", [
    ("1.299,00", 1299.0), ("323,10", 323.1), ("12.5", 12.5), ("2.000", 2000.0), ("7", 7.0), ("1.234.567,89", 1234567.89),
])
def test_parse_number_handles_brazilian_formats(token, value):
    assert parse_number(token) == value


def test_collect_evidence_reads_numbers_pages_and_formatted_strings():
    numbers, pages = collect_evidence({
        "price_brl": 359.0, "price_formatted": "R$ 1.299,00", "flag": True,
        "passages": [{"page": 4, "text": "devolução em até 7 (sete) dias corridos; frete R$ 35,00"}],
    })
    assert {359.0, 1299.0, 7.0, 35.0, 4.0} <= numbers
    assert True not in numbers and 1.0 not in numbers  # booleans are not numbers
    assert pages == {4}


def test_answer_with_supported_figures_is_accepted():
    text = "Custa R$ 1.299,00 (10% off), em até 12x, entrega em 3 dias úteis. Pedido 1. Manual, p. 4."
    assert codes(text, numbers={1299.0, 10.0, 12.0, 3.0, 1.0}, pages={4}) == []


@pytest.mark.parametrize("text, expected", [
    ("Custa R$ 999,00.", "ungrounded_money"),
    ("Tem 20% de desconto.", "ungrounded_percent"),
    ("Você tem 30 dias para trocar.", "ungrounded_quantity"),
    ("Parcelamos em 12 (doze) parcelas.", "ungrounded_quantity"),
    ("Dá para pagar em 6x.", "ungrounded_quantity"),
    ("Restam 4 unidades em estoque.", "ungrounded_quantity"),
    ("O pedido 77 já foi enviado.", "ungrounded_identifier"),
    ("Segundo o manual, página 5.", "ungrounded_citation"),
    ("Chamei get_order_details para você.", "internal_leak"),
    ("   ", "empty_answer"),
])
def test_unsupported_claims_are_flagged(text, expected):
    assert expected in codes(text, numbers={1.0, 2.0}, pages={4})


def test_the_same_invented_value_is_reported_once():
    assert codes("R$ 999,00 hoje, repito: R$ 999,00.") == ["ungrounded_money"]


def test_prose_without_figures_and_model_names_are_not_flagged():
    text = "O Kalani KAL-700T e o Martin D-28 são ótimas opções; me diga seu orçamento que eu busco."
    assert codes(text) == []


def test_numbers_typed_by_the_customer_do_not_count_as_evidence():
    # The validator only sees tool/RAG evidence; a figure echoed from the question is still unsupported.
    assert codes("Sim, o desconto é de 30%.", numbers=set()) == ["ungrounded_percent"]


def test_float_rounding_does_not_cause_false_rejections():
    numbers, _ = collect_evidence({"promotional_price_brl": 323.1})
    assert codes("Por R$ 323,10", numbers=numbers) == []
