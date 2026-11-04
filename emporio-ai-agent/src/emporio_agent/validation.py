"""Deterministic grounding checks applied to the model's final answer (pipeline step 6).

The rule is simple: every checkable factual claim in the answer (money, percentages,
quantities/deadlines/installments, order/product identifiers and manual page citations)
must also appear in what the tools or the policy retriever returned during the session.
A claim without support means the model invented or computed something, so the answer is
rejected instead of being shown to the customer.

Scope and limits: this catches fabricated *numbers and citations*. It cannot judge whether
a prose statement without figures is supported (that is handled by the system prompt and
the no-evidence gate in the agent), and it is intentionally lenient about *which* field a
number came from: any number present in the evidence counts as supported.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

# pt-BR numbers: "1.299,00" (thousands dot, decimal comma) or "12.5" / "12,5".
_NUMBER = r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?"
NUMBER_RE = re.compile(_NUMBER)
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?")

_MONEY_RE = re.compile(rf"R\$\s*({_NUMBER})", re.IGNORECASE)
_PERCENT_RE = re.compile(rf"({_NUMBER})\s*%")
_QUANTITY_RE = re.compile(
    rf"({_NUMBER})\s*(?:\([^)]{{1,25}}\)\s*)?"
    r"(?:dias?|horas?|semanas?|meses|m[eê]s|anos?|parcelas?|vezes|unidades?)\b",
    re.IGNORECASE,
)
_INSTALLMENT_X_RE = re.compile(r"\b(\d+)x\b", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(r"\b(?:pedido|produto|id)\s*(?:n[ºo°.]{0,2}\s*)?#?\s*(\d+)\b", re.IGNORECASE)
_CITATION_RE = re.compile(r"\b(?:p[áa]ginas?|p[áa]gs?\.?|pp?\.)\s*(\d+)", re.IGNORECASE)
_LEAK_RE = re.compile(r"\b(?:get_[a-z_]+|search_[a-z_]+|handoff_to_human)\b|\"ok\"\s*:", re.IGNORECASE)


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str  # written in Portuguese: it is fed back to the model on retry


def parse_number(token: str) -> float:
    if _THOUSANDS_RE.fullmatch(token):
        return float(token.replace(".", "").replace(",", "."))
    return float(token.replace(",", "."))


def collect_evidence(result: Any) -> tuple[set[float], set[int]]:
    """Gather every number (and every ``page`` value) found anywhere in a tool result."""
    numbers: set[float] = set()
    pages: set[int] = set()

    def walk(node: Any, key: str | None = None) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            numbers.add(round(float(node), 2))
            if key == "page" and isinstance(node, int):
                pages.add(node)
        elif isinstance(node, str):
            for token in NUMBER_RE.findall(node):
                numbers.add(round(parse_number(token), 2))
        elif isinstance(node, dict):
            for child_key, child in node.items():
                walk(child, child_key)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child, key)

    walk(result)
    return numbers, pages


def validate_answer(text: str, grounded_numbers: set[float], grounded_pages: set[int]) -> list[Violation]:
    """Return the list of problems found in ``text`` (empty list = answer accepted)."""
    if not text or not text.strip():
        return [Violation("empty_answer", "A resposta veio vazia.")]

    supported = {round(value, 2) for value in grounded_numbers}
    violations: list[Violation] = []
    seen: set[tuple[str, str]] = set()

    def check(code: str, regex: re.Pattern, describe) -> None:
        for match in regex.finditer(text):
            value = round(parse_number(match.group(1)), 2)
            key = (code, match.group(0).strip())
            if value not in supported and key not in seen:
                seen.add(key)
                violations.append(Violation(code, describe(match.group(0).strip())))

    check("ungrounded_money", _MONEY_RE,
          lambda raw: f"O valor \"{raw}\" não aparece nos resultados das ferramentas (não calcule nem estime valores).")
    check("ungrounded_percent", _PERCENT_RE,
          lambda raw: f"O percentual \"{raw}\" não aparece nos resultados das ferramentas.")
    check("ungrounded_quantity", _QUANTITY_RE,
          lambda raw: f"A quantidade/prazo \"{raw}\" não aparece nos resultados das ferramentas.")
    check("ungrounded_quantity", _INSTALLMENT_X_RE,
          lambda raw: f"O parcelamento \"{raw}\" não aparece nos resultados das ferramentas.")
    check("ungrounded_identifier", _IDENTIFIER_RE,
          lambda raw: f"O identificador \"{raw}\" não aparece nos resultados das ferramentas.")

    for match in _CITATION_RE.finditer(text):
        page = int(match.group(1))
        key = ("ungrounded_citation", match.group(0).strip())
        if page not in grounded_pages and key not in seen:
            seen.add(key)
            violations.append(Violation(
                "ungrounded_citation",
                f"A citação \"{match.group(0).strip()}\" não corresponde a nenhum trecho do manual recuperado.",
            ))

    if _LEAK_RE.search(text):
        violations.append(Violation(
            "internal_leak", "A resposta expõe nomes de ferramentas ou JSON interno ao cliente."))
    return violations
