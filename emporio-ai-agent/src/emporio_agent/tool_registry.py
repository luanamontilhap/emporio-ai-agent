"""Tool schemas exposed to the LLM and the controlled executor behind them.

The model never touches data directly. Every call is: parsed -> validated against its JSON
schema -> checked by privacy guards -> executed by a deterministic function -> serialized
back as JSON. Failures are returned to the model as ``{"ok": false, "error": ...}`` instead
of raising, so it can correct itself (for example, by asking the customer for a missing field).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import re
import time
from typing import Any, Callable, Protocol

from .llm_client import ToolCall
from .prompts import HANDOFF_REASONS
from .session import SessionState
from .tools import StoreTools

logger = logging.getLogger(__name__)

POLICY_TOOL = "search_store_policy"
HANDOFF_TOOL = "handoff_to_human"
ORDER_TOOLS = {"get_order_details", "get_customer_orders"}
MAX_FAILED_VERIFICATIONS = 3
MIN_PHONE_DIGITS = 10


class PolicyRetriever(Protocol):
    def retrieve(self, question: str, top_k: int = 4, min_score: float = 0.20) -> dict: ...


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required or [], "additionalProperties": False},
    }}


TOOL_DEFINITIONS: list[dict] = [
    _fn("search_products",
        "Busca produtos no catálogo da loja. Retorna preço, promoção ativa, estoque e disponibilidade. "
        "Use palavras-chave curtas (ex.: 'violão', 'Fender', 'ukulele'), não frases inteiras.",
        {"query": {"type": "string", "maxLength": 80,
                   "description": "Palavra-chave curta no nome, descrição ou especificações."},
         "category": {"type": "string", "maxLength": 60,
                      "description": "Nome da categoria (ex.: Guitarras, Baixos, Violões, Ukuleles, Teclados e Pianos)."},
         "max_price_brl": {"type": "number", "minimum": 0, "description": "Preço máximo em reais."},
         "available_only": {"type": "boolean", "description": "Se verdadeiro (padrão), só produtos à venda com estoque."},
         "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Máximo de resultados (padrão 5)."}}),
    _fn("get_product_details",
        "Detalhes de um produto pelo ID exato: nome, categoria, preço, status, estoque e promoção.",
        {"product_id": {"type": "integer", "minimum": 1}}, ["product_id"]),
    _fn("get_product_price",
        "Preço final de um produto conforme a forma de pagamento. Aplica o desconto de PIX somente se não houver promoção ativa. "
        "Use esta ferramenta em vez de calcular descontos.",
        {"product_id": {"type": "integer", "minimum": 1},
         "payment_method": {"type": "string", "enum": ["standard", "pix", "credit_card", "debit_card", "boleto"],
                            "description": "Forma de pagamento; 'standard' quando o cliente não informou."}},
        ["product_id"]),
    _fn("get_order_details",
        "Consulta um pedido. Exige o número do pedido E o telefone cadastrado, ambos informados pelo cliente nesta conversa.",
        {"order_id": {"type": "integer", "minimum": 1},
         "customer_phone": {"type": "string", "maxLength": 30,
                            "description": "Telefone exatamente como o cliente informou."}},
        ["order_id", "customer_phone"]),
    _fn("get_customer_orders",
        "Lista os pedidos de um cliente a partir do telefone cadastrado informado por ele nesta conversa.",
        {"customer_phone": {"type": "string", "maxLength": 30,
                            "description": "Telefone exatamente como o cliente informou."}},
        ["customer_phone"]),
    _fn("get_store_information",
        "Endereço, telefone, e-mail, horário de funcionamento e escopo da loja.", {}),
    _fn(POLICY_TOOL,
        "Busca trechos do Manual de Políticas e Procedimentos (pagamento, parcelamento, trocas, devoluções, arrependimento, "
        "frete, entrega, promoções, garantia, privacidade, atendimento). Retorna trechos com página e seção.",
        {"question": {"type": "string", "minLength": 3, "maxLength": 500,
                      "description": "A dúvida do cliente reescrita como pergunta completa e autocontida."}},
        ["question"]),
    _fn(HANDOFF_TOOL,
        "Encerra o atendimento automático e encaminha o cliente. Use para assuntos fora do escopo, acessórios, falta de "
        "evidência, divergência no manual, reclamações ou ações que exigem uma pessoa. O texto ao cliente é padronizado.",
        {"reason": {"type": "string", "enum": list(HANDOFF_REASONS)},
         "note": {"type": "string", "maxLength": 300, "description": "Nota interna curta (não é mostrada ao cliente)."}},
        ["reason"]),
]
TOOL_SCHEMAS = {item["function"]["name"]: item["function"]["parameters"] for item in TOOL_DEFINITIONS}

# Source-document inconsistencies that must be disclosed instead of silently resolved
# (documented in the README's assumptions). Each entry is (keywords, also_match_evidence, note): keywords
# always match the customer's question; they match the retrieved text only when the flag is True.
KNOWN_POLICY_INCONSISTENCIES = (
    (("parcel",), True,
     "INCONSISTÊNCIA CONHECIDA: o manual traz regras divergentes de parcela mínima no cartão de crédito "
     "(tabela de pagamentos, p. 3: R$ 100,00 para até 12x; seção 3.1: faixas diferentes por quantidade de parcelas). "
     "Informe a divergência e use handoff_to_human com reason=\"policy_conflict\"; não escolha uma regra."),
    (("whatsapp", "telefone", "contato"), False,
     "INCONSISTÊNCIA CONHECIDA: o manual lista (67) 3341-4444 como telefone/WhatsApp (p. 2) e (67) 3321-4500 como canal "
     "de WhatsApp (p. 6). Informe os dois e peça que o cliente confirme com a loja qual deve usar."),
)


# --- argument validation ---------------------------------------------------------------------

def _coerce(name: str, value: Any, spec: dict) -> tuple[Any, str | None]:
    kind = spec.get("type")
    if kind == "integer":
        if isinstance(value, bool):
            return None, f"{name} must be an integer."
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        elif isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
            value = int(value)
        if not isinstance(value, int):
            return None, f"{name} must be an integer."
    elif kind == "number":
        if isinstance(value, bool):
            return None, f"{name} must be a number."
        if isinstance(value, str):
            try:
                value = float(value.strip().replace(",", "."))
            except ValueError:
                return None, f"{name} must be a number."
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return None, f"{name} must be a finite number."
    elif kind == "boolean":
        if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
            value = value.strip().casefold() == "true"
        if not isinstance(value, bool):
            return None, f"{name} must be true or false."
    elif kind == "string":
        if not isinstance(value, str):
            return None, f"{name} must be a string."
        value = value.strip()
        if len(value) < spec.get("minLength", 0):
            return None, f"{name} is too short."
        if len(value) > spec.get("maxLength", 10_000):
            return None, f"{name} is too long (max {spec['maxLength']} characters)."
        if "enum" in spec:
            match = next((option for option in spec["enum"] if option.casefold() == value.casefold()), None)
            if match is None:
                return None, f"{name} must be one of {spec['enum']}."
            value = match
    if "minimum" in spec and value < spec["minimum"]:
        return None, f"{name} must be >= {spec['minimum']}."
    if "maximum" in spec and value > spec["maximum"]:
        return None, f"{name} must be <= {spec['maximum']}."
    return value, None


def validate_arguments(schema: dict, raw: dict) -> tuple[dict | None, str | None]:
    """Validate/coerce ``raw`` against the small JSON-schema subset used by our tools."""
    properties = schema.get("properties", {})
    unknown = sorted(set(raw) - set(properties))
    if unknown:
        return None, f"Unknown argument(s): {unknown}. Allowed: {sorted(properties)}."
    for name in schema.get("required", []):
        if raw.get(name) is None:
            return None, f"Missing required argument: {name}."
    cleaned = {}
    for name, value in raw.items():
        if value is None:
            continue
        coerced, error = _coerce(name, value, properties[name])
        if error:
            return None, error
        cleaned[name] = coerced
    return cleaned, None


# --- dispatcher ------------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolTrace:
    name: str
    arguments: dict  # personal data masked
    ok: bool
    error: str | None
    duration_ms: float


@dataclass
class TurnContext:
    session: SessionState
    user_messages: list[str]  # every user message of the session, including the current one


@dataclass(frozen=True)
class ToolOutcome:
    result: dict
    content: str  # JSON string sent back to the model
    trace: ToolTrace
    handoff_reason: str | None = None


def _json_default(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else str(value)  # numpy scalars -> Python


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _mask(arguments: dict) -> dict:
    masked = {}
    for key, value in arguments.items():
        if "phone" in key and isinstance(value, str):
            tail = _digits(value)[-4:]
            masked[key] = f"***{tail}" if tail else "***"
        else:
            masked[key] = value
    return masked


def has_positive_evidence(tool_name: str, result: dict) -> bool:
    """True when a tool result actually supports an answer (not just 'nothing found')."""
    if not result.get("ok") or tool_name == HANDOFF_TOOL:
        return False
    if tool_name == POLICY_TOOL:
        return bool(result.get("found_evidence"))
    if tool_name == "search_products":
        return result.get("total_matches", 0) > 0
    return True


class ToolDispatcher:
    def __init__(self, tools: StoreTools, policy: PolicyRetriever, *, rag_top_k: int = 4, rag_min_score: float = 0.20):
        self.tools = tools
        self.policy = policy
        self.rag_top_k = rag_top_k
        self.rag_min_score = rag_min_score
        self.schemas = TOOL_DEFINITIONS
        self._handlers: dict[str, Callable[[dict, TurnContext], dict]] = {
            "search_products": self._search_products,
            "get_product_details": lambda a, c: self.tools.get_product_details(a["product_id"]),
            "get_product_price": lambda a, c: self.tools.get_product_price(
                a["product_id"], a.get("payment_method", "standard")),
            "get_order_details": self._order_details,
            "get_customer_orders": self._customer_orders,
            "get_store_information": lambda a, c: self.tools.get_store_information(),
            POLICY_TOOL: self._search_policy,
        }

    def execute(self, call: ToolCall, ctx: TurnContext) -> ToolOutcome:
        started = time.perf_counter()
        parsed = self._parse_json(call.arguments)
        handoff_reason = None
        arguments: dict = parsed if isinstance(parsed, dict) else {}

        if call.name == HANDOFF_TOOL:
            # Lenient on purpose: an unknown reason still hands off (as "needs_human").
            reason = str(arguments.get("reason", "")).strip()
            handoff_reason = reason if reason in HANDOFF_REASONS else "needs_human"
            arguments = {"reason": handoff_reason, "note": str(arguments.get("note", ""))[:300]}
            result: dict = {"ok": True, "handoff": handoff_reason}
        elif call.name not in self._handlers:
            result = {"ok": False, "error": f"Unknown tool {call.name!r}. Available: {sorted(self._handlers)}."}
        elif not isinstance(parsed, dict):
            result = {"ok": False, "error": "Tool arguments must be a valid JSON object."}
        else:
            cleaned, error = validate_arguments(TOOL_SCHEMAS[call.name], parsed)
            if error:
                result = {"ok": False, "error": f"Invalid arguments for {call.name}: {error}"}
            else:
                arguments = cleaned
                result = self._run(call.name, cleaned, ctx)

        result = json.loads(json.dumps(result, ensure_ascii=False, default=_json_default))
        trace = ToolTrace(
            name=call.name, arguments=_mask(arguments), ok=bool(result.get("ok")),
            error=result.get("error"), duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return ToolOutcome(result=result, content=json.dumps(result, ensure_ascii=False),
                           trace=trace, handoff_reason=handoff_reason)

    @staticmethod
    def _parse_json(raw: str) -> Any:
        try:
            return json.loads(raw) if raw and raw.strip() else {}
        except json.JSONDecodeError:
            return None

    def _run(self, name: str, arguments: dict, ctx: TurnContext) -> dict:
        try:
            return self._handlers[name](arguments, ctx)
        except Exception:  # a tool bug must not crash the conversation or leak a stack trace
            logger.exception("Tool %s failed", name)
            return {"ok": False, "error": "Internal error while running the tool."}

    # -- handlers -----------------------------------------------------------------------------

    def _search_products(self, args: dict, ctx: TurnContext) -> dict:
        return self.tools.search_products(
            query=args.get("query"), category=args.get("category"),
            max_price_brl=args.get("max_price_brl"),
            available_only=args.get("available_only", True), limit=args.get("limit", 5),
        )

    def _verified_phone(self, phone: str, ctx: TurnContext) -> dict | None:
        """Return an error result if the phone may not be used, else None.

        Guards: (1) the number must have been typed by the customer in this conversation,
        so the model cannot guess or enumerate phones; (2) a few failed lookups lock the
        order tools for the rest of the session.
        """
        if ctx.session.failed_verifications >= MAX_FAILED_VERIFICATIONS:
            return {"ok": False, "error": "Too many failed verification attempts in this session. "
                                          "Do not retry; refer the customer to human support.",
                    "suggested_handoff": "needs_human"}
        digits = _digits(phone)
        if len(digits) < MIN_PHONE_DIGITS:
            return {"ok": False, "error": "The phone number is incomplete. Ask the customer for the full registered phone (with area code)."}
        if not any(digits in _digits(message) for message in ctx.user_messages):
            return {"ok": False, "error": "This phone number was not provided by the customer in this conversation. "
                                          "Ask the customer for their registered phone number; never guess it."}
        return None

    def _order_details(self, args: dict, ctx: TurnContext) -> dict:
        blocked = self._verified_phone(args["customer_phone"], ctx)
        if blocked:
            return blocked
        result = self.tools.get_order_details(args["order_id"], args["customer_phone"])
        if not result.get("ok"):
            ctx.session.failed_verifications += 1
        return result

    def _customer_orders(self, args: dict, ctx: TurnContext) -> dict:
        blocked = self._verified_phone(args["customer_phone"], ctx)
        if blocked:
            return blocked
        result = self.tools.get_customer_orders(args["customer_phone"])
        if not result.get("ok"):
            ctx.session.failed_verifications += 1
        return result

    def _search_policy(self, args: dict, ctx: TurnContext) -> dict:
        retrieval = self.policy.retrieve(args["question"], top_k=self.rag_top_k, min_score=self.rag_min_score)
        if not retrieval.get("ok"):
            return {"ok": False, "error": retrieval.get("error", "Policy retrieval failed.")}
        passages = [{"page": item["page"], "section": item["section"], "text": item["text"],
                     "relevance": item["score"]} for item in retrieval["results"]]
        result: dict = {
            "ok": True, "found_evidence": bool(passages),
            "source": "Manual de Políticas e Procedimentos, versão 2.1 (junho de 2025)",
            "passages": passages,
        }
        if not passages:
            result["instruction"] = ("Nenhum trecho relevante foi encontrado. Não responda por suposição: diga que não "
                                     "encontrou e encaminhe ao atendimento humano.")
        question = args["question"].casefold()
        evidence_text = " ".join(p["text"] for p in passages).casefold()
        notes = [
            note for keywords, in_evidence, note in KNOWN_POLICY_INCONSISTENCIES
            if any(k in question or (in_evidence and k in evidence_text) for k in keywords)
        ]
        if notes:
            result["notes"] = notes
        return result
