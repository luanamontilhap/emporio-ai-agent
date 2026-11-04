"""Support-agent orchestration: the six-step pipeline.

1. Input        a natural-language question (plus the session history).
2. LLM          decides whether a tool is needed (native tool calling).
3. Tools        deterministic product / price / order / store-info functions.
4. RAG          ``search_store_policy`` retrieves manual passages (also a tool the LLM calls).
5. LLM          receives the tool results and writes the answer.
6. Validation   the draft is rejected if its figures or citations lack evidence; out-of-scope
                and no-evidence cases are answered with fixed hand-off templates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Literal

from .config import PROJECT_ROOT, Settings
from .llm_client import ChatClient, LLMError, OpenAICompatibleClient
from .prompts import (EMPTY_QUESTION_MESSAGE, SYSTEM_PROMPT, format_contact, render_handoff,
                      render_technical_error, validation_feedback)
from .session import SessionState
from .tool_registry import (POLICY_TOOL, PolicyRetriever, ToolDispatcher, ToolTrace, TurnContext,
                            has_positive_evidence)
from .tools import StoreTools
from .validation import Violation, collect_evidence, validate_answer

logger = logging.getLogger(__name__)

Status = Literal["answered", "handoff", "abstained", "error"]


@dataclass
class AgentResponse:
    text: str
    status: Status
    handoff_reason: str | None = None
    tool_trace: list[ToolTrace] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)  # [{"page": 4, "section": "..."}]
    violations: list[Violation] = field(default_factory=list)  # problems found in rejected drafts
    llm_calls: int = 0


@dataclass
class _Turn:
    trace: list[ToolTrace] = field(default_factory=list)
    citations: dict[tuple[int, str], dict] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)
    positive_evidence: bool = False
    policy_miss: bool = False
    llm_calls: int = 0
    tool_rounds: int = 0
    validation_retries: int = 0


class SupportAgent:
    def __init__(self, llm: ChatClient, dispatcher: ToolDispatcher, settings: Settings):
        self.llm = llm
        self.dispatcher = dispatcher
        self.settings = settings
        # Contact data comes from the deterministic store-information tool, never from the model.
        self._contact = format_contact(dispatcher.tools.get_store_information())

    def new_session(self) -> SessionState:
        return SessionState()

    def ask(self, question: str, session: SessionState | None = None) -> AgentResponse:
        session = session if session is not None else SessionState()
        question = (question or "").strip()
        if not question:
            return AgentResponse(text=EMPTY_QUESTION_MESSAGE, status="answered")

        settings = self.settings
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, *session.history,
                                {"role": "user", "content": question}]
        ctx = TurnContext(
            session=session,
            user_messages=[m["content"] for m in session.history if m["role"] == "user"] + [question],
        )
        turn = _Turn()
        max_llm_calls = settings.max_tool_rounds + settings.max_validation_retries + 1

        while turn.llm_calls < max_llm_calls:
            try:
                reply = self.llm.complete(messages, self.dispatcher.schemas)
            except LLMError:
                logger.exception("LLM call failed")
                return AgentResponse(text=render_technical_error(self._contact), status="error",
                                     tool_trace=turn.trace, llm_calls=turn.llm_calls)
            turn.llm_calls += 1

            # Steps 2-4: the model asked for tools (or for a hand-off).
            if reply.tool_calls:
                turn.tool_rounds += 1
                if turn.tool_rounds > settings.max_tool_rounds:
                    break
                messages.append(reply.to_message())
                handoff = self._run_tool_calls(reply.tool_calls, messages, ctx, turn)
                if handoff:
                    return self._finish(session, question, turn, "handoff", handoff)
                continue

            # Steps 5-6: the model wrote an answer; validate before showing it.
            draft = (reply.content or "").strip()
            if turn.policy_miss and not turn.positive_evidence:
                # The manual was searched and nothing relevant came back: never let the model improvise.
                return self._finish(session, question, turn, "abstained", "insufficient_evidence")

            violations = validate_answer(draft, session.grounded_numbers, session.grounded_pages)
            if not violations:
                return self._finish(session, question, turn, "answered", text=draft)

            turn.violations.extend(violations)
            if turn.validation_retries < settings.max_validation_retries:
                turn.validation_retries += 1
                messages.append({"role": "assistant", "content": draft})
                messages.append({"role": "user", "content": validation_feedback(violations)})
                continue
            return self._finish(session, question, turn, "abstained", "insufficient_evidence")

        return self._finish(session, question, turn, "handoff", "needs_human")  # round budget exhausted

    # -- internals ----------------------------------------------------------------------------

    def _run_tool_calls(self, calls, messages: list[dict], ctx: TurnContext, turn: _Turn) -> str | None:
        """Execute the model's tool calls in order. Returns a hand-off reason if one was requested."""
        for call in calls:
            outcome = self.dispatcher.execute(call, ctx)
            turn.trace.append(outcome.trace)
            if outcome.handoff_reason:
                return outcome.handoff_reason
            messages.append({"role": "tool", "tool_call_id": call.id, "content": outcome.content})

            numbers, pages = collect_evidence(outcome.result)
            ctx.session.grounded_numbers |= numbers
            ctx.session.grounded_pages |= pages
            if has_positive_evidence(call.name, outcome.result):
                turn.positive_evidence = True
            if call.name == POLICY_TOOL:
                if not outcome.result.get("found_evidence"):
                    turn.policy_miss = True
                for passage in outcome.result.get("passages", []):
                    turn.citations[(passage["page"], passage["section"])] = {
                        "page": passage["page"], "section": passage["section"]}
        return None

    def _finish(self, session: SessionState, question: str, turn: _Turn, status: Status,
                reason: str | None = None, text: str | None = None) -> AgentResponse:
        if text is None:
            text = render_handoff(reason or "needs_human", self._contact)
        session.remember_turn(question, text, self.settings.max_history_messages)
        return AgentResponse(
            text=text, status=status, handoff_reason=reason, tool_trace=turn.trace,
            citations=list(turn.citations.values()) if status == "answered" else [],
            violations=turn.violations, llm_calls=turn.llm_calls,
        )


def build_agent(settings: Settings, *, project_root: Path = PROJECT_ROOT, llm: ChatClient | None = None,
                policy: PolicyRetriever | None = None) -> SupportAgent:
    """Wire the real components. Any of them can be replaced (tests inject fakes)."""
    from .data_loader import StoreData

    tools = StoreTools(StoreData(project_root / "data"))
    if policy is None:
        from .policy_rag import PolicyRAG  # imported lazily: loads the embedding model
        policy = PolicyRAG(project_root / "documents" / "politicas_da_loja.pdf")
    dispatcher = ToolDispatcher(tools, policy, rag_top_k=settings.rag_top_k, rag_min_score=settings.rag_min_score)
    return SupportAgent(llm or OpenAICompatibleClient(settings), dispatcher, settings)
