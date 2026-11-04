"""Conversation state kept between turns."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    """State for one customer conversation.

    ``history`` keeps only user/assistant text (never tool payloads), so personal data
    returned by tools is not replayed into later prompts. ``grounded_numbers`` and
    ``grounded_pages`` accumulate everything the tools/RAG have returned in this session;
    the answer validator accepts numeric claims only if they appear there.
    """

    history: list[dict] = field(default_factory=list)
    grounded_numbers: set[float] = field(default_factory=set)
    grounded_pages: set[int] = field(default_factory=set)
    failed_verifications: int = 0

    def remember_turn(self, question: str, answer: str, max_messages: int) -> None:
        self.history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        if max_messages <= 0:
            self.history.clear()
            return
        del self.history[:-max_messages]
        if self.history and self.history[0]["role"] != "user":
            del self.history[0]
