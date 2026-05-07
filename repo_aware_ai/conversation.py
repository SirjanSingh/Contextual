"""Conversation history management for multi-turn dialogues."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    question: str
    answer: str
    sources: List[str]


@dataclass
class ConversationHistory:
    """Bounded ring buffer of recent conversation turns.

    Thread-safe so the FastAPI server can serve concurrent /query requests
    without corrupting history.
    """

    max_turns: int = 5
    _history: List[ConversationTurn] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_turn(self, question: str, answer: str, sources: List[str]) -> None:
        turn = ConversationTurn(question=question, answer=answer, sources=list(sources))
        with self._lock:
            self._history.append(turn)
            if len(self._history) > self.max_turns:
                self._history = self._history[-self.max_turns :]

    def get_context(self, include_sources: bool = False) -> str:
        with self._lock:
            snapshot = list(self._history)

        if not snapshot:
            return ""

        parts = ["CONVERSATION HISTORY:"]
        for i, turn in enumerate(snapshot, 1):
            parts.append(f"\nQ{i}: {turn.question}")
            parts.append(f"A{i}: {turn.answer}")
            if include_sources and turn.sources:
                parts.append(f"Sources: {', '.join(turn.sources[:3])}")
        parts.append("\n---\n")
        return "\n".join(parts)

    def get_last_question(self) -> str | None:
        with self._lock:
            return self._history[-1].question if self._history else None

    def clear(self) -> None:
        with self._lock:
            self._history.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._history)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0
