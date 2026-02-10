"""Conversation history management for multi-turn dialogues."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    question: str
    answer: str
    sources: List[str]


@dataclass
class ConversationHistory:
    """Manages conversation history for context-aware follow-up questions."""
    
    max_turns: int = 5  # Keep last 5 turns
    _history: List[ConversationTurn] = field(default_factory=list)
    
    def add_turn(self, question: str, answer: str, sources: List[str]) -> None:
        """Add a new conversation turn."""
        turn = ConversationTurn(question=question, answer=answer, sources=sources)
        self._history.append(turn)
        
        # Keep only last max_turns
        if len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns:]
    
    def get_context(self, include_sources: bool = False) -> str:
        """Get formatted conversation history for context.
        
        Args:
            include_sources: Whether to include source citations in history.
        
        Returns:
            Formatted conversation history string.
        """
        if not self._history:
            return ""
        
        context_parts = ["CONVERSATION HISTORY:"]
        
        for i, turn in enumerate(self._history, 1):
            context_parts.append(f"\nQ{i}: {turn.question}")
            context_parts.append(f"A{i}: {turn.answer}")
            
            if include_sources and turn.sources:
                sources_str = ", ".join(turn.sources[:3])  # Show max 3 sources
                context_parts.append(f"Sources: {sources_str}")
        
        context_parts.append("\n---\n")
        return "\n".join(context_parts)
    
    def get_last_question(self) -> str | None:
        """Get the last question asked."""
        if not self._history:
            return None
        return self._history[-1].question
    
    def clear(self) -> None:
        """Clear conversation history."""
        self._history.clear()
    
    def __len__(self) -> int:
        """Return number of turns in history."""
        return len(self._history)
    
    @property
    def is_empty(self) -> bool:
        """Check if history is empty."""
        return len(self._history) == 0
