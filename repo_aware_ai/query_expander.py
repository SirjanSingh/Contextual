"""Query expansion using LLM to generate alternative phrasings."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


EXPANSION_PROMPT = """You are a code search query expander. Given a user question about a codebase, generate 2-3 alternative phrasings that would help find relevant code.

Rules:
- Keep expansions focused on code/programming concepts
- Include both specific (function names, class names) and general (concepts) variants
- Output ONLY the expanded queries, one per line, numbered 1-3
- Do NOT include the original query
- Be concise — each expansion should be under 20 words

User question: {question}

Expanded queries:"""


@dataclass
class QueryExpander:
    """Generates alternative query phrasings using the LLM for broader retrieval coverage."""
    
    _client: object = field(default=None, repr=False)
    _types: object = field(default=None, repr=False)
    model: str = ""
    
    @classmethod
    def from_llm_client(cls, llm_client) -> QueryExpander:
        """Create a QueryExpander reusing the LLMClient's API connection."""
        expander = cls()
        expander._client = llm_client._client
        expander._types = llm_client._types
        expander.model = llm_client.model
        return expander
    
    def expand(self, question: str, max_expansions: int = 3) -> List[str]:
        """Generate expanded queries from a user question.
        
        Args:
            question: The original user question.
            max_expansions: Maximum number of expanded queries to generate.
        
        Returns:
            List of queries: original + expanded variants.
        """
        if not self._client:
            return [question]
        
        try:
            prompt = EXPANSION_PROMPT.format(question=question)
            
            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    self._types.Content(
                        role="user",
                        parts=[self._types.Part(text=prompt)]
                    )
                ],
                config=self._types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=256,
                ),
            )
            
            if not response.text:
                return [question]
            
            # Parse numbered lines from response
            expanded = _parse_expansions(response.text, max_expansions)
            
            # Always include the original question first
            return [question] + expanded
            
        except Exception:
            # On any error, fall back to just the original query
            return [question]


def _parse_expansions(text: str, max_count: int) -> List[str]:
    """Parse numbered expansions from LLM response text."""
    lines = text.strip().splitlines()
    expansions = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove numbering like "1.", "1)", "1:", "- "
        cleaned = re.sub(r'^[\d]+[.):\-]\s*', '', line).strip()
        cleaned = re.sub(r'^[-•]\s*', '', cleaned).strip()
        
        if cleaned and len(cleaned) > 3:
            expansions.append(cleaned)
        
        if len(expansions) >= max_count:
            break
    
    return expansions
