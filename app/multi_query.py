"""Multi-query retrieval: decompose complex questions into sub-questions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


DECOMPOSE_PROMPT = """You are a code question decomposer. Break this complex question into 2-3 simpler, focused sub-questions that each target a specific piece of information in a codebase.

Rules:
- Each sub-question should be answerable independently
- Keep sub-questions specific and concise (under 15 words each)
- Focus on code-level concepts (functions, classes, files, modules)
- Output ONLY the sub-questions, numbered 1-3
- Do NOT include the original question

Question: {question}

Sub-questions:"""


def _is_complex_query(question: str) -> bool:
    """Heuristic check for whether a question is complex enough to decompose.
    
    Returns True for questions that are long or contain compound indicators.
    """
    # Short/simple questions don't benefit from decomposition
    words = question.split()
    if len(words) < 10:
        return False
    
    # Compound indicators
    compound_markers = [
        " and ", " also ", " as well as ",
        " relate to ", " relationship between ",
        " compare ", " difference between ",
        " how does ", " what happens when ",
    ]
    
    question_lower = question.lower()
    return any(marker in question_lower for marker in compound_markers)


@dataclass
class MultiQueryGenerator:
    """Decomposes complex questions into simpler sub-questions for retrieval."""
    
    _client: object = field(default=None, repr=False)
    _types: object = field(default=None, repr=False)
    model: str = ""
    
    @classmethod
    def from_llm_client(cls, llm_client) -> MultiQueryGenerator:
        """Create a MultiQueryGenerator reusing the LLMClient's API connection."""
        gen = cls()
        gen._client = llm_client._client
        gen._types = llm_client._types
        gen.model = llm_client.model
        return gen
    
    def generate(self, question: str) -> List[str]:
        """Generate sub-questions for a complex question.
        
        For simple questions, returns just the original. For complex ones,
        returns the original + decomposed sub-questions.
        
        Args:
            question: The user's question.
        
        Returns:
            List of questions: original + sub-questions (if complex).
        """
        if not self._client:
            return [question]
        
        # Skip decomposition for simple questions
        if not _is_complex_query(question):
            return [question]
        
        try:
            prompt = DECOMPOSE_PROMPT.format(question=question)
            
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
            
            # Parse sub-questions
            sub_questions = _parse_sub_questions(response.text)
            
            if not sub_questions:
                return [question]
            
            # Always include original question first
            return [question] + sub_questions
            
        except Exception:
            return [question]


def _parse_sub_questions(text: str) -> List[str]:
    """Parse numbered sub-questions from LLM response."""
    lines = text.strip().splitlines()
    questions = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove numbering
        cleaned = re.sub(r'^[\d]+[.):\-]\s*', '', line).strip()
        cleaned = re.sub(r'^[-•]\s*', '', cleaned).strip()
        
        if cleaned and len(cleaned) > 5:
            questions.append(cleaned)
        
        if len(questions) >= 3:
            break
    
    return questions
