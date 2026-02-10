"""LLM client using Google Gemini API."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .config import get_config
from .retriever import RetrievedChunk


SYSTEM_PROMPT = """You are a repo-aware coding assistant.

Hard rules:
- Use ONLY the provided REPO CONTEXT.
- Do NOT invent files, functions, or behavior.
- If you cannot answer from the context OR make a supported inference, say exactly:
  Not found in the retrieved repository context.

Supported inference rule:
- You MAY infer behavior that follows directly from the shown code.
- Every inference MUST cite an Evidence quote from the context.

Output format (must follow):
Answer:
- <1-4 bullets>

Evidence:
- "<direct quote>" (source: path:start-end)
- "<direct quote>" (source: path:start-end)

Sources:
- path:start-end
- path:start-end
"""


def _format_context(chunks: List[RetrievedChunk], max_chars: int = 15000) -> str:
    """Build a compact context block. Keep within a safe budget."""
    parts: List[str] = []
    used = 0
    for c in chunks:
        header = f"\n---\nFILE: {c.source} [{c.start_char}-{c.end_char}] (score={c.score:.3f})\n"
        body = c.text.strip()
        block = header + body + "\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "".join(parts).strip()


@dataclass
class LLMClient:
    """LLM client using Google Gemini API with new google.genai package."""
    
    model: str = "models/gemini-2.0-flash"
    temperature: float = 0.2
    _client: object = field(default=None, repr=False, init=False)
    
    def __post_init__(self) -> None:
        """Initialize the Gemini client."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai is required. Install with:\n"
                "  pip install google-genai"
            )
        
        config = get_config()
        self._client = genai.Client(api_key=config.google_api_key)
        self.model = config.gemini_model
        self._types = types
    
    def answer(self, question: str, chunks: List[RetrievedChunk], conversation_context: str = "") -> str:
        """Generate an answer based on retrieved chunks.
        
        Args:
            question: The user's question.
            chunks: Retrieved code chunks as context.
            conversation_context: Optional conversation history for follow-up questions.
        
        Returns:
            The generated answer string.
        """
        context = _format_context(chunks)
        
        # Include conversation history if provided
        full_context = conversation_context + context if conversation_context else context
        
        user_prompt = f"""REPO CONTEXT:
{full_context}

QUESTION:
{question}

Answer ONLY if the information is explicitly present in the REPO CONTEXT.
If not, say: "Not found in the retrieved repository context."
Follow the Answer/Evidence/Sources format. Do not add extra meta text
"""
        
        try:
            # Create contents with system instruction and user prompt
            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    self._types.Content(
                        role="user",
                        parts=[self._types.Part(text=SYSTEM_PROMPT + "\n\n" + user_prompt)]
                    )
                ],
                config=self._types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=2048,
                ),
            )
            
            # Extract text from response
            if response.text:
                return response.text.strip()
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    return "".join(part.text for part in candidate.content.parts if hasattr(part, 'text')).strip()
            
            return "No response generated. The model may have blocked the content."
                
        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg.upper() or "AUTHENTICATION" in error_msg.upper():
                return (
                    f"API Authentication Error: {e}\n\n"
                    "Tip: Ensure GOOGLE_API_KEY is set correctly in your .env file.\n"
                    "Get your key from: https://aistudio.google.com/app/apikey"
                )
            elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg.upper():
                return (
                    f"Rate Limit Error: {e}\n\n"
                    "Tip: You've hit the API rate limit. Wait 60 seconds and try again.\n"
                    "Free tier: 15 requests per minute."
                )
            return f"LLM error: {e}"
