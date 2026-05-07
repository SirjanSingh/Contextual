"""LLM client using Google Gemini API."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List

from ._retry import gemini_retry
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
    """Build a compact context block within a safe budget."""
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


def _build_prompt(question: str, chunks: List[RetrievedChunk], conversation_context: str) -> str:
    context = _format_context(chunks)
    full_context = (conversation_context + context) if conversation_context else context
    return SYSTEM_PROMPT + "\n\n" + (
        f"REPO CONTEXT:\n{full_context}\n\n"
        f"QUESTION:\n{question}\n\n"
        'Answer ONLY if the information is explicitly present in the REPO CONTEXT.\n'
        'If not, say: "Not found in the retrieved repository context."\n'
        "Follow the Answer/Evidence/Sources format. Do not add extra meta text\n"
    )


def _humanize_error(exc: Exception) -> str:
    msg = str(exc)
    upper = msg.upper()
    if "API_KEY" in upper or "AUTHENTICATION" in upper or "UNAUTHENTICATED" in upper:
        return (
            f"API authentication error: {msg}\n\n"
            "Tip: ensure GOOGLE_API_KEY is set in your .env. "
            "Get a key at https://aistudio.google.com/app/apikey"
        )
    if "429" in msg or "RESOURCE_EXHAUSTED" in upper:
        return (
            f"Rate limit exceeded: {msg}\n\n"
            "Tip: wait a minute and retry. Free tier is ~15 requests/min."
        )
    return f"LLM error: {msg}"


@dataclass
class LLMClient:
    """Gemini LLM client with retry and streaming support."""

    model: str = "models/gemini-2.5-flash"
    temperature: float = 0.2
    max_output_tokens: int = 2048
    _client: object = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "google-genai is required. Install with `pip install google-genai`."
            ) from e

        config = get_config()
        self._client = genai.Client(api_key=config.google_api_key)
        self.model = config.gemini_model
        self._types = types

    def _gen_config(self):
        return self._types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    def _content(self, prompt: str):
        return [
            self._types.Content(
                role="user",
                parts=[self._types.Part(text=prompt)],
            )
        ]

    @gemini_retry
    def _generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            contents=self._content(prompt),
            config=self._gen_config(),
        )
        if getattr(response, "text", None):
            return response.text.strip()
        if getattr(response, "candidates", None):
            cand = response.candidates[0]
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                return "".join(p.text for p in parts if getattr(p, "text", None)).strip()
        return "No response generated. The model may have blocked the content."

    def answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        conversation_context: str = "",
    ) -> str:
        prompt = _build_prompt(question, chunks, conversation_context)
        try:
            return self._generate(prompt)
        except Exception as e:  # noqa: BLE001 — surface a friendly message
            return _humanize_error(e)

    def stream_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        conversation_context: str = "",
    ) -> Iterator[str]:
        """Stream answer chunks as they arrive from Gemini.

        Yields incremental text chunks. Falls back to a single yield if the
        SDK does not support streaming on this version.
        """
        prompt = _build_prompt(question, chunks, conversation_context)

        # google-genai >= 0.3 exposes generate_content_stream.
        stream_fn = getattr(self._client.models, "generate_content_stream", None)
        if stream_fn is None:
            try:
                yield self._generate(prompt)
            except Exception as e:  # noqa: BLE001
                yield _humanize_error(e)
            return

        try:
            for chunk in stream_fn(
                model=self.model,
                contents=self._content(prompt),
                config=self._gen_config(),
            ):
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception as e:  # noqa: BLE001
            yield _humanize_error(e)
