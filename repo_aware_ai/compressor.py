"""Contextual compression to extract only relevant lines from retrieved chunks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .retriever import RetrievedChunk


COMPRESSION_PROMPT = """You are a code context compressor. Given a question and a code chunk, extract ONLY the lines that are directly relevant to answering the question.

Rules:
- Return the relevant lines VERBATIM (do not modify, summarize, or explain)
- Include surrounding context lines (imports, class/function signatures) if they help understanding
- If no lines are relevant, respond with exactly: NONE
- Do NOT add commentary, explanations, or markdown formatting
- Output raw code/text only

QUESTION: {question}

CODE CHUNK (from {source}):
{chunk_text}

RELEVANT LINES:"""


@dataclass
class ContextCompressor:
    """Compresses retrieved chunks by extracting only question-relevant lines."""
    
    _client: object = field(default=None, repr=False)
    _types: object = field(default=None, repr=False)
    model: str = ""
    
    @classmethod
    def from_llm_client(cls, llm_client) -> ContextCompressor:
        """Create a ContextCompressor reusing the LLMClient's API connection."""
        compressor = cls()
        compressor._client = llm_client._client
        compressor._types = llm_client._types
        compressor.model = llm_client.model
        return compressor
    
    def compress(
        self,
        question: str,
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """Compress chunks by extracting only relevant lines.
        
        Args:
            question: The user's question.
            chunks: Retrieved chunks to compress.
        
        Returns:
            Compressed chunks with only relevant content. Chunks with no
            relevant content are dropped entirely.
        """
        if not self._client or not chunks:
            return chunks
        
        compressed = []
        
        for chunk in chunks:
            try:
                prompt = COMPRESSION_PROMPT.format(
                    question=question,
                    source=chunk.source,
                    chunk_text=chunk.text,
                )
                
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=[
                        self._types.Content(
                            role="user",
                            parts=[self._types.Part(text=prompt)]
                        )
                    ],
                    config=self._types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1024,
                    ),
                )
                
                if not response.text:
                    compressed.append(chunk)
                    continue
                
                extracted = response.text.strip()
                
                # Skip chunk if nothing relevant
                if extracted.upper() == "NONE" or len(extracted) < 5:
                    continue
                
                # Create new chunk with compressed text
                compressed.append(
                    RetrievedChunk(
                        text=extracted,
                        source=chunk.source,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        score=chunk.score,
                    )
                )
                
            except Exception:
                # On error, keep original chunk
                compressed.append(chunk)
        
        # If compression removed everything, return original chunks
        if not compressed:
            return chunks
        
        return compressed
