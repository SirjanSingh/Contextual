"""Reranker using cross-encoder for improved retrieval quality."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .retriever import RetrievedChunk


@dataclass
class Reranker:
    """Reranks retrieved chunks using a cross-encoder model.
    
    Cross-encoders are more accurate than bi-encoders (used for initial retrieval)
    because they can see both query and document together, but they're slower.
    
    Strategy: Use fast bi-encoder for initial retrieval (top-20), then use
    cross-encoder to rerank to final top-k.
    """
    
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _model: object = None
    
    def __post_init__(self) -> None:
        """Lazy load the model (only when first used)."""
        pass
    
    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for reranking. Install with:\n"
                "  pip install sentence-transformers"
            )
        
        self._model = CrossEncoder(self.model_name)
    
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 6
    ) -> List[RetrievedChunk]:
        """Rerank chunks using cross-encoder.
        
        Args:
            query: The user's question.
            chunks: Retrieved chunks from initial search.
            top_k: Number of top chunks to return after reranking.
        
        Returns:
            Top-k chunks sorted by cross-encoder score.
        """
        if not chunks:
            return []
        
        # Lazy load model
        self._load_model()
        
        # Create query-document pairs
        pairs = [[query, chunk.text] for chunk in chunks]
        
        # Score all pairs
        scores = self._model.predict(pairs)
        
        # Combine chunks with scores and sort
        chunk_scores = list(zip(chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k chunks (update their scores)
        reranked = []
        for chunk, score in chunk_scores[:top_k]:
            # Create new chunk with updated score
            reranked.append(
                RetrievedChunk(
                    text=chunk.text,
                    source=chunk.source,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    score=float(score),  # Use cross-encoder score
                )
            )
        
        return reranked
