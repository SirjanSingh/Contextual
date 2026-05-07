"""Hybrid search combining BM25 (keyword) and vector (semantic) search."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from .embedder import Embedder
from .retriever import RetrievedChunk
from .debug import log_json


@dataclass
class BM25Index:
    """BM25 index for keyword-based search."""
    
    bm25: BM25Okapi
    metadata: List[Dict]
    tokenized_corpus: List[List[str]]
    
    @staticmethod
    def simple_tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()
    
    @classmethod
    def build(cls, chunks: List[Dict]) -> BM25Index:
        """Build BM25 index from chunks.
        
        Args:
            chunks: List of chunk metadata dicts with 'text' field.
        
        Returns:
            BM25Index instance.
        """
        # Tokenize all chunks
        tokenized_corpus = [cls.simple_tokenize(c["text"]) for c in chunks]
        
        # Build BM25 index
        bm25 = BM25Okapi(tokenized_corpus)
        
        return cls(
            bm25=bm25,
            metadata=chunks,
            tokenized_corpus=tokenized_corpus,
        )
    
    def search(self, query: str, top_k: int = 20) -> List[tuple[int, float]]:
        """Search using BM25.
        
        Args:
            query: Query string.
            top_k: Number of top results to return.
        
        Returns:
            List of (index, score) tuples sorted by score descending.
        """
        tokenized_query = self.simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def save(self, path: Path) -> None:
        """Save BM25 index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path) -> Optional[BM25Index]:
        """Load BM25 index from disk."""
        if not path.exists():
            return None
        
        with open(path, "rb") as f:
            return pickle.load(f)


def reciprocal_rank_fusion(
    rankings: List[List[tuple[int, float]]],
    k: int = 60
) -> List[tuple[int, float]]:
    """Combine multiple rankings using Reciprocal Rank Fusion (RRF).
    
    RRF formula: score = Σ(1 / (k + rank))
    
    Args:
        rankings: List of rankings, each is a list of (index, score) tuples.
        k: Constant for RRF (default: 60, standard value).
    
    Returns:
        Combined ranking as list of (index, score) tuples sorted by score descending.
    """
    rrf_scores: Dict[int, float] = {}
    
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking, start=1):
            if idx not in rrf_scores:
                rrf_scores[idx] = 0.0
            rrf_scores[idx] += 1.0 / (k + rank)
    
    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [(idx, score) for idx, score in sorted_items]


def hybrid_retrieve(
    *,
    faiss_index: faiss.Index,
    bm25_index: BM25Index,
    metadata: List[Dict],
    embedder: Embedder,
    question: str,
    top_k: int = 6,
    retrieve_k: int = 20,
) -> List[RetrievedChunk]:
    """Retrieve using hybrid search (BM25 + vector with RRF).
    
    Args:
        faiss_index: FAISS vector index.
        bm25_index: BM25 keyword index.
        metadata: Chunk metadata.
        embedder: Embedder for query encoding.
        question: Query string.
        top_k: Final number of chunks to return.
        retrieve_k: Number of candidates to retrieve from each method.
    
    Returns:
        Top-k chunks ranked by hybrid RRF score.
    """
    # 1. Vector search
    q_vec = embedder.embed_query(question).astype(np.float32).reshape(1, -1)
    vector_scores, vector_ids = faiss_index.search(q_vec, retrieve_k)
    vector_scores = vector_scores[0].tolist()
    vector_ids = vector_ids[0].tolist()
    
    # Convert to ranking (index, score)
    vector_ranking = [
        (int(idx), float(score))
        for idx, score in zip(vector_ids, vector_scores)
        if idx >= 0 and idx < len(metadata)
    ]
    
    # 2. BM25 search
    bm25_ranking = bm25_index.search(question, top_k=retrieve_k)
    
    # Filter invalid indices
    bm25_ranking = [
        (idx, score)
        for idx, score in bm25_ranking
        if idx >= 0 and idx < len(metadata)
    ]
    
    # 3. Combine with RRF
    combined_ranking = reciprocal_rank_fusion([vector_ranking, bm25_ranking])
    
    # 4. Convert to RetrievedChunk objects
    results: List[RetrievedChunk] = []
    
    for idx, rrf_score in combined_ranking[:top_k]:
        m = metadata[idx]
        results.append(
            RetrievedChunk(
                text=m["text"],
                source=m["source"],
                start_char=int(m["start_char"]),
                end_char=int(m["end_char"]),
                score=float(rrf_score),
            )
        )
    
    # Log results
    log_json(
        "hybrid_retrieved_chunks",
        [
            {
                "source": r.source,
                "start": r.start_char,
                "end": r.end_char,
                "score": r.score,
                "length": len(r.text),
            }
            for r in results
        ],
    )
    
    return results
