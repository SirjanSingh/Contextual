from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import faiss

from .embedder import Embedder
from .debug import log_json


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    start_char: int
    end_char: int
    score: float


def retrieve(
    *,
    index: faiss.Index,
    metadata: List[Dict],
    embedder: Embedder,
    question: str,
    top_k: int = 6,
) -> List[RetrievedChunk]:
    q = embedder.embed_query(question).astype(np.float32)
    q = q.reshape(1, -1)

    scores, ids = index.search(q, top_k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results: List[RetrievedChunk] = []

    for score, idx in zip(scores, ids):
        if idx < 0 or idx >= len(metadata):
            continue

        m = metadata[idx]
        results.append(
            RetrievedChunk(
                text=m["text"],
                source=m["source"],
                start_char=int(m["start_char"]),
                end_char=int(m["end_char"]),
                score=float(score),
            )
        )

    # Log AFTER collecting all results (important)
    log_json(
        "retrieved_chunks",
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
