from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import faiss
import numpy as np

from .debug import log_json
from .embedder import Embedder

if TYPE_CHECKING:
    from .repo_map.graph import KnowledgeGraph
    from .repo_map.types import RepoMapData


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
    metadata: list[dict],
    embedder: Embedder,
    question: str,
    top_k: int = 6,
) -> list[RetrievedChunk]:
    q = embedder.embed_query(question).astype(np.float32)
    q = q.reshape(1, -1)

    scores, ids = index.search(q, top_k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results: list[RetrievedChunk] = []

    for score, idx in zip(scores, ids, strict=False):
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


def retrieve_with_graph_context(
    *,
    index: faiss.Index,
    metadata: list[dict],
    embedder: Embedder,
    question: str,
    top_k: int = 6,
    repo_graph: KnowledgeGraph | None = None,
    repo_map: RepoMapData | None = None,
    graph_hops: int = 1,
    graph_bonus: float = 0.05,
) -> list[RetrievedChunk]:
    """Retrieve chunks and expand with graph-related symbols.

    After initial FAISS retrieval, looks up each chunk's file in the repo map,
    finds related symbols (callers/callees/same community) and adds their
    chunks to the context — giving the LLM a fuller picture of connected code.

    Falls back to plain `retrieve()` if repo_graph is None.
    """
    initial = retrieve(
        index=index, metadata=metadata, embedder=embedder, question=question, top_k=top_k
    )

    if repo_graph is None:
        return initial

    # Build file -> chunks lookup from metadata
    file_chunks: dict[str, list[dict]] = {}
    for m in metadata:
        fp = m.get("source", "")
        file_chunks.setdefault(fp, []).append(m)

    seen_keys: set[tuple] = {(c.source, c.start_char, c.end_char) for c in initial}
    extra: list[RetrievedChunk] = []

    # Build membership map for community-mates lookup
    membership_map: dict[str, str] = {}
    if repo_map:
        for m in repo_map.communities.memberships:
            membership_map[m.node_id] = m.community_id

    for chunk in initial:
        # Find all symbols in this file
        file_symbols = [
            n
            for n in repo_graph.nodes_by_file(chunk.source)
            if n.label in ("Function", "Class", "Method")
        ]

        related_files: set[str] = set()

        for sym in file_symbols:
            # Add files of callees and callers (1 hop)
            for rel in repo_graph.outgoing(sym.id, "CALLS"):
                n = repo_graph.get_node(rel.target_id)
                if n and rel.confidence >= 0.5:
                    related_files.add(n.properties.file_path)
            for rel in repo_graph.incoming(sym.id, "CALLS"):
                n = repo_graph.get_node(rel.source_id)
                if n and rel.confidence >= 0.5:
                    related_files.add(n.properties.file_path)

            # Add community-mate files (same cluster)
            comm_id = membership_map.get(sym.id)
            if comm_id and repo_map:
                for mb in repo_map.communities.memberships:
                    if mb.community_id == comm_id and mb.node_id != sym.id:
                        n = repo_graph.get_node(mb.node_id)
                        if n:
                            related_files.add(n.properties.file_path)

        # Pull first chunk from each related file (with slight score penalty)
        for fp in related_files:
            if fp == chunk.source:
                continue
            for m in file_chunks.get(fp, [])[:1]:
                key = (m["source"], m["start_char"], m["end_char"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    extra.append(
                        RetrievedChunk(
                            text=m["text"],
                            source=m["source"],
                            start_char=int(m["start_char"]),
                            end_char=int(m["end_char"]),
                            score=chunk.score * (1.0 - graph_bonus),
                        )
                    )

    # Merge and re-sort by score, keep top_k * 2 for downstream reranker
    combined = initial + extra
    combined.sort(key=lambda c: -c.score)
    return combined[: top_k * 2]
