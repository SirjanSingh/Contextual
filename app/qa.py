from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .loader import load_repo_files
from .chunker import chunk_files, Chunk
from .embedder import Embedder
from .indexer import build_or_load_index, try_load_index
from .retriever import retrieve, RetrievedChunk
from .llm import LLMClient


@dataclass
class QAEngine:
    repo_root: Path
    cache_base: Path
    embedder: Embedder
    llm: LLMClient
    chunk_size: int = 1800
    overlap: int = 250
    top_k: int = 6

    index = None
    metadata: List[Dict] | None = None
    cache_dir: Path | None = None

    def build(self, force_rebuild: bool = False) -> None:
        repo_files = load_repo_files(self.repo_root)
        chunks = chunk_files(repo_files, chunk_size=self.chunk_size, overlap=self.overlap)

        # Try cache first (fast path)
        dim = 768  # Google text-embedding-004
        if not force_rebuild:
            loaded = try_load_index(
                repo_root=self.repo_root,
                chunks=chunks,
                cache_base=self.cache_base,
                dim=dim,
            )
            if loaded is not None:
                self.index, self.metadata, _ = loaded
                return

        # Cache miss (or forced rebuild): compute embeddings and build index.
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        index, metadata, _ = build_or_load_index(
            repo_root=self.repo_root,
            chunks=chunks,
            embeddings=embeddings,
            cache_base=self.cache_base,
            force_rebuild=True,
        )
        self.index, self.metadata = index, metadata


    def ask(self, question: str) -> Tuple[str, List[str]]:
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index not built. Call build() first.")

        chunks = retrieve(
            index=self.index,
            metadata=self.metadata,
            embedder=self.embedder,
            question=question,
            top_k=self.top_k,
        )

        answer = self.llm.answer(question, chunks)

        sources = []
        for c in chunks:
            sources.append(f"{c.source}:{c.start_char}-{c.end_char}")
        return answer, sources
