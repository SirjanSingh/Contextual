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
from .reranker import Reranker
from .conversation import ConversationHistory


@dataclass
class QAEngine:
    repo_root: Path
    cache_base: Path
    embedder: Embedder
    llm: LLMClient
    chunk_size: int = 1800
    overlap: int = 250
    top_k: int = 6
    use_reranker: bool = True  # Enable reranking by default
    use_conversation: bool = True  # Enable conversation history by default
    use_hybrid_search: bool = True  # Enable hybrid search by default

    index = None
    metadata: List[Dict] | None = None
    cache_dir: Path | None = None
    _reranker: Reranker | None = None
    _conversation: ConversationHistory | None = None
    _bm25_index: object | None = None  # BM25Index from hybrid_search

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
                load_bm25=self.use_hybrid_search,
            )
            if loaded is not None:
                self.index, self.metadata, self.cache_dir, self._bm25_index = loaded
                return

        # Cache miss (or forced rebuild): compute embeddings and build index.
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        index, metadata, cache_dir, bm25_index = build_or_load_index(
            repo_root=self.repo_root,
            chunks=chunks,
            embeddings=embeddings,
            cache_base=self.cache_base,
            force_rebuild=True,
            build_bm25=self.use_hybrid_search,
        )
        self.index, self.metadata, self.cache_dir, self._bm25_index = index, metadata, cache_dir, bm25_index


    def ask(self, question: str) -> Tuple[str, List[str]]:
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index not built. Call build() first.")

        # Initialize conversation history if enabled
        if self.use_conversation and self._conversation is None:
            self._conversation = ConversationHistory()
        
        # Retrieve more chunks if using reranker (will be filtered down)
        retrieve_k = self.top_k * 3 if self.use_reranker else self.top_k
        
        # Use hybrid search if enabled and BM25 index exists
        if self.use_hybrid_search and self._bm25_index is not None:
            from .hybrid_search import hybrid_retrieve
            chunks = hybrid_retrieve(
                faiss_index=self.index,
                bm25_index=self._bm25_index,
                metadata=self.metadata,
                embedder=self.embedder,
                question=question,
                top_k=retrieve_k if self.use_reranker else self.top_k,
                retrieve_k=retrieve_k * 2,  # Get more candidates for RRF
            )
        else:
            # Standard vector-only retrieval
            chunks = retrieve(
                index=self.index,
                metadata=self.metadata,
                embedder=self.embedder,
                question=question,
                top_k=retrieve_k,
            )
        
        # Rerank if enabled
        if self.use_reranker and len(chunks) > self.top_k:
            if self._reranker is None:
                self._reranker = Reranker()
            chunks = self._reranker.rerank(question, chunks, top_k=self.top_k)
        
        # Get conversation context if enabled
        conversation_context = ""
        if self.use_conversation and self._conversation and not self._conversation.is_empty:
            conversation_context = self._conversation.get_context(include_sources=False)

        answer = self.llm.answer(question, chunks, conversation_context)

        sources = []
        for c in chunks:
            sources.append(f"{c.source}:{c.start_char}-{c.end_char}")
        
        # Add to conversation history if enabled
        if self.use_conversation and self._conversation:
            self._conversation.add_turn(question, answer, sources)
        
        return answer, sources
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        if self._conversation:
            self._conversation.clear()
