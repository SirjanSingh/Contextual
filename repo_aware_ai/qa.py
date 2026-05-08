"""End-to-end RAG pipeline orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("rai.qa")

from .chunker import chunk_files
from .compressor import ContextCompressor
from .conversation import ConversationHistory
from .embedder import Embedder
from .indexer import build_or_load_index, try_load_index
from .llm import LLMClient
from .loader import load_repo_files
from .multi_query import MultiQueryGenerator
from .query_expander import QueryExpander
from .reranker import Reranker
from .retriever import RetrievedChunk, retrieve


@dataclass
class QAEngine:
    repo_root: Path
    cache_base: Path
    embedder: Embedder
    llm: LLMClient
    chunk_size: int = 1800
    overlap: int = 250
    top_k: int = 6
    use_reranker: bool = True
    use_conversation: bool = True
    use_hybrid_search: bool = True
    use_query_expansion: bool = True
    use_compression: bool = True
    use_ast_chunking: bool = False
    use_multi_query: bool = True
    use_graph_context: bool = True

    index = None
    metadata: list[dict] | None = None
    cache_dir: Path | None = None
    _reranker: Reranker | None = None
    _conversation: ConversationHistory | None = None
    _bm25_index: object | None = None
    _query_expander: QueryExpander | None = None
    _compressor: ContextCompressor | None = None
    _multi_query: MultiQueryGenerator | None = None
    _repo_map: object | None = None
    _repo_graph: object | None = None

    # ──────────────────────────────────────────────
    # Index lifecycle
    # ──────────────────────────────────────────────
    def build(self, force_rebuild: bool = False) -> None:
        repo_files = load_repo_files(self.repo_root)
        chunks = chunk_files(
            repo_files,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            use_ast=self.use_ast_chunking,
        )

        dim = self.embedder.dimension
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
                self._build_repo_map(repo_files, force_rebuild=False)
                return

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
        self.index = index
        self.metadata = metadata
        self.cache_dir = cache_dir
        self._bm25_index = bm25_index
        self._build_repo_map(repo_files, force_rebuild=force_rebuild)

    def _build_repo_map(self, repo_files, force_rebuild: bool = False) -> None:
        """Build repo map graph. Non-fatal if it fails (optional dep)."""
        if self.cache_dir is None:
            return
        try:
            from .repo_map import build_repo_map

            self._repo_map, self._repo_graph = build_repo_map(
                repo_files=repo_files,
                cache_dir=self.cache_dir,
                force_rebuild=force_rebuild,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[QA] repo map build failed (non-fatal): %s", e)

    # ──────────────────────────────────────────────
    # Query pipeline
    # ──────────────────────────────────────────────
    def _prepare(self, question: str) -> tuple[list[RetrievedChunk], list[str], str]:
        """Run everything up to (but not including) the final LLM answer.

        Returns (final_chunks, sources, conversation_context).
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index not built. Call build() first.")

        logger.info("[QA] ask() called: question=%r", question)

        if self.use_conversation and self._conversation is None:
            self._conversation = ConversationHistory()

        retrieve_k = self.top_k * 3 if self.use_reranker else self.top_k

        # 1. Decompose compound questions
        base_queries = [question]
        if self.use_multi_query:
            t0 = time.time()
            if self._multi_query is None:
                self._multi_query = MultiQueryGenerator.from_llm_client(self.llm)
            base_queries = self._multi_query.generate(question)
            logger.info(
                "[QA] multi_query: %d sub-queries (%.0fms)",
                len(base_queries),
                (time.time() - t0) * 1000,
            )

        # 2. Expand each base query
        queries: list[str] = []
        if self.use_query_expansion:
            t0 = time.time()
            if self._query_expander is None:
                self._query_expander = QueryExpander.from_llm_client(self.llm)
            for bq in base_queries:
                queries.extend(self._query_expander.expand(bq))
            logger.info(
                "[QA] query_expansion: %d queries (%.0fms)",
                len(queries),
                (time.time() - t0) * 1000,
            )
        else:
            queries = list(base_queries)

        # Dedupe queries
        seen_q: set = set()
        queries = [
            q for q in queries if not (q.strip().lower() in seen_q or seen_q.add(q.strip().lower()))
        ]

        # 3. Retrieve per query
        all_chunks: list[RetrievedChunk] = []
        seen_keys: set = set()
        t0 = time.time()
        for q in queries:
            if self.use_hybrid_search and self._bm25_index is not None:
                from .hybrid_search import hybrid_retrieve

                q_chunks = hybrid_retrieve(
                    faiss_index=self.index,
                    bm25_index=self._bm25_index,
                    metadata=self.metadata,
                    embedder=self.embedder,
                    question=q,
                    top_k=retrieve_k if self.use_reranker else self.top_k,
                    retrieve_k=retrieve_k * 2,
                )
            elif self.use_graph_context and self._repo_graph is not None:
                from .retriever import retrieve_with_graph_context

                q_chunks = retrieve_with_graph_context(
                    index=self.index,
                    metadata=self.metadata,
                    embedder=self.embedder,
                    question=q,
                    top_k=retrieve_k,
                    repo_graph=self._repo_graph,
                    repo_map=self._repo_map,
                )
            else:
                q_chunks = retrieve(
                    index=self.index,
                    metadata=self.metadata,
                    embedder=self.embedder,
                    question=q,
                    top_k=retrieve_k,
                )

            for c in q_chunks:
                key = (c.source, c.start_char, c.end_char)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_chunks.append(c)

        chunks = all_chunks
        logger.info(
            "[QA] retrieval: %d unique chunks (%.0fms)", len(chunks), (time.time() - t0) * 1000
        )

        # 4. Rerank
        if self.use_reranker and len(chunks) > self.top_k:
            t0 = time.time()
            if self._reranker is None:
                self._reranker = Reranker()
            chunks = self._reranker.rerank(question, chunks, top_k=self.top_k)
            logger.info(
                "[QA] reranking: %d chunks (%.0fms)", len(chunks), (time.time() - t0) * 1000
            )

        # 5. Compress
        if self.use_compression:
            t0 = time.time()
            if self._compressor is None:
                self._compressor = ContextCompressor.from_llm_client(self.llm)
            chunks = self._compressor.compress(question, chunks)
            logger.info(
                "[QA] compression: %d chunks (%.0fms)", len(chunks), (time.time() - t0) * 1000
            )

        # 6. Conversation context
        conversation_context = ""
        if self.use_conversation and self._conversation and not self._conversation.is_empty:
            conversation_context = self._conversation.get_context(include_sources=False)

        sources = [f"{c.source}:{c.start_char}-{c.end_char}" for c in chunks]
        return chunks, sources, conversation_context

    def ask(self, question: str) -> tuple[str, list[str]]:
        t_overall = time.time()
        chunks, sources, conv_ctx = self._prepare(question)

        t0 = time.time()
        answer = self.llm.answer(question, chunks, conv_ctx)
        logger.info("[QA] llm.answer: (%.0fms)", (time.time() - t0) * 1000)

        if self.use_conversation and self._conversation:
            self._conversation.add_turn(question, answer, sources)

        logger.info("[QA] total: %.2fs", time.time() - t_overall)
        return answer, sources

    def stream_ask(self, question: str) -> tuple[Iterator[str], list[str]]:
        """Stream the answer as it is generated.

        Returns (text_iterator, sources). The iterator yields incremental
        text chunks; the conversation history is updated after the iterator
        is exhausted.
        """
        chunks, sources, conv_ctx = self._prepare(question)

        accumulated: list[str] = []

        def gen() -> Iterator[str]:
            for piece in self.llm.stream_answer(question, chunks, conv_ctx):
                accumulated.append(piece)
                yield piece
            if self.use_conversation and self._conversation:
                self._conversation.add_turn(question, "".join(accumulated), sources)

        return gen(), sources

    def clear_conversation(self) -> None:
        if self._conversation:
            self._conversation.clear()
