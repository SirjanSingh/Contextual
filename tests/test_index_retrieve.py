"""End-to-end indexer + retriever test using a fake embedder.

We don't hit the real Gemini API in tests — the FakeEmbedder fixture in
conftest.py provides a deterministic embedding good enough to verify the
indexer / retriever / cache code paths.
"""

from __future__ import annotations

from repo_aware_ai.chunker import chunk_files
from repo_aware_ai.indexer import build_or_load_index, try_load_index
from repo_aware_ai.loader import load_repo_files
from repo_aware_ai.retriever import retrieve


def test_build_then_retrieve_then_cache_hits(tiny_repo, fake_embedder, tmp_path):
    cache = tmp_path / "cache"

    repo_files = load_repo_files(tiny_repo)
    chunks = chunk_files(repo_files, chunk_size=400, overlap=50)
    assert chunks, "expected at least one chunk from the tiny repo"

    embeddings = fake_embedder.embed_texts([c.text for c in chunks])
    index, metadata, cdir, _bm25 = build_or_load_index(
        repo_root=tiny_repo,
        chunks=chunks,
        embeddings=embeddings,
        cache_base=cache,
        force_rebuild=True,
        build_bm25=False,
    )
    # FAISS rows must align 1:1 with metadata.
    assert index.ntotal == len(metadata) == len(chunks)

    # A retrieval query about the symbols actually in the repo should hit
    # one of the matching files.
    results = retrieve(
        index=index,
        metadata=metadata,
        embedder=fake_embedder,
        question="add function",
        top_k=3,
    )
    assert results
    sources = {r.source for r in results}
    # Either the python file or the markdown should be retrieved — both
    # contain relevant tokens. We just want *some* sensible result.
    assert any("alpha" in s or "README" in s or "beta" in s for s in sources)

    # Second call without force-rebuild should hit the cache and return
    # equivalent shape.
    loaded = try_load_index(
        repo_root=tiny_repo,
        chunks=chunks,
        cache_base=cache,
        dim=fake_embedder.dimension,
        load_bm25=False,
    )
    assert loaded is not None, "cache should have been written and reloadable"
    cached_index, cached_meta, _cdir, _bm25 = loaded
    assert cached_index.ntotal == len(cached_meta) == len(metadata)


def test_cache_invalidates_on_file_change(tiny_repo, fake_embedder, tmp_path):
    cache = tmp_path / "cache"

    repo_files = load_repo_files(tiny_repo)
    chunks = chunk_files(repo_files, chunk_size=400, overlap=50)
    embeddings = fake_embedder.embed_texts([c.text for c in chunks])
    build_or_load_index(
        repo_root=tiny_repo,
        chunks=chunks,
        embeddings=embeddings,
        cache_base=cache,
        force_rebuild=True,
        build_bm25=False,
    )

    # Mutate a tracked file → fingerprint should change → cache should miss.
    (tiny_repo / "src" / "alpha.py").write_text("def changed():\n    return 42\n", encoding="utf-8")

    new_files = load_repo_files(tiny_repo)
    new_chunks = chunk_files(new_files, chunk_size=400, overlap=50)

    loaded = try_load_index(
        repo_root=tiny_repo,
        chunks=new_chunks,
        cache_base=cache,
        dim=fake_embedder.dimension,
        load_bm25=False,
    )
    assert loaded is None, "stale cache should not be served"
