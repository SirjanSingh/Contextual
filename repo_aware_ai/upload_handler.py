"""Handle directory uploads and trigger indexing with progress."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import UploadFile

from .chunker import chunk_files
from .embedder import Embedder
from .indexer import build_or_load_index
from .llm import LLMClient
from .loader import load_repo_files
from .progress_tracker import UploadProgress


async def save_uploaded_files(files: list[UploadFile], progress: UploadProgress) -> str:
    """Save uploaded files to a temp directory preserving relative paths."""
    temp_dir = Path(tempfile.mkdtemp(prefix="contextual_upload_"))
    progress.update(stage="uploading", progress=0)

    for i, file in enumerate(files):
        relative_path = file.filename or f"file_{i}"
        target = temp_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        target.write_bytes(content)

        pct = ((i + 1) / len(files)) * 30  # uploading is 0-30%
        progress.update(
            progress=pct,
            files_processed=i + 1,
            current_file=relative_path,
        )

    progress.update(repo_path=str(temp_dir))
    return str(temp_dir)


def process_repository_sync(
    repo_path: str,
    cache_base: str,
    progress: UploadProgress,
    embedder: Embedder,
    llm: LLMClient,
) -> None:
    """Run the full indexing pipeline with progress updates (blocking)."""

    repo_root = Path(repo_path)
    cache = Path(cache_base)

    # Stage: scanning (30-50%)
    progress.update(stage="scanning", progress=30, current_file="Scanning directory...")
    repo_files = load_repo_files(repo_root)
    progress.update(
        progress=40,
        total_files=len(repo_files),
        current_file=f"Found {len(repo_files)} files",
    )

    # Stage: parsing (50-80%)
    progress.update(stage="parsing", progress=50, current_file="Chunking files...")
    chunks = chunk_files(repo_files, chunk_size=1800, overlap=250)
    progress.update(
        progress=65,
        chunks_created=len(chunks),
        current_file=f"Created {len(chunks)} chunks",
    )

    # Stage: embedding (80-95%)
    progress.update(stage="embedding", progress=70, current_file="Generating embeddings...")
    texts = [c.text for c in chunks]

    # Embed in batches with progress
    batch_size = 100
    import numpy as np

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_emb = embedder.embed_texts(batch)
        all_embeddings.append(batch_emb)

        pct = 70 + ((i + batch_size) / max(len(texts), 1)) * 20
        progress.update(
            progress=min(pct, 90),
            current_file=f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks",
        )

    embeddings = (
        np.vstack(all_embeddings)
        if all_embeddings
        else np.empty((0, embedder.dimension), dtype=np.float32)
    )

    # Stage: indexing (95-100%)
    progress.update(stage="indexing", progress=92, current_file="Building FAISS index...")
    index, metadata, cache_dir, bm25 = build_or_load_index(
        repo_root=repo_root,
        chunks=chunks,
        embeddings=embeddings,
        cache_base=cache,
        force_rebuild=True,
        build_bm25=True,
    )

    progress.update(
        stage="complete",
        progress=100,
        current_file="Repository indexed successfully",
    )

    return {
        "repo_path": str(repo_root),
        "cache_dir": str(cache_dir),
        "total_files": len(repo_files),
        "total_chunks": len(chunks),
        "index_size": index.ntotal if index else 0,
    }
