from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str      # relative file path
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 1800,
    overlap: int = 250,
) -> List[Chunk]:
    """
    Character-based chunking with overlap.
    - For codebases, char-chunking is robust and fast.
    - We'll switch to token-based later if needed.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Chunk] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end]
        if piece.strip():
            chunks.append(Chunk(text=piece, source=source, start_char=start, end_char=end))

        # advance
        start = end - overlap
        if start < 0:
            start = 0

        if end == n:
            break

    return chunks


def chunk_files(repo_files, chunk_size: int = 1800, overlap: int = 250, use_ast: bool = False) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    
    if use_ast:
        from .ast_chunker import ast_chunk_python
    
    for rf in repo_files:
        if use_ast and rf.path.endswith(".py"):
            all_chunks.extend(ast_chunk_python(rf.text, rf.path, max_chunk_size=chunk_size))
        else:
            all_chunks.extend(chunk_text(rf.text, rf.path, chunk_size=chunk_size, overlap=overlap))
    
    return all_chunks
