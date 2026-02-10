from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from .chunker import Chunk


def _repo_fingerprint(repo_root: Path, files: List[str]) -> str:
    """
    Lightweight fingerprint: hash of paths + (mtime_ns, size).
    Good enough for a local dev tool.
    """
    h = hashlib.sha256()
    h.update(str(repo_root.resolve()).encode("utf-8"))
    for rel in sorted(files):
        p = repo_root / rel
        try:
            st = p.stat()
            h.update(rel.encode("utf-8"))
            h.update(str(st.st_mtime_ns).encode("utf-8"))
            h.update(str(st.st_size).encode("utf-8"))
        except OSError:
            # if file disappears mid-run, ignore
            continue
    return h.hexdigest()


def _cache_dir(base_cache: Path, repo_root: Path) -> Path:
    repo_id = hashlib.sha256(str(repo_root.resolve()).encode("utf-8")).hexdigest()[:16]
    return base_cache / repo_id


def try_load_index(
    *,
    repo_root: str | Path,
    chunks: List[Chunk],
    cache_base: str | Path,
    dim: int,
    load_bm25: bool = False,
) -> Tuple[faiss.Index, List[Dict], Path, object | None] | None:
    """Try to load a cached index/metadata for this repo+file set.

    Returns (index, metadata, cache_dir, bm25_index) if a valid cache exists, else None.
    """
    repo_root = Path(repo_root).resolve()
    cache_base = Path(cache_base).resolve()
    cache_base.mkdir(parents=True, exist_ok=True)

    cdir = _cache_dir(cache_base, repo_root)
    index_path = cdir / "index.faiss"
    meta_path = cdir / "metadata.pkl"
    manifest_path = cdir / "manifest.json"
    bm25_path = cdir / "bm25_index.pkl"

    if not (index_path.exists() and meta_path.exists() and manifest_path.exists()):
        return None

    file_list = list({c.source for c in chunks})
    fp = _repo_fingerprint(repo_root, file_list)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fp:
            return None
        if int(manifest.get("dim", -1)) != int(dim):
            return None
        index = faiss.read_index(str(index_path))
        metadata = pickle.loads(meta_path.read_bytes())
        
        # Load BM25 index if requested and exists
        bm25_index = None
        if load_bm25 and bm25_path.exists():
            from .hybrid_search import BM25Index
            bm25_index = BM25Index.load(bm25_path)
        
        return index, metadata, cdir, bm25_index
    except Exception:
        return None


def build_or_load_index(
    *,
    repo_root: str | Path,
    chunks: List[Chunk],
    embeddings: np.ndarray,
    cache_base: str | Path,
    force_rebuild: bool = False,
    build_bm25: bool = False,
) -> Tuple[faiss.Index, List[Dict], Path, object | None]:
    """
    Saves/loads:
      - FAISS index: index.faiss
      - metadata: metadata.pkl (list of dicts aligned with vectors)
      - manifest: manifest.json (fingerprint etc.)
      - BM25 index: bm25_index.pkl (optional, if build_bm25=True)
    """
    repo_root = Path(repo_root).resolve()
    cache_base = Path(cache_base).resolve()
    cache_base.mkdir(parents=True, exist_ok=True)

    cdir = _cache_dir(cache_base, repo_root)
    cdir.mkdir(parents=True, exist_ok=True)

    index_path = cdir / "index.faiss"
    meta_path = cdir / "metadata.pkl"
    manifest_path = cdir / "manifest.json"
    bm25_path = cdir / "bm25_index.pkl"

    file_list = list({c.source for c in chunks})
    fp = _repo_fingerprint(repo_root, file_list)

    if (not force_rebuild) and index_path.exists() and meta_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") == fp and manifest.get("dim") == int(embeddings.shape[1]):
                index = faiss.read_index(str(index_path))
                metadata = pickle.loads(meta_path.read_bytes())
                
                # Load BM25 index if requested and exists
                bm25_index = None
                if build_bm25 and bm25_path.exists():
                    from .hybrid_search import BM25Index
                    bm25_index = BM25Index.load(bm25_path)
                
                return index, metadata, cdir, bm25_index
        except Exception:
            pass  # fall through to rebuild

    # Rebuild
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be (N, D)")
    n, d = embeddings.shape

    # cosine via IP on normalized vectors
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    metadata: List[Dict] = []
    for c in chunks:
        metadata.append({
            "source": c.source,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "text": c.text,  # kept to avoid re-reading files during QA
        })

    faiss.write_index(index, str(index_path))
    meta_path.write_bytes(pickle.dumps(metadata))

    manifest = {
        "repo_root": str(repo_root),
        "fingerprint": fp,
        "dim": int(d),
        "num_chunks": int(n),
        "created_at": int(__import__("time").time()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    # Build BM25 index if requested
    bm25_index = None
    if build_bm25:
        from .hybrid_search import BM25Index
        bm25_index = BM25Index.build(metadata)
        bm25_index.save(bm25_path)
        print(f"[+] BM25 index saved to cache")
    
    # Helpful build diagnostics
    unique_sources = sorted({c.source for c in chunks})
    print(f"[+] Indexed chunks: {len(chunks)}")
    print(f"[+] Indexed files: {len(unique_sources)}")
    print("[+] Sample files:", unique_sources[:10])

    return index, metadata, cdir, bm25_index
