"""FastAPI web server for Contextual RAG system."""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, Request, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rai.server")

from .config import get_config
from .embedder import Embedder
from .llm import LLMClient
from .qa import QAEngine
from .progress_tracker import create_progress, get_progress, remove_progress
from .upload_handler import save_uploaded_files, process_repository_sync

# ──────────────────────────────────────────────
# App & state
# ──────────────────────────────────────────────

# Extension-friendly version identifier
VERSION = "2.0.0"

# Configurable port — extension sets RAI_PORT env var before spawning
PORT = int(os.environ.get("RAI_PORT", "8360"))

# Where FAISS index/cache files are stored.
# The VS Code extension sets RAI_DATA_DIR to its globalStorageUri (a writable
# path that persists across updates and is isolated from the user's workspace).
# Falls back to a local data/index path for CLI / dev use.
_DATA_DIR = Path(os.environ.get("RAI_DATA_DIR", "data/index"))

app = FastAPI(title="Contextual – Neural Code Nexus", version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ──
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.0f}ms)")
    return response


# Global engine state
_engine: Optional[QAEngine] = None
_engine_lock = threading.Lock()
_index_status = "none"  # none | building | ready | error
_index_info: dict = {}


def _get_engine() -> Optional[QAEngine]:
    return _engine


# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    repo_path: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


class IndexStatus(BaseModel):
    status: str
    info: dict = {}


class HealthResponse(BaseModel):
    status: str
    version: str = VERSION
    model: str = ""
    embedding_model: str = ""


class UploadResponse(BaseModel):
    upload_id: str
    total_files: int
    message: str


class IndexDirectoryRequest(BaseModel):
    repo_path: str  # Absolute path on disk


class ChunkResult(BaseModel):
    source: str
    text: str
    start_char: int
    end_char: int
    score: float


class SearchResponse(BaseModel):
    chunks: List[ChunkResult]


class FileContextResponse(BaseModel):
    chunks: List[ChunkResult]
    related_files: List[str]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        config = get_config()
        logger.debug("Health check OK")
        return HealthResponse(
            status="ok",
            version=VERSION,
            model=config.gemini_model,
            embedding_model=config.embedding_model,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status=f"error: {e}", version=VERSION)


@app.get("/index/status", response_model=IndexStatus)
async def index_status():
    return IndexStatus(status=_index_status, info=_index_info)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    engine = _get_engine()
    if engine is None or engine.index is None:
        raise HTTPException(status_code=400, detail="No repository indexed yet. Upload a directory first.")

    logger.info(f"Query: {req.question[:100]}")
    t0 = time.time()
    try:
        answer, sources = engine.ask(req.question)
        logger.info(f"Query answered in {(time.time()-t0)*1000:.0f}ms, {len(sources)} sources")
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """SSE stream: yields partial answer tokens."""
    engine = _get_engine()
    if engine is None or engine.index is None:
        raise HTTPException(status_code=400, detail="No repository indexed yet.")

    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Run retrieval in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            answer, sources = await loop.run_in_executor(None, engine.ask, req.question)

            # Stream tokens word-by-word (backend doesn't yet support true streaming;
            # this simulates it — upgrade later when google-genai streaming is wired up)
            words = answer.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == len(words) - 1 else word + " "
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0)  # yield control

            # Send sources as final event
            import json
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
            yield "event: done\ndata: \n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/search", response_model=SearchResponse)
async def semantic_search(req: QueryRequest):
    """Return top-k chunks without LLM answer (for hover/codelens)."""
    engine = _get_engine()
    if engine is None or engine.index is None:
        raise HTTPException(status_code=400, detail="No repository indexed yet.")

    try:
        from .retriever import retrieve

        top_k = 8  # More results for search (no reranker overhead)
        if engine.use_hybrid_search and engine._bm25_index is not None:
            from .hybrid_search import hybrid_retrieve
            chunks = hybrid_retrieve(
                faiss_index=engine.index,
                bm25_index=engine._bm25_index,
                metadata=engine.metadata,
                embedder=engine.embedder,
                question=req.question,
                top_k=top_k,
                retrieve_k=top_k * 2,
            )
        else:
            chunks = retrieve(
                index=engine.index,
                metadata=engine.metadata,
                embedder=engine.embedder,
                question=req.question,
                top_k=top_k,
            )

        return SearchResponse(
            chunks=[
                ChunkResult(
                    source=c.source,
                    text=c.text,
                    start_char=c.start_char,
                    end_char=c.end_char,
                    score=c.score,
                )
                for c in chunks
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/context/file", response_model=FileContextResponse)
async def file_context(file_path: str):
    """Return all indexed chunks for a specific file + related files."""
    engine = _get_engine()
    if engine is None or engine.metadata is None:
        raise HTTPException(status_code=400, detail="No repository indexed yet.")

    try:
        # Normalize path separators for comparison
        norm_path = file_path.replace("\\", "/")

        # Filter metadata for this file
        file_chunks = [
            ChunkResult(
                source=m["source"],
                text=m["text"],
                start_char=int(m["start_char"]),
                end_char=int(m["end_char"]),
                score=1.0,
            )
            for m in engine.metadata
            if m.get("source", "").replace("\\", "/").endswith(norm_path)
            or norm_path.endswith(m.get("source", "").replace("\\", "/"))
        ]

        # Find related files via semantic search on first chunk's text
        related_files: List[str] = []
        if file_chunks and engine.index is not None:
            from .retriever import retrieve
            sample_text = file_chunks[0].text[:500]
            related = retrieve(
                index=engine.index,
                metadata=engine.metadata,
                embedder=engine.embedder,
                question=sample_text,
                top_k=10,
            )
            seen = set()
            for r in related:
                src = r.source
                if src not in seen and not norm_path.endswith(src.replace("\\", "/")):
                    seen.add(src)
                    related_files.append(src)

        return FileContextResponse(chunks=file_chunks, related_files=related_files[:5])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/directory")
async def index_directory(req: IndexDirectoryRequest):
    """Index a local directory by path (no file upload needed)."""
    global _engine, _index_status, _index_info
    repo_path = Path(req.repo_path)

    if not repo_path.exists() or not repo_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.repo_path}")

    logger.info(f"Index directory requested: {repo_path}")
    progress = create_progress(total_files=0)

    def _index_worker():
        global _engine, _index_status, _index_info
        try:
            _index_status = "building"
            logger.info(f"[indexer] Building index for {repo_path}...")
            t0 = time.time()
            embedder = Embedder()
            llm = LLMClient()

            engine = QAEngine(
                repo_root=repo_path,
                cache_base=_DATA_DIR,
                embedder=embedder,
                llm=llm,
            )
            engine.build(force_rebuild=False)

            with _engine_lock:
                _engine = engine

            chunk_count = len(engine.metadata) if engine.metadata else 0
            elapsed = time.time() - t0
            _index_info = {
                "repo_path": str(repo_path),
                "chunk_count": chunk_count,
                "build_time_s": round(elapsed, 2),
            }
            _index_status = "ready"
            logger.info(f"[indexer] Index ready: {chunk_count} chunks in {elapsed:.1f}s")
        except Exception as e:
            _index_status = "error"
            _index_info = {"error": str(e)}
            logger.exception(f"[indexer] Index failed: {e}")

    thread = threading.Thread(target=_index_worker, daemon=True)
    thread.start()

    return {
        "status": "indexing",
        "upload_id": progress.upload_id,
        "message": f"Indexing started for {req.repo_path}",
    }


@app.post("/index/rebuild")
async def rebuild_index():
    global _index_status
    engine = _get_engine()
    if engine is None:
        raise HTTPException(status_code=400, detail="No repository loaded.")
    _index_status = "building"
    try:
        engine.build(force_rebuild=True)
        _index_status = "ready"
        return {"status": "ok", "message": "Index rebuilt successfully"}
    except Exception as e:
        _index_status = "error"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/directory", response_model=UploadResponse)
async def upload_directory(files: List[UploadFile] = File(...)):
    global _engine, _index_status, _index_info

    progress = create_progress(total_files=len(files))

    # Save files async
    repo_path = await save_uploaded_files(files, progress)

    # Run indexing in a background thread
    def _index_worker():
        global _engine, _index_status, _index_info
        try:
            _index_status = "building"
            embedder = Embedder()
            llm = LLMClient()

            result = process_repository_sync(
                repo_path=repo_path,
                cache_base=_DATA_DIR,
                progress=progress,
                embedder=embedder,
                llm=llm,
            )
            _index_info = result or {}

            # Build a QAEngine for serving queries
            engine = QAEngine(
                repo_root=Path(repo_path),
                cache_base=_DATA_DIR,
                embedder=embedder,
                llm=llm,
            )
            engine.build(force_rebuild=False)  # Will load from cache we just built

            with _engine_lock:
                _engine = engine
            _index_status = "ready"

        except Exception as e:
            progress.update(stage="error", current_file=str(e))
            progress.errors.append(str(e))
            _index_status = "error"

    thread = threading.Thread(target=_index_worker, daemon=True)
    thread.start()

    return UploadResponse(
        upload_id=progress.upload_id,
        total_files=len(files),
        message="Upload started. Track progress with /upload/progress/{upload_id}",
    )


@app.get("/upload/progress/{upload_id}")
async def upload_progress(upload_id: str):
    progress = get_progress(upload_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    return progress.to_dict()


@app.websocket("/ws/indexing/{upload_id}")
async def ws_indexing(websocket: WebSocket, upload_id: str):
    await websocket.accept()
    try:
        while True:
            progress = get_progress(upload_id)
            if progress is None:
                await websocket.send_json({"error": "Upload not found"})
                break

            await websocket.send_json(progress.to_dict())

            if progress.stage in ("complete", "error"):
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.delete("/repository/clear")
async def clear_repository():
    global _engine, _index_status, _index_info
    with _engine_lock:
        if _engine:
            _engine.clear_conversation()
        _engine = None
    _index_status = "none"
    _index_info = {}
    return {"status": "ok", "message": "Repository cleared"}


# ──────────────────────────────────────────────
# Graph endpoints (for Repo Map visualization)
# ──────────────────────────────────────────────

@app.get("/graph/dependencies")
async def dependency_graph():
    """Return file-to-file dependency edges from import analysis."""
    engine = _get_engine()
    if engine is None or engine.metadata is None:
        raise HTTPException(status_code=400, detail="No repository indexed yet.")

    import re
    from collections import defaultdict

    # Build file nodes
    file_chunks: dict = defaultdict(list)
    for m in engine.metadata:
        file_chunks[m.get("source", "unknown")].append(m)

    nodes = [
        {"id": src, "label": Path(src).name, "type": "file", "chunkCount": len(chunks)}
        for src, chunks in file_chunks.items()
    ]

    # Parse import edges from chunk text
    edges = []
    seen_edges = set()
    import_pattern = re.compile(
        r"^(?:import|from)\s+([\w\.]+)", re.MULTILINE
    )
    for src, chunks in file_chunks.items():
        for chunk in chunks:
            text = chunk.get("text", "")
            for match in import_pattern.finditer(text):
                module = match.group(1).split(".")[0]
                # Match to actual files in index
                for target in file_chunks:
                    if Path(target).stem == module or Path(target).name == module:
                        edge_key = (src, target)
                        if edge_key not in seen_edges and src != target:
                            seen_edges.add(edge_key)
                            edges.append({"source": src, "target": target, "type": "import"})

    return {"nodes": nodes, "edges": edges}


@app.get("/graph/clusters")
async def semantic_clusters():
    """Return k-means clusters of FAISS vectors."""
    engine = _get_engine()
    if engine is None or engine.index is None or engine.metadata is None:
        raise HTTPException(status_code=400, detail="No repository indexed yet.")

    try:
        import numpy as np
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return {"clusters": [], "error": "sklearn not installed"}

        n_vectors = engine.index.ntotal
        k = min(8, max(2, n_vectors // 10))

        # Reconstruct vectors from FAISS index
        vectors = np.zeros((n_vectors, engine.index.d), dtype=np.float32)
        for i in range(n_vectors):
            engine.index.reconstruct(i, vectors[i])

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)

        clusters = []
        for cluster_id in range(k):
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            files_in_cluster = list({
                engine.metadata[i].get("source", "unknown")
                for i in cluster_indices
                if i < len(engine.metadata)
            })
            clusters.append({
                "id": cluster_id,
                "centroid_label": f"Cluster {cluster_id}",
                "files": files_in_cluster,
                "size": len(cluster_indices),
            })

        return {"clusters": clusters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Auto-index on startup (for dev_server.py)
# ──────────────────────────────────────────────

_auto_index_path = os.environ.get("RAI_AUTO_INDEX")
if _auto_index_path:
    @app.on_event("startup")
    async def _auto_index():
        logger.info(f"Auto-indexing: {_auto_index_path}")
        import httpx
        port = int(os.environ.get("RAI_PORT", "8360"))
        async with httpx.AsyncClient() as client:
            await client.post(
                f"http://127.0.0.1:{port}/index/directory",
                json={"repo_path": _auto_index_path},
                timeout=5.0,
            )


# ──────────────────────────────────────────────
# /dev/ debug endpoint
# ──────────────────────────────────────────────

@app.get("/dev/debug")
async def dev_debug():
    """Debug info for the test dashboard."""
    engine = _get_engine()
    return {
        "index_status": _index_status,
        "index_info": _index_info,
        "engine_loaded": engine is not None,
        "chunk_count": len(engine.metadata) if engine and engine.metadata else 0,
        "has_bm25": engine._bm25_index is not None if engine else False,
        "use_reranker": engine.use_reranker if engine else None,
        "use_hybrid_search": engine.use_hybrid_search if engine else None,
        "use_query_expansion": engine.use_query_expansion if engine else None,
        "use_compression": engine.use_compression if engine else None,
        "use_multi_query": engine.use_multi_query if engine else None,
        "version": VERSION,
    }


# ──────────────────────────────────────────────
# /dev/ test dashboard (separate from prod frontend)
# ──────────────────────────────────────────────

@app.get("/dev/", response_class=HTMLResponse)
@app.get("/dev", response_class=HTMLResponse)
async def dev_dashboard():
    """Test dashboard — does NOT touch the production frontend."""
    return _DEV_DASHBOARD_HTML


# ──────────────────────────────────────────────
# Serve frontend static files (if built)
# ──────────────────────────────────────────────

_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")


# ──────────────────────────────────────────────
# Test Dashboard HTML (self-contained)
# ──────────────────────────────────────────────

_DEV_DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Repo AI — Dev Dashboard</title>
  <style>
    :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --fg: #c9d1d9; --accent: #58a6ff; --green: #3fb950; --red: #f85149; --yellow: #d29922; }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: var(--bg); color: var(--fg); font-family: 'Segoe UI', -apple-system, sans-serif; font-size: 14px; }
    .layout { display: grid; grid-template-columns: 320px 1fr; grid-template-rows: auto 1fr; height: 100vh; }
    header { grid-column: 1 / -1; padding: 12px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 16px; background: var(--card); }
    header h1 { font-size: 16px; color: var(--accent); font-weight: 700; }
    .status-badge { padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
    .status-none { background: var(--border); color: var(--fg); }
    .status-building { background: #d2992233; color: var(--yellow); }
    .status-ready { background: #3fb95033; color: var(--green); }
    .status-error { background: #f8514933; color: var(--red); }
    .sidebar { border-right: 1px solid var(--border); overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 8px; }
    .main { overflow-y: auto; padding: 16px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
    .card h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--accent); margin-bottom: 10px; }
    .btn { background: #1f6feb; color: white; border: none; border-radius: 6px; padding: 8px 16px; cursor: pointer; font-size: 13px; width: 100%; text-align: left; transition: background 0.2s; }
    .btn:hover { background: #388bfd; }
    .btn:disabled { opacity: 0.5; cursor: default; }
    .btn.danger { background: #da3633; }
    .btn.danger:hover { background: #f85149; }
    .btn.secondary { background: var(--border); color: var(--fg); }
    .btn.secondary:hover { background: #484f58; }
    input, textarea { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; color: var(--fg); font-size: 13px; width: 100%; font-family: inherit; }
    input:focus, textarea:focus { outline: none; border-color: var(--accent); }
    textarea { resize: vertical; min-height: 60px; font-family: 'Cascadia Code', 'Fira Code', monospace; }
    /* ── Generic result box ── */
    .result { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 12px; margin-top: 12px; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 12px; white-space: pre-wrap; overflow-x: auto; max-height: 400px; overflow-y: auto; }
    /* ── Chunk cards ── */
    .chunk-list { display: flex; flex-direction: column; gap: 12px; margin-top: 12px; }
    .chunk-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; transition: border-color 0.2s; }
    .chunk-card:hover { border-color: var(--accent); }
    .chunk-header { display: flex; align-items: center; gap: 10px; padding: 8px 14px; border-bottom: 1px solid var(--border); background: #1c2128; }
    .chunk-file { font-size: 13px; font-weight: 600; color: var(--accent); font-family: 'Cascadia Code', monospace; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .chunk-score { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; background: #1f6feb33; color: var(--accent); white-space: nowrap; }
    .chunk-range { font-size: 10px; color: #8b949e; white-space: nowrap; }
    .chunk-code { margin: 0; padding: 12px 14px; background: #0d1117; color: #c9d1d9; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 12px; overflow-x: auto; white-space: pre; line-height: 1.6; max-height: 260px; overflow-y: auto; }
    .chunk-code::-webkit-scrollbar { width: 6px; height: 6px; }
    .chunk-code::-webkit-scrollbar-track { background: #161b22; }
    .chunk-code::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    /* ── Query answer card ── */
    .answer-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-top: 12px; }
    .answer-header { padding: 10px 14px; background: #1c2128; border-bottom: 1px solid var(--border); font-size: 12px; font-weight: 700; color: var(--green); letter-spacing: 0.5px; text-transform: uppercase; }
    .answer-body { padding: 16px; font-size: 14px; line-height: 1.7; white-space: pre-wrap; color: var(--fg); }
    .sources-list { padding: 0 14px 14px; display: flex; flex-wrap: wrap; gap: 6px; }
    .source-chip { font-size: 11px; padding: 2px 10px; border-radius: 12px; background: #1f6feb22; border: 1px solid #1f6feb55; color: var(--accent); font-family: 'Cascadia Code', monospace; }
    /* ── JSON syntax highlight ── */
    .json-key { color: #79c0ff; }
    .json-str { color: #a5d6ff; }
    .json-num { color: #ffa657; }
    .json-bool { color: #ff7b72; }
    .json-null { color: #8b949e; }
    .result.error { border-color: var(--red); color: var(--red); }
    .result.success { border-color: var(--green); }
    .timing { font-size: 11px; color: #8b949e; margin-top: 6px; }
    .tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
    .tab { padding: 8px 20px; cursor: pointer; border-bottom: 2px solid transparent; color: #8b949e; font-size: 13px; font-weight: 500; }
    .tab:hover { color: var(--fg); }
    .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .stat { text-align: center; }
    .stat .val { font-size: 28px; font-weight: 700; color: var(--accent); }
    .stat .label { font-size: 11px; color: #8b949e; }
    .log-line { padding: 2px 0; border-bottom: 1px solid #21262d; font-family: monospace; font-size: 11px; }
    #repomap-canvas { width: 100%; height: 500px; border-radius: 8px; background: #0d1117; display: block; }
    .input-group { display: flex; gap: 8px; margin-bottom: 8px; }
    .input-group input { flex: 1; }
    .sep { border-top: 1px solid var(--border); margin: 8px 0; }
    /* ── Info cards ── */
    .info-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-top: 12px; }
    .info-card-header { display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: #1c2128; border-bottom: 1px solid var(--border); }
    .info-card-title { font-size: 13px; font-weight: 700; color: var(--fg); flex: 1; }
    .info-card-body { padding: 14px; }
    .pill { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; padding: 3px 10px; border-radius: 12px; font-family: 'Cascadia Code', monospace; white-space: nowrap; }
    .pill-blue { background: #1f6feb22; border: 1px solid #1f6feb55; color: var(--accent); }
    .pill-green { background: #3fb95022; border: 1px solid #3fb95055; color: var(--green); }
    .pill-yellow { background: #d2992222; border: 1px solid #d2992255; color: var(--yellow); }
    .pill-red { background: #f8514922; border: 1px solid #f8514955; color: var(--red); }
    .pill-gray { background: #30363d44; border: 1px solid #30363d; color: #8b949e; }
    .pill-purple { background: #bc8cff22; border: 1px solid #bc8cff55; color: #bc8cff; }
    .node-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .node-table th { text-align: left; padding: 6px 10px; font-size: 10px; text-transform: uppercase; letter-spacing: .5px; color: #8b949e; border-bottom: 1px solid var(--border); }
    .node-table td { padding: 5px 10px; border-bottom: 1px solid #21262d; vertical-align: top; }
    .node-table tr:hover td { background: #1c2128; }
    .node-table tr:last-child td { border-bottom: none; }
    .feat-grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(180px,1fr)); gap: 8px; }
    .feat-item { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 6px; background: #0d1117; border: 1px solid var(--border); font-size: 12px; }
    .feat-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .cluster-grid { display: flex; flex-direction: column; gap: 10px; margin-top: 12px; }
    .cluster-item { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
    .cluster-head { display: flex; align-items: center; gap: 10px; padding: 8px 14px; border-bottom: 1px solid var(--border); }
    .cluster-files { padding: 10px 14px; display: flex; flex-wrap: wrap; gap: 6px; background: #0d1117; }
    .stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }
    .stat-pill { text-align: center; padding: 10px 20px; border-radius: 8px; background: #0d1117; border: 1px solid var(--border); }
    .stat-pill .sv { font-size: 24px; font-weight: 700; color: var(--accent); }
    .stat-pill .sl { font-size: 10px; text-transform: uppercase; letter-spacing: .5px; color: #8b949e; margin-top: 2px; }
  </style>
</head>
<body>
  <div class="layout">
    <header>
      <h1>🧪 Repo AI — Dev Dashboard</h1>
      <span id="status-badge" class="status-badge status-none">—</span>
      <span id="chunk-count" style="font-size:12px;color:#8b949e"></span>
      <span style="margin-left:auto;font-size:11px;color:#8b949e" id="version"></span>
    </header>

    <!-- Sidebar: Quick Actions -->
    <div class="sidebar">
      <div class="card">
        <h3>⚡ Quick Actions</h3>
        <button class="btn" onclick="callEndpoint('GET', '/health')">🏥 Health Check</button>
        <div style="height:6px"></div>
        <button class="btn" onclick="callEndpoint('GET', '/index/status')">📊 Index Status</button>
        <div style="height:6px"></div>
        <button class="btn" onclick="callEndpoint('GET', '/dev/debug')">🔧 Debug Info</button>
      </div>

      <div class="card">
        <h3>📁 Index Directory</h3>
        <input id="repo-path" placeholder="G:/projs/repo-aware-ai" />
        <div style="height:6px"></div>
        <button class="btn" onclick="indexDir()">Index Folder</button>
        <div style="height:6px"></div>
        <button class="btn danger" onclick="callEndpoint('DELETE', '/repository/clear')">🗑 Clear Index</button>
      </div>

      <div class="card">
        <h3>🔍 Search</h3>
        <input id="search-query" placeholder="embedder, loader, etc." />
        <div style="height:6px"></div>
        <button class="btn" onclick="searchChunks()">Search Chunks</button>
      </div>

      <div class="card">
        <h3>💬 Query</h3>
        <textarea id="question" placeholder="How does the indexing pipeline work?"></textarea>
        <div style="height:6px"></div>
        <button class="btn" onclick="askQuestion()">Ask Question</button>
      </div>

      <div class="card">
        <h3>🗺 Graph</h3>
        <button class="btn" onclick="loadGraph()">Load Dependencies</button>
        <div style="height:6px"></div>
        <button class="btn secondary" onclick="loadClusters()">Load Clusters</button>
      </div>
    </div>

    <!-- Main Panel -->
    <div class="main">
      <div class="tabs">
        <div class="tab active" onclick="switchTab('results')">📋 Results</div>
        <div class="tab" onclick="switchTab('repomap')">🗺 Repo Map</div>
        <div class="tab" onclick="switchTab('stats')">📊 Stats</div>
      </div>

      <!-- Results Tab -->
      <div id="tab-results" class="tab-content active">
        <div id="result-area">
          <div class="card" style="text-align:center;padding:40px;color:#8b949e">
            <div style="font-size:32px;margin-bottom:8px">🧪</div>
            Use the sidebar to test endpoints.<br/>
            Results will appear here.
          </div>
        </div>
      </div>

      <!-- Repo Map Tab -->
      <div id="tab-repomap" class="tab-content">
        <div class="card" style="padding:0;overflow:hidden">
          <div style="padding:10px 14px;display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--border)">
            <span style="font-size:14px;font-weight:600;color:var(--accent)">🗺 Repo Map</span>
            <input id="map-search" placeholder="Filter files..." style="width:200px;margin-left:auto" />
            <button class="btn secondary" style="width:auto" onclick="loadAndRenderMap()">↻ Refresh</button>
          </div>
          <canvas id="repomap-canvas"></canvas>
        </div>
        <div id="map-tooltip" style="position:fixed;background:var(--card);border:1px solid var(--border);border-radius:6px;padding:8px 12px;font-size:12px;pointer-events:none;opacity:0;transition:opacity 0.15s;z-index:100"></div>
        <div id="map-stats" style="margin-top:8px;font-size:12px;color:#8b949e"></div>
      </div>

      <!-- Stats Tab -->
      <div id="tab-stats" class="tab-content">
        <div class="grid-2">
          <div class="card stat"><div class="val" id="stat-chunks">—</div><div class="label">Chunks</div></div>
          <div class="card stat"><div class="val" id="stat-files">—</div><div class="label">Files</div></div>
          <div class="card stat"><div class="val" id="stat-status">—</div><div class="label">Status</div></div>
          <div class="card stat"><div class="val" id="stat-version">—</div><div class="label">Version</div></div>
        </div>
        <div class="card" style="margin-top:12px">
          <h3>Engine Config</h3>
          <pre id="engine-config" class="result" style="margin-top:0">Loading...</pre>
        </div>
      </div>
    </div>
  </div>

  <!-- Three.js for Repo Map -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script>
    const BASE = '';
    let graphData = null, clusterData = null;

    // ── Tab switching ──
    function switchTab(name) {
      document.querySelectorAll('.tab').forEach((t, i) => {
        const tabs = ['results','repomap','stats'];
        t.classList.toggle('active', tabs[i] === name);
      });
      document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
      document.getElementById('tab-' + name).classList.add('active');
      if (name === 'stats') refreshStats();
      if (name === 'repomap' && !graphData) loadAndRenderMap();
    }

    // ── Syntax-highlight JSON ──
    function highlightJson(obj) {
      const raw = JSON.stringify(obj, null, 2);
      return raw.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/("[^"]*")\s*:/g, '<span class="json-key">$1</span>:')
        .replace(/:\s*("[^"]*")/g, ': <span class="json-str">$1</span>')
        .replace(/:\s*(-?\d+\.?\d*)/g, ': <span class="json-num">$1</span>')
        .replace(/:\s*(true|false)/g, ': <span class="json-bool">$1</span>')
        .replace(/:\s*(null)/g, ': <span class="json-null">$1</span>');
    }

    // ── Render chunk cards (for /search) ──
    function renderChunks(chunks, elapsed, status) {
      const area = document.getElementById('result-area');
      if (!chunks || !chunks.length) {
        area.innerHTML = '<div class="result" style="color:var(--yellow)">No chunks returned.</div>';
        return;
      }
      const header = `<div class="timing">POST /search → ${status} (${elapsed}ms) · <strong style="color:var(--green)">${chunks.length} chunks</strong></div>`;
      const cards = chunks.map((c, i) => {
        const scoreColor = c.score > 0.4 ? 'var(--green)' : c.score > 0.2 ? 'var(--yellow)' : 'var(--accent)';
        const pct = (c.score * 100).toFixed(1);
        const escaped = c.text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        return `
          <div class="chunk-card">
            <div class="chunk-header">
              <span style="color:#8b949e;font-size:11px;font-weight:700;min-width:22px">#${i+1}</span>
              <span class="chunk-file">📄 ${c.source}</span>
              <span class="chunk-score" style="background:${scoreColor}22;color:${scoreColor}">${pct}%</span>
              <span class="chunk-range">${c.start_char}–${c.end_char}</span>
            </div>
            <pre class="chunk-code">${escaped}</pre>
          </div>`;
      }).join('');
      area.innerHTML = header + '<div class="chunk-list">' + cards + '</div>';
    }

    // ── Render query answer ──
    function renderAnswer(data, elapsed, status) {
      const area = document.getElementById('result-area');
      const escaped = (data.answer || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      const sourceChips = (data.sources || []).map(s =>
        `<span class="source-chip">📄 ${s}</span>`
      ).join('');
      area.innerHTML = `
        <div class="timing">POST /query → ${status} (${elapsed}ms)</div>
        <div class="answer-card">
          <div class="answer-header">✅ Answer</div>
          <div class="answer-body">${escaped}</div>
          ${data.sources?.length ? `<div style="padding:6px 14px;font-size:11px;color:#8b949e;font-weight:700;letter-spacing:.5px;text-transform:uppercase">Sources (${data.sources.length})</div><div class="sources-list">${sourceChips}</div>` : ''}
        </div>`;
    }

    // cluster palette
    const C_COLORS = ['#58a6ff','#3fb950','#d29922','#f78166','#bc8cff','#76e3ea','#ffa657','#ff7b72'];

    // ── Render: /health ──
    function renderHealth(d, elapsed) {
      const ok = d.status === 'ok';
      const area = document.getElementById('result-area');
      area.innerHTML = `
        <div class="timing">GET /health → ${ok ? 200 : 500} (${elapsed}ms)</div>
        <div class="info-card">
          <div class="info-card-header">
            <span style="font-size:18px">${ok ? '🟢' : '🔴'}</span>
            <span class="info-card-title">Backend Health</span>
            <span class="pill ${ok ? 'pill-green' : 'pill-red'}">${d.status}</span>
          </div>
          <div class="info-card-body">
            <div class="stat-row">
              <div class="stat-pill"><div class="sv">v${d.version||'—'}</div><div class="sl">Version</div></div>
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap">
              <span class="pill pill-blue">🤖 ${d.model||'—'}</span>
              <span class="pill pill-purple">🧬 ${d.embedding_model||'—'}</span>
            </div>
          </div>
        </div>`;
    }

    // ── Render: /index/status ──
    function renderIndexStatus(d, elapsed) {
      const area = document.getElementById('result-area');
      const color = {none:'pill-gray',building:'pill-yellow',ready:'pill-green',error:'pill-red'}[d.status] || 'pill-gray';
      const info = d.info || {};
      const rows = Object.entries(info).map(([k,v]) =>
        `<tr><td style="color:#8b949e;padding:4px 10px">${k}</td><td style="padding:4px 10px;font-family:'Cascadia Code',monospace;color:var(--fg)">${JSON.stringify(v)}</td></tr>`
      ).join('');
      area.innerHTML = `
        <div class="timing">GET /index/status → 200 (${elapsed}ms)</div>
        <div class="info-card">
          <div class="info-card-header">
            <span style="font-size:16px">📊</span>
            <span class="info-card-title">Index Status</span>
            <span class="pill ${color}">${d.status}</span>
          </div>
          <div class="info-card-body">${rows ? `<table class="node-table">${rows}</table>` : '<span style="color:#8b949e;font-size:12px">No info yet</span>'}</div>
        </div>`;
    }

    // ── Render: /dev/debug ──
    function renderDebug(d, elapsed) {
      const area = document.getElementById('result-area');
      const flags = [
        ['use_hybrid_search','🔀 Hybrid Search'],['use_reranker','🏆 Reranker'],
        ['use_query_expansion','🔭 Query Expansion'],['use_compression','🗜 Compression'],
        ['use_multi_query','📡 Multi-Query'],['has_bm25','📖 BM25 Index'],
        ['engine_loaded','⚙️ Engine Loaded'],
      ];
      const feats = flags.map(([k,label]) => {
        const val = d[k];
        const on = val === true;
        const off = val === false;
        const dot = on ? '#3fb950' : off ? '#f85149' : '#8b949e';
        const txt = val === null || val === undefined ? 'N/A' : String(val);
        return `<div class="feat-item"><div class="feat-dot" style="background:${dot}"></div><span>${label}</span><span style="margin-left:auto;color:${dot};font-size:11px;font-weight:700">${txt}</span></div>`;
      }).join('');
      area.innerHTML = `
        <div class="timing">GET /dev/debug → 200 (${elapsed}ms)</div>
        <div class="info-card">
          <div class="info-card-header">
            <span style="font-size:16px">🔧</span>
            <span class="info-card-title">Engine Debug</span>
            <span class="pill pill-blue">${d.chunk_count || 0} chunks</span>
            <span class="pill pill-gray">v${d.version||'—'}</span>
          </div>
          <div class="info-card-body"><div class="feat-grid">${feats}</div></div>
        </div>`;
    }

    // ── Render: /graph/dependencies ──
    function renderGraph(d, elapsed) {
      const area = document.getElementById('result-area');
      const nodes = d.nodes || [], edges = d.edges || [];
      // top nodes by chunk count
      const sorted = [...nodes].sort((a,b) => (b.chunkCount||0)-(a.chunkCount||0));
      const rows = sorted.slice(0, 60).map(n => {
        const bar = Math.min(100, ((n.chunkCount||0) / (sorted[0]?.chunkCount||1)) * 100);
        return `<tr>
          <td style="font-family:'Cascadia Code',monospace;color:var(--accent);max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${n.id}">${n.label||n.id}</td>
          <td style="color:#8b949e;font-family:'Cascadia Code',monospace;font-size:11px;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${n.id}</td>
          <td><div style="display:flex;align-items:center;gap:8px">
            <div style="width:80px;height:6px;background:#21262d;border-radius:3px"><div style="width:${bar}%;height:100%;background:var(--accent);border-radius:3px"></div></div>
            <span style="color:var(--fg);font-size:11px">${n.chunkCount||0}</span>
          </div></td>
        </tr>`;
      }).join('');
      // edge summary grouped by source
      const edgeMap = {};
      for (const e of edges) { (edgeMap[e.source]=edgeMap[e.source]||[]).push(e.target); }
      const topEdges = Object.entries(edgeMap).sort((a,b)=>b[1].length-a[1].length).slice(0,8);
      const edgeRows = topEdges.map(([src,targets]) =>
        `<tr><td style="font-family:'Cascadia Code',monospace;color:var(--accent)">${src}</td><td>${targets.slice(0,5).map(t=>`<span class="pill pill-gray">${t}</span>`).join(' ')}</td></tr>`
      ).join('');
      area.innerHTML = `
        <div class="timing">GET /graph/dependencies → 200 (${elapsed}ms)</div>
        <div class="stat-row" style="margin-top:12px">
          <div class="stat-pill"><div class="sv">${nodes.length}</div><div class="sl">Files</div></div>
          <div class="stat-pill"><div class="sv">${edges.length}</div><div class="sl">Dependencies</div></div>
          <div class="stat-pill"><div class="sv">${Math.round(edges.length/(nodes.length||1)*10)/10}</div><div class="sl">Avg Deps</div></div>
        </div>
        <div class="info-card">
          <div class="info-card-header"><span>📦</span><span class="info-card-title">Files by Chunk Count</span><span style="color:#8b949e;font-size:11px">top ${Math.min(sorted.length,60)}</span></div>
          <div style="overflow-y:auto;max-height:280px"><table class="node-table"><thead><tr><th>Label</th><th>Path</th><th>Chunks</th></tr></thead><tbody>${rows}</tbody></table></div>
        </div>
        ${edgeRows ? `<div class="info-card" style="margin-top:10px">
          <div class="info-card-header"><span>🔗</span><span class="info-card-title">Top Import Relations</span></div>
          <div style="overflow-y:auto;max-height:220px"><table class="node-table"><thead><tr><th>File</th><th>Imports</th></tr></thead><tbody>${edgeRows}</tbody></table></div>
        </div>` : ''}`;
    }

    // ── Render: /graph/clusters ──
    function renderClusters(d, elapsed) {
      const area = document.getElementById('result-area');
      const clusters = d.clusters || [];
      if (!clusters.length) {
        area.innerHTML = `<div class="timing">GET /graph/clusters → 200 (${elapsed}ms)</div><div class="result" style="color:var(--yellow)">No clusters returned.</div>`;
        return;
      }
      const totalFiles = clusters.reduce((s,c)=>s+(c.files||[]).length,0);
      const clusterCards = clusters.map((c, i) => {
        const col = C_COLORS[i % C_COLORS.length];
        const files = c.files || [];
        const chips = files.slice(0, 20).map(f =>
          `<span class="pill" style="background:${col}18;border:1px solid ${col}44;color:${col}">${f}</span>`
        ).join('');
        const more = files.length > 20 ? `<span class="pill pill-gray">+${files.length-20} more</span>` : '';
        return `<div class="cluster-item">
          <div class="cluster-head">
            <div style="width:12px;height:12px;border-radius:50%;background:${col};flex-shrink:0"></div>
            <span style="font-weight:700;color:var(--fg)">Cluster ${c.id ?? i}</span>
            <span class="pill" style="background:${col}22;border:1px solid ${col}55;color:${col}">${files.length} files</span>
            <span style="margin-left:auto;font-size:10px;color:#8b949e">${Math.round(files.length/totalFiles*100)}% of repo</span>
          </div>
          <div class="cluster-files">${chips}${more}</div>
        </div>`;
      }).join('');
      area.innerHTML = `
        <div class="timing">GET /graph/clusters → 200 (${elapsed}ms)</div>
        <div class="stat-row" style="margin-top:12px">
          <div class="stat-pill"><div class="sv">${clusters.length}</div><div class="sl">Clusters</div></div>
          <div class="stat-pill"><div class="sv">${totalFiles}</div><div class="sl">Files</div></div>
          <div class="stat-pill"><div class="sv">${Math.round(totalFiles/clusters.length)}</div><div class="sl">Avg Size</div></div>
        </div>
        <div class="cluster-grid">${clusterCards}</div>`;
    }

    // ── Render: /repository/clear ──
    function renderClear(d, elapsed) {
      const area = document.getElementById('result-area');
      area.innerHTML = `
        <div class="timing">DELETE /repository/clear → 200 (${elapsed}ms)</div>
        <div class="info-card">
          <div class="info-card-header"><span>🗑</span><span class="info-card-title">Repository Cleared</span><span class="pill pill-red">cleared</span></div>
          <div class="info-card-body" style="color:#8b949e;font-size:13px">${d.message || 'Index and engine state reset.'}</div>
        </div>`;
    }

    // ── Render: /index/directory ──
    function renderIndexStart(d, elapsed) {
      const area = document.getElementById('result-area');
      area.innerHTML = `
        <div class="timing">POST /index/directory → 200 (${elapsed}ms)</div>
        <div class="info-card">
          <div class="info-card-header"><span>🔄</span><span class="info-card-title">Indexing Started</span><span class="pill pill-yellow">building</span></div>
          <div class="info-card-body" style="font-size:13px;color:#8b949e">
            ${d.message||''}<br/>
            ${d.upload_id ? `<span style="margin-top:8px;display:inline-block" class="pill pill-blue">Upload ID: ${d.upload_id}</span>` : ''}
          </div>
        </div>`;
    }

    // ── Generic endpoint caller ──
    async function callEndpoint(method, path, body = null) {
      const area = document.getElementById('result-area');
      area.innerHTML = '<div class="result" style="color:var(--yellow)">⏳ Loading...</div>';
      switchTab('results');
      const t0 = performance.now();
      try {
        const opts = { method, headers: {} };
        if (body) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
        const res = await fetch(BASE + path, opts);
        const data = await res.json();
        const elapsed = (performance.now() - t0).toFixed(0);
        // Smart rendering based on endpoint
        if (path === '/search' && data.chunks) {
          renderChunks(data.chunks, elapsed, res.status);
        } else if (path === '/query' && data.answer !== undefined) {
          renderAnswer(data, elapsed, res.status);
        } else if (path === '/health') {
          renderHealth(data, elapsed);
        } else if (path === '/index/status') {
          renderIndexStatus(data, elapsed);
        } else if (path === '/dev/debug') {
          renderDebug(data, elapsed);
        } else if (path === '/graph/dependencies') {
          renderGraph(data, elapsed);
        } else if (path === '/graph/clusters') {
          renderClusters(data, elapsed);
        } else if (path === '/repository/clear') {
          renderClear(data, elapsed);
        } else if (path === '/index/directory') {
          renderIndexStart(data, elapsed);
        } else {
          const cls = res.ok ? 'success' : 'error';
          area.innerHTML = `<div class="result ${cls}" style="font-family:'Cascadia Code',monospace">${highlightJson(data)}</div><div class="timing">${method} ${path} → ${res.status} (${elapsed}ms)</div>`;
        }
        return data;
      } catch (e) {
        area.innerHTML = `<div class="result error">❌ ${e.message}</div>`;
        return null;
      }
    }

    async function indexDir() {
      const path = document.getElementById('repo-path').value.trim();
      if (!path) { alert('Enter a folder path'); return; }
      await callEndpoint('POST', '/index/directory', { repo_path: path });
      pollStatus();
    }

    async function searchChunks() {
      const q = document.getElementById('search-query').value.trim();
      if (!q) return;
      await callEndpoint('POST', '/search', { question: q });
    }

    async function askQuestion() {
      const q = document.getElementById('question').value.trim();
      if (!q) return;
      await callEndpoint('POST', '/query', { question: q });
    }

    async function loadGraph() {
      const data = await callEndpoint('GET', '/graph/dependencies');
      if (data) graphData = data;
    }

    async function loadClusters() {
      const data = await callEndpoint('GET', '/graph/clusters');
      if (data) clusterData = data;
    }

    // ── Status polling ──
    async function pollStatus() {
      try {
        const res = await fetch(BASE + '/index/status');
        const data = await res.json();
        const badge = document.getElementById('status-badge');
        badge.textContent = data.status;
        badge.className = 'status-badge status-' + data.status;
        const chunks = data.info?.chunk_count;
        document.getElementById('chunk-count').textContent = chunks ? `${chunks} chunks` : '';
        if (data.status === 'building') setTimeout(pollStatus, 1000);
      } catch(e) {}
    }
    setInterval(pollStatus, 5000);
    pollStatus();

    // Health at start
    fetch(BASE + '/health').then(r => r.json()).then(d => {
      document.getElementById('version').textContent = `v${d.version} • ${d.model}`;
    }).catch(() => {});

    // ── Stats Tab ──
    async function refreshStats() {
      try {
        const data = await (await fetch(BASE + '/dev/debug')).json();
        document.getElementById('stat-chunks').textContent = data.chunk_count || '—';
        document.getElementById('stat-status').textContent = data.index_status;
        document.getElementById('stat-version').textContent = data.version;
        document.getElementById('stat-files').textContent = data.index_info?.chunk_count ? '~' + Math.round(data.chunk_count / 15) : '—';
        document.getElementById('engine-config').textContent = JSON.stringify(data, null, 2);
      } catch(e) {}
    }

    // ── Repo Map (Three.js) ──
    async function loadAndRenderMap() {
      try {
        const [graph, clusters] = await Promise.all([
          fetch(BASE + '/graph/dependencies').then(r => r.json()),
          fetch(BASE + '/graph/clusters').then(r => r.json()).catch(() => ({clusters:[]})),
        ]);
        graphData = graph;
        clusterData = clusters;
        renderMap(graph, clusters);
        document.getElementById('map-stats').textContent =
          `${graph.nodes?.length || 0} files • ${graph.edges?.length || 0} dependencies • ${clusters.clusters?.length || 0} clusters`;
      } catch(e) {
        document.getElementById('map-stats').textContent = '⚠ ' + e.message;
      }
    }

    let mapRenderer, mapScene, mapCamera, mapPivot, mapNodeObjects = [];

    function renderMap(graph, clusters) {
      const canvas = document.getElementById('repomap-canvas');
      const tooltip = document.getElementById('map-tooltip');
      const W = canvas.clientWidth, H = canvas.clientHeight;

      // Cleanup previous
      if (mapRenderer) mapRenderer.dispose();

      mapRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
      mapRenderer.setSize(W, H);
      mapRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

      mapScene = new THREE.Scene();
      mapScene.fog = new THREE.FogExp2(0x0d1117, 0.002);

      mapCamera = new THREE.PerspectiveCamera(60, W / H, 0.1, 2000);
      mapCamera.position.set(0, 0, 300);

      mapScene.add(new THREE.AmbientLight(0x404080, 0.5));
      const pl = new THREE.PointLight(0x58a6ff, 2, 500);
      pl.position.set(0, 100, 100);
      mapScene.add(pl);

      mapPivot = new THREE.Group();
      mapScene.add(mapPivot);

      // Simple orbit
      let drag = false, pX = 0, pY = 0, rX = 0, rY = 0, zoom = 300;
      canvas.onmousedown = e => { drag = true; pX = e.clientX; pY = e.clientY; };
      canvas.onmouseup = () => drag = false;
      canvas.onmousemove = e => {
        if (!drag) { checkMapHover(e); return; }
        rY += (e.clientX - pX) * 0.5; rX += (e.clientY - pY) * 0.5;
        pX = e.clientX; pY = e.clientY;
        mapPivot.rotation.y = THREE.MathUtils.degToRad(rY);
        mapPivot.rotation.x = THREE.MathUtils.degToRad(rX);
      };
      canvas.onwheel = e => {
        zoom = Math.max(50, Math.min(800, zoom + e.deltaY * 0.3));
        mapCamera.position.z = zoom;
      };

      // Build nodes
      const colors = [0x58a6ff, 0x3fb950, 0xd29922, 0xf78166, 0xbc8cff, 0x76e3ea, 0xffa657, 0xff7b72];
      const fileCluster = {};
      if (clusters?.clusters) {
        for (const c of clusters.clusters) for (const f of c.files) fileCluster[f] = c.id;
      }

      const nodes = graph.nodes || [], edges = graph.edges || [];
      const phi = Math.PI * (3 - Math.sqrt(5)), radius = Math.max(80, nodes.length * 3);
      mapNodeObjects = [];

      nodes.forEach((node, i) => {
        const y = 1 - (i / (nodes.length - 1 || 1)) * 2;
        const r = Math.sqrt(1 - y * y);
        const theta = phi * i;
        const pos = new THREE.Vector3(Math.cos(theta) * r * radius, y * radius, Math.sin(theta) * r * radius);
        const size = Math.max(2, Math.min(12, node.chunkCount * 0.5));
        const cid = fileCluster[node.id] ?? 0;
        const color = colors[cid % colors.length];
        const geo = new THREE.SphereGeometry(size, 16, 16);
        const mat = new THREE.MeshPhongMaterial({ color, emissive: color, emissiveIntensity: 0.2, transparent: true, opacity: 0.85 });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.copy(pos);
        mesh.userData = { node, color };
        mapPivot.add(mesh);
        mapNodeObjects.push(mesh);
      });

      // Edges
      const nodePos = {};
      mapNodeObjects.forEach(m => nodePos[m.userData.node.id] = m.position);
      const lm = new THREE.LineBasicMaterial({ color: 0x30363d, transparent: true, opacity: 0.3 });
      for (const edge of edges) {
        const from = nodePos[edge.source], to = nodePos[edge.target];
        if (!from || !to) continue;
        mapPivot.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([from, to]), lm));
      }

      // Search filter
      document.getElementById('map-search').oninput = function() {
        const q = this.value.toLowerCase();
        mapNodeObjects.forEach(m => {
          const match = !q || (m.userData.node.label || '').toLowerCase().includes(q);
          m.material.opacity = match ? 0.85 : 0.1;
          m.material.emissiveIntensity = match ? (q ? 0.6 : 0.2) : 0;
        });
      };

      // Animate
      const raycaster = new THREE.Raycaster(), mouse = new THREE.Vector2();
      function animate() {
        requestAnimationFrame(animate);
        if (!drag) mapPivot.rotation.y += 0.001;
        mapRenderer.render(mapScene, mapCamera);
      }
      animate();

      function checkMapHover(e) {
        const rect = canvas.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, mapCamera);
        const hits = raycaster.intersectObjects(mapNodeObjects);
        if (hits.length) {
          const n = hits[0].object.userData.node;
          tooltip.innerHTML = `<strong>${n.label}</strong><br/>${n.chunkCount} chunks<br/><span style="color:#8b949e">${n.id}</span>`;
          tooltip.style.opacity = '1';
          tooltip.style.left = (e.clientX + 12) + 'px';
          tooltip.style.top = (e.clientY - 30) + 'px';
          canvas.style.cursor = 'pointer';
        } else {
          tooltip.style.opacity = '0';
          canvas.style.cursor = 'default';
        }
      }
    }
  </script>
</body>
</html>
"""
