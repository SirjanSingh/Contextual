# Repo-Aware AI — Copilot Instructions

## Project overview

A "talk to your codebase" RAG tool powered by **Google Gemini** (not Ollama).
Three surfaces share one Python backend: CLI, web app, and VS Code extension.

Package: `repo_aware_ai/` (installed as `repo-aware-ai` / `repo-aware-ai-server` console scripts).
See `pyproject.toml` for extras: `reranker`, `graph`, `all`, `dev`.

## Architecture

```
repo_aware_ai/
├── loader.py           # file walker + ignore-list (fnmatch venv patterns)
├── chunker.py          # character-window chunking (default 1800 / 250)
├── ast_chunker.py      # AST-aware chunking for Python
├── embedder.py         # gemini-embedding-001, batched, L2-normalised
├── indexer.py          # FAISS IndexFlatIP + metadata.pkl (MUST stay 1:1 aligned)
├── retriever.py        # cosine similarity top-k
├── hybrid_search.py    # BM25 + vector via reciprocal rank fusion
├── reranker.py         # cross-encoder ms-marco-MiniLM-L-6-v2
├── compressor.py       # LLM extracts relevant lines per chunk
├── query_expander.py   # 2-3 paraphrase variants
├── multi_query.py      # decomposes compound questions
├── conversation.py     # bounded history (Lock-protected)
├── _retry.py           # @gemini_retry (tenacity) for 429/503/500
├── qa.py               # pipeline orchestrator (_prepare + ask/stream_ask)
├── llm.py              # Gemini wrapper (answer + stream_answer)
├── server.py           # FastAPI; indexing on background thread
└── repo_map/           # tree-sitter → symbol graph → Leiden communities
```

## Critical invariants

- **FAISS ↔ metadata alignment**: Row `i` in FAISS index must always match entry `i` in `metadata.pkl`. Never mutate one without the other.
- **POSIX paths**: All stored paths use `/` relative to repo root. Convert Windows paths with `.replace("\\", "/")` before storage.
- **Context-only LLM**: `llm.py` system prompt forbids answering outside retrieved context. Fallback: `"Not found in the retrieved repository context."`.
- **Loader ignore list**: `loader.py` uses both an exact-name set (`DEFAULT_IGNORE_DIRS`) and `fnmatch` glob patterns (`DEFAULT_IGNORE_DIR_PATTERNS`) to catch dynamic venv names like `.venv311`. Adding new ignore rules goes in both if appropriate.
- **Background indexing must settle**: Errors during indexing must flip the global status to `"error"` so polling clients stop.

## Running the project

```bash
# Backend dev server
python dev_server.py --port 18360 --repo ./some-project
# Or with uvicorn directly
uvicorn repo_aware_ai.server:app --host 127.0.0.1 --port 18360

# CLI REPL
repo-aware-ai --repo .

# Frontend dev
cd frontend && npm run dev   # → http://localhost:5173 (proxied to backend)

# Extension
cd extension && npm run watch
```

> On Windows, ports around 8360 are often reserved by Hyper-V. Use `--port 18360`.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GOOGLE_API_KEY` | required | Gemini API key |
| `GEMINI_MODEL` | `models/gemini-2.5-flash` | LLM model |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | 768-dim embedding model |
| `RAI_PORT` | `8360` | Server port |
| `RAI_DATA_DIR` | `data/index` | FAISS cache root |
| `RAI_GEMINI_MAX_RETRIES` | `4` | Max Gemini retry attempts |

## Streaming

`qa.py` exposes two call paths:
- `ask(question)` → `str` (blocking, full answer)
- `stream_ask(question)` → generator of `str` chunks

`server.py` `/query/stream` endpoint formats chunks as SSE events:
```
event: chunk
data: {"text": "..."}

event: sources
data: [{"path": "...", ...}]

event: done
data: {}
```

Both the web frontend (`frontend/src/api/client.ts`) and VS Code extension
(`extension/src/services/backendClient.ts`) parse this SSE stream via
`queryStream()` which yields typed `StreamEvent` objects.

## VS Code extension activation

1. `BackendBootstrap.ensure()` — finds Python ≥ 3.10, creates venv, installs package
2. `BackendProcess.start()` — spawns uvicorn sidecar, polls `/health`
3. Auto-indexes open workspace
4. Registers commands, webviews, CodeLens, status bar

Key services: `backendBootstrap.ts`, `backendProcess.ts`, `backendClient.ts`.
Commands always registered (even on failure): `repoAwareAI.showLogs`, `repoAwareAI.setApiKey`.

## Cache layout

`data/index/<repo_id>/`:
```
faiss.index     # FAISS vectors
metadata.pkl    # chunk metadata (1:1 with FAISS rows)
manifest.json   # fingerprint + build params
bm25.pkl        # BM25 index (optional)
repo_map.json   # symbol graph + communities (optional)
```

Fingerprint = SHA256(repo_path + sorted file paths + mtime_ns + size). Any change forces a full rebuild.

## Test suite

```bash
pip install -e .[dev]
pytest tests/ -v
```

Tests in `tests/`: `test_loader.py`, `test_chunker.py`, `test_conversation.py`, `test_index_retrieve.py`.
`conftest.py` provides `tiny_repo` fixture and `FakeEmbedder` (deterministic, no API calls).
