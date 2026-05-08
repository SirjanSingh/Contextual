# Architecture

This document describes how Repo-Aware AI is put together. For a higher-level
overview, start with the [README](../README.md).

## High-level shape

```
                 ┌───────────────────┐
                 │    User surface   │
                 │  CLI · Web · IDE  │
                 └────────┬──────────┘
                          │ HTTP / direct call
                          ▼
                ┌────────────────────┐
                │   FastAPI server   │  repo_aware_ai/server.py
                └────────┬───────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │      QAEngine        │  repo_aware_ai/qa.py
              │  (pipeline orchestrator) │
              └─┬─────────────┬──────┘
   ┌────────────┘             └────────────┐
   ▼                                       ▼
[Indexing path]                    [Query path]
loader → chunker → embedder       multi_query → expand → retrieve →
  → indexer (FAISS)                 rerank → compress → llm
  → repo_map (tree-sitter)
```

## Surfaces

### CLI (`main.py`)
Thin wrapper that builds a `QAEngine` and runs an interactive REPL. Useful for
local debugging without touching the server. Installed as the
`repo-aware-ai` console script.

### Web server (`repo_aware_ai/server.py`)
A FastAPI app exposing the full pipeline. Key endpoint groups:

| Group | Endpoints | Purpose |
|-------|-----------|---------|
| Health | `/health`, `/index/status` | Readiness + liveness |
| Indexing | `/index/directory`, `/index/rebuild`, `/upload/directory`, `/upload/progress/{id}`, `/ws/indexing/{id}` | Trigger and monitor index builds |
| Query | `/query`, `/query/stream`, `/search`, `/context/file` | Answers, search, file context |
| Repo map | `/graph/repo-map`, `/graph/community/{id}`, `/graph/process/{id}`, `/graph/symbol/{id}`, `/graph/neighborhood/{id}`, `/graph/dependencies`, `/graph/clusters`, `/graph/symbols` | Code graph + community detection |
| Misc | `/repository/clear`, `/dev/`, `/dev/debug` | Admin and dev dashboard |

The server keeps a single `QAEngine` in module-level state behind a lock.
Indexing runs on a background thread so HTTP stays responsive.

Static frontend assets are mounted at `/` from `frontend/dist` if present.

Installed as `repo-aware-ai-server`.

### Web frontend (`frontend/`)
React 18 + Vite + Tailwind + Zustand + Three.js. Renders:

- **Chat view** — query box, streamed answer, source citations.
- **Repo map view** — canvas-based force-directed graph of communities,
  process timeline, drill-down detail panels.
- **Three.js scene** — ambient particle / neural-network visual overlay.

Talks to the backend via Vite's dev proxy or direct `/` mount in production.

### VS Code extension (`extension/`)
TypeScript, bundled with esbuild. Activation sequence:

1. **Bootstrap** (`backendBootstrap.ts`) — finds system Python ≥ 3.10, creates
   a venv under `globalStorageUri/venv`, and `pip install`s the backend. Fast-path
   skips install if `repo_aware_ai.server` is already importable in the venv.
2. **Spawn** (`backendProcess.ts`) — starts `python -m uvicorn repo_aware_ai.server:app`
   as a sidecar, polls `/health`, then emits `onReady`.
3. **Auto-index** — indexes the open workspace automatically once the backend is ready.

Exposes:

- Commands: ask question, explain selection, find related code, explain repo,
  rebuild index, show backend logs, set API key.
- Webview views: chat panel (live streaming), repo map panel.
- Providers: CodeLens (related-chunk hints) and Hover ("used in …").
- Status bar: chunk count + indexing state (down / indexing / ready / error).

The chat panel consumes `/query/stream` SSE and appends tokens in real-time.
The `openSource` handler resolves workspace-relative paths and opens the file at
the chunk's start character offset.

## Pipeline modules

All inside `repo_aware_ai/`.

| Module | Role |
|--------|------|
| `loader.py` | Walks the repo, filters by extension, drops ignored directories. |
| `chunker.py` / `ast_chunker.py` | Character-window chunking (default) or AST-aware chunks for Python. |
| `embedder.py` | Google `gemini-embedding-001` calls, batched and L2-normalized. |
| `indexer.py` | FAISS `IndexFlatIP` plus a fingerprint manifest for cache validation. |
| `retriever.py` | Cosine similarity search (and a graph-aware variant that pulls in symbol neighbours). |
| `hybrid_search.py` | BM25 + vector via reciprocal rank fusion. |
| `reranker.py` | Cross-encoder reranking with `ms-marco-MiniLM-L-6-v2`. |
| `compressor.py` | LLM extracts only the question-relevant lines per chunk. |
| `query_expander.py` | Generates 2–3 paraphrases for query diversity. |
| `multi_query.py` | Heuristic decomposition of compound questions. |
| `conversation.py` | Bounded ring-buffer of recent turns. |
| `qa.py` | Wires it all together. `_prepare()` runs everything up to the LLM call; `ask()` / `stream_ask()` finish it. |
| `llm.py` | Gemini wrapper. `answer()` for one-shot; `stream_answer()` yields incremental text chunks. |
| `_retry.py` | `@gemini_retry` decorator (tenacity): retries 429 / RESOURCE\_EXHAUSTED / 503 / 500 up to `RAI_GEMINI_MAX_RETRIES` times. |
| `config.py` | `.env` loading + defaults. |
| `debug.py` | Best-effort JSON debug logs. |
| `progress_tracker.py` | Upload / indexing progress with stages and ETA. |
| `upload_handler.py` | Saves uploaded files and runs the indexing pipeline. |
| `repo_map/` | Symbol graph + community detection (see below). |

## The repo-map subsystem (`repo_aware_ai/repo_map/`)

Optional feature that produces a structural understanding of the codebase.
Gracefully degrades if `tree-sitter` or `python-igraph` are unavailable.

| Module | Role |
|--------|------|
| `parsing.py` | Tree-sitter walks for Python / JS / TS extracting functions, classes, methods, imports. |
| `relationships.py` | Builds CALLS / IMPORTS / EXTENDS / CONTAINS edges. |
| `graph.py` | In-memory `KnowledgeGraph` with adjacency indexes. |
| `communities.py` | Leiden community detection with heuristic labelling. |
| `processes.py` | Detects multi-step call chains as named processes. |
| `cache.py` | Persists `repo_map.json` alongside the FAISS index. |
| `types.py` | Dataclasses for nodes / relationships / communities / processes. |

## Caching

Everything keys off a fingerprint:

```
SHA256(repo_path + sorted(file_paths) + mtime_ns + size)
```

Cache layout under `data/index/<repo_id>/`:

```
faiss.index        # FAISS vectors
metadata.pkl       # chunk metadata aligned 1:1 with FAISS rows
manifest.json      # fingerprint + build params
bm25.pkl           # optional BM25 index
repo_map.json      # optional repo map
```

Any change to file paths, mtimes, or sizes invalidates the cache and forces a
full rebuild on next start.

## Critical invariants

- **FAISS ↔ metadata alignment.** Row `i` in the FAISS index must always
  correspond to entry `i` in `metadata.pkl`. Any code that mutates one and not
  the other corrupts every subsequent retrieval.
- **POSIX paths internally.** All chunk source paths are stored with `/`
  separators relative to the repo root. Windows paths must be normalised
  before storage (see `chunker.py`).
- **Context-only LLM prompt.** The system prompt in `llm.py` forbids the
  model from answering from outside the retrieved context. The fallback
  string is `"Not found in the retrieved repository context."`.
- **Background indexing must always settle.** Errors during indexing must
  flip the global status to `"error"` so the client stops polling forever.
