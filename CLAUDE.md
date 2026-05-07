# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A "talk to your codebase" RAG (Retrieval-Augmented Generation) system. Users point it at a repository and ask natural language questions; it retrieves relevant code chunks and answers using Google Gemini. Three interfaces exist: CLI, web app, and VS Code extension.

## Commands

### Python Backend
```bash
# Install dependencies
pip install -r requirements.txt

# CLI (requires GOOGLE_API_KEY in .env)
python main.py --repo <path>
python main.py --repo . --rebuild           # force index rebuild
python main.py --repo . --topk 10 --chunk_size 2000 --temperature 0.1 --no-rerank --no-hybrid

# Dev server (web + API)
python dev_server.py --port 8360 --repo ./some-project

# Run backend directly
uvicorn app.server:app --host 127.0.0.1 --port 8360
```

### Frontend (React)
```bash
cd frontend
npm install
npm run dev      # Vite dev server → http://localhost:5173
npm run build
npm run preview
```

### VS Code Extension
```bash
cd extension
npm install
npm run build
npm run watch    # watch mode during development
npm run lint
npm run package  # produce .vsix
```

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `GOOGLE_API_KEY` | Yes | — | Google Gemini API key |
| `GEMINI_MODEL` | No | `models/gemini-2.5-flash` | LLM model |
| `EMBEDDING_MODEL` | No | `gemini-embedding-001` | Embedding model (768-dim) |
| `RAI_PORT` | No | `8360` | Backend port (set by extension) |
| `RAI_DATA_DIR` | No | `data/index` | FAISS cache directory |

## Architecture

### 7-Stage RAG Pipeline

```
Repository Files → Loader → Chunker → Embedder → Indexer → Retriever → Enhancement Layers → LLM → Answer + Sources
```

| Stage | File | Notes |
|-------|------|-------|
| Loader | `app/loader.py` | Filters by extension; ignores `.git`, `node_modules`, `.venv`, `data/`, `debug_logs/` |
| Chunker | `app/chunker.py`, `app/ast_chunker.py` | Default: 1800 chars / 250 overlap; AST mode for Python preserves function/class boundaries |
| Embedder | `app/embedder.py` | Gemini embeddings, batched (max 100/batch) |
| Indexer | `app/indexer.py` | FAISS CPU index; fingerprint-based cache (SHA256 of paths+mtime+size) → `data/index/<repo_id>/` |
| Retriever | `app/retriever.py` | Cosine similarity over FAISS |
| Hybrid Search | `app/hybrid_search.py` | BM25 + vector combined (optional) |
| Reranker | `app/reranker.py` | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Compressor | `app/compressor.py` | LLM extracts only question-relevant lines |
| Query Expansion | `app/query_expander.py` | Generate synonym/related queries |
| Multi-Query | `app/multi_query.py` | Decompose complex question into 3-5 sub-queries |
| Conversation | `app/conversation.py` | Last 5 turns of history |
| QA Orchestrator | `app/qa.py` | Wires all stages together |

### Web Server (`app/server.py`)

FastAPI on port 8360. Key endpoints:
- `POST /index/directory` — trigger indexing
- `GET /index/status` — progress polling
- `POST /query` — ask a question → `{answer, sources}`
- `GET /graph/dependencies` — code dependency graph
- `GET /` — serves built React frontend

### VS Code Extension (`extension/`)

On activation: checks API key → spawns Python sidecar process → polls `/health` → auto-indexes workspace → registers commands/providers.

Key services: `backendProcess.ts` (sidecar lifecycle), `backendClient.ts` (HTTP calls), `indexManager.ts`, `cacheManager.ts` (SecretStorage for API key).

Commands: ask question, explain selection, find related code, explain repo, rebuild index.

Providers: CodeLens (action hints on functions), hover tooltips.

### Frontend (`frontend/`)

React 18 + TypeScript + Vite + Tailwind CSS + Zustand. Entry: `src/main.tsx` → `src/App.tsx`. API calls via `src/api/client.ts`. Includes a Three.js 3D repo visualization (`src/components/three/`).

## Critical Design Constraints

**FAISS index / metadata alignment**: The FAISS vector index and the metadata list must stay 1:1 aligned. Any mismatch corrupts source attribution. Do not modify `app/indexer.py` in ways that could desync these.

**Cache invalidation**: Fingerprint = SHA256(repo_path + sorted file paths + mtime_ns + size). Any file change forces a full index rebuild.

**Path handling**: Internal paths use POSIX separators (`/`) relative to repo root. Windows paths must be converted with `.replace("\\", "/")` before storage.

**LLM prompt discipline**: The system prompt in `app/llm.py` enforces strict context-only answers. The LLM must not hallucinate beyond retrieved chunks. Fallback: `"Not found in the retrieved repository context."` Required output format: answer bullets → evidence quotes → sources (`path/file.py:start_char-end_char`).

**Debug logs**: `app/debug.py` writes JSON logs of retrieved chunks to `debug_logs/`. The loader explicitly ignores this directory to avoid indexing debug output.
