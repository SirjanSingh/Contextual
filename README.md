# Repo-Aware AI

> Talk to your codebase. A local RAG (retrieval-augmented generation) assistant that
> indexes any repository, retrieves the relevant code chunks, and answers natural-language
> questions with file-anchored citations — powered by Google Gemini.

Three ways to use it:

| Surface | What it gives you |
|---------|-------------------|
| **CLI** | Interactive REPL: `repo-aware-ai --repo .` |
| **Web app** | Chat UI + force-directed repo map at `http://localhost:8360` |
| **VS Code extension** | Inline `Ask` / `Explain selection` / `Find related` + chat sidebar + repo map panel |

All three share the same Python backend and FAISS vector index.

---

## Quick start

### 1. Get a Google API key

Create one at [Google AI Studio](https://aistudio.google.com/app/apikey) — the free
tier is plenty for personal use.

### 2. Install the backend

```bash
git clone https://github.com/SirjanSingh/repo-aware-ai
cd repo-aware-ai

python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -e .[all]
```

> The `[all]` extra pulls in the cross-encoder reranker and the tree-sitter / igraph
> graph-analysis stack. To skip the heavy ML bits, use `pip install -e .` for the
> minimum viable install.

### 3. Configure your API key

```bash
cp .env.example .env
# then edit .env and paste your key into GOOGLE_API_KEY=
```

### 4. Pick an interface

**CLI**
```bash
repo-aware-ai --repo /path/to/your/project
```

**Web app** (FastAPI backend + built React frontend)
```bash
# build the frontend (one-time)
cd frontend && npm install && npm run build && cd ..

# start the server
repo-aware-ai-server --repo /path/to/your/project
# open http://localhost:8360
```

**VS Code extension** — see [extension/README.md](extension/README.md).
On first activation the extension bootstraps its own Python venv and installs the
backend automatically — no manual `pip install` needed.

---

## What it does

```
Repository ──► Loader ──► Chunker ──► Embedder ──► FAISS index ─┐
                                                                 │
              ┌──────────────────────────────────────────────────┘
              ▼
   ┌─► Hybrid retrieval (BM25 + vector) ─► Reranker ─► Compressor ─► Gemini ─► Answer + citations
   │
   └─► Repo-map (tree-sitter symbols → graph → Leiden communities → process traces)
```

Features that come on by default:

- **Hybrid retrieval** — combines BM25 keyword scoring with vector similarity via reciprocal rank fusion.
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` re-scores the top candidates for relevance.
- **Contextual compression** — Gemini extracts only the question-relevant lines from each chunk before answering.
- **Multi-query decomposition** — breaks compound questions into sub-questions and merges results.
- **Conversation memory** — last 5 turns of history feed the next answer.
- **Repo map** — tree-sitter parses Python / JS / TS into a symbol graph, detects communities of related code with Leiden, and traces likely call processes.

Each can be disabled via CLI flags (`--no-rerank`, `--no-hybrid`, `--no-expand`, `--no-compress`, `--no-multi-query`).

---

## Configuration

Environment variables (also accepted in `.env`):

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `GOOGLE_API_KEY` | yes | — | Google Gemini API key |
| `GEMINI_MODEL` | no | `models/gemini-2.5-flash` | LLM for answers and compression |
| `EMBEDDING_MODEL` | no | `gemini-embedding-001` | 768-dim embedding model |
| `RAI_PORT` | no | `8360` | Backend HTTP port (use `18360` on Windows — ports near 8360 are often reserved by Hyper-V) |
| `RAI_DATA_DIR` | no | `data/index` | FAISS index + metadata cache root |
| `RAI_AUTO_INDEX` | no | — | Path to auto-index when the server starts |

CLI flags for `repo-aware-ai` (CLI Q&A REPL):

| Flag | Default | Purpose |
|------|---------|---------|
| `--repo` | required | Repository to index |
| `--cache` | `data/index` | Index cache directory |
| `--rebuild` | false | Force rebuild from scratch |
| `--topk` | 6 | Chunks fed to the LLM |
| `--temperature` | 0.2 | LLM sampling temperature |
| `--chunk_size` / `--overlap` | 1800 / 250 | Chunking parameters |
| `--ast-chunk` | false | AST-based chunking for Python |
| `--no-rerank` / `--no-hybrid` / `--no-expand` / `--no-compress` / `--no-multi-query` | — | Disable individual stages |

---

## Repository layout

```
repo-aware-ai/
├── repo_aware_ai/         # Python package (the RAG engine)
│   ├── server.py          # FastAPI app
│   ├── qa.py              # End-to-end pipeline orchestration
│   ├── loader/chunker/embedder/indexer/retriever
│   ├── hybrid_search.py reranker.py compressor.py
│   ├── query_expander.py multi_query.py conversation.py
│   └── repo_map/          # Symbol graph, communities, processes
├── extension/             # VS Code extension (TypeScript, esbuild)
├── frontend/              # React + Vite + Tailwind web app
├── evaluation/            # Eval harness + question set
├── docs/                  # Architecture, contributing, archive
├── main.py                # CLI entry point
├── dev_server.py          # Dev server launcher
└── pyproject.toml
```

For deeper reading: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Status & roadmap

Phases 1–4 are complete. The plan lives at [docs/ROADMAP.md](docs/ROADMAP.md).

Currently working: CLI, web backend, web frontend (chat + repo map), VS Code extension (bootstrap + streaming chat + repo map panel).
Up next: Phase 5 — GitHub Actions CI, PyPI publish, VS Code Marketplace.

---

## Contributing

PRs welcome. Read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for setup,
coding conventions, and how to run the test suite.

## License

[MIT](LICENSE) © Sirjan Singh
