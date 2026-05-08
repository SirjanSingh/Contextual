# Roadmap

The repo is mid-refactor. The goal is a polished v1.0 with three first-class
surfaces: CLI, web app, and VS Code extension.

## Phase 1 — Project foundation ✅
- [x] Snapshot pre-refactor WIP
- [x] Rename `app/` → `repo_aware_ai/` for proper Python packaging
- [x] `pyproject.toml` (installable as `repo-aware-ai` console script)
- [x] Pin core deps; move heavy ML to optional extras
- [x] Repo-root `LICENSE`, `.env.example`, `.editorconfig`
- [x] `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `ROADMAP.md`
- [x] Move stale prompts and notes to `docs/archive/`

## Phase 2 — Backend hardening ✅
- [x] Real Gemini streaming for `/query/stream` via `stream_ask()` generator
- [x] Add `tenacity`-based retries on Gemini API calls (429 / RESOURCE_EXHAUSTED / 503 / 500) via `_retry.py`
- [x] Thread-safety on `ConversationHistory` (added `threading.Lock`)
- [x] Derive embedding dimension from the embedder, drop the hardcoded 768 (`_MODEL_DIMS` + `dimension` property)
- [x] Make `/search` `top_k` a request parameter
- [x] Surface background-indexing errors reliably (status flips to `"error"`)
- [x] Smoke test suite (`pytest`, 14 tests across loader / chunker / conversation / indexer)

## Phase 3 — VS Code extension end-to-end ✅
- [x] Bootstrap a venv on first activation and `pip install` the backend (`backendBootstrap.ts`)
- [x] Reliable install-source resolution: checks local pyproject.toml → falls back to PyPI
- [x] Wire `chatPanel` `openSource` handler (open file at start character)
- [x] Strip dead endpoints from `backendClient.ts`; connected `/query/stream`
- [x] Switch chat to `/query/stream` for live token streaming with blinking cursor
- [x] Surface backend errors as VS Code notifications with "Show Logs" action
- [x] Status-bar states: down / indexing / ready / error
- [x] "Show backend logs" command (always registered, even if activation fails)
- [x] `PYTHONIOENCODING=utf-8` + `PYTHONUNBUFFERED=1` for reliable sidecar I/O

## Phase 4 — Frontend polish ✅
- [x] `ErrorBoundary` around the app
- [x] `VITE_BACKEND_URL` env support (read via `loadEnv`; WS URL auto-derived)
- [x] Render real model + embedding model from `/health` (`setBackendInfo()` in store)
- [x] Typed `RepoMapSummary`, `SymbolDetail`, `CommunityDetail`, `ProcessDetail` in `client.ts`
- [x] Tightened `useStore` — removed dead `showRepoExplorer` / `showUploadZone` state

## Phase 5 — OSS readiness
- [ ] GitHub Actions CI: lint, typecheck, build (all surfaces), tests
- [ ] Issue + PR templates, `CODE_OF_CONDUCT.md`
- [ ] Demo GIF in README
- [ ] Publish backend to PyPI
- [ ] Publish extension to VS Code Marketplace

## Beyond v1.0 (deferred)
- Multi-repo indexing in one engine
- Local embedding fallback (no external API)
- Persistent conversation memory across sessions
- Web app: code editor with inline RAG hover
- Extension: dependency-graph TreeView fed by `/graph/dependencies`
