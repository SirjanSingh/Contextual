# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI: backend tests on Linux + Windows, py3.10/3.11/3.12; extension and frontend typecheck/build.
- Issue templates (bug, feature) and pull request template.
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1).
- `.vscodeignore` so packaged `.vsix` ships only built output.

## [0.1.0] — 2026-05-09

First public preview. Phases 1–4 of the refactor are complete.

### Added
- **Backend (`repo_aware_ai/`)**
  - Installable Python package with `repo-aware-ai` and `repo-aware-ai-server` console scripts.
  - Real Gemini streaming via `stream_ask()` and `/query/stream` SSE endpoint.
  - Tenacity-based retries on 429 / 503 / 500 Gemini errors (`_retry.py`).
  - Thread-safe `ConversationHistory` (Lock-protected).
  - Embedder derives dimension from the model dict, no more hardcoded 768.
  - `/search` accepts a `top_k` request parameter.
  - Background indexing errors flip status to `"error"` instead of hanging.
  - Pytest smoke tests covering loader, chunker, conversation, and index/retrieve cycle.

- **VS Code extension (`extension/`)**
  - End-to-end activation: bootstrap a private venv, `pip install` the backend, spawn uvicorn sidecar, auto-index workspace.
  - Live token streaming in the chat panel (with blinking cursor while the answer is being generated).
  - "Show Backend Logs" and "Set API Key" commands always registered, even when activation fails.
  - Status-bar states: down / indexing / ready / error.
  - `openSource` handler opens cited files at the chunk's start character.

- **Web frontend (`frontend/`)**
  - `ErrorBoundary` around the app.
  - `VITE_BACKEND_URL` env support; WebSocket URL auto-derived.
  - Real model + embedding model rendered from `/health`.
  - Typed repo-map summary, symbol, community, and process detail responses.

### Fixed
- Loader pruning ignores dynamic venv names (`.venv311`, `myenv`, `venv-py312`) via `fnmatch` glob patterns. A repo that previously produced 144,288 chunks now produces 397.
- Lockfiles (`package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, `poetry.lock`, `uv.lock`) are excluded from indexing.
- Auto-index startup hook no longer crashes on Windows: it now calls the route handler in-process instead of making an HTTP self-loopback before uvicorn binds the socket.
