# Repo-Aware AI — Stabilization & Debug Plan

This plan is split into **3 phases**. Phase 1 gives you a local test server + debug logs so you can actually see what's happening. Phase 2 fixes the bugs. Phase 3 is verification.

## Phase 1: Local Test Server + Debug Logging

- [x] Fix config model names → `gemini-2.5-flash` + `gemini-embedding-001`
- [x] Add `dev_server.py` — standalone backend with `--reload` and auto-index
- [x] Add test dashboard at `/dev/` endpoint (separate from prod frontend)
  - Endpoint tester (health, index, query, search, graph)
  - RepoMap 3D visualization for testing
  - Live server logs panel
- [x] Add Python `logging` to `server.py`
- [x] Add Python `logging` to `qa.py`, `embedder.py`, `indexer.py`
- [x] Change extension backend log-level from `warning` → `info`

## Phase 2: Bug Fixes

- [ ] Fix auto-index on save calling full `rebuildIndex()` (too heavy)
- [ ] Fix CodeLens N+1 problem (50 requests per file)
- [ ] Fix hover firing on common keywords
- [ ] Fix RepoMap error handling (hangs on "Loading...")
- [ ] Add folder selection to extension

## Phase 3: Verification

- [ ] Test backend standalone via `dev_server.py` + test dashboard
- [ ] Verify each endpoint returns correct responses
- [ ] Test extension with debug output channel visible
