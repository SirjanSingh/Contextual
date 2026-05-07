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

## Phase 2 — Backend hardening
- [ ] Real Gemini streaming for `/query/stream` (or remove if unsupported)
- [ ] Add `tenacity`-based retries on Gemini API calls (auth / 429 / transient)
- [ ] Thread-safety on `ConversationHistory`
- [ ] Derive embedding dimension from the embedder, drop the hardcoded 768
- [ ] Make `/search` `top_k` a request parameter
- [ ] Surface background-indexing errors reliably (status flips to `"error"`)
- [ ] Smoke test suite (`pytest`, ~10 tests, mocked LLM)

## Phase 3 — VS Code extension end-to-end
- [ ] Bootstrap a venv on first activation and `pip install` the backend
- [ ] Reliable `_resolveBackendRoot()` (installed CLI → bundled source → repo)
- [ ] Wire `chatPanel` `openSource` handler (open file at line)
- [ ] Strip dead endpoints from `backendClient.ts` or connect them
- [ ] Fix Windows path normalisation in `findRelated.ts`
- [ ] Switch chat to `/query/stream` for streaming UX
- [ ] Surface backend errors as VS Code notifications, not silent
- [ ] Status-bar states: down / indexing / ready / error
- [ ] "Show backend logs" command

## Phase 4 — Frontend polish
- [ ] Drop unused deps (`react-syntax-highlighter`, `react-markdown`)
- [ ] `ErrorBoundary` around the app
- [ ] `VITE_BACKEND_URL` env support
- [ ] Render real model from `/health` (not hardcoded badge)
- [ ] Loading skeletons for repo-map detail panels
- [ ] Tighten `useStore` types — kill `any` in `RepoMapView`

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
