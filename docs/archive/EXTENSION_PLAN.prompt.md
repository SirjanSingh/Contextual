# Repo-Aware AI — VS Code Extension Implementation Plan

> **Purpose:** Step-by-step plan for shipping repo-aware-ai as a VS Code extension.  
> **Agent:** Hand each phase to Claude Sonnet 4.6 with the exact instructions below.  
> **Date:** March 2026

---

## Project Structure (Final Target)

```
repo-aware-ai/
├── extension/                          # NEW — VS Code extension (TypeScript)
│   ├── package.json                    # Extension manifest + contributions
│   ├── tsconfig.json
│   ├── esbuild.js                      # Bundler config
│   ├── .vscodeignore
│   ├── src/
│   │   ├── extension.ts                # Activation + deactivation
│   │   ├── constants.ts                # Port, endpoints, config keys
│   │   ├── services/
│   │   │   ├── backendProcess.ts       # Spawn/kill Python sidecar
│   │   │   ├── backendClient.ts        # HTTP + WebSocket client
│   │   │   ├── indexManager.ts         # Auto-index on save, status tracking
│   │   │   └── cacheManager.ts         # Extension settings ↔ backend config
│   │   ├── providers/
│   │   │   ├── codeLensProvider.ts     # Inline "Related chunks" annotations
│   │   │   ├── hoverProvider.ts        # RAG-powered hover explanations
│   │   │   ├── searchProvider.ts       # Semantic search as workspace search
│   │   │   └── diagnosticProvider.ts   # Dead code / weak reference warnings
│   │   ├── commands/
│   │   │   ├── askQuestion.ts          # Quick-pick → query → inline result
│   │   │   ├── explainSelection.ts     # Right-click → explain code
│   │   │   ├── findRelated.ts          # Find semantically related files/funcs
│   │   │   ├── explainRepo.ts          # Generate repo summary
│   │   │   └── rebuildIndex.ts         # Force re-index
│   │   ├── views/
│   │   │   ├── chatPanel.ts            # Webview-based chat sidebar
│   │   │   ├── repoMapPanel.ts         # 3D dependency graph webview
│   │   │   └── statusBar.ts            # "Index: 1,247 chunks ✓" in status bar
│   │   └── webview/
│   │       ├── chat/                   # React chat UI (lightweight)
│   │       └── repo-map/              # Three.js graph (port from frontend)
├── app/                                # EXISTING — Python backend (keep as-is)
│   ├── server.py                       # FastAPI (runs as sidecar)
│   └── ...
├── main.py                             # EXISTING — CLI entry
├── requirements.txt                    # EXISTING
└── frontend/                           # EXISTING — Standalone web UI (optional)
```

---

## PHASE 0 — Backend Prep (Python side modifications)

**Goal:** Make the existing backend extension-friendly without breaking the web UI.

**Files to read first:** `app/server.py`, `app/qa.py`, `app/llm.py`, `app/config.py`, `app/retriever.py`, `app/indexer.py`

### Step 0.1 — Add direct workspace indexing endpoint

Currently `/upload/directory` requires multipart file upload. For VS Code, we need a simpler path.

**File:** `app/server.py`  
**Add new endpoint:**
```python
class IndexDirectoryRequest(BaseModel):
    repo_path: str  # Absolute path on disk

@app.post("/index/directory")
async def index_directory(req: IndexDirectoryRequest):
    """Index a local directory by path (no file upload needed)."""
```
This endpoint takes a local filesystem path and indexes it directly — the extension just sends the workspace folder path.

### Step 0.2 — Add streaming response endpoint

**File:** `app/server.py`
```python
from fastapi.responses import StreamingResponse

@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """SSE stream: yields partial answer tokens."""
```
This requires modifying `LLMClient.answer()` in `app/llm.py` to yield tokens via `generate_content_stream()` instead of blocking.

### Step 0.3 — Add semantic search endpoint (no LLM, just retrieval)

**File:** `app/server.py`
```python
@app.post("/search")
async def semantic_search(req: QueryRequest):
    """Return top-k chunks without LLM answer (for hover/codelens)."""
    # Returns: { chunks: [{ source, text, start_char, end_char, score }] }
```
This is critical — CodeLens and Hover providers need fast retrieval without waiting for LLM.

### Step 0.4 — Add file-level context endpoint

**File:** `app/server.py`
```python
@app.get("/context/file")
async def file_context(file_path: str):
    """Return all indexed chunks for a specific file + related files."""
    # Returns: { chunks: [...], related_files: [...] }
```
Powers hover tooltips and CodeLens — "what's indexed about THIS file?"

### Step 0.5 — Configurable port via env var

**File:** `app/server.py` or startup script
```python
PORT = int(os.environ.get("RAI_PORT", "8360"))
```
Use a non-standard port (8360) to avoid conflicts. Extension will set this env var when spawning.

### Step 0.6 — Health endpoint returns version

```python
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "model": ..., "embedding_model": ...}
```
Extension checks version compatibility.

---

## PHASE 1 — Extension Scaffold

**Goal:** Working VS Code extension that spawns backend and shows status.

### Step 1.1 — Initialize extension project

```powershell
cd g:\projs\repo-aware-ai
mkdir extension
cd extension
npm init -y
npm install -D @types/vscode typescript esbuild @vscode/vsce
```

### Step 1.2 — Create `extension/package.json` manifest

This is the most important file — it declares everything the extension does:

```jsonc
{
  "name": "repo-aware-ai",
  "displayName": "Repo-Aware AI",
  "description": "Understands your codebase structure — semantic search, code radar, and repo mapping",
  "version": "0.1.0",
  "engines": { "vscode": "^1.85.0" },
  "categories": ["AI", "Other"],
  "activationEvents": ["onStartupFinished"],
  "main": "./dist/extension.js",
  "contributes": {
    "commands": [
      { "command": "repoAwareAI.askQuestion", "title": "Ask Codebase", "category": "Repo AI" },
      { "command": "repoAwareAI.explainSelection", "title": "Explain Selection", "category": "Repo AI" },
      { "command": "repoAwareAI.findRelated", "title": "Find Related Code", "category": "Repo AI" },
      { "command": "repoAwareAI.explainRepo", "title": "Explain This Repository", "category": "Repo AI" },
      { "command": "repoAwareAI.rebuildIndex", "title": "Rebuild Index", "category": "Repo AI" },
      { "command": "repoAwareAI.openChat", "title": "Open Chat Panel", "category": "Repo AI" },
      { "command": "repoAwareAI.openRepoMap", "title": "Open Repo Map", "category": "Repo AI" }
    ],
    "configuration": {
      "title": "Repo-Aware AI",
      "properties": {
        "repoAwareAI.googleApiKey": { "type": "string", "description": "Google API Key for Gemini" },
        "repoAwareAI.pythonPath": { "type": "string", "default": "python", "description": "Path to Python executable" },
        "repoAwareAI.port": { "type": "number", "default": 8360, "description": "Backend server port" },
        "repoAwareAI.topK": { "type": "number", "default": 6 },
        "repoAwareAI.chunkSize": { "type": "number", "default": 1800 },
        "repoAwareAI.useReranker": { "type": "boolean", "default": true },
        "repoAwareAI.useHybridSearch": { "type": "boolean", "default": true },
        "repoAwareAI.useQueryExpansion": { "type": "boolean", "default": true },
        "repoAwareAI.useCompression": { "type": "boolean", "default": true },
        "repoAwareAI.autoIndex": { "type": "boolean", "default": true, "description": "Auto-reindex on file save" }
      }
    },
    "menus": {
      "editor/context": [
        { "command": "repoAwareAI.explainSelection", "when": "editorHasSelection", "group": "repoai" },
        { "command": "repoAwareAI.findRelated", "group": "repoai" }
      ]
    },
    "keybindings": [
      { "command": "repoAwareAI.askQuestion", "key": "ctrl+shift+a", "mac": "cmd+shift+a" },
      { "command": "repoAwareAI.findRelated", "key": "ctrl+shift+r", "mac": "cmd+shift+r" }
    ],
    "viewsContainers": {
      "activitybar": [
        { "id": "repoAwareAI", "title": "Repo AI", "icon": "resources/icon.svg" }
      ]
    },
    "views": {
      "repoAwareAI": [
        { "type": "webview", "id": "repoAwareAI.chatView", "name": "Chat" },
        { "id": "repoAwareAI.indexStatus", "name": "Index Status" }
      ]
    }
  }
}
```

### Step 1.3 — Create `extension/src/extension.ts`

```typescript
// Activation: spawn backend, register commands, start status bar
// Deactivation: kill backend process, dispose providers
export function activate(context: vscode.ExtensionContext) {
    // 1. Start sidecar backend (backendProcess.ts)
    // 2. Wait for /health to respond
    // 3. Send workspace path to /index/directory
    // 4. Register all commands
    // 5. Register CodeLens + Hover providers
    // 6. Create status bar item
    // 7. Watch workspace for file saves → trigger re-index
}
```

### Step 1.4 — Create `extension/src/services/backendProcess.ts`

```typescript
// Spawns: python -m uvicorn app.server:app --port {PORT} --host 127.0.0.1
// Sets env: GOOGLE_API_KEY, RAI_PORT
// Monitors: stdout/stderr → output channel "Repo AI Backend"
// Health check: poll /health every 500ms until ready (max 30s)
// Kill: on deactivation or extension host shutdown
```

The backend project root (where `app/` lives) is resolved relative to the extension install path. During development, it's the workspace root.

### Step 1.5 — Create `extension/src/services/backendClient.ts`

```typescript
export class BackendClient {
    constructor(private baseUrl: string) {}
    
    async health(): Promise<HealthResponse>
    async query(question: string): Promise<{ answer: string; sources: string[] }>
    async queryStream(question: string): AsyncGenerator<string>  // SSE
    async search(query: string, topK?: number): Promise<ChunkResult[]>
    async fileContext(filePath: string): Promise<FileContextResult>
    async indexDirectory(repoPath: string): Promise<void>
    async indexStatus(): Promise<{ status: string; info: object }>
    async rebuildIndex(): Promise<void>
    async clearRepository(): Promise<void>
    connectProgress(uploadId: string): WebSocket
}
```

### Step 1.6 — Create `extension/src/views/statusBar.ts`

```typescript
// Shows: "$(database) Repo AI: Indexing..." or "$(check) Repo AI: 1,247 chunks"
// Click: opens command palette with Repo AI commands
// Polls /index/status every 5 seconds when idle
// Polls every 500ms during indexing
```

### Step 1.7 — Build & test locally

```powershell
cd extension
npm run build
# Press F5 in VS Code → launches Extension Development Host
# Verify: status bar appears, backend starts, index builds
```

**Deliverable Phase 1:** Extension activates, spawns backend, indexes workspace, shows status bar.

---

## PHASE 2 — Core Commands (The Chat + Search)

**Goal:** Users can ask questions and find related code.

**Files to read first:** `extension/src/services/backendClient.ts`, `app/server.py` endpoints

### Step 2.1 — `askQuestion` command

```typescript
// Ctrl+Shift+A → InputBox "Ask about your codebase..."
// → POST /query → show answer in:
//   Option A: Notification with "Show Full" button → opens temp markdown doc
//   Option B: Chat webview panel (preferred)
// Sources: clickable links that open the file at the right position
```

### Step 2.2 — `explainSelection` command

```typescript
// Right-click selected code → "Explain Selection"
// Sends selected text + file path as context to /query
// Question: "Explain what this code does and how it fits in the codebase: ```{selection}```"
// Shows result in panel beside the editor
```

### Step 2.3 — `findRelated` command

```typescript
// Ctrl+Shift+R → uses current file or selection
// → POST /search with file content as query
// → Shows QuickPick list of related files/chunks
// → Clicking opens file at chunk location
// This is the "killer demo" — instantly find similar code
```

### Step 2.4 — `explainRepo` command

```typescript
// Queries: "Describe the overall architecture of this project:
//   main entry points, core modules, data flow, and key design patterns"
// → Opens a new markdown preview tab with the result
// → Optionally cached as .repo-summary.md
```

### Step 2.5 — `rebuildIndex` command

```typescript
// POST /index/rebuild
// Shows progress notification with cancel button
// Updates status bar during rebuild
```

### Step 2.6 — Chat webview sidebar

```typescript
// Registers WebviewViewProvider for the sidebar panel
// HTML/CSS/JS (simple — no React needed for MVP):
//   - Message list (markdown rendered)
//   - Input box at bottom
//   - Source links clickable → vscode.open
//   - Streaming token display (via SSE from /query/stream)
```

For MVP, use plain HTML + VS Code's webview toolkit CSS. Port the React chat later.

**Deliverable Phase 2:** Full question-answer loop, right-click explain, find-related, repo summary.

---

## PHASE 3 — Code Radar (The Differentiator)

**Goal:** Inline intelligence that no other extension provides.

**Files to read first:** `extension/src/services/backendClient.ts`, VS Code `vscode.languages` API docs

### Step 3.1 — CodeLens Provider

```typescript
// Registers for: python, javascript, typescript, go, rust, java, etc.
// For each function/class definition:
//   1. Parse symbol name from document symbols API
//   2. POST /search with symbol name as query
//   3. Show: "🔍 N related chunks across M files"
//   4. Click → show QuickPick of related chunks
// Caching: Cache results per file, invalidate on save
// Debouncing: Only compute when file is idle for 2 seconds
// Limit: Only top 50 symbols per file to avoid perf issues
```

### Step 3.2 — Hover Provider

```typescript
// When hovering over a function/class name:
//   1. POST /search with hovered symbol name
//   2. Show hover card:
//      "**Used in:** file1.py, file2.py (3 references)
//       **Related to:** similar_function(), helper_func()
//       **Purpose:** <1-sentence from nearest doc chunk>"
// Caching: LRU cache of recent hovers (100 entries)
// Timeout: 2s max, hide if backend slow
```

### Step 3.3 — Auto-index on save

```typescript
// workspace.onDidSaveTextDocument → 
//   debounce 5s → POST /index/rebuild
// Only triggers if file is in indexed extensions
// Shows subtle status bar spinner during re-index
// Configurable: repoAwareAI.autoIndex setting
```

### Step 3.4 — File-level decorations

```typescript
// In the Explorer tree, show badges:
//   - Files with high chunk density: "📊 12 chunks"
//   - Recently queried files: subtle highlight
// Uses TreeDecorationProvider API
```

**Deliverable Phase 3:** CodeLens annotations on functions, hover cards, auto-reindexing.

---

## PHASE 4 — Repo Map (Visual Differentiator)

**Goal:** Interactive 3D codebase visualization in a webview.

**Files to read first:** `frontend/src/components/three/Scene.tsx`, `app/indexer.py`, `app/ast_chunker.py`

### Step 4.1 — Add graph data endpoints to backend

**File:** `app/server.py`
```python
@app.get("/graph/dependencies")
async def dependency_graph():
    """Return file-to-file dependency edges from import analysis."""
    # Parse imports from indexed chunks
    # Return: { nodes: [{id, label, type, chunkCount}], edges: [{source, target, type}] }

@app.get("/graph/clusters")  
async def semantic_clusters():
    """Return k-means clusters of FAISS vectors."""
    # Run sklearn.cluster.KMeans on stored vectors
    # Return: { clusters: [{ id, centroid_label, files: [...] }] }
```

### Step 4.2 — Build Repo Map webview

```typescript
// Webview panel (not sidebar — needs space)
// Uses Three.js (port from frontend/src/components/three/)
// Shows:
//   - Nodes = files (size = chunk count)
//   - Edges = import relationships
//   - Colors = semantic clusters
//   - Click node → opens file
//   - Hover node → shows file summary
//   - Search box → highlights matching nodes
// Communication: postMessage between extension ↔ webview
```

### Step 4.3 — Change impact visualization

```typescript
// When a file is modified (dirty):
//   1. POST /search with file content
//   2. Get semantically coupled files
//   3. Highlight them in the repo map
//   4. Show notification: "3 files may need changes too"
```

**Deliverable Phase 4:** Interactive 3D repo map, semantic clusters, change impact.

---

## PHASE 5 — Polish & Ship

### Step 5.1 — Onboarding flow

```
On first activation:
  1. Check for GOOGLE_API_KEY → if missing, prompt with input box
  2. Check Python availability → if missing, show install link
  3. Auto-install Python requirements (pip install -r requirements.txt)
  4. Index workspace → show progress
  5. Show welcome walkthrough (contributes.walkthroughs in package.json)
```

### Step 5.2 — Settings UI

```
contributes.configuration already declared in Phase 1.
Add a dedicated settings webview with:
  - API key input (stored in SecretStorage, NOT settings.json)
  - Model selector dropdown
  - Feature toggles (reranker, hybrid, compression, etc.)
  - Chunk size slider
  - "Test Connection" button
```

### Step 5.3 — Error handling & resilience

```
- Backend crash → auto-restart (max 3 times)
- Network timeout → retry with exponential backoff
- API key invalid → clear error message + settings link
- Large repo warning → "This repo has 50k+ files, indexing may take a while"
- Graceful degradation: if backend down, disable CodeLens/Hover, show status bar warning
```

### Step 5.4 — Performance optimization

```
- CodeLens: batch requests, cache per file, invalidate on save
- Hover: LRU cache (100 entries), 2s timeout
- Search: debounce all user-triggered queries by 300ms
- Backend: connection pooling, keep-alive
- Startup: lazy-activate (don't spawn backend until first command)
```

### Step 5.5 — Package & publish

```powershell
cd extension
npx vsce package          # Creates .vsix file
npx vsce publish          # Publishes to VS Code Marketplace
```

**Bundle strategy:**
- Extension: esbuild-bundled TypeScript → single `dist/extension.js`
- Backend: Ship as Python source in `backend/` folder inside extension
- Requirements: `pip install` on first activation (with progress)
- Alternative: PyInstaller binary for zero-Python-dependency install (Phase 5+)

### Step 5.6 — Marketplace listing

```
- Icon: Design a logo (neural network + code brackets)
- Screenshots: CodeLens, Hover, Chat, Repo Map
- README: Feature comparison table vs Copilot/Cody/Continue
- Tags: ai, rag, codebase, semantic-search, code-understanding
- Demo GIF: 30-second workflow showing find-related + ask
```

---

## Implementation Order (Optimized for Impact)

| Order | Phase | What | Impact |
|-------|-------|------|--------|
| 1 | 0.1-0.3 | Backend prep (new endpoints) | Unblocks everything |
| 2 | 1.1-1.7 | Extension scaffold + sidecar | Foundation |
| 3 | 2.3 | `findRelated` command | **Highest wow-factor, lowest effort** |
| 4 | 2.1 | `askQuestion` command | Core utility |
| 5 | 2.6 | Chat sidebar (basic) | Expected baseline |
| 6 | 3.1 | CodeLens provider | **The differentiator** |
| 7 | 3.2 | Hover provider | Compounds CodeLens |
| 8 | 3.3 | Auto-index on save | Quality of life |
| 9 | 2.2 | Explain selection | Right-click bonus |
| 10 | 2.4 | Explain repo | Onboarding feature |
| 11 | 4.1-4.3 | Repo Map | Visual differentiator |
| 12 | 5.1-5.6 | Polish + publish | Ship it |

---

## Key Guidance for Claude Sonnet 4.6

When handing off each phase, include these instructions:

1. **"Read these files first"** — always list the exact files the agent needs to read before modifying
2. **"Don't break existing functionality"** — the web UI and CLI must keep working
3. **"Use the VS Code API, not hacks"** — `vscode.languages.registerCodeLensProvider`, not regex parsing
4. **"Test with F5"** — every step should be verifiable in Extension Development Host
5. **"Keep the backend communication simple"** — HTTP fetch + WebSocket, no gRPC or complex protocols
6. **Backend is the brain, extension is the hands** — don't duplicate RAG logic in TypeScript

---

## Current Backend Reference

### Existing Endpoints (app/server.py)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Server health & model info |
| GET | `/index/status` | Current indexing status |
| POST | `/query` | Query indexed repository |
| POST | `/index/rebuild` | Force rebuild index |
| POST | `/upload/directory` | Upload repo files (multipart) |
| GET | `/upload/progress/{upload_id}` | Poll progress |
| WS | `/ws/indexing/{upload_id}` | Live progress stream |
| DELETE | `/repository/clear` | Clear indexed repo |

### Schemas

```python
QueryRequest:  { question: str, repo_path?: str }
QueryResponse: { answer: str, sources: List[str] }
```

### Config (app/config.py)

```
GOOGLE_API_KEY   — Required
GEMINI_MODEL     — Default: gemini-2.0-flash
EMBEDDING_MODEL  — Default: gemini-embedding-001
```

### QAEngine (app/qa.py)

```python
QAEngine(
    repo_root, cache_base, embedder, llm,
    chunk_size=1800, overlap=250, top_k=6,
    use_reranker=True, use_hybrid_search=True,
    use_query_expansion=True, use_compression=True,
    use_ast_chunking=False, use_multi_query=True
)
# .build(force_rebuild=False) → indexes repo
# .ask(question) → (answer, sources)
# .clear_conversation() → resets history
```

---

## Start Here

```
Phase 0 + Phase 1 → first prompt to Claude Sonnet 4.6
"Read app/server.py, app/qa.py, app/config.py, app/llm.py, app/indexer.py, app/retriever.py.
Add the new endpoints (Steps 0.1-0.6) without breaking existing endpoints.
Then scaffold the VS Code extension (Steps 1.1-1.7) with sidecar backend spawning and status bar."
```
