# Repo-Aware AI

> **Talk to your codebase.** Ask natural-language questions and get answers grounded
> in your actual code — with file-anchored citations. Powered by Google Gemini and a
> local RAG (retrieval-augmented generation) engine.

Repo-Aware AI indexes your workspace, retrieves the most relevant code chunks with
hybrid search + cross-encoder reranking, and answers in the editor. Nothing about
your code is stored on a third-party server — the index lives on your machine and
only the retrieved snippets are sent to Gemini to compose an answer.

---

## Features

- **Ask Codebase** (`Ctrl+Shift+A` / `Cmd+Shift+A`) — ask anything about the repo; answers cite `path/file.py:start-end`.
- **Explain Selection** — right-click any code → get a plain-language explanation in context.
- **Find Related Code** (`Ctrl+Shift+R` / `Cmd+Shift+R`) — jump to semantically related code, not just text matches.
- **Explain This Repository** — a high-level tour of what the project does and how it's organized.
- **Chat sidebar** — a dedicated Repo AI view with streaming answers.
- **Repo Map panel** — a dependency/community graph of your code.
- **CodeLens & hover hints** — lightweight action hints on functions.
- **Auto-index on save** — the index stays fresh as you work (toggleable).

## Requirements

- **Python 3.10+** on your `PATH` (or set `repoAwareAI.pythonPath`). On first run the
  extension creates its **own private virtual environment** and installs the backend
  automatically — you do **not** need to `pip install` anything yourself.
- A **Google Gemini API key** — create a free one at
  [Google AI Studio](https://aistudio.google.com/app/apikey).

## Getting started

1. Install the extension.
2. When prompted, paste your Google API key (stored securely in VS Code SecretStorage),
   or run **`Repo AI: Set API Key`** from the Command Palette.
3. Wait for first-run setup — the extension bootstraps a Python venv, installs the
   backend, and indexes your workspace. Progress is shown in notifications and the
   **Repo AI Backend** output channel (**`Repo AI: Show Backend Logs`**).
4. Press `Ctrl+Shift+A` and ask away.

> **First-run notes:** the initial setup downloads Python dependencies (and, if the
> reranker is enabled, a ~500 MB cross-encoder model on first query). This is a
> one-time cost. Subsequent activations reuse the cached venv and index.

## Extension settings

| Setting | Default | Description |
|---|---|---|
| `repoAwareAI.googleApiKey` | — | Gemini API key. **Prefer the `Set API Key` command** — it uses SecretStorage and won't leak into settings exports. |
| `repoAwareAI.pythonPath` | `python` | System Python (≥ 3.10) used to bootstrap the private venv. |
| `repoAwareAI.port` | `8360` | Backend sidecar port. |
| `repoAwareAI.topK` | `6` | Chunks fed to the LLM per question. |
| `repoAwareAI.chunkSize` | `1800` | Chunk size in characters. |
| `repoAwareAI.useReranker` | `true` | Cross-encoder reranking. |
| `repoAwareAI.useHybridSearch` | `true` | BM25 + vector retrieval. |
| `repoAwareAI.useQueryExpansion` | `true` | Synonym/related-query expansion. |
| `repoAwareAI.useCompression` | `true` | Extract only question-relevant lines before answering. |
| `repoAwareAI.autoIndex` | `true` | Re-index on file save. |

## Privacy

Your repository is indexed **locally**. The FAISS index and metadata live under the
extension's global storage on your machine. Only the natural-language question and the
small set of retrieved code snippets are sent to Google Gemini to generate an answer.
Your API key is stored in VS Code SecretStorage and never written to `settings.json`.

## Commands

All commands are under the **Repo AI** category in the Command Palette: Ask Codebase,
Explain Selection, Find Related Code, Explain This Repository, Rebuild Index, Open Chat
Panel, Open Repo Map, Set API Key, Show Backend Logs.

## Troubleshooting

- **"Python 3.10+ not found"** — install Python and set `repoAwareAI.pythonPath` to its full path.
- **Backend won't start** — run **`Repo AI: Show Backend Logs`** and check the output.
- **Answers seem stale** — run **`Repo AI: Rebuild Index`**.
- **Port conflict (Windows)** — set `repoAwareAI.port` to e.g. `18360`; ports near 8360 are often reserved by Hyper-V.

## License

[MIT](LICENSE) © Sirjan Singh. Source: <https://github.com/SirjanSingh/repo-aware-ai>
