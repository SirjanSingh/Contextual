# Contributing

Thanks for your interest. The project is in active refactor toward a polished
v1.0 — small focused PRs are very welcome.

## Dev setup

```bash
git clone https://github.com/SirjanSingh/repo-aware-ai
cd repo-aware-ai

# Backend
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -e .[all,dev]

cp .env.example .env
# add your GOOGLE_API_KEY

# Frontend
cd frontend && npm install && cd ..

# Extension
cd extension && npm install && cd ..
```

## Running things

```bash
# CLI
repo-aware-ai --repo .

# Dev server (auto-reload)
repo-aware-ai-server --repo .

# Frontend dev (proxies to localhost:8360)
cd frontend && npm run dev

# Extension watch build (then F5 in VS Code with the extension folder open)
cd extension && npm run watch
```

## Testing

```bash
pytest                    # backend tests
ruff check .              # lint
mypy repo_aware_ai        # type check

cd frontend && npm run build   # tsc + vite build
cd extension && npm run build  # esbuild
```

## Coding conventions

- **Python**: `ruff` formatting and `mypy` clean for new code in
  `repo_aware_ai/`. Type hints on public functions. Keep modules focused — the
  pipeline benefits from small composable steps.
- **TypeScript**: strict mode is on for both frontend and extension. Avoid
  `any` in new code. Prefer named exports.
- **Commits**: imperative present tense ("add streaming endpoint", not
  "added"). Conventional prefixes (`feat:`, `fix:`, `chore:`, `docs:`,
  `refactor:`) are encouraged but not enforced.
- **No drive-by reformats**. Keep diffs minimal; changes you didn't intend
  make review hard.

## Before opening a PR

- Tests pass: `pytest && ruff check . && mypy repo_aware_ai`
- Frontend builds: `cd frontend && npm run build`
- Extension builds: `cd extension && npm run build`
- For UI changes, include a screenshot or short clip in the PR description.
- If you touched the indexing or retrieval pipeline, confirm an end-to-end
  query still returns sensible results on a small repo.

## Reporting bugs

Open a GitHub issue with:

- What you ran (CLI command, server URL, extension version)
- What you expected
- What you saw (full traceback / VS Code output channel logs)
- Your OS, Python version, and `pip freeze`

## Adding a new pipeline stage

The query pipeline lives in [`repo_aware_ai/qa.py`](../repo_aware_ai/qa.py). New
stages should:

1. Live in their own module under `repo_aware_ai/`.
2. Lazily import any heavy dependencies.
3. Be optional via a flag on `QAEngine.__init__` and a CLI switch.
4. Degrade gracefully if their dependency is missing — log a warning, return
   the input unchanged.

## Code of conduct

Be kind. Assume good faith. We're all here to make a useful tool.
