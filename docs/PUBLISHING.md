# Publishing Guide

Repo-Aware AI ships as **two artifacts** that must be released in order:

1. **`repo-aware-ai` on PyPI** — the Python backend. The VS Code extension installs
   this from PyPI at first activation, so **PyPI must be published first.**
2. **The VS Code extension** on the Visual Studio Marketplace (and optionally Open VSX).

---

## Prerequisites (one-time)

- A [PyPI account](https://pypi.org/account/register/) with the name `repo-aware-ai`
  available (check <https://pypi.org/project/repo-aware-ai/>). Create a scoped
  **API token** under Account Settings → API tokens.
- A [Visual Studio Marketplace publisher](https://marketplace.visualstudio.com/manage)
  with ID **`repo-aware-ai`** (must match `publisher` in `extension/package.json`).
  Create a **Personal Access Token (PAT)** in Azure DevOps with
  *Marketplace → Manage* scope. See the
  [official guide](https://code.visualstudio.com/api/working-with-extensions/publishing-extension).
- (Optional) An [Open VSX](https://open-vsx.org) account for Cursor / VSCodium users.
- A **public** GitHub repo at the URL declared in `pyproject.toml` /
  `extension/package.json` (`github.com/SirjanSingh/repo-aware-ai`).

---

## 1. Publish the backend to PyPI

```bash
# from repo root, in the venv
pip install build twine
python -m build                 # → dist/*.whl + dist/*.tar.gz
python -m twine check dist/*    # must PASS
python -m twine upload dist/*   # paste your PyPI API token
```

Verify it installs cleanly in a throwaway env:

```bash
python -m venv /tmp/verify && /tmp/verify/bin/pip install "repo-aware-ai[all]"
/tmp/verify/bin/repo-aware-ai --help
```

> Bump `version` in **both** `pyproject.toml` and `extension/package.json` for every
> release — PyPI rejects re-uploads of an existing version.

## 2. Publish the VS Code extension

```bash
cd extension
npm install
npx vsce package --no-dependencies     # sanity-check the .vsix locally first
npx vsce login repo-aware-ai           # paste your Azure DevOps PAT
npx vsce publish --no-dependencies
```

(Optional) Open VSX:

```bash
npx ovsx publish repo-aware-ai-<version>.vsix -p <OPEN_VSX_TOKEN>
```

`--no-dependencies` is correct here because the extension is bundled with esbuild
(`vscode:prepublish`), so node_modules need not be packaged.

## 3. (Optional) Automate with GitHub Actions

Add a `release.yml` triggered on tags that runs the two steps above using
repository secrets `PYPI_API_TOKEN` and `VSCE_PAT`. See
[Publish VS Code Extension action](https://github.com/marketplace/actions/publish-vs-code-extension).

---

## Pre-release checklist

- [ ] `pytest` green; `extension` `tsc` clean; `frontend` `npm run build` clean.
- [ ] Versions bumped in `pyproject.toml` **and** `extension/package.json`.
- [ ] `CHANGELOG.md` (root and `extension/`) updated.
- [ ] `python -m twine check dist/*` PASSES.
- [ ] No secrets committed (`.env` is gitignored; confirm with `git ls-files | grep env`).
- [ ] GitHub repo public; README renders correctly.
- [ ] PyPI published and `pip install "repo-aware-ai[all]"` verified **before** the
      extension goes live (the extension depends on it at runtime).
- [ ] Replace `Development Status :: 3 - Alpha` in `pyproject.toml` when leaving alpha.
