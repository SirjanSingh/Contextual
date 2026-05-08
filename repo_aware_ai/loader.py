"""Repository file loader.

Walks a repo, filters by extension, prunes ignored directories, and reads
file text safely.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("rai.loader")


# Directories with these EXACT names are pruned during the walk.
DEFAULT_IGNORE_DIRS: set[str] = {
    # VCS
    ".git",
    ".hg",
    ".svn",
    ".bzr",
    # IDE / editor
    ".idea",
    ".vscode",
    ".vs",
    ".history",
    # Python caches and builds
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".cache",
    ".eggs",
    "build",
    "dist",
    "out",
    "wheelhouse",
    "site-packages",
    # Node / JS
    "node_modules",
    ".next",
    ".nuxt",
    ".expo",
    ".parcel-cache",
    ".turbo",
    ".svelte-kit",
    ".astro",
    ".vercel",
    ".netlify",
    # Coverage / docs build
    "coverage",
    "htmlcov",
    "_build",
    # Other language build dirs
    "target",
    "bin",
    "obj",
    "vendor",
    ".gradle",
    ".m2",
    # Scratch / data
    ".agent",
    ".claude",
    "tmp",
    # This project's own writable dirs
    "data",
    "debug_logs",
}


# Directories matching any of these glob patterns are pruned. Catches
# user-named venvs like `.venv311`, `myenv`, `venv-py312`, etc.
DEFAULT_IGNORE_DIR_PATTERNS: tuple[str, ...] = (
    ".venv*",
    "venv*",
    "env-*",
    "*.egg-info",
)


DEFAULT_IGNORE_FILES: set[str] = {
    ".env",
    ".env.local",
    ".env.production",
    ".DS_Store",
    "Thumbs.db",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "uv.lock",
}


DEFAULT_ALLOWED_EXTS: set[str] = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".ini",
    ".cfg",
}


# Files larger than this are skipped wholesale. Big files are usually
# generated artifacts, model weights, or vendored libs that pollute the index.
DEFAULT_MAX_FILE_BYTES = 2_000_000


@dataclass(frozen=True)
class RepoFile:
    """A file under the repo with its (POSIX-relative) path and contents."""

    path: str  # POSIX-style relative path from repo root
    abs_path: str  # absolute path on disk
    text: str


def _matches_pattern(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _safe_read_text(p: Path, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> str:
    """Read text safely; skip oversize files; tolerate encoding issues."""
    try:
        if p.stat().st_size > max_bytes:
            return ""
    except OSError:
        return ""

    try:
        return p.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        try:
            return p.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""
    except Exception:
        return ""


def load_repo_files(
    repo_root: str | Path,
    allowed_exts: set[str] = DEFAULT_ALLOWED_EXTS,
    ignore_dirs: set[str] = DEFAULT_IGNORE_DIRS,
    ignore_dir_patterns: Iterable[str] = DEFAULT_IGNORE_DIR_PATTERNS,
    ignore_files: set[str] = DEFAULT_IGNORE_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> list[RepoFile]:
    """Walk `repo_root` and return readable text files matching `allowed_exts`.

    Pruning rules:
    - Directories whose basename is in `ignore_dirs`
      OR matches any glob in `ignore_dir_patterns` are skipped wholesale.
    - Hidden directories starting with `.` (other than the repo root itself)
      are kept by default — many useful tracked directories like `.github/`
      are hidden. Specific dot-dirs we don't want appear in the ignore list.
    - Files in `ignore_files` are skipped by basename.
    - Files larger than `max_file_bytes` are skipped.
    - Empty / whitespace-only files are skipped.

    Returns: list of RepoFile with POSIX-separated relative paths.
    """
    repo_root = Path(repo_root).resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise ValueError(f"repo_root is not a folder: {repo_root}")

    patterns = tuple(ignore_dir_patterns)
    results: list[RepoFile] = []
    skipped_dirs = 0

    for root, dirs, files in os.walk(repo_root):
        # Prune ignored dirs in-place so os.walk skips them.
        keep: list[str] = []
        for d in dirs:
            if d in ignore_dirs or _matches_pattern(d, patterns):
                skipped_dirs += 1
                continue
            keep.append(d)
        dirs[:] = keep

        for name in files:
            if name in ignore_files:
                continue
            p = Path(root) / name
            if p.suffix.lower() not in allowed_exts:
                continue

            text = _safe_read_text(p, max_bytes=max_file_bytes)
            if not text.strip():
                continue

            rel = str(p.relative_to(repo_root)).replace("\\", "/")
            results.append(RepoFile(path=rel, abs_path=str(p), text=text))

    logger.info(
        "[loader] %s -> %d files (%d dirs pruned)",
        repo_root,
        len(results),
        skipped_dirs,
    )
    return results
