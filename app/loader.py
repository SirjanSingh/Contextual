from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set


DEFAULT_IGNORE_DIRS: Set[str] = {
    ".git", ".hg", ".svn",
    ".idea", ".vscode",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "dist", "build", "out", "coverage",
    ".next", ".nuxt",
    ".venv", "venv", "env",
    ".venv_new",
    "debug_logs", "data"
}

DEFAULT_IGNORE_FILES: Set[str] = {
    ".env", ".env.local", ".DS_Store",
}

DEFAULT_ALLOWED_EXTS: Set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".md", ".json", ".yaml", ".yml", ".toml",
    ".txt", ".ini", ".cfg",
}


@dataclass(frozen=True)
class RepoFile:
    path: str      # relative path from repo root (posix style)
    abs_path: str  # absolute path on disk
    text: str      # file contents


def _safe_read_text(p: Path, max_bytes: int = 2_000_000) -> str:
    """Read text safely; skip very large files; handle encoding issues."""
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
    allowed_exts: Set[str] = DEFAULT_ALLOWED_EXTS,
    ignore_dirs: Set[str] = DEFAULT_IGNORE_DIRS,
    ignore_files: Set[str] = DEFAULT_IGNORE_FILES,
) -> List[RepoFile]:
    repo_root = Path(repo_root).resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise ValueError(f"repo_root is not a folder: {repo_root}")

    results: List[RepoFile] = []

    for root, dirs, files in os.walk(repo_root):
        # prune ignored dirs in-place
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for name in files:
            if name in ignore_files:
                continue
            p = Path(root) / name
            if p.suffix.lower() not in allowed_exts:
                continue

            text = _safe_read_text(p)
            if not text.strip():
                continue

            rel = str(p.relative_to(repo_root)).replace("\\", "/")
            results.append(RepoFile(path=rel, abs_path=str(p), text=text))

    return results
