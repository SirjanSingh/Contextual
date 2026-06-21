"""Test fixtures shared across the suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """A minimal on-disk repo with a few code files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "alpha.py").write_text(
        "def add(a, b):\n    return a + b\n\nclass Adder:\n    def add(self, a, b):\n        return a + b\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "beta.js").write_text(
        "export function multiply(x, y) {\n  return x * y;\n}\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Tiny repo\n\nFor testing.\n", encoding="utf-8")
    # Make sure ignored dirs really are ignored
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("// should be ignored\n", encoding="utf-8")
    return tmp_path


class FakeEmbedder:
    """Deterministic, API-free embedder for tests.

    Embeds text by hashing the lowercase character histogram into a small fixed
    vector — purely so retrieval has *something* sensible to rank against.
    """

    def __init__(self, dim: int = 32):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for ch in text.lower():
            v[ord(ch) % self._dim] += 1.0
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
        return v

    def embed_texts(self, texts):
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec(query)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
