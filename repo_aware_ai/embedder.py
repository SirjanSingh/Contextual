"""Embedder using Google GenAI embeddings."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from ._retry import gemini_retry
from .config import get_config

logger = logging.getLogger("rai.embedder")


# Known output dimensions for supported embedding models.
# google-genai supports `output_dimensionality` to truncate, but we keep
# native dims by default for best quality.
_MODEL_DIMS = {
    "gemini-embedding-001": 768,
    "text-embedding-004": 768,
    "models/embedding-001": 768,
}
_DEFAULT_DIM = 768


@dataclass
class Embedder:
    """Embedder using Google's gemini-embedding model.

    Produces L2-normalized float32 vectors suitable for FAISS inner-product
    search. Default model is gemini-embedding-001 (768 dimensions).
    """

    model_name: str = "gemini-embedding-001"
    output_dim: int | None = None  # None = use native model dim
    _dim: int = field(default=_DEFAULT_DIM, repr=False, init=False)

    def __post_init__(self) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "google-genai is required. Install with `pip install google-genai`."
            ) from e

        config = get_config()
        if getattr(config, "embedding_model", None):
            self.model_name = config.embedding_model

        # Resolve dimension: explicit override > known native dim > default.
        if self.output_dim is not None:
            self._dim = self.output_dim
        else:
            self._dim = _MODEL_DIMS.get(self.model_name.split("/")[-1], _DEFAULT_DIM)

        self._client = genai.Client(api_key=config.google_api_key)
        self._types = types

    @gemini_retry
    def _embed_batch(self, batch: list[str], task_type: str) -> list[list[float]]:
        result = self._client.models.embed_content(
            model=self.model_name,
            contents=batch,
            config=self._types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._dim,
            ),
        )
        return [emb.values for emb in result.embeddings]

    def embed_texts(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Embed many texts. Returns (N, dim) L2-normalized float32 array."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        batch_size = min(batch_size, 100)  # API hard limit
        all_embeddings: list[list[float]] = []

        logger.info(
            "Embedding %d texts in batches of %d (%s)", len(texts), batch_size, self.model_name
        )
        t0 = time.time()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._embed_batch(batch, "RETRIEVAL_DOCUMENT"))
            if i + batch_size < len(texts):
                time.sleep(0.1)  # gentle pacing

        embeddings = np.asarray(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings /= norms

        logger.info("Embedded %d texts in %.2fs", len(texts), time.time() - t0)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (dim,) L2-normalized float32 array."""
        values = self._embed_batch([query], "RETRIEVAL_QUERY")[0]
        embedding = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        return embedding

    @property
    def dimension(self) -> int:
        """Output embedding dimension."""
        return self._dim
