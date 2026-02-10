"""Embedder using Google GenAI embeddings."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .config import get_config


@dataclass
class Embedder:
    """Embedder using Google's gemini-embedding model.
    
    Uses the new google.genai package with gemini-embedding-001.
    Produces 768-dimensional normalized embeddings.
    """
    
    model_name: str = "gemini-embedding-001"
    _dim: int = field(default=768, repr=False, init=False)
    
    def __post_init__(self) -> None:
        """Initialize the Google GenAI client."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai is required. Install with:\n"
                "  pip install google-genai"
            )
        
        config = get_config()
        self._client = genai.Client(api_key=config.google_api_key)
        self._types = types
        # Use gemini-embedding-001 if not specified in config
        if hasattr(config, 'embedding_model') and config.embedding_model:
            self.model_name = config.embedding_model
    
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Embed a list of texts into normalized vectors.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Maximum number of texts to embed per API call (API limit is 100).
        
        Returns:
            numpy array of shape (N, 768) with float32 normalized embeddings.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dim)
        
        # API has a hard limit of 100 requests per batch
        if batch_size > 100:
            batch_size = 100
        
        all_embeddings = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Embed the batch
            result = self._client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=self._types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768
                )
            )
            
            # Extract embeddings from result
            batch_embeddings = [emb.values for emb in result.embeddings]
            all_embeddings.extend(batch_embeddings)
            
            # Optional: Add a small delay between batches to avoid rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        # Convert to numpy array
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.
        
        Args:
            query: The query string to embed.
        
        Returns:
            numpy array of shape (768,) with float32 normalized embedding.
        """
        result = self._client.models.embed_content(
            model=self.model_name,
            contents=query,
            config=self._types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )
        
        # Extract embedding from response
        embedding = np.array(result.embeddings[0].values, dtype=np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension (768 for gemini-embedding-001)."""
        return self._dim
