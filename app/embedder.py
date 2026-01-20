"""Embedder using Google GenAI embeddings."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .config import get_config


@dataclass
class Embedder:
    """Embedder using Google's text-embedding model.
    
    Uses the new google.genai package.
    Produces 768-dimensional normalized embeddings.
    """
    
    model_name: str = "text-embedding-004"
    _client: object = field(default=None, repr=False, init=False)
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
        self.model_name = config.embedding_model
        self._types = types
    
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Embed a list of texts into normalized vectors.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Not used with new API (processes individually to avoid rate limits).
        
        Returns:
            numpy array of shape (N, 768) with float32 normalized embeddings.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dim)
        
        all_embeddings = []
        
        # Process each text individually with small delay to avoid rate limits
        for i, text in enumerate(texts):
            # Add small delay to avoid rate limiting (except for first request)
            if i > 0:
                time.sleep(0.1)  # 100ms delay between requests
            
            result = self._client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            
            # Extract embedding from response
            if hasattr(result, 'embeddings') and len(result.embeddings) > 0:
                embedding = result.embeddings[0].values
                all_embeddings.append(embedding)
            else:
                raise ValueError(f"Unexpected result format: {type(result)}")
        
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
        )
        
        # Extract embedding from response
        if hasattr(result, 'embeddings') and len(result.embeddings) > 0:
            embedding = np.array(result.embeddings[0].values, dtype=np.float32)
        else:
            raise ValueError(f"Unexpected result format: {type(result)}")
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension (768 for text-embedding-004)."""
        return self._dim
