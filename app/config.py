"""Configuration management for Google API credentials."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for Google API services."""
    
    google_api_key: str
    gemini_model: str = "models/gemini-2.0-flash"
    embedding_model: str = "gemini-embedding-001"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables.
        
        Environment variables:
            GOOGLE_API_KEY: Required. Your Google API key.
            GEMINI_MODEL: Optional. Model name for chat (default: gemini-2.0-flash-exp).
            EMBEDDING_MODEL: Optional. Model name for embeddings (default: text-embedding-004).
        """
        # Try to load from .env file if it exists
        env_path = Path(".env")
        if env_path.exists():
            _load_dotenv(env_path)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required.\n"
                "Set it via:\n"
                "  - Environment variable: $env:GOOGLE_API_KEY = 'your_key'\n"
                "  - Or create a .env file with: GOOGLE_API_KEY=your_key"
            )
        
        return cls(
            google_api_key=api_key,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-004"),
        )
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.google_api_key:
            raise ValueError("Google API key is required")
        if not self.gemini_model:
            raise ValueError("Gemini model name is required")
        if not self.embedding_model:
            raise ValueError("Embedding model name is required")


def _load_dotenv(path: Path) -> None:
    """Simple .env file loader (no external dependency)."""
    try:
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass  # Ignore errors reading .env


# Singleton instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config
