"""Configuration management for Google API credentials."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for Google API services.

    Supports two auth backends:
      - Gemini API key (``GOOGLE_API_KEY``), the default.
      - Vertex AI (``use_vertex``), selected automatically when Vertex env vars
        are present. Auth is either a short-lived access token
        (``VERTEX_ACCESS_TOKEN``) or Application Default Credentials (ADC).
    """

    google_api_key: str | None = None
    gemini_model: str = "models/gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"

    # Vertex AI backend (optional; auto-detected from env).
    use_vertex: bool = False
    vertex_access_token: str | None = None
    vertex_project: str | None = None
    vertex_location: str = "us-central1"

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables.

        Gemini API key mode (default):
            GOOGLE_API_KEY: Required. Your Google Gemini API key.
            GEMINI_MODEL: Optional. Chat model (default: models/gemini-2.5-flash).
            EMBEDDING_MODEL: Optional. Embedding model (default: gemini-embedding-001).

        Vertex AI mode (auto-selected when VERTEX_ACCESS_TOKEN or
        GOOGLE_CLOUD_PROJECT is set):
            GOOGLE_CLOUD_PROJECT: Required. GCP project ID.
            GOOGLE_CLOUD_LOCATION: Optional. Region (default: us-central1).
            VERTEX_ACCESS_TOKEN: Optional. Short-lived OAuth token; if unset,
                Application Default Credentials (ADC) are used.
            VERTEX_MODEL: Optional. Chat model (default: gemini-2.5-flash).
            EMBEDDING_MODEL: Optional. Embedding model (default: gemini-embedding-001).
        """
        # Try to load from .env file if it exists
        env_path = Path(".env")
        if env_path.exists():
            _load_dotenv(env_path)

        embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

        vertex_token = os.getenv("VERTEX_ACCESS_TOKEN")
        vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT")
        use_vertex = bool(vertex_token or vertex_project)

        if use_vertex:
            if not vertex_project:
                raise ValueError(
                    "Vertex AI mode requires a project. Set GOOGLE_CLOUD_PROJECT "
                    "in your .env (alongside VERTEX_ACCESS_TOKEN or ADC)."
                )
            return cls(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                gemini_model=os.getenv("VERTEX_MODEL", "gemini-2.5-flash"),
                embedding_model=embedding_model,
                use_vertex=True,
                vertex_access_token=vertex_token,
                vertex_project=vertex_project,
                vertex_location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required.\n"
                "Set it via:\n"
                "  - Environment variable: $env:GOOGLE_API_KEY = 'your_key'\n"
                "  - Or create a .env file with: GOOGLE_API_KEY=your_key\n"
                "  - Or use Vertex AI: set GOOGLE_CLOUD_PROJECT (+ VERTEX_ACCESS_TOKEN or ADC)"
            )

        return cls(
            google_api_key=api_key,
            gemini_model=os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"),
            embedding_model=embedding_model,
        )

    def validate(self) -> None:
        """Validate the configuration."""
        if self.use_vertex:
            if not self.vertex_project:
                raise ValueError("Vertex AI project (GOOGLE_CLOUD_PROJECT) is required")
        elif not self.google_api_key:
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
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def make_genai_client(config: Config):
    """Build a google-genai client for the configured backend.

    Vertex AI mode uses ``VERTEX_ACCESS_TOKEN`` if present, otherwise falls back
    to Application Default Credentials (ADC). Gemini mode uses the API key.
    """
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(
            "google-genai is required. Install with `pip install google-genai`."
        ) from e

    if config.use_vertex:
        credentials = None
        if config.vertex_access_token:
            from google.oauth2.credentials import Credentials

            # Wrap the short-lived token; None lets the SDK use ADC instead.
            credentials = Credentials(token=config.vertex_access_token)
        return genai.Client(
            vertexai=True,
            project=config.vertex_project,
            location=config.vertex_location,
            credentials=credentials,
        )

    return genai.Client(api_key=config.google_api_key)
