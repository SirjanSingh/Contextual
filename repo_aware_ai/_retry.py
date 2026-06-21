"""Retry helpers for Google Gemini API calls.

The Gemini SDK occasionally fails with transient 5xx errors and 429 rate-limit
errors. We treat both as retryable with exponential backoff. Authentication
errors (4xx aside from 429) bubble immediately so the user sees the real
problem.

If `tenacity` is not installed, the decorator becomes a no-op so the package
still works on a minimal install.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger("rai.retry")

F = TypeVar("F", bound=Callable[..., object])


def _is_retryable(exc: BaseException) -> bool:
    """Heuristic: retry on rate limit + transient server errors only."""
    msg = str(exc).upper()
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "RATE" in msg:
        return True
    if "503" in msg or "504" in msg or "UNAVAILABLE" in msg or "DEADLINE_EXCEEDED" in msg:
        return True
    if "500" in msg and "INTERNAL" in msg:
        return True
    return False


def _no_op(fn: F) -> F:
    return fn


try:
    from tenacity import (
        before_sleep_log,
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential,
    )

    # Defaults are tuned so the cumulative backoff crosses the 60s window that
    # Vertex per-minute token quotas reset on (so a saturated embedding quota
    # clears on its own instead of surfacing as a hard error). Override via env.
    _MAX_ATTEMPTS = int(os.environ.get("RAI_GEMINI_MAX_RETRIES", "6"))
    _MAX_WAIT = int(os.environ.get("RAI_GEMINI_MAX_WAIT", "30"))

    gemini_retry: Callable[[F], F] = retry(  # type: ignore[assignment]
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=_MAX_WAIT),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
except ImportError:  # pragma: no cover
    logger.debug("tenacity not installed; Gemini calls run without retry.")
    gemini_retry = _no_op
