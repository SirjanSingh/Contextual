import json
from datetime import datetime
from pathlib import Path

# Absolute path anchored to repo root (parent of the `app/` package).
# Using __file__ prevents CWD-dependent failures when the server is spawned
# from a different working directory (e.g. by the VS Code extension).
DEBUG_DIR = Path(__file__).parent.parent / "debug_logs"
try:
    DEBUG_DIR.mkdir(exist_ok=True)
except Exception:
    pass


def log_text(name: str, text: str) -> None:
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = DEBUG_DIR / f"{ts}_{name}.txt"
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass  # Debug logging is non-fatal


def log_json(name: str, data) -> None:
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = DEBUG_DIR / f"{ts}_{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass  # Debug logging is non-fatal


def log_kv(name: str, **kwargs) -> None:
    lines = [f"{k}: {v}" for k, v in kwargs.items()]
    log_text(name, "\n".join(lines))
