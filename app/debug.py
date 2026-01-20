from pathlib import Path
from datetime import datetime
import json


DEBUG_DIR = Path("debug_logs")
DEBUG_DIR.mkdir(exist_ok=True)


def log_text(name: str, text: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DEBUG_DIR / f"{ts}_{name}.txt"
    path.write_text(text, encoding="utf-8")


def log_json(name: str, data):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DEBUG_DIR / f"{ts}_{name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def log_kv(name: str, **kwargs):
    lines = [f"{k}: {v}" for k, v in kwargs.items()]
    log_text(name, "\n".join(lines))
