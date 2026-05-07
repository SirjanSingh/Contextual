"""
Dev server — run the backend standalone for local testing.

Usage:
    python dev_server.py                         # start on port 8360
    python dev_server.py --port 9000             # custom port
    python dev_server.py --repo ./some-project   # auto-index a folder on startup

Then open: http://localhost:8360/dev/  (test dashboard)

Or test with curl:
    curl http://localhost:8360/health
    curl -X POST http://localhost:8360/index/directory -H "Content-Type: application/json" -d '{"repo_path": "G:/projs/repo-aware-ai"}'
    curl http://localhost:8360/index/status
    curl -X POST http://localhost:8360/query -H "Content-Type: application/json" -d '{"question": "how does indexing work?"}'
    curl http://localhost:8360/graph/dependencies
    curl http://localhost:8360/graph/clusters
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Repo-Aware AI Dev Server")
    parser.add_argument("--port", type=int, default=8360, help="Port (default: 8360)")
    parser.add_argument("--repo", type=str, default=None, help="Auto-index this folder on startup")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--data-dir", type=str, default="data/index", help="Index cache directory")
    args = parser.parse_args()

    # Load .env
    env_path = Path(".env")
    if env_path.exists():
        print(f"[dev] Loading .env from {env_path.resolve()}")
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    else:
        print("[dev] WARNING: No .env file found. Set GOOGLE_API_KEY manually.")

    # Set env vars the server expects
    os.environ["RAI_PORT"] = str(args.port)
    os.environ["RAI_DATA_DIR"] = args.data_dir

    if args.repo:
        os.environ["RAI_AUTO_INDEX"] = str(Path(args.repo).resolve())

    # Verify API key
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("[dev] ERROR: GOOGLE_API_KEY not set. Create a .env file with:")
        print("        GOOGLE_API_KEY=your_key_here")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════╗
║         Repo-Aware AI — Dev Server               ║
╠══════════════════════════════════════════════════╣
║  Dashboard : http://{args.host}:{args.port}/dev/             ║
║  Health    : http://{args.host}:{args.port}/health            ║
║  API Key   : {api_key[:8]}...{api_key[-4:]}                  ║
║  Data Dir  : {args.data_dir:<35s}║
║  Reload    : {'ON' if not args.no_reload else 'OFF'}                                ║
╚══════════════════════════════════════════════════╝
""")

    print("[dev] Starting uvicorn...\n")

    import uvicorn
    uvicorn.run(
        "repo_aware_ai.server:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level="info",
        reload_dirs=["repo_aware_ai"] if not args.no_reload else None,
    )


if __name__ == "__main__":
    main()
