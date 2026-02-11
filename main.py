from __future__ import annotations

import argparse
from pathlib import Path

from app.embedder import Embedder
from app.llm import LLMClient
from app.qa import QAEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repo-Aware AI Assistant (Google API)")
    p.add_argument("--repo", required=True, help="Path to the target repository")
    p.add_argument("--cache", default="data/index", help="Cache dir for FAISS index + metadata")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild the index")
    p.add_argument("--topk", type=int, default=6, help="Top-k chunks to retrieve")
    p.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    p.add_argument("--chunk_size", type=int, default=1800, help="Chunk size (characters)")
    p.add_argument("--overlap", type=int, default=250, help="Chunk overlap (characters)")
    p.add_argument("--no-rerank", action="store_true", help="Disable reranking (faster but less accurate)")
    p.add_argument("--no-hybrid", action="store_true", help="Disable hybrid search (vector-only)")
    p.add_argument("--no-expand", action="store_true", help="Disable query expansion")
    p.add_argument("--no-compress", action="store_true", help="Disable contextual compression")
    p.add_argument("--ast-chunk", action="store_true", help="Enable AST-based chunking for Python files (requires --rebuild)")
    p.add_argument("--no-multi-query", action="store_true", help="Disable multi-query decomposition")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo).resolve()
    cache_base = Path(args.cache).resolve()

    print("[+] Initializing Google AI clients...")
    
    try:
        embedder = Embedder()
        llm = LLMClient(temperature=args.temperature)
    except ValueError as e:
        print(f"\n[!] Configuration Error:\n{e}")
        print("\n[i] Create a .env file with your Google API key:")
        print("    GOOGLE_API_KEY=your_key_here")
        print("\n[i] Get your key from: https://aistudio.google.com/app/apikey")
        return
    except ImportError as e:
        print(f"\n[!] Missing dependency:\n{e}")
        print("\n[i] Install with: pip install -r requirements.txt")
        return

    engine = QAEngine(
        repo_root=repo_root,
        cache_base=cache_base,
        embedder=embedder,
        llm=llm,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.topk,
        use_reranker=not args.no_rerank,  # Invert the flag
        use_hybrid_search=not args.no_hybrid,  # Invert the flag
        use_query_expansion=not args.no_expand,  # Invert the flag
        use_compression=not args.no_compress,  # Invert the flag
        use_ast_chunking=args.ast_chunk,
        use_multi_query=not args.no_multi_query,
    )

    print(f"[+] Repo: {repo_root}")
    print(f"[+] Cache: {cache_base}")
    print(f"[+] Model: {llm.model}")
    print("[+] Building/loading index...")
    engine.build(force_rebuild=args.rebuild)
    if engine.cache_dir:
        print(f"[+] Index ready in: {engine.cache_dir}")

    print("\nType a question, or 'exit' to quit.\n")
    while True:
        try:
            q = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("bye")
            break
        if q.lower() in {"clear", "reset"}:
            engine.clear_conversation()
            print("[+] Conversation history cleared")
            continue

        ans, sources = engine.ask(q)
        print("\nANSWER:\n" + ans.strip())
        print("\nSOURCES:")
        for s in sources:
            print(" - " + s)
        print("")


if __name__ == "__main__":
    main()
