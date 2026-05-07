"""Repo Map — knowledge graph builder for codebase structure analysis.

Public API:
    build_repo_map(repo_files, cache_dir, force_rebuild) -> (RepoMapData, KnowledgeGraph)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

from ..loader import RepoFile
from .types import RepoMapData
from .graph import KnowledgeGraph
from .parsing import parse_files
from .relationships import (
    build_import_relationships,
    build_call_relationships,
    build_heritage_relationships,
    build_structure_relationships,
)
from .communities import detect_communities
from .processes import detect_processes
from .cache import save_repo_map, load_repo_map

logger = logging.getLogger("rai.repo_map")


def build_repo_map(
    repo_files: List[RepoFile],
    cache_dir: Path,
    force_rebuild: bool = False,
) -> Tuple[RepoMapData, KnowledgeGraph]:
    """Build (or load cached) repo map from repo files.

    Returns (RepoMapData, KnowledgeGraph).
    Cache lives in cache_dir/repo_map.json — same directory as FAISS index,
    so cache invalidation is automatically in sync with the FAISS fingerprint.
    """
    if not force_rebuild:
        cached = load_repo_map(cache_dir)
        if cached is not None:
            return cached

    t_start = time.time()
    logger.info(f"[repo_map] building from {len(repo_files)} files …")

    # Stage 1: Parse symbols
    graph = KnowledgeGraph()
    symbol_table = parse_files(graph, repo_files)

    # Stage 2: Build relationships
    import_map = build_import_relationships(graph, repo_files, symbol_table)
    build_call_relationships(graph, repo_files, symbol_table, import_map)
    build_heritage_relationships(graph, repo_files, symbol_table, import_map)
    build_structure_relationships(graph)

    # Stage 3: Community detection
    community_result = detect_communities(graph)

    # Stage 4: Process detection
    process_result = detect_processes(graph, community_result.memberships)

    data = RepoMapData(
        communities=community_result,
        processes=process_result,
        stats={
            "node_count": graph.node_count,
            "relationship_count": graph.relationship_count,
            "file_count": len(repo_files),
            "build_time_s": round(time.time() - t_start, 2),
        },
    )

    save_repo_map(data, graph, cache_dir)
    logger.info(f"[repo_map] built in {time.time()-t_start:.1f}s "
                f"nodes={graph.node_count} rels={graph.relationship_count} "
                f"communities={len(community_result.communities)} "
                f"processes={len(process_result.processes)}")

    return data, graph
