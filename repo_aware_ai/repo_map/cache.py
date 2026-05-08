"""Serialization and caching for RepoMapData.

Stores repo_map.json alongside the FAISS index in the same cache directory.
Cache is considered valid as long as the FAISS fingerprint matches
(i.e., if the FAISS index is fresh, so is the repo map).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .graph import KnowledgeGraph
from .types import RepoMapData

logger = logging.getLogger("rai.repo_map.cache")

REPO_MAP_FILE = "repo_map.json"


def save_repo_map(data: RepoMapData, graph: KnowledgeGraph, cache_dir: Path) -> None:
    """Serialize graph + RepoMapData to cache_dir/repo_map.json."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / REPO_MAP_FILE

    payload = {
        "graph": graph.to_dict(),
        "communities": data.communities.to_dict(),
        "processes": data.processes.to_dict(),
        "stats": data.stats,
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        logger.info(f"[cache] saved repo map → {path} ({path.stat().st_size // 1024}KB)")
    except Exception as e:
        logger.warning(f"[cache] failed to save repo map: {e}")


def load_repo_map(cache_dir: Path) -> tuple | None:
    """Load (RepoMapData, KnowledgeGraph) from cache. Returns None if missing/corrupt."""
    path = cache_dir / REPO_MAP_FILE
    if not path.exists():
        return None

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        graph = KnowledgeGraph.from_dict(payload.get("graph", {}))

        from .types import CommunityDetectionResult, ProcessDetectionResult

        communities = CommunityDetectionResult.from_dict(payload.get("communities", {}))
        processes = ProcessDetectionResult.from_dict(payload.get("processes", {}))

        data = RepoMapData(
            communities=communities,
            processes=processes,
            stats=payload.get("stats", {}),
        )

        logger.info(
            f"[cache] loaded repo map ← {path} "
            f"(nodes={graph.node_count} rels={graph.relationship_count})"
        )
        return data, graph

    except Exception as e:
        logger.warning(f"[cache] failed to load repo map: {e}")
        return None
