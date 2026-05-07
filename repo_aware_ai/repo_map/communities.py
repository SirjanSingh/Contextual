"""Community detection using the Leiden algorithm via igraph.

Ported from GitNexus community-processor.ts.
Groups symbols into functional modules based on CALLS/EXTENDS edges.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple

from .types import (
    CommunityDetectionResult, CommunityMembership, CommunityNode,
    GraphRelationship,
)
from .graph import KnowledgeGraph

logger = logging.getLogger("rai.repo_map.communities")

# Confidence threshold — filter low-confidence edges for large graphs
MIN_CONFIDENCE_LARGE = 0.5
LARGE_GRAPH_THRESHOLD = 10_000

# Generic folder names to skip for label generation
GENERIC_FOLDERS = {
    "src", "lib", "core", "utils", "util", "helpers", "helper",
    "common", "shared", "internal", "pkg", "app", "main",
    "components", "services", "models", "views", "controllers",
}


def _get_igraph():
    try:
        import igraph
        return igraph
    except ImportError:
        logger.warning("python-igraph not installed; community detection disabled")
        return None


# ── Label heuristics ──────────────────────────────────────────

def _heuristic_label_from_nodes(node_ids: List[str], graph: KnowledgeGraph) -> str:
    """Generate a human-readable label for a community."""
    folder_counts: Counter = Counter()
    for nid in node_ids:
        node = graph.get_node(nid)
        if not node:
            continue
        fp = node.properties.file_path
        if not fp:
            continue
        parts = PurePosixPath(fp).parts
        for part in parts[:-1]:  # skip filename
            if part.lower() not in GENERIC_FOLDERS:
                folder_counts[part] += 1

    if folder_counts:
        return folder_counts.most_common(1)[0][0].title()

    # Fallback: common prefix of node names
    names = []
    for nid in node_ids[:20]:
        node = graph.get_node(nid)
        if node:
            short = node.properties.name.split(".")[-1]
            names.append(short)
    if names:
        # Find common CamelCase prefix
        prefix = names[0]
        for name in names[1:]:
            while prefix and not name.startswith(prefix):
                prefix = prefix[:-1]
        if len(prefix) >= 3:
            return prefix.title()

    return ""


def _calculate_cohesion(node_ids: List[str], graph: KnowledgeGraph) -> float:
    """Internal edge ratio as a cohesion measure (0-1)."""
    id_set = set(node_ids[:50])  # sample up to 50 for performance
    total = internal = 0
    for nid in id_set:
        for rel in graph.outgoing(nid):
            if rel.type in ("CALLS", "EXTENDS", "IMPLEMENTS"):
                total += 1
                if rel.target_id in id_set:
                    internal += 1
    return (internal / total) if total > 0 else 0.0


# ── Main algorithm ────────────────────────────────────────────

def detect_communities(graph: KnowledgeGraph) -> CommunityDetectionResult:
    """Run Leiden community detection on the symbol graph.

    Returns CommunityDetectionResult with communities and memberships.
    """
    ig = _get_igraph()
    if ig is None:
        return CommunityDetectionResult(stats={"error": "igraph not available"})

    # Collect symbol nodes (Function, Class, Method, Interface only)
    symbol_labels = {"Function", "Class", "Method", "Interface"}
    symbol_nodes = [n for n in graph.iter_nodes() if n.label in symbol_labels]

    if len(symbol_nodes) < 2:
        return CommunityDetectionResult(stats={"total_communities": 0, "nodes_processed": len(symbol_nodes)})

    is_large = len(symbol_nodes) > LARGE_GRAPH_THRESHOLD
    min_confidence = MIN_CONFIDENCE_LARGE if is_large else 0.0

    # Build node index
    node_index: Dict[str, int] = {n.id: i for i, n in enumerate(symbol_nodes)}

    # Build edge list from CALLS/EXTENDS/IMPLEMENTS
    edges: List[Tuple[int, int]] = []
    seen_edges = set()
    for rel in graph.iter_relationships():
        if rel.type not in ("CALLS", "EXTENDS", "IMPLEMENTS"):
            continue
        if rel.confidence < min_confidence:
            continue
        src_idx = node_index.get(rel.source_id)
        tgt_idx = node_index.get(rel.target_id)
        if src_idx is None or tgt_idx is None or src_idx == tgt_idx:
            continue
        edge_key = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            edges.append(edge_key)

    if not edges:
        # No clustering possible — put everything in one community
        return _single_community(symbol_nodes, graph)

    # Build igraph Graph
    ig_graph = ig.Graph(n=len(symbol_nodes), edges=edges, directed=False)

    # Filter degree-1 nodes for large graphs
    if is_large:
        degrees = ig_graph.degree()
        keep = [i for i, d in enumerate(degrees) if d > 1]
        if len(keep) < len(symbol_nodes):
            ig_graph = ig_graph.subgraph(keep)
            symbol_nodes = [symbol_nodes[i] for i in keep]
            node_index = {n.id: i for i, n in enumerate(symbol_nodes)}

    resolution = 2.0 if is_large else 1.0

    try:
        membership_result = ig_graph.community_leiden(
            objective_function="modularity",
            resolution=resolution,
            n_iterations=3 if is_large else 2,
        )
        partition = list(membership_result.membership)
        modularity = ig_graph.modularity(partition)
    except Exception as e:
        logger.warning(f"Leiden failed: {e}; falling back to single community")
        return _single_community(symbol_nodes, graph)

    # Group nodes by community
    community_groups: Dict[int, List[str]] = defaultdict(list)
    for node, comm_id in zip(symbol_nodes, partition):
        community_groups[comm_id].append(node.id)

    # Build results
    communities: List[CommunityNode] = []
    memberships: List[CommunityMembership] = []
    comm_id_map: Dict[int, str] = {}

    for raw_id, node_ids in community_groups.items():
        if len(node_ids) < 2:
            continue  # skip singletons

        comm_str_id = f"comm_{raw_id}"
        comm_id_map[raw_id] = comm_str_id

        label = _heuristic_label_from_nodes(node_ids, graph) or f"Cluster_{raw_id}"
        cohesion = _calculate_cohesion(node_ids, graph)

        community = CommunityNode(
            id=comm_str_id,
            label=label,
            heuristic_label=label,
            cohesion=cohesion,
            symbol_count=len(node_ids),
        )
        communities.append(community)

        # Add Community node to graph
        from .types import GraphNode, NodeProperties
        graph.add_node(GraphNode(
            id=f"Community:{comm_str_id}",
            label="Community",
            properties=NodeProperties(
                name=label,
                heuristic_label=label,
                cohesion=cohesion,
                symbol_count=len(node_ids),
            ),
        ))

        for nid in node_ids:
            memberships.append(CommunityMembership(node_id=nid, community_id=comm_str_id))
            # MEMBER_OF edge
            rel_id = f"MEMBER_OF:{nid}:{comm_str_id}"
            if not graph.get_relationship(rel_id):
                graph.add_relationship(GraphRelationship(
                    id=rel_id,
                    source_id=nid,
                    target_id=f"Community:{comm_str_id}",
                    type="MEMBER_OF", confidence=1.0,
                    reason="leiden_community",
                ))

    stats = {
        "total_communities": len(communities),
        "modularity": round(modularity, 4),
        "nodes_processed": len(symbol_nodes),
    }
    logger.info(f"[communities] communities={len(communities)} modularity={modularity:.4f}")

    return CommunityDetectionResult(communities=communities, memberships=memberships, stats=stats)


def _single_community(symbol_nodes, graph: KnowledgeGraph) -> CommunityDetectionResult:
    """Fallback: put all nodes in one community."""
    from .types import GraphNode, NodeProperties
    node_ids = [n.id for n in symbol_nodes]
    comm_str_id = "comm_0"
    label = "All"
    graph.add_node(GraphNode(
        id=f"Community:{comm_str_id}",
        label="Community",
        properties=NodeProperties(name=label, symbol_count=len(node_ids)),
    ))
    memberships = []
    for nid in node_ids:
        memberships.append(CommunityMembership(node_id=nid, community_id=comm_str_id))
        rel_id = f"MEMBER_OF:{nid}:{comm_str_id}"
        if not graph.get_relationship(rel_id):
            graph.add_relationship(GraphRelationship(
                id=rel_id,
                source_id=nid, target_id=f"Community:{comm_str_id}",
                type="MEMBER_OF", confidence=1.0, reason="fallback",
            ))
    community = CommunityNode(
        id=comm_str_id, label=label, heuristic_label=label,
        cohesion=1.0, symbol_count=len(node_ids),
    )
    return CommunityDetectionResult(
        communities=[community], memberships=memberships,
        stats={"total_communities": 1, "nodes_processed": len(node_ids)},
    )
