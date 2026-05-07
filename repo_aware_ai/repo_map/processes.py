"""Process (execution flow) detection.

Ported from GitNexus process-processor.ts + entry-point-scoring.ts.
Detects execution flows through the codebase via BFS from scored entry points.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .types import (
    CommunityMembership, GraphRelationship,
    ProcessDetectionResult, ProcessNode, ProcessStep,
)
from .graph import KnowledgeGraph

logger = logging.getLogger("rai.repo_map.processes")


# ── Config ────────────────────────────────────────────────────

@dataclass
class ProcessDetectionConfig:
    max_trace_depth: int = 10
    max_branching: int = 4
    max_processes: int = 75
    min_steps: int = 3
    min_confidence: float = 0.5


# ── Entry point scoring ───────────────────────────────────────

# Patterns that indicate a likely entry point (1.5x multiplier)
_ENTRY_PATTERNS = [
    re.compile(r"^(main|init|bootstrap|start|run|setup|configure|initialize)$"),
    re.compile(r"^handle[A-Z_]"),
    re.compile(r"^on[A-Z_]"),
    re.compile(r"^process[A-Z_]"),
    re.compile(r"^execute[A-Z_]"),
    re.compile(r"^perform[A-Z_]"),
    re.compile(r"^dispatch[A-Z_]"),
    re.compile(r"^trigger[A-Z_]"),
    re.compile(r"^fire[A-Z_]"),
    re.compile(r"^emit[A-Z_]"),
    re.compile(r".*Controller$"),
    re.compile(r".*Handler$"),
    re.compile(r".*Service$"),
    re.compile(r"^(get|post|put|delete|patch)_"),
    re.compile(r"^api_"),
    re.compile(r"^view_"),
    re.compile(r"^use[A-Z]"),  # React hooks
    re.compile(r"^app$"),
]

# Patterns that indicate a utility (0.3x penalty)
_UTILITY_PATTERNS = [
    re.compile(r"^(get|set|is|has|can|should|will|did)[A-Z_]"),
    re.compile(r"^_"),
    re.compile(r"^(format|parse|validate|convert|transform)"),
    re.compile(r"^(log|debug|error|warn|info)$"),
    re.compile(r"^(to|from)[A-Z]"),
    re.compile(r"^(encode|decode)"),
    re.compile(r"^(serialize|deserialize)"),
    re.compile(r"^(clone|copy|deep)"),
    re.compile(r"^(merge|extend|assign)"),
    re.compile(r"^(filter|map|reduce|sort|find)$"),
    re.compile(r"(Helper|Util|Utils)$"),
    re.compile(r"^(util|helper)s?$"),
]

# Test file patterns
_TEST_FILE_PATTERNS = [
    ".test.", ".spec.", "__tests__/", "__mocks__/",
    "/test/", "/tests/", "/testing/",
    "_test.py", "/test_", "_test.go", "/src/test/",
]


def _is_test_file(file_path: str) -> bool:
    p = file_path.lower().replace("\\", "/")
    return any(pat in p for pat in _TEST_FILE_PATTERNS)


def _score_entry_point(
    name: str, is_exported: bool, callee_count: int, caller_count: int,
) -> float:
    """Score a symbol as a potential process entry point."""
    if callee_count == 0:
        return 0.0

    # Base: calls many but is called by few
    base = callee_count / (caller_count + 1)

    # Export multiplier
    export_mult = 2.0 if is_exported else 1.0

    # Name multiplier
    short_name = name.split(".")[-1]  # strip class prefix
    name_mult = 1.0
    if any(p.search(short_name) for p in _UTILITY_PATTERNS):
        name_mult = 0.3
    elif any(p.match(short_name) for p in _ENTRY_PATTERNS):
        name_mult = 1.5

    return base * export_mult * name_mult


# ── BFS trace ─────────────────────────────────────────────────

def _bfs_trace(
    entry_id: str,
    graph: KnowledgeGraph,
    config: ProcessDetectionConfig,
) -> List[str]:
    """BFS from entry_id through CALLS edges. Returns ordered node list."""
    visited: Set[str] = set()
    path: List[str] = [entry_id]
    visited.add(entry_id)
    queue: deque = deque([(entry_id, 0)])

    while queue:
        current_id, depth = queue.popleft()
        if depth >= config.max_trace_depth:
            continue
        outgoing = [
            r for r in graph.outgoing(current_id, rel_type="CALLS")
            if r.confidence >= config.min_confidence
        ]
        outgoing.sort(key=lambda r: -r.confidence)
        outgoing = outgoing[: config.max_branching]

        for rel in outgoing:
            nid = rel.target_id
            if nid not in visited:
                visited.add(nid)
                path.append(nid)
                queue.append((nid, depth + 1))

    return path


# ── Deduplication ─────────────────────────────────────────────

def _deduplicate_traces(traces: List[List[str]]) -> List[List[str]]:
    """Remove subset traces and keep longest per (entry, terminal) pair."""
    # Sort longest first
    traces = sorted(traces, key=len, reverse=True)

    # Subset removal: remove trace A if all its nodes appear in longer trace B
    kept: List[List[str]] = []
    for trace in traces:
        trace_set = set(trace)
        is_subset = any(
            trace_set.issubset(set(other)) and trace != other
            for other in kept
        )
        if not is_subset:
            kept.append(trace)

    # Per entry→terminal dedup: keep longest
    seen_endpoints: Dict[Tuple[str, str], int] = {}
    final: List[List[str]] = []
    for trace in kept:
        key = (trace[0], trace[-1])
        if key not in seen_endpoints:
            seen_endpoints[key] = len(trace)
            final.append(trace)

    return final


# ── Label generation ──────────────────────────────────────────

def _trace_label(trace: List[str], graph: KnowledgeGraph) -> str:
    entry = graph.get_node(trace[0])
    terminal = graph.get_node(trace[-1])
    entry_name = entry.properties.name.split(".")[-1] if entry else trace[0]
    term_name = terminal.properties.name.split(".")[-1] if terminal else trace[-1]
    return f"{entry_name} -> {term_name}"


# ── Main algorithm ────────────────────────────────────────────

def detect_processes(
    graph: KnowledgeGraph,
    memberships: List[CommunityMembership],
    config: Optional[ProcessDetectionConfig] = None,
) -> ProcessDetectionResult:
    """Detect execution processes via entry-point scoring + BFS tracing."""
    if config is None:
        config = ProcessDetectionConfig()

    # Build membership lookup: node_id -> community_id
    membership_map: Dict[str, str] = {m.node_id: m.community_id for m in memberships}

    # Count callers and callees per symbol
    callee_count: Dict[str, int] = defaultdict(int)
    caller_count: Dict[str, int] = defaultdict(int)
    for rel in graph.iter_relationships():
        if rel.type == "CALLS" and rel.confidence >= config.min_confidence:
            callee_count[rel.source_id] += 1
            caller_count[rel.target_id] += 1

    # Score all symbols and select top candidates
    candidate_labels = {"Function", "Method"}
    scored: List[Tuple[float, str]] = []
    for node in graph.iter_nodes():
        if node.label not in candidate_labels:
            continue
        if _is_test_file(node.properties.file_path):
            continue
        score = _score_entry_point(
            name=node.properties.name,
            is_exported=node.properties.is_exported,
            callee_count=callee_count[node.id],
            caller_count=caller_count[node.id],
        )
        if score > 0:
            scored.append((score, node.id))

    scored.sort(reverse=True)
    candidates = [nid for _, nid in scored[:200]]

    # BFS from each candidate
    raw_traces: List[List[str]] = []
    for entry_id in candidates:
        trace = _bfs_trace(entry_id, graph, config)
        if len(trace) >= config.min_steps:
            raw_traces.append(trace)

    # Deduplicate
    traces = _deduplicate_traces(raw_traces)
    traces = traces[: config.max_processes]

    # Build ProcessNode objects
    processes: List[ProcessNode] = []
    steps: List[ProcessStep] = []

    for idx, trace in enumerate(traces):
        entry_id = trace[0]
        terminal_id = trace[-1]
        label = _trace_label(trace, graph)
        proc_id = f"proc_{idx}_{entry_id.split(':')[-1].lower()[:20]}"

        # Determine community involvement
        communities_touched = list({
            membership_map[nid]
            for nid in trace
            if nid in membership_map
        })
        process_type = (
            "cross_community" if len(communities_touched) > 1 else "intra_community"
        )

        proc = ProcessNode(
            id=proc_id,
            label=label,
            heuristic_label=label,
            process_type=process_type,
            step_count=len(trace),
            communities=communities_touched,
            trace=trace,
            entry_point_id=entry_id,
            terminal_id=terminal_id,
        )
        processes.append(proc)

        # Add Process node to graph
        from .types import GraphNode, NodeProperties
        graph.add_node(GraphNode(
            id=f"Process:{proc_id}",
            label="Process",
            properties=NodeProperties(
                name=label,
                heuristic_label=label,
                process_type=process_type,
                step_count=len(trace),
            ),
        ))

        # STEP_IN_PROCESS relationships
        for step_num, nid in enumerate(trace, start=1):
            step_rel_id = f"STEP_IN_PROCESS:{proc_id}:{step_num}"
            graph.add_relationship(GraphRelationship(
                id=step_rel_id,
                source_id=nid,
                target_id=f"Process:{proc_id}",
                type="STEP_IN_PROCESS", confidence=1.0,
                reason="process_step", step=step_num,
            ))
            steps.append(ProcessStep(process_id=proc_id, node_id=nid, step=step_num))

    cross_community = sum(1 for p in processes if p.process_type == "cross_community")
    avg_steps = sum(p.step_count for p in processes) / max(len(processes), 1)

    stats = {
        "total_processes": len(processes),
        "cross_community_count": cross_community,
        "avg_step_count": round(avg_steps, 2),
        "entry_points_found": len(candidates),
    }
    logger.info(f"[processes] processes={len(processes)} cross_community={cross_community}")

    return ProcessDetectionResult(processes=processes, steps=steps, stats=stats)
