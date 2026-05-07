"""Data types for the repo map knowledge graph.

Ported from GitNexus graph/types.ts — Python dataclass equivalents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


# ── Node & Relationship type literals ──────────────────────────

NodeLabel = Literal[
    "Function", "Class", "Method", "Interface",
    "File", "Folder", "Community", "Process",
]

RelationshipType = Literal[
    "CALLS", "IMPORTS", "EXTENDS", "IMPLEMENTS",
    "CONTAINS", "HAS_METHOD", "HAS_PROPERTY",
    "MEMBER_OF", "STEP_IN_PROCESS",
]


# ── Graph primitives ──────────────────────────────────────────

@dataclass
class NodeProperties:
    name: str
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    language: str = ""
    is_exported: bool = False
    # Community / Process specific
    heuristic_label: str = ""
    cohesion: float = 0.0
    symbol_count: int = 0
    process_type: str = ""      # "intra_community" | "cross_community"
    step_count: int = 0

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v or k == "name"}

    @classmethod
    def from_dict(cls, d: Dict) -> NodeProperties:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GraphNode:
    id: str
    label: NodeLabel
    properties: NodeProperties

    def to_dict(self) -> Dict:
        return {"id": self.id, "label": self.label, "properties": self.properties.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict) -> GraphNode:
        return cls(
            id=d["id"],
            label=d["label"],
            properties=NodeProperties.from_dict(d.get("properties", {})),
        )


@dataclass
class GraphRelationship:
    id: str
    source_id: str
    target_id: str
    type: RelationshipType
    confidence: float = 1.0
    reason: str = ""
    step: Optional[int] = None  # 1-indexed for STEP_IN_PROCESS

    def to_dict(self) -> Dict:
        d: Dict = {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "confidence": self.confidence,
            "reason": self.reason,
        }
        if self.step is not None:
            d["step"] = self.step
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> GraphRelationship:
        return cls(
            id=d["id"],
            source_id=d["source_id"],
            target_id=d["target_id"],
            type=d["type"],
            confidence=d.get("confidence", 1.0),
            reason=d.get("reason", ""),
            step=d.get("step"),
        )


# ── Community detection results ───────────────────────────────

@dataclass
class CommunityNode:
    id: str
    label: str
    heuristic_label: str
    cohesion: float
    symbol_count: int

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict) -> CommunityNode:
        return cls(**d)


@dataclass
class CommunityMembership:
    node_id: str
    community_id: str

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict) -> CommunityMembership:
        return cls(**d)


@dataclass
class CommunityDetectionResult:
    communities: List[CommunityNode] = field(default_factory=list)
    memberships: List[CommunityMembership] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "communities": [c.to_dict() for c in self.communities],
            "memberships": [m.to_dict() for m in self.memberships],
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> CommunityDetectionResult:
        return cls(
            communities=[CommunityNode.from_dict(c) for c in d.get("communities", [])],
            memberships=[CommunityMembership.from_dict(m) for m in d.get("memberships", [])],
            stats=d.get("stats", {}),
        )


# ── Process detection results ─────────────────────────────────

@dataclass
class ProcessStep:
    process_id: str
    node_id: str
    step: int  # 1-indexed

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict) -> ProcessStep:
        return cls(**d)


@dataclass
class ProcessNode:
    id: str
    label: str
    heuristic_label: str
    process_type: str  # "intra_community" | "cross_community"
    step_count: int
    communities: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    entry_point_id: str = ""
    terminal_id: str = ""

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict) -> ProcessNode:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProcessDetectionResult:
    processes: List[ProcessNode] = field(default_factory=list)
    steps: List[ProcessStep] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "processes": [p.to_dict() for p in self.processes],
            "steps": [s.to_dict() for s in self.steps],
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> ProcessDetectionResult:
        return cls(
            processes=[ProcessNode.from_dict(p) for p in d.get("processes", [])],
            steps=[ProcessStep.from_dict(s) for s in d.get("steps", [])],
            stats=d.get("stats", {}),
        )


# ── Top-level container ───────────────────────────────────────

@dataclass
class RepoMapData:
    nodes: List[GraphNode] = field(default_factory=list)
    relationships: List[GraphRelationship] = field(default_factory=list)
    communities: CommunityDetectionResult = field(default_factory=CommunityDetectionResult)
    processes: ProcessDetectionResult = field(default_factory=ProcessDetectionResult)
    stats: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "relationships": [r.to_dict() for r in self.relationships],
            "communities": self.communities.to_dict(),
            "processes": self.processes.to_dict(),
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> RepoMapData:
        return cls(
            nodes=[GraphNode.from_dict(n) for n in d.get("nodes", [])],
            relationships=[GraphRelationship.from_dict(r) for r in d.get("relationships", [])],
            communities=CommunityDetectionResult.from_dict(d.get("communities", {})),
            processes=ProcessDetectionResult.from_dict(d.get("processes", {})),
            stats=d.get("stats", {}),
        )
