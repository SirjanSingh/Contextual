"""KnowledgeGraph container — dict-backed graph with node/relationship CRUD."""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional

from .types import GraphNode, GraphRelationship


class KnowledgeGraph:
    """In-memory knowledge graph backed by dictionaries for O(1) lookup."""

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._relationships: Dict[str, GraphRelationship] = {}
        # Adjacency indexes for fast traversal
        self._outgoing: Dict[str, List[str]] = {}  # node_id -> [rel_ids]
        self._incoming: Dict[str, List[str]] = {}  # node_id -> [rel_ids]

    # ── Node operations ───────────────────────────────────────

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        # Remove associated relationships
        for rel_id in list(self._outgoing.get(node_id, [])):
            self._remove_relationship(rel_id)
        for rel_id in list(self._incoming.get(node_id, [])):
            self._remove_relationship(rel_id)
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)
        return True

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def iter_nodes(self) -> Iterator[GraphNode]:
        return iter(self._nodes.values())

    def nodes_by_label(self, label: str) -> List[GraphNode]:
        return [n for n in self._nodes.values() if n.label == label]

    def nodes_by_file(self, file_path: str) -> List[GraphNode]:
        return [n for n in self._nodes.values() if n.properties.file_path == file_path]

    # ── Relationship operations ───────────────────────────────

    def add_relationship(self, rel: GraphRelationship) -> None:
        self._relationships[rel.id] = rel
        self._outgoing.setdefault(rel.source_id, []).append(rel.id)
        self._incoming.setdefault(rel.target_id, []).append(rel.id)

    def get_relationship(self, rel_id: str) -> Optional[GraphRelationship]:
        return self._relationships.get(rel_id)

    def _remove_relationship(self, rel_id: str) -> None:
        rel = self._relationships.pop(rel_id, None)
        if rel:
            out = self._outgoing.get(rel.source_id, [])
            if rel_id in out:
                out.remove(rel_id)
            inc = self._incoming.get(rel.target_id, [])
            if rel_id in inc:
                inc.remove(rel_id)

    @property
    def relationship_count(self) -> int:
        return len(self._relationships)

    def iter_relationships(self) -> Iterator[GraphRelationship]:
        return iter(self._relationships.values())

    def relationships_by_type(self, rel_type: str) -> List[GraphRelationship]:
        return [r for r in self._relationships.values() if r.type == rel_type]

    def outgoing(self, node_id: str, rel_type: Optional[str] = None) -> List[GraphRelationship]:
        rels = [self._relationships[rid] for rid in self._outgoing.get(node_id, [])
                if rid in self._relationships]
        if rel_type:
            rels = [r for r in rels if r.type == rel_type]
        return rels

    def incoming(self, node_id: str, rel_type: Optional[str] = None) -> List[GraphRelationship]:
        rels = [self._relationships[rid] for rid in self._incoming.get(node_id, [])
                if rid in self._relationships]
        if rel_type:
            rels = [r for r in rels if r.type == rel_type]
        return rels

    # ── Neighborhood (for API) ────────────────────────────────

    def neighborhood(self, node_id: str, hops: int = 2) -> "KnowledgeGraph":
        """Return subgraph within N hops of a node."""
        visited = set()
        frontier = {node_id}
        for _ in range(hops):
            next_frontier = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                for rel in self.outgoing(nid):
                    next_frontier.add(rel.target_id)
                for rel in self.incoming(nid):
                    next_frontier.add(rel.source_id)
            frontier = next_frontier - visited
        visited.update(frontier)

        sub = KnowledgeGraph()
        for nid in visited:
            node = self.get_node(nid)
            if node:
                sub.add_node(node)
        for rel in self.iter_relationships():
            if rel.source_id in visited and rel.target_id in visited:
                sub.add_relationship(rel)
        return sub

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "relationships": [r.to_dict() for r in self._relationships.values()],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "KnowledgeGraph":
        g = cls()
        for nd in d.get("nodes", []):
            g.add_node(GraphNode.from_dict(nd))
        for rd in d.get("relationships", []):
            g.add_relationship(GraphRelationship.from_dict(rd))
        return g
