"""AST-based symbol extraction using tree-sitter.

Extracts functions, classes, methods, imports from Python/JS/TS files
and populates a KnowledgeGraph with symbol nodes.
"""
from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Set, Tuple

from ..loader import RepoFile
from .types import GraphNode, GraphRelationship, NodeProperties
from .graph import KnowledgeGraph

logger = logging.getLogger("rai.repo_map.parsing")

# ── Types ─────────────────────────────────────────────────────

# (file_path, name) -> node_id
SymbolTable = Dict[Tuple[str, str], str]

# file_path -> set of imported file paths
ImportMap = Dict[str, Set[str]]

# ── Language detection ────────────────────────────────────────

LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}


def _detect_language(file_path: str) -> Optional[str]:
    suffix = PurePosixPath(file_path).suffix.lower()
    return LANG_MAP.get(suffix)


# ── tree-sitter initialization ────────────────────────────────

_parsers: Dict[str, object] = {}


def _get_parser(language: str):
    """Lazy-load tree-sitter parser for a language."""
    if language in _parsers:
        return _parsers[language]

    try:
        from tree_sitter import Language, Parser
    except ImportError:
        logger.warning("tree-sitter not installed; AST parsing disabled")
        return None

    try:
        if language == "python":
            import tree_sitter_python as grammar
            lang = Language(grammar.language())
        elif language == "javascript":
            import tree_sitter_javascript as grammar
            lang = Language(grammar.language())
        elif language == "typescript":
            import tree_sitter_typescript
            lang = Language(tree_sitter_typescript.language_typescript())
        else:
            return None

        parser = Parser(lang)
        _parsers[language] = parser
        return parser
    except Exception as e:
        logger.warning(f"Failed to load tree-sitter for {language}: {e}")
        return None


# ── Node ID helpers ───────────────────────────────────────────

def _node_id(label: str, file_path: str, name: str) -> str:
    return f"{label}:{file_path}:{name}"


def _rel_id(rel_type: str, source: str, target: str) -> str:
    return f"{rel_type}:{source}->{target}"


# ── Python extraction ─────────────────────────────────────────

def _walk_python_node(
    node, file_path: str, graph: KnowledgeGraph,
    symbol_table: SymbolTable, parent_class: Optional[str],
) -> None:
    for child in node.children:
        if child.type == "class_definition":
            name_node = child.child_by_field_name("name")
            if not name_node:
                continue
            name = name_node.text.decode("utf-8")
            nid = _node_id("Class", file_path, name)
            is_exported = not name.startswith("_")

            graph.add_node(GraphNode(
                id=nid, label="Class",
                properties=NodeProperties(
                    name=name, file_path=file_path,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    language="python", is_exported=is_exported,
                ),
            ))
            symbol_table[(file_path, name)] = nid

            # Record base classes for heritage processing
            superclasses = child.child_by_field_name("superclasses")
            if superclasses:
                for arg in superclasses.children:
                    if arg.type == "identifier":
                        base_name = arg.text.decode("utf-8")
                        symbol_table[(file_path, f"__extends__{name}__{base_name}")] = nid

            body = child.child_by_field_name("body")
            if body:
                _walk_python_node(body, file_path, graph, symbol_table, parent_class=name)

        elif child.type == "function_definition":
            name_node = child.child_by_field_name("name")
            if not name_node:
                continue
            raw_name = name_node.text.decode("utf-8")

            if parent_class:
                label = "Method"
                qualified = f"{parent_class}.{raw_name}"
                nid = _node_id("Method", file_path, qualified)
            else:
                label = "Function"
                qualified = raw_name
                nid = _node_id("Function", file_path, raw_name)

            is_exported = not raw_name.startswith("_")
            graph.add_node(GraphNode(
                id=nid, label=label,
                properties=NodeProperties(
                    name=qualified, file_path=file_path,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    language="python", is_exported=is_exported,
                ),
            ))
            symbol_table[(file_path, qualified)] = nid
            if parent_class:
                symbol_table[(file_path, raw_name)] = nid  # short name lookup

        elif child.type == "decorated_definition":
            _walk_python_node(child, file_path, graph, symbol_table, parent_class)

        elif child.child_count > 0:
            _walk_python_node(child, file_path, graph, symbol_table, parent_class)


# ── JS/TS extraction ──────────────────────────────────────────

def _walk_js_node(
    node, file_path: str, language: str,
    graph: KnowledgeGraph, symbol_table: SymbolTable,
    parent_class: Optional[str], is_exported: bool,
) -> None:
    for child in node.children:
        if child.type == "export_statement":
            _walk_js_node(child, file_path, language, graph, symbol_table, parent_class, is_exported=True)
            continue

        if child.type == "class_declaration":
            name_node = child.child_by_field_name("name")
            if not name_node:
                continue
            name = name_node.text.decode("utf-8")
            nid = _node_id("Class", file_path, name)
            graph.add_node(GraphNode(
                id=nid, label="Class",
                properties=NodeProperties(
                    name=name, file_path=file_path,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    language=language, is_exported=is_exported,
                ),
            ))
            symbol_table[(file_path, name)] = nid

            # Heritage
            for hc in child.children:
                if hc.type == "class_heritage":
                    for clause in hc.children:
                        if clause.type == "extends_clause":
                            for vc in clause.children:
                                if vc.type == "identifier":
                                    base = vc.text.decode("utf-8")
                                    symbol_table[(file_path, f"__extends__{name}__{base}")] = nid

            body = child.child_by_field_name("body")
            if body:
                _walk_js_node(body, file_path, language, graph, symbol_table, parent_class=name, is_exported=False)
            continue

        if child.type == "interface_declaration" and language == "typescript":
            name_node = child.child_by_field_name("name")
            if not name_node:
                continue
            name = name_node.text.decode("utf-8")
            nid = _node_id("Interface", file_path, name)
            graph.add_node(GraphNode(
                id=nid, label="Interface",
                properties=NodeProperties(
                    name=name, file_path=file_path,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    language=language, is_exported=is_exported,
                ),
            ))
            symbol_table[(file_path, name)] = nid
            continue

        if child.type == "function_declaration":
            name_node = child.child_by_field_name("name")
            if not name_node:
                continue
            name = name_node.text.decode("utf-8")
            nid = _node_id("Function", file_path, name)
            graph.add_node(GraphNode(
                id=nid, label="Function",
                properties=NodeProperties(
                    name=name, file_path=file_path,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    language=language, is_exported=is_exported,
                ),
            ))
            symbol_table[(file_path, name)] = nid
            continue

        if child.type == "method_definition" and parent_class:
            name_node = child.child_by_field_name("name")
            if not name_node:
                continue
            raw_name = name_node.text.decode("utf-8")
            qualified = f"{parent_class}.{raw_name}"
            nid = _node_id("Method", file_path, qualified)
            graph.add_node(GraphNode(
                id=nid, label="Method",
                properties=NodeProperties(
                    name=qualified, file_path=file_path,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    language=language, is_exported=False,
                ),
            ))
            symbol_table[(file_path, qualified)] = nid
            symbol_table[(file_path, raw_name)] = nid
            continue

        # Arrow/function expressions: const foo = () => {}
        if child.type == "lexical_declaration":
            for decl in child.children:
                if decl.type == "variable_declarator":
                    name_node = decl.child_by_field_name("name")
                    value_node = decl.child_by_field_name("value")
                    if (name_node and value_node
                            and value_node.type in ("arrow_function", "function_expression", "function")):
                        name = name_node.text.decode("utf-8")
                        nid = _node_id("Function", file_path, name)
                        graph.add_node(GraphNode(
                            id=nid, label="Function",
                            properties=NodeProperties(
                                name=name, file_path=file_path,
                                start_line=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                language=language, is_exported=is_exported,
                            ),
                        ))
                        symbol_table[(file_path, name)] = nid
            continue

        if (child.child_count > 0
                and child.type not in ("string", "template_string", "comment", "template_literal")):
            _walk_js_node(child, file_path, language, graph, symbol_table, parent_class, is_exported=False)


# ── Structure nodes ───────────────────────────────────────────

def _add_structure_nodes(graph: KnowledgeGraph, files: List[RepoFile]) -> None:
    """Add File/Folder nodes and CONTAINS relationships."""
    seen_folders: Set[str] = set()
    rel_counter = 0

    for f in files:
        lang = _detect_language(f.path)
        if not lang:
            continue

        file_id = f"File:{f.path}"
        graph.add_node(GraphNode(
            id=file_id, label="File",
            properties=NodeProperties(name=PurePosixPath(f.path).name, file_path=f.path, language=lang),
        ))

        parts = list(PurePosixPath(f.path).parts[:-1])
        for i in range(len(parts)):
            folder_path = "/".join(parts[: i + 1])
            folder_id = f"Folder:{folder_path}"
            if folder_path not in seen_folders:
                seen_folders.add(folder_path)
                graph.add_node(GraphNode(
                    id=folder_id, label="Folder",
                    properties=NodeProperties(name=parts[i], file_path=folder_path),
                ))
                if i > 0:
                    parent_id = f"Folder:{'/'.join(parts[:i])}"
                    rel_counter += 1
                    graph.add_relationship(GraphRelationship(
                        id=f"CONTAINS:ff:{rel_counter}",
                        source_id=parent_id, target_id=folder_id,
                        type="CONTAINS", reason="folder_hierarchy",
                    ))

        if parts:
            parent_folder = f"Folder:{'/'.join(parts)}"
            rel_counter += 1
            graph.add_relationship(GraphRelationship(
                id=f"CONTAINS:fi:{rel_counter}",
                source_id=parent_folder, target_id=file_id,
                type="CONTAINS", reason="file_in_folder",
            ))

        for node in graph.nodes_by_file(f.path):
            if node.label in ("Function", "Class", "Method", "Interface"):
                rel_counter += 1
                graph.add_relationship(GraphRelationship(
                    id=f"CONTAINS:fs:{rel_counter}",
                    source_id=file_id, target_id=node.id,
                    type="CONTAINS", reason="symbol_in_file",
                ))


# ── Public API ────────────────────────────────────────────────

def parse_files(graph: KnowledgeGraph, files: List[RepoFile]) -> SymbolTable:
    """Parse all files, populate graph with symbol nodes, return SymbolTable."""
    symbol_table: SymbolTable = {}
    parsed = skipped = 0

    for f in files:
        lang = _detect_language(f.path)
        if not lang:
            continue
        parser = _get_parser(lang)
        if parser is None:
            skipped += 1
            continue
        try:
            tree = parser.parse(f.text.encode("utf-8"))
            if lang == "python":
                _walk_python_node(tree.root_node, f.path, graph, symbol_table, parent_class=None)
            else:
                _walk_js_node(tree.root_node, f.path, lang, graph, symbol_table, parent_class=None, is_exported=False)
            parsed += 1
        except Exception as e:
            logger.warning(f"Failed to parse {f.path}: {e}")
            skipped += 1

    _add_structure_nodes(graph, files)
    logger.info(f"[parsing] parsed={parsed} skipped={skipped} nodes={graph.node_count}")
    return symbol_table
