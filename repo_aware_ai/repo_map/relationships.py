"""Relationship building for the repo map knowledge graph.

Builds IMPORTS, CALLS, EXTENDS, HAS_METHOD relationships between symbols.
Ported from GitNexus import-processor.ts, call-processor.ts,
heritage-processor.ts, structure-processor.ts.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from ..loader import RepoFile
from .graph import KnowledgeGraph
from .parsing import ImportMap, SymbolTable, _detect_language, _get_parser, _rel_id
from .types import GraphRelationship

logger = logging.getLogger("rai.repo_map.relationships")


# ── Import resolution helpers ─────────────────────────────────


def _build_file_set(files: list[RepoFile]) -> set[str]:
    return {f.path for f in files}


def _try_extensions(base: str, file_set: set[str]) -> str | None:
    """Try resolving a base path with common extensions."""
    for ext in (
        ".py",
        "/__init__.py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        "/index.ts",
        "/index.tsx",
        "/index.js",
        "/index.jsx",
    ):
        candidate = base + ext
        if candidate in file_set:
            return candidate
    if base in file_set:
        return base
    return None


def _resolve_python_import(import_path: str, current_file: str, file_set: set[str]) -> str | None:
    """Resolve a Python import path to a file in the repo."""
    # Relative import (PEP 328)
    if import_path.startswith("."):
        dot_count = len(import_path) - len(import_path.lstrip("."))
        module_part = import_path[dot_count:]
        parts = current_file.replace("\\", "/").split("/")[:-1]
        if dot_count - 1 > len(parts):
            return None
        for _ in range(1, dot_count):
            if parts:
                parts.pop()
        if module_part:
            parts.extend(module_part.replace(".", "/").split("/"))
        base = "/".join(parts)
        return _try_extensions(base, file_set)

    # Single segment — check same directory first (proximity)
    if "." not in import_path:
        importer_dir = "/".join(current_file.replace("\\", "/").split("/")[:-1])
        if importer_dir:
            candidate = _try_extensions(f"{importer_dir}/{import_path}", file_set)
            if candidate:
                return candidate

    # Fallback: suffix matching across the whole file set
    parts = import_path.replace(".", "/").split("/")
    suffix = "/".join(parts)
    for fp in file_set:
        fp_norm = fp.replace("\\", "/")
        if fp_norm.endswith(suffix + ".py") or fp_norm.endswith(suffix + "/__init__.py"):
            return fp
    return None


def _resolve_js_import(import_path: str, current_file: str, file_set: set[str]) -> str | None:
    """Resolve a JS/TS import path to a file in the repo."""
    if not import_path.startswith("."):
        # Bare import — try suffix matching
        parts = [p for p in import_path.split("/") if p]
        for fp in file_set:
            fp_norm = fp.replace("\\", "/")
            for ext in (".ts", ".tsx", ".js", ".jsx"):
                if fp_norm.endswith("/".join(parts) + ext):
                    return fp
            if fp_norm.endswith("/".join(parts) + "/index.ts") or fp_norm.endswith(
                "/".join(parts) + "/index.js"
            ):
                return fp
        return None

    # Relative import
    current_dir = current_file.replace("\\", "/").split("/")[:-1]
    for part in import_path.split("/"):
        if part == "." or part == "":
            continue
        elif part == "..":
            if current_dir:
                current_dir.pop()
        else:
            current_dir.append(part)
    base = "/".join(current_dir)
    return _try_extensions(base, file_set)


# ── 1. Import relationships ───────────────────────────────────


def _extract_python_imports(tree, file_path: str) -> list[str]:
    """Extract import paths from a Python AST."""
    imports = []

    def walk(node):
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    imports.append(child.text.decode("utf-8"))
                elif child.type == "aliased_import":
                    for c in child.children:
                        if c.type == "dotted_name":
                            imports.append(c.text.decode("utf-8"))
                            break
        elif node.type == "import_from_statement":
            mod = node.child_by_field_name("module_name")
            if mod:
                imports.append(mod.text.decode("utf-8"))
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return imports


def _extract_js_imports(tree, file_path: str) -> list[str]:
    """Extract import sources from a JS/TS AST."""
    imports = []

    def walk(node):
        if node.type == "import_statement":
            src = node.child_by_field_name("source")
            if src:
                raw = src.text.decode("utf-8").strip("'\"`")
                imports.append(raw)
        elif node.type == "export_statement":
            src = node.child_by_field_name("source")
            if src:
                raw = src.text.decode("utf-8").strip("'\"`")
                imports.append(raw)
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return imports


def build_import_relationships(
    graph: KnowledgeGraph,
    files: list[RepoFile],
    symbol_table: SymbolTable,
) -> ImportMap:
    """Parse imports and create IMPORTS edges. Returns ImportMap."""
    file_set = _build_file_set(files)
    import_map: ImportMap = defaultdict(set)
    rel_counter = 0

    for f in files:
        lang = _detect_language(f.path)
        if not lang:
            continue
        parser = _get_parser(lang)
        if not parser:
            continue
        try:
            tree = parser.parse(f.text.encode("utf-8"))
            if lang == "python":
                raw_imports = _extract_python_imports(tree, f.path)
                resolver = _resolve_python_import
            else:
                raw_imports = _extract_js_imports(tree, f.path)
                resolver = _resolve_js_import
        except Exception:
            continue

        for imp in raw_imports:
            resolved = resolver(imp, f.path, file_set)
            if not resolved or resolved == f.path:
                continue
            import_map[f.path].add(resolved)

            # File-level IMPORTS edge
            src_id = f"File:{f.path}"
            tgt_id = f"File:{resolved}"
            if graph.has_node(src_id) and graph.has_node(tgt_id):
                rel_id = _rel_id("IMPORTS", src_id, tgt_id)
                if not graph.get_relationship(rel_id):
                    rel_counter += 1
                    graph.add_relationship(
                        GraphRelationship(
                            id=rel_id,
                            source_id=src_id,
                            target_id=tgt_id,
                            type="IMPORTS",
                            confidence=1.0,
                            reason="file_import",
                        )
                    )

    logger.info(f"[relationships] import edges={rel_counter}")
    return dict(import_map)


# ── 2. Call relationships ─────────────────────────────────────


def _collect_calls(node, calls: list[str]) -> None:
    """Collect all call targets from an AST node."""
    if node.type == "call":
        # Python: call.function
        fn = node.child_by_field_name("function")
        if fn:
            if fn.type == "identifier":
                calls.append(fn.text.decode("utf-8"))
            elif fn.type == "attribute":
                attr = fn.child_by_field_name("attribute")
                if attr:
                    calls.append(attr.text.decode("utf-8"))
    elif node.type == "call_expression":
        # JS/TS
        fn = node.child_by_field_name("function")
        if fn:
            if fn.type == "identifier":
                calls.append(fn.text.decode("utf-8"))
            elif fn.type in ("member_expression", "call_expression"):
                prop = fn.child_by_field_name("property")
                if prop:
                    calls.append(prop.text.decode("utf-8"))
    elif node.type == "new_expression":
        ctor = node.child_by_field_name("constructor")
        if ctor and ctor.type == "identifier":
            calls.append(ctor.text.decode("utf-8"))

    for child in node.children:
        _collect_calls(child, calls)


def _resolve_call(
    call_name: str,
    from_file: str,
    symbol_table: SymbolTable,
    import_map: ImportMap,
) -> tuple[str | None, float]:
    """Resolve a function call name to a node_id with confidence score.

    Returns (node_id, confidence) or (None, 0).
    Tier 1: same file   → confidence 0.95
    Tier 2: imported    → confidence 0.90
    Tier 3: global      → confidence 0.50
    """
    # Tier 1: same file
    nid = symbol_table.get((from_file, call_name))
    if nid:
        return nid, 0.95

    # Tier 2: imported files
    imported = import_map.get(from_file, set())
    candidates = []
    for (fp, name), nid in symbol_table.items():
        if name == call_name:
            if fp in imported:
                return nid, 0.90
            candidates.append(nid)

    # Tier 3: global (single unambiguous match only)
    if len(candidates) == 1:
        return candidates[0], 0.50

    return None, 0.0


def build_call_relationships(
    graph: KnowledgeGraph,
    files: list[RepoFile],
    symbol_table: SymbolTable,
    import_map: ImportMap,
) -> None:
    """Build CALLS relationships between callables."""
    rel_counter = 0

    for f in files:
        lang = _detect_language(f.path)
        if not lang:
            continue
        parser = _get_parser(lang)
        if not parser:
            continue

        # Collect per-function calls by scanning function body AST nodes
        try:
            tree = parser.parse(f.text.encode("utf-8"))
        except Exception:
            continue

        # Find all callable symbols in this file
        file_symbols = {name: nid for (fp, name), nid in symbol_table.items() if fp == f.path}

        if not file_symbols:
            continue

        # For each symbol, collect calls within its subtree
        # We do a single full-file pass and attribute calls to the enclosing function
        _extract_file_calls(
            tree.root_node,
            f.path,
            lang,
            graph,
            symbol_table,
            import_map,
            rel_counter,
        )
        # rel_counter is updated inside, but Python ints are immutable; use list
    # Re-run with mutable counter
    _rel_count = [0]
    for f in files:
        lang = _detect_language(f.path)
        if not lang:
            continue
        parser = _get_parser(lang)
        if not parser:
            continue
        try:
            tree = parser.parse(f.text.encode("utf-8"))
            _extract_file_calls_v2(
                tree.root_node, f.path, lang, graph, symbol_table, import_map, _rel_count
            )
        except Exception:
            continue

    logger.info(f"[relationships] call edges={_rel_count[0]}")


def _extract_file_calls_v2(
    root,
    file_path: str,
    language: str,
    graph: KnowledgeGraph,
    symbol_table: SymbolTable,
    import_map: ImportMap,
    counter: list[int],
) -> None:
    """Walk AST, find enclosing function context, extract and record calls."""

    def get_current_symbol(node) -> str | None:
        """Find the nearest enclosing function/method symbol for a node."""
        # Walk up via parent chain isn't available; use name lookups
        return None  # handled in the recursive walk below

    def walk(node, enclosing_nid: str | None):
        # Detect enclosing scope changes
        if node.type in ("function_definition",):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = name_node.text.decode("utf-8")
                nid = symbol_table.get((file_path, name))
                if not nid:
                    # Try method lookup
                    for (fp, n), v in symbol_table.items():
                        if fp == file_path and (n == name or n.endswith(f".{name}")):
                            nid = v
                            break
                if nid:
                    enclosing_nid = nid
        elif node.type in ("function_declaration", "method_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = name_node.text.decode("utf-8")
                nid = symbol_table.get((file_path, name))
                if not nid:
                    for (fp, n), v in symbol_table.items():
                        if fp == file_path and (n == name or n.endswith(f".{name}")):
                            nid = v
                            break
                if nid:
                    enclosing_nid = nid

        if enclosing_nid:
            # Detect calls at this level
            calls: list[str] = []
            _collect_calls_shallow(node, calls)
            for call_name in calls:
                if call_name == symbol_table.get((file_path, "__name__"), ""):
                    continue
                target_nid, confidence = _resolve_call(
                    call_name, file_path, symbol_table, import_map
                )
                if target_nid and target_nid != enclosing_nid and confidence >= 0.5:
                    rel_id = _rel_id("CALLS", enclosing_nid, target_nid)
                    if not graph.get_relationship(rel_id):
                        graph.add_relationship(
                            GraphRelationship(
                                id=rel_id,
                                source_id=enclosing_nid,
                                target_id=target_nid,
                                type="CALLS",
                                confidence=confidence,
                                reason=f"tier:{confidence:.2f}",
                            )
                        )
                        counter[0] += 1

        for child in node.children:
            walk(child, enclosing_nid)

    walk(root, None)


def _collect_calls_shallow(node, calls: list[str]) -> None:
    """Collect direct calls (non-recursive into nested function bodies)."""
    if node.type in (
        "function_definition",
        "function_declaration",
        "method_definition",
        "arrow_function",
        "function_expression",
        "function",
    ):
        return  # Don't recurse into nested function bodies at this level
    if node.type == "call":
        fn = node.child_by_field_name("function")
        if fn:
            if fn.type == "identifier":
                calls.append(fn.text.decode("utf-8"))
            elif fn.type == "attribute":
                attr = fn.child_by_field_name("attribute")
                if attr:
                    calls.append(attr.text.decode("utf-8"))
    elif node.type == "call_expression":
        fn = node.child_by_field_name("function")
        if fn:
            if fn.type == "identifier":
                calls.append(fn.text.decode("utf-8"))
            elif fn.type in ("member_expression",):
                prop = fn.child_by_field_name("property")
                if prop:
                    calls.append(prop.text.decode("utf-8"))
    elif node.type == "new_expression":
        ctor = node.child_by_field_name("constructor")
        if ctor and ctor.type == "identifier":
            calls.append(ctor.text.decode("utf-8"))

    for child in node.children:
        _collect_calls_shallow(child, calls)


def _extract_file_calls(root, file_path, lang, graph, symbol_table, import_map, counter):
    pass  # replaced by _extract_file_calls_v2


# ── 3. Heritage relationships ─────────────────────────────────


def build_heritage_relationships(
    graph: KnowledgeGraph,
    files: list[RepoFile],
    symbol_table: SymbolTable,
    import_map: ImportMap,
) -> None:
    """Build EXTENDS relationships from class inheritance."""
    rel_counter = 0

    for (fp, key), class_nid in list(symbol_table.items()):
        if not key.startswith("__extends__"):
            continue
        # Key format: __extends__{ClassName}__{BaseName}
        _, _, class_name, base_name = key.split("__", 3)

        # Resolve base class
        base_nid, confidence = _resolve_call(base_name, fp, symbol_table, import_map)
        if not base_nid:
            continue

        rel_id = _rel_id("EXTENDS", class_nid, base_nid)
        if not graph.get_relationship(rel_id):
            graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    source_id=class_nid,
                    target_id=base_nid,
                    type="EXTENDS",
                    confidence=confidence,
                    reason="class_inheritance",
                )
            )
            rel_counter += 1

    logger.info(f"[relationships] heritage edges={rel_counter}")


# ── 4. Structure relationships (HAS_METHOD) ───────────────────


def build_structure_relationships(graph: KnowledgeGraph) -> None:
    """Build HAS_METHOD edges from Class → Method nodes."""
    rel_counter = 0

    class_nodes = {
        n.properties.file_path + ":" + n.properties.name: n for n in graph.nodes_by_label("Class")
    }

    for method_node in graph.nodes_by_label("Method"):
        qualified = method_node.properties.name  # e.g. "QAEngine.build"
        if "." not in qualified:
            continue
        class_name = qualified.rsplit(".", 1)[0]
        file_path = method_node.properties.file_path
        key = file_path + ":" + class_name
        class_node = class_nodes.get(key)
        if not class_node:
            continue
        rel_id = _rel_id("HAS_METHOD", class_node.id, method_node.id)
        if not graph.get_relationship(rel_id):
            graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    source_id=class_node.id,
                    target_id=method_node.id,
                    type="HAS_METHOD",
                    confidence=1.0,
                    reason="class_method",
                )
            )
            rel_counter += 1

    logger.info(f"[relationships] has_method edges={rel_counter}")
