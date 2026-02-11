"""AST-based chunking for Python files.

Parses Python source into function/class boundaries for more semantically
meaningful chunks instead of arbitrary character splits.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional

from .chunker import Chunk, chunk_text


def ast_chunk_python(
    text: str,
    source: str,
    max_chunk_size: int = 3000,
) -> List[Chunk]:
    """Chunk a Python file by AST node boundaries (functions, classes).
    
    Extracts top-level functions, classes, and their methods as individual
    chunks. Falls back to character-based chunking if parsing fails.
    
    Args:
        text: Python source code.
        source: Relative file path (for metadata).
        max_chunk_size: Maximum size per chunk in characters. Nodes larger
            than this are split using character-based chunking.
    
    Returns:
        List of Chunk objects aligned to code structure.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Fall back to character chunking for unparseable files
        return chunk_text(text, source, chunk_size=1800, overlap=250)
    
    lines = text.splitlines(keepends=True)
    chunks: List[Chunk] = []
    
    # Collect module-level preamble (imports, module docstring, constants)
    preamble_end = _find_preamble_end(tree, lines)
    if preamble_end > 0:
        preamble_text = "".join(lines[:preamble_end])
        if preamble_text.strip():
            start_char = 0
            end_char = len(preamble_text)
            chunks.append(Chunk(
                text=preamble_text,
                source=source,
                start_char=start_char,
                end_char=end_char,
            ))
    
    # Extract top-level nodes (functions, classes)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk = _node_to_chunk(node, lines, source, text)
            if chunk:
                if len(chunk.text) > max_chunk_size:
                    # Split large functions with character chunking
                    chunks.extend(chunk_text(chunk.text, source,
                                            chunk_size=max_chunk_size, overlap=250))
                else:
                    chunks.append(chunk)
        
        elif isinstance(node, ast.ClassDef):
            # Extract class body and its methods separately
            class_chunks = _extract_class_chunks(node, lines, source, text, max_chunk_size)
            chunks.extend(class_chunks)
    
    # If AST produced no chunks (unusual), fall back
    if not chunks:
        return chunk_text(text, source, chunk_size=1800, overlap=250)
    
    return chunks


def _find_preamble_end(tree: ast.Module, lines: List[str]) -> int:
    """Find the line number where the preamble (imports, docstring) ends."""
    first_def_line = len(lines)
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_def_line = min(first_def_line, node.lineno - 1)
            break
    
    return first_def_line


def _node_to_chunk(
    node: ast.AST,
    lines: List[str],
    source: str,
    full_text: str,
) -> Optional[Chunk]:
    """Convert an AST node to a Chunk."""
    start_line = node.lineno - 1  # 0-indexed
    end_line = node.end_lineno  # exclusive
    
    if end_line is None:
        return None
    
    node_text = "".join(lines[start_line:end_line])
    if not node_text.strip():
        return None
    
    # Calculate character offsets
    start_char = sum(len(lines[i]) for i in range(start_line))
    end_char = start_char + len(node_text)
    
    return Chunk(
        text=node_text,
        source=source,
        start_char=start_char,
        end_char=end_char,
    )


def _extract_class_chunks(
    class_node: ast.ClassDef,
    lines: List[str],
    source: str,
    full_text: str,
    max_chunk_size: int,
) -> List[Chunk]:
    """Extract chunks from a class: class header + individual methods.
    
    If the entire class is small enough, returns it as one chunk.
    Otherwise, splits into class header + separate method chunks.
    """
    class_chunk = _node_to_chunk(class_node, lines, source, full_text)
    if not class_chunk:
        return []
    
    # If class is small enough, keep as single chunk
    if len(class_chunk.text) <= max_chunk_size:
        return [class_chunk]
    
    chunks: List[Chunk] = []
    
    # Extract class header (signature + docstring + class-level code before first method)
    methods = [n for n in ast.iter_child_nodes(class_node)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    
    if methods:
        # Class header: from class definition to first method
        header_start = class_node.lineno - 1
        header_end = methods[0].lineno - 1
        header_text = "".join(lines[header_start:header_end])
        
        if header_text.strip():
            start_char = sum(len(lines[i]) for i in range(header_start))
            chunks.append(Chunk(
                text=header_text,
                source=source,
                start_char=start_char,
                end_char=start_char + len(header_text),
            ))
        
        # Each method as a separate chunk, prepended with class signature for context
        class_sig = f"class {class_node.name}:\n"
        
        for method in methods:
            method_chunk = _node_to_chunk(method, lines, source, full_text)
            if method_chunk:
                # Prepend class name for context
                contextualized_text = class_sig + method_chunk.text
                
                if len(contextualized_text) > max_chunk_size:
                    sub_chunks = chunk_text(contextualized_text, source,
                                           chunk_size=max_chunk_size, overlap=250)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(Chunk(
                        text=contextualized_text,
                        source=source,
                        start_char=method_chunk.start_char,
                        end_char=method_chunk.end_char,
                    ))
    else:
        # No methods — split with character chunking
        chunks.extend(chunk_text(class_chunk.text, source,
                                 chunk_size=max_chunk_size, overlap=250))
    
    return chunks
