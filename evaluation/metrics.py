"""Evaluation metrics for RAG system."""
from __future__ import annotations

from typing import Dict, List, Set


def source_precision(predicted_sources: List[str], expected_sources: List[str]) -> float:
    """Calculate precision of retrieved sources.
    
    Precision = (relevant sources retrieved) / (total sources retrieved)
    
    Args:
        predicted_sources: List of source file paths from retrieval.
        expected_sources: List of expected source file paths.
    
    Returns:
        Precision score between 0 and 1.
    """
    if not predicted_sources:
        return 0.0
    
    # Extract just the filename from full paths for comparison
    predicted_files = {_extract_filename(s) for s in predicted_sources}
    expected_files = {_extract_filename(s) for s in expected_sources}
    
    relevant_retrieved = predicted_files & expected_files
    
    return len(relevant_retrieved) / len(predicted_files)


def source_recall(predicted_sources: List[str], expected_sources: List[str]) -> float:
    """Calculate recall of retrieved sources.
    
    Recall = (relevant sources retrieved) / (total relevant sources)
    
    Args:
        predicted_sources: List of source file paths from retrieval.
        expected_sources: List of expected source file paths.
    
    Returns:
        Recall score between 0 and 1.
    """
    if not expected_sources:
        return 1.0 if not predicted_sources else 0.0
    
    # Extract just the filename from full paths for comparison
    predicted_files = {_extract_filename(s) for s in predicted_sources}
    expected_files = {_extract_filename(s) for s in expected_sources}
    
    relevant_retrieved = predicted_files & expected_files
    
    return len(relevant_retrieved) / len(expected_files)


def source_f1(predicted_sources: List[str], expected_sources: List[str]) -> float:
    """Calculate F1 score of retrieved sources.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        predicted_sources: List of source file paths from retrieval.
        expected_sources: List of expected source file paths.
    
    Returns:
        F1 score between 0 and 1.
    """
    precision = source_precision(predicted_sources, expected_sources)
    recall = source_recall(predicted_sources, expected_sources)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def answer_keyword_score(answer: str, expected_keywords: List[str]) -> float:
    """Calculate keyword overlap between answer and expected keywords.
    
    Score = (matched keywords) / (total expected keywords)
    
    Args:
        answer: Generated answer text.
        expected_keywords: List of expected keywords in answer.
    
    Returns:
        Keyword score between 0 and 1.
    """
    if not expected_keywords:
        return 1.0
    
    answer_lower = answer.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    
    return matches / len(expected_keywords)


def _extract_filename(source: str) -> str:
    """Extract filename from source path or reference.
    
    Examples:
        - 'app/loader.py:123-456' -> 'loader.py'
        - 'app/loader.py' -> 'loader.py'
        - 'g:\\projs\\repo\\app\\loader.py' -> 'loader.py'
    """
    # Remove line numbers if present
    if ':' in source:
        source = source.split(':')[0]
    
    # Get filename
    if '/' in source:
        return source.split('/')[-1]
    elif '\\' in source:
        return source.split('\\')[-1]
    else:
        return source


def calculate_all_metrics(
    predicted_sources: List[str],
    expected_sources: List[str],
    answer: str,
    expected_keywords: List[str],
    retrieval_time: float,
    total_time: float,
) -> Dict[str, float]:
    """Calculate all evaluation metrics.
    
    Args:
        predicted_sources: Retrieved source references.
        expected_sources: Expected source files.
        answer: Generated answer.
        expected_keywords: Expected keywords in answer.
        retrieval_time: Time taken to retrieve chunks (seconds).
        total_time: Time taken for full QA pipeline (seconds).
    
    Returns:
        Dictionary of metric name to score.
    """
    return {
        "source_precision": source_precision(predicted_sources, expected_sources),
        "source_recall": source_recall(predicted_sources, expected_sources),
        "source_f1": source_f1(predicted_sources, expected_sources),
        "answer_keyword_score": answer_keyword_score(answer, expected_keywords),
        "retrieval_latency": retrieval_time,
        "total_latency": total_time,
    }
