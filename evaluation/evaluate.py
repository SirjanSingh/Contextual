"""Evaluation script for RAG system."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from app.embedder import Embedder
from app.llm import LLMClient
from app.qa import QAEngine
from evaluation.metrics import calculate_all_metrics


def load_test_set(test_set_path: Path) -> List[Dict]:
    """Load test questions from JSON file."""
    with open(test_set_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def evaluate_question(
    engine: QAEngine,
    question_data: Dict,
) -> Dict:
    """Evaluate a single question.
    
    Args:
        engine: QA engine instance.
        question_data: Test question with expected results.
    
    Returns:
        Evaluation results for this question.
    """
    question = question_data["question"]
    expected_sources = question_data["expected_sources"]
    expected_keywords = question_data["expected_answer_contains"]
    
    # Measure retrieval and total time
    start_time = time.time()
    answer, sources = engine.ask(question)
    total_time = time.time() - start_time
    
    # For now, we approximate retrieval time as 80% of total time
    # In a real scenario, you'd instrument the retrieval function
    retrieval_time = total_time * 0.8
    
    # Calculate metrics
    metrics = calculate_all_metrics(
        predicted_sources=sources,
        expected_sources=expected_sources,
        answer=answer,
        expected_keywords=expected_keywords,
        retrieval_time=retrieval_time,
        total_time=total_time,
    )
    
    return {
        "question_id": question_data["id"],
        "question": question,
        "answer": answer,
        "sources": sources,
        "expected_sources": expected_sources,
        "expected_keywords": expected_keywords,
        "metrics": metrics,
        "category": question_data["category"],
    }


def aggregate_metrics(results: List[Dict]) -> Dict:
    """Aggregate metrics across all questions.
    
    Args:
        results: List of evaluation results.
    
    Returns:
        Aggregated metrics.
    """
    if not results:
        return {}
    
    # Aggregate numeric metrics
    metric_names = [
        "source_precision",
        "source_recall",
        "source_f1",
        "answer_keyword_score",
        "retrieval_latency",
        "total_latency",
    ]
    
    aggregated = {}
    
    for metric_name in metric_names:
        values = [r["metrics"][metric_name] for r in results]
        aggregated[f"avg_{metric_name}"] = sum(values) / len(values)
        aggregated[f"min_{metric_name}"] = min(values)
        aggregated[f"max_{metric_name}"] = max(values)
    
    # Category breakdown
    category_counts = {}
    category_scores = {}
    
    for result in results:
        category = result["category"]
        if category not in category_counts:
            category_counts[category] = 0
            category_scores[category] = []
        
        category_counts[category] += 1
        category_scores[category].append(result["metrics"]["source_f1"])
    
    aggregated["category_breakdown"] = {
        cat: {
            "count": category_counts[cat],
            "avg_source_f1": sum(category_scores[cat]) / len(category_scores[cat]),
        }
        for cat in category_counts
    }
    
    return aggregated


def save_results(results: List[Dict], aggregated: Dict, config_name: str, output_dir: Path) -> Path:
    """Save evaluation results to file.
    
    Args:
        results: Individual question results.
        aggregated: Aggregated metrics.
        config_name: Configuration name for this run.
        output_dir: Output directory for results.
    
    Returns:
        Path to saved results file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config_name}_{timestamp}.json"
    output_path = output_dir / filename
    
    output = {
        "config": config_name,
        "timestamp": timestamp,
        "aggregated_metrics": aggregated,
        "individual_results": results,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    return output_path


def print_summary(aggregated: Dict, config_name: str) -> None:
    """Print summary of evaluation results."""
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY: {config_name}")
    print(f"{'='*60}\n")
    
    print("Source Retrieval Metrics:")
    print(f"  Precision: {aggregated['avg_source_precision']:.3f}")
    print(f"  Recall:    {aggregated['avg_source_recall']:.3f}")
    print(f"  F1 Score:  {aggregated['avg_source_f1']:.3f}")
    
    print("\nAnswer Quality:")
    print(f"  Keyword Score: {aggregated['avg_answer_keyword_score']:.3f}")
    
    print("\nLatency:")
    print(f"  Retrieval: {aggregated['avg_retrieval_latency']:.3f}s")
    print(f"  Total:     {aggregated['avg_total_latency']:.3f}s")
    
    print("\nCategory Breakdown:")
    for category, stats in aggregated["category_breakdown"].items():
        print(f"  {category:12s}: {stats['count']} questions, F1={stats['avg_source_f1']:.3f}")
    
    print(f"\n{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--repo", required=True, help="Path to repository to evaluate on")
    parser.add_argument("--config", default="default", help="Configuration name for this run")
    parser.add_argument("--test-set", default="evaluation/test_set.json", help="Path to test set JSON")
    parser.add_argument("--output-dir", default="evaluation/results", help="Output directory for results")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid search")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--topk", type=int, default=6, help="Top-k chunks to retrieve")
    
    args = parser.parse_args()
    
    # Load test set
    test_set_path = Path(args.test_set)
    questions = load_test_set(test_set_path)
    print(f"[+] Loaded {len(questions)} questions from {test_set_path}")
    
    # Initialize QA engine
    print("[+] Initializing QA engine...")
    embedder = Embedder()
    llm = LLMClient(temperature=0.2)
    
    engine = QAEngine(
        repo_root=Path(args.repo),
        cache_base=Path("data/index"),
        embedder=embedder,
        llm=llm,
        top_k=args.topk,
        use_reranker=not args.no_rerank,
        use_hybrid_search=not args.no_hybrid,
        use_conversation=False,  # Disable for evaluation (each question independent)
    )
    
    print("[+] Building/loading index...")
    engine.build(force_rebuild=False)
    
    # Evaluate each question
    print(f"[+] Evaluating {len(questions)} questions...\n")
    results = []
    
    for i, question_data in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question_data['question'][:60]}...")
        result = evaluate_question(engine, question_data)
        results.append(result)
        print(f"    F1: {result['metrics']['source_f1']:.3f}, "
              f"Keywords: {result['metrics']['answer_keyword_score']:.3f}")
    
    # Aggregate and save
    aggregated = aggregate_metrics(results)
    output_path = save_results(results, aggregated, args.config, Path(args.output_dir))
    
    print(f"\n[+] Results saved to: {output_path}")
    
    # Print summary
    print_summary(aggregated, args.config)


if __name__ == "__main__":
    main()
