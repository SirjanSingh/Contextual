# Evaluation Framework

This directory contains the evaluation framework for assessing RAG system performance.

## Structure

```
evaluation/
├── test_set.json       # Test questions with expected results
├── metrics.py          # Evaluation metrics implementation
├── evaluate.py         # Main evaluation script
└── results/            # Timestamped evaluation results
```

## Quick Start

### Run Evaluation

```bash
# Baseline evaluation (vector-only, no reranking)
python -m evaluation.evaluate --repo g:\projs\repo-aware-ai --config baseline --no-hybrid --no-rerank

# Hybrid search evaluation
python -m evaluation.evaluate --repo g:\projs\repo-aware-ai --config hybrid

# Full evaluation (hybrid + reranking)
python -m evaluation.evaluate --repo g:\projs\repo-aware-ai --config full
```

### View Results

Results are saved to `evaluation/results/` with timestamps. Each result file contains:

- Configuration name and timestamp
- Aggregated metrics (averages, min, max)
- Individual question results with answers and scores
- Category breakdown

## Test Set Format

The test set (`test_set.json`) contains questions with:

- **id**: Unique question identifier
- **question**: The question text
- **expected_answer_contains**: Keywords expected in the answer
- **expected_sources**: File paths expected in retrieved sources
- **category**: Question type (direct, conceptual, multi-hop, edge-case)

Example:

```json
{
  "id": "q001",
  "question": "What does the Embedder class do?",
  "expected_answer_contains": ["embed", "Google AI", "text-embedding-004"],
  "expected_sources": ["app/embedder.py"],
  "category": "direct"
}
```

## Metrics

### Source Accuracy

- **Precision**: Relevant sources / Retrieved sources
- **Recall**: Relevant sources / Expected sources
- **F1 Score**: Harmonic mean of precision and recall

### Answer Quality

- **Keyword Score**: Matched keywords / Expected keywords

### Performance

- **Retrieval Latency**: Time to retrieve chunks (seconds)
- **Total Latency**: End-to-end question → answer time (seconds)

## Adding New Questions

To add new test questions:

1. Open `test_set.json`
2. Add a new question object following the format above
3. Choose appropriate expected sources and keywords
4. Assign a category (direct, conceptual, multi-hop, edge-case)

### Category Guidelines

- **direct**: Simple, single-hop questions about specific code elements
- **conceptual**: Questions requiring understanding of how things work
- **multi-hop**: Questions requiring information from multiple files
- **edge-case**: Questions testing boundary conditions or error cases

## Comparing Configurations

To compare different configurations:

1. Run evaluations with different `--config` names
2. Compare the saved JSON files in `results/`
3. Look for improvements in F1 score and keyword score

Example comparison:

```bash
# Run baseline
python -m evaluation.evaluate --repo . --config baseline --no-hybrid --no-rerank

# Run with hybrid search
python -m evaluation.evaluate --repo . --config hybrid

# Compare the two result files
```

## Interpreting Results

### Good Performance

- Source F1 > 0.7
- Keyword Score > 0.6
- Consistent performance across categories

### Areas for Improvement

- Low precision: Too many irrelevant chunks retrieved
- Low recall: Missing relevant sources
- Low keyword score: Answer quality needs improvement
- High latency: Performance optimization needed
