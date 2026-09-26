from __future__ import annotations

from src.evaluation.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_faithfulness,
    calculate_answer_relevance,
    compute_benchmark_metrics,
)


def test_retrieval_metrics_precision_recall_mrr():
    retrieved = ["paper.pdf", "notes.pdf", "other.pdf"]
    target = "paper.pdf"

    pk = calculate_precision_at_k(retrieved, target)
    rk = calculate_recall_at_k(retrieved, target)
    mrr = calculate_mrr(retrieved, target)

    assert pk == round(1 / 3, 4)
    assert rk == 1.0
    assert mrr == 1.0


def test_retrieval_mrr_second_rank():
    retrieved = ["other.pdf", "paper.pdf", "notes.pdf"]
    target = "paper.pdf"

    mrr = calculate_mrr(retrieved, target)
    assert mrr == 0.5


def test_generation_faithfulness_and_relevance():
    context = "Differential privacy adds noise to gradients during training."
    answer = "Differential privacy adds noise to gradients."
    question = "What does differential privacy add to gradients?"

    faith = calculate_faithfulness(answer, context)
    rel = calculate_answer_relevance(answer, question)

    assert faith > 0.8
    assert rel > 0.5


def test_compute_benchmark_metrics():
    eval_data = [
        {"precision_at_k": 0.8, "recall_at_k": 1.0, "mrr": 1.0, "faithfulness": 0.9, "answer_relevance": 0.85},
        {"precision_at_k": 0.6, "recall_at_k": 1.0, "mrr": 0.5, "faithfulness": 0.8, "answer_relevance": 0.75},
    ]

    metrics = compute_benchmark_metrics(eval_data)
    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == 0.7
    assert metrics.mrr == 0.75
    assert metrics.faithfulness == 0.85
