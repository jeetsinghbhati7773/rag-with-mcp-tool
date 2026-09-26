from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.assistant.assistant import KnowledgeAssistant
from src.evaluation.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_faithfulness,
    calculate_answer_relevance,
    compute_benchmark_metrics,
)
from src.schemas.models import EvaluationMetrics

logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


def run_rag_evaluation(
    assistant: KnowledgeAssistant,
    dataset_path: Path = DATASET_PATH,
) -> Tuple[EvaluationMetrics, List[Dict[str, Any]]]:
    """
    Runs automated evaluation across benchmark Q&A pairs in dataset.json.
    Returns (aggregated_metrics, per_item_results).
    """
    if not dataset_path.exists():
        logger.warning("Evaluation dataset file not found at %s", dataset_path)
        return EvaluationMetrics(), []

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    for item in dataset:
        q = item.get("question", "")
        target_doc = item.get("document_name", "")

        # Run assistant retrieval
        retrieved_docs, citations, conf = assistant.retriever.retrieve(q)
        doc_names = [d.metadata.get("document_name", "") for d in retrieved_docs]

        pk = calculate_precision_at_k(doc_names, target_doc)
        rk = calculate_recall_at_k(doc_names, target_doc)
        mrr = calculate_mrr(doc_names, target_doc)

        context_str = assistant.retriever.build_context_prompt(retrieved_docs)
        gold = item.get("gold_answer", "")
        faith = calculate_faithfulness(gold, context_str)
        rel = calculate_answer_relevance(gold, q)

        results.append({
            "id": item.get("id"),
            "question": q,
            "target_document": target_doc,
            "retrieved_count": len(retrieved_docs),
            "precision_at_k": pk,
            "recall_at_k": rk,
            "mrr": mrr,
            "faithfulness": faith,
            "answer_relevance": rel,
        })

    summary = compute_benchmark_metrics(results)
    return summary, results


if __name__ == "__main__":
    from src.rag.vector_store import ChromaVectorStore
    vs = ChromaVectorStore()
    ast = KnowledgeAssistant(vs)
    metrics, details = run_rag_evaluation(ast)
    print("=== RAG EVALUATION BENCHMARK METRICS ===")
    print(f"Recall@5:       {metrics.recall_at_k:.2f}")
    print(f"Precision@5:    {metrics.precision_at_k:.2f}")
    print(f"MRR:            {metrics.mrr:.2f}")
    print(f"Hit Rate:       {metrics.hit_rate:.2f}")
    print(f"Faithfulness:   {metrics.faithfulness:.2f}")
    print(f"Relevance:      {metrics.answer_relevance:.2f}")
