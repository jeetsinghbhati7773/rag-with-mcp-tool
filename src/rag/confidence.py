from __future__ import annotations

from typing import List, Tuple
from langchain_core.documents import Document
from src.schemas.models import ConfidenceScore
from src.config import settings


def calculate_retrieval_confidence(
    reranked_docs: List[Tuple[Document, float]]
) -> ConfidenceScore:
    """
    Calculates evidence-based confidence score from retrieved/reranked document signals.
    Does NOT ask the LLM 'how confident are you?'; bases score purely on retrieval evidence metrics.
    """
    if not reranked_docs:
        return ConfidenceScore(
            level="Insufficient evidence",
            score=0.0,
            explanation="No relevant document chunks were found in the knowledge base.",
        )

    # Calculate average and max composite scores from top reranked chunks
    scores = [score for _, score in reranked_docs]
    top_score = max(scores)
    avg_score = sum(scores) / len(scores)

    combined_metric = (top_score * 0.7) + (avg_score * 0.3)
    combined_metric = round(min(1.0, max(0.0, combined_metric)), 3)

    if combined_metric >= settings.confidence_high_threshold:
        return ConfidenceScore(
            level="High confidence",
            score=combined_metric,
            explanation=f"Strong evidence retrieved (Top Rerank Score: {top_score:.2f}, Avg Score: {avg_score:.2f}).",
        )
    elif combined_metric >= settings.confidence_moderate_threshold:
        return ConfidenceScore(
            level="Moderate confidence",
            score=combined_metric,
            explanation=f"Partial evidence retrieved (Top Rerank Score: {top_score:.2f}, Avg Score: {avg_score:.2f}).",
        )
    else:
        return ConfidenceScore(
            level="Insufficient evidence",
            score=combined_metric,
            explanation=f"Low relevance scores across retrieved chunks (Top Score: {top_score:.2f}). Context may be insufficient.",
        )
