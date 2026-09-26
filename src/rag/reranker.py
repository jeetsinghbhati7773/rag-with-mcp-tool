from __future__ import annotations

import math
import re
from typing import List, Tuple
from langchain_core.documents import Document


class ChunkReranker:
    """
    Reranks retrieved candidate document chunks using a multi-signal scoring model:
    - Distance score (from ChromaDB vector similarity)
    - Keyword term overlap (BM25 style frequency matching)
    - Exact match / title match bonuses
    """

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3,
    ) -> List[Tuple[Document, float]]:
        """
        Reranks documents and returns list of (Document, composite_score) tuples.
        Composite score is normalized between 0.0 and 1.0.
        """
        if not documents:
            return []

        query_terms = set(re.findall(r"\w+", query.lower()))
        scored_docs: List[Tuple[Document, float]] = []

        for doc in documents:
            content_lower = doc.page_content.lower()
            meta = doc.metadata or {}

            # 1. Base vector distance score (Chroma distance metric where lower distance is higher similarity)
            dist = meta.get("_distance", 0.5)
            # Convert L2 / Cosine distance to similarity range [0.0, 1.0]
            vector_sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

            # 2. Term Frequency / Keyword overlap score
            doc_words = re.findall(r"\w+", content_lower)
            doc_word_count = max(1, len(doc_words))
            matched_terms = sum(1 for term in query_terms if term in content_lower)
            keyword_score = (matched_terms / max(1, len(query_terms))) if query_terms else 0.0

            # 3. Exact phrase match bonus
            exact_match_bonus = 0.2 if query.lower() in content_lower else 0.0

            # Composite final score calculation
            final_score = (vector_sim * 0.55) + (keyword_score * 0.35) + exact_match_bonus
            final_score = round(max(0.0, min(1.0, final_score)), 4)

            # Store score on document metadata
            doc.metadata["rerank_score"] = final_score
            scored_docs.append((doc, final_score))

        # Sort descending by composite score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]
