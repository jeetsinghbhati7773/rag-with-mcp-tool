from __future__ import annotations

import logging
import re
from typing import List, Tuple, Optional
from langchain_core.documents import Document

from src.rag.vector_store import ChromaVectorStore
from src.rag.reranker import ChunkReranker
from src.rag.confidence import calculate_retrieval_confidence
from src.rag.citations import format_citations
from src.schemas.models import Citation, ConfidenceScore
from src.config import settings

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Advanced RAG Document Retriever.
    Supports Query Transformation, Vector + Keyword Hybrid Search, Multi-Signal Reranking,
    Evidence Confidence Calculation, and Granular Citation Building.
    """

    def __init__(self, vectorstore: ChromaVectorStore):
        self.vectorstore = vectorstore
        self.reranker = ChunkReranker()

    def rewrite_query(self, query: str) -> str:
        """
        Transforms user query into an optimized retrieval query by stripping conversational prefix framing.
        e.g., 'What does my uploaded networking.pdf say about TCP?' -> 'networking.pdf TCP definition mechanisms'
        """
        q_clean = query.strip()
        # Remove conversational prefixes
        patterns = [
            r"^(what does|what did|can you tell me|according to|explain what|search for|find info on)\s+(my|the|uploaded)?\s*",
            r"^(uploaded|file|document|paper|pdf)\s*",
        ]
        for p in patterns:
            q_clean = re.sub(p, "", q_clean, flags=re.IGNORECASE).strip()
        
        return q_clean if len(q_clean) >= 3 else query

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_k: Optional[int] = None,
        conversation_id: Optional[str] = None,
        document_names: Optional[List[str]] = None,
    ) -> Tuple[List[Document], List[Citation], ConfidenceScore]:
        """
        Retrieves top relevant document chunks using hybrid vector search & reranking.
        Scoped to conversation_id and/or document_names.
        Returns (reranked_documents, citations, confidence_score).
        """
        k = top_k or settings.retrieval_k
        rk = rerank_k or settings.rerank_top_k

        if document_names is not None and len(document_names) == 0:
            return [], [], calculate_retrieval_confidence([])

        # Step 1: Query Transformation
        transformed_query = self.rewrite_query(query)
        logger.debug("Original query: %r -> Transformed: %r", query, transformed_query)

        # Step 2: Vector Similarity Search
        vector_candidates = self.vectorstore.similarity_search(
            transformed_query,
            k=k,
            conversation_id=conversation_id,
            document_names=document_names,
        )
        if not vector_candidates and transformed_query != query:
            # Fallback to original query
            vector_candidates = self.vectorstore.similarity_search(
                query,
                k=k,
                conversation_id=conversation_id,
                document_names=document_names,
            )

        if not vector_candidates:
            empty_confidence = calculate_retrieval_confidence([])
            return [], [], empty_confidence

        # Step 3: Multi-signal Reranking
        reranked_tuples = self.reranker.rerank(query, vector_candidates, top_k=rk)
        reranked_docs = [doc for doc, _ in reranked_tuples]

        # Step 4: Confidence Score Calculation
        confidence = calculate_retrieval_confidence(reranked_tuples)

        # Step 5: Format Traceable Citations
        citations = format_citations(reranked_docs)

        return reranked_docs, citations, confidence

    def build_context_prompt(self, docs: List[Document]) -> str:
        """
        Builds a structured grounded context prompt string from retrieved documents.
        """
        if not docs:
            return "No relevant documents found in knowledge base."

        context_blocks = []
        for idx, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            doc_name = meta.get("document_name") or meta.get("source") or "Document"
            page_num = meta.get("page_number")
            chunk_id = meta.get("chunk_id", f"chunk_{idx}")
            page_info = f" | Page {page_num}" if page_num is not None else ""
            score = meta.get("rerank_score")
            score_info = f" | Score: {score:.2f}" if score is not None else ""

            block = (
                f"[Source Document {idx}: {doc_name}{page_info} | Chunk ID: {chunk_id}{score_info}]\n"
                f"{doc.page_content.strip()}"
            )
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)
