from __future__ import annotations

import pytest
from pathlib import Path
from langchain_core.documents import Document
from src.rag.vector_store import ChromaVectorStore
from src.rag.retriever import DocumentRetriever


def test_chroma_vector_store_add_and_retrieve(monkeypatch):
    # Prevent Rust ONNX binding instantiation in test environment
    monkeypatch.setattr("src.rag.embeddings.get_embedding_model", lambda: None)

    store = ChromaVectorStore(persist_directory=Path(":memory:"))

    docs = [
        Document(
            page_content="Differential privacy ensures statistical privacy for database queries.",
            metadata={"document_name": "privacy.pdf", "page_number": 8, "chunk_id": "c1"},
        ),
        Document(
            page_content="Transformers use multi-head self-attention mechanisms.",
            metadata={"document_name": "attention.pdf", "page_number": 2, "chunk_id": "c2"},
        ),
    ]

    # Force fallback memory in unit test environment to prevent Windows Rust PyO3 crash
    monkeypatch.setattr(store, "_get_collection", lambda: None)

    added = store.add_documents(docs)
    assert added == 2
    assert store.count_chunks() == 2

    retriever = DocumentRetriever(store)
    retrieved, citations, confidence = retriever.retrieve("differential privacy", top_k=1)
    assert len(retrieved) == 1
    assert "differential privacy" in retrieved[0].page_content.lower()
    assert citations[0].document_name == "privacy.pdf"
    assert citations[0].page_number == 8
    assert confidence.level in ("High confidence", "Moderate confidence", "Insufficient evidence")
