from __future__ import annotations

from pathlib import Path
from src.rag.vector_store import ChromaVectorStore
from src.rag.ingestion import ingest_document_bytes
from src.utils.hashing import calculate_sha256


def test_ingestion_duplicate_detection(monkeypatch):
    monkeypatch.setattr("src.rag.embeddings.get_embedding_model", lambda: None)

    store = ChromaVectorStore(persist_directory=Path(":memory:"))
    monkeypatch.setattr(store, "_get_collection", lambda: None)

    content = b"Sample document text content for SHA256 duplicate testing."
    filename = "test_doc.txt"

    # First ingestion
    added, msg, is_dup = ingest_document_bytes(content, filename, store)
    assert is_dup is False
    assert added > 0

    # Second ingestion of exact same content
    added_dup, msg_dup, is_dup_2 = ingest_document_bytes(content, filename, store)
    assert is_dup_2 is True
    assert added_dup == 0
    assert "already indexed" in msg_dup


def test_document_deletion(monkeypatch):
    monkeypatch.setattr("src.rag.embeddings.get_embedding_model", lambda: None)

    store = ChromaVectorStore(persist_directory=Path(":memory:"))
    monkeypatch.setattr(store, "_get_collection", lambda: None)

    content = b"Content to be deleted from ChromaDB vectorstore memory."
    filename = "to_delete.txt"

    ingest_document_bytes(content, filename, store)
    assert store.count_chunks() > 0

    deleted_count = store.delete_document(filename)
    assert deleted_count > 0
    assert store.count_chunks() == 0
