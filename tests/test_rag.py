from __future__ import annotations

from langchain_core.documents import Document
from src.rag.loaders import load_txt_bytes, load_document_bytes
from src.rag.chunking import split_documents


def test_load_txt_bytes_creates_document():
    text = "Hello world! This is a test document content for RAG."
    docs = load_txt_bytes(text.encode("utf-8"), "test.txt")
    assert len(docs) == 1
    assert docs[0].page_content == text
    assert docs[0].metadata["document_name"] == "test.txt"


def test_split_documents_preserves_metadata():
    docs = [
        Document(
            page_content="Word " * 300,
            metadata={"document_name": "sample.pdf", "page_number": 3},
        )
    ]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["document_name"] == "sample.pdf"
        assert c.metadata["page_number"] == 3
        assert "chunk_id" in c.metadata
