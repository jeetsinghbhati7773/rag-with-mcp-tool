from __future__ import annotations

from langchain_core.documents import Document
from src.rag.citations import format_citations, render_citations_markdown


def test_format_citations_preserves_page_numbers():
    docs = [
        Document(
            page_content="Some chunk text content",
            metadata={"document_name": "paper.pdf", "page_number": 12, "chunk_id": "c100", "rerank_score": 0.88},
        ),
        Document(
            page_content="Another chunk text content",
            metadata={"document_name": "readme.txt", "page_number": None, "chunk_id": "c101", "rerank_score": 0.65},
        ),
    ]

    citations = format_citations(docs)
    assert len(citations) == 2
    assert citations[0].document_name == "paper.pdf"
    assert citations[0].page_number == 12
    assert citations[0].score == 0.88
    assert citations[1].document_name == "readme.txt"
    assert citations[1].page_number is None

    md = render_citations_markdown(citations)
    assert "**[1] paper.pdf** | Page 12" in md
    assert "**[2] readme.txt**" in md
