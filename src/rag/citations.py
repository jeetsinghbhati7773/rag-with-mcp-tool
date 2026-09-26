from __future__ import annotations

from typing import List
from langchain_core.documents import Document
from src.schemas.models import Citation


def format_citations(documents: List[Document]) -> List[Citation]:
    """
    Extracts citation objects from retrieved LangChain documents.
    Preserves document_name, page_number (without hallucinating missing numbers),
    chunk_id, score, and a text snippet.
    """
    citations = []
    seen_ids = set()

    for doc in documents:
        meta = doc.metadata or {}
        chunk_id = meta.get("chunk_id", "chunk")
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        doc_name = meta.get("document_name") or meta.get("source") or "Document"
        page_num = meta.get("page_number")
        score = meta.get("rerank_score")

        snippet = doc.page_content.strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "…"

        citations.append(
            Citation(
                document_name=doc_name,
                page_number=page_num if isinstance(page_num, int) else None,
                snippet=snippet,
                chunk_id=chunk_id,
                score=score,
            )
        )

    return citations


def render_citations_markdown(citations: List[Citation]) -> str:
    """
    Renders structured Markdown citations list for assistant responses with clickable/traceable references.
    """
    if not citations:
        return ""

    lines = ["\n\n### 📑 Sources & References"]
    for idx, c in enumerate(citations, 1):
        page_str = f" | Page {c.page_number}" if c.page_number is not None else ""
        chunk_str = f" | Chunk: `{c.chunk_id}`"
        score_str = f" | Score: `{c.score:.2f}`" if c.score is not None else ""
        lines.append(
            f"**[{idx}] {c.document_name}**{page_str}{chunk_str}{score_str}\n"
            f"> *\"{c.snippet}\"*\n"
        )

    return "\n".join(lines)
