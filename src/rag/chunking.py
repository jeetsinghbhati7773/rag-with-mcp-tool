from __future__ import annotations

import uuid
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Splits documents into semantic chunks while preserving document_name,
    page_number, chunk_id, file_hash, and source metadata.
    """
    c_size = chunk_size or settings.chunk_size
    c_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Document] = []
    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        doc_name = doc.metadata.get("document_name", "document")
        page_num = doc.metadata.get("page_number")

        for idx, chunk in enumerate(doc_chunks):
            chunk_uuid = str(uuid.uuid4())[:8]
            chunk_id = f"{doc_name}_p{page_num if page_num else 1}_c{idx+1}_{chunk_uuid}"

            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["document_name"] = doc_name
            chunk.metadata["page_number"] = page_num
            chunk.metadata["source"] = doc.metadata.get("source", doc_name)
            chunks.append(chunk)

    return chunks
