from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document

from src.rag.loaders import load_document_bytes
from src.rag.chunking import split_documents
from src.rag.vector_store import ChromaVectorStore
from src.utils.hashing import calculate_sha256
from src.security.validators import validate_file_upload

logger = logging.getLogger(__name__)


def ingest_document_bytes(
    file_bytes: bytes,
    filename: str,
    vectorstore: ChromaVectorStore,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Tuple[int, str, bool]:
    """
    Ingests document bytes into ChromaDB vectorstore with SHA-256 duplicate detection.
    Returns: (chunks_added, status_message, is_duplicate)
    """
    # 1. Validate size & format
    is_valid, err_msg = validate_file_upload(filename, len(file_bytes))
    if not is_valid:
        return 0, err_msg, False

    # 2. SHA-256 Hash check for duplicates
    file_hash = calculate_sha256(file_bytes)
    if vectorstore.has_document_hash(file_hash):
        existing_doc = vectorstore.get_document_by_hash(file_hash)
        doc_name = existing_doc.get("document_name", filename)
        msg = f"Document '{doc_name}' (SHA-256: {file_hash[:8]}...) is already indexed in ChromaDB."
        logger.info(msg)
        return 0, msg, True

    # 3. Load & chunk document
    raw_docs = load_document_bytes(file_bytes, filename)
    if not raw_docs:
        return 0, f"Could not extract text content from '{filename}'.", False

    total_pages = len(set(d.metadata.get("page_number") for d in raw_docs if d.metadata.get("page_number") is not None))
    created_at = datetime.now().isoformat()

    # 4. Attach rich metadata to raw docs before chunking
    doc_id = f"doc_{file_hash[:12]}"
    for idx, doc in enumerate(raw_docs):
        doc.metadata["document_id"] = doc_id
        doc.metadata["document_name"] = filename
        doc.metadata["file_hash"] = file_hash
        doc.metadata["file_size_bytes"] = len(file_bytes)
        doc.metadata["created_at"] = created_at
        if total_pages > 0:
            doc.metadata["total_pages"] = total_pages

    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # 5. Ingest into vector store
    added_count = vectorstore.add_documents(chunks)
    msg = f"Successfully indexed '{filename}' ({added_count} chunks created)."
    return added_count, msg, False
