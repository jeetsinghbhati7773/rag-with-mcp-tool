from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple, Optional
from src.database.connection import get_db
from src.database.repositories.document_repository import DocumentRepository
from src.schemas.document import DocumentCreate
from src.rag.vector_store import ChromaVectorStore
from src.rag.ingestion import ingest_document_bytes

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Coordinates PostgreSQL document metadata storage and ChromaDB vector chunk indexing.
    """

    def __init__(self, vectorstore: Optional[ChromaVectorStore] = None):
        self.repo = DocumentRepository()
        self.vectorstore = vectorstore or ChromaVectorStore()

    def process_and_index_document(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> Tuple[int, str, bool]:
        """
        Calculates file hash, checks PostgreSQL document registry, indexes chunks into ChromaDB,
        and saves document metadata to PostgreSQL.
        """
        # 1. Ingest into vectorstore & compute hash
        chunks_added, msg, is_dup = ingest_document_bytes(file_bytes, filename, self.vectorstore)
        if is_dup:
            return 0, msg, True

        if chunks_added > 0:
            # 2. Record document metadata in PostgreSQL
            import os
            from src.utils.hashing import calculate_sha256
            file_hash = calculate_sha256(file_bytes)
            ext = os.path.splitext(filename)[1].lower()

            doc_create = DocumentCreate(
                filename=filename,
                file_hash=file_hash,
                file_type=ext,
                file_size=len(file_bytes),
                page_count=1,
                chunk_count=chunks_added,
                status="indexed",
            )

            try:
                with get_db() as db:
                    if db is not None:
                        self.repo.create(db, doc_create)
                        logger.info("Saved document metadata for '%s' to PostgreSQL.", filename)
            except Exception as e:
                logger.warning("Could not write document metadata to PostgreSQL: %s", e)

        return chunks_added, msg, False

    def list_documents(self) -> List[Dict[str, Any]]:
        """Lists indexed documents from PostgreSQL or ChromaDB fallback."""
        try:
            with get_db() as db:
                if db is not None:
                    docs = self.repo.get_all(db)
                    if docs:
                        return [
                            {
                                "Document Name": d.filename,
                                "Chunks": d.chunk_count,
                                "Pages": d.page_count,
                                "Size": f"{d.file_size / 1024:.1f} KB",
                                "Hash": d.file_hash[:8],
                                "Status": d.status,
                            }
                            for d in docs
                        ]
        except Exception as e:
            logger.warning("Could not read document registry from PostgreSQL: %s", e)

        return self.vectorstore.list_documents()

    def delete_document(self, filename: str) -> int:
        """Deletes document chunks from ChromaDB and metadata from PostgreSQL."""
        deleted_count = self.vectorstore.delete_document(filename)
        try:
            with get_db() as db:
                if db is not None:
                    self.repo.delete(db, filename)
        except Exception as e:
            logger.warning("Could not delete document metadata from PostgreSQL: %s", e)
        return deleted_count


document_service = DocumentService()
