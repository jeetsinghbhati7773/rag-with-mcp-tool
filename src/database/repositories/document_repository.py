from __future__ import annotations

import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.database.models import Document
from src.schemas.document import DocumentCreate

logger = logging.getLogger(__name__)


class DocumentRepository:
    """SQLAlchemy 2.x Repository for managing Document metadata persistence in PostgreSQL."""

    def create(self, db: Session, doc_in: DocumentCreate) -> Document:
        doc = Document(
            filename=doc_in.filename,
            file_hash=doc_in.file_hash,
            file_type=doc_in.file_type,
            file_size=doc_in.file_size,
            page_count=doc_in.page_count,
            chunk_count=doc_in.chunk_count,
            status=doc_in.status,
        )
        db.add(doc)
        db.flush()
        return doc

    def get_by_hash(self, db: Session, file_hash: str) -> Optional[Document]:
        stmt = select(Document).where(Document.file_hash == file_hash)
        return db.execute(stmt).scalar_one_or_none()

    def get_by_filename(self, db: Session, filename: str) -> Optional[Document]:
        stmt = select(Document).where(Document.filename == filename)
        return db.execute(stmt).scalar_one_or_none()

    def get_all(self, db: Session) -> List[Document]:
        stmt = select(Document).order_by(Document.uploaded_at.desc())
        return list(db.execute(stmt).scalars().all())

    def delete(self, db: Session, filename: str) -> bool:
        doc = self.get_by_filename(db, filename)
        if not doc:
            return False
        db.delete(doc)
        db.flush()
        return True
