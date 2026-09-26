from __future__ import annotations

import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, or_, delete

from src.database.models import Conversation, Message, ConversationFile
from src.schemas.conversation import MessageCreate

logger = logging.getLogger(__name__)


class ConversationRepository:
    """SQLAlchemy 2.x Repository for managing Conversations, Messages, and Files in PostgreSQL/Database."""

    def create_conversation(self, db: Session, title: str = "New Conversation", convo_id: Optional[str] = None) -> Conversation:
        kwargs = {"title": title}
        if convo_id:
            kwargs["id"] = convo_id
        convo = Conversation(**kwargs)
        db.add(convo)
        db.flush()
        return convo

    def list_conversations(self, db: Session, search_query: Optional[str] = None, limit: int = 100) -> List[Conversation]:
        stmt = select(Conversation).order_by(Conversation.updated_at.desc())
        if search_query and search_query.strip():
            pattern = f"%{search_query.strip()}%"
            stmt = stmt.where(
                or_(
                    Conversation.title.ilike(pattern),
                    Conversation.summary.ilike(pattern),
                )
            )
        stmt = stmt.limit(limit)
        return list(db.execute(stmt).scalars().all())

    def get_conversation(self, db: Session, convo_id: str) -> Optional[Conversation]:
        stmt = select(Conversation).where(Conversation.id == convo_id)
        return db.execute(stmt).scalar_one_or_none()

    def update_title(self, db: Session, convo_id: str, title: str) -> Optional[Conversation]:
        convo = self.get_conversation(db, convo_id)
        if convo:
            convo.title = title
            db.flush()
        return convo

    def update_summary(self, db: Session, convo_id: str, summary: str) -> Optional[Conversation]:
        convo = self.get_conversation(db, convo_id)
        if convo:
            convo.summary = summary
            db.flush()
        return convo

    def delete_conversation(self, db: Session, convo_id: str) -> bool:
        convo = self.get_conversation(db, convo_id)
        if convo:
            db.delete(convo)
            db.flush()
            return True
        return False

    def add_message(self, db: Session, convo_id: str, message_in: MessageCreate) -> Message:
        msg = Message(
            conversation_id=convo_id,
            role=message_in.role,
            content=message_in.content,
            query_type=message_in.query_type,
            confidence_json=message_in.confidence_json,
            citations_json=message_in.citations_json,
            tool_results_json=message_in.tool_results_json,
        )
        db.add(msg)
        # Touch conversation updated_at
        convo = self.get_conversation(db, convo_id)
        if convo:
            from datetime import datetime
            convo.updated_at = datetime.utcnow()
        db.flush()
        return msg

    def get_messages(self, db: Session, convo_id: str, limit: int = 200) -> List[Message]:
        stmt = select(Message).where(Message.conversation_id == convo_id).order_by(Message.created_at.asc()).limit(limit)
        return list(db.execute(stmt).scalars().all())

    def add_conversation_file(
        self,
        db: Session,
        convo_id: str,
        file_id: str,
        filename: str,
        file_hash: str,
        file_size: int = 0,
        status: str = "ready",
    ) -> ConversationFile:
        cf = ConversationFile(
            conversation_id=convo_id,
            file_id=file_id,
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            status=status,
        )
        db.add(cf)
        db.flush()
        return cf

    def get_conversation_files(self, db: Session, convo_id: str) -> List[ConversationFile]:
        stmt = select(ConversationFile).where(ConversationFile.conversation_id == convo_id).order_by(ConversationFile.uploaded_at.asc())
        return list(db.execute(stmt).scalars().all())

    def delete_conversation_file(self, db: Session, convo_id: str, filename: str) -> bool:
        stmt = delete(ConversationFile).where(
            ConversationFile.conversation_id == convo_id,
            ConversationFile.filename == filename,
        )
        res = db.execute(stmt)
        db.flush()
        return res.rowcount > 0
