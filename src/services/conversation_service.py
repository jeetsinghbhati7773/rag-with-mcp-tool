from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from src.database.connection import get_db
from src.database.repositories.conversation_repository import ConversationRepository
from src.schemas.conversation import MessageCreate

logger = logging.getLogger(__name__)


class ConversationService:
    """Manages multi-turn conversation sessions and message history in PostgreSQL."""

    def __init__(self):
        self.repo = ConversationRepository()

    def create_conversation(self, title: str = "New Conversation") -> Optional[str]:
        try:
            with get_db() as db:
                if db is not None:
                    convo = self.repo.create_conversation(db, title=title)
                    return convo.id
        except Exception as e:
            logger.warning("Could not create conversation in PostgreSQL: %s", e)
        return None

    def log_message(self, convo_id: str, role: str, content: str) -> bool:
        try:
            with get_db() as db:
                if db is not None:
                    self.repo.add_message(db, convo_id, MessageCreate(role=role, content=content))
                    return True
        except Exception as e:
            logger.warning("Could not log message to PostgreSQL: %s", e)
        return False

    def get_messages(self, convo_id: str) -> List[Dict[str, str]]:
        try:
            with get_db() as db:
                if db is not None:
                    messages = self.repo.get_messages(db, convo_id)
                    return [{"role": m.role, "content": m.content} for m in messages]
        except Exception as e:
            logger.warning("Could not retrieve messages from PostgreSQL: %s", e)
        return []


conversation_service = ConversationService()
