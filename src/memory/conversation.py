from __future__ import annotations

from typing import List, Dict, Any, Optional
from src.schemas.models import ChatMessage
from src.memory.summarizer import ConversationSummarizer
from src.config import settings


class ConversationMemory:
    """
    Multi-turn conversation memory manager.
    Includes automated conversation summarization when message history exceeds limit.
    """

    def __init__(self, system_prompt: str = "You are an intelligent AI Knowledge Assistant."):
        self.system_prompt = system_prompt
        self.messages: List[ChatMessage] = []
        self.summary: str = ""
        self.summarizer = ConversationSummarizer()

    def add_user_message(self, content: str):
        self.messages.append(ChatMessage(role="user", content=content))
        self._check_and_summarize()

    def add_assistant_message(
        self,
        content: str,
        citations: list = None,
        query_type: Optional[Any] = None,
        confidence: Optional[Any] = None,
    ):
        self.messages.append(
            ChatMessage(
                role="assistant",
                content=content,
                citations=citations or [],
                query_type=query_type,
                confidence=confidence,
            )
        )
        self._check_and_summarize()

    def _check_and_summarize(self):
        if len(self.messages) > settings.max_chat_history:
            history_dicts = [{"role": m.role, "content": m.content} for m in self.messages]
            self.summary = self.summarizer.summarize(history_dicts)

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        history = [{"role": "system", "content": self.system_prompt}]
        for msg in self.messages:
            history.append({"role": msg.role, "content": msg.content})
        return history

    def clear(self):
        self.messages.clear()
        self.summary = ""
