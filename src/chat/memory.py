from __future__ import annotations

from typing import List, Dict, Any
from src.schemas.models import ChatMessage


class ConversationMemory:
    """
    Multi-turn conversation memory manager.
    Keeps conversation history distinct from persistent vector database document storage.
    """

    def __init__(self, system_prompt: str = "You are an intelligent AI Knowledge Assistant."):
        self.system_prompt = system_prompt
        self.messages: List[ChatMessage] = []

    def add_user_message(self, content: str):
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str, citations: list = None):
        self.messages.append(
            ChatMessage(role="assistant", content=content, citations=citations or [])
        )

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        history = [{"role": "system", "content": self.system_prompt}]
        for msg in self.messages:
            history.append({"role": msg.role, "content": msg.content})
        return history

    def clear(self):
        self.messages.clear()
