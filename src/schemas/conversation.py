from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class MessageCreate(BaseModel):
    role: str
    content: str
    query_type: Optional[str] = None
    confidence_json: Optional[str] = None
    citations_json: Optional[str] = None
    tool_results_json: Optional[str] = None


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    query_type: Optional[str] = None
    confidence_json: Optional[str] = None
    citations_json: Optional[str] = None
    tool_results_json: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True


class ConversationFileResponse(BaseModel):
    id: str
    conversation_id: str
    file_id: str
    filename: str
    file_hash: str
    file_size: int = 0
    status: str = "ready"
    uploaded_at: str

    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    title: str = "New Conversation"


class ConversationResponse(BaseModel):
    id: str
    title: str
    summary: Optional[str] = None
    created_at: str
    updated_at: str
    messages: List[MessageResponse] = Field(default_factory=list)
    files: List[ConversationFileResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True
