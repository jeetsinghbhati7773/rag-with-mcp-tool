from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from src.schemas.models import (
    QueryType,
    ExecutionPlan,
    Citation,
    ToolCallResult,
    ConfidenceScore,
)


class AssistantState(BaseModel):
    """
    State container tracking execution life cycle of an incoming assistant request.
    """
    query: str
    query_type: QueryType = QueryType.GENERAL
    plan: Optional[ExecutionPlan] = None
    retrieved_documents: List[Any] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    confidence: Optional[ConfidenceScore] = None
    tool_results: List[ToolCallResult] = Field(default_factory=list)
    final_answer: str = ""
    status_messages: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
