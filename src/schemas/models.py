from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


class QueryType(str, Enum):
    GENERAL = "general"
    DOCUMENT = "document"
    WEB = "web"
    TOOL = "tool"
    HYBRID = "hybrid"
    AMBIGUOUS = "ambiguous"


class AttachedFile(BaseModel):
    file_id: str
    filename: str
    file_size_bytes: int = 0
    status: str = "ready"
    chunks_count: int = 0
    collection: Optional[str] = None


class RouteResult(BaseModel):
    route: QueryType
    confidence: float
    reason: str
    document_ids: List[str] = Field(default_factory=list)
    semantic_score: float = 0.0



class ExpenseSchema(BaseModel):
    amount: float = Field(..., gt=0, description="Expense amount (must be > 0)")
    category: str = Field(..., min_length=1, description="Expense category e.g. shopping, groceries, travel")
    description: str = Field(..., min_length=1, description="Detailed description of item or service")
    date: str = Field(..., description="ISO formatted date YYYY-MM-DD")

    @field_validator("amount")
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError("Expense amount must be strictly greater than 0.")
        return round(v, 2)


class ExpenseRecord(ExpenseSchema):
    id: str = Field(..., description="Unique UUID for expense record")
    created_at: str = Field(..., description="ISO timestamp when record was added")


class DocumentMetadata(BaseModel):
    document_id: str
    document_name: str
    file_hash: str
    page_number: Optional[int] = None
    chunk_id: str
    source: str
    created_at: Optional[str] = None
    total_pages: Optional[int] = None
    file_size_bytes: Optional[int] = None
    embedding_model: Optional[str] = "all-MiniLM-L6-v2"


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: DocumentMetadata
    score: Optional[float] = None


class Citation(BaseModel):
    document_name: str
    page_number: Optional[int] = None
    snippet: str
    chunk_id: str
    score: Optional[float] = None


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    tool_name: str
    result: Any
    is_error: bool = False
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


class ExecutionPlan(BaseModel):
    query_type: QueryType
    reason: str
    use_rag: bool = False
    use_mcp_tools: bool = False
    use_web_search: bool = False
    target_tool_names: List[str] = Field(default_factory=list)
    rewritten_query: Optional[str] = None


class RouterDecision(BaseModel):
    mode: Literal["general", "document_rag", "mcp_tool", "hybrid"]
    query_type: QueryType
    reason: str
    tool_name: Optional[str] = None
    search_query: Optional[str] = None


class ConfidenceScore(BaseModel):
    level: Literal["High confidence", "Moderate confidence", "Insufficient evidence"]
    score: float
    explanation: str


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    citations: List[Citation] = Field(default_factory=list)
    tool_calls: List[ToolCallRequest] = Field(default_factory=list)
    tool_results: List[ToolCallResult] = Field(default_factory=list)
    query_type: Optional[QueryType] = None
    confidence: Optional[ConfidenceScore] = None


class EvaluationMetrics(BaseModel):
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    hit_rate: float = 0.0
    mrr: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_relevance: float = 0.0
    citation_correctness: float = 0.0
