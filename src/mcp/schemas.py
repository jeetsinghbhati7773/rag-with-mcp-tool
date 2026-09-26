from __future__ import annotations

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class MCPToolSchema(BaseModel):
    name: str
    description: str
    args_schema: Optional[Dict[str, Any]] = None
    server_name: Optional[str] = "local"
    status: Literal["available", "unavailable", "executing", "failed"] = "available"


class MCPCallLog(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    status: str
    execution_time_ms: float
    result: Any = None
    error_reason: Optional[str] = None
