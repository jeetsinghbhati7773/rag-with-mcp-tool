from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    filename: str
    file_hash: str
    file_type: str
    file_size: int
    page_count: int = 1
    chunk_count: int = 0
    status: str = "indexed"


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_hash: str
    file_type: str
    file_size: int
    page_count: int
    chunk_count: int
    status: str
    uploaded_at: str

    class Config:
        from_attributes = True
