from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ExpenseCreate(BaseModel):
    """Pydantic schema for creating a new expense record."""

    amount: Decimal = Field(..., gt=0, description="Expense amount, must be > 0")
    category: str = Field(..., min_length=1, max_length=100, description="Expense category")
    description: str = Field(..., min_length=1, max_length=500, description="Expense item description")
    expense_date: date = Field(..., description="Date of transaction (YYYY-MM-DD)")

    @field_validator("amount")
    def validate_amount(cls, v: Decimal) -> Decimal:
        if v <= Decimal("0"):
            raise ValueError("Amount must be strictly greater than 0.")
        return round(v, 2)

    @field_validator("category")
    def validate_category(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("description")
    def validate_description(cls, v: str) -> str:
        return v.strip()


class ExpenseUpdate(BaseModel):
    """Pydantic schema for updating an existing expense record."""

    amount: Optional[Decimal] = Field(None, gt=0)
    category: Optional[str] = Field(None, min_length=1)
    description: Optional[str] = Field(None, min_length=1)
    expense_date: Optional[date] = None


class ExpenseResponse(BaseModel):
    """Pydantic schema for returning expense record details."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    amount: float
    category: str
    description: str
    expense_date: str
    created_at: str
    updated_at: str


class ExpenseSummaryResponse(BaseModel):
    """Pydantic schema for returning expense summary statistics."""

    total_records: int
    total_spend: float
    category_breakdown: Dict[str, float]
    filter_applied: Dict[str, Any]
    records: List[ExpenseResponse]
