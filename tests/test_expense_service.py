from __future__ import annotations

from datetime import date
from decimal import Decimal
import pytest

from src.services.expense_service import ExpenseService
from src.schemas.expense import ExpenseCreate


def test_expense_service_add_and_list():
    service = ExpenseService()

    res = service.add_expense(
        amount=200,
        category="shopping",
        description="snake and chips",
        expense_date="2026-09-18",
    )

    assert res["success"] is True
    exp = res["expense"]
    assert exp["amount"] == 200.0
    assert exp["category"] == "shopping"
    assert exp["date"] == "2026-09-18"

    summary_res = service.summarize_expenses()
    assert summary_res["success"] is True
    assert summary_res["summary"]["total_records"] >= 1
