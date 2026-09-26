from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
import pytest

from src.schemas.models import ExpenseSchema
from src.mcp.expense_store import ExpenseStore
from src.mcp.expense_parser import resolve_relative_date, extract_expense_from_query
from src.mcp.tools import add_expense, list_expenses, summarize_expenses


@pytest.fixture
def temp_store(tmp_path):
    json_file = tmp_path / "expenses.json"
    return ExpenseStore(file_path=json_file)


def test_relative_date_resolution():
    now = datetime(2026, 9, 19)
    assert resolve_relative_date("today", current_date=now) == "2026-09-19"
    assert resolve_relative_date("yesterday", current_date=now) == "2026-09-18"
    assert resolve_relative_date("tomorrow", current_date=now) == "2026-09-20"
    assert resolve_relative_date("2026-09-18", current_date=now) == "2026-09-18"


def test_pydantic_validation_valid():
    exp = ExpenseSchema(amount=200, category="shopping", description="snake and chips", date="2026-09-18")
    assert exp.amount == 200.0
    assert exp.category == "shopping"
    assert exp.description == "snake and chips"
    assert exp.date == "2026-09-18"


def test_pydantic_validation_invalid_amount():
    with pytest.raises(ValueError):
        ExpenseSchema(amount=-50, category="food", description="lunch", date="2026-09-18")

    with pytest.raises(ValueError):
        ExpenseSchema(amount=0, category="food", description="lunch", date="2026-09-18")


def test_expense_store_add_and_persistence(temp_store):
    res = temp_store.add_expense(amount=200, category="shopping", description="snake and chips", date="2026-09-18")
    assert res["success"] is True
    record = res["expense"]
    assert record["amount"] == 200.0
    assert record["category"] == "shopping"
    assert "id" in record
    assert "created_at" in record

    # Verify JSON file on disk
    loaded = temp_store.load_expenses()
    assert len(loaded) == 1
    assert loaded[0]["id"] == record["id"]


def test_expense_store_filtering_and_summarization(temp_store):
    temp_store.add_expense(amount=200, category="shopping", description="snake and chips", date="2026-09-18")
    temp_store.add_expense(amount=500, category="groceries", description="weekly groceries", date="2026-09-19")

    # Filter by date
    d18 = temp_store.get_expenses(date="2026-09-18")
    assert len(d18) == 1
    assert d18[0]["amount"] == 200.0

    # Summarize
    summary = temp_store.summarize_expenses()
    assert summary["total_records"] == 2
    assert summary["total_spend"] == 700.0
    assert summary["category_breakdown"]["Shopping"] == 200.0
    assert summary["category_breakdown"]["Groceries"] == 500.0


def test_rule_based_expense_extraction():
    query = "add an expense of 200 for shopping yesterday for snake and chips"
    parsed = extract_expense_from_query(query)
    assert parsed is not None
    assert parsed.amount == 200.0
    assert parsed.category == "shopping"
    assert "snake" in parsed.description.lower() or "chips" in parsed.description.lower()
