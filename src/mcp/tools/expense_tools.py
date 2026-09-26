from __future__ import annotations

import logging
from typing import Optional
from langchain_core.tools import tool
from src.services.expense_service import expense_service
from src.mcp.expense_parser import resolve_relative_date

logger = logging.getLogger(__name__)


@tool
def add_expense(amount: float, category: str, description: str, expense_date: str) -> dict:
    """
    Adds a new validated expense transaction to PostgreSQL database via ExpenseService.
    Inputs: amount (> 0), category, description, expense_date (YYYY-MM-DD).
    """
    iso_date = resolve_relative_date(expense_date)
    return expense_service.add_expense(
        amount=amount,
        category=category,
        description=description,
        expense_date=iso_date,
    )


@tool
def list_expenses(
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
) -> dict:
    """
    Retrieves stored expense records directly from PostgreSQL database via ExpenseService.
    """
    iso_date = resolve_relative_date(date) if date else None
    iso_start = resolve_relative_date(start_date) if start_date else None
    iso_end = resolve_relative_date(end_date) if end_date else None

    return expense_service.list_expenses(
        target_date=iso_date,
        start_date=iso_start,
        end_date=iso_end,
        category=category,
    )


@tool
def summarize_expenses(
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
) -> dict:
    """
    Calculates spending summary statistics (total spend, count, category breakdown) directly from PostgreSQL database.
    """
    iso_date = resolve_relative_date(date) if date else None
    iso_start = resolve_relative_date(start_date) if start_date else None
    iso_end = resolve_relative_date(end_date) if end_date else None

    return expense_service.summarize_expenses(
        target_date=iso_date,
        start_date=iso_start,
        end_date=iso_end,
        category=category,
    )


@tool
def update_expense(
    expense_id: str,
    amount: Optional[float] = None,
    category: Optional[str] = None,
    description: Optional[str] = None,
    expense_date: Optional[str] = None,
) -> dict:
    """
    Updates an existing expense transaction in PostgreSQL by expense_id.
    """
    iso_date = resolve_relative_date(expense_date) if expense_date else None
    return expense_service.update_expense(
        expense_id=expense_id,
        amount=amount,
        category=category,
        description=description,
        expense_date=iso_date,
    )


@tool
def delete_expense(expense_id: str) -> dict:
    """
    Deletes an expense transaction from PostgreSQL by expense_id.
    """
    return expense_service.delete_expense(expense_id=expense_id)
