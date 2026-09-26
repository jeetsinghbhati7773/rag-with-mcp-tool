from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional

from src.database.connection import get_db, database_health_check
from src.database.repositories.expense_repository import ExpenseRepository
from src.schemas.expense import ExpenseCreate, ExpenseUpdate, ExpenseResponse, ExpenseSummaryResponse
from src.mcp.expense_store import expense_store as fallback_json_store

logger = logging.getLogger(__name__)


class ExpenseService:
    """
    Business logic and Service Layer for Expense Management.
    Handles Pydantic validation, database transactions, PostgreSQL persistence via ExpenseRepository,
    and fallback dataset integration if database is offline.
    """

    def __init__(self):
        self.repo = ExpenseRepository()

    def add_expense(
        self,
        amount: float | Decimal,
        category: str,
        description: str,
        expense_date: str | date,
    ) -> Dict[str, Any]:
        """
        Validates input schema and creates a new expense transaction.
        Saves directly to PostgreSQL when available, or fallback JSON dataset if offline.
        """
        # Convert date if string
        if isinstance(expense_date, str):
            try:
                parsed_date = datetime.strptime(expense_date.strip(), "%Y-%m-%d").date()
            except ValueError:
                parsed_date = datetime.now().date()
        else:
            parsed_date = expense_date

        try:
            expense_in = ExpenseCreate(
                amount=Decimal(str(amount)),
                category=category,
                description=description,
                expense_date=parsed_date,
            )
        except Exception as ve:
            logger.warning("Expense validation error: %s", ve)
            return {
                "success": False,
                "error": f"Validation Error: {str(ve)}",
            }

        # Try PostgreSQL Insertion
        try:
            with get_db() as db:
                if db is not None:
                    record = self.repo.create(db, expense_in)
                    return {
                        "success": True,
                        "expense": {
                            "id": record.id,
                            "amount": float(record.amount),
                            "category": record.category,
                            "description": record.description,
                            "date": record.expense_date.strftime("%Y-%m-%d"),
                            "created_at": record.created_at.isoformat(),
                        },
                        "storage": "PostgreSQL",
                    }
        except Exception as db_err:
            logger.warning("PostgreSQL transaction failed: %s; using local dataset fallback.", db_err)

        # Fallback to local JSON store
        res = fallback_json_store.add_expense(
            amount=float(expense_in.amount),
            category=expense_in.category,
            description=expense_in.description,
            date=expense_in.expense_date.strftime("%Y-%m-%d"),
        )
        res["storage"] = "Local JSON Dataset"
        return res

    def list_expenses(
        self,
        target_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieves matching expenses from PostgreSQL or local JSON fallback."""
        d_target = datetime.strptime(target_date, "%Y-%m-%d").date() if target_date else None
        d_start = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
        d_end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None

        try:
            with get_db() as db:
                if db is not None:
                    records = self.repo.get_filtered(
                        db,
                        target_date=d_target,
                        start_date=d_start,
                        end_date=d_end,
                        category=category,
                    )
                    formatted = [
                        {
                            "id": r.id,
                            "amount": float(r.amount),
                            "category": r.category,
                            "description": r.description,
                            "date": r.expense_date.strftime("%Y-%m-%d"),
                            "created_at": r.created_at.isoformat(),
                        }
                        for r in records
                    ]
                    return {
                        "success": True,
                        "count": len(formatted),
                        "expenses": formatted,
                        "storage": "PostgreSQL",
                    }
        except Exception as e:
            logger.warning("PostgreSQL query failed: %s; using fallback.", e)

        records = fallback_json_store.get_expenses(
            date=target_date,
            start_date=start_date,
            end_date=end_date,
            category=category,
        )
        return {"success": True, "count": len(records), "expenses": records, "storage": "Local JSON Dataset"}

    def update_expense(
        self,
        expense_id: str,
        amount: Optional[float] = None,
        category: Optional[str] = None,
        description: Optional[str] = None,
        expense_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Updates an expense record by ID."""
        parsed_date = datetime.strptime(expense_date, "%Y-%m-%d").date() if expense_date else None
        update_in = ExpenseUpdate(
            amount=Decimal(str(amount)) if amount is not None else None,
            category=category,
            description=description,
            expense_date=parsed_date,
        )

        try:
            with get_db() as db:
                if db is not None:
                    updated = self.repo.update(db, expense_id, update_in)
                    if updated:
                        return {
                            "success": True,
                            "expense": {
                                "id": updated.id,
                                "amount": float(updated.amount),
                                "category": updated.category,
                                "description": updated.description,
                                "date": updated.expense_date.strftime("%Y-%m-%d"),
                                "updated_at": updated.updated_at.isoformat(),
                            },
                        }
                    return {"success": False, "error": f"Expense ID '{expense_id}' not found."}
        except Exception as e:
            logger.error("PostgreSQL update failed: %s", e)
            return {"success": False, "error": str(e)}

    def delete_expense(self, expense_id: str) -> Dict[str, Any]:
        """Deletes an expense record by ID."""
        try:
            with get_db() as db:
                if db is not None:
                    success = self.repo.delete(db, expense_id)
                    if success:
                        return {"success": True, "message": f"Expense '{expense_id}' deleted from PostgreSQL."}
                    return {"success": False, "error": f"Expense ID '{expense_id}' not found."}
        except Exception as e:
            logger.error("PostgreSQL delete failed: %s", e)

        success = fallback_json_store.delete_expense(expense_id)
        if success:
            return {"success": True, "message": f"Expense '{expense_id}' deleted from JSON dataset."}
        return {"success": False, "error": f"Expense ID '{expense_id}' not found."}

    def summarize_expenses(
        self,
        target_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculates spending statistics from PostgreSQL or local JSON dataset."""
        d_target = datetime.strptime(target_date, "%Y-%m-%d").date() if target_date else None
        d_start = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
        d_end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None

        try:
            with get_db() as db:
                if db is not None:
                    summary = self.repo.calculate_summary(
                        db,
                        target_date=d_target,
                        start_date=d_start,
                        end_date=d_end,
                        category=category,
                    )
                    formatted_records = [
                        {
                            "id": r.id,
                            "amount": float(r.amount),
                            "category": r.category,
                            "description": r.description,
                            "date": r.expense_date.strftime("%Y-%m-%d"),
                        }
                        for r in summary["records"]
                    ]
                    return {
                        "success": True,
                        "summary": {
                            "total_records": summary["total_records"],
                            "total_spend": summary["total_spend"],
                            "category_breakdown": summary["category_breakdown"],
                            "records": formatted_records,
                            "storage": "PostgreSQL",
                        },
                    }
        except Exception as e:
            logger.warning("PostgreSQL summarize failed: %s; using JSON fallback.", e)

        summary = fallback_json_store.summarize_expenses(
            date=target_date,
            start_date=start_date,
            end_date=end_date,
            category=category,
        )
        return {"success": True, "summary": summary}


# Global expense service instance
expense_service = ExpenseService()
