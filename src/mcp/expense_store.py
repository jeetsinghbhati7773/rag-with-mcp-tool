from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.schemas.models import ExpenseSchema, ExpenseRecord

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
EXPENSES_JSON_FILE = DATA_DIR / "expenses.json"


class ExpenseStore:
    """
    Local JSON Dataset persistence layer for Expense Management.
    Handles file I/O, validation, UUID generation, filtering, and summarization.
    """

    def __init__(self, file_path: Path = EXPENSES_JSON_FILE):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def load_expenses(self) -> List[Dict[str, Any]]:
        """Loads all recorded expense items from data/expenses.json."""
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error loading dataset expenses from %s: %s", self.file_path, e)
            return []

    def save_expenses(self, expenses: List[Dict[str, Any]]) -> bool:
        """Saves expenses list atomically to data/expenses.json."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(expenses, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error("Error saving dataset expenses to %s: %s", self.file_path, e)
            return False

    def add_expense(
        self,
        amount: float,
        category: str,
        description: str,
        date: str,
    ) -> Dict[str, Any]:
        """
        Validates input, generates UUID & created_at timestamp, and appends expense record.
        Returns result payload: {"success": True, "expense": {...}}
        """
        # Validate schema via Pydantic model
        try:
            validated = ExpenseSchema(
                amount=amount,
                category=category.strip().lower(),
                description=description.strip(),
                date=date.strip(),
            )
        except Exception as ve:
            logger.warning("Expense validation failed: %s", ve)
            return {
                "success": False,
                "error": f"Validation Error: {str(ve)}",
            }

        new_record = ExpenseRecord(
            id=str(uuid.uuid4()),
            amount=validated.amount,
            category=validated.category,
            description=validated.description,
            date=validated.date,
            created_at=datetime.now().isoformat(),
        ).model_dump()

        expenses = self.load_expenses()
        expenses.append(new_record)

        if self.save_expenses(expenses):
            logger.info("Successfully added expense record %s to %s", new_record["id"], self.file_path)
            return {
                "success": True,
                "expense": new_record,
            }
        else:
            return {
                "success": False,
                "error": "Failed to write record to local dataset file.",
            }

    def get_expenses(
        self,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves and filters recorded expenses from dataset."""
        expenses = self.load_expenses()
        filtered = []

        for exp in expenses:
            exp_date = exp.get("date", "")
            exp_cat = exp.get("category", "").lower()

            if date and exp_date != date:
                continue
            if start_date and exp_date < start_date:
                continue
            if end_date and exp_date > end_date:
                continue
            if category and category.lower() not in exp_cat:
                continue

            filtered.append(exp)

        return filtered

    def delete_expense(self, expense_id: str) -> bool:
        """Deletes an expense record by its unique UUID."""
        expenses = self.load_expenses()
        initial_len = len(expenses)
        updated = [e for e in expenses if e.get("id") != expense_id]

        if len(updated) < initial_len:
            self.save_expenses(updated)
            return True
        return False

    def summarize_expenses(
        self,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculates aggregate statistics from the dataset (total spend, count, category breakdown).
        """
        filtered = self.get_expenses(
            date=date,
            start_date=start_date,
            end_date=end_date,
            category=category,
        )

        total_amount = sum(e.get("amount", 0.0) for e in filtered)
        category_breakdown: Dict[str, float] = {}

        for e in filtered:
            cat = e.get("category", "uncategorized").title()
            category_breakdown[cat] = round(category_breakdown.get(cat, 0.0) + e.get("amount", 0.0), 2)

        return {
            "total_records": len(filtered),
            "total_spend": round(total_amount, 2),
            "category_breakdown": category_breakdown,
            "filter_applied": {
                "date": date,
                "start_date": start_date,
                "end_date": end_date,
                "category": category,
            },
            "records": filtered,
        }


# Global expense store instance
expense_store = ExpenseStore()
