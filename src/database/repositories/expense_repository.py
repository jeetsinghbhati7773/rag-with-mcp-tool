from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_

from src.database.models import Expense
from src.schemas.expense import ExpenseCreate, ExpenseUpdate

logger = logging.getLogger(__name__)


class ExpenseRepository:
    """
    SQLAlchemy 2.x Repository for managing Expense entity persistence in PostgreSQL.
    """

    def create(self, db: Session, expense_in: ExpenseCreate) -> Expense:
        """Inserts a new Expense record into PostgreSQL."""
        expense = Expense(
            amount=float(expense_in.amount),
            category=expense_in.category.lower(),
            description=expense_in.description,
            expense_date=expense_in.expense_date,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(expense)
        db.flush()
        return expense

    def get_by_id(self, db: Session, expense_id: str) -> Optional[Expense]:
        """Retrieves single Expense by ID."""
        stmt = select(Expense).where(Expense.id == expense_id)
        return db.execute(stmt).scalar_one_or_none()

    def get_all(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Expense]:
        """Retrieves paginated Expense records sorted descending by expense_date."""
        stmt = select(Expense).order_by(Expense.expense_date.desc(), Expense.created_at.desc()).offset(skip).limit(limit)
        return list(db.execute(stmt).scalars().all())

    def get_filtered(
        self,
        db: Session,
        target_date: Optional[date] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        category: Optional[str] = None,
    ) -> List[Expense]:
        """Queries Expense records matching given filter conditions."""
        stmt = select(Expense)
        conditions = []

        if target_date:
            conditions.append(Expense.expense_date == target_date)
        if start_date:
            conditions.append(Expense.expense_date >= start_date)
        if end_date:
            conditions.append(Expense.expense_date <= end_date)
        if category:
            conditions.append(Expense.category.ilike(f"%{category.strip()}%"))

        if conditions:
            stmt = stmt.where(and_(*conditions))

        stmt = stmt.order_by(Expense.expense_date.desc(), Expense.created_at.desc())
        return list(db.execute(stmt).scalars().all())

    def update(self, db: Session, expense_id: str, update_in: ExpenseUpdate) -> Optional[Expense]:
        """Updates fields of an existing Expense record."""
        expense = self.get_by_id(db, expense_id)
        if not expense:
            return None

        if update_in.amount is not None:
            expense.amount = float(update_in.amount)
        if update_in.category is not None:
            expense.category = update_in.category.lower()
        if update_in.description is not None:
            expense.description = update_in.description
        if update_in.expense_date is not None:
            expense.expense_date = update_in.expense_date

        expense.updated_at = datetime.utcnow()
        db.flush()
        return expense

    def delete(self, db: Session, expense_id: str) -> bool:
        """Deletes Expense record by ID."""
        expense = self.get_by_id(db, expense_id)
        if not expense:
            return False
        db.delete(expense)
        db.flush()
        return True

    def calculate_summary(
        self,
        db: Session,
        target_date: Optional[date] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculates total spend and category breakdown directly using SQL aggregates."""
        filtered_records = self.get_filtered(
            db,
            target_date=target_date,
            start_date=start_date,
            end_date=end_date,
            category=category,
        )

        total_spend = sum(float(r.amount) for r in filtered_records)
        category_breakdown: Dict[str, float] = {}

        for r in filtered_records:
            cat = r.category.title()
            category_breakdown[cat] = round(category_breakdown.get(cat, 0.0) + float(r.amount), 2)

        return {
            "total_records": len(filtered_records),
            "total_spend": round(total_spend, 2),
            "category_breakdown": category_breakdown,
            "records": filtered_records,
        }
