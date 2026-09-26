from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.database.models import Expense, Document, Conversation, Message
from src.database.repositories.expense_repository import ExpenseRepository
from src.schemas.expense import ExpenseCreate


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_expense_model_crud(in_memory_db):
    repo = ExpenseRepository()

    expense_in = ExpenseCreate(
        amount=Decimal("200.00"),
        category="shopping",
        description="snake and chips",
        expense_date=date(2026, 9, 18),
    )

    created = repo.create(in_memory_db, expense_in)
    assert created.id is not None
    assert float(created.amount) == 200.0
    assert created.category == "shopping"

    retrieved = repo.get_by_id(in_memory_db, created.id)
    assert retrieved is not None
    assert retrieved.description == "snake and chips"

    filtered = repo.get_filtered(in_memory_db, target_date=date(2026, 9, 18))
    assert len(filtered) == 1

    summary = repo.calculate_summary(in_memory_db)
    assert summary["total_records"] == 1
    assert summary["total_spend"] == 200.0
    assert summary["category_breakdown"]["Shopping"] == 200.0


def test_document_and_conversation_models(in_memory_db):
    doc = Document(
        filename="test.pdf",
        file_hash="hash123",
        file_type=".pdf",
        file_size=1024,
        page_count=5,
        chunk_count=20,
        status="indexed",
    )
    in_memory_db.add(doc)
    in_memory_db.commit()

    retrieved_doc = in_memory_db.query(Document).filter_by(file_hash="hash123").first()
    assert retrieved_doc is not None
    assert retrieved_doc.chunk_count == 20

    convo = Conversation(title="Test Convo")
    in_memory_db.add(convo)
    in_memory_db.commit()

    msg = Message(conversation_id=convo.id, role="user", content="Hello test")
    in_memory_db.add(msg)
    in_memory_db.commit()

    retrieved_msg = in_memory_db.query(Message).filter_by(conversation_id=convo.id).first()
    assert retrieved_msg is not None
    assert retrieved_msg.content == "Hello test"
