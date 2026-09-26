from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.database.repositories.conversation_repository import ConversationRepository
from src.schemas.conversation import MessageCreate
from src.assistant.assistant import generate_conversation_title


@pytest.fixture
def db_session():
    """In-memory SQLite database session fixture for testing repositories."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_create_and_list_conversations(db_session):
    repo = ConversationRepository()
    c1 = repo.create_conversation(db_session, title="Transformers Study")
    c2 = repo.create_conversation(db_session, title="Python Decorators")

    convos = repo.list_conversations(db_session)
    assert len(convos) == 2
    titles = [c.title for c in convos]
    assert "Transformers Study" in titles
    assert "Python Decorators" in titles


def test_search_conversations(db_session):
    repo = ConversationRepository()
    repo.create_conversation(db_session, title="Transformers Self Attention")
    repo.create_conversation(db_session, title="Python Asyncio Event Loop")

    results = repo.list_conversations(db_session, search_query="transformers")
    assert len(results) == 1
    assert results[0].title == "Transformers Self Attention"


def test_add_and_get_messages(db_session):
    repo = ConversationRepository()
    c = repo.create_conversation(db_session, title="Math Q&A")

    repo.add_message(
        db_session,
        c.id,
        MessageCreate(role="user", content="What is matrix multiplication?"),
    )
    repo.add_message(
        db_session,
        c.id,
        MessageCreate(role="assistant", content="Matrix multiplication is an operation...", query_type="general"),
    )

    msgs = repo.get_messages(db_session, c.id)
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"
    assert msgs[1].query_type == "general"


def test_conversation_file_association(db_session):
    repo = ConversationRepository()
    c = repo.create_conversation(db_session, title="Paper Analysis")

    cf = repo.add_conversation_file(
        db_session,
        convo_id=c.id,
        file_id="f123",
        filename="paper.pdf",
        file_hash="hash123",
        file_size=1024,
    )
    assert cf.filename == "paper.pdf"

    files = repo.get_conversation_files(db_session, c.id)
    assert len(files) == 1
    assert files[0].filename == "paper.pdf"


def test_delete_conversation(db_session):
    repo = ConversationRepository()
    c = repo.create_conversation(db_session, title="Temporary Chat")
    repo.add_message(db_session, c.id, MessageCreate(role="user", content="Hello"))

    deleted = repo.delete_conversation(db_session, c.id)
    assert deleted is True
    assert repo.get_conversation(db_session, c.id) is None
