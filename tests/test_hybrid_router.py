from __future__ import annotations

import pytest
from src.assistant.router import HybridQueryRouter, QueryRouter
from src.schemas.models import QueryType, ExecutionPlan


def test_hybrid_router_no_files_shortcut():
    router = HybridQueryRouter()
    plan = router.route_query(
        query="What is the capital of France?",
        attached_files=[],
        chat_history=[],
        vectorstore=None,
    )
    assert plan.query_type == QueryType.GENERAL
    assert plan.use_rag is False
    assert "No attached files" in plan.reason


def test_hybrid_router_explicit_doc_keyword():
    router = HybridQueryRouter()
    plan = router.route_query(
        query="What does page 5 of my uploaded pdf say about attention mechanism?",
        attached_files=[{"filename": "transformers.pdf"}],
        chat_history=[],
        vectorstore=None,
    )
    assert plan.query_type == QueryType.DOCUMENT
    assert plan.use_rag is True


def test_hybrid_router_general_query_with_attached_file():
    router = HybridQueryRouter()
    plan = router.route_query(
        query="Write a Python program to reverse a string.",
        attached_files=[{"filename": "Machine_Learning_Notes.pdf"}],
        chat_history=[],
        vectorstore=None,
    )
    assert plan.query_type == QueryType.GENERAL
    assert plan.use_rag is False


def test_hybrid_router_follow_up_context():
    router = HybridQueryRouter()
    chat_history = [
        {"role": "user", "content": "What is PCA in Machine_Learning_Notes.pdf?"},
        {"role": "assistant", "content": "Principal Component Analysis (PCA) is a dimensionality reduction technique based on Document citations: Machine_Learning_Notes.pdf page 3."},
    ]
    plan = router.route_query(
        query="Explain it with a detailed code example.",
        attached_files=[{"filename": "Machine_Learning_Notes.pdf"}],
        chat_history=chat_history,
        vectorstore=None,
    )
    assert plan.query_type == QueryType.DOCUMENT
    assert plan.use_rag is True


def test_hybrid_router_expense_tool():
    router = HybridQueryRouter()
    plan = router.route_query(
        query="Add $50 spent on books yesterday",
        attached_files=[],
        chat_history=[],
        vectorstore=None,
    )
    assert plan.query_type == QueryType.TOOL
    assert "expense" in plan.target_tool_names


def test_hybrid_router_stock_tool():
    router = HybridQueryRouter()
    plan = router.route_query(
        query="What is the current stock price of TSLA?",
        attached_files=[],
        chat_history=[],
        vectorstore=None,
    )
    assert plan.query_type == QueryType.TOOL
    assert "get_stock_price" in plan.target_tool_names


def test_query_router_backwards_compatibility():
    router = QueryRouter()
    plan = router.analyze_query("Explain neural networks", has_documents=False)
    assert plan.query_type == QueryType.GENERAL

    plan_doc = router.analyze_query("According to my paper, what is transformer?", has_documents=True)
    assert plan_doc.query_type == QueryType.DOCUMENT
