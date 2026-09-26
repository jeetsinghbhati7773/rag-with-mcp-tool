from __future__ import annotations

from src.assistant.router import QueryRouter
from src.schemas.models import QueryType


def test_router_detects_general_chat():
    router = QueryRouter()
    plan = router.analyze_query("What is the capital of France?", has_documents=False)
    assert plan.query_type == QueryType.GENERAL
    assert plan.use_rag is False


def test_router_detects_document_rag():
    router = QueryRouter()
    plan = router.analyze_query("What does my uploaded paper say about differential privacy?", has_documents=True)
    assert plan.query_type == QueryType.DOCUMENT
    assert plan.use_rag is True


def test_router_detects_stock_tool():
    router = QueryRouter()
    plan = router.analyze_query("What is the current stock price of AAPL?", has_documents=True)
    assert plan.query_type == QueryType.TOOL
    assert "get_stock_price" in plan.target_tool_names


def test_router_detects_expense_tool():
    router = QueryRouter()
    plan = router.analyze_query("I spent $45 on lunch today, add expense", has_documents=False)
    assert plan.query_type == QueryType.TOOL
    assert "expense" in plan.target_tool_names


def test_router_detects_web_search():
    router = QueryRouter()
    plan = router.analyze_query("Search the web for latest RAG benchmarks", has_documents=False)
    assert plan.query_type == QueryType.WEB
    assert "duckduckgo_search" in plan.target_tool_names


def test_router_detects_hybrid_mode():
    router = QueryRouter()
    plan = router.analyze_query("Compare my uploaded paper with latest web search benchmarks", has_documents=True)
    assert plan.query_type == QueryType.HYBRID
    assert plan.use_rag is True
    assert plan.use_mcp_tools is True
