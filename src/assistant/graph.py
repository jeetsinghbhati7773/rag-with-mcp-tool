from __future__ import annotations

import logging
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from src.schemas.models import QueryType, ExecutionPlan
from src.assistant.router import HybridQueryRouter

logger = logging.getLogger(__name__)


class AssistantWorkflowState(TypedDict, total=False):
    query: str
    attached_files: List[Dict[str, Any]]
    chat_history: List[Dict[str, str]]
    plan: Optional[ExecutionPlan]
    retrieved_docs: List[Any]
    citations: List[Any]
    confidence: Optional[Any]
    tool_results: List[Any]
    final_answer: str
    status: str


def context_manager_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    """Prepares execution context and normalizes inputs."""
    query = state.get("query", "").strip()
    attached_files = state.get("attached_files", [])
    chat_history = state.get("chat_history", [])
    return {
        "query": query,
        "attached_files": attached_files,
        "chat_history": chat_history,
        "status": "context_ready",
    }


def query_router_node(state: AssistantWorkflowState, router: HybridQueryRouter, vectorstore: Any = None) -> Dict[str, Any]:
    """Executes 4-Stage Hybrid Routing."""
    query = state.get("query", "")
    attached_files = state.get("attached_files", [])
    chat_history = state.get("chat_history", [])
    
    plan = router.route_query(
        query=query,
        attached_files=attached_files,
        chat_history=chat_history,
        vectorstore=vectorstore,
    )
    return {"plan": plan, "status": "routed"}


def route_decision_edge(state: AssistantWorkflowState) -> str:
    """Conditional edge routing to downstream nodes based on plan execution type."""
    plan = state.get("plan")
    if not plan:
        return "general_chat"
    
    if plan.query_type == QueryType.DOCUMENT or plan.use_rag:
        return "retrieve_docs"
    elif plan.query_type in (QueryType.TOOL, QueryType.WEB):
        return "mcp_tool_execution"
    elif plan.query_type == QueryType.AMBIGUOUS:
        return "clarification"
    else:
        return "general_chat"


def build_assistant_graph(router: HybridQueryRouter, vectorstore: Any = None) -> Any:
    """Builds compiled LangGraph StateGraph workflow for query processing."""
    graph = StateGraph(AssistantWorkflowState)

    graph.add_node("context_manager", context_manager_node)
    graph.add_node("query_router", lambda s: query_router_node(s, router, vectorstore))

    graph.add_edge(START, "context_manager")
    graph.add_edge("context_manager", "query_router")

    graph.add_conditional_edges(
        "query_router",
        route_decision_edge,
        {
            "retrieve_docs": END,
            "mcp_tool_execution": END,
            "general_chat": END,
            "clarification": END,
        },
    )

    return graph.compile()
