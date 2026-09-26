from __future__ import annotations

import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage

from src.schemas.models import QueryType, ExecutionPlan, RouterDecision, RouteResult
from src.llm.models import llm

logger = logging.getLogger(__name__)


EXPLICIT_DOC_KEYWORDS = [
    "pdf", "docx", "txt", "doc", "file", "paper", "upload", "according to",
    "my document", "my paper", "in the document", "uploaded file", "this document",
    "the pdf", "the file", "page ", "table ", "section ", "document says",
    "what does the document", "from the file", "from the paper"
]


class HybridQueryRouter:
    """
    4-Stage Hybrid Query Router:
    1. Step 1 (No files): If no attached files / vector documents, return GENERAL_QUERY immediately (0 LLM/DB calls).
    2. Step 2 (Explicit Keyword): Check for explicit tool calls (expenses, stock, web search)
       and explicit document references.
    3. Step 3 (Semantic Relevance): Perform ChromaDB top-k lookup to check vector similarity.
    4. Step 4 (LLM Context Router): Evaluate query + conversation history + document previews for ambiguity.
    """

    def __init__(self, similarity_threshold: float = 0.60):
        self.similarity_threshold = similarity_threshold

    def route_query(
        self,
        query: str,
        attached_files: Optional[List[Dict[str, Any]]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        vectorstore: Any = None,
        available_tools: Optional[List[str]] = None,
    ) -> ExecutionPlan:
        q_lower = query.lower()
        attached_files = attached_files or []
        chat_history = chat_history or []
        
        db_count = 0
        if vectorstore is not None:
            try:
                db_count = vectorstore.count_chunks()
            except Exception:
                pass

        has_files = len(attached_files) > 0 or db_count > 0

        doc_refs = any(k in q_lower for k in EXPLICIT_DOC_KEYWORDS)
        is_expense = any(k in q_lower for k in ["expense", "expens", "add ₹", "add $", "spent", "list expenses", "my expenses", "cost", "purchased", "bought"])
        is_stock = any(k in q_lower for k in ["stock", "price of", "aapl", "tsla", "ticker", "market quote", "share price"])
        is_web = any(k in q_lower for k in ["search the web", "web search", "latest news", "current information", "recent paper", "benchmarks"])

        # -------------------------------------------------------------
        # Step 1 — Check HYBRID and tool keywords
        # -------------------------------------------------------------
        if has_files and doc_refs and (is_web or is_stock or is_expense or "compare" in q_lower):
            return ExecutionPlan(
                query_type=QueryType.HYBRID,
                reason="Explicit reference asking to compare document with web/tools.",
                use_rag=True,
                use_mcp_tools=True,
                use_web_search=is_web,
                target_tool_names=["duckduckgo_search"] if is_web else [],
            )

        if is_expense:
            return ExecutionPlan(
                query_type=QueryType.TOOL,
                reason="Query matches expense management tool intent.",
                use_mcp_tools=True,
                target_tool_names=["add_expense", "list_expenses", "expense"],
            )

        if is_stock:
            return ExecutionPlan(
                query_type=QueryType.TOOL,
                reason="Query matches stock price lookup tool intent.",
                use_mcp_tools=True,
                target_tool_names=["get_stock_price"],
            )

        if is_web:
            return ExecutionPlan(
                query_type=QueryType.WEB,
                reason="Query requests live web search or online information.",
                use_mcp_tools=True,
                use_web_search=True,
                target_tool_names=["duckduckgo_search"],
            )

        if not has_files:
            return ExecutionPlan(
                query_type=QueryType.GENERAL,
                reason="No attached files or vector documents found; defaulting to General Chat.",
                use_mcp_tools=True,
            )

        # -------------------------------------------------------------
        # Step 2 — Explicit keyword reference
        # -------------------------------------------------------------
        if doc_refs:
            return ExecutionPlan(
                query_type=QueryType.DOCUMENT,
                reason="Explicit reference to attached document detected in query.",
                use_rag=True,
            )

        # -------------------------------------------------------------
        # Step 3 — Semantic Relevance Check (ChromaDB Vector Similarity)
        # -------------------------------------------------------------
        max_similarity = 0.0
        retrieved_samples: List[Tuple[str, str, float]] = []
        if vectorstore is not None and has_files:
            try:
                results_with_score = vectorstore.similarity_search_with_score(query, k=3)
                if results_with_score:
                    for doc, score in results_with_score:
                        # Convert distance score to similarity [0, 1]
                        sim = max(0.0, 1.0 - (score if isinstance(score, (int, float)) else 0.5))
                        if sim > max_similarity:
                            max_similarity = sim
                        doc_name = doc.metadata.get("document_name") or doc.metadata.get("source") or "doc"
                        retrieved_samples.append((doc_name, doc.page_content[:200], sim))
            except Exception as e:
                logger.warning("Semantic similarity check failed in router: %s", e)

        if max_similarity >= self.similarity_threshold:
            return ExecutionPlan(
                query_type=QueryType.DOCUMENT,
                reason=f"High semantic relevance ({max_similarity:.2f}) found in attached document chunks.",
                use_rag=True,
            )

        # -------------------------------------------------------------
        # Step 4 — Conversation Context Tracking & LLM Router
        # -------------------------------------------------------------
        is_follow_up_doc = False
        if chat_history:
            recent_turns = chat_history[-4:]
            for turn in recent_turns:
                turn_content = turn.get("content", "").lower()
                if any(k in turn_content for k in ["citation", "source", "document", "page ", "pdf", "section"]):
                    is_follow_up_doc = True
                    break

        follow_up_phrases = ["explain", "example", "elaborate", "tell me more", "what about", "how so", "summarize"]
        if is_follow_up_doc and any(p in q_lower for p in follow_up_phrases):
            return ExecutionPlan(
                query_type=QueryType.DOCUMENT,
                reason="Follow-up query in an active document conversation context.",
                use_rag=True,
            )

        if max_similarity < 0.35 and not is_follow_up_doc:
            return ExecutionPlan(
                query_type=QueryType.GENERAL,
                reason=f"Query has low similarity ({max_similarity:.2f}) to attached documents; routing to General Chat.",
                use_mcp_tools=True,
            )

        return self._llm_classify(query, attached_files, chat_history, retrieved_samples)

    def _llm_classify(
        self,
        query: str,
        attached_files: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]],
        retrieved_samples: List[Tuple[str, str, float]],
    ) -> ExecutionPlan:
        """Invokes LLM for ambiguous query routing decision."""
        file_names = [f.get("filename", "") for f in attached_files if f.get("filename")]
        file_str = ", ".join(file_names) if file_names else "Attached Knowledge Base Files"

        samples_str = "\n".join([f"- Document {d}: {txt}..." for d, txt, _ in retrieved_samples[:2]]) if retrieved_samples else "None"
        recent_history = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history[-3:]]) if chat_history else "None"

        system_prompt = (
            "You are an intent router for a Knowledge Assistant. Determine if the user's query requires "
            "answering from the attached documents or general AI knowledge.\n\n"
            "Categories:\n"
            "- DOCUMENT_QUERY: The user is asking about, seeking information from, or following up on the attached documents.\n"
            "- GENERAL_QUERY: The user query is generic (coding, math, general advice, trivia) and does NOT depend on the attached documents.\n"
            "- AMBIGUOUS: The query is vague or unclear.\n\n"
            "Respond ONLY with valid JSON in this format:\n"
            '{"route": "DOCUMENT_QUERY" | "GENERAL_QUERY" | "AMBIGUOUS", "reason": "brief rationale"}'
        )

        user_content = (
            f"Attached Files: {file_str}\n"
            f"Recent Chat History:\n{recent_history}\n"
            f"Retrieved Document Snippets:\n{samples_str}\n\n"
            f"User Query: {query}"
        )

        try:
            res = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_content)])
            text = str(res.content).strip()
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                route = str(data.get("route", "")).upper()
                reason = data.get("reason", "LLM classifier decision")
                if "DOCUMENT" in route:
                    return ExecutionPlan(
                        query_type=QueryType.DOCUMENT,
                        reason=f"LLM Router: {reason}",
                        use_rag=True,
                    )
                elif "AMBIGUOUS" in route:
                    return ExecutionPlan(
                        query_type=QueryType.AMBIGUOUS,
                        reason=f"LLM Router detected ambiguous query: {reason}",
                        use_rag=True,
                    )
        except Exception as e:
            logger.warning("LLM router classification fallback: %s", e)

        return ExecutionPlan(
            query_type=QueryType.GENERAL,
            reason="Ambiguous query defaulted to General Chat.",
            use_mcp_tools=True,
        )


class QueryRouter(HybridQueryRouter):
    """Backwards-compatible QueryRouter class wrapping HybridQueryRouter."""

    def analyze_query(
        self,
        query: str,
        has_documents: bool = False,
        available_tools: Optional[List[str]] = None,
    ) -> ExecutionPlan:
        attached = [{"filename": "doc"}] if has_documents else []
        return self.route_query(
            query=query,
            attached_files=attached,
            available_tools=available_tools,
        )


def route_user_query(query: str, has_documents: bool = False, mcp_tools_available: bool = True) -> RouterDecision:
    """Legacy helper function maintaining backwards compatibility for router decision model."""
    router = QueryRouter()
    plan = router.analyze_query(query, has_documents=has_documents)

    mode_map = {
        QueryType.GENERAL: "general",
        QueryType.DOCUMENT: "document_rag",
        QueryType.WEB: "mcp_tool",
        QueryType.TOOL: "mcp_tool",
        QueryType.HYBRID: "hybrid",
        QueryType.AMBIGUOUS: "general",
    }

    t_name = plan.target_tool_names[0] if plan.target_tool_names else None
    return RouterDecision(
        mode=mode_map.get(plan.query_type, "general"),
        query_type=plan.query_type,
        reason=plan.reason,
        tool_name=t_name,
    )
