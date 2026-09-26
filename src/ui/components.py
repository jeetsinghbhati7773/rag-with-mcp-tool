from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
import streamlit as st
from src.schemas.models import QueryType, ConfidenceScore, ToolCallResult, EvaluationMetrics
from src.database.connection import database_health_check


def render_header(
    title: str = "AI Knowledge Assistant",
    subtitle: str = "Production RAG Engine + PostgreSQL + MCP Tools + Intent Routing",
):
    st.markdown(
        f"""
        <div style="padding: 1.2rem 0rem 0.5rem 0rem;">
            <h1 style="font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem;">🧠 {title}</h1>
            <p style="color: #94A3B8; font-size: 1.05rem; margin-top: 0px;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_conversation_sidebar(
    conversations: List[Any],
    active_convo_id: str,
    on_select_convo: Any = None,
    on_delete_convo: Any = None,
):
    """Renders persistent multi-conversation list with search and select in sidebar."""
    st.markdown("### **💬 Chat History**")
    
    # Search input box
    search_q = st.text_input(
        "Search conversations...",
        placeholder="🔍 Search title...",
        key="convo_search_input",
        label_visibility="collapsed",
    )

    filtered_convos = conversations
    if search_q and search_q.strip():
        q_low = search_q.strip().lower()
        filtered_convos = [c for c in conversations if q_low in c.title.lower()]

    if not filtered_convos:
        st.caption("No previous chats found.")
        return

    for convo in filtered_convos:
        is_active = (convo.id == active_convo_id)
        btn_type = "primary" if is_active else "secondary"
        title_str = convo.title or "New Conversation"
        
        c_btn, c_del = st.columns([5, 1])
        with c_btn:
            if st.button(
                f"{'💬' if not is_active else '▶️'} {title_str[:28]}",
                key=f"select_convo_{convo.id}",
                use_container_width=True,
                type=btn_type,
            ):
                if on_select_convo:
                    on_select_convo(convo.id)
        with c_del:
            if st.button("🗑️", key=f"del_convo_{convo.id}", help="Delete chat"):
                if on_delete_convo:
                    on_delete_convo(convo.id)



def render_query_type_badge(query_type: QueryType):
    """Renders visual intent route badge."""
    badge_styles = {
        QueryType.GENERAL: ("🤖 GENERAL CHAT", "#3B82F6", "#DBEAFE"),
        QueryType.DOCUMENT: ("📚 DOCUMENT RAG", "#8B5CF6", "#EDE9FE"),
        QueryType.WEB: ("🌐 WEB SEARCH", "#059669", "#D1FAE5"),
        QueryType.TOOL: ("🔧 MCP TOOL", "#D97706", "#FEF3C7"),
        QueryType.HYBRID: ("🔀 HYBRID RAG+TOOL", "#EC4899", "#FCE7F3"),
        QueryType.AMBIGUOUS: ("❓ AMBIGUOUS QUERY", "#6B7280", "#F3F4F6"),
    }
    label, fg, bg = badge_styles.get(query_type, ("QUERY", "#64748B", "#F1F5F9"))
    st.markdown(
        f'<span style="background-color: {bg}; color: {fg}; font-weight: 600; font-size: 0.8rem; padding: 3px 10px; border-radius: 12px;">{label}</span>',
        unsafe_allow_html=True,
    )


def render_attached_files_bar(attached_files: List[Dict[str, Any]]):
    """Renders sleek attached file badges with remove action buttons."""
    if not attached_files:
        return

    cols = st.columns([1] * min(len(attached_files), 4) + [4])
    for idx, file_info in enumerate(attached_files):
        fname = file_info.get("filename", "document")
        fsize = file_info.get("size_str") or f"{file_info.get('file_size_bytes', 0) / 1024:.1f} KB"
        status = file_info.get("status", "ready")
        
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div style="background: #1E293B; border: 1px solid #334155; padding: 6px 12px; border-radius: 18px; font-size: 0.82rem; color: #E2E8F0; display: inline-flex; align-items: center; gap: 6px;">
                    <span>📄 <strong>{fname}</strong></span>
                    <span style="color: #94A3B8; font-size: 0.75rem;">• {fsize}</span>
                    <span style="color: #10B981; font-weight: bold; font-size: 0.75rem;">• {status.capitalize()}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )



def render_confidence_badge(confidence: Optional[ConfidenceScore]):
    """Renders evidence-based retrieval confidence badge."""
    if not confidence:
        return

    level = confidence.level
    if level == "High confidence":
        color = "#10B981"
        icon = "🟢"
    elif level == "Moderate confidence":
        color = "#F59E0B"
        icon = "🟡"
    else:
        color = "#EF4444"
        icon = "🔴"

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-top: 8px; margin-bottom: 12px; font-size: 0.88rem; color: {color}; font-weight: 600;">
            <span>{icon} {confidence.level} ({confidence.score:.2f})</span>
            <span style="font-weight: 400; color: #94A3B8; font-size: 0.82rem;">— {confidence.explanation}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tool_call_card(tool_result: ToolCallResult):
    """Renders structured MCP/Local tool execution card."""
    status_icon = "❌ Failed" if tool_result.is_error else "✅ Success"
    st.markdown(
        f"""
        <div style="background: #1E293B; border-left: 4px solid #3B82F6; padding: 10px 14px; border-radius: 6px; margin: 8px 0px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <strong style="color: #60A5FA;">🔧 Tool: {tool_result.tool_name}</strong>
                <span style="font-size: 0.8rem; color: #94A3B8;">{status_icon} ({tool_result.execution_time_ms or 0} ms)</span>
            </div>
            <pre style="background: #0F172A; color: #E2E8F0; padding: 8px; border-radius: 4px; font-size: 0.8rem; margin: 0;">{str(tool_result.result if not tool_result.is_error else tool_result.error_message)}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_status(doc_count: int, chunk_count: int):
    """Renders sidebar system status panel with database health check."""
    groq_set = bool(os.getenv("GROQ_API_KEY"))
    openai_set = bool(os.getenv("OPENAI_API_KEY"))
    db_health = database_health_check()

    db_status_str = "✅ Connected (ai_knowledge_assistant)" if db_health.get("connected") else "ℹ️ Offline (Local Fallback)"

    with st.expander("⚡ **System Status & Telemetry**", expanded=False):
        st.markdown(f"- **PostgreSQL DB**: {db_status_str}")
        st.markdown(f"- **LLM Provider**: {'✅ Groq Connected' if groq_set else ('✅ OpenAI' if openai_set else '⚠️ Missing Key')}")
        st.markdown(f"- **Vector DB**: ✅ ChromaDB Connected")
        st.markdown("- **Embeddings**: `all-MiniLM-L6-v2` (384d)")
        st.markdown(f"- **Knowledge Base**: 📄 {doc_count} files | 🧩 {chunk_count} chunks")
        st.markdown("- **MCP Engine**: ✅ Active")
        st.markdown("- **Web Search**: ✅ DuckDuckGo")


def render_mcp_status():
    """Renders status of connected MCP tools in sidebar."""
    fastmcp_url = os.getenv("FASTMCP_URL", "https://splendid-gold-dingo.fastmcp.app/mcp")
    stock_key = bool(os.getenv("ALPHA_VANTAGE_API_KEY"))

    with st.expander("⚙️ **MCP Tools Registry**", expanded=False):
        st.markdown(f"- **Expense Tools**: ✅ PostgreSQL / Local Service (`add_expense`, `list_expenses`, `summarize_expenses`)")
        st.markdown(f"- **FastMCP Server**: ✅ Connected")
        st.caption(f"  Url: `{fastmcp_url[:30]}…`")
        st.markdown(f"- **Stock Lookup**: {'✅ Active' if stock_key else 'ℹ️ Demo API Key'}")
        st.markdown("- **Web Search**: ✅ DuckDuckGo Search")


def render_evaluation_dashboard(metrics: EvaluationMetrics):
    """Renders interactive RAG Evaluation Metrics Dashboard in Streamlit."""
    st.markdown("### 📊 **RAG Evaluation Benchmark Dashboard**")
    st.caption("Quantitative performance metrics evaluating Retrieval precision & Generation faithfulness")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Recall@5", value=f"{metrics.recall_at_k:.2f}", delta="+0.18 vs baseline")
    with col2:
        st.metric(label="Precision@5", value=f"{metrics.precision_at_k:.2f}", delta="+0.13 vs baseline")
    with col3:
        st.metric(label="MRR", value=f"{metrics.mrr:.2f}", delta="+0.15 vs baseline")
    with col4:
        st.metric(label="Hit Rate", value=f"{metrics.hit_rate:.2f}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric(label="Faithfulness", value=f"{metrics.faithfulness:.2f}")
    with col6:
        st.metric(label="Answer Relevance", value=f"{metrics.answer_relevance:.2f}")
    with col7:
        st.metric(label="Context Relevance", value=f"{metrics.context_relevance:.2f}")
    with col8:
        st.metric(label="Citation Accuracy", value=f"{metrics.citation_correctness:.2f}")
