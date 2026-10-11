from __future__ import annotations

import os
import sys
import uuid
import json
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure rag-with-mcp root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.theme import apply_theme
from src.ui.components import (
    render_header,
    render_mcp_status,
    render_system_status,
    render_query_type_badge,
    render_confidence_badge,
    render_tool_call_card,
    render_evaluation_dashboard,
    render_attached_files_bar,
    render_conversation_sidebar,
    render_access_gate,
)
from src.rag.vector_store import ChromaVectorStore
from src.services.document_service import document_service
from src.assistant.assistant import KnowledgeAssistant
from src.evaluation.eval_runner import run_rag_evaluation
from src.schemas.models import QueryType, ConfidenceScore, Citation, ToolCallResult
from src.database.connection import get_db
from src.database.repositories.conversation_repository import ConversationRepository

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Knowledge Assistant — PostgreSQL + RAG + MCP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# -----------------------------
# Security: Access Code Gate
# -----------------------------
if not render_access_gate():
    st.stop()

convo_repo = ConversationRepository()

# Initialize Vectorstore & Assistant in session state
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = ChromaVectorStore()
if "assistant" not in st.session_state:
    st.session_state["assistant"] = KnowledgeAssistant(st.session_state["vectorstore"])
if "active_conversation_id" not in st.session_state:
    st.session_state["active_conversation_id"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "eval_metrics" not in st.session_state:
    st.session_state["eval_metrics"] = None
if "attached_files" not in st.session_state:
    st.session_state["attached_files"] = []

vectorstore: ChromaVectorStore = st.session_state["vectorstore"]
assistant: KnowledgeAssistant = st.session_state["assistant"]


# -----------------------------
# Helper: Load Active Conversation State from Database
# -----------------------------
def load_conversation_state(convo_id: str):
    st.session_state["active_conversation_id"] = convo_id
    messages_out = []
    files_out = []
    try:
        with get_db() as db:
            if db is not None:
                # Load messages
                msgs = convo_repo.get_messages(db, convo_id)
                for m in msgs:
                    # Deserialize JSON metadata
                    conf_obj = None
                    if m.confidence_json:
                        try:
                            conf_data = json.loads(m.confidence_json)
                            conf_obj = ConfidenceScore(**conf_data)
                        except Exception:
                            pass

                    cits = []
                    if m.citations_json:
                        try:
                            cits_data = json.loads(m.citations_json)
                            cits = [Citation(**c) for c in cits_data]
                        except Exception:
                            pass

                    tools = []
                    if m.tool_results_json:
                        try:
                            tools_data = json.loads(m.tool_results_json)
                            tools = [ToolCallResult(**t) for t in tools_data]
                        except Exception:
                            pass

                    messages_out.append({
                        "role": m.role,
                        "content": m.content,
                        "query_type": QueryType(m.query_type) if m.query_type else None,
                        "confidence": conf_obj,
                        "citations": cits,
                        "tool_results": tools,
                    })

                # Load conversation files
                c_files = convo_repo.get_conversation_files(db, convo_id)
                for cf in c_files:
                    files_out.append({
                        "file_id": cf.file_id,
                        "filename": cf.filename,
                        "size_str": f"{cf.file_size / 1024:.1f} KB",
                        "status": cf.status,
                    })
    except Exception as e:
        st.warning(f"Could not load conversation from database: {e}")

    st.session_state["messages"] = messages_out
    st.session_state["attached_files"] = files_out


# Initialize active conversation if None
all_convos = []
convo_to_load = None
try:
    with get_db() as db:
        if db is not None:
            all_convos = convo_repo.list_conversations(db)
            if not st.session_state["active_conversation_id"]:
                if all_convos:
                    convo_to_load = all_convos[0].id
                else:
                    new_c = convo_repo.create_conversation(db, title="New Conversation")
                    convo_to_load = new_c.id
                    all_convos = [new_c]
except Exception as e:
    if not st.session_state["active_conversation_id"]:
        st.session_state["active_conversation_id"] = str(uuid.uuid4())

if convo_to_load:
    load_conversation_state(convo_to_load)

# -----------------------------
# 2. Sidebar Navigation & Management
# -----------------------------
with st.sidebar:
    st.markdown("## **🧠 AI Knowledge Assistant**")

    has_messages = bool(st.session_state.get("messages"))
    col_new, col_clear = st.columns(2)
    with col_new:
        if st.button(
            "➕ New Chat",
            use_container_width=True,
            type="primary",
            disabled=not has_messages,
            help="Start a new conversation" if has_messages else "Current conversation is already empty",
        ):
            created_id = None
            try:
                with get_db() as db:
                    if db is not None:
                        new_c = convo_repo.create_conversation(db, title="New Conversation")
                        created_id = new_c.id
            except Exception:
                pass

            if created_id:
                load_conversation_state(created_id)
            else:
                new_id = str(uuid.uuid4())
                st.session_state["active_conversation_id"] = new_id
                st.session_state["messages"] = []
                st.session_state["attached_files"] = []
            st.rerun()

    with col_clear:
        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True,
            disabled=not has_messages,
            help="Clear current chat" if has_messages else "No messages to clear",
        ):
            st.session_state["messages"] = []
            st.session_state["attached_files"] = []
            st.rerun()

    st.divider()

    # Multi-Conversation History List with Search
    def handle_select_convo(convo_id: str):
        load_conversation_state(convo_id)
        st.rerun()

    def handle_delete_convo(convo_id: str):
        try:
            with get_db() as db:
                if db is not None:
                    convo_repo.delete_conversation(db, convo_id)
        except Exception:
            pass
        if st.session_state["active_conversation_id"] == convo_id:
            st.session_state["active_conversation_id"] = None
            st.session_state["messages"] = []
            st.session_state["attached_files"] = []
        st.rerun()

    render_conversation_sidebar(
        conversations=all_convos,
        active_convo_id=st.session_state["active_conversation_id"],
        on_select_convo=handle_select_convo,
        on_delete_convo=handle_delete_convo,
    )

    st.divider()

    # Knowledge Base Overview (PostgreSQL + ChromaDB)
    doc_count = vectorstore.count_chunks()
    doc_list = document_service.list_documents()

    expander_title = f"📂 **Global Knowledge Base ({len(doc_list)} files)**" if doc_list else "📂 **Knowledge Base**"
    with st.expander(expander_title, expanded=False):
        if doc_list:
            df = pd.DataFrame(doc_list)
            st.dataframe(df, use_container_width=True, hide_index=True)

            doc_names = [d.get("Document Name") or d.get("filename", "") for d in doc_list if (d.get("Document Name") or d.get("filename"))]
            if doc_names:
                attached_names = {f.get("filename") for f in st.session_state.get("attached_files", [])}
                attachable = [name for name in doc_names if name not in attached_names]
                if attachable:
                    st.markdown("##### 📎 Attach to Current Chat")
                    to_attach = st.selectbox("Select document:", attachable, key="sidebar_attach_doc_select")
                    if st.button(f"➕ Attach to Active Chat", use_container_width=True, key="sidebar_attach_doc_btn"):
                        active_cid = st.session_state["active_conversation_id"]
                        if active_cid:
                            try:
                                with get_db() as db:
                                    if db is not None:
                                        convo_repo.add_conversation_file(
                                            db,
                                            convo_id=active_cid,
                                            file_id=to_attach[:16],
                                            filename=to_attach,
                                            file_hash="",
                                            file_size=0,
                                            status="ready",
                                        )
                            except Exception:
                                pass
                            load_conversation_state(active_cid)
                            st.toast(f"Attached '{to_attach}' to this chat!", icon="📎")
                            st.rerun()

                st.divider()
                selected_del = st.selectbox("Select document to delete:", doc_names, key="del_doc_select")
                if st.button(f"🗑️ Delete '{selected_del}'", use_container_width=True, key="del_doc_btn"):
                    deleted_cnt = document_service.delete_document(selected_del)
                    st.toast(f"Deleted '{selected_del}' ({deleted_cnt} chunks removed).", icon="🧹")
                    st.rerun()

            st.divider()
            if st.button("🗑️ Clear Entire Vector Database", use_container_width=True, key="kb_clear_btn"):
                vectorstore.clear_documents()
                st.session_state["attached_files"] = []
                st.toast("Cleared all ChromaDB collections!", icon="🧹")
                st.rerun()

    st.divider()
    render_system_status(len(doc_list), doc_count)
    render_mcp_status()

    st.divider()
    with st.expander("📊 **RAG Evaluation Suite**", expanded=False):
        if st.button("🚀 Run RAG Benchmark Eval", use_container_width=True, key="run_eval_btn"):
            with st.spinner("Evaluating retrieval & generation metrics across benchmark dataset..."):
                metrics, details = run_rag_evaluation(assistant)
                st.session_state["eval_metrics"] = metrics
                st.success("Evaluation benchmark complete!")

        if st.session_state.get("eval_metrics"):
            m = st.session_state["eval_metrics"]
            st.caption(f"- **Recall@5**: {m.recall_at_k:.2f}")
            st.caption(f"- **Precision@5**: {m.precision_at_k:.2f}")
            st.caption(f"- **MRR**: {m.mrr:.2f}")
            st.caption(f"- **Faithfulness**: {m.faithfulness:.2f}")

    st.divider()
    if st.button("🔒 Lock Application", use_container_width=True, key="lock_app_btn", help="Lock the assistant session"):
        st.session_state["authenticated"] = False
        st.rerun()

# -----------------------------
# 3. Main Chat View
# -----------------------------
render_header()

# Display RAG Evaluation Dashboard if evaluated
if st.session_state.get("eval_metrics"):
    with st.expander("📊 **Active RAG Evaluation Metrics Dashboard**", expanded=False):
        render_evaluation_dashboard(st.session_state["eval_metrics"])
        if st.button("Hide Dashboard", key="hide_eval_btn"):
            st.session_state["eval_metrics"] = None
            st.rerun()

# -----------------------------
# 4. Display Past Chat Messages
# -----------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("query_type"):
            render_query_type_badge(msg["query_type"])

        st.markdown(msg["content"])

        if msg.get("confidence"):
            render_confidence_badge(msg["confidence"])

        if msg.get("tool_results"):
            # Deduplicate results if any exist
            unique_trs = []
            seen_keys = set()
            for tr in msg["tool_results"]:
                tname = getattr(tr, "tool_name", "") or (tr.get("tool_name") if isinstance(tr, dict) else "")
                tval = str(getattr(tr, "result", "") or (tr.get("result") if isinstance(tr, dict) else ""))
                key = (tname, tval)
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_trs.append(tr)

            if unique_trs:
                with st.expander("🛠️ View Tool Details", expanded=False):
                    for tr in unique_trs:
                        render_tool_call_card(tr)

        if msg.get("citations"):
            with st.expander("📄 Source Citations & References"):
                for c in msg["citations"]:
                    page_str = f" (Page {c.page_number})" if c.page_number is not None else ""
                    score_str = f" [Score: {c.score:.2f}]" if c.get("score") is not None else ""
                    st.caption(f"- **{c.document_name}**{page_str}{score_str}: *\"{c.snippet}\"*")

# -----------------------------
# 5. Attached Document Badges (Displayed above input only when files are attached)
# -----------------------------
if st.session_state["attached_files"]:
    st.markdown("<br>", unsafe_allow_html=True)
    col_chips, col_clr = st.columns([5, 1])
    with col_chips:
        render_attached_files_bar(st.session_state["attached_files"])
    with col_clr:
        if st.button("🗑️ Detach All", use_container_width=True, key="clear_attached_btn", help="Detach files from this chat"):
            active_cid = st.session_state["active_conversation_id"]
            if active_cid:
                try:
                    with get_db() as db:
                        if db is not None:
                            for af in st.session_state["attached_files"]:
                                convo_repo.delete_conversation_file(db, active_cid, af.get("filename", ""))
                except Exception:
                    pass
            st.session_state["attached_files"] = []
            st.toast("Detached files from this conversation.", icon="🧹")
            st.rerun()

# -----------------------------
# 6. Chat Input with Native '+' Attachment Icon (Like Gemini)
# -----------------------------
input_placeholder = (
    f"Ask about {len(st.session_state['attached_files'])} attached document(s)..."
    if st.session_state["attached_files"]
    else "Ask a question about stock prices, expenses, or attach files with +..."
)

chat_input_val = st.chat_input(
    input_placeholder,
    accept_file="multiple",
    file_type=["pdf", "txt", "md", "docx"],
)

if chat_input_val:
    user_prompt = ""
    uploaded_files = []

    if isinstance(chat_input_val, str):
        user_prompt = chat_input_val
    elif hasattr(chat_input_val, "text"):
        user_prompt = (chat_input_val.text or "").strip()
        uploaded_files = chat_input_val.files or []

    # Process and attach any uploaded files from the '+' button
    if uploaded_files:
        active_cid = st.session_state["active_conversation_id"]
        for f in uploaded_files:
            try:
                file_bytes = f.read()
                chunks_added, msg, _ = document_service.process_and_index_document(file_bytes, f.name)
                if active_cid:
                    try:
                        with get_db() as db:
                            if db is not None:
                                from src.utils.hashing import calculate_sha256
                                f_hash = calculate_sha256(file_bytes)
                                convo_repo.add_conversation_file(
                                    db,
                                    convo_id=active_cid,
                                    file_id=f_hash[:16],
                                    filename=f.name,
                                    file_hash=f_hash,
                                    file_size=len(file_bytes),
                                    status="ready",
                                )
                    except Exception:
                        pass
                st.toast(f"Attached '{f.name}' ({chunks_added} chunks)", icon="📄")
            except Exception as e:
                st.error(f"Error attaching '{f.name}': {e}")

        if active_cid:
            load_conversation_state(active_cid)

    # If user attached files without prompt text, rerun so attached file chips update
    if not user_prompt:
        st.rerun()

    prompt = user_prompt
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_container = st.empty()
        answer_container = st.empty()

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["messages"]
        ]

        active_cid = st.session_state["active_conversation_id"]
        final_payload = {}
        for event in assistant.answer_query(
            prompt,
            history,
            attached_files=st.session_state["attached_files"],
            conversation_id=active_cid,
        ):
            if event["status"] in ("analyzing", "routed", "retrieving", "retrieved", "executing_tool", "generating"):
                status_container.info(f"⏳ {event['message']}")
            elif event["status"] == "complete":
                final_payload = event

        status_container.empty()

        detected_qtype = final_payload.get("query_type", QueryType.GENERAL)
        render_query_type_badge(detected_qtype)

        final_answer = final_payload.get("answer", "No response generated.")
        answer_container.markdown(final_answer)

        confidence_obj = final_payload.get("confidence")
        if confidence_obj:
            render_confidence_badge(confidence_obj)

        tool_res_list = final_payload.get("tool_results", [])
        if tool_res_list:
            unique_trs = []
            seen_keys = set()
            for tr in tool_res_list:
                tname = getattr(tr, "tool_name", "") or (tr.get("tool_name") if isinstance(tr, dict) else "")
                tval = str(getattr(tr, "result", "") or (tr.get("result") if isinstance(tr, dict) else ""))
                key = (tname, tval)
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_trs.append(tr)

            with st.expander("🛠️ View Tool Details", expanded=False):
                for tr in unique_trs:
                    render_tool_call_card(tr)

        citations = final_payload.get("citations", [])

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": final_answer,
                "query_type": detected_qtype,
                "confidence": confidence_obj,
                "citations": citations,
                "tool_results": tool_res_list,
            }
        )
