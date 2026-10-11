from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Generator, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from src.llm.models import llm
from src.rag.vector_store import ChromaVectorStore
from src.rag.retriever import DocumentRetriever
from src.rag.citations import render_citations_markdown
from src.mcp.registry import global_registry
from src.mcp.executor import execute_tool_safe
from src.mcp.expense_parser import extract_expense_from_query
from src.mcp.tools import add_expense
from src.assistant.router import HybridQueryRouter, QueryRouter
from src.assistant.context import ContextManager
from src.assistant.state import AssistantState
from src.memory.summarizer import ConversationSummarizer
from src.schemas.models import Citation, ToolCallResult, QueryType, ConfidenceScore

from src.database.connection import get_db
from src.database.repositories.conversation_repository import ConversationRepository
from src.schemas.conversation import MessageCreate

logger = logging.getLogger(__name__)


def generate_conversation_title(query: str) -> str:
    """Generates a concise 3-6 word conversation title based on initial user prompt."""
    try:
        from src.llm.models import llm
        from langchain_core.messages import SystemMessage, HumanMessage
        prompt = "Create a concise, professional 3 to 6 word title summarizing the user's question topic. Return ONLY the title text, no quotes or markdown."
        res = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
        title = str(res.content).strip().strip('"')
        return title[:60] if title else query[:40]
    except Exception:
        return query[:40]


def format_expense_confirmation(expense: dict) -> str:
    """Formats a concise, clean confirmation response for expense addition."""
    amount = expense.get("amount", 0.0)
    category = expense.get("category", "General").title()
    desc = expense.get("description", "Expense").capitalize()
    raw_date = expense.get("date", "")
    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%d")
        formatted_date = dt.strftime("%d %b %Y")
    except Exception:
        formatted_date = raw_date

    return (
        f"✅ **Expense added**\n\n"
        f"**₹{amount:.0f}** · {category}\n"
        f"{desc}\n"
        f"{formatted_date}"
    )


class KnowledgeAssistant:
    """
    Master AI Knowledge Assistant orchestrating 4-Stage Hybrid Query Router, Hybrid RAG,
    MCP Tool Registry, Structured Expense Pipeline, Confidence Scorer, and LLM Generator.
    """

    def __init__(self, vectorstore: Optional[ChromaVectorStore] = None):
        self.vectorstore = vectorstore or ChromaVectorStore()
        self.retriever = DocumentRetriever(self.vectorstore)
        self.router = HybridQueryRouter()
        self.context_manager = ContextManager()
        self.summarizer = ConversationSummarizer()
        self.registry = global_registry
        self.convo_repo = ConversationRepository()

    @staticmethod
    def _recover_failed_generation(err: Exception) -> Optional[Tuple[str, dict]]:
        """
        Parses failed tool generation errors (e.g. from Groq models like openai/gpt-oss-120b)
        where the model emitted tokens like <|channel|>commentary or unparsed JSON.
        """
        err_str = str(err)
        match = re.search(r'["\']failed_generation["\']\s*:\s*["\'](\{.*?\})["\']', err_str, re.DOTALL)
        if not match:
            match = re.search(r'failed_generation[\\\'"]*:\s*[\\\'"]*(\{.*?\})[\\\'"]*', err_str, re.DOTALL)
        if match:
            try:
                raw_json = match.group(1).replace(r"\'", "'").replace(r'\"', '"').replace(r"\\", "\\")
                data = json.loads(raw_json)
                raw_name = data.get("name", "")
                clean_name = re.sub(r"<\|.*?\|>.*", "", raw_name).strip()
                args = data.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                return clean_name, args if isinstance(args, dict) else {}
            except Exception as parse_e:
                logger.debug("Failed to parse failed_generation: %s", parse_e)
        return None

    def answer_query(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        attached_files: Optional[List[Dict[str, Any]]] = None,
        conversation_id: Optional[str] = None,
        mode: str = "auto",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Processes query and yields streaming lifecycle status events and final answer payload.
        """
        attached_filenames = [
            f.get("filename") for f in (attached_files or []) if f.get("filename")
        ]
        has_docs = len(attached_filenames) > 0
        available_tools = [t.name for t in self.registry.list_tools()]

        # Auto-title conversation if needed
        if conversation_id:
            try:
                with get_db() as db:
                    if db is not None:
                        convo = self.convo_repo.get_conversation(db, conversation_id)
                        if convo and (convo.title == "New Conversation" or not convo.title):
                            new_title = generate_conversation_title(query)
                            self.convo_repo.update_title(db, conversation_id, new_title)
                        # Save User Message to DB
                        self.convo_repo.add_message(
                            db,
                            conversation_id,
                            MessageCreate(role="user", content=query),
                        )
            except Exception as e:
                logger.warning("Could not sync conversation user message to DB: %s", e)

        # Step 1: Intelligent Intent Routing
        yield {"status": "analyzing", "message": "Analyzing query intent and context with Hybrid Router..."}
        plan = self.router.route_query(
            query=query,
            attached_files=attached_files,
            chat_history=chat_history,
            vectorstore=self.vectorstore,
            available_tools=available_tools,
        )

        if mode == "document_rag" and has_docs:
            plan.query_type = QueryType.DOCUMENT
            plan.use_rag = True
            plan.use_mcp_tools = False
        elif mode == "general":
            plan.query_type = QueryType.GENERAL
            plan.use_rag = False
            plan.use_mcp_tools = True

        yield {
            "status": "routed",
            "message": f"Route: `{plan.query_type.value.upper()}` — {plan.reason}",
            "plan": plan,
        }

        # Step 2: Check for Expense Intent & Direct Structured Pipeline Execution
        q_lower = query.lower()
        if plan.query_type == QueryType.TOOL and any(k in q_lower for k in ["expense", "expens", "add ₹", "add $", "spent", "cost", "purchased", "bought"]):
            if any(k in q_lower for k in ["add", "spent", "paid", "record", "bought", "cost", "expens"]):
                yield {"status": "executing_tool", "message": "Parsing structured expense data..."}
                parsed_expense = extract_expense_from_query(query)

                if parsed_expense is None:
                    yield {
                        "status": "complete",
                        "answer": "Please specify the expense amount and description (e.g., 'Add ₹200 for shopping yesterday for snake and chips').",
                        "query_type": plan.query_type,
                        "citations": [],
                        "tool_results": [],
                    }
                    return

                yield {"status": "executing_tool", "message": f"Saving expense to local dataset (data/expenses.json)..."}
                add_tool = self.registry.get_tool("add_expense") or add_expense
                res_obj = execute_tool_safe(add_tool, parsed_expense.model_dump())

                if not res_obj.is_error and isinstance(res_obj.result, dict) and res_obj.result.get("success"):
                    final_ans = format_expense_confirmation(res_obj.result["expense"])
                else:
                    err_msg = res_obj.error_message or "Failed to record expense."
                    final_ans = f"❌ **Error adding expense:** {err_msg}"

                yield {
                    "status": "complete",
                    "answer": final_ans,
                    "query_type": plan.query_type,
                    "citations": [],
                    "tool_results": [res_obj],
                }
                return

        # Step 3: Document RAG Retrieval, Reranking & Confidence Scoring
        retrieved_docs = []
        citations: List[Citation] = []
        confidence: Optional[ConfidenceScore] = None
        context_str = ""

        if plan.use_rag:
            if not attached_filenames:
                yield {
                    "status": "complete",
                    "answer": "ℹ️ There are no documents attached to this conversation. Please use the **📎 Attach Files** button at the bottom left to attach or select a document to query.",
                    "query_type": plan.query_type,
                    "citations": [],
                    "tool_results": [],
                }
                return

            yield {"status": "retrieving", "message": f"Searching across {len(attached_filenames)} attached document(s)..."}
            retrieved_docs, citations, confidence = self.retriever.retrieve(
                query,
                conversation_id=conversation_id,
                document_names=attached_filenames,
            )
            if retrieved_docs:
                context_str = self.retriever.build_context_prompt(retrieved_docs)
                yield {
                    "status": "retrieved",
                    "message": f"Retrieved & reranked {len(retrieved_docs)} context chunk(s). Confidence: **{confidence.level}** ({confidence.score:.2f}).",
                    "citations": citations,
                    "confidence": confidence,
                }
            else:
                yield {
                    "status": "retrieved",
                    "message": f"No relevant context found in attached document(s): {', '.join(attached_filenames)}.",
                    "confidence": confidence,
                }

        # Step 4: Summarize Context if history is long
        summary_context = ""
        if len(chat_history) >= 8:
            summary_context = self.summarizer.summarize(chat_history)

        # Step 5: Build Messages with Prompt Injection Guard
        messages = self.context_manager.build_messages(
            query=query,
            chat_history=chat_history,
            document_context=context_str,
            summary_context=summary_context,
        )

        # Step 6: Tool Binding & LLM Execution
        tool_results: List[ToolCallResult] = []
        final_answer = ""
        tools_to_bind = self.registry.list_tools() if plan.use_mcp_tools else []

        if tools_to_bind:
            try:
                llm_with_tools = llm.bind_tools(tools_to_bind)
                yield {"status": "generating", "message": "Evaluating query with tool binding..."}

                response = None
                try:
                    response = llm_with_tools.invoke(messages)
                except Exception as invoke_err:
                    recovered = self._recover_failed_generation(invoke_err)
                    if recovered:
                        t_name, t_args = recovered
                        yield {"status": "executing_tool", "message": f"Executing Tool: `{t_name}` args={t_args}"}
                        target_tool = self.registry.get_tool(t_name)
                        if target_tool:
                            res_obj = execute_tool_safe(target_tool, t_args)
                            tool_results.append(res_obj)
                            yield {
                                "status": "tool_result",
                                "message": f"Tool `{t_name}` completed in {res_obj.execution_time_ms} ms.",
                                "result": res_obj.result,
                            }
                            synth_prompt = (
                                f"User question: {query}\n\n"
                                f"Tool ({t_name}) result:\n{res_obj.result}\n\n"
                                f"Please provide a direct, helpful, and natural language answer to the user based on the tool result. "
                                f"Do not call any functions or tools."
                            )
                            yield {"status": "generating", "message": "Synthesizing response from tool output..."}
                            final_res = llm.invoke([HumanMessage(content=synth_prompt)])
                            final_answer = final_res.content
                    if not final_answer:
                        raise invoke_err

                if not final_answer and response and hasattr(response, "tool_calls") and response.tool_calls:
                    messages.append(response)
                    seen_calls = set()
                    for tc in response.tool_calls:
                        raw_name = tc.get("name", "")
                        t_name = re.sub(r"<\|.*?\|>.*", "", raw_name).strip()
                        t_args = tc.get("args", {})
                        tc_id = tc.get("id")

                        call_key = (t_name, json.dumps(t_args, sort_keys=True) if isinstance(t_args, dict) else str(t_args))
                        if call_key in seen_calls:
                            continue
                        seen_calls.add(call_key)

                        yield {"status": "executing_tool", "message": f"Executing Tool: `{t_name}` args={t_args}"}

                        target_tool = self.registry.get_tool(t_name)
                        if target_tool:
                            res_obj = execute_tool_safe(target_tool, t_args)
                            tool_results.append(res_obj)
                            yield {
                                "status": "tool_result",
                                "message": f"Tool `{t_name}` completed in {res_obj.execution_time_ms} ms.",
                                "result": res_obj.result,
                            }

                            if not res_obj.is_error and t_name == "add_expense" and isinstance(res_obj.result, dict) and res_obj.result.get("success"):
                                final_answer = format_expense_confirmation(res_obj.result["expense"])
                            else:
                                if tc_id:
                                    messages.append(ToolMessage(content=str(res_obj.result), tool_call_id=tc_id))
                                else:
                                    messages.append(AIMessage(content=f"Tool `{t_name}` output: {res_obj.result}"))

                    if not final_answer and tool_results:
                        yield {"status": "generating", "message": "Synthesizing response with tool outputs..."}
                        tools_context = "\n\n".join([
                            f"Tool: {tr.tool_name}\nResult:\n{tr.result}" for tr in tool_results if not tr.is_error
                        ])
                        synth_prompt = (
                            f"User query: {query}\n\n"
                            f"Tool Results:\n{tools_context}\n\n"
                            f"Please provide a comprehensive, clear, and direct response answering the user's question using the information above. "
                            f"Do not call any additional tools."
                        )
                        try:
                            final_res = llm.invoke([HumanMessage(content=synth_prompt)])
                            final_answer = final_res.content
                        except Exception:
                            final_response = llm_with_tools.invoke(messages)
                            final_answer = final_response.content
                elif not final_answer and response:
                    final_answer = response.content

            except Exception as e:
                logger.warning("Tool-bound LLM invocation warning: %s; falling back to direct prompt", e)
                yield {"status": "generating", "message": "Generating final response..."}
                try:
                    direct_prompt = (
                        f"User query: {query}\n\n"
                        f"Please provide a direct, helpful, and natural language response to the user. "
                        f"Do not call any tools."
                    )
                    res_msg = llm.invoke([HumanMessage(content=direct_prompt)])
                    final_answer = res_msg.content
                except Exception as fallback_err:
                    logger.error("LLM fallback failed: %s", fallback_err)
                    final_answer = f"Error generating response: {str(fallback_err)}"
        else:
            yield {"status": "generating", "message": "Generating response..."}
            try:
                res_msg = llm.invoke(messages)
                final_answer = res_msg.content
            except Exception as e:
                logger.error("LLM invocation failed: %s", e)
                final_answer = f"Error generating response: {str(e)}"

        # Save Assistant Response to DB
        if conversation_id:
            try:
                with get_db() as db:
                    if db is not None:
                        conf_json = confidence.model_dump_json() if confidence else None
                        cits_json = json.dumps([c.model_dump() for c in citations]) if citations else None
                        tools_json = json.dumps([t.model_dump() for t in tool_results]) if tool_results else None
                        self.convo_repo.add_message(
                            db,
                            conversation_id,
                            MessageCreate(
                                role="assistant",
                                content=final_answer,
                                query_type=plan.query_type.value,
                                confidence_json=conf_json,
                                citations_json=cits_json,
                                tool_results_json=tools_json,
                            ),
                        )
            except Exception as e:
                logger.warning("Could not sync assistant response message to DB: %s", e)

        yield {
            "status": "complete",
            "answer": final_answer,
            "query_type": plan.query_type,
            "citations": citations,
            "confidence": confidence,
            "tool_results": tool_results,
        }
