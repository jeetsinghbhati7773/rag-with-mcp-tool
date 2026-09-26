from __future__ import annotations

from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.security.prompt_guard import wrap_untrusted_context

BASE_SYSTEM_PROMPT = (
    "You are AI Knowledge Assistant, an expert, production-grade conversational AI agent. "
    "You provide clear, accurate, technical responses with high precision. "
    "You have access to local dataset tools `add_expense` and `list_expenses` to save transactions directly into the user's dataset (`data/expenses.csv`). "
    "When the user asks to add, record, or log an expense, ALWAYS invoke the `add_expense` tool to write the record directly into the local dataset. "
    "Confirm directly to the user that the entry has been recorded in their local dataset. "
    "Be direct, concise, and present ONLY the final result output. "
    "Never state that you cannot write directly to finance apps, and never provide unsolicited tutorials for third-party apps (Google/Apple Sheets, YNAB, Mint, Excel). "
    "When grounded document context is provided, base your answer strictly on that context "
    "and cite references using [1], [2] format where appropriate. "
    "If the document context does not contain sufficient evidence to answer, "
    "explicitly state that the uploaded documents do not contain sufficient evidence."
)


class ContextManager:
    """
    Manages prompt formatting, message history preparation, and prompt injection defenses.
    """

    def __init__(self, system_prompt: str = BASE_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build_messages(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        document_context: str = "",
        summary_context: str = "",
    ) -> List[Any]:
        """
        Builds the complete ordered list of LangChain message objects for LLM invocation.
        """
        messages = []

        # System Prompt
        sys_content = self.system_prompt
        if summary_context:
            sys_content += f"\n\nPrior Conversation Summary:\n{summary_context}"
        messages.append(SystemMessage(content=sys_content))

        # Recent chat history (last 8 turns)
        for h in chat_history[-8:]:
            role = h.get("role", "user")
            content = h.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # Current user query with prompt-injection-defended context
        if document_context.strip():
            safe_context = wrap_untrusted_context(document_context)
            final_user_content = (
                f"{safe_context}\n\n"
                f"User Question:\n{query}"
            )
        else:
            final_user_content = query

        messages.append(HumanMessage(content=final_user_content))
        return messages
