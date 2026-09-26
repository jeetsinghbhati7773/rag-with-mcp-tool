from __future__ import annotations

import logging
from typing import List, Dict
from src.llm.models import llm

logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """
    Summarizes conversation history when turn count exceeds safety thresholds,
    reducing token footprint while preserving long-term conversational context.
    """

    def summarize(self, history: List[Dict[str, str]]) -> str:
        """
        Summarizes past chat history into a concise 2-3 sentence overview.
        """
        if not history or len(history) < 6:
            return ""

        convo_text = []
        for msg in history[:-4]:  # Summarize older messages, leave recent 4 turns intact
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            convo_text.append(f"{role}: {content}")

        history_str = "\n".join(convo_text)
        prompt = (
            "Summarize the following conversation history into a concise 2-3 sentence technical context summary. "
            "Focus on topics discussed, user goals, and uploaded document references:\n\n"
            f"{history_str}"
        )

        try:
            res = llm.invoke([{"role": "user", "content": prompt}])
            return res.content.strip()
        except Exception as e:
            logger.warning("Conversation summarization warning: %s", e)
            return ""
