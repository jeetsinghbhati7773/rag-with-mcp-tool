from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from src.schemas.models import ExpenseSchema
from src.llm.models import llm

logger = logging.getLogger(__name__)

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def resolve_relative_date(date_phrase: str, current_date: Optional[datetime] = None) -> str:
    """
    Resolves natural language relative date phrases to ISO YYYY-MM-DD string.
    Supports: today, yesterday, tomorrow, last week, Monday, last Monday, etc.
    """
    now = current_date or datetime.now()
    phrase = date_phrase.strip().lower()

    # Match YYYY-MM-DD directly
    if re.match(r"^\d{4}-\d{2}-\d{2}$", phrase):
        return phrase

    if "today" in phrase:
        return now.strftime("%Y-%m-%d")
    elif "yesterday" in phrase:
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    elif "tomorrow" in phrase:
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "last week" in phrase:
        return (now - timedelta(days=7)).strftime("%Y-%m-%d")

    # Match weekday names (e.g. "Monday", "last Monday")
    for idx, day_name in enumerate(WEEKDAYS):
        if day_name in phrase:
            is_last = "last" in phrase
            current_weekday = now.weekday()
            days_ago = (current_weekday - idx) % 7
            if days_ago == 0 or is_last:
                days_ago += 7
            target_date = now - timedelta(days=days_ago)
            return target_date.strftime("%Y-%m-%d")

    # Default fallback to current date
    return now.strftime("%Y-%m-%d")


def extract_expense_from_query(query: str) -> Optional[ExpenseSchema]:
    """
    Extracts structured ExpenseSchema object from natural language user query.
    Uses LLM invocation with strict JSON schema enforcing amount > 0, category, description, and date.
    Rejects incomplete queries lacking valid amount or details.
    """
    now_iso = datetime.now().strftime("%Y-%m-%d")
    now_day = datetime.now().strftime("%A")

    system_prompt = (
        f"You are a strict expense extraction parser. Today's date is {now_iso} ({now_day}).\n"
        "Extract the expense fields from the user query into JSON matching this exact structure:\n"
        "{\n"
        '  "amount": float,\n'
        '  "category": str,\n'
        '  "description": str,\n'
        '  "date": "YYYY-MM-DD"\n'
        "}\n\n"
        "Rules:\n"
        "1. amount must be a number strictly > 0.\n"
        "2. Resolve relative dates ('yesterday', 'today', 'Monday') into exact YYYY-MM-DD relative to today.\n"
        "3. If amount is missing, 0, or negative, return null.\n"
        "4. Output strictly raw JSON without markdown or explanations."
    )

    try:
        # Try Pydantic structured output if supported
        if hasattr(llm, "with_structured_output"):
            try:
                structured_llm = llm.with_structured_output(ExpenseSchema)
                res = structured_llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ])
                if isinstance(res, ExpenseSchema):
                    # Ensure date is resolved
                    res.date = resolve_relative_date(res.date)
                    return res
            except Exception as e_struct:
                logger.debug("with_structured_output fallback: %s", e_struct)

        # Fallback JSON parsing
        res = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ])
        raw_content = res.content.strip()

        # Clean markdown wrappers if present
        raw_content = re.sub(r"^```json\s*", "", raw_content)
        raw_content = re.sub(r"```$", "", raw_content).strip()

        if not raw_content or raw_content.lower() == "null":
            return None

        import json
        data = json.loads(raw_content)
        if not data or not isinstance(data, dict):
            return None

        date_val = resolve_relative_date(data.get("date", now_iso))
        data["date"] = date_val

        return ExpenseSchema(**data)

    except Exception as e:
        logger.warning("Expense parsing error for query %r: %s", query, e)
        # Try regex rule-based extraction as fallback
        return rule_based_expense_extraction(query)


def rule_based_expense_extraction(query: str) -> Optional[ExpenseSchema]:
    """Fallback rule-based regex expense extractor for standard queries."""
    q_lower = query.lower()

    # Regex for currency/amount e.g., 200, $200, ₹350
    amt_match = re.search(r"(?:[\$\₹\€\£]|rs\.?|inr)?\s*(\d+(?:\.\d{1,2})?)", q_lower)
    if not amt_match:
        return None

    try:
        amount = float(amt_match.group(1))
        if amount <= 0:
            return None
    except ValueError:
        return None

    # Date phrase extraction
    date_val = resolve_relative_date(q_lower)

    # Category heuristic
    category = "general"
    if any(w in q_lower for w in ["shopping", "clothes", "store", "snake", "chips", "food", "dinner", "lunch"]):
        category = "shopping" if "shopping" in q_lower else "food"
    elif any(w in q_lower for w in ["groceries", "supermarket"]):
        category = "groceries"
    elif any(w in q_lower for w in ["electricity", "bill", "utility"]):
        category = "utilities"
    elif any(w in q_lower for w in ["travel", "flight", "taxi", "uber"]):
        category = "travel"

    # Description heuristic
    desc_match = re.search(r"for\s+([^0-9\.\,]+)", q_lower)
    desc = desc_match.group(1).strip() if desc_match else "expense transaction"

    try:
        return ExpenseSchema(
            amount=amount,
            category=category,
            description=desc,
            date=date_val,
        )
    except Exception:
        return None
