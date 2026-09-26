from __future__ import annotations

import logging
import os
import requests
from langchain_core.tools import tool

from src.mcp.expense_tools import (
    add_expense,
    list_expenses,
    summarize_expenses,
    update_expense,
    delete_expense,
)

logger = logging.getLogger(__name__)

# Search tool loader
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun(region="us-en")
except Exception as e:
    logger.warning("DuckDuckGo search tool unavailable: %s", e)
    search_tool = None


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given ticker symbol (e.g. 'AAPL', 'TSLA', 'IBM')
    using Alpha Vantage. Uses environment variable ALPHA_VANTAGE_API_KEY (falls back to 'demo').
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if isinstance(data, dict) and not data.get("Global Quote") and (
            data.get("Note") or data.get("Information") or data.get("Error Message")
        ):
            return {
                "error": (
                    "Alpha Vantage rate limit reached or missing API key. "
                    "Set ALPHA_VANTAGE_API_KEY in .env."
                ),
                "raw": data,
            }
        return data
    except Exception as e:
        logger.warning("Stock price lookup failed for %r: %s", symbol, e)
        return {"error": str(e)}


__all__ = [
    "get_stock_price",
    "search_tool",
    "add_expense",
    "list_expenses",
    "summarize_expenses",
    "update_expense",
    "delete_expense",
]
