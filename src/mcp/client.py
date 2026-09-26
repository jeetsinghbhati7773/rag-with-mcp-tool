from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from typing import List, Dict, Any, Optional

from langchain_core.tools import BaseTool
from src.mcp.tools import get_stock_price, search_tool

logger = logging.getLogger(__name__)


def run_async(coro):
    """Safely executes an async coroutine in a dedicated thread to avoid interfering with Streamlit/Tornado event loop."""
    def _run():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        return future.result()



DEFAULT_MCP_SERVERS = {
    "expense": {
        "transport": "streamable_http",
        "url": os.getenv("FASTMCP_URL", "https://splendid-gold-dingo.fastmcp.app/mcp"),
    }
}


def load_mcp_tools(server_config: Optional[Dict[str, Any]] = None) -> List[BaseTool]:
    """
    Loads remote tools from MCP servers via MultiServerMCPClient.
    Wraps async tools with sync execution support to prevent invocation crashes.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        config = server_config or DEFAULT_MCP_SERVERS
        
        async def _fetch():
            client = MultiServerMCPClient(config)
            return await client.get_tools()

        raw_tools = run_async(_fetch())

        wrapped_tools = []
        for t in raw_tools:
            if hasattr(t, "coroutine") and t.coroutine and not getattr(t, "func", None):
                coro = t.coroutine
                t.func = lambda *args, _c=coro, **kwargs: run_async(_c(*args, **kwargs))
            wrapped_tools.append(t)
        return wrapped_tools
    except Exception as e:
        logger.warning("Could not load remote MCP tools: %s", e)
        return []


def get_all_tools(include_mcp: bool = True) -> List[BaseTool]:
    """
    Returns combined list of local tools (Stock price, DuckDuckGo) and remote MCP tools.
    """
    tools = []
    if search_tool:
        tools.append(search_tool)
    tools.append(get_stock_price)

    if include_mcp:
        mcp_tools = load_mcp_tools()
        tools.extend(mcp_tools)

    return tools
