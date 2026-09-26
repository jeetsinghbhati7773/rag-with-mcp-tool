from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool

from src.mcp.client import load_mcp_tools, DEFAULT_MCP_SERVERS
from src.mcp.tools import (
    get_stock_price,
    search_tool,
    add_expense,
    list_expenses,
    summarize_expenses,
    update_expense,
    delete_expense,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    First-class Tool Registry managing local tools and remote MCP server tools.
    Provides tool lookup, discovery, parameter validation, and status reporting.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self.register_defaults()

    def register_defaults(self):
        """Registers local default tools (Stock price lookup, DuckDuckGo web search, Expense Service tools)."""
        if search_tool:
            self.register_tool(search_tool)
        self.register_tool(get_stock_price)
        self.register_tool(add_expense)
        self.register_tool(list_expenses)
        self.register_tool(summarize_expenses)
        self.register_tool(update_expense)
        self.register_tool(delete_expense)

    def register_tool(self, tool_obj: BaseTool):
        """Registers a LangChain or MCP tool instance into the registry."""
        t_name = getattr(tool_obj, "name", str(tool_obj))
        self._tools[t_name] = tool_obj
        logger.debug("Registered tool in registry: %s", t_name)

    def load_remote_mcp_tools(self, server_config: Optional[Dict[str, Any]] = None) -> int:
        """Discovers and registers tools from remote MCP servers."""
        remote_tools = load_mcp_tools(server_config or DEFAULT_MCP_SERVERS)
        added_count = 0
        for t in remote_tools:
            self.register_tool(t)
            added_count += 1
        logger.info("Discovered and registered %d remote MCP tools.", added_count)
        return added_count

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Looks up a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """Returns list of all currently registered tools."""
        return list(self._tools.values())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns structured metadata list of registered tools for UI status panels."""
        schemas = []
        for name, t in self._tools.items():
            schemas.append({
                "name": name,
                "description": getattr(t, "description", ""),
                "type": "Local Tool" if name in (
                    "add_expense", "list_expenses", "summarize_expenses", "update_expense", "delete_expense", "get_stock_price", "duckduckgo_search"
                ) else "MCP Remote",
                "status": "Available",
            })
        return schemas


# Global registry instance
global_registry = ToolRegistry()
