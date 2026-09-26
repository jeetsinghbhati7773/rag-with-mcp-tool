from __future__ import annotations

import logging
from typing import List, Dict, Any
from src.mcp.registry import global_registry

logger = logging.getLogger(__name__)


def discover_and_register_mcp_tools(server_config: Dict[str, Any] = None) -> int:
    """Discovers remote MCP tools and registers them in global tool registry."""
    return global_registry.load_remote_mcp_tools(server_config)
