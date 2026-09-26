from __future__ import annotations


class MCPException(Exception):
    """Base exception for MCP operations."""
    pass


class MCPConnectionError(MCPException):
    """Raised when connecting to an MCP server fails."""
    pass


class MCPToolNotFoundError(MCPException):
    """Raised when a requested tool is not registered in the registry."""
    pass


class MCPExecutionError(MCPException):
    """Raised when tool execution fails."""
    pass


def sanitize_mcp_error(error: Exception) -> str:
    """Sanitizes raw system exception into user-friendly message without leaking credentials or internal paths."""
    err_str = str(error)
    if "GROQ_API_KEY" in err_str or "OPENAI_API_KEY" in err_str:
        return "Tool execution failed: Authentication key error."
    if "Connection refused" in err_str or "timeout" in err_str.lower():
        return "MCP server connection timed out or is unavailable."
    return f"Execution error: {err_str}"
