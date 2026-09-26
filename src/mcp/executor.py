from __future__ import annotations

import logging
import time
from typing import Any, Dict
from langchain_core.tools import BaseTool

from src.mcp.client import run_async
from src.mcp.errors import sanitize_mcp_error
from src.schemas.models import ToolCallResult

logger = logging.getLogger(__name__)


def execute_tool_safe(tool_obj: BaseTool, args: dict) -> ToolCallResult:
    """
    Safely executes a LangChain / MCP tool.
    Measures latency (execution_time_ms), handles sync and async execution paths,
    and returns structured, sanitized tool results.
    """
    tool_name = getattr(tool_obj, "name", "tool")
    start_time = time.perf_counter()

    try:
        res = tool_obj.invoke(args)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        return ToolCallResult(
            tool_name=tool_name,
            result=res,
            is_error=False,
            execution_time_ms=elapsed_ms,
        )
    except Exception as e_sync:
        logger.debug("Sync execution for %s failed (%s); trying async execution...", tool_name, e_sync)
        try:
            res = run_async(tool_obj.ainvoke(args))
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return ToolCallResult(
                tool_name=tool_name,
                result=res,
                is_error=False,
                execution_time_ms=elapsed_ms,
            )
        except Exception as e_async:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            sanitized_err = sanitize_mcp_error(e_async)
            err_msg = f"Tool '{tool_name}' failed to execute: {sanitized_err}"
            logger.error("Async execution for tool %s failed: %s", tool_name, err_msg)
            return ToolCallResult(
                tool_name=tool_name,
                result=None,
                is_error=True,
                error_message=err_msg,
                execution_time_ms=elapsed_ms,
            )
