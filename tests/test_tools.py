from __future__ import annotations

import types
from src.mcp.tools import get_stock_price
from src.mcp.executor import execute_tool_safe


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_get_stock_price_demo_key_handling(monkeypatch):
    def fake_get(url, timeout=10):
        return _FakeResp({"Global Quote": {"05. price": "150.00"}})

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    res = get_stock_price.invoke({"symbol": "IBM"})
    assert res.get("Global Quote", {}).get("05. price") == "150.00"


def test_execute_tool_safe_handles_errors():
    class DummyTool:
        name = "dummy"
        def invoke(self, args):
            raise ValueError("Tool execution failed test")

    res = execute_tool_safe(DummyTool(), {})
    assert res.is_error is True
    assert "dummy" in res.error_message
