import pytest

from app.exceptions import ToolFailureError
from app.tools.base import EchoTool
from app.tools.registry import ToolRegistry


def test_register_and_get_tool():
    registry = ToolRegistry()
    registry.register(EchoTool())
    tool = registry.get("echo")
    assert tool.name == "echo"
    assert "echo" in registry.names()


def test_execute_echo_tool():
    registry = ToolRegistry()
    registry.register(EchoTool())
    result = registry.execute("echo", message="hi")
    assert result.success is True
    assert result.data == {"echo": "hi"}


def test_unknown_tool_fails_cleanly():
    registry = ToolRegistry()
    with pytest.raises(ToolFailureError) as exc:
        registry.get("search_curriculum")
    assert "Unknown tool" in exc.value.message
    assert exc.value.status_code == 502


def test_duplicate_register_fails():
    registry = ToolRegistry()
    registry.register(EchoTool())
    with pytest.raises(ToolFailureError):
        registry.register(EchoTool())
