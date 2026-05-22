"""Tests for Tool Registry and built-in tools."""
import pytest
from tools.registry import ToolRegistry, ToolResult


@pytest.fixture
def registry():
    return ToolRegistry(project_root="/home/michael/SCM")


class TestToolRegistry:
    def test_all_six_tools_registered(self, registry):
        tools = registry.list_tools()
        assert "execute_python" in tools
        assert "read_file" in tools
        assert "write_file" in tools
        assert "list_directory" in tools
        assert "web_search" in tools
        assert "run_bash" in tools
        assert len(tools) == 6

    def test_unknown_tool_returns_error(self, registry):
        result = registry.execute("nonexistent", {})
        assert not result.success
        assert "Unknown" in result.error

    def test_get_tool_returns_none_for_unknown(self, registry):
        assert registry.get("nonexistent") is None

    def test_get_tool_returns_def_for_known(self, registry):
        tool = registry.get("execute_python")
        assert tool is not None
        assert tool.name == "execute_python"


class TestExecutePython:
    def test_simple_expression(self, registry):
        result = registry.execute("execute_python", {"code": "print(2 + 2)"})
        assert result.success
        assert "4" in result.data["stdout"]

    def test_failing_code(self, registry):
        result = registry.execute("execute_python", {"code": "1/0"})
        assert not result.success
        assert result.data["returncode"] != 0

    def test_timeout_is_enforced(self, registry):
        result = registry.execute("execute_python", {"code": "while True: pass"})
        assert not result.success
        assert "timed out" in result.error.lower()


class TestFileOps:
    def test_read_existing_file(self, registry):
        result = registry.execute("read_file", {"path": "requirements.txt"})
        assert result.success
        assert "pydantic" in result.data["content"]

    def test_read_nonexistent_file(self, registry):
        result = registry.execute("read_file", {"path": "nonexistent.xyz"})
        assert not result.success
        assert "not found" in result.error.lower()

    def test_write_and_read_round_trip(self, registry):
        test_content = "hello from tool test"
        write_result = registry.execute(
            "write_file",
            {"path": "/tmp/tool_test.txt", "content": test_content}
        )
        # /tmp is outside project_root, should be denied
        assert not write_result.success
        assert "denied" in write_result.error.lower()

    def test_write_inside_project(self, registry):
        result = registry.execute(
            "write_file",
            {"path": "tests/tool_test_output.txt", "content": "test123"}
        )
        assert result.success
        assert result.data["written"] == 7

    def test_list_directory(self, registry):
        result = registry.execute("list_directory", {"path": "tests"})
        assert result.success
        names = [e["name"] for e in result.data["entries"]]
        assert "test_schemas.py" in names or len(names) > 0

    def test_list_directory_denied_outside_project(self, registry):
        result = registry.execute("list_directory", {"path": "/etc"})
        assert not result.success
        assert "denied" in result.error.lower()


class TestRunBash:
    def test_allowed_command(self, registry):
        result = registry.execute("run_bash", {"command": "echo hello"})
        assert result.success
        assert "hello" in result.data["stdout"]

    def test_disallowed_command(self, registry):
        result = registry.execute("run_bash", {"command": "rm -rf /"})
        assert not result.success
        assert "not allowed" in result.error.lower()

    def test_pwd_stays_in_project(self, registry):
        result = registry.execute("run_bash", {"command": "pwd"})
        assert result.success
        assert "SCM" in result.data["stdout"]


class TestWebSearch:
    def test_web_search_stub(self, registry):
        result = registry.execute("web_search", {"query": "python testing"})
        assert result.success
        assert "stub" in result.data["message"].lower()


class TestToolResult:
    def test_result_has_duration(self, registry):
        result = registry.execute("execute_python", {"code": "print(1)"})
        assert result.duration_ms > 0

    def test_error_result_has_duration(self, registry):
        result = registry.execute("execute_python", {"code": "1/0"})
        assert result.duration_ms > 0
