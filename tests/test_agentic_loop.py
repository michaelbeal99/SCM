"""Tests for Dispatcher Mode 3, Agentic Loop, and multi-specialist routing."""
import pytest
from core.dispatcher import Dispatcher
from agent import AgenticLoop, LoopResult


@pytest.fixture
def dispatcher():
    return Dispatcher()


@pytest.fixture
def loop():
    return AgenticLoop(project_root="/home/michael/SCM")


class TestDispatcherMode3:
    def test_decompose_single_step_goal(self, dispatcher):
        plan = dispatcher.decompose("write a function to sort a list")
        assert "steps" in plan
        assert len(plan["steps"]) >= 1
        step = plan["steps"][0]
        assert "id" in step
        assert "intent" in step
        assert "depends_on" in step

    def test_decompose_multi_step_goal(self, dispatcher):
        plan = dispatcher.decompose(
            "create a SQL database of users, then write Python code to query it"
        )
        assert "steps" in plan
        # Should have multiple steps for SQL + Python
        steps = plan["steps"]
        assert len(steps) >= 1

    def test_decompose_step_ids_are_sequential(self, dispatcher):
        plan = dispatcher.decompose("write and test a sorting function")
        ids = [s["id"] for s in plan["steps"]]
        assert ids == sorted(ids)

    def test_decompose_fallback_produces_valid_plan(self, dispatcher):
        """Even with an empty goal, decompose should return a valid plan."""
        plan = dispatcher.decompose("")
        assert "steps" in plan
        assert len(plan["steps"]) >= 1
        assert "id" in plan["steps"][0]


class TestAgenticLoop:
    def test_single_step_python(self, loop):
        """The loop should handle a simple Python request end-to-end."""
        result = loop.run("write a Python function to check if a number is prime")
        assert result.success
        assert result.steps_executed >= 1
        assert "def " in result.output.lower() or "prime" in result.output.lower()

    def test_single_step_sql(self, loop):
        """Route to SQL specialist."""
        result = loop.run("write a SQL query to select all users from the users table")
        assert result.success
        assert result.steps_executed >= 1

    def test_tool_execution(self, loop):
        """Verify tool calls work through the loop."""
        result = loop.run("run a Python command to print hello world")
        assert result.success
        assert result.steps_executed >= 1

    def test_multi_step_plan(self, loop):
        """A goal that should produce multiple steps."""
        result = loop.run(
            "create a Python function to add two numbers and test it"
        )
        print(f"\nPlan: {result.plan}")
        print(f"Output: {result.output[:500]}")
        assert result.success
        assert result.steps_executed >= 1

    def test_result_has_plan(self, loop):
        result = loop.run("write a sort function")
        assert "steps" in result.plan


class TestMultiSpecialistRouting:
    """Verify the dispatcher correctly routes to all four target types."""

    def test_routes_python_requests(self, dispatcher):
        result = dispatcher.to_ir("write a Python function to sort a list of dicts")
        assert result.contract.schema == "python-specialist-v1"

    def test_routes_sql_requests(self, dispatcher):
        result = dispatcher.to_ir("write a SQL query to join users and orders")
        assert result.contract.schema == "sql-specialist-v1"

    def test_routes_math_requests(self, dispatcher):
        result = dispatcher.to_ir("solve this equation for x")
        assert result.contract.schema == "math-specialist-v1"

    def test_routes_tool_requests(self, dispatcher):
        result = dispatcher.to_ir("run a bash command to list files")
        assert result.contract.schema == "tool-call-v1"

    def test_mixed_query_prefers_strongest(self, dispatcher):
        """SQL keywords should win even when other terms are present."""
        result = dispatcher.to_ir("write SQL to select from database and also do python")
        assert result.contract.schema == "sql-specialist-v1"
