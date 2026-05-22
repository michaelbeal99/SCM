"""Tests for Dispatcher Mode 1 — English to JSON contract routing."""
import pytest
from core.dispatcher import Dispatcher, DispatchResult, _keyword_score, _best_task
from core.schemas import (
    PythonSpecialistContract,
    SQLSpecialistContract,
    MathSpecialistContract,
    ToolCallContract,
)


@pytest.fixture
def dispatcher():
    """Create a Dispatcher instance. Model calls are fallback only."""
    return Dispatcher()


class TestKeywordScoring:
    def test_python_keywords_score_high(self):
        score = _keyword_score("write a Python function to sort a list", 
                               ["python", "function", "def", "sort", "list", "code"])
        assert score >= 2

    def test_sql_keywords_score_high(self):
        score = _keyword_score("write a SQL query to join tables", 
                               ["sql", "query", "join", "table", "database"])
        assert score >= 2

    def test_no_match_returns_zero(self):
        score = _keyword_score("hello world", ["python", "sql", "math"])
        assert score == 0


class TestTaskDetection:
    def test_generate_detected(self):
        assert _best_task("write a python function") == "generate"

    def test_debug_detected(self):
        assert _best_task("fix this bug in my code") == "debug"

    def test_refactor_detected(self):
        assert _best_task("refactor this messy function") == "refactor"

    def test_optimize_detected(self):
        assert _best_task("make this loop faster and more efficient") == "optimize"

    def test_defaults_to_generate(self):
        assert _best_task("xyzzy") == "generate"


class TestDispatcherMode1:
    def test_python_request_routed_correctly(self, dispatcher):
        result = dispatcher.to_ir("write a Python function to sort a list of dicts by date")
        assert isinstance(result.contract, PythonSpecialistContract)
        assert result.contract.schema == "python-specialist-v1"
        assert result.routing_method == "rule"

    def test_sql_request_routed_correctly(self, dispatcher):
        result = dispatcher.to_ir("write a SQL query to select users from database")
        assert isinstance(result.contract, SQLSpecialistContract)
        assert result.contract.schema == "sql-specialist-v1"

    def test_math_request_routed_correctly(self, dispatcher):
        result = dispatcher.to_ir("solve this calculus equation for x")
        assert isinstance(result.contract, MathSpecialistContract)
        assert result.contract.schema == "math-specialist-v1"

    def test_tool_request_routed_correctly(self, dispatcher):
        result = dispatcher.to_ir("run a bash command to list files")
        assert isinstance(result.contract, ToolCallContract)
        assert result.contract.schema == "tool-call-v1"

    def test_result_is_serializable(self, dispatcher):
        result = dispatcher.to_ir("write a Python function to sort a list")
        json_str = result.contract.model_dump_json()
        assert "python-specialist-v1" in json_str
        # Verify we can round-trip
        c2 = PythonSpecialistContract.model_validate_json(json_str)
        assert c2.task == result.contract.task

    def test_intent_is_extracted(self, dispatcher):
        result = dispatcher.to_ir("write a Python function to filter a list of dicts by key")
        intent = result.contract.intent
        assert len(intent) > 0
        # Should contain meaningful words, not stopwords
        assert "a" not in intent.split()
        assert "to" not in intent.split()

    def test_context_is_none_by_default(self, dispatcher):
        result = dispatcher.to_ir("write a Python function")
        assert result.contract.context is None

    def test_dispatch_result_has_metadata(self, dispatcher):
        result = dispatcher.to_ir("write a Python function to sort a list of dicts")
        assert result.routing_method in ("rule", "keyword", "model")
        assert 0 <= result.confidence <= 1.0


class TestDispatcherEdgeCases:
    def test_empty_request(self, dispatcher):
        result = dispatcher.to_ir("")
        # Should default to Python specialist gracefully
        assert result.contract.schema == "python-specialist-v1"

    def test_ambiguous_request(self, dispatcher):
        """A request with no strong domain keywords should still produce a contract."""
        result = dispatcher.to_ir("help me with something")
        assert result.contract is not None
        assert result.contract.schema in (
            "python-specialist-v1", "sql-specialist-v1",
            "math-specialist-v1", "tool-call-v1",
        )

    def test_mixed_domain_prefers_strongest_signal(self, dispatcher):
        """Python keywords should win when mixed with weak signals."""
        result = dispatcher.to_ir("write a python function and also do some math")
        assert isinstance(result.contract, PythonSpecialistContract)
