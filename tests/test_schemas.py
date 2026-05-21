"""Tests for Pydantic schema validation and serialization."""
import pytest
from core.schemas import (
    BaseContract,
    PythonSpecialistContract,
    SQLSpecialistContract,
    MathSpecialistContract,
    ToolCallContract,
    NLRequest,
    NLResponse,
    ScanResult,
)


class TestBaseContract:
    def test_schema_field_is_required(self):
        with pytest.raises(Exception):
            BaseContract()

    def test_schema_field_serializes(self):
        c = BaseContract(schema="test-v1")
        assert c.schema == "test-v1"
        data = c.model_dump()
        assert data["schema"] == "test-v1"


class TestPythonSpecialistContract:
    def test_minimal_contract(self):
        c = PythonSpecialistContract(
            task="generate",
            intent="sort list dict",
        )
        assert c.schema == "python-specialist-v1"
        assert c.task == "generate"
        assert c.python_version == "3.11"
        assert c.inputs == []
        assert c.constraints == []

    def test_full_contract(self):
        c = PythonSpecialistContract(
            task="refactor",
            intent="optimize loop nested",
            inputs=[{"name": "data", "type": "list_dict"}],
            outputs=[{"type": "list_dict"}],
            constraints=["pure_function", "no_side_effects"],
            python_version="3.11",
            context="def process(items): pass",
        )
        assert len(c.inputs) == 1
        assert c.inputs[0].name == "data"

    def test_invalid_task_rejected(self):
        with pytest.raises(Exception):
            PythonSpecialistContract(task="invalid_task", intent="test")

    def test_round_trip_json(self):
        c = PythonSpecialistContract(
            task="debug",
            intent="null pointer dereference",
        )
        json_str = c.model_dump_json()
        c2 = PythonSpecialistContract.model_validate_json(json_str)
        assert c2.task == "debug"
        assert c2.schema == "python-specialist-v1"


class TestSQLSpecialistContract:
    def test_minimal_contract(self):
        c = SQLSpecialistContract(
            task="generate",
            intent="select join filter",
        )
        assert c.schema == "sql-specialist-v1"
        assert c.dialect == "sqlite"
        assert c.output_format == "SELECT"

    def test_with_tables(self):
        c = SQLSpecialistContract(
            task="optimize",
            intent="index scan seq",
            tables=["users", "orders"],
            dialect="postgresql",
        )
        assert "users" in c.tables


class TestMathSpecialistContract:
    def test_solve_algebra(self):
        c = MathSpecialistContract(
            task="solve",
            domain="algebra",
            expression="x^2 - 4 = 0",
        )
        assert c.schema == "math-specialist-v1"
        assert c.output_format == "step_by_step"

    def test_invalid_domain_rejected(self):
        with pytest.raises(Exception):
            MathSpecialistContract(
                task="solve",
                domain="topology",
                expression="x",
            )


class TestToolCallContract:
    def test_minimal_tool_call(self):
        c = ToolCallContract(tool="execute_python")
        assert c.schema == "tool-call-v1"
        assert c.timeout_ms == 5000
        assert c.sandbox is True

    def test_with_args(self):
        c = ToolCallContract(
            tool="read_file",
            args={"path": "/tmp/test.py"},
            timeout_ms=3000,
        )
        assert c.args["path"] == "/tmp/test.py"
        assert c.timeout_ms == 3000


class TestNLSchema:
    def test_nl_request(self):
        nl = NLRequest(
            placeholder_id="__NL_0__",
            req="docstring",
            ctx="sorts items by date",
            intent_context="sort list dict date",
        )
        assert nl.placeholder_id == "__NL_0__"
        assert nl.req == "docstring"
        assert nl.max == 50  # default

    def test_nl_response(self):
        resp = NLResponse(
            fragments={"__NL_0__": "Sort a list of items.", "__NL_1__": "items"}
        )
        assert resp.fragments["__NL_0__"] == "Sort a list of items."

    def test_scan_result(self):
        nl = NLRequest(
            placeholder_id="__NL_0__",
            req="comment",
            ctx="explain loop",
        )
        sr = ScanResult(
            template='x = [1, 2, 3]  # __NL_0__',
            requests=[nl],
        )
        assert "__NL_0__" in sr.template
        assert len(sr.requests) == 1

    def test_invalid_req_rejected(self):
        with pytest.raises(Exception):
            NLRequest(
                placeholder_id="__NL_0__",
                req="novel",
                ctx="test",
            )
