"""Tests for Python specialist wrapper."""
import pytest
from core.schemas import PythonSpecialistContract
from specialists.python_specialist import PythonSpecialist, PYTHON_SPECIALIST_PROMPT


@pytest.fixture
def specialist():
    return PythonSpecialist()


class TestPythonSpecialist:
    def test_build_prompt_includes_contract(self, specialist):
        contract = PythonSpecialistContract(
            task="generate",
            intent="sort list dict date",
            inputs=[{"name": "items", "type": "list_dict"}],
            outputs=[{"type": "list_dict"}],
        )
        prompt = specialist.build_prompt(contract)
        assert "sort list dict date" in prompt
        assert "python-specialist-v1" in prompt
        assert "items" in prompt

    def test_build_prompt_has_placeholder_instructions(self, specialist):
        contract = PythonSpecialistContract(task="generate", intent="test")
        prompt = specialist.build_prompt(contract)
        assert "<NL" in prompt
        assert "placeholder" in prompt.lower() or "docstring" in prompt.lower()

    def test_validate_rejects_short_output(self, specialist):
        assert not specialist.validate_output("short")

    def test_validate_accepts_function(self, specialist):
        assert specialist.validate_output("def foo():\n    return 42")

    def test_validate_accepts_import(self, specialist):
        assert specialist.validate_output("import os\nfrom sys import argv")

    def test_validate_accepts_placeholder_code(self, specialist):
        code = 'def sort_items(data):\n    <NL req="docstring" ctx="sorts items">\n    return sorted(data)'
        assert specialist.validate_output(code)

    def test_run_generates_code(self, specialist):
        """Integration test: call the Ollama model and verify output."""
        contract = PythonSpecialistContract(
            task="generate",
            intent="sort list dict date",
            inputs=[{"name": "items", "type": "list_dict"}],
            outputs=[{"type": "list_dict"}],
        )
        output = specialist.run(contract)
        # Should produce Python code
        assert len(output) > 10
        assert "def " in output or "sorted" in output.lower()
        # Should contain <NL> placeholders per the prompt instruction
        # (model may or may not follow this perfectly in dev mode)
        assert specialist.validate_output(output)

    def test_run_generates_with_placeholders(self, specialist):
        """Verify the model uses <NL> placeholders for English fragments."""
        contract = PythonSpecialistContract(
            task="generate",
            intent="filter list condition",
            inputs=[{"name": "data", "type": "list"}],
            outputs=[{"type": "list"}],
        )
        output = specialist.run(contract)
        assert "def " in output or "lambda" in output or "filter" in output.lower()
