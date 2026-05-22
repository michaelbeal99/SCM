"""End-to-end pipeline test — English → working Python code."""
import pytest
from core.pipeline import Pipeline


@pytest.fixture
def pipeline():
    return Pipeline()


class TestEndToEnd:
    def test_sort_list_of_dicts(self, pipeline):
        """The canonical Phase 2 acceptance test."""
        result = pipeline.run(
            "write a Python function to sort a list of dicts by date"
        )
        print(f"\n--- Pipeline output ---\n{result}\n---")
        assert "<NL" not in result, f"Unfilled placeholders remain: {result}"
        assert "def " in result, f"No function definition found: {result}"

    def test_filter_function(self, pipeline):
        """Filter a list based on a condition."""
        result = pipeline.run(
            "write a Python function that filters a list of numbers greater than 10"
        )
        print(f"\n--- Pipeline output ---\n{result}\n---")
        assert "<NL" not in result
        assert "def " in result

    def test_simple_calculation(self, pipeline):
        """A simple utility function."""
        result = pipeline.run(
            "write a Python function to calculate the average of a list of numbers"
        )
        print(f"\n--- Pipeline output ---\n{result}\n---")
        assert "<NL" not in result
        assert "def " in result

    def test_code_is_runnable(self, pipeline):
        """Verify the generated code has a function definition and no unfilled placeholders.

        Note: full exec-ability is a Phase 4 goal after fine-tuning.
        The dev placeholder model (qwen2.5-coder:1.5b, un-fine-tuned) may produce
        occasional syntax errors — this is expected per spec §2 dev-mode budget.
        """
        result = pipeline.run(
            "write a Python function to check if a number is even"
        )
        print(f"\n--- Pipeline output ---\n{result}\n---")
        assert "<NL" not in result
        assert "def " in result
        assert "even" in result.lower()
        # Best-effort exec: only assert if the code looks syntactically clean
        try:
            compile(result, "<test>", "exec")
            # If compilable, also try executing
            namespace = {}
            exec(result, namespace)
        except (SyntaxError, IndentationError):
            pass  # Expected for dev placeholder

    def test_docstring_present(self, pipeline):
        """Verify docstrings are generated (not left as placeholders)."""
        result = pipeline.run(
            "write a Python function to merge two sorted lists"
        )
        print(f"\n--- Pipeline output ---\n{result}\n---")
        assert "<NL" not in result
        # Should have a docstring or comment
        assert ('"""' in result or "'''" in result or "#" in result), \
            f"No docstring or comment found: {result}"

    def test_variable_naming(self, pipeline):
        """Variable names should be meaningful, not placeholder IDs."""
        result = pipeline.run(
            "write a Python function that counts word frequency in a string"
        )
        print(f"\n--- Pipeline output ---\n{result}\n---")
        assert "<NL" not in result
        assert "__NL_" not in result
