"""Tests for Math specialist."""
import pytest
from core.schemas import MathSpecialistContract
from specialists.math_specialist import MathSpecialist


@pytest.fixture
def specialist():
    return MathSpecialist()


class TestMathSpecialist:
    def test_build_prompt_includes_contract(self, specialist):
        contract = MathSpecialistContract(
            task="solve",
            domain="algebra",
            expression="x^2 - 4 = 0",
        )
        prompt = specialist.build_prompt(contract)
        assert "x^2" in prompt
        assert "algebra" in prompt
        assert "math-specialist-v1" in prompt

    def test_build_prompt_has_placeholder_instructions(self, specialist):
        contract = MathSpecialistContract(
            task="solve",
            domain="algebra",
            expression="x + 1 = 0",
        )
        prompt = specialist.build_prompt(contract)
        assert "<NL" in prompt

    def test_validate_rejects_short_output(self, specialist):
        assert not specialist.validate_output("ab")

    def test_validate_accepts_equation(self, specialist):
        assert specialist.validate_output("x = 2")

    def test_validate_accepts_solution(self, specialist):
        assert specialist.validate_output("The solution is x = 5")

    def test_run_solve_algebra(self, specialist):
        """Integration test: solve a simple algebra equation."""
        contract = MathSpecialistContract(
            task="solve",
            domain="algebra",
            expression="x^2 - 9 = 0",
            output_format="step_by_step",
        )
        output = specialist.run(contract)
        print(f"\nMath output:\n{output}")
        assert len(output) > 5
        assert "3" in output or "-3" in output or "±" in output

    def test_run_calculate(self, specialist):
        """Calculate a simple expression."""
        contract = MathSpecialistContract(
            task="calculate",
            domain="algebra",
            expression="2 + 2 * 3",
            output_format="result_only",
        )
        output = specialist.run(contract)
        print(f"\nMath output:\n{output}")
        assert len(output) > 0
