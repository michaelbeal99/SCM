"""Tests for SQL specialist."""
import pytest
from core.schemas import SQLSpecialistContract
from specialists.sql_specialist import SQLSpecialist


@pytest.fixture
def specialist():
    return SQLSpecialist()


class TestSQLSpecialist:
    def test_build_prompt_includes_contract(self, specialist):
        contract = SQLSpecialistContract(
            task="generate",
            intent="select join filter",
            tables=["users", "orders"],
            dialect="sqlite",
        )
        prompt = specialist.build_prompt(contract)
        assert "users" in prompt
        assert "orders" in prompt
        assert "sql-specialist-v1" in prompt

    def test_build_prompt_has_placeholder_instructions(self, specialist):
        contract = SQLSpecialistContract(task="generate", intent="select")
        prompt = specialist.build_prompt(contract)
        assert "<NL" in prompt

    def test_validate_rejects_short_output(self, specialist):
        assert not specialist.validate_output("short")

    def test_validate_accepts_select(self, specialist):
        assert specialist.validate_output("SELECT * FROM users")

    def test_validate_accepts_insert(self, specialist):
        assert specialist.validate_output("INSERT INTO users VALUES (1)")

    def test_validate_accepts_join(self, specialist):
        assert specialist.validate_output(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        )

    def test_run_generates_sql(self, specialist):
        """Integration test: call Ollama and verify SQL output."""
        contract = SQLSpecialistContract(
            task="generate",
            intent="select join filter date range",
            tables=["users", "orders"],
            dialect="sqlite",
        )
        output = specialist.run(contract)
        print(f"\nSQL output:\n{output}")
        assert len(output) > 5
        assert specialist.validate_output(output)

    def test_run_select_query(self, specialist):
        """Generate a simple SELECT."""
        contract = SQLSpecialistContract(
            task="generate",
            intent="select filter where active",
            tables=["users"],
            output_format="SELECT",
            dialect="sqlite",
        )
        output = specialist.run(contract)
        print(f"\nSQL output:\n{output}")
        assert "SELECT" in output.upper()
        assert "FROM" in output.upper()
