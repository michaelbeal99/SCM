"""SQL specialist — generates SQL queries from CDV contracts.

Model: qwen2.5-coder:1.5b (dev placeholder)
Target: CDV-compressed fine-tuned variant (Phase 4)

Critical rule: NEVER generates English strings, comments, or error messages
directly. All English fragments MUST be <NL ...> placeholders.
"""

from core.schemas import SQLSpecialistContract
from specialists.base import SpecialistBase


SQL_SPECIALIST_PROMPT = """You are a SQL query generator. You receive typed JSON instructions and produce only SQL code.

CRITICAL RULES:
1. Output ONLY SQL. No explanations, no markdown fences.
2. NEVER write English comments or descriptions yourself.
3. Instead, use placeholders: <NL req="comment" ctx="describe query purpose" max="50" tone="technical">
4. Use placeholders for: query comments, column descriptions, error messages

Placeholder format:
  <NL req="TYPE" ctx="HUMAN_DESCRIPTION" max="TOKENS" tone="technical|user_facing|terse">

Valid req types: docstring, comment, string, error_msg, varname, log_msg

Example:
  Input: {{"schema":"sql-specialist-v1","task":"generate","intent":"select join filter date","tables":["users","orders"],"dialect":"sqlite"}}
  Output:
<NL req="comment" ctx="selects users with recent orders" max="30">
SELECT u.id, u.name, o.total
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.order_date > <NL req="string" ctx="date threshold value" max="10" tone="terse">

Now generate SQL for this contract:
{contract_json}

SQL only — no markdown:"""


class SQLSpecialist(SpecialistBase):
    """Generates SQL queries from typed CDV contracts."""

    model: str = "qwen2.5-coder:1.5b"

    def build_prompt(self, contract: SQLSpecialistContract) -> str:
        """Build a generation prompt from a SQL specialist contract."""
        contract_json = contract.model_dump_json(indent=2)
        return SQL_SPECIALIST_PROMPT.format(contract_json=contract_json)

    def validate_output(self, output: str) -> bool:
        """Check that output contains SQL statements."""
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE",
                        "DROP", "ALTER", "FROM", "WHERE", "JOIN", "GROUP",
                        "ORDER", "HAVING", "LIMIT", "INNER", "LEFT", "RIGHT"]
        upper = output.upper()
        return len(output) > 5 and any(kw in upper for kw in sql_keywords)
