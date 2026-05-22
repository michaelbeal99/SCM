"""Python specialist — generates Python code from CDV contracts.

Model: qwen2.5-coder:1.5b (dev placeholder)
Target: CDV-compressed fine-tuned variant (~350MB, Phase 4)

Critical rule: NEVER generates English strings, comments, docstrings,
variable names, or error messages directly. All English fragments MUST
be emitted as <NL ...> placeholders for the Dispatcher (Mode 2) to fill.
"""

from core.schemas import PythonSpecialistContract
from specialists.base import SpecialistBase


PYTHON_SPECIALIST_PROMPT = """You are a Python code generator. You receive typed JSON instructions and produce only Python code.

CRITICAL RULES:
1. Output ONLY Python code. No explanations, no markdown fences.
2. NEVER write English comments, docstrings, error messages, variable names, or log messages yourself.
3. Instead, use placeholders: <NL req="docstring" ctx="describe needed content" max="50" tone="technical">
4. Use placeholders for: docstrings, comments, error messages, user-facing strings, log messages, variable names that need English meaning

Placeholder format:
  <NL req="TYPE" ctx="HUMAN_DESCRIPTION" max="TOKENS" tone="technical|user_facing|terse" style="snake_case|title_case|sentence">

Valid req types: docstring, comment, string, error_msg, varname, log_msg

Example:
  Input: {{"task":"generate","intent":"sort list dict date","inputs":[{{"name":"items","type":"list_dict"}}],"outputs":[{{"type":"list_dict"}}]}}
  Output:
def sort_by_date(items):
    <NL req="docstring" ctx="sorts list of dicts by date key" max="30" tone="technical">
    return sorted(items, key=lambda x: x.get(<NL req="varname" ctx="date field name" max="10" tone="terse" style="snake_case">))

Now generate code for this contract:
{contract_json}

Python code only — no markdown:"""


class PythonSpecialist(SpecialistBase):
    """Generates Python code from typed CDV contracts."""

    model: str = "qwen2.5-coder:1.5b"

    def build_prompt(self, contract: PythonSpecialistContract) -> str:
        """Build a generation prompt from a Python specialist contract."""
        contract_json = contract.model_dump_json(indent=2)
        return PYTHON_SPECIALIST_PROMPT.format(contract_json=contract_json)

    def validate_output(self, output: str) -> bool:
        """Check that output contains Python code (not just English prose).

        In Phase 2, we accept any output that doesn't appear to be a refusal.
        Strict CDV validation comes in Phase 4.
        """
        # At minimum, output should contain code-like content
        return len(output) > 10 and (
            "def " in output
            or "class " in output
            or "import " in output
            or "return " in output
            or "=" in output
        )
