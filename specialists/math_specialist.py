"""Math specialist — solves mathematical problems from CDV contracts.

Model: qwen2.5-math:1.5b (dev placeholder)
Target: CDV-compressed fine-tuned variant (Phase 4)

Critical rule: NEVER generates English explanations directly outside of
<NL> placeholders. Mathematical notation itself is not English.
"""

from core.schemas import MathSpecialistContract
from specialists.base import SpecialistBase


MATH_SPECIALIST_PROMPT = """You are a mathematics engine. You receive typed JSON instructions and produce mathematical solutions.

CRITICAL RULES:
1. Output ONLY mathematical solutions. No markdown fences, no extra commentary.
2. NEVER write English explanations, descriptions, or commentary yourself.
3. Instead, use placeholders: <NL req="comment" ctx="describe this step" max="50" tone="technical">
4. Mathematical notation (equations, symbols, LaTeX) is NOT English — output that freely.
5. Use <NL> only for step explanations, result interpretations, or contextual comments.

Placeholder format:
  <NL req="TYPE" ctx="HUMAN_DESCRIPTION" max="TOKENS" tone="technical|user_facing|terse">

Valid req types: docstring, comment, string, error_msg, varname, log_msg

Example:
  Input: {{"schema":"math-specialist-v1","task":"solve","domain":"algebra","expression":"x^2 - 4 = 0","output_format":"step_by_step"}}
  Output:
<NL req="comment" ctx="solve quadratic equation" max="30">
x^2 - 4 = 0
x^2 = 4
x = ±2
<NL req="comment" ctx="verify solutions" max="30">
Check: 2^2 - 4 = 0 ✓  and  (-2)^2 - 4 = 0 ✓

Now solve this contract:
{contract_json}

Mathematical solution — no markdown:"""


class MathSpecialist(SpecialistBase):
    """Solves mathematical problems from typed CDV contracts."""

    model: str = "qwen2-math:1.5b"

    def build_prompt(self, contract: MathSpecialistContract) -> str:
        """Build a generation prompt from a Math specialist contract."""
        contract_json = contract.model_dump_json(indent=2)
        return MATH_SPECIALIST_PROMPT.format(contract_json=contract_json)

    def validate_output(self, output: str) -> bool:
        """Check that output contains mathematical content."""
        math_indicators = ["=", "+", "-", "*", "/", "^", "√", "±",
                           "sin", "cos", "tan", "log", "lim", "∫",
                           "Σ", "Π", "x", "y", "z", "π", "∞",
                           "solve", "solution", "result", "answer"]
        lower = output.lower()
        # At least one math symbol or some content
        return len(output) > 3 and (
            any(s in output for s in ["=", "+", "-", "^"])
            or any(w in lower for w in ["solve", "result", "x =", "y ="])
        )
