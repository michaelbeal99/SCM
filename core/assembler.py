"""Assembler — deterministic output assembly.

Substitutes NL fragment placeholders back into the output template.
No model required. Pure string substitution with validation.
"""

from .schemas import NLResponse


class Assembler:
    """Deterministic assembler that substitutes NL fragments into templates."""

    def assemble(self, template: str, fragments: dict[str, str]) -> str:
        """Substitute placeholder IDs with generated NL fragments.

        Args:
            template: Specialist output with __NL_N__ placeholders.
            fragments: Mapping of placeholder_id → generated text.

        Returns:
            Final assembled output with all placeholders filled.
        """
        result = template
        for placeholder_id, text in fragments.items():
            result = result.replace(placeholder_id, text)
        return result

    def validate(self, output: str) -> bool:
        """Check that no unreplaced placeholders remain."""
        return "__NL_" not in output
