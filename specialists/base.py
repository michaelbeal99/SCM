"""Base class for all specialist modules.

All specialists share these properties:
- Input: Always a typed JSON contract (CDV vocabulary only)
- Output: Domain content with <NL> placeholders for English fragments
- Never process raw English directly
- Managed by Ollama (automatic load/unload)
"""

import json
from abc import ABC, abstractmethod
from typing import Any

import ollama


class SpecialistBase(ABC):
    """Abstract base for domain specialists."""

    model: str
    ollama_host: str = "http://localhost:11434"

    @abstractmethod
    def build_prompt(self, contract: Any) -> str:
        """Build a generation prompt from a typed JSON contract.

        The prompt must instruct the model to use <NL ...> placeholders
        instead of generating English strings directly.
        """
        ...

    @abstractmethod
    def validate_output(self, output: str) -> bool:
        """Check that the output contains no raw English where <NL> is expected."""
        ...

    def run(self, contract: Any) -> str:
        """Execute the specialist: build prompt → call Ollama → return output."""
        prompt = self.build_prompt(contract)
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": 0.3,
                "num_predict": 1024,
            },
        )
        return response["response"].strip()
