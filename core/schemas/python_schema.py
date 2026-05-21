"""Python specialist contract — schema: python-specialist-v1."""
from pydantic import BaseModel, Field
from typing import Literal, Optional
from .base import BaseContract


class InputSpec(BaseModel):
    """A named input parameter with its CDV type."""
    name: str
    type: str  # CDV type — will become Literal[...] once CDV vocabulary is built


class OutputSpec(BaseModel):
    """Expected output type."""
    type: str  # CDV type


class PythonSpecialistContract(BaseContract):
    """Contract for the Python specialist.

    All fields use CDV vocabulary — no raw English.
    """
    schema: Literal["python-specialist-v1"] = "python-specialist-v1"

    task: Literal["generate", "debug", "refactor", "explain", "optimize"] = Field(
        ..., description="Operation to perform"
    )
    intent: str = Field(
        ..., description="CDV terms describing the user's intent"
    )
    inputs: list[InputSpec] = Field(
        default_factory=list, description="Input parameter specifications"
    )
    outputs: list[OutputSpec] = Field(
        default_factory=list, description="Expected output types"
    )
    constraints: list[str] = Field(
        default_factory=list, description="CDV constraint terms"
    )
    python_version: str = Field(
        default="3.11", description="Target Python version"
    )
    context: Optional[str] = Field(
        default=None, description="Optional existing code context"
    )
