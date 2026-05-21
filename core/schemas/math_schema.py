"""Math specialist contract — schema: math-specialist-v1."""
from pydantic import BaseModel, Field
from typing import Literal


class MathSpecialistContract(BaseModel):
    """Contract for the Math specialist."""
    model_config = {"protected_namespaces": ()}
    schema: Literal["math-specialist-v1"] = "math-specialist-v1"

    task: Literal["solve", "prove", "simplify", "calculate"] = Field(
        ..., description="Operation to perform"
    )
    domain: Literal["algebra", "calculus", "statistics", "linear_algebra"] = Field(
        ..., description="Mathematical domain"
    )
    expression: str = Field(
        ..., description="LaTeX or symbolic notation of the problem"
    )
    output_format: Literal["step_by_step", "result_only", "code"] = Field(
        default="step_by_step", description="How to present the result"
    )
    context: str | None = Field(
        default=None, description="Optional additional context"
    )
