"""NL request/response schemas for placeholder generation."""
from pydantic import BaseModel, Field
from typing import Literal


class NLRequest(BaseModel):
    """A single natural-language fragment request extracted from a placeholder.

    Maps to <NL req=... ctx=... max=... tone=... style=...> tokens in specialist output.
    """
    placeholder_id: str = Field(
        ..., description="Stable ID for substitution (e.g. '__NL_0__')"
    )
    req: Literal["docstring", "comment", "string", "error_msg", "varname", "log_msg"] = Field(
        ..., description="Type of NL content needed"
    )
    ctx: str = Field(
        ..., description="Human description of needed content"
    )
    max: int = Field(
        default=50, ge=1, le=500, description="Maximum token count"
    )
    tone: Literal["technical", "user_facing", "terse"] = Field(
        default="technical", description="Desired tone"
    )
    style: Literal["snake_case", "title_case", "sentence"] = Field(
        default="sentence", description="Text style"
    )
    intent_context: str = Field(
        default="", description="Original user intent for context injection"
    )


class NLResponse(BaseModel):
    """Map of placeholder IDs to generated NL fragments."""
    fragments: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of placeholder_id → generated English text"
    )


class ScanResult(BaseModel):
    """Output of the Placeholder Scanner."""
    template: str = Field(
        ..., description="Specialist output with placeholders replaced by stable IDs"
    )
    requests: list[NLRequest] = Field(
        default_factory=list, description="Extracted NL requests to be fulfilled"
    )
