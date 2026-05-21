"""Tool call contract — schema: tool-call-v1."""
from pydantic import BaseModel, Field
from typing import Literal, Any


class ToolCallContract(BaseModel):
    """Contract for invoking a tool from the agentic loop."""
    model_config = {"protected_namespaces": ()}
    schema: Literal["tool-call-v1"] = "tool-call-v1"

    tool: str = Field(
        ..., description="Name of the tool to invoke"
    )
    args: dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )
    timeout_ms: int = Field(
        default=5000, ge=100, le=30000, description="Timeout in milliseconds"
    )
    sandbox: bool = Field(
        default=True, description="Whether to run in isolated sandbox"
    )
