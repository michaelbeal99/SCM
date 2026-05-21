"""Base schema shared by all JSON contracts."""
from pydantic import BaseModel, Field
from typing import Literal


class BaseContract(BaseModel):
    """Every inter-module message carries a schema discriminator."""
    model_config = {"protected_namespaces": ()}
    schema: str = Field(..., description="Schema discriminator (e.g. 'python-specialist-v1')")


# Known schema identifiers that the Dispatcher can route to
SchemaId = Literal[
    "python-specialist-v1",
    "sql-specialist-v1",
    "math-specialist-v1",
    "tool-call-v1",
]
