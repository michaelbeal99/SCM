"""SQL specialist contract — schema: sql-specialist-v1."""
from pydantic import BaseModel, Field
from typing import Literal, Optional
from .base import BaseContract


class SQLSpecialistContract(BaseContract):
    """Contract for the SQL specialist."""
    schema: Literal["sql-specialist-v1"] = "sql-specialist-v1"

    task: Literal["generate", "optimize", "explain"] = Field(
        ..., description="Operation to perform"
    )
    intent: str = Field(
        ..., description="CDV terms describing the user's intent"
    )
    tables: list[str] = Field(
        default_factory=list, description="Table names involved"
    )
    filters: dict = Field(
        default_factory=dict, description="Filter conditions"
    )
    output_format: Literal["SELECT", "INSERT", "UPDATE", "DELETE"] = Field(
        default="SELECT", description="SQL statement type"
    )
    dialect: Literal["postgresql", "mysql", "sqlite"] = Field(
        default="sqlite", description="SQL dialect"
    )
    context: Optional[str] = Field(
        default=None, description="Optional existing schema or query context"
    )
