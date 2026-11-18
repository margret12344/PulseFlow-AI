from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TicketCreate(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    description: str = Field(min_length=1, max_length=10000)


class TicketRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    description: str
    status: str
    priority: str | None
    created_at: datetime
    resolved_at: datetime | None


class TicketStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "Open",
        "Assigned",
        "In Progress",
        "Resolved",
        "Closed",
    ]
class TicketEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    ticket_id: UUID
    event_type: str
    old_values: dict[str, Any] | None
    new_values: dict[str, Any] | None
    created_at: datetime    