from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TicketCreate(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )
    location_id: UUID | None = None
    description: str = Field(min_length=1, max_length=10000)


class TicketRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    description: str
    status: str
    priority: str | None
    created_at: datetime
    resolved_at: datetime | None
    assigned_department_id: UUID | None
    location_id: UUID | None


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


class DepartmentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None


class TicketAssignmentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    department_id: UUID


class TokenRead(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    full_name: str
    email: str
    department_id: UUID | None
    is_active: bool


class UserCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: str = Field(min_length=1, max_length=150)
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=12, max_length=128)
    department_id: UUID | None = None
    role: Literal["requester", "support"] = "requester"


class UserActiveUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_active: bool = Field(strict=True)


class CategoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    default_department_id: UUID | None


class LocationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None


class TicketUpdate(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    description: str | None = Field(
        default=None,
        min_length=1,
        max_length=10000,
    )
    location_id: UUID | None = None

    @model_validator(mode="after")
    def validate_changes(self) -> "TicketUpdate":
        if not self.model_fields_set:
            raise ValueError("Provide at least one field to update.")

        if "description" in self.model_fields_set and self.description is None:
            raise ValueError("Description cannot be null.")

        return self


class TicketAssignmentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    ticket_id: UUID
    department_id: UUID
    assigned_by: UUID | None
    assigned_at: datetime
    ended_at: datetime | None
