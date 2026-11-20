from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException
from sqlalchemy import false, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from pulseflow.api.dependencies import get_current_user, get_db
from pulseflow.database.models.hospital import Role, User, UserRole
from pulseflow.database.models.tickets import Ticket


@dataclass(frozen=True)
class TicketAccess:
    user: User
    roles: frozenset[str]


def get_ticket_access(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> TicketAccess:
    try:
        roles = frozenset(
            db.scalars(
                select(Role.name)
                .join(UserRole, UserRole.role_id == Role.id)
                .where(UserRole.user_id == user.id)
            ).all()
        )
    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Authorization service is unavailable.",
        ) from None

    if not roles.intersection({"admin", "support", "requester"}):
        raise HTTPException(
            status_code=403,
            detail="Ticket access is not permitted.",
        )

    return TicketAccess(user=user, roles=roles)


def ticket_read_condition(access: TicketAccess):
    if "admin" in access.roles:
        return Ticket.id.is_not(None)

    if "support" in access.roles:
        if access.user.department_id is None:
            return false()

        return Ticket.assigned_department_id == access.user.department_id

    return Ticket.created_by == access.user.id


def require_ticket_editor(
    access: Annotated[TicketAccess, Depends(get_ticket_access)],
) -> TicketAccess:
    if not access.roles.intersection({"admin", "support"}):
        raise HTTPException(
            status_code=403,
            detail="Ticket updates require support or administrator access.",
        )

    return access
