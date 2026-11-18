from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import pulseflow.database.models  # noqa: F401
from pulseflow.api.dependencies import get_db
from pulseflow.api.schemas import TicketCreate, TicketRead
from pulseflow.database.models.tickets import Ticket

router = APIRouter(prefix="/tickets", tags=["Tickets"])


@router.post(
    "",
    response_model=TicketRead,
    status_code=status.HTTP_201_CREATED,
)
def create_ticket(
    payload: TicketCreate,
    db: Annotated[Session, Depends(get_db)],
) -> Ticket:
    ticket = Ticket(description=payload.description)

    try:
        db.add(ticket)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not save the ticket.",
        ) from None

    return ticket
@router.get("", response_model=list[TicketRead])
def list_tickets(
    db: Annotated[Session, Depends(get_db)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ticket_status: Annotated[
        str | None,
        Query(alias="status", min_length=1, max_length=30),
    ] = None,
) -> list[Ticket]:
    statement = select(Ticket)

    if ticket_status is not None:
        statement = statement.where(Ticket.status == ticket_status)

    statement = (
        statement
        .order_by(Ticket.created_at.desc(), Ticket.id.desc())
        .offset(offset)
        .limit(limit)
    )

    try:
        return list(db.scalars(statement).all())
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not retrieve tickets.",
        ) from None
@router.get("/{ticket_id}", response_model=TicketRead)
def get_ticket(
    ticket_id: UUID,
    db: Annotated[Session, Depends(get_db)],
) -> Ticket:
    try:
        ticket = db.get(Ticket, ticket_id)
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not retrieve the ticket.",
        ) from None

    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ticket not found.",
        )

    return ticket