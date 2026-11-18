from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import pulseflow.database.models  # noqa: F401
from pulseflow.api.dependencies import get_db
from pulseflow.api.schemas import (
    TicketCreate,
    TicketEventRead,
    TicketRead,
    TicketStatusUpdate,
)
from pulseflow.database.models.tickets import Ticket, TicketEvent

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

ALLOWED_TRANSITIONS = {
    "Open": {"In Progress"},
    "Assigned": {"In Progress"},
    "In Progress": {"Resolved"},
    "Resolved": {"Closed", "Open"},
    "Closed": set(),
}


@router.patch("/{ticket_id}/status", response_model=TicketRead)
def update_ticket_status(
    ticket_id: UUID,
    payload: TicketStatusUpdate,
    db: Annotated[Session, Depends(get_db)],
) -> Ticket:
    try:
        statement = (
            select(Ticket)
            .where(Ticket.id == ticket_id)
            .with_for_update()
        )
        ticket = db.scalar(statement)

        if ticket is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ticket not found.",
            )

        old_status = ticket.status
        new_status = payload.status

        if new_status == old_status:
            return ticket

        if new_status not in ALLOWED_TRANSITIONS.get(old_status, set()):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot change status from {old_status} to {new_status}.",
            )

        old_resolved_at = ticket.resolved_at

        if new_status == "Resolved":
            ticket.resolved_at = datetime.now(UTC)
        elif new_status == "Open":
            ticket.resolved_at = None

        ticket.status = new_status

        event = TicketEvent(
            ticket_id=ticket.id,
            event_type="status_changed",
            old_values={
                "status": old_status,
                "resolved_at": (
                    old_resolved_at.isoformat()
                    if old_resolved_at is not None
                    else None
                ),
            },
            new_values={
                "status": new_status,
                "resolved_at": (
                    ticket.resolved_at.isoformat()
                    if ticket.resolved_at is not None
                    else None
                ),
            },
        )

        db.add(event)
        db.commit()

    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not update the ticket.",
        ) from None

    return ticket

@router.get(
    "/{ticket_id}/events",
    response_model=list[TicketEventRead],
)
def list_ticket_events(
    ticket_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[TicketEvent]:
    try:
        ticket = db.get(Ticket, ticket_id)

        if ticket is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ticket not found.",
            )

        statement = (
            select(TicketEvent)
            .where(TicketEvent.ticket_id == ticket_id)
            .order_by(TicketEvent.created_at.asc(), TicketEvent.id.asc())
            .offset(offset)
            .limit(limit)
        )

        return list(db.scalars(statement).all())

    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not retrieve ticket events.",
        ) from None