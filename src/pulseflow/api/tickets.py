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
    TicketAssignmentCreate,
    TicketCreate,
    TicketEventRead,
    TicketRead,
    TicketStatusUpdate,
)
from pulseflow.database.models.hospital import Department
from pulseflow.database.models.tickets import Ticket, TicketAssignment, TicketEvent

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
        statement.order_by(Ticket.created_at.desc(), Ticket.id.desc())
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
        statement = select(Ticket).where(Ticket.id == ticket_id).with_for_update()
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
                    old_resolved_at.isoformat() if old_resolved_at is not None else None
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
@router.post("/{ticket_id}/assignments", response_model=TicketRead)
def assign_ticket(
    ticket_id: UUID,
    payload: TicketAssignmentCreate,
    db: Annotated[Session, Depends(get_db)],
) -> Ticket:
    try:
        ticket = db.scalar(
            select(Ticket)
            .where(Ticket.id == ticket_id)
            .with_for_update()
        )

        if ticket is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ticket not found.",
            )

        department = db.get(Department, payload.department_id)

        if department is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Department not found.",
            )

        if ticket.status not in {"Open", "Assigned"}:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Only Open or Assigned tickets can be assigned.",
            )

        active_assignments = list(
            db.scalars(
                select(TicketAssignment).where(
                    TicketAssignment.ticket_id == ticket_id,
                    TicketAssignment.ended_at.is_(None),
                )
            ).all()
        )

        if (
            ticket.status == "Assigned"
            and ticket.assigned_department_id == department.id
            and len(active_assignments) == 1
            and active_assignments[0].department_id == department.id
        ):
            return ticket

        changed_at = datetime.now(UTC)
        old_department_id = ticket.assigned_department_id
        old_status = ticket.status

        for assignment in active_assignments:
            assignment.ended_at = changed_at

        db.add(
            TicketAssignment(
                ticket_id=ticket.id,
                department_id=department.id,
                assigned_at=changed_at,
            )
        )

        ticket.assigned_department_id = department.id
        ticket.status = "Assigned"

        db.add(
            TicketEvent(
                ticket_id=ticket.id,
                event_type="department_assigned",
                old_values={
                    "status": old_status,
                    "assigned_department_id": (
                        str(old_department_id)
                        if old_department_id is not None
                        else None
                    ),
                },
                new_values={
                    "status": ticket.status,
                    "assigned_department_id": str(department.id),
                },
            )
        )

        db.commit()

    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not assign the ticket.",
        ) from None

    return ticket