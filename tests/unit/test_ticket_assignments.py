from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from pulseflow.api.dependencies import get_db
from pulseflow.api.tickets import router
from pulseflow.database.models.hospital import Department
from pulseflow.database.models.tickets import Ticket, TicketAssignment, TicketEvent


@pytest.fixture
def assignment_case():
    ticket = Ticket(
        id=uuid4(),
        description="Synthetic printer issue.",
        status="Open",
        priority=None,
        created_at=datetime.now(UTC),
        resolved_at=None,
        assigned_department_id=None,
    )
    department = Department(id=uuid4(), name="IT Support")

    db = MagicMock(spec=Session)
    db.scalar.return_value = ticket
    db.get.return_value = department
    db.scalars.return_value.all.return_value = []

    app = FastAPI()
    app.include_router(router)

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db

    with TestClient(app) as client:
        yield client, db, ticket, department


def post_assignment(client, ticket, department):
    return client.post(
        f"/tickets/{ticket.id}/assignments",
        json={"department_id": str(department.id)},
    )


def test_assign_ticket(assignment_case):
    client, db, ticket, department = assignment_case

    response = post_assignment(client, ticket, department)

    assert response.status_code == 200
    assert response.json()["status"] == "Assigned"
    assert response.json()["assigned_department_id"] == str(department.id)

    added = [call.args[0] for call in db.add.call_args_list]
    assignment = next(item for item in added if isinstance(item, TicketAssignment))
    event = next(item for item in added if isinstance(item, TicketEvent))

    assert len(added) == 2
    assert assignment.ticket_id == ticket.id
    assert assignment.department_id == department.id
    assert event.old_values == {
        "status": "Open",
        "assigned_department_id": None,
    }
    assert event.new_values == {
        "status": "Assigned",
        "assigned_department_id": str(department.id),
    }
    db.commit.assert_called_once()


def test_missing_ticket(assignment_case):
    client, db, ticket, department = assignment_case
    db.scalar.return_value = None

    response = post_assignment(client, ticket, department)

    assert response.status_code == 404
    assert response.json()["detail"] == "Ticket not found."
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_missing_department(assignment_case):
    client, db, ticket, department = assignment_case
    db.get.return_value = None

    response = post_assignment(client, ticket, department)

    assert response.status_code == 404
    assert response.json()["detail"] == "Department not found."
    assert ticket.assigned_department_id is None
    db.add.assert_not_called()
    db.commit.assert_not_called()


@pytest.mark.parametrize("ticket_status", ["In Progress", "Resolved", "Closed"])
def test_reject_assignment_for_other_statuses(assignment_case, ticket_status):
    client, db, ticket, department = assignment_case
    ticket.status = ticket_status

    response = post_assignment(client, ticket, department)

    assert response.status_code == 409
    assert ticket.status == ticket_status
    assert ticket.assigned_department_id is None
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_repeated_assignment_does_not_add_records(assignment_case):
    client, db, ticket, department = assignment_case
    ticket.status = "Assigned"
    ticket.assigned_department_id = department.id

    existing = TicketAssignment(
        ticket_id=ticket.id,
        department_id=department.id,
        assigned_at=datetime.now(UTC),
        ended_at=None,
    )
    db.scalars.return_value.all.return_value = [existing]

    response = post_assignment(client, ticket, department)

    assert response.status_code == 200
    assert existing.ended_at is None
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_reassignment_ends_previous_assignment(assignment_case):
    client, db, ticket, department = assignment_case
    previous_department_id = uuid4()
    ticket.status = "Assigned"
    ticket.assigned_department_id = previous_department_id

    previous = TicketAssignment(
        ticket_id=ticket.id,
        department_id=previous_department_id,
        assigned_at=datetime.now(UTC),
        ended_at=None,
    )
    db.scalars.return_value.all.return_value = [previous]

    response = post_assignment(client, ticket, department)

    assert response.status_code == 200

    added = [call.args[0] for call in db.add.call_args_list]
    new_assignment = next(item for item in added if isinstance(item, TicketAssignment))
    event = next(item for item in added if isinstance(item, TicketEvent))

    assert previous.ended_at is not None
    assert previous.ended_at == new_assignment.assigned_at
    assert new_assignment.department_id == department.id
    assert event.old_values["assigned_department_id"] == str(previous_department_id)
    db.commit.assert_called_once()


def test_commit_failure_returns_503_and_requests_rollback(assignment_case):
    client, db, ticket, department = assignment_case
    db.commit.side_effect = SQLAlchemyError("Simulated database failure")

    response = post_assignment(client, ticket, department)

    assert response.status_code == 503
    assert response.json()["detail"] == "Could not assign the ticket."
    db.rollback.assert_called_once()
