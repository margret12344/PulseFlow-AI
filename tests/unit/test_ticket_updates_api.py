from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from pulseflow.api.dependencies import get_db
from pulseflow.api.ticket_permissions import TicketAccess, get_ticket_access
from pulseflow.api.tickets import router
from pulseflow.database.models.hospital import User
from pulseflow.database.models.tickets import Ticket, TicketAssignment, TicketEvent


@pytest.fixture
def ticket_case():
    user = User(
        id=uuid4(),
        full_name="Synthetic Requester",
        email="requester@pulseflow.example",
        is_active=True,
    )
    ticket = Ticket(
        id=uuid4(),
        description="Original printer issue.",
        status="Open",
        priority=None,
        created_at=datetime.now(UTC),
        resolved_at=None,
        assigned_department_id=None,
        location_id=None,
        created_by=user.id,
    )
    db = MagicMock(spec=Session)
    db.scalar.return_value = ticket
    db.scalars.return_value.all.return_value = []

    access = TicketAccess(user=user, roles=frozenset({"requester"}))
    app = FastAPI()
    app.include_router(router)

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_ticket_access] = lambda: access

    with TestClient(app) as client:
        yield client, db, ticket, access


def test_update_description_records_actor_and_changes(ticket_case):
    client, db, ticket, access = ticket_case

    response = client.patch(
        f"/tickets/{ticket.id}",
        json={"description": "Updated printer issue."},
    )

    assert response.status_code == 200
    assert response.json()["description"] == "Updated printer issue."
    event = db.add.call_args.args[0]
    assert isinstance(event, TicketEvent)
    assert event.actor_id == access.user.id
    assert event.ticket_id == ticket.id
    assert event.event_type == "ticket_updated"
    assert event.old_values == {"description": "Original printer issue."}
    assert event.new_values == {"description": "Updated printer issue."}
    db.commit.assert_called_once()


def test_clear_location_records_previous_value(ticket_case):
    client, db, ticket, _ = ticket_case
    previous_location = uuid4()
    ticket.location_id = previous_location

    response = client.patch(
        f"/tickets/{ticket.id}",
        json={"location_id": None},
    )

    assert response.status_code == 200
    assert ticket.location_id is None
    event = db.add.call_args.args[0]
    assert event.old_values == {"location_id": str(previous_location)}
    assert event.new_values == {"location_id": None}
    db.get.assert_not_called()


def test_unknown_location_does_not_change_ticket(ticket_case):
    client, db, ticket, _ = ticket_case
    db.get.return_value = None

    response = client.patch(
        f"/tickets/{ticket.id}",
        json={
            "description": "Must not be applied.",
            "location_id": str(uuid4()),
        },
    )

    assert response.status_code == 404
    assert ticket.description == "Original printer issue."
    db.add.assert_not_called()
    db.commit.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"description": None},
        {"description": "   "},
        {"location_id": "invalid"},
        {"status": "Resolved"},
    ],
)
def test_invalid_update_is_rejected(ticket_case, payload):
    client, db, ticket, _ = ticket_case

    response = client.patch(f"/tickets/{ticket.id}", json=payload)

    assert response.status_code == 422
    db.add.assert_not_called()
    db.commit.assert_not_called()


@pytest.mark.parametrize(
    "ticket_status",
    ["Assigned", "In Progress", "Resolved", "Closed"],
)
def test_requester_cannot_edit_non_open_ticket(ticket_case, ticket_status):
    client, db, ticket, _ = ticket_case
    ticket.status = ticket_status

    response = client.patch(
        f"/tickets/{ticket.id}",
        json={"description": "Rejected change."},
    )

    assert response.status_code == 409
    assert ticket.description == "Original printer issue."
    db.commit.assert_not_called()


def test_unchanged_update_does_not_create_event(ticket_case):
    client, db, ticket, _ = ticket_case

    response = client.patch(
        f"/tickets/{ticket.id}",
        json={"description": ticket.description},
    )

    assert response.status_code == 200
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_hidden_or_missing_ticket_returns_404(ticket_case):
    client, db, ticket, _ = ticket_case
    db.scalar.return_value = None

    response = client.patch(
        f"/tickets/{ticket.id}",
        json={"description": "Rejected change."},
    )

    assert response.status_code == 404
    db.commit.assert_not_called()


def test_update_database_failure_rolls_back(ticket_case):
    client, db, ticket, _ = ticket_case
    db.commit.side_effect = SQLAlchemyError("Synthetic failure")

    response = client.patch(
        f"/tickets/{ticket.id}",
        json={"description": "Updated description."},
    )

    assert response.status_code == 503
    db.rollback.assert_called_once()


def test_assignment_history(ticket_case):
    client, db, ticket, access = ticket_case
    assignment = TicketAssignment(
        id=uuid4(),
        ticket_id=ticket.id,
        department_id=uuid4(),
        assigned_by=uuid4(),
        assigned_at=datetime.now(UTC),
        ended_at=None,
    )
    db.scalars.return_value.all.return_value = [assignment]

    response = client.get(f"/tickets/{ticket.id}/assignments")

    assert response.status_code == 200
    assert response.json()[0]["id"] == str(assignment.id)
    assert response.json()[0]["ended_at"] is None

    lookup = db.scalar.call_args.args[0].compile()
    assert access.user.id in lookup.params.values()


def test_assignment_history_hidden_ticket_returns_404(ticket_case):
    client, db, ticket, _ = ticket_case
    db.scalar.return_value = None

    response = client.get(f"/tickets/{ticket.id}/assignments")

    assert response.status_code == 404
    db.scalars.assert_not_called()


def test_filters_preserve_requester_scope(ticket_case):
    client, db, _, access = ticket_case
    category_id = uuid4()
    department_id = uuid4()
    location_id = uuid4()

    response = client.get(
        "/tickets",
        params={
            "status": "Open",
            "priority": "High",
            "category_id": str(category_id),
            "department_id": str(department_id),
            "location_id": str(location_id),
            "search": "printer",
            "offset": 2,
            "limit": 5,
        },
    )

    assert response.status_code == 200
    statement = db.scalars.call_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    values = list(compiled.params.values())

    for value in (
        access.user.id,
        "Open",
        "High",
        category_id,
        department_id,
        location_id,
        "printer",
        2,
        5,
    ):
        assert value in values

    assert "tickets.created_by =" in str(compiled)


def test_blank_search_is_rejected(ticket_case):
    client, db, _, _ = ticket_case

    response = client.get("/tickets", params={"search": "   "})

    assert response.status_code == 422
    db.scalars.assert_not_called()


def test_ticket_routes_are_unique():
    routes = [
        (method, route.path) for route in router.routes for method in route.methods
    ]

    assert len(routes) == len(set(routes)) == 8
