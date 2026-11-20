import os
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from pulseflow.api.dependencies import get_current_user, get_db
from pulseflow.api.tickets import router
from pulseflow.core.database import get_database_url
from pulseflow.database.models.hospital import Department, Role, User, UserRole
from pulseflow.database.models.tickets import Ticket, TicketAssignment, TicketEvent


@pytest.fixture
def database_case():
    configured_url = os.getenv("TEST_DATABASE_URL")

    if not configured_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests.")

    test_url = make_url(configured_url)
    application_url = make_url(get_database_url())

    if (
        test_url.get_backend_name() != "postgresql"
        or test_url.database != "pulseflow_test"
        or test_url.database == application_url.database
    ):
        pytest.fail("Use a separate PostgreSQL database named pulseflow_test.")

    engine = create_engine(test_url)

    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            session = Session(
                bind=connection,
                expire_on_commit=False,
                join_transaction_mode="create_savepoint",
            )

            try:
                department = Department(name=f"Test Department {uuid4()}")
                requester = User(
                    full_name="Integration Requester",
                    email=f"{uuid4()}@pulseflow.example",
                    password_hash="unused",
                    is_active=True,
                )
                other_user = User(
                    full_name="Other Requester",
                    email=f"{uuid4()}@pulseflow.example",
                    password_hash="unused",
                    is_active=True,
                )
                admin = User(
                    full_name="Integration Admin",
                    email=f"{uuid4()}@pulseflow.example",
                    password_hash="unused",
                    is_active=True,
                )
                session.add_all([department, requester, other_user, admin])
                session.flush()

                for user, role_name in (
                    (requester, "requester"),
                    (other_user, "requester"),
                    (admin, "admin"),
                ):
                    role = session.scalar(select(Role).where(Role.name == role_name))

                    if role is None:
                        role = Role(name=role_name)
                        session.add(role)
                        session.flush()

                    session.add(UserRole(user_id=user.id, role_id=role.id))

                ticket = Ticket(
                    description="Original integration printer issue.",
                    created_by=requester.id,
                )
                session.add(ticket)
                session.commit()

                identity = {"user": requester}
                app = FastAPI()
                app.include_router(router)

                def override_db():
                    yield session

                app.dependency_overrides[get_db] = override_db
                app.dependency_overrides[get_current_user] = lambda: identity["user"]

                with TestClient(app) as client:
                    yield {
                        "client": client,
                        "session": session,
                        "ticket": ticket,
                        "requester": requester,
                        "other_user": other_user,
                        "admin": admin,
                        "department": department,
                        "identity": identity,
                    }

            finally:
                session.close()

                if transaction.is_active:
                    transaction.rollback()
    finally:
        engine.dispose()


def test_update_persists_ticket_and_event(database_case):
    case = database_case
    ticket_id = case["ticket"].id

    response = case["client"].patch(
        f"/tickets/{ticket_id}",
        json={"description": "Updated integration printer issue."},
    )

    assert response.status_code == 200

    session = case["session"]
    session.expire_all()
    stored_ticket = session.get(Ticket, ticket_id)
    stored_event = session.scalar(
        select(TicketEvent).where(TicketEvent.ticket_id == ticket_id)
    )

    assert stored_ticket.description == "Updated integration printer issue."
    assert stored_event is not None
    assert stored_event.actor_id == case["requester"].id
    assert stored_event.old_values == {
        "description": "Original integration printer issue."
    }
    assert stored_event.new_values == {
        "description": "Updated integration printer issue."
    }


def test_other_requester_cannot_read_or_edit_ticket(database_case):
    case = database_case
    ticket_id = case["ticket"].id
    case["identity"]["user"] = case["other_user"]
    client = case["client"]

    assert client.get(f"/tickets/{ticket_id}").status_code == 404
    assert client.get(f"/tickets/{ticket_id}/events").status_code == 404
    assert client.get(f"/tickets/{ticket_id}/assignments").status_code == 404

    response = client.patch(
        f"/tickets/{ticket_id}",
        json={"description": "Unauthorized change."},
    )

    assert response.status_code == 404

    listed = client.get("/tickets", params={"search": "integration"})
    assert listed.status_code == 200
    assert str(ticket_id) not in {item["id"] for item in listed.json()}

    case["session"].expire_all()
    assert (
        case["session"].get(Ticket, ticket_id).description
        == "Original integration printer issue."
    )


def test_assignment_persists_and_repeated_request_is_idempotent(database_case):
    case = database_case
    case["identity"]["user"] = case["admin"]
    ticket_id = case["ticket"].id
    department_id = case["department"].id
    path = f"/tickets/{ticket_id}/assignments"
    payload = {"department_id": str(department_id)}

    first = case["client"].post(path, json=payload)
    repeated = case["client"].post(path, json=payload)

    assert first.status_code == 200
    assert repeated.status_code == 200

    session = case["session"]
    session.expire_all()

    assignments = list(
        session.scalars(
            select(TicketAssignment).where(TicketAssignment.ticket_id == ticket_id)
        )
    )
    events = list(
        session.scalars(select(TicketEvent).where(TicketEvent.ticket_id == ticket_id))
    )

    assert len(assignments) == 1
    assert assignments[0].department_id == department_id
    assert assignments[0].assigned_by == case["admin"].id
    assert assignments[0].ended_at is None
    assert len(events) == 1
    assert events[0].event_type == "department_assigned"

    stored_ticket = session.get(Ticket, ticket_id)
    assert stored_ticket.status == "Assigned"
    assert stored_ticket.assigned_department_id == department_id

    history = case["client"].get(path)
    assert history.status_code == 200
    assert history.json()[0]["id"] == str(assignments[0].id)


def test_event_failure_rolls_back_ticket_update(database_case):
    case = database_case
    session = case["session"]
    ticket_id = case["ticket"].id

    def invalidate_event(session, flush_context, instances):
        for item in session.new:
            if isinstance(item, TicketEvent):
                item.actor_id = uuid4()

    event.listen(session, "before_flush", invalidate_event)

    try:
        response = case["client"].patch(
            f"/tickets/{ticket_id}",
            json={"description": "This must be rolled back."},
        )
    finally:
        event.remove(session, "before_flush", invalidate_event)

    assert response.status_code == 503

    session.expire_all()
    stored_ticket = session.get(Ticket, ticket_id)

    assert stored_ticket.description == "Original integration printer issue."
    assert (
        session.scalar(select(TicketEvent).where(TicketEvent.ticket_id == ticket_id))
        is None
    )
