from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import Column, MetaData, Table, Uuid, create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from pulseflow.api.ticket_permissions import (
    TicketAccess,
    get_ticket_access,
    require_ticket_editor,
    ticket_read_condition,
)
from pulseflow.database.models.hospital import User
from pulseflow.database.models.tickets import Ticket


@pytest.fixture
def access_case():
    user_id = uuid4()
    department_id = uuid4()

    user = User(
        id=user_id,
        full_name="Synthetic User",
        email="user@pulseflow.example",
        password_hash="unused-in-this-test",
        is_active=True,
        department_id=department_id,
    )

    ids = {name: uuid4() for name in ("own", "department", "other", "unassigned")}

    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata = MetaData()

    tickets_table = Table(
        "tickets",
        metadata,
        Column("id", Uuid, primary_key=True),
        Column("created_by", Uuid, nullable=True),
        Column("assigned_department_id", Uuid, nullable=True),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            tickets_table.insert(),
            [
                {
                    "id": ids["own"],
                    "created_by": user_id,
                    "assigned_department_id": uuid4(),
                },
                {
                    "id": ids["department"],
                    "created_by": uuid4(),
                    "assigned_department_id": department_id,
                },
                {
                    "id": ids["other"],
                    "created_by": uuid4(),
                    "assigned_department_id": uuid4(),
                },
                {
                    "id": ids["unassigned"],
                    "created_by": None,
                    "assigned_department_id": None,
                },
            ],
        )

    try:
        with Session(engine) as session:
            yield session, user, ids
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("role", "expected_names"),
    [
        ("requester", {"own"}),
        ("support", {"department"}),
        ("admin", {"own", "department", "other", "unassigned"}),
    ],
)
def test_read_scope_selects_only_allowed_tickets(access_case, role, expected_names):
    session, user, ids = access_case
    access = TicketAccess(user=user, roles=frozenset({role}))

    actual = set(
        session.scalars(select(Ticket.id).where(ticket_read_condition(access))).all()
    )

    assert actual == {ids[name] for name in expected_names}


def test_support_without_department_cannot_read_unassigned_tickets(access_case):
    session, user, _ = access_case
    user.department_id = None
    access = TicketAccess(user=user, roles=frozenset({"support"}))

    actual = session.scalars(
        select(Ticket.id).where(ticket_read_condition(access))
    ).all()

    assert actual == []


@pytest.mark.parametrize("role", ["admin", "support", "requester"])
def test_get_ticket_access_loads_recognized_role(access_case, role):
    _, user, _ = access_case
    db = MagicMock(spec=Session)
    db.scalars.return_value.all.return_value = [role]

    access = get_ticket_access(user=user, db=db)

    assert access.user is user
    assert access.roles == frozenset({role})


@pytest.mark.parametrize("roles", [[], ["unknown"]])
def test_get_ticket_access_rejects_missing_or_unknown_roles(access_case, roles):
    _, user, _ = access_case
    db = MagicMock(spec=Session)
    db.scalars.return_value.all.return_value = roles

    with pytest.raises(HTTPException) as error:
        get_ticket_access(user=user, db=db)

    assert error.value.status_code == 403


def test_get_ticket_access_handles_database_failure(access_case):
    _, user, _ = access_case
    db = MagicMock(spec=Session)
    db.scalars.side_effect = SQLAlchemyError("Synthetic failure")

    with pytest.raises(HTTPException) as error:
        get_ticket_access(user=user, db=db)

    assert error.value.status_code == 503
    assert error.value.detail == "Authorization service is unavailable."


@pytest.mark.parametrize("role", ["admin", "support"])
def test_editor_allows_admin_and_support(access_case, role):
    _, user, _ = access_case
    access = TicketAccess(user=user, roles=frozenset({role}))

    assert require_ticket_editor(access) is access


def test_editor_rejects_requester(access_case):
    _, user, _ = access_case
    access = TicketAccess(user=user, roles=frozenset({"requester"}))

    with pytest.raises(HTTPException) as error:
        require_ticket_editor(access)

    assert error.value.status_code == 403
