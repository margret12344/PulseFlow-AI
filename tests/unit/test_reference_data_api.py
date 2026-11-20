from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from pulseflow.api.dependencies import get_current_user, get_db
from pulseflow.api.lookups import router as lookups_router
from pulseflow.api.tickets import router as tickets_router
from pulseflow.database.models.hospital import Category, Location, User


@pytest.fixture
def reference_case():
    user = User(
        id=uuid4(),
        full_name="Synthetic Requester",
        email="requester@pulseflow.example",
        password_hash="unused-in-this-test",
        is_active=True,
    )
    location = Location(
        id=uuid4(),
        name="Reception",
        description=None,
    )
    category = Category(
        id=uuid4(),
        name="Network",
        default_department_id=uuid4(),
    )

    db = MagicMock(spec=Session)
    db.get.return_value = location

    def simulate_commit():
        ticket = db.add.call_args.args[0]
        ticket.id = uuid4()
        ticket.status = "Open"
        ticket.created_at = datetime.now(UTC)

    db.commit.side_effect = simulate_commit

    app = FastAPI()
    app.include_router(lookups_router)
    app.include_router(tickets_router)

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: user

    with TestClient(app) as client:
        yield client, db, user, location, category


def test_list_categories(reference_case):
    client, db, _, _, category = reference_case
    db.scalars.return_value.all.return_value = [category]

    response = client.get("/categories")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": str(category.id),
            "name": category.name,
            "default_department_id": str(category.default_department_id),
        }
    ]


def test_list_locations(reference_case):
    client, db, _, location, _ = reference_case
    db.scalars.return_value.all.return_value = [location]

    response = client.get("/locations")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": str(location.id),
            "name": location.name,
            "description": None,
        }
    ]


@pytest.mark.parametrize("path", ["/categories", "/locations"])
def test_reference_data_requires_login(reference_case, path):
    client, db, _, _, _ = reference_case
    client.app.dependency_overrides.pop(get_current_user)

    response = client.get(path)

    assert response.status_code == 401
    db.scalars.assert_not_called()


def test_create_ticket_with_location(reference_case):
    client, db, user, location, _ = reference_case

    response = client.post(
        "/tickets",
        json={
            "description": "Synthetic printer issue.",
            "location_id": str(location.id),
        },
    )

    assert response.status_code == 201
    assert response.json()["location_id"] == str(location.id)

    ticket = db.add.call_args.args[0]
    assert ticket.created_by == user.id
    assert ticket.location_id == location.id

    db.get.assert_called_once_with(Location, location.id)
    db.commit.assert_called_once()


def test_create_ticket_without_location(reference_case):
    client, db, user, _, _ = reference_case

    response = client.post(
        "/tickets",
        json={"description": "Synthetic printer issue."},
    )

    assert response.status_code == 201
    assert response.json()["location_id"] is None
    assert db.add.call_args.args[0].created_by == user.id
    db.get.assert_not_called()
    db.commit.assert_called_once()


def test_create_ticket_rejects_missing_location(reference_case):
    client, db, _, _, _ = reference_case
    db.get.return_value = None

    response = client.post(
        "/tickets",
        json={
            "description": "Synthetic printer issue.",
            "location_id": str(uuid4()),
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Location not found."
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_create_ticket_rejects_invalid_location_id(reference_case):
    client, db, _, _, _ = reference_case

    response = client.post(
        "/tickets",
        json={
            "description": "Synthetic printer issue.",
            "location_id": "invalid-id",
        },
    )

    assert response.status_code == 422
    db.add.assert_not_called()
    db.commit.assert_not_called()
