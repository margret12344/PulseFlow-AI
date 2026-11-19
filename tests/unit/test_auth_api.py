from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from pulseflow.api.auth import router as auth_router
from pulseflow.api.dependencies import get_db
from pulseflow.api.users import router as users_router
from pulseflow.core import security
from pulseflow.core.auth_config import AuthSettings
from pulseflow.database.models.hospital import User

TEST_PASSWORD = "Synthetic-Auth-Password-123!"
TEST_SECRET = "synthetic-auth-api-secret-" * 4


@pytest.fixture
def auth_case(monkeypatch):
    settings = AuthSettings(
        secret_key=TEST_SECRET,
        access_token_expire_minutes=30,
    )
    monkeypatch.setattr(security, "get_auth_settings", lambda: settings)

    user = User(
        id=uuid4(),
        full_name="Synthetic User",
        email="user@pulseflow.example",
        password_hash=security.hash_password(TEST_PASSWORD),
        department_id=None,
        is_active=True,
    )

    db = MagicMock(spec=Session)
    db.scalar.return_value = user
    db.get.return_value = user

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(users_router)

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db

    with TestClient(app) as client:
        yield client, db, user


def login(client, password=TEST_PASSWORD):
    return client.post(
        "/auth/token",
        data={
            "username": "user@pulseflow.example",
            "password": password,
        },
    )


def bearer_headers(user):
    token = security.create_access_token(user.id)
    return {"Authorization": f"Bearer {token}"}


def test_login_and_read_current_user(auth_case):
    client, db, user = auth_case

    response = login(client)

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert security.decode_access_token(body["access_token"]) == user.id

    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(user.id)
    assert response.json()["email"] == user.email
    assert "password" not in response.json()
    assert "password_hash" not in response.json()
    db.get.assert_called_once_with(User, user.id)


@pytest.mark.parametrize(
    "scenario",
    ["wrong_password", "missing_user", "inactive_user", "unknown_hash"],
)
def test_login_rejects_invalid_credentials(auth_case, scenario):
    client, db, user = auth_case
    password = TEST_PASSWORD

    if scenario == "wrong_password":
        password = "Incorrect-Password"
    elif scenario == "missing_user":
        db.scalar.return_value = None
    elif scenario == "inactive_user":
        user.is_active = False
    else:
        user.password_hash = "not-a-supported-hash"

    response = login(client, password=password)

    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect email or password."}
    assert response.headers["www-authenticate"] == "Bearer"
    assert "access_token" not in response.json()


def test_login_handles_database_failure(auth_case):
    client, db, _ = auth_case
    db.scalar.side_effect = SQLAlchemyError("Synthetic internal details")

    response = login(client)

    assert response.status_code == 503
    assert response.json() == {"detail": "Authentication service is unavailable."}


def test_current_user_requires_token(auth_case):
    client, db, _ = auth_case

    response = client.get("/users/me")

    assert response.status_code == 401
    db.get.assert_not_called()


def test_current_user_rejects_invalid_token(auth_case):
    client, db, _ = auth_case

    response = client.get(
        "/users/me",
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401
    db.get.assert_not_called()


def test_current_user_rejects_expired_token(auth_case):
    client, db, user = auth_case
    now = datetime.now(UTC)

    token = jwt.encode(
        {
            "sub": str(user.id),
            "iat": now - timedelta(hours=2),
            "exp": now - timedelta(hours=1),
            "iss": security.JWT_ISSUER,
            "aud": security.JWT_AUDIENCE,
            "token_type": "access",
        },
        TEST_SECRET,
        algorithm=security.JWT_ALGORITHM,
    )

    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
    db.get.assert_not_called()


def test_current_user_rejects_deleted_user(auth_case):
    client, db, user = auth_case
    headers = bearer_headers(user)
    db.get.return_value = None

    response = client.get("/users/me", headers=headers)

    assert response.status_code == 401


def test_existing_token_rejected_after_account_deactivation(auth_case):
    client, db, user = auth_case
    headers = bearer_headers(user)

    assert client.get("/users/me", headers=headers).status_code == 200

    user.is_active = False

    response = client.get("/users/me", headers=headers)

    assert response.status_code == 401
    assert db.get.call_count == 2


def test_current_user_handles_database_failure(auth_case):
    client, db, user = auth_case
    headers = bearer_headers(user)
    db.get.side_effect = SQLAlchemyError("Synthetic internal details")

    response = client.get("/users/me", headers=headers)

    assert response.status_code == 503
    assert response.json() == {"detail": "Authentication service is unavailable."}


def test_requester_cannot_list_users(auth_case):
    client, db, user = auth_case
    db.scalar.return_value = None

    response = client.get("/users", headers=bearer_headers(user))

    assert response.status_code == 403
    assert response.json()["detail"] == "Administrator access required."
    db.scalars.assert_not_called()
