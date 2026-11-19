from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt
import pytest
from jwt.exceptions import InvalidTokenError

from pulseflow.core import security
from pulseflow.core.auth_config import AuthSettings

TEST_SECRET = "synthetic-test-secret-" * 4


@pytest.fixture
def token_case(monkeypatch):
    settings = AuthSettings(
        secret_key=TEST_SECRET,
        access_token_expire_minutes=30,
    )

    monkeypatch.setattr(security, "get_auth_settings", lambda: settings)

    user_id = uuid4()
    now = datetime.now(UTC)

    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=30),
        "iss": security.JWT_ISSUER,
        "aud": security.JWT_AUDIENCE,
        "token_type": "access",
    }

    return user_id, payload


def sign_payload(payload, secret=TEST_SECRET, algorithm="HS256"):
    return jwt.encode(payload, secret, algorithm=algorithm)


def test_access_token_round_trip(token_case):
    user_id, _ = token_case

    token = security.create_access_token(user_id)

    assert security.decode_access_token(token) == user_id


def test_created_token_claims(token_case):
    user_id, _ = token_case

    token = security.create_access_token(user_id)
    payload = jwt.decode(
        token,
        TEST_SECRET,
        algorithms=["HS256"],
        issuer=security.JWT_ISSUER,
        audience=security.JWT_AUDIENCE,
    )

    assert payload["sub"] == str(user_id)
    assert payload["token_type"] == "access"
    assert payload["exp"] - payload["iat"] == 30 * 60
    assert "password" not in payload
    assert "password_hash" not in payload


def test_reject_expired_token(token_case):
    _, payload = token_case
    payload["iat"] = datetime.now(UTC) - timedelta(hours=2)
    payload["exp"] = datetime.now(UTC) - timedelta(hours=1)

    with pytest.raises(InvalidTokenError):
        security.decode_access_token(sign_payload(payload))


def test_reject_wrong_signing_key(token_case):
    _, payload = token_case

    token = sign_payload(payload, secret="different-synthetic-secret-" * 4)

    with pytest.raises(InvalidTokenError):
        security.decode_access_token(token)


def test_reject_modified_payload(token_case):
    user_id, payload = token_case
    original = sign_payload(payload)

    payload["sub"] = str(uuid4())
    modified = sign_payload(payload)

    tampered = ".".join(modified.split(".")[:2]) + "." + original.split(".")[2]

    with pytest.raises(InvalidTokenError):
        security.decode_access_token(tampered)

    assert security.decode_access_token(original) == user_id


@pytest.mark.parametrize(
    "claim",
    ["sub", "iat", "exp", "iss", "aud", "token_type"],
)
def test_reject_missing_required_claim(token_case, claim):
    _, payload = token_case
    del payload[claim]

    with pytest.raises(InvalidTokenError):
        security.decode_access_token(sign_payload(payload))


@pytest.mark.parametrize(
    ("claim", "value"),
    [
        ("iss", "another-issuer"),
        ("aud", "another-api"),
        ("token_type", "refresh"),
        ("sub", "invalid-user-id"),
    ],
)
def test_reject_invalid_claim(token_case, claim, value):
    _, payload = token_case
    payload[claim] = value

    with pytest.raises(InvalidTokenError):
        security.decode_access_token(sign_payload(payload))


def test_reject_unapproved_algorithm(token_case):
    _, payload = token_case
    token = sign_payload(payload, algorithm="HS512")

    with pytest.raises(InvalidTokenError):
        security.decode_access_token(token)


def test_reject_malformed_token(token_case):
    with pytest.raises(InvalidTokenError):
        security.decode_access_token("not-a-jwt")


def test_password_hash_and_verification():
    password = "Synthetic-Test-Password-123!"
    hashed = security.hash_password(password)

    assert hashed != password
    assert security.verify_password(password, hashed)
    assert not security.verify_password("Incorrect-Password", hashed)


def test_reject_empty_password():
    with pytest.raises(ValueError, match="Password cannot be empty"):
        security.hash_password("")
