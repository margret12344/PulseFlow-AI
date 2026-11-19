from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash

from pulseflow.core.auth_config import get_auth_settings

password_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("Password cannot be empty.")

    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return password_hasher.verify(password, password_hash)

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "pulseflow-ai"
JWT_AUDIENCE = "pulseflow-api"


def create_access_token(user_id: UUID) -> str:
    settings = get_auth_settings()
    now = datetime.now(UTC)

    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(
            minutes=settings.access_token_expire_minutes
        ),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "token_type": "access",
    }

    return jwt.encode(
        payload,
        settings.secret_key.get_secret_value(),
        algorithm=JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> UUID:
    settings = get_auth_settings()

    payload = jwt.decode(
        token,
        settings.secret_key.get_secret_value(),
        algorithms=[JWT_ALGORITHM],
        issuer=JWT_ISSUER,
        audience=JWT_AUDIENCE,
        options={
            "require": [
                "sub",
                "iat",
                "exp",
                "iss",
                "aud",
                "token_type",
            ]
        },
    )

    if payload["token_type"] != "access":
        raise InvalidTokenError("Invalid token type.")

    try:
        return UUID(payload["sub"])
    except (ValueError, TypeError, AttributeError) as exc:
        raise InvalidTokenError("Invalid token subject.") from exc