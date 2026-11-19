import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

load_dotenv()


class AuthSettings(BaseModel):
    secret_key: SecretStr = Field(min_length=64)
    access_token_expire_minutes: int = Field(default=30, ge=1, le=1440)


@lru_cache(maxsize=1)
def get_auth_settings() -> AuthSettings:
    secret_key = os.getenv("JWT_SECRET_KEY")

    if (
        not secret_key
        or secret_key.startswith("GENERATE_")
        or secret_key.startswith("PASTE_")
    ):
        raise ValueError("Configure a generated JWT_SECRET_KEY in .env.")

    return AuthSettings(
        secret_key=secret_key,
        access_token_expire_minutes=os.getenv(
            "JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"
        ),
    )