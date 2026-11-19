from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pwdlib.exceptions import UnknownHashError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from pulseflow.api.dependencies import get_db
from pulseflow.api.schemas import TokenRead
from pulseflow.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from pulseflow.database.models.hospital import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

DUMMY_PASSWORD_HASH = hash_password("Synthetic-Dummy-Password-Only")


@router.post("/token", response_model=TokenRead)
def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
) -> TokenRead:
    try:
        user = db.scalar(
            select(User).where(User.email == form.username.strip().lower())
        )
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is unavailable.",
        ) from None

    stored_hash = user.password_hash if user else DUMMY_PASSWORD_HASH

    try:
        password_valid = verify_password(form.password, stored_hash)
    except UnknownHashError:
        password_valid = False

    if user is None or not password_valid or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenRead(access_token=create_access_token(user.id))
