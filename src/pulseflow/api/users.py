from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from pulseflow.api.dependencies import get_current_user, get_db, require_admin
from pulseflow.api.schemas import UserActiveUpdate, UserCreate, UserRead
from pulseflow.core.security import hash_password
from pulseflow.database.models.hospital import Department, Role, User, UserRole

router = APIRouter(prefix="/users", tags=["Users"])

Database = Annotated[Session, Depends(get_db)]
Admin = Annotated[User, Depends(require_admin)]


@router.get("/me", response_model=UserRead)
def read_current_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    return current_user


@router.post(
    "",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
)
def create_user(payload: UserCreate, db: Database, admin: Admin) -> User:
    full_name = payload.full_name.strip()
    email = payload.email.strip().lower()

    if not full_name:
        raise HTTPException(
            status_code=422,
            detail="Full name cannot be blank.",
        )

    if (
        email.count("@") != 1
        or not all(email.split("@"))
        or any(character.isspace() for character in email)
    ):
        raise HTTPException(
            status_code=422,
            detail="Invalid email address.",
        )

    try:
        existing = db.scalar(select(User.id).where(func.lower(User.email) == email))

        if existing is not None:
            raise HTTPException(
                status_code=409,
                detail="Email already exists.",
            )

        if payload.department_id is not None:
            department = db.get(Department, payload.department_id)

            if department is None:
                raise HTTPException(
                    status_code=404,
                    detail="Department not found.",
                )

        role = db.scalar(select(Role).where(Role.name == payload.role))

        if role is None:
            role = Role(name=payload.role)
            db.add(role)
            db.flush()

        user = User(
            full_name=full_name,
            email=email,
            password_hash=hash_password(payload.password),
            department_id=payload.department_id,
            is_active=True,
        )
        db.add(user)
        db.flush()

        db.add(UserRole(user_id=user.id, role_id=role.id))
        db.commit()

    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Account data conflicts with an existing record. Retry.",
        ) from None
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="Could not create the user.",
        ) from None

    return user


@router.get("", response_model=list[UserRead])
def list_users(
    db: Database,
    admin: Admin,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[User]:
    try:
        statement = (
            select(User)
            .order_by(User.created_at.desc(), User.id.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(db.scalars(statement).all())
    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve users.",
        ) from None


@router.patch("/{user_id}/active", response_model=UserRead)
def update_user_active(
    user_id: UUID,
    payload: UserActiveUpdate,
    db: Database,
    admin: Admin,
) -> User:
    if user_id == admin.id and not payload.is_active:
        raise HTTPException(
            status_code=409,
            detail="You cannot deactivate your own account.",
        )

    try:
        user = db.scalar(select(User).where(User.id == user_id).with_for_update())

        if user is None:
            raise HTTPException(
                status_code=404,
                detail="User not found.",
            )

        has_admin_role = db.scalar(
            select(Role.id)
            .join(UserRole, UserRole.role_id == Role.id)
            .where(UserRole.user_id == user_id, Role.name == "admin")
        )

        if has_admin_role is not None and not payload.is_active:
            raise HTTPException(
                status_code=409,
                detail="Administrator deactivation is not supported here.",
            )

        user.is_active = payload.is_active
        db.commit()

    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="Could not update the user.",
        ) from None

    return user
