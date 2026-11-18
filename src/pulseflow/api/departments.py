from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import pulseflow.database.models  # noqa: F401
from pulseflow.api.dependencies import get_db
from pulseflow.api.schemas import DepartmentRead
from pulseflow.database.models.hospital import Department

router = APIRouter(prefix="/departments", tags=["Departments"])


@router.get("", response_model=list[DepartmentRead])
def list_departments(
    db: Annotated[Session, Depends(get_db)],
) -> list[Department]:
    try:
        statement = select(Department).order_by(Department.name)
        return list(db.scalars(statement).all())
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not retrieve departments.",
        ) from None
