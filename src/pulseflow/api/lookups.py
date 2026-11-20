from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from pulseflow.api.dependencies import get_current_user, get_db
from pulseflow.api.schemas import CategoryRead, LocationRead
from pulseflow.database.models.hospital import Category, Location, User

router = APIRouter(tags=["Reference Data"])

Database = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.get("/categories", response_model=list[CategoryRead])
def list_categories(
    db: Database,
    current_user: CurrentUser,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[Category]:
    try:
        statement = (
            select(Category)
            .order_by(Category.name, Category.id)
            .offset(offset)
            .limit(limit)
        )
        return list(db.scalars(statement).all())
    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve categories.",
        ) from None


@router.get("/locations", response_model=list[LocationRead])
def list_locations(
    db: Database,
    current_user: CurrentUser,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[Location]:
    try:
        statement = (
            select(Location)
            .order_by(Location.name, Location.id)
            .offset(offset)
            .limit(limit)
        )
        return list(db.scalars(statement).all())
    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve locations.",
        ) from None
