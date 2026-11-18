from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from pulseflow.api.dependencies import get_db
from pulseflow.api.tickets import router as tickets_router

app = FastAPI(
    title="PulseFlow AI",
    description="Hospital IT Support Ticket Management API",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

app.include_router(tickets_router)

@app.get("/health/db")
def database_health_check(
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, str]:
    try:
        db.execute(text("SELECT 1")).scalar_one()
    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Database is unavailable.",
        ) from None

    return {"status": "ok", "database": "connected"}