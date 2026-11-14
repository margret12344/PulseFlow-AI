
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is not configured.")

    return database_url


def get_engine() -> Engine:
    return create_engine(
        get_database_url(),
        pool_pre_ping=True,
    )


def test_connection() -> None:
    engine = get_engine()

    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            value = result.scalar_one()

            if value != 1:
                raise RuntimeError("Unexpected database test result.")

        print("Database connection successful!")

    finally:
        engine.dispose()
