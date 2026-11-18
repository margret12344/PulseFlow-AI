import uuid

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

import pulseflow.database.models  # noqa: F401
from pulseflow.core.database import get_engine
from pulseflow.data_engineering.profiling import load_dataset
from pulseflow.data_engineering.quality import validate_dataset
from pulseflow.database.models.hospital import Category, Department, Location
from pulseflow.database.models.tickets import Ticket


def prepare_tickets() -> pd.DataFrame:
    df = load_dataset()

    checks = validate_dataset(df)
    failed_checks = [name for name, result in checks.items() if not result["passed"]]

    if failed_checks:
        raise ValueError(f"Data quality checks failed: {failed_checks}")

    prepared = df.copy()

    for column in ("created_at", "resolved_at"):
        prepared[column] = (
            pd.to_datetime(prepared[column], errors="raise")
            .dt.tz_localize(
                "Africa/Cairo",
                ambiguous="raise",
                nonexistent="raise",
            )
            .dt.tz_convert("UTC")
        )

    prepared["source_ticket_id"] = prepared["ticket_id"].astype(str)

    if prepared["source_ticket_id"].str.len().gt(100).any():
        raise ValueError("Source ticket IDs must not exceed 100 characters.")

    if prepared["source_ticket_id"].duplicated().any():
        raise ValueError("Duplicate source ticket IDs after conversion.")
    invalid_dates = prepared["resolved_at"].notna() & (
        prepared["resolved_at"] < prepared["created_at"]
    )

    if invalid_dates.any():
        raise ValueError("Resolution time cannot precede creation time.")
    return prepared


def import_tickets() -> None:
    prepared = prepare_tickets()
    engine = get_engine()
    inserted_count = 0

    try:
        with Session(engine) as session:
            with session.begin():
                for name in sorted(prepared["department"].unique()):
                    session.execute(
                        insert(Department)
                        .values(id=uuid.uuid4(), name=name)
                        .on_conflict_do_nothing(index_elements=["name"])
                    )

                department_ids = dict(
                    session.execute(select(Department.name, Department.id)).all()
                )

                category_pairs = prepared[["category", "department"]].drop_duplicates()

                for category_name, department_name in category_pairs.itertuples(
                    index=False, name=None
                ):
                    department_id = department_ids[department_name]

                    session.execute(
                        insert(Category)
                        .values(
                            id=uuid.uuid4(),
                            name=category_name,
                            default_department_id=department_id,
                        )
                        .on_conflict_do_nothing(index_elements=["name"])
                    )

                    category = session.scalar(
                        select(Category).where(Category.name == category_name)
                    )

                    if category.default_department_id != department_id:
                        raise ValueError(
                            f"Existing category mapping conflicts: {category_name}"
                        )

                category_ids = dict(
                    session.execute(select(Category.name, Category.id)).all()
                )

                for name in sorted(prepared["location"].dropna().unique()):
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError("Location names must be non-empty text.")

                    if len(name) > 100:
                        raise ValueError(
                            "Location names must not exceed 100 characters."
                        )

                    session.execute(
                        insert(Location)
                        .values(id=uuid.uuid4(), name=name)
                        .on_conflict_do_nothing(index_elements=["name"])
                    )

                location_ids = dict(
                    session.execute(select(Location.name, Location.id)).all()
                )

                for row in prepared.itertuples(index=False):
                    location_id = (
                        None if pd.isna(row.location) else location_ids[row.location]
                    )
                    resolved_at = (
                        None
                        if pd.isna(row.resolved_at)
                        else row.resolved_at.to_pydatetime()
                    )

                    inserted_id = session.scalar(
                        insert(Ticket)
                        .values(
                            id=uuid.uuid4(),
                            source_ticket_id=row.source_ticket_id,
                            description=row.description,
                            category_id=category_ids[row.category],
                            priority=row.priority,
                            status=row.status,
                            location_id=location_id,
                            assigned_department_id=department_ids[row.department],
                            created_at=row.created_at.to_pydatetime(),
                            resolved_at=resolved_at,
                        )
                        .on_conflict_do_nothing(index_elements=["source_ticket_id"])
                        .returning(Ticket.id)
                    )

                    if inserted_id is not None:
                        inserted_count += 1

        print(f"Inserted tickets: {inserted_count}")
        print(f"Skipped existing tickets: {len(prepared) - inserted_count}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    import_tickets()
