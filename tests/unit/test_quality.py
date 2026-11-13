
import pandas as pd

from pulseflow.data_engineering.quality import validate_dataset


def create_sample_dataset():
    return pd.DataFrame(
        {
            "ticket_id": ["HSP-0001", "HSP-0002"],
            "description": [
                "The network is not working",
                "Printer connection failed",
            ],
            "category": ["Network", "Software Failure"],
            "department": ["Network Team", "IT Support"],
            "priority": ["High", "Medium"],
            "location": ["ICU", None],
            "status": ["Resolved", "Open"],
            "created_at": pd.to_datetime(
                ["2026-01-01 10:00", "2026-01-02 12:00"]
            ),
            "resolved_at": pd.to_datetime(
                ["2026-01-01 12:00", None]
            ),
        }
    )


def test_valid_dataset_passes_all_checks():
    df = create_sample_dataset()

    checks = validate_dataset(df)

    assert len(checks) == 10
    assert all(result["passed"] for result in checks.values())


def test_detects_missing_required_columns():
    df = create_sample_dataset().drop(columns=["priority"])

    checks = validate_dataset(df)

    assert not checks["required_columns"]["passed"]
    assert checks["required_columns"]["missing_columns"] == [
        "priority"
    ]
    assert len(checks) == 1


def test_detects_duplicate_ticket_ids():
    df = create_sample_dataset()
    df.loc[1, "ticket_id"] = "HSP-0001"

    checks = validate_dataset(df)

    assert not checks["unique_ticket_ids"]["passed"]
    assert checks["unique_ticket_ids"]["duplicates"] == 1


def test_detects_empty_descriptions():
    df = create_sample_dataset()
    df.loc[0, "description"] = "   "

    checks = validate_dataset(df)

    assert not checks["non_empty_descriptions"]["passed"]
    assert checks["non_empty_descriptions"]["invalid_count"] == 1


def test_detects_invalid_priorities():
    df = create_sample_dataset()
    df.loc[0, "priority"] = "Urgent"

    checks = validate_dataset(df)

    assert not checks["valid_priorities"]["passed"]
    assert checks["valid_priorities"]["unexpected_values"] == [
        "Urgent"
    ]


def test_detects_invalid_statuses():
    df = create_sample_dataset()
    df.loc[0, "status"] = "Pending"

    checks = validate_dataset(df)

    assert not checks["valid_statuses"]["passed"]
    assert checks["valid_statuses"]["unexpected_values"] == [
        "Pending"
    ]


def test_detects_incorrect_category_department_mapping():
    df = create_sample_dataset()
    df.loc[0, "department"] = "IT Support"

    checks = validate_dataset(df)

    assert not checks[
        "valid_category_department_mapping"
    ]["passed"]


def test_detects_invalid_date_order():
    df = create_sample_dataset()
    df.loc[0, "resolved_at"] = pd.Timestamp("2026-01-01 09:00")

    checks = validate_dataset(df)

    assert not checks["valid_date_order"]["passed"]


def test_unresolved_tickets_must_not_have_resolution_date():
    df = create_sample_dataset()
    df.loc[1, "resolved_at"] = pd.Timestamp("2026-01-02 14:00")

    checks = validate_dataset(df)

    result = checks["unresolved_tickets_have_no_resolution_date"]

    assert not result["passed"]
    assert result["invalid_count"] == 1


def test_resolved_tickets_must_have_resolution_date():
    df = create_sample_dataset()
    df.loc[0, "resolved_at"] = pd.NaT

    checks = validate_dataset(df)

    result = checks["resolved_tickets_have_resolution_date"]

    assert not result["passed"]
    assert result["invalid_count"] == 1


def test_detects_unexpected_categories():
    df = create_sample_dataset()
    df.loc[0, "category"] = "Unknown Category"

    checks = validate_dataset(df)

    assert not checks["valid_categories"]["passed"]
    assert checks["valid_categories"]["unexpected_values"] == [
        "Unknown Category"
    ]
