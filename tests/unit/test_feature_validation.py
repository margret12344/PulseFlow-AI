
import pandas as pd
import pytest

from pulseflow.data_engineering.feature_validation import (
    validate_processed_dataset,
)


def create_sample_processed_dataset():
    return pd.DataFrame(
        {
            "ticket_id": ["HSP-0001", "HSP-0002"],
            "description": [
                "The network is not working",
                "مشكلة في الشبكة Network",
            ],
            "category": ["Network", "Network"],
            "department": ["Network Team", "Network Team"],
            "priority": ["High", "Medium"],
            "location": ["ICU", None],
            "status": ["Closed", "Open"],
            "created_at": pd.to_datetime(
                ["2026-01-01 10:00", "2026-01-02 12:00"]
            ),
            "resolved_at": pd.to_datetime(
                ["2026-01-01 12:00", None]
            ),
            "description_length": [26, 23],
            "description_word_count": [5, 4],
            "description_language": ["English", "Mixed"],
            "has_location": [True, False],
            "created_year": [2026, 2026],
            "created_month": [1, 1],
            "created_day": [1, 2],
            "created_hour": [10, 12],
            "created_day_of_week": [3, 4],
            "created_is_weekend": [False, True],
            "resolution_time_hours": [2.0, None],
        }
    )


def test_valid_processed_dataset_passes_all_checks():
    df = create_sample_processed_dataset()

    checks = validate_processed_dataset(df)

    failed_checks = {
        name: result
        for name, result in checks.items()
        if not result["passed"]
    }

    assert len(checks) == 9, checks
    assert not failed_checks, failed_checks


def test_detects_incorrect_description_length():
    df = create_sample_processed_dataset()
    df.loc[0, "description_length"] = 100

    checks = validate_processed_dataset(df)

    assert not checks["description_length_consistency"]["passed"]
    assert checks["description_length_consistency"]["invalid_count"] == 1


def test_detects_incorrect_word_count():
    df = create_sample_processed_dataset()
    df.loc[0, "description_word_count"] = 99

    checks = validate_processed_dataset(df)

    assert not checks["description_word_count_consistency"]["passed"]


def test_detects_incorrect_location_feature():
    df = create_sample_processed_dataset()
    df.loc[0, "has_location"] = False

    checks = validate_processed_dataset(df)

    assert not checks["location_feature_consistency"]["passed"]


def test_detects_invalid_creation_hour():
    df = create_sample_processed_dataset()
    df.loc[0, "created_hour"] = 25

    checks = validate_processed_dataset(df)

    assert not checks["valid_creation_hour"]["passed"]


def test_detects_invalid_day_of_week():
    df = create_sample_processed_dataset()
    df.loc[0, "created_day_of_week"] = 7

    checks = validate_processed_dataset(df)

    assert not checks["valid_creation_day_of_week"]["passed"]


def test_detects_incorrect_weekend_feature():
    df = create_sample_processed_dataset()
    df.loc[1, "created_is_weekend"] = False

    checks = validate_processed_dataset(df)

    assert not checks["weekend_feature_consistency"]["passed"]


def test_detects_incorrect_creation_time_features():
    df = create_sample_processed_dataset()
    df.loc[0, "created_hour"] = 15

    checks = validate_processed_dataset(df)

    assert not checks["creation_time_consistency"]["passed"]


def test_detects_negative_resolution_time():
    df = create_sample_processed_dataset()
    df.loc[0, "resolution_time_hours"] = -2.0

    checks = validate_processed_dataset(df)

    assert not checks["non_negative_resolution_time"]["passed"]


def test_detects_incorrect_resolution_time():
    df = create_sample_processed_dataset()
    df.loc[0, "resolution_time_hours"] = 5.0

    checks = validate_processed_dataset(df)

    assert not checks["resolution_time_consistency"]["passed"]


def test_rejects_missing_required_feature():
    df = create_sample_processed_dataset()
    df = df.drop(columns=["created_hour"])

    with pytest.raises(
        ValueError,
        match="Missing engineered features",
    ):
        validate_processed_dataset(df)
