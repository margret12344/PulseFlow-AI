
import pandas as pd

REQUIRED_FEATURES = [
    "description_length",
    "description_word_count",
    "description_language",
    "has_location",
    "created_year",
    "created_month",
    "created_day",
    "created_hour",
    "created_day_of_week",
    "created_is_weekend",
    "resolution_time_hours",
]


def validate_processed_dataset(df: pd.DataFrame) -> dict:
    """Validate engineered features in the processed dataset."""

    missing_features = set(REQUIRED_FEATURES) - set(df.columns)

    if missing_features:
        raise ValueError(
            f"Missing engineered features: {sorted(missing_features)}"
        )

    checks = {}

    def add_check(name: str, invalid_count: int, details: str):
        checks[name] = {
            "passed": bool(invalid_count == 0),
            "invalid_count": int(invalid_count),
            "details": details,
        }

    # 1. Description length consistency
    expected_length = df["description"].str.len()
    invalid = df["description_length"].ne(expected_length)
    add_check(
        "description_length_consistency",
        invalid.sum(),
        "Description length must match the original text length.",
    )

    # 2. Description word count consistency
    expected_word_count = df["description"].str.split().str.len()
    invalid = df["description_word_count"].ne(expected_word_count)
    add_check(
        "description_word_count_consistency",
        invalid.sum(),
        "Word count must match the number of whitespace-separated words.",
    )

    # 3. Location feature consistency
    expected_location = (
        df["location"].notna()
        & df["location"].str.strip().ne("")
    )
    invalid = df["has_location"].ne(expected_location)
    add_check(
        "location_feature_consistency",
        invalid.sum(),
        "has_location must reflect whether a location is available.",
    )

    # 4. Creation hour range
    invalid = ~df["created_hour"].between(0, 23)
    add_check(
        "valid_creation_hour",
        invalid.sum(),
        "Creation hour must be between 0 and 23.",
    )

    # 5. Creation weekday range
    invalid = ~df["created_day_of_week"].between(0, 6)
    add_check(
        "valid_creation_day_of_week",
        invalid.sum(),
        "Creation weekday must be between 0 and 6.",
    )

    # 6. Weekend feature consistency
    expected_weekend = df["created_day_of_week"].isin([4, 5])
    invalid = df["created_is_weekend"].ne(expected_weekend)
    add_check(
        "weekend_feature_consistency",
        invalid.sum(),
        "Friday and Saturday must be marked as weekend days.",
    )

    # 7. Creation time feature consistency
    invalid = (
        df["created_hour"].ne(df["created_at"].dt.hour)
        | df["created_day_of_week"].ne(df["created_at"].dt.dayofweek)
    )
    add_check(
        "creation_time_consistency",
        invalid.sum(),
        "Derived time features must match created_at.",
    )

    # 8. Resolution time must not be negative
    invalid = (
        df["resolution_time_hours"].notna()
        & df["resolution_time_hours"].lt(0)
    )
    add_check(
        "non_negative_resolution_time",
        invalid.sum(),
        "Resolution time cannot be negative.",
    )

    # 9. Resolution time calculation consistency
    expected_resolution = (
        df["resolved_at"] - df["created_at"]
    ).dt.total_seconds() / 3600

    actual_resolution = df["resolution_time_hours"]

    matches = (
        actual_resolution.isna() & expected_resolution.isna()
    ) | (
        actual_resolution.notna()
        & expected_resolution.notna()
        & actual_resolution.sub(expected_resolution).abs().lt(1e-9)
    )

    add_check(
        "resolution_time_consistency",
        (~matches).sum(),
        "Resolution time must match resolved_at minus created_at.",
    )

    return checks


def run_feature_validation(df: pd.DataFrame) -> dict:
    """Run and print all processed dataset validation checks."""

    checks = validate_processed_dataset(df)

    passed = sum(
        result["passed"] for result in checks.values()
    )
    failed = len(checks) - passed

    print("\n" + "=" * 60)
    print("PULSEFLOW AI - PROCESSED DATA VALIDATION")
    print("=" * 60)

    for name, result in checks.items():
        status = "PASSED" if result["passed"] else "FAILED"
        print(
            f"{status}: {name} "
            f"(invalid rows: {result['invalid_count']})"
        )

    print("-" * 60)
    print(f"Passed: {passed}/{len(checks)}")
    print(f"Failed: {failed}/{len(checks)}")

    if failed:
        failed_checks = [
            name
            for name, result in checks.items()
            if not result["passed"]
        ]
        raise ValueError(
            f"Processed data validation failed: {failed_checks}"
        )

    return checks
