
from pulseflow.data_engineering.profiling import load_dataset

CATEGORY_DEPARTMENT_MAPPING = {
    "Software Failure": "IT Support",
    "Facilities": "Maintenance",
    "Medical Equipment Failure": "Biomedical Engineering",
    "Network": "Network Team",
    "Access/Permissions": "Security & Access",
    "Billing": "Finance",
}

ALLOWED_PRIORITIES = {
    "Critical",
    "High",
    "Medium",
    "Low",
}

ALLOWED_STATUSES = {
    "Resolved",
    "Closed",
    "Open",
    "In Progress",
    "Assigned",
}

UNRESOLVED_STATUSES = {
    "Open",
    "In Progress",
    "Assigned",
}

REQUIRED_COLUMNS = [
    "ticket_id",
    "description",
    "category",
    "department",
    "priority",
    "location",
    "status",
    "created_at",
    "resolved_at",
]


def validate_dataset(df):
    """Validate the dataset against business and structural rules."""

    checks = {}

    # 1. Required columns
    missing_columns = sorted(
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    checks["required_columns"] = {
        "passed": bool(not missing_columns),
        "missing_columns": missing_columns,
    }

    if missing_columns:
        return checks

    # 2. Unique ticket IDs
    duplicate_ticket_ids = df["ticket_id"].duplicated().sum()

    checks["unique_ticket_ids"] = {
        "passed": bool(duplicate_ticket_ids == 0),
        "duplicates": int(duplicate_ticket_ids),
    }

    # 3. Non-empty descriptions
    invalid_descriptions = (
        df["description"].isna()
        | df["description"].astype(str).str.strip().eq("")
    )

    checks["non_empty_descriptions"] = {
        "passed": bool(not invalid_descriptions.any()),
        "invalid_count": int(invalid_descriptions.sum()),
    }

    # 4. Valid priorities
    unexpected_priorities = (
        set(df["priority"].dropna()) - ALLOWED_PRIORITIES
    )

    checks["valid_priorities"] = {
        "passed": bool(not unexpected_priorities),
        "unexpected_values": sorted(unexpected_priorities),
    }

    # 5. Valid statuses
    unexpected_statuses = (
        set(df["status"].dropna()) - ALLOWED_STATUSES
    )

    checks["valid_statuses"] = {
        "passed": bool(not unexpected_statuses),
        "unexpected_values": sorted(unexpected_statuses),
    }

    # 6. Category and department mapping
    mapping_valid = all(
        df.loc[
            df["category"] == category,
            "department",
        ].eq(department).all()
        for category, department in CATEGORY_DEPARTMENT_MAPPING.items()
    )

    checks["valid_category_department_mapping"] = {
        "passed": bool(mapping_valid),
    }

    # 7. Valid date order
    invalid_date_order = (
        df["resolved_at"].notna()
        & (df["resolved_at"] < df["created_at"])
    )

    checks["valid_date_order"] = {
        "passed": bool(not invalid_date_order.any()),
    }

    # 8. Unresolved tickets must not have resolution dates
    unresolved = df["status"].isin(UNRESOLVED_STATUSES)

    invalid_unresolved_dates = (
        df.loc[unresolved, "resolved_at"].notna().sum()
    )

    checks["unresolved_tickets_have_no_resolution_date"] = {
        "passed": bool(invalid_unresolved_dates == 0),
        "invalid_count": int(invalid_unresolved_dates),
    }

    # 9. Resolved tickets must have resolution dates
    resolved = ~unresolved

    invalid_resolved_dates = (
        df.loc[resolved, "resolved_at"].isna().sum()
    )

    checks["resolved_tickets_have_resolution_date"] = {
        "passed": bool(invalid_resolved_dates == 0),
        "invalid_count": int(invalid_resolved_dates),
    }

    # 10. Valid categories
    unexpected_categories = (
        set(df["category"].dropna())
        - set(CATEGORY_DEPARTMENT_MAPPING)
    )

    checks["valid_categories"] = {
        "passed": bool(not unexpected_categories),
        "unexpected_values": sorted(unexpected_categories),
    }

    return checks


def run_quality_checks():
    """Run and display all data quality checks."""

    df = load_dataset()
    checks = validate_dataset(df)

    print("=" * 60)
    print("PULSEFLOW AI - DATA QUALITY REPORT")
    print("=" * 60)

    passed_count = 0

    for check_name, result in checks.items():
        passed = result["passed"]

        if passed:
            passed_count += 1

        print(
            f"\n{check_name}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

        for key, value in result.items():
            if key != "passed":
                print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print(f"Passed: {passed_count}/{len(checks)}")
    print(f"Failed: {len(checks) - passed_count}/{len(checks)}")
    print("=" * 60)

    return checks


if __name__ == "__main__":
    run_quality_checks()
