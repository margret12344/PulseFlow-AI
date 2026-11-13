
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "tickets_clean.xlsx"
REPORT_DIR = PROJECT_ROOT / "data" / "interim" / "reports"


EXPECTED_COLUMNS = [
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

CATEGORICAL_COLUMNS = [
    "category",
    "department",
    "priority",
    "location",
    "status",
]


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure that all expected dataset columns are present."""

    missing_columns = sorted(
        set(EXPECTED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Dataset is missing expected columns: {missing_columns}"
        )


def load_dataset(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw ticket dataset without modifying the source file."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    return pd.read_excel(
        file_path,
        sheet_name="tickets",
    )


def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate a comprehensive data profiling report."""

    validate_columns(df)

    missing_values = df.isna().sum()

    duplicate_rows = int(df.duplicated().sum())
    duplicate_ticket_ids = int(
        df["ticket_id"].duplicated().sum()
    )

    value_distributions = {}

    for column in CATEGORICAL_COLUMNS:
        distribution = {}

        for value, count in df[column].value_counts(
            dropna=False
        ).items():
            key = "<MISSING>" if pd.isna(value) else value

            if hasattr(key, "item"):
                key = key.item()

            distribution[key] = int(count)

        value_distributions[column] = distribution

    invalid_date_order = int(
        (
            df["resolved_at"].notna()
            & df["created_at"].notna()
            & (df["resolved_at"] < df["created_at"])
        ).sum()
    )

    resolution_time = (
        df["resolved_at"] - df["created_at"]
    ).dt.total_seconds() / 3600

    resolution_time_summary = {
        metric: (
            None if pd.isna(value) else float(value)
        )
        for metric, value in resolution_time.describe().items()
    }

    report = {
        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": df.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(count)
            for column, count in missing_values.items()
        },
        "duplicate_rows": duplicate_rows,
        "duplicate_ticket_ids": duplicate_ticket_ids,
        "value_distributions": value_distributions,
        "invalid_date_order": invalid_date_order,
        "resolution_time_hours": resolution_time_summary,
    }

    return report


def save_profile_report(report: dict) -> Path:
    """Save the profiling report as a JSON file."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = REPORT_DIR / "data_profiling_report.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
            allow_nan=False,
        )

    return report_path


def run_profiling() -> dict:
    """Run the complete dataset profiling process."""

    df = load_dataset()

    report = profile_dataset(df)
    report_path = save_profile_report(report)

    print("=" * 60)
    print("PULSEFLOW AI - DATA PROFILING REPORT")
    print("=" * 60)

    print(f"Dataset shape: {report['dataset_shape']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"Duplicate ticket IDs: {report['duplicate_ticket_ids']}")
    print(f"Invalid date order: {report['invalid_date_order']}")

    print("\nMissing values:")
    for column, count in report["missing_values"].items():
        print(f"  {column}: {count}")

    print("\nValue distributions:")
    for column, distribution in report["value_distributions"].items():
        print(f"\n{column}:")
        for value, count in distribution.items():
            print(f"  {value}: {count}")

    print("\nResolution time (hours):")
    for metric, value in report["resolution_time_hours"].items():
        print(f"  {metric}: {value}")

    print(f"\nReport saved to: {report_path}")

    return report


if __name__ == "__main__":
    run_profiling()
