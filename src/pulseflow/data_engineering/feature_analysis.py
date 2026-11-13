
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_FILE = (
    PROJECT_ROOT / "data" / "processed" / "tickets_processed.parquet"
)
REPORT_DIR = PROJECT_ROOT / "data" / "interim" / "reports"
REPORT_FILE = REPORT_DIR / "feature_analysis_report.json"

REQUIRED_FEATURES = [
    "description_length",
    "description_word_count",
    "description_language",
    "has_location",
    "created_hour",
    "created_day_of_week",
    "created_is_weekend",
    "location",
]


def load_processed_dataset(
    file_path: Path = PROCESSED_FILE,
) -> pd.DataFrame:
    """Load the processed dataset from Parquet."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {file_path}. "
            "Run the data engineering pipeline first."
        )

    return pd.read_parquet(file_path)


def calculate_distribution(series: pd.Series) -> dict:
    """Calculate category counts and percentages."""
    values = series.astype("string").fillna("<MISSING>")
    counts = values.value_counts(dropna=False)

    total = len(values)

    return {
        str(value): {
            "count": int(count),
            "percentage": round(float(count / total * 100), 2),
        }
        for value, count in counts.items()
    }


def calculate_numeric_summary(series: pd.Series) -> dict:
    """Calculate descriptive statistics for a numeric feature."""
    summary = series.describe()

    return {
        str(metric): (
            None if pd.isna(value) else round(float(value), 2)
        )
        for metric, value in summary.items()
    }


def analyze_features(df: pd.DataFrame) -> dict:
    """Analyze distributions and statistics of engineered features."""
    missing_features = set(REQUIRED_FEATURES) - set(df.columns)

    if missing_features:
        raise ValueError(
            f"Missing engineered features: {sorted(missing_features)}"
        )

    total_tickets = len(df)

    if total_tickets == 0:
        raise ValueError("Cannot analyze an empty dataset.")

    report = {
        "dataset": {
            "rows": int(total_tickets),
            "columns": int(len(df.columns)),
        },
        "text_features": {
            "description_language": calculate_distribution(
                df["description_language"]
            ),
            "description_length": calculate_numeric_summary(
                df["description_length"]
            ),
            "description_word_count": calculate_numeric_summary(
                df["description_word_count"]
            ),
        },
        "location_feature": {
            "has_location": calculate_distribution(df["has_location"]),
            "missing_location_count": int(
                df["location"].isna().sum()
            ),
            "missing_location_percentage": round(
                float(df["location"].isna().mean() * 100), 2
            ),
        },
        "creation_time_features": {
            "created_hour": calculate_distribution(df["created_hour"]),
            "created_day_of_week": calculate_distribution(
                df["created_day_of_week"]
            ),
            "created_is_weekend": calculate_distribution(
                df["created_is_weekend"]
            ),
        },
    }

    return report


def save_feature_report(report: dict) -> Path:
    """Save feature analysis results as a JSON report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    return REPORT_FILE


def run_feature_analysis() -> dict:
    """Load processed data, analyze features, and save the report."""
    print("\n" + "=" * 60)
    print("PULSEFLOW AI - FEATURE ANALYSIS")
    print("=" * 60)

    print("\n[1/3] Loading processed dataset...")
    df = load_processed_dataset()
    print(f"Loaded successfully: {df.shape}")

    print("\n[2/3] Analyzing engineered features...")
    report = analyze_features(df)
    print("Feature analysis completed.")

    print("\nText language distribution:")
    for language, values in report["text_features"][
        "description_language"
    ].items():
        print(
            f"  {language}: {values['count']} "
            f"({values['percentage']}%)"
        )

    print("\nDescription length:")
    for metric, value in report["text_features"][
        "description_length"
    ].items():
        print(f"  {metric}: {value}")

    print("\nDescription word count:")
    for metric, value in report["text_features"][
        "description_word_count"
    ].items():
        print(f"  {metric}: {value}")

    print("\nLocation coverage:")
    location_report = report["location_feature"]
    print(
        f"  Has location: "
        f"{location_report['has_location'].get('True', {}).get('count', 0)}"
    )
    print(
        f"  Missing location: "
        f"{location_report['missing_location_count']} "
        f"({location_report['missing_location_percentage']}%)"
    )

    print("\nWeekend distribution:")
    for value, values in report["creation_time_features"][
        "created_is_weekend"
    ].items():
        print(
            f"  {value}: {values['count']} "
            f"({values['percentage']}%)"
        )

    print("\n[3/3] Saving feature analysis report...")
    report_path = save_feature_report(report)
    print(f"Report saved to: {report_path}")

    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS COMPLETED")
    print("=" * 60)

    return report


if __name__ == "__main__":
    run_feature_analysis()
