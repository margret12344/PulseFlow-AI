
from pathlib import Path

import pandas as pd

from pulseflow.data_engineering.profiling import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "tickets_processed.parquet"


def detect_language(text: str) -> str:
    """Detect the dominant language using Arabic character presence."""

    has_arabic = any(
        "\u0600" <= char <= "\u06FF"
        for char in text
    )

    has_english = any(
        ("a" <= char.lower() <= "z")
        for char in text
    )

    if has_arabic and has_english:
        return "Mixed"

    if has_arabic:
        return "Arabic"

    if has_english:
        return "English"

    return "Unknown"



def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create structured features without modifying the raw dataset."""
    processed = df.copy()

    # Text features
    processed["description_length"] = processed["description"].str.len()
    processed["description_word_count"] = (
        processed["description"].str.split().str.len()
    )
    processed["description_language"] = processed["description"].apply(
        detect_language
    )

    # Location feature
    processed["has_location"] = (
        processed["location"].notna()
        & processed["location"].str.strip().ne("")
    )

    # Ticket creation date features
    processed["created_year"] = processed["created_at"].dt.year
    processed["created_month"] = processed["created_at"].dt.month
    processed["created_day"] = processed["created_at"].dt.day
    processed["created_hour"] = processed["created_at"].dt.hour
    processed["created_day_of_week"] = processed["created_at"].dt.dayofweek

    # Friday = 4, Saturday = 5 (Egyptian weekend)
    processed["created_is_weekend"] = (
        processed["created_day_of_week"].isin([4, 5])
    )

    # Resolution time: analytical metric, NOT a prediction input
    processed["resolution_time_hours"] = (
        processed["resolved_at"] - processed["created_at"]
    ).dt.total_seconds() / 3600

    return processed



def save_processed_dataset(df: pd.DataFrame) -> Path:
    """Save the processed dataset in Parquet format."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_FILE, index=False)

    return OUTPUT_FILE


def run_cleaning_pipeline() -> pd.DataFrame:
    """Load, transform, and save the processed dataset."""

    raw_df = load_dataset()

    processed_df = transform_dataset(raw_df)

    output_path = save_processed_dataset(processed_df)

    print("=" * 60)
    print("PULSEFLOW AI - DATA TRANSFORMATION")
    print("=" * 60)

    print(f"Original shape: {raw_df.shape}")
    print(f"Processed shape: {processed_df.shape}")

    print("\nDetected languages:")
    print(
        processed_df["description_language"]
        .value_counts()
        .to_string()
    )

    print("\nResolution time summary:")
    print(
        processed_df["resolution_time_hours"]
        .describe()
        .to_string()
    )

    print(f"\nProcessed dataset saved to: {output_path}")

    return processed_df


if __name__ == "__main__":
    run_cleaning_pipeline()
