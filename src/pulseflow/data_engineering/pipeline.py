
import json
from pathlib import Path

from pulseflow.data_engineering.cleaning import (
    save_processed_dataset,
    transform_dataset,
)
from pulseflow.data_engineering.feature_analysis import (
    analyze_features,
    save_feature_report,
)
from pulseflow.data_engineering.feature_validation import (
    validate_processed_dataset,
)
from pulseflow.data_engineering.profiling import (
    load_dataset,
    profile_dataset,
    save_profile_report,
)
from pulseflow.data_engineering.quality import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = PROJECT_ROOT / "data" / "interim" / "reports"


def save_quality_report(checks: dict) -> Path:
    """Save data quality validation results as JSON."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "data_quality_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(checks, file, ensure_ascii=False, indent=4)

    return report_path


def save_feature_validation_report(checks: dict) -> Path:
    """Save processed dataset validation results as JSON."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "processed_data_validation_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(checks, file, ensure_ascii=False, indent=4)

    return report_path


def run_pipeline():
    """Execute the complete data engineering pipeline."""

    print("\n" + "=" * 60)
    print("PULSEFLOW AI - DATA ENGINEERING PIPELINE")
    print("=" * 60)

    # Step 1: Data Ingestion
    print("\n[1/7] Loading raw dataset...")

    raw_df = load_dataset()

    print(f"Loaded successfully: {raw_df.shape}")

    # Step 2: Data Profiling
    print("\n[2/7] Running data profiling...")

    profile = profile_dataset(raw_df)
    profile_path = save_profile_report(profile)

    print("Profiling completed.")
    print(f"Report saved to: {profile_path}")

    # Step 3: Data Quality
    print("\n[3/7] Running data quality checks...")

    checks = validate_dataset(raw_df)
    quality_path = save_quality_report(checks)

    passed = sum(
        result["passed"]
        for result in checks.values()
    )
    failed = len(checks) - passed

    print(f"Passed: {passed}/{len(checks)}")
    print(f"Failed: {failed}/{len(checks)}")
    print(f"Report saved to: {quality_path}")

    if failed > 0:
        failed_checks = [
            name
            for name, result in checks.items()
            if not result["passed"]
        ]

        raise ValueError(
            f"Data quality validation failed: {failed_checks}"
        )

    # Step 4: Data Transformation
    print("\n[4/7] Transforming dataset...")

    processed_df = transform_dataset(raw_df)

    print(f"Transformation completed: {processed_df.shape}")

    # Step 5: Processed Data Validation
    print("\n[5/7] Validating processed dataset...")

    validation_checks = validate_processed_dataset(processed_df)

    validation_path = save_feature_validation_report(
        validation_checks
    )

    passed = sum(
        result["passed"]
        for result in validation_checks.values()
    )
    failed = len(validation_checks) - passed

    print(f"Passed: {passed}/{len(validation_checks)}")
    print(f"Failed: {failed}/{len(validation_checks)}")
    print(f"Report saved to: {validation_path}")

    if failed > 0:
        failed_checks = [
            name
            for name, result in validation_checks.items()
            if not result["passed"]
        ]

        raise ValueError(
            f"Processed data validation failed: {failed_checks}"
        )

    # Step 6: Save Processed Dataset
    print("\n[6/7] Saving processed dataset...")

    output_path = save_processed_dataset(processed_df)

    print(f"Saved to: {output_path}")

    # Step 7: Feature Analysis
    print("\n[7/7] Analyzing engineered features...")

    feature_report = analyze_features(processed_df)
    feature_report_path = save_feature_report(feature_report)

    print("Feature analysis completed.")
    print(f"Report saved to: {feature_report_path}")

    print("\n" + "=" * 60)
    print("DATA ENGINEERING PIPELINE COMPLETED")
    print("=" * 60)

    return processed_df


if __name__ == "__main__":
    run_pipeline()
