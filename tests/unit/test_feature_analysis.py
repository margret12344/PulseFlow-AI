
import json

import pandas as pd
import pytest

import pulseflow.data_engineering.feature_analysis as feature_analysis


def create_processed_dataset():
    """Create a small processed dataset for testing."""

    return pd.DataFrame(
        {
            "ticket_id": ["HSP-001", "HSP-002", "HSP-003"],
            "description_length": [10, 20, 30],
            "description_word_count": [2, 4, 6],
            "description_language": ["Arabic", "English", "Arabic"],
            "has_location": [True, False, True],
            "created_hour": [9, 10, 11],
            "created_day_of_week": [0, 1, 2],
            "created_is_weekend": [False, False, True],
            "location": ["ICU", None, "Emergency"],
        }
    )


def test_load_processed_dataset_success(tmp_path, monkeypatch):
    file_path = tmp_path / "tickets.parquet"
    file_path.touch()

    expected_df = create_processed_dataset()

    def mock_read_parquet(path):
        assert path == file_path
        return expected_df

    monkeypatch.setattr(
        feature_analysis.pd,
        "read_parquet",
        mock_read_parquet,
    )

    result = feature_analysis.load_processed_dataset(file_path)

    pd.testing.assert_frame_equal(result, expected_df)


def test_load_processed_dataset_file_not_found(tmp_path):
    file_path = tmp_path / "missing.parquet"

    with pytest.raises(
        FileNotFoundError,
        match="Run the data engineering pipeline first",
    ):
        feature_analysis.load_processed_dataset(file_path)


def test_calculate_distribution():
    series = pd.Series(
        ["Arabic", "English", "Arabic", None]
    )

    result = feature_analysis.calculate_distribution(series)

    assert result == {
        "Arabic": {
            "count": 2,
            "percentage": 50.0,
        },
        "English": {
            "count": 1,
            "percentage": 25.0,
        },
        "<MISSING>": {
            "count": 1,
            "percentage": 25.0,
        },
    }


def test_calculate_distribution_empty_series():
    series = pd.Series([], dtype="object")

    result = feature_analysis.calculate_distribution(series)

    assert result == {}


def test_calculate_numeric_summary():
    series = pd.Series([1, 2, 3])

    result = feature_analysis.calculate_numeric_summary(series)

    assert result == {
        "count": 3.0,
        "mean": 2.0,
        "std": 1.0,
        "min": 1.0,
        "25%": 1.5,
        "50%": 2.0,
        "75%": 2.5,
        "max": 3.0,
    }


def test_calculate_numeric_summary_empty_series():
    series = pd.Series([], dtype="float64")

    result = feature_analysis.calculate_numeric_summary(series)

    assert result["count"] == 0.0
    assert result["mean"] is None
    assert result["std"] is None
    assert result["min"] is None
    assert result["max"] is None


def test_analyze_features_success():
    df = create_processed_dataset()

    report = feature_analysis.analyze_features(df)

    assert report["dataset"] == {
        "rows": 3,
        "columns": 9,
    }

    assert report["text_features"]["description_language"] == {
        "Arabic": {
            "count": 2,
            "percentage": 66.67,
        },
        "English": {
            "count": 1,
            "percentage": 33.33,
        },
    }

    assert report["location_feature"]["missing_location_count"] == 1
    assert (
        report["location_feature"]["missing_location_percentage"]
        == 33.33
    )

    assert report["location_feature"]["has_location"]["True"]["count"] == 2

    assert (
        report["creation_time_features"]["created_is_weekend"]["False"][
            "count"
        ]
        == 2
    )


def test_analyze_features_missing_required_features():
    df = create_processed_dataset()

    df = df.drop(columns=["description_language"])

    with pytest.raises(
        ValueError,
        match="Missing engineered features",
    ):
        feature_analysis.analyze_features(df)


def test_analyze_features_missing_location():
    df = create_processed_dataset()

    df = df.drop(columns=["location"])

    with pytest.raises(
        ValueError,
        match="Missing engineered features",
    ):
        feature_analysis.analyze_features(df)


def test_analyze_features_empty_dataset():
    df = create_processed_dataset().iloc[0:0]

    with pytest.raises(
        ValueError,
        match="Cannot analyze an empty dataset",
    ):
        feature_analysis.analyze_features(df)


def test_save_feature_report(tmp_path, monkeypatch):
    report = {
        "dataset": {
            "rows": 3,
            "columns": 9,
        },
        "location_feature": {
            "missing_location_count": 1,
        },
    }

    report_dir = tmp_path / "reports"
    report_file = report_dir / "feature_analysis_report.json"

    monkeypatch.setattr(
        feature_analysis,
        "REPORT_DIR",
        report_dir,
    )

    monkeypatch.setattr(
        feature_analysis,
        "REPORT_FILE",
        report_file,
    )

    result = feature_analysis.save_feature_report(report)

    assert result == report_file
    assert report_file.exists()

    with report_file.open("r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report == report


def test_run_feature_analysis(monkeypatch, tmp_path, capsys):
    df = create_processed_dataset()

    report = {
        "text_features": {
            "description_language": {
                "Arabic": {
                    "count": 2,
                    "percentage": 66.67,
                },
                "English": {
                    "count": 1,
                    "percentage": 33.33,
                },
            },
            "description_length": {
                "count": 3.0,
                "mean": 20.0,
            },
            "description_word_count": {
                "count": 3.0,
                "mean": 4.0,
            },
        },
        "location_feature": {
            "has_location": {
                "True": {
                    "count": 2,
                    "percentage": 66.67,
                },
            },
            "missing_location_count": 1,
            "missing_location_percentage": 33.33,
        },
        "creation_time_features": {
            "created_is_weekend": {
                "False": {
                    "count": 2,
                    "percentage": 66.67,
                },
                "True": {
                    "count": 1,
                    "percentage": 33.33,
                },
            },
        },
    }

    report_path = tmp_path / "feature_analysis_report.json"

    monkeypatch.setattr(
        feature_analysis,
        "load_processed_dataset",
        lambda: df,
    )

    monkeypatch.setattr(
        feature_analysis,
        "analyze_features",
        lambda data: report,
    )

    monkeypatch.setattr(
        feature_analysis,
        "save_feature_report",
        lambda data: report_path,
    )

    result = feature_analysis.run_feature_analysis()

    assert result == report

    captured = capsys.readouterr()

    assert "FEATURE ANALYSIS" in captured.out
    assert "Arabic" in captured.out
    assert "Missing location: 1" in captured.out
    assert "FEATURE ANALYSIS COMPLETED" in captured.out
