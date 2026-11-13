
import json

import pandas as pd
import pytest

import pulseflow.data_engineering.pipeline as pipeline


def record_call(calls, name, result):
    """Create a mock function that records its execution."""

    def mock_function(*args, **kwargs):
        calls.append(name)
        return result

    return mock_function


def test_save_quality_report(tmp_path, monkeypatch):
    checks = {
        "unique_ticket_ids": {
            "passed": True,
            "duplicates": 0,
        },
        "valid_priorities": {
            "passed": True,
            "unexpected_values": [],
        },
    }

    monkeypatch.setattr(
        pipeline,
        "REPORT_DIR",
        tmp_path / "reports",
    )

    report_path = pipeline.save_quality_report(checks)

    assert report_path.exists()
    assert report_path.name == "data_quality_report.json"

    with report_path.open("r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report == checks


def test_save_feature_validation_report(tmp_path, monkeypatch):
    checks = {
        "description_length_consistency": {
            "passed": True,
        },
        "valid_creation_hour": {
            "passed": True,
        },
    }

    monkeypatch.setattr(
        pipeline,
        "REPORT_DIR",
        tmp_path / "reports",
    )

    report_path = pipeline.save_feature_validation_report(checks)

    assert report_path.exists()
    assert report_path.name == (
        "processed_data_validation_report.json"
    )

    with report_path.open("r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report == checks


def test_run_pipeline_success(monkeypatch, tmp_path, capsys):
    calls = []

    raw_df = pd.DataFrame(
        {
            "ticket_id": ["HSP-0001"],
            "description": ["Network failure"],
        }
    )

    processed_df = pd.DataFrame(
        {
            "ticket_id": ["HSP-0001"],
            "description_length": [15],
        }
    )

    profile_report = {"dataset_shape": {"rows": 1}}
    quality_checks = {
        "unique_ticket_ids": {"passed": True},
        "valid_priorities": {"passed": True},
    }
    validation_checks = {
        "description_length_consistency": {"passed": True},
    }
    feature_report = {"features": ["description_length"]}

    monkeypatch.setattr(
        pipeline,
        "load_dataset",
        record_call(calls, "load_dataset", raw_df),
    )

    monkeypatch.setattr(
        pipeline,
        "profile_dataset",
        record_call(calls, "profile_dataset", profile_report),
    )

    monkeypatch.setattr(
        pipeline,
        "save_profile_report",
        record_call(calls, "save_profile_report", tmp_path / "profile.json"),
    )

    monkeypatch.setattr(
        pipeline,
        "validate_dataset",
        record_call(calls, "validate_dataset", quality_checks),
    )

    monkeypatch.setattr(
        pipeline,
        "save_quality_report",
        record_call(calls, "save_quality_report", tmp_path / "quality.json"),
    )

    monkeypatch.setattr(
        pipeline,
        "transform_dataset",
        record_call(calls, "transform_dataset", processed_df),
    )

    monkeypatch.setattr(
        pipeline,
        "validate_processed_dataset",
        record_call(
            calls,
            "validate_processed_dataset",
            validation_checks,
        ),
    )

    monkeypatch.setattr(
        pipeline,
        "save_feature_validation_report",
        record_call(
            calls,
            "save_feature_validation_report",
            tmp_path / "validation.json",
        ),
    )

    monkeypatch.setattr(
        pipeline,
        "save_processed_dataset",
        record_call(
            calls,
            "save_processed_dataset",
            tmp_path / "processed.parquet",
        ),
    )

    monkeypatch.setattr(
        pipeline,
        "analyze_features",
        record_call(calls, "analyze_features", feature_report),
    )

    monkeypatch.setattr(
        pipeline,
        "save_feature_report",
        record_call(
            calls,
            "save_feature_report",
            tmp_path / "features.json",
        ),
    )

    result = pipeline.run_pipeline()

    expected_calls = [
        "load_dataset",
        "profile_dataset",
        "save_profile_report",
        "validate_dataset",
        "save_quality_report",
        "transform_dataset",
        "validate_processed_dataset",
        "save_feature_validation_report",
        "save_processed_dataset",
        "analyze_features",
        "save_feature_report",
    ]

    assert calls == expected_calls
    pd.testing.assert_frame_equal(result, processed_df)

    captured = capsys.readouterr()

    assert "DATA ENGINEERING PIPELINE" in captured.out
    assert "DATA ENGINEERING PIPELINE COMPLETED" in captured.out


def test_run_pipeline_stops_when_quality_validation_fails(
    monkeypatch,
    tmp_path,
):
    raw_df = pd.DataFrame(
        {"ticket_id": ["HSP-0001"]}
    )

    quality_checks = {
        "unique_ticket_ids": {"passed": False},
        "valid_priorities": {"passed": True},
    }

    transform_called = False

    monkeypatch.setattr(
        pipeline,
        "load_dataset",
        lambda: raw_df,
    )

    monkeypatch.setattr(
        pipeline,
        "profile_dataset",
        lambda df: {},
    )

    monkeypatch.setattr(
        pipeline,
        "save_profile_report",
        lambda report: tmp_path / "profile.json",
    )

    monkeypatch.setattr(
        pipeline,
        "validate_dataset",
        lambda df: quality_checks,
    )

    monkeypatch.setattr(
        pipeline,
        "save_quality_report",
        lambda checks: tmp_path / "quality.json",
    )

    def mock_transform(df):
        nonlocal transform_called
        transform_called = True
        return df

    monkeypatch.setattr(
        pipeline,
        "transform_dataset",
        mock_transform,
    )

    with pytest.raises(
        ValueError,
        match="Data quality validation failed",
    ):
        pipeline.run_pipeline()

    assert transform_called is False


def test_run_pipeline_stops_when_processed_validation_fails(
    monkeypatch,
    tmp_path,
):
    raw_df = pd.DataFrame(
        {"ticket_id": ["HSP-0001"]}
    )

    processed_df = pd.DataFrame(
        {"description_length": [15]}
    )

    quality_checks = {
        "unique_ticket_ids": {"passed": True},
    }

    validation_checks = {
        "description_length_consistency": {"passed": False},
    }

    save_processed_called = False

    monkeypatch.setattr(
        pipeline,
        "load_dataset",
        lambda: raw_df,
    )

    monkeypatch.setattr(
        pipeline,
        "profile_dataset",
        lambda df: {},
    )

    monkeypatch.setattr(
        pipeline,
        "save_profile_report",
        lambda report: tmp_path / "profile.json",
    )

    monkeypatch.setattr(
        pipeline,
        "validate_dataset",
        lambda df: quality_checks,
    )

    monkeypatch.setattr(
        pipeline,
        "save_quality_report",
        lambda checks: tmp_path / "quality.json",
    )

    monkeypatch.setattr(
        pipeline,
        "transform_dataset",
        lambda df: processed_df,
    )

    monkeypatch.setattr(
        pipeline,
        "validate_processed_dataset",
        lambda df: validation_checks,
    )

    monkeypatch.setattr(
        pipeline,
        "save_feature_validation_report",
        lambda checks: tmp_path / "validation.json",
    )

    def mock_save_processed(df):
        nonlocal save_processed_called
        save_processed_called = True
        return tmp_path / "processed.parquet"

    monkeypatch.setattr(
        pipeline,
        "save_processed_dataset",
        mock_save_processed,
    )

    with pytest.raises(
        ValueError,
        match="Processed data validation failed",
    ):
        pipeline.run_pipeline()

    assert save_processed_called is False
