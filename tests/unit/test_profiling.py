
import json

import pandas as pd
import pytest

import pulseflow.data_engineering.profiling as profiling


def create_sample_dataset():
    """Create a small dataset for profiling tests."""

    return pd.DataFrame(
        {
            "ticket_id": [
                "HSP-0001",
                "HSP-0002",
                "HSP-0003",
            ],
            "description": [
                "Network is not working",
                "Printer connection failed",
                "Cannot access the system",
            ],
            "category": [
                "Network",
                "Software Failure",
                "Access/Permissions",
            ],
            "department": [
                "Network Team",
                "IT Support",
                "Security & Access",
            ],
            "priority": [
                "High",
                "Medium",
                "Low",
            ],
            "location": [
                "ICU",
                None,
                "Emergency",
            ],
            "status": [
                "Resolved",
                "Open",
                "Closed",
            ],
            "created_at": pd.to_datetime(
                [
                    "2026-01-01 10:00",
                    "2026-01-02 12:00",
                    "2026-01-03 08:00",
                ]
            ),
            "resolved_at": pd.to_datetime(
                [
                    "2026-01-01 12:00",
                    None,
                    "2026-01-03 10:00",
                ]
            ),
        }
    )


def test_load_dataset_success(tmp_path):
    df = create_sample_dataset()
    file_path = tmp_path / "tickets.xlsx"

    df.to_excel(
        file_path,
        sheet_name="tickets",
        index=False,
    )

    loaded_df = profiling.load_dataset(file_path)

    assert loaded_df.shape == (3, 9)
    assert loaded_df["ticket_id"].tolist() == [
        "HSP-0001",
        "HSP-0002",
        "HSP-0003",
    ]


def test_load_dataset_file_not_found(tmp_path):
    file_path = tmp_path / "missing.xlsx"

    with pytest.raises(
        FileNotFoundError,
        match="Dataset not found",
    ):
        profiling.load_dataset(file_path)


def test_load_dataset_missing_sheet(tmp_path):
    file_path = tmp_path / "tickets.xlsx"

    pd.DataFrame(
        {"ticket_id": ["HSP-0001"]}
    ).to_excel(
        file_path,
        sheet_name="other_sheet",
        index=False,
    )

    with pytest.raises(ValueError):
        profiling.load_dataset(file_path)


def test_validate_columns_success():
    df = create_sample_dataset()

    profiling.validate_columns(df)


def test_validate_columns_detects_missing_columns():
    df = create_sample_dataset().drop(
        columns=["priority", "location"]
    )

    with pytest.raises(
        ValueError,
        match="Dataset is missing expected columns",
    ):
        profiling.validate_columns(df)


def test_profile_dataset_shape():
    df = create_sample_dataset()

    report = profiling.profile_dataset(df)

    assert report["dataset_shape"] == {
        "rows": 3,
        "columns": 9,
    }


def test_profile_dataset_columns_and_data_types():
    df = create_sample_dataset()

    report = profiling.profile_dataset(df)

    assert report["columns"] == df.columns.tolist()
    assert report["data_types"]["ticket_id"] == "object"
    assert report["data_types"]["created_at"] == "datetime64[ns]"


def test_profile_dataset_missing_values():
    df = create_sample_dataset()

    report = profiling.profile_dataset(df)

    assert report["missing_values"]["location"] == 1
    assert report["missing_values"]["resolved_at"] == 1


def test_profile_dataset_duplicate_rows_and_ticket_ids():
    df = create_sample_dataset()

    df.loc[2] = df.loc[1]
    df["ticket_id"] = "HSP-0001"

    report = profiling.profile_dataset(df)

    assert report["duplicate_rows"] == 1
    assert report["duplicate_ticket_ids"] == 2


def test_profile_dataset_value_distributions():
    df = create_sample_dataset()

    report = profiling.profile_dataset(df)

    assert report["value_distributions"]["priority"] == {
        "High": 1,
        "Medium": 1,
        "Low": 1,
    }

    assert report["value_distributions"]["category"]["Network"] == 1


def test_profile_dataset_missing_value_distribution():
    df = create_sample_dataset()

    report = profiling.profile_dataset(df)

    assert report["value_distributions"]["location"]["<MISSING>"] == 1


def test_profile_dataset_invalid_date_order():
    df = create_sample_dataset()

    df.loc[0, "resolved_at"] = pd.Timestamp(
        "2026-01-01 09:00"
    )

    report = profiling.profile_dataset(df)

    assert report["invalid_date_order"] == 1


def test_profile_dataset_resolution_time_summary():
    df = create_sample_dataset()

    report = profiling.profile_dataset(df)

    summary = report["resolution_time_hours"]

    assert summary["count"] == 2.0
    assert summary["mean"] == 2.0
    assert summary["min"] == 2.0
    assert summary["max"] == 2.0


def test_profile_dataset_missing_resolution_times():
    df = create_sample_dataset()

    df["resolved_at"] = pd.NaT

    report = profiling.profile_dataset(df)

    summary = report["resolution_time_hours"]

    assert summary["count"] == 0.0
    assert summary["mean"] is None
    assert summary["min"] is None
    assert summary["max"] is None


def test_profile_dataset_empty_dataset():
    df = create_sample_dataset().iloc[0:0].copy()

    report = profiling.profile_dataset(df)

    assert report["dataset_shape"]["rows"] == 0
    assert report["duplicate_rows"] == 0
    assert report["duplicate_ticket_ids"] == 0


def test_profile_dataset_does_not_modify_original_data():
    df = create_sample_dataset()
    original_df = df.copy(deep=True)

    profiling.profile_dataset(df)

    pd.testing.assert_frame_equal(df, original_df)


def test_save_profile_report(tmp_path, monkeypatch):
    df = create_sample_dataset()
    report = profiling.profile_dataset(df)

    monkeypatch.setattr(
        profiling,
        "REPORT_DIR",
        tmp_path / "reports",
    )

    report_path = profiling.save_profile_report(report)

    assert report_path.exists()
    assert report_path.name == "data_profiling_report.json"

    with report_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_report = json.load(file)

    assert saved_report["dataset_shape"] == {
        "rows": 3,
        "columns": 9,
    }


def test_save_profile_report_creates_valid_json_for_missing_times(
    tmp_path,
    monkeypatch,
):
    df = create_sample_dataset()
    df["resolved_at"] = pd.NaT

    report = profiling.profile_dataset(df)

    monkeypatch.setattr(
        profiling,
        "REPORT_DIR",
        tmp_path / "reports",
    )

    report_path = profiling.save_profile_report(report)

    with report_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_report = json.load(file)

    assert saved_report["resolution_time_hours"]["mean"] is None


def test_run_profiling(monkeypatch, tmp_path, capsys):
    df = create_sample_dataset()

    monkeypatch.setattr(
        profiling,
        "load_dataset",
        lambda: df,
    )

    monkeypatch.setattr(
        profiling,
        "REPORT_DIR",
        tmp_path / "reports",
    )

    report = profiling.run_profiling()

    captured = capsys.readouterr()

    assert report["dataset_shape"]["rows"] == 3
    assert "DATA PROFILING REPORT" in captured.out
    assert "Report saved to:" in captured.out
