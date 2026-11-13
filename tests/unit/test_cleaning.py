
import pandas as pd
import pandas.testing as pdt

import pulseflow.data_engineering.cleaning as cleaning
from pulseflow.data_engineering.cleaning import (
    detect_language,
    transform_dataset,
)


def create_sample_dataset():
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
        }
    )


def test_detect_language():
    assert detect_language("Hello world") == "English"
    assert detect_language("مرحبا بالعالم") == "Arabic"
    assert detect_language("Hello مرحبا") == "Mixed"
    assert detect_language("12345") == "Unknown"


def test_transform_preserves_all_rows():
    raw_df = create_sample_dataset()

    processed_df = transform_dataset(raw_df)

    assert len(processed_df) == len(raw_df)


def test_transform_adds_expected_features():
    raw_df = create_sample_dataset()

    processed_df = transform_dataset(raw_df)

    expected_features = {
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
    }

    assert expected_features.issubset(processed_df.columns)


def test_resolution_time_calculation():
    raw_df = create_sample_dataset()

    processed_df = transform_dataset(raw_df)

    assert processed_df.loc[0, "resolution_time_hours"] == 2.0
    assert pd.isna(
        processed_df.loc[1, "resolution_time_hours"]
    )


def test_transform_does_not_modify_raw_data():
    raw_df = create_sample_dataset()
    original_df = raw_df.copy(deep=True)

    transform_dataset(raw_df)

    pdt.assert_frame_equal(raw_df, original_df)


def test_missing_location_is_preserved():
    raw_df = create_sample_dataset()

    processed_df = transform_dataset(raw_df)

    assert pd.isna(processed_df.loc[1, "location"])


def test_text_and_location_features():
    raw_df = create_sample_dataset()

    processed_df = transform_dataset(raw_df)

    assert processed_df.loc[0, "description_word_count"] == 5
    assert processed_df.loc[0, "has_location"]
    assert not processed_df.loc[1, "has_location"]


def test_creation_time_features():
    raw_df = create_sample_dataset()

    processed_df = transform_dataset(raw_df)

    assert processed_df.loc[0, "created_hour"] == 10
    assert processed_df.loc[0, "created_day_of_week"] == 3

    # 2026-01-02 is Friday
    assert processed_df.loc[1, "created_day_of_week"] == 4
    assert processed_df.loc[1, "created_is_weekend"]




def test_save_processed_dataset(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "ticket_id": ["HSP-001", "HSP-002"],
            "description": ["Network failure", "Software error"],
        }
    )

    output_dir = tmp_path / "processed"
    output_file = output_dir / "tickets_processed.parquet"

    monkeypatch.setattr(cleaning, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(cleaning, "OUTPUT_FILE", output_file)

    result = cleaning.save_processed_dataset(df)

    assert result == output_file
    assert output_file.exists()

    saved_df = pd.read_parquet(output_file)

    pd.testing.assert_frame_equal(saved_df, df)


def test_run_cleaning_pipeline(tmp_path, monkeypatch, capsys):
    raw_df = pd.DataFrame(
        {
            "ticket_id": ["HSP-001", "HSP-002"],
            "description": [
                "Network failure",
                "مشكلة في الشبكة",
            ],
            "location": ["ICU", None],
            "created_at": pd.to_datetime(
                ["2026-01-01 09:00:00", "2026-01-02 10:00:00"]
            ),
            "resolved_at": pd.to_datetime(
                ["2026-01-01 11:00:00", None]
            ),
        }
    )

    output_dir = tmp_path / "processed"
    output_file = output_dir / "tickets_processed.parquet"

    monkeypatch.setattr(
        cleaning,
        "load_dataset",
        lambda: raw_df,
    )

    monkeypatch.setattr(
        cleaning,
        "OUTPUT_DIR",
        output_dir,
    )

    monkeypatch.setattr(
        cleaning,
        "OUTPUT_FILE",
        output_file,
    )

    result = cleaning.run_cleaning_pipeline()

    assert result.shape == (2, 16)

    assert result["description_language"].tolist() == [
        "English",
        "Arabic",
    ]

    assert result["has_location"].tolist() == [True, False]

    assert result["resolution_time_hours"].iloc[0] == 2.0
    assert pd.isna(result["resolution_time_hours"].iloc[1])

    assert output_file.exists()

    saved_df = pd.read_parquet(output_file)

    pd.testing.assert_frame_equal(saved_df, result)

    captured = capsys.readouterr()

    assert "DATA TRANSFORMATION" in captured.out
    assert "Original shape:" in captured.out
    assert "Processed shape:" in captured.out
    assert "Processed dataset saved to:" in captured.out
