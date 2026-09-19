"""Tests for data ingestion module."""

import numpy as np
import pandas as pd
import pytest

from autoencoder.data.ingestion import detect_sensor_columns, validate_dataframe, read_parquet_robust


class TestReadParquetRobust:
    def test_matches_plain_read_parquet_no_list_columns(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
        path = tmp_path / "plain.parquet"
        df.to_parquet(path)
        result = read_parquet_robust(str(path))
        pd.testing.assert_frame_equal(result, pd.read_parquet(path))

    def test_matches_plain_read_parquet_with_list_columns(self, tmp_path):
        # Exercises the chunk-by-chunk list-column path (the actual fix);
        # at this scale plain pd.read_parquet also succeeds, so this checks
        # the workaround doesn't corrupt data on the happy path -- the real
        # ArrowNotImplementedError this works around only manifests at much
        # larger scale (pyarrow's internal ~128K-row chunking), impractical
        # to reproduce in a fast unit test.
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "tags": [["a", "b"], [], ["c"]],
        })
        path = tmp_path / "with_lists.parquet"
        df.to_parquet(path)
        result = read_parquet_robust(str(path))
        assert result["id"].tolist() == [1, 2, 3]
        assert [list(x) for x in result["tags"]] == [["a", "b"], [], ["c"]]

    def test_all_columns_present_regardless_of_position(self, tmp_path):
        # Column order isn't guaranteed to match the source exactly (list
        # columns are reassembled after scalar ones), but every column and
        # its values must still be present and correct -- this codebase
        # always accesses columns by name, never by position.
        df = pd.DataFrame({
            "z_col": [1, 2],
            "tags": [["a"], ["b"]],
            "a_col": [3, 4],
        })
        path = tmp_path / "order.parquet"
        df.to_parquet(path)
        result = read_parquet_robust(str(path))
        assert set(result.columns) == {"z_col", "tags", "a_col"}
        assert list(result["z_col"]) == [1, 2]
        assert list(result["a_col"]) == [3, 4]
        assert [list(x) for x in result["tags"]] == [["a"], ["b"]]


class TestDetectSensorColumns:
    def test_detects_numeric(self, synthetic_df, sensor_columns):
        detected = detect_sensor_columns(synthetic_df)
        assert detected == sorted(sensor_columns)

    def test_excludes_timestamp(self, synthetic_df):
        detected = detect_sensor_columns(synthetic_df)
        assert "timestamp" not in detected

    def test_excludes_custom(self, synthetic_df):
        detected = detect_sensor_columns(synthetic_df, exclude_columns=["sensor_00"])
        assert "sensor_00" not in detected


class TestValidateDataframe:
    def test_valid(self, synthetic_df, sensor_columns):
        errors = validate_dataframe(synthetic_df, expected_sensors=sensor_columns)
        assert errors == []

    def test_missing_timestamp(self, synthetic_df):
        errors = validate_dataframe(synthetic_df, timestamp_column="nonexistent")
        assert any("nonexistent" in e for e in errors)

    def test_missing_sensor(self, synthetic_df):
        errors = validate_dataframe(synthetic_df, expected_sensors=["sensor_00", "fake"])
        assert any("fake" in str(e) for e in errors)

    def test_non_numeric(self):
        df = pd.DataFrame({"timestamp": ["2025-01-01"], "s1": [1.0], "s2": ["text"]})
        errors = validate_dataframe(df, expected_sensors=["s1", "s2"])
        assert any("Non-numeric" in e for e in errors)

    def test_auto_detect(self):
        df = pd.DataFrame({"timestamp": ["2025-01-01"], "temp": [100.0], "pressure": [50.0]})
        errors = validate_dataframe(df)
        assert errors == []
