"""Tests for data ingestion module."""

import numpy as np
import pandas as pd
import pytest

from autoencoder.data.ingestion import detect_sensor_columns, validate_dataframe


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
