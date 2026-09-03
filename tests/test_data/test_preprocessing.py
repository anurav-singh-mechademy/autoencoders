"""Tests for preprocessing module -- scaling and window construction."""

import numpy as np
import pandas as pd
import pytest

from autoencoder.data.preprocessing import (
    apply_scaling,
    construct_windows,
    construct_windows_with_metadata,
    fit_robust_scaler,
    remove_low_variance_columns,
    WINDOW_ROWS,
)


class TestRemoveLowVarianceColumns:
    def test_removes_constant_columns(self, sensor_columns):
        n_rows = 500
        rng = np.random.default_rng(42)
        data = {"timestamp": pd.date_range("2025-01-01", periods=n_rows, freq="15s")}
        for col in sensor_columns:
            data[col] = rng.normal(50, 5, n_rows)
        # Make 2 columns constant
        data[sensor_columns[0]] = 1.0
        data[sensor_columns[1]] = 0.0
        df = pd.DataFrame(data)

        result = remove_low_variance_columns(df, sensor_columns)
        assert sensor_columns[0] in result.removed_columns
        assert sensor_columns[1] in result.removed_columns
        assert len(result.kept_columns) == len(sensor_columns) - 2

    def test_keeps_normal_columns(self, synthetic_df, sensor_columns):
        result = remove_low_variance_columns(synthetic_df, sensor_columns)
        assert len(result.removed_columns) == 0
        assert result.kept_columns == sensor_columns

    def test_near_constant_removed(self, sensor_columns):
        n_rows = 500
        rng = np.random.default_rng(42)
        data = {"timestamp": pd.date_range("2025-01-01", periods=n_rows, freq="15s")}
        for col in sensor_columns:
            data[col] = rng.normal(50, 5, n_rows)
        # Near-constant: variance ~1e-8
        data[sensor_columns[3]] = 5.0 + rng.normal(0, 1e-5, n_rows)
        df = pd.DataFrame(data)

        result = remove_low_variance_columns(df, sensor_columns, variance_threshold=1e-5)
        assert sensor_columns[3] in result.removed_columns

    def test_custom_threshold(self, synthetic_df, sensor_columns):
        # Very high threshold should remove everything
        result = remove_low_variance_columns(synthetic_df, sensor_columns, variance_threshold=1e6)
        assert len(result.removed_columns) == len(sensor_columns)
        assert len(result.kept_columns) == 0

    def test_variances_dict(self, synthetic_df, sensor_columns):
        result = remove_low_variance_columns(synthetic_df, sensor_columns)
        assert len(result.variances) == len(sensor_columns)
        for col in sensor_columns:
            assert col in result.variances
            assert result.variances[col] > 0


class TestRobustScaler:
    def test_fit_scaler(self, synthetic_df, sensor_columns):
        result = fit_robust_scaler(synthetic_df, sensor_columns)
        assert result.n_sensors == len(sensor_columns)
        assert result.medians.shape == (len(sensor_columns),)
        assert result.iqrs.shape == (len(sensor_columns),)
        assert result.sensor_columns == sensor_columns

    def test_apply_scaling_preserves_shape(self, synthetic_df, sensor_columns):
        scaler_result = fit_robust_scaler(synthetic_df, sensor_columns)
        scaled = apply_scaling(synthetic_df, scaler_result)
        assert scaled.shape == synthetic_df.shape
        assert "timestamp" in scaled.columns

    def test_scaled_values_differ(self, synthetic_df, sensor_columns):
        scaler_result = fit_robust_scaler(synthetic_df, sensor_columns)
        scaled = apply_scaling(synthetic_df, scaler_result)
        # Scaled median should be approximately 0
        for col in sensor_columns:
            assert abs(scaled[col].median()) < 0.5

    def test_non_sensor_columns_unchanged(self, synthetic_df, sensor_columns):
        scaler_result = fit_robust_scaler(synthetic_df, sensor_columns)
        scaled = apply_scaling(synthetic_df, scaler_result)
        pd.testing.assert_series_equal(scaled["timestamp"], synthetic_df["timestamp"])


class TestConstructWindows:
    def test_window_count(self, synthetic_df, sensor_columns):
        windows = construct_windows(synthetic_df, sensor_columns=sensor_columns)
        expected = len(synthetic_df) // WINDOW_ROWS
        assert len(windows) == expected

    def test_window_shape(self, synthetic_df, sensor_columns):
        windows = construct_windows(synthetic_df, sensor_columns=sensor_columns)
        assert windows[0].shape == (WINDOW_ROWS, len(sensor_columns))

    def test_drop_incomplete(self, sensor_columns):
        # 130 rows => 1 complete window, 10 leftover dropped
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=130, freq="15s"),
            **{col: np.random.randn(130) for col in sensor_columns},
        })
        windows = construct_windows(df, sensor_columns=sensor_columns, drop_incomplete=True)
        assert len(windows) == 1

    def test_keep_incomplete(self, sensor_columns):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=130, freq="15s"),
            **{col: np.random.randn(130) for col in sensor_columns},
        })
        windows = construct_windows(df, sensor_columns=sensor_columns, drop_incomplete=False)
        assert len(windows) == 2
        assert windows[1].shape[0] == 10


class TestConstructWindowsWithMetadata:
    def test_metadata_fields(self, synthetic_df, sensor_columns):
        windows = construct_windows_with_metadata(
            synthetic_df, sensor_columns=sensor_columns,
        )
        w = windows[0]
        assert "data" in w
        assert "start_time" in w
        assert "end_time" in w
        assert "window_id" in w
        assert w["window_id"] == 0
        assert w["data"].shape == (WINDOW_ROWS, len(sensor_columns))
