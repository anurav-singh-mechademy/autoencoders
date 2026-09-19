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
    remove_null_or_stuck_columns,
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


class TestRemoveNullOrStuckColumns:
    """A sensor flat/missing in MOST windows can still have healthy
    whole-file variance if it has a few genuine excursions spread across a
    long series -- exactly the case remove_low_variance_columns can't
    catch. These tests build that scenario directly: 10 windows of 120 rows
    (synthetic_df/sensor_columns from conftest), one sensor flat in 9 of 10
    windows with a single real excursion in the 10th.

    Null and stuck are independent, separately-thresholded checks: by
    default (max_stuck_pct=100.0), a column is NEVER dropped for merely
    being stuck -- a window-fraction can't exceed 100%, so only a genuinely
    null-dominated column gets removed unless max_stuck_pct is explicitly
    lowered."""

    def test_keeps_healthy_columns(self, synthetic_df, sensor_columns):
        result = remove_null_or_stuck_columns(synthetic_df, sensor_columns, window_size=WINDOW_ROWS)
        assert result.removed_columns == []
        assert result.kept_columns == sensor_columns

    def test_default_never_drops_for_stuck_alone(self, synthetic_df, sensor_columns):
        df = synthetic_df.copy()
        target = sensor_columns[0]
        values = df[target].to_numpy().copy()
        values[:] = 42.0  # stuck in EVERY window (100%)
        df[target] = values

        # Confirm the premise: the low-variance filter does NOT catch this
        # (a perfectly constant column IS actually near-zero variance --
        # this specific case would be caught by remove_low_variance_columns;
        # the point here is remove_null_or_stuck_columns itself, at its
        # default max_stuck_pct=100, never removes for stuck regardless).
        result = remove_null_or_stuck_columns(df, sensor_columns, window_size=WINDOW_ROWS)
        assert target not in result.removed_columns
        assert result.stuck_pct[target] == pytest.approx(100.0)

    def test_removes_column_null_in_most_windows(self, synthetic_df, sensor_columns):
        df = synthetic_df.copy()
        target = sensor_columns[1]
        values = df[target].to_numpy().copy()
        values[: 6 * WINDOW_ROWS] = np.nan  # null in windows 0-5 (60%)
        df[target] = values

        result = remove_null_or_stuck_columns(df, sensor_columns, window_size=WINDOW_ROWS, max_null_pct=50.0)
        assert target in result.removed_columns
        assert result.null_pct[target] == pytest.approx(60.0)

    def test_explicit_max_stuck_pct_still_removes_frequently_stuck_column(self, synthetic_df, sensor_columns):
        df = synthetic_df.copy()
        target = sensor_columns[0]
        values = df[target].to_numpy().copy()
        for w in range(9):
            values[w * WINDOW_ROWS:(w + 1) * WINDOW_ROWS] = 42.0
        df[target] = values

        # 90% stuck: default (max_stuck_pct=100) keeps it...
        default_result = remove_null_or_stuck_columns(df, sensor_columns, window_size=WINDOW_ROWS)
        assert target not in default_result.removed_columns
        assert default_result.stuck_pct[target] == pytest.approx(90.0)
        # ...but an explicit, lower max_stuck_pct still removes it.
        strict_result = remove_null_or_stuck_columns(df, sensor_columns, window_size=WINDOW_ROWS, max_stuck_pct=50.0)
        assert target in strict_result.removed_columns

    def test_pct_reported_for_every_column(self, synthetic_df, sensor_columns):
        result = remove_null_or_stuck_columns(synthetic_df, sensor_columns, window_size=WINDOW_ROWS)
        assert set(result.null_pct.keys()) == set(sensor_columns)
        assert set(result.stuck_pct.keys()) == set(sensor_columns)
        assert all(p == pytest.approx(0.0) for p in result.null_pct.values())
        assert all(p == pytest.approx(0.0) for p in result.stuck_pct.values())


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

    def test_tail_compression_none_matches_default(self, synthetic_df, sensor_columns):
        scaler_result = fit_robust_scaler(synthetic_df, sensor_columns)
        scaled_default = apply_scaling(synthetic_df, scaler_result)
        scaled_explicit_none = apply_scaling(synthetic_df, scaler_result, tail_compression_scale=None)
        pd.testing.assert_frame_equal(scaled_default, scaled_explicit_none)

    def test_tail_compression_near_identity_for_small_values(self, synthetic_df, sensor_columns):
        # asinh(x) ~= x for |x| << c, so well-behaved (non-outlier) sensor
        # readings should barely move.
        scaler_result = fit_robust_scaler(synthetic_df, sensor_columns)
        uncompressed = apply_scaling(synthetic_df, scaler_result)
        compressed = apply_scaling(synthetic_df, scaler_result, tail_compression_scale=20.0)
        diff = (compressed[sensor_columns] - uncompressed[sensor_columns]).abs()
        assert (diff < 0.05).values.all()

    def test_tail_compression_bounds_extreme_outliers(self, synthetic_df, sensor_columns):
        # A glitch reading thousands of sigma out should compress to a
        # tame, bounded magnitude instead of passing through linearly.
        df = synthetic_df.copy()
        col = sensor_columns[0]
        scaler_result = fit_robust_scaler(df, sensor_columns)
        glitch_df = df.copy()
        glitch_df.loc[0, col] = df[col].median() + scaler_result.scaler.scale_[0] * 1e9

        uncompressed = apply_scaling(glitch_df, scaler_result)
        compressed = apply_scaling(glitch_df, scaler_result, tail_compression_scale=20.0)

        assert abs(uncompressed.loc[0, col]) > 1e8
        assert abs(compressed.loc[0, col]) < 1000

    def test_tail_compression_preserves_relative_ordering(self, synthetic_df, sensor_columns):
        # A more extreme outlier should still compress to a larger value
        # than a milder one -- unlike a hard clip, where both would
        # saturate to the same plateau.
        col = sensor_columns[0]
        scaler_result = fit_robust_scaler(synthetic_df, sensor_columns)

        mild_df = synthetic_df.copy()
        mild_df.loc[0, col] = synthetic_df[col].median() + scaler_result.scaler.scale_[0] * 100

        severe_df = synthetic_df.copy()
        severe_df.loc[0, col] = synthetic_df[col].median() + scaler_result.scaler.scale_[0] * 1e6

        mild_compressed = apply_scaling(mild_df, scaler_result, tail_compression_scale=20.0)
        severe_compressed = apply_scaling(severe_df, scaler_result, tail_compression_scale=20.0)

        assert severe_compressed.loc[0, col] > mild_compressed.loc[0, col]


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
