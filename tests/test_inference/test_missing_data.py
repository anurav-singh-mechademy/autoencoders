"""Tests for missing data handling in inference."""

import numpy as np
import pytest

from autoencoder.inference.missing_data import check_nulls, forward_fill, assess_window_quality


class TestCheckNulls:
    def test_no_nulls(self):
        window = np.random.rand(120, 20)
        info = check_nulls(window)
        assert info["null_count"] == 0
        assert not info["has_nulls"]

    def test_with_nulls(self):
        window = np.random.rand(120, 20)
        window[0, 0] = np.nan
        window[10, 5] = np.nan
        window[11, 5] = np.nan
        info = check_nulls(window)
        assert info["null_count"] == 3
        assert info["has_nulls"]

    def test_null_pct_per_sensor(self):
        window = np.ones((100, 5))
        window[:10, 0] = np.nan  # 10% null in sensor 0
        info = check_nulls(window)
        assert info["null_pct_per_sensor"][0] == pytest.approx(10.0)
        assert info["null_pct_per_sensor"][1] == pytest.approx(0.0)


class TestForwardFill:
    def test_fills_single_gap(self):
        window = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
        filled, info = forward_fill(window, max_consecutive=3)
        assert filled[1, 0] == 1.0  # forward-filled from row 0
        assert info["total_filled"] == 1

    def test_respects_max_consecutive(self):
        window = np.ones((10, 1))
        window[2:7, 0] = np.nan  # 5 consecutive nulls
        filled, info = forward_fill(window, max_consecutive=3)
        # First 3 should be filled, last 2 should remain NaN
        assert not np.isnan(filled[2, 0])
        assert not np.isnan(filled[4, 0])
        assert np.isnan(filled[5, 0])
        assert info["still_has_nulls"]

    def test_no_nulls_unchanged(self):
        window = np.random.rand(120, 20)
        filled, info = forward_fill(window)
        np.testing.assert_array_equal(filled, window)
        assert info["total_filled"] == 0


class TestAssessWindowQuality:
    def test_clean_window_is_usable(self):
        window = np.random.rand(120, 20)
        result = assess_window_quality(window)
        assert result["usable"]
        assert result["quality_flags"] == []

    def test_small_gap_is_usable(self):
        window = np.random.rand(120, 20)
        window[5, 3] = np.nan  # single null
        result = assess_window_quality(window)
        assert result["usable"]
        assert not np.isnan(result["filled_window"]).any()

    def test_large_gap_not_usable(self):
        window = np.random.rand(120, 20)
        window[:20, 0] = np.nan  # 20 consecutive nulls in sensor 0
        result = assess_window_quality(window, max_null_pct_per_sensor=5.0, max_consecutive_nulls=3)
        assert not result["usable"]
        assert result["filled_window"] is None
