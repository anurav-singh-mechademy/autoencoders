"""Tests for drift detection."""

import numpy as np
import pytest

from autoencoder.monitoring.drift import rolling_error_trend, ks_test_drift, per_sensor_drift


class TestRollingErrorTrend:
    def test_flat_trend(self):
        daily = [1.0] * 30
        result = rolling_error_trend(daily, window_days=7)
        assert result["trend_slope"] == pytest.approx(0.0, abs=1e-6)

    def test_increasing_trend(self):
        daily = list(range(60))  # 0, 1, 2, ..., 59
        result = rolling_error_trend(daily, window_days=7)
        assert result["trend_slope"] > 0
        assert result["is_increasing"]

    def test_short_series(self):
        daily = [1.0, 2.0, 3.0]
        result = rolling_error_trend(daily, window_days=7)
        assert result["n_days"] == 3
        assert not result["is_increasing"]

    def test_returns_rolling_mean(self):
        daily = [1.0] * 14
        result = rolling_error_trend(daily, window_days=7)
        assert len(result["rolling_mean"]) == 8  # 14 - 7 + 1


class TestKsTestDrift:
    def test_same_distribution(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 200)
        b = rng.normal(0, 1, 200)
        result = ks_test_drift(a, b)
        assert not result["is_drifted"]

    def test_different_distribution(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 200)
        b = rng.normal(5, 1, 200)  # shifted mean
        result = ks_test_drift(a, b)
        assert result["is_drifted"]
        assert result["ks_statistic"] > 0.5

    def test_output_keys(self):
        a = np.random.rand(50)
        b = np.random.rand(50)
        result = ks_test_drift(a, b)
        assert "ks_statistic" in result
        assert "p_value" in result
        assert "is_drifted" in result


class TestPerSensorDrift:
    def test_basic(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, (100, 5))
        recent = rng.normal(0, 1, (100, 5))
        recent[:, 2] += 5  # drift sensor 2
        results = per_sensor_drift(baseline, recent)
        assert len(results) == 5
        # Sensor 2 should be at the top (most drifted)
        assert results[0]["sensor_index"] == 2
        assert results[0]["is_drifted"]
