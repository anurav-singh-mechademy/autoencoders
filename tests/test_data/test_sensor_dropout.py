"""Tests for sensor dropout filtering."""

import numpy as np
import pytest

from autoencoder.data.sensor_dropout import detect_sensor_dropout_windows, remove_sensor_dropout_windows


class TestDetectSensorDropoutWindows:
    def test_no_dropout(self, synthetic_windows):
        """Healthy synthetic windows have no nulls or stuck sensors."""
        result = detect_sensor_dropout_windows(synthetic_windows, threshold_pct=5.0)
        assert result.n_dropped == 0
        assert result.n_total == len(synthetic_windows)
        assert len(result.bad_sensor_pct) == len(synthetic_windows)

    def test_null_sensors_flagged(self):
        """A window where >5% of sensors are null should be excluded."""
        rng = np.random.default_rng(42)
        windows = [rng.normal(50.0, 5.0, (120, 20)) for _ in range(3)]
        # Flag 2/20 = 10% of sensors as null in window 1
        windows[1][:, 0] = np.nan
        windows[1][:, 1] = np.nan

        result = detect_sensor_dropout_windows(windows, threshold_pct=5.0)
        assert result.is_dropped[1] == True
        assert result.n_dropped == 1

    def test_stuck_sensors_flagged(self):
        """A window where a sensor never changes should be excluded."""
        rng = np.random.default_rng(42)
        windows = [rng.normal(50.0, 5.0, (120, 20)) for _ in range(3)]
        # 2/20 = 10% stuck sensors in window 0
        windows[0][:, 0] = 1.0
        windows[0][:, 1] = 1.0

        result = detect_sensor_dropout_windows(windows, threshold_pct=5.0)
        assert result.is_dropped[0] == True

    def test_below_threshold_not_flagged(self):
        """A single null/stuck sensor (5% of 20) should not exceed a 5% threshold."""
        rng = np.random.default_rng(42)
        windows = [rng.normal(50.0, 5.0, (120, 20)) for _ in range(3)]
        windows[0][:, 0] = 1.0  # 1/20 = 5%, not > 5%

        result = detect_sensor_dropout_windows(windows, threshold_pct=5.0)
        assert result.is_dropped[0] == False

    def test_threshold_used_recorded(self, synthetic_windows):
        result = detect_sensor_dropout_windows(synthetic_windows, threshold_pct=7.5)
        assert result.threshold_used == 7.5


class TestRemoveSensorDropoutWindows:
    def test_remove(self, synthetic_windows):
        rng = np.random.default_rng(42)
        windows = [w.copy() for w in synthetic_windows]
        windows[2][:, :3] = np.nan  # 3/20 = 15% null sensors

        result = detect_sensor_dropout_windows(windows, threshold_pct=5.0)
        clean = remove_sensor_dropout_windows(windows, result)
        assert len(clean) == len(windows) - 1
