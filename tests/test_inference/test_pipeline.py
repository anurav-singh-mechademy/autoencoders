"""Tests for the inference pipeline."""

import numpy as np
import torch
import pytest
from sklearn.preprocessing import RobustScaler

from autoencoder.model.architecture import Autoencoder
from autoencoder.inference.pipeline import infer_window, is_equipment_running


@pytest.fixture
def trained_model():
    """A small trained autoencoder for testing."""
    model = Autoencoder(n_sensors=20, latent_dim=8)
    model.eval()
    return model


@pytest.fixture
def fitted_scaler():
    """A RobustScaler fitted on random data."""
    scaler = RobustScaler()
    scaler.fit(np.random.rand(500, 20))
    return scaler


class TestInferWindow:
    def test_basic_inference(self, trained_model, fitted_scaler):
        window = np.random.rand(120, 20).astype(np.float32)
        result = infer_window(window, trained_model, fitted_scaler)
        assert result.usable
        assert isinstance(result.window_score, float)
        assert result.row_errors.shape == (120,)
        assert result.sensor_errors.shape == (20,)
        assert result.reconstruction.shape == (120, 20)
        assert len(result.quality_flags) == 0

    def test_with_sensor_names(self, trained_model, fitted_scaler):
        window = np.random.rand(120, 20).astype(np.float32)
        names = [f"sensor_{i}" for i in range(20)]
        result = infer_window(window, trained_model, fitted_scaler, sensor_names=names, top_k=5)
        assert len(result.top_contributors) == 5
        assert all("name" in c for c in result.top_contributors)

    def test_window_with_nulls_fillable(self, trained_model, fitted_scaler):
        window = np.random.rand(120, 20).astype(np.float32)
        window[5, 3] = np.nan  # single gap, fillable
        result = infer_window(window, trained_model, fitted_scaler)
        assert result.usable
        assert not np.isnan(result.window_score)

    def test_window_with_one_bad_sensor_is_masked_not_rejected(self, trained_model, fitted_scaler):
        # 1/20 = 5% of sensors null-dominant, below the default 30% window
        # threshold -- still usable, with that sensor excluded from scoring.
        window = np.random.rand(120, 20).astype(np.float32)
        window[:50, 0] = np.nan  # 50 consecutive nulls in sensor 0
        result = infer_window(window, trained_model, fitted_scaler)
        assert result.usable
        assert not np.isnan(result.window_score)
        assert result.masked_sensors == [0]
        assert all(c["index"] != 0 for c in result.top_contributors)

    def test_window_with_too_many_bad_sensors_not_usable(self, trained_model, fitted_scaler):
        # 10/20 = 50% of sensors null-dominant, above the default 30% window
        # threshold -- the whole window is rejected.
        window = np.random.rand(120, 20).astype(np.float32)
        for col in range(10):
            window[:50, col] = np.nan
        result = infer_window(window, trained_model, fitted_scaler)
        assert not result.usable
        assert np.isnan(result.window_score)
        assert len(result.quality_flags) > 0
        assert result.masked_sensors == list(range(10))


class TestTailCompressionScale:
    def test_extreme_value_scored_differently_with_compression(self, trained_model, fitted_scaler):
        # A window with one extreme outlier produces a huge raw z-score
        # after scaler.transform(); tail_compression_scale must actually be
        # applied (not silently ignored) for already_scaled=False callers,
        # or thresholds calibrated under compression get compared against
        # uncompressed scores.
        window = np.random.rand(120, 20).astype(np.float32)
        window[0, 0] = 1e6  # extreme glitch value

        uncompressed = infer_window(window, trained_model, fitted_scaler, tail_compression_scale=None)
        compressed = infer_window(window, trained_model, fitted_scaler, tail_compression_scale=20.0)

        assert uncompressed.usable and compressed.usable
        assert uncompressed.window_score != compressed.window_score

    def test_ignored_when_already_scaled(self, trained_model, fitted_scaler):
        window = np.random.rand(120, 20).astype(np.float32)
        already_scaled = infer_window(window, trained_model, fitted_scaler, already_scaled=True, tail_compression_scale=20.0)
        already_scaled_no_compression = infer_window(window, trained_model, fitted_scaler, already_scaled=True, tail_compression_scale=None)
        # already_scaled=True skips scaling entirely, so tail_compression_scale must have no effect.
        assert already_scaled.window_score == already_scaled_no_compression.window_score


class TestEquipmentRunningFilter:
    def test_equipment_not_running(self, trained_model, fitted_scaler):
        window = np.random.rand(120, 20).astype(np.float32)
        window[:, 5] = 0.0  # sensor 5 all zeros (shutdown)
        result = infer_window(
            window, trained_model, fitted_scaler,
            running_sensor_index=5, running_threshold=10.0,
        )
        assert not result.usable
        assert "equipment_not_running" in result.quality_flags

    def test_equipment_running(self, trained_model, fitted_scaler):
        window = np.random.rand(120, 20).astype(np.float32)
        window[:, 5] = 50.0  # sensor 5 well above threshold
        result = infer_window(
            window, trained_model, fitted_scaler,
            running_sensor_index=5, running_threshold=10.0,
        )
        assert result.usable

    def test_no_running_check(self, trained_model, fitted_scaler):
        window = np.random.rand(120, 20).astype(np.float32)
        window[:, 5] = 0.0
        result = infer_window(window, trained_model, fitted_scaler)
        # Without running_sensor_index, no check — should still be usable
        assert result.usable


class TestIsEquipmentRunning:
    def test_all_above_threshold(self):
        window = np.full((120, 10), 100.0)
        assert is_equipment_running(window, sensor_index=0, threshold=50.0)

    def test_all_below_threshold(self):
        window = np.full((120, 10), 5.0)
        assert not is_equipment_running(window, sensor_index=0, threshold=50.0)

    def test_partial_above(self):
        window = np.full((120, 10), 100.0)
        window[:60, 0] = 0.0  # 50% below
        # Exactly 50% above, min_pct=50 → True
        assert is_equipment_running(window, sensor_index=0, threshold=50.0, min_pct=50.0)

    def test_with_nans(self):
        window = np.full((120, 10), 100.0)
        window[:20, 0] = np.nan  # 20 NaN rows
        assert is_equipment_running(window, sensor_index=0, threshold=50.0)


class TestWithSensorBaselines:
    def test_baselines_in_result(self, trained_model, fitted_scaler):
        window = np.random.rand(120, 20).astype(np.float32)
        baselines = np.ones(20) * 0.01
        result = infer_window(
            window, trained_model, fitted_scaler,
            sensor_baselines=baselines,
        )
        assert result.usable
        assert result.error_ratios is not None
        assert result.error_ratios.shape == (20,)
        assert result.sensor_flags is not None
        # sensor_flags is exactly (error_ratios > flag_threshold) -- the
        # binary flag is derived from the continuous ratio, not independent.
        np.testing.assert_array_equal(result.sensor_flags, (result.error_ratios > 3.0).astype(int))
        assert result.pct_flagged_sensors is not None
        assert result.sensors_anomalous is not None

    def test_no_baselines_no_flags(self, trained_model, fitted_scaler):
        window = np.random.rand(120, 20).astype(np.float32)
        result = infer_window(window, trained_model, fitted_scaler)
        assert result.error_ratios is None
        assert result.sensor_flags is None
        assert result.pct_flagged_sensors is None
        assert result.sensors_anomalous is None
