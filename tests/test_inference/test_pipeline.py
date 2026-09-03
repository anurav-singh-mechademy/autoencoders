"""Tests for the inference pipeline."""

import numpy as np
import torch
import pytest
from sklearn.preprocessing import RobustScaler

from autoencoder.model.architecture import Autoencoder
from autoencoder.explain.fastshap import FastSHAPExplainer
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

    def test_window_with_too_many_nulls(self, trained_model, fitted_scaler):
        window = np.random.rand(120, 20).astype(np.float32)
        window[:50, 0] = np.nan  # 50 consecutive nulls
        result = infer_window(window, trained_model, fitted_scaler)
        assert not result.usable
        assert np.isnan(result.window_score)
        assert len(result.quality_flags) > 0


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
        assert result.sensor_flags is not None
        assert result.pct_flagged_sensors is not None
        assert result.sensors_anomalous is not None

    def test_no_baselines_no_flags(self, trained_model, fitted_scaler):
        window = np.random.rand(120, 20).astype(np.float32)
        result = infer_window(window, trained_model, fitted_scaler)
        assert result.sensor_flags is None
        assert result.pct_flagged_sensors is None
        assert result.sensors_anomalous is None


class TestWithExplainer:
    def test_attribution_values_populated(self, trained_model, fitted_scaler):
        explainer = FastSHAPExplainer(n_sensors=20, hidden_dim=16)
        explainer.eval()
        window = np.random.rand(120, 20).astype(np.float32)
        result = infer_window(window, trained_model, fitted_scaler, explainer=explainer)
        assert result.usable
        assert result.attribution_method == "fastshap"
        assert result.attribution_values is not None
        assert result.attribution_values.shape == (20,)

    def test_no_explainer_defaults_to_heuristic(self, trained_model, fitted_scaler):
        window = np.random.rand(120, 20).astype(np.float32)
        result = infer_window(window, trained_model, fitted_scaler)
        assert result.attribution_method == "heuristic"
        assert result.attribution_values is None


class TestWithIntegratedGradients:
    def test_attribution_values_populated(self, trained_model, fitted_scaler):
        window = np.random.rand(120, 20).astype(np.float32)
        result = infer_window(window, trained_model, fitted_scaler, use_integrated_gradients=True, ig_n_steps=10)
        assert result.usable
        assert result.attribution_method == "integrated_gradients"
        assert result.attribution_values is not None
        assert result.attribution_values.shape == (20,)

    def test_explainer_and_ig_together_raises(self, trained_model, fitted_scaler):
        explainer = FastSHAPExplainer(n_sensors=20, hidden_dim=16)
        window = np.random.rand(120, 20).astype(np.float32)
        with pytest.raises(ValueError):
            infer_window(window, trained_model, fitted_scaler, explainer=explainer, use_integrated_gradients=True)
