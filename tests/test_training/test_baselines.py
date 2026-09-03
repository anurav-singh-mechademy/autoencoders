"""Tests for compute_sensor_baselines."""

import numpy as np
import pytest

from autoencoder.model.architecture import Autoencoder
from autoencoder.training.trainer import compute_sensor_baselines


@pytest.fixture
def small_model():
    model = Autoencoder(n_sensors=10, latent_dim=8)
    model.eval()
    return model


class TestComputeSensorBaselines:
    def test_output_shape(self, small_model):
        rng = np.random.default_rng(42)
        windows = rng.normal(size=(5, 120, 10)).astype(np.float32)
        baselines = compute_sensor_baselines(small_model, windows)
        assert baselines.shape == (10,)

    def test_all_positive(self, small_model):
        rng = np.random.default_rng(42)
        windows = rng.normal(size=(5, 120, 10)).astype(np.float32)
        baselines = compute_sensor_baselines(small_model, windows)
        assert np.all(baselines >= 0)

    def test_single_window(self, small_model):
        rng = np.random.default_rng(42)
        windows = rng.normal(size=(1, 120, 10)).astype(np.float32)
        baselines = compute_sensor_baselines(small_model, windows)
        assert baselines.shape == (10,)
        assert np.all(np.isfinite(baselines))
