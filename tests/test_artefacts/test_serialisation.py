"""Tests for artefact save/load including sensor baselines."""

import json
import numpy as np
import torch
import pytest
from sklearn.preprocessing import RobustScaler

from autoencoder.model.architecture import Autoencoder
from autoencoder.explain.fastshap import FastSHAPExplainer
from autoencoder.artefacts.serialisation import (
    save_artefacts, load_artefacts, save_explainer, load_explainer,
)


@pytest.fixture
def model_and_artefacts(tmp_path):
    """Create a model and dummy artefacts for testing."""
    model = Autoencoder(n_sensors=10, latent_dim=8)
    scaler = RobustScaler()
    scaler.fit(np.random.rand(100, 10))
    thresholds = {"green_yellow": 0.5, "yellow_red": 1.0}
    metadata = {"n_sensors": 10, "latent_dim": 8, "epochs": 5}
    training_errors = np.random.rand(50)
    sensor_baselines = np.random.rand(10) * 0.1
    return model, scaler, thresholds, metadata, training_errors, sensor_baselines, tmp_path


class TestSaveLoadWithBaselines:
    def test_round_trip(self, model_and_artefacts):
        model, scaler, thresholds, metadata, errors, baselines, tmp_path = model_and_artefacts
        save_artefacts(tmp_path, model, scaler, thresholds, metadata, errors, baselines)
        loaded = load_artefacts(tmp_path)
        assert len(loaded) == 6
        _, _, _, _, _, loaded_baselines = loaded
        assert loaded_baselines is not None
        np.testing.assert_allclose(loaded_baselines, baselines, rtol=1e-5)

    def test_load_without_baselines_file(self, model_and_artefacts):
        model, scaler, thresholds, metadata, errors, _, tmp_path = model_and_artefacts
        # Save without baselines
        save_artefacts(tmp_path, model, scaler, thresholds, metadata, errors)
        loaded = load_artefacts(tmp_path)
        assert len(loaded) == 6
        _, _, _, _, _, loaded_baselines = loaded
        assert loaded_baselines is None

    def test_baselines_file_exists(self, model_and_artefacts):
        model, scaler, thresholds, metadata, errors, baselines, tmp_path = model_and_artefacts
        save_artefacts(tmp_path, model, scaler, thresholds, metadata, errors, baselines)
        assert (tmp_path / "sensor_baselines.npy").exists()

    def test_no_baselines_file_when_none(self, model_and_artefacts):
        model, scaler, thresholds, metadata, errors, _, tmp_path = model_and_artefacts
        save_artefacts(tmp_path, model, scaler, thresholds, metadata, errors)
        assert not (tmp_path / "sensor_baselines.npy").exists()


class TestSaveLoadExplainer:
    def test_round_trip(self, tmp_path):
        explainer = FastSHAPExplainer(n_sensors=10, hidden_dim=16)
        save_explainer(tmp_path, explainer)
        loaded = load_explainer(tmp_path)
        assert loaded is not None
        for p1, p2 in zip(explainer.state_dict().values(), loaded.state_dict().values()):
            torch.testing.assert_close(p1, p2)

    def test_weights_file_exists(self, tmp_path):
        explainer = FastSHAPExplainer(n_sensors=10, hidden_dim=16)
        save_explainer(tmp_path, explainer)
        assert (tmp_path / "explainer_weights.pt").exists()
        assert (tmp_path / "explainer_metadata.json").exists()

    def test_missing_explainer_returns_none(self, tmp_path):
        assert load_explainer(tmp_path) is None

    def test_existing_artefacts_unaffected_by_explainer_absence(self, model_and_artefacts):
        """save_artefacts/load_artefacts keep their 6-tuple contract regardless of explainer."""
        model, scaler, thresholds, metadata, errors, baselines, tmp_path = model_and_artefacts
        save_artefacts(tmp_path, model, scaler, thresholds, metadata, errors, baselines)
        loaded = load_artefacts(tmp_path)
        assert len(loaded) == 6
        assert load_explainer(tmp_path) is None
